"""
Tree Speculative Decoding for Layer-Skip Draft Models.

Supports:
  (A) Single-exit tree: draft generates tree from one exit layer (real early
      exit for speed). Target verifies with tree attention mask.
  (B) Multi-exit tree: draft generates candidates from multiple exits in one
      forward (wider tree, more diverse candidates). Target verifies combined tree.

Key functions:
  build_draft_tree()        — single-exit tree construction
  build_multi_exit_tree()   — multi-exit tree construction
  verify_tree()             — target verification with tree mask
  early_exit_context()      — temporarily truncate model for real speedup
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import random

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache


# ======================================================================
# Data structures
# ======================================================================

@dataclass
class DraftTree:
    """Flat representation of a draft tree."""
    tokens: torch.Tensor       # [n_nodes] token IDs
    parents: torch.Tensor      # [n_nodes] parent index (-1 for roots)
    depths: torch.Tensor       # [n_nodes] depth 0..gamma-1
    scores: torch.Tensor       # [n_nodes] cumulative log-prob
    # Per-node draft probability p_draft(this token | parent path)  used by ratio verify.
    # Optional — only populated when build_draft_tree_incremental is called with
    # draft_mode='argmax_ratio' or 'sample'.
    probs: torch.Tensor = None  # [n_nodes] float in [0, 1] or None
    # Per-depth counts (for slicing)
    depth_counts: List[int] = field(default_factory=list)
    # device
    device: torch.device = torch.device("cpu")

    @property
    def n_nodes(self):
        return self.tokens.shape[0]

    def _parents_cpu(self) -> List[int]:
        # One GPU→CPU sync; cached per-call site via callers that keep the list.
        return self.parents.cpu().tolist()

    def get_ancestor_indices(self, node_idx: int) -> List[int]:
        """Trace back from node_idx to root, return list of indices (inclusive)."""
        parents_cpu = self._parents_cpu()
        path = []
        j = node_idx
        while j >= 0:
            path.append(j)
            j = parents_cpu[j]
        path.reverse()
        return path

    def get_leaf_paths(self) -> List[List[int]]:
        """Return all root-to-leaf paths (each path = list of node indices)."""
        parents_cpu = self._parents_cpu()
        n = self.n_nodes
        children = [[] for _ in range(n)]
        for i in range(n):
            p = parents_cpu[i]
            if p >= 0:
                children[p].append(i)
        leaves = [i for i in range(n) if len(children[i]) == 0]
        paths = []
        for leaf in leaves:
            path = []
            j = leaf
            while j >= 0:
                path.append(j)
                j = parents_cpu[j]
            path.reverse()
            paths.append(path)
        return paths


# ======================================================================
# Early exit context manager
# ======================================================================

@contextmanager
def early_exit_context(model, exit_layer):
    """Temporarily truncate model.model.layers to `exit_layer` layers.

    Inside this context, the model forward only runs layers [0, exit_layer),
    then applies model.model.norm + model.lm_head. KV cache and
    output_hidden_states work normally (just fewer layers).

    This gives REAL speedup: fewer transformer blocks = less compute.
    """
    base = model.model if hasattr(model, 'model') else model
    original_layers = base.layers
    base.layers = original_layers[:exit_layer]
    try:
        yield model
    finally:
        base.layers = original_layers


# ======================================================================
# Tree attention mask construction
# ======================================================================

def build_tree_attn_mask(tree: DraftTree, prefix_len: int,
                         dtype=torch.bfloat16) -> torch.Tensor:
    """Build 4D attention mask for target verification of the tree.

    Returns: [1, 1, n_nodes, prefix_len + n_nodes] float tensor.
    Convention: 0.0 = attend, finfo.min = block.
    """
    n = tree.n_nodes
    total = prefix_len + n
    min_val = torch.finfo(dtype).min
    mask = torch.full((1, 1, n, total), min_val, device=tree.device, dtype=dtype)

    for i in range(n):
        # Attend to entire prefix
        mask[0, 0, i, :prefix_len] = 0.0
        # Attend to ancestors (including self)
        j = i
        while j >= 0:
            mask[0, 0, i, prefix_len + j] = 0.0
            j = tree.parents[j].item()

    return mask


def build_tree_position_ids(tree: DraftTree, prefix_len: int) -> torch.Tensor:
    """Position IDs: prefix_len + depth for each node."""
    return (prefix_len + tree.depths).unsqueeze(0).long()  # [1, n_nodes]


# ======================================================================
# Single-exit tree construction
# ======================================================================

@torch.inference_mode()
def build_draft_tree(
    model,
    input_ids: torch.Tensor,
    exit_layer: int,
    gamma: int = 7,
    top_k=10,
    budget: int = 63,
) -> Tuple[DraftTree, float]:
    """Build a draft tree using one exit layer with real early exit.

    Args:
        model: draft model (AutoModelForCausalLM)
        input_ids: [1, prefix_len]
        exit_layer: which layer to exit at (1-indexed, e.g., 20 for layers 0-19)
        gamma: max tree depth
        top_k: branching factor per node. Either:
          - int: same top_k at all depths (leads to exponential growth)
          - list[int] of length >= gamma: depth-aware branching, e.g.,
            [4, 3, 2, 2, 1] (EAGLE-style narrowing with depth). top_k[d] is
            the number of children per leaf at depth d.
        budget: max total tree nodes (global cap, hard limit via greedy prune)

    Returns:
        (DraftTree, draft_time_seconds)
    """
    # Normalize top_k to list of length gamma
    if isinstance(top_k, int):
        top_k_list = [top_k] * gamma
    else:
        top_k_list = list(top_k)
        if len(top_k_list) < gamma:
            top_k_list = top_k_list + [top_k_list[-1]] * (gamma - len(top_k_list))
    device = input_ids.device
    prefix_len = input_ids.shape[1]
    mdtype = next(model.parameters()).dtype
    min_val = torch.finfo(mdtype).min

    node_tokens = []    # Python ints
    node_parents = []   # Python ints
    node_depths = []    # Python ints
    node_scores = []    # Python floats
    depth_counts = []

    t0 = time.time()

    with early_exit_context(model, exit_layer):
        # Prefix forward
        out = model(input_ids, use_cache=True)
        cache = out.past_key_values

        # Depth 0: top-k roots from prefix logits
        last_logits = out.logits[:, -1, :].float()
        log_probs = F.log_softmax(last_logits, dim=-1)
        n_roots = min(top_k_list[0], budget)
        topk_s, topk_i = log_probs.topk(n_roots, dim=-1)  # [1, n_roots]

        # One sync to bulk-transfer roots
        roots_tok = topk_i[0].tolist()
        roots_sc = topk_s[0].tolist()
        for r in range(n_roots):
            node_tokens.append(roots_tok[r])
            node_parents.append(-1)
            node_depths.append(0)
            node_scores.append(roots_sc[r])
        depth_counts.append(n_roots)

        # Depth 1..gamma-1: expand leaves, prune
        for depth in range(1, gamma):
            n_leaves = depth_counts[-1]
            if n_leaves == 0:
                break
            leaf_start = len(node_tokens) - n_leaves
            leaf_indices = list(range(leaf_start, leaf_start + n_leaves))

            # Leaf tokens: build on CPU, one H2D
            leaf_toks = torch.tensor(
                [node_tokens[i] for i in leaf_indices],
                device=device, dtype=torch.long).unsqueeze(0)  # [1, n_leaves]

            pos_ids = torch.full(
                (1, n_leaves), prefix_len + depth - 1,
                device=device, dtype=torch.long)

            # Mask: Current leaves NOT yet in cache; nodes_in_cache = older ancestors.
            nodes_in_cache = len(node_tokens) - n_leaves
            cache_len = prefix_len + nodes_in_cache
            total_kv = cache_len + n_leaves

            # Build bool mask on CPU, one H2D, then convert to additive float.
            mask_bool = torch.zeros(n_leaves, total_kv, dtype=torch.bool)
            mask_bool[:, :prefix_len] = True  # all leaves attend full prefix
            for li, leaf_idx in enumerate(leaf_indices):
                j = leaf_idx
                while j >= 0:
                    mask_bool[li, prefix_len + j] = True
                    j = node_parents[j]
                mask_bool[li, cache_len + li] = True  # self
            mask = torch.where(
                mask_bool.to(device, non_blocking=True),
                torch.zeros((), dtype=mdtype, device=device),
                torch.full((), min_val, dtype=mdtype, device=device),
            ).unsqueeze(0).unsqueeze(0)

            fout = model(
                input_ids=leaf_toks,
                attention_mask=mask,
                position_ids=pos_ids,
                past_key_values=cache,
                use_cache=True,
            )
            cache = fout.past_key_values
            new_logits = fout.logits[0].float()                    # [n_leaves, V]
            new_lp = F.log_softmax(new_logits, dim=-1)

            # Top-k per leaf as tensor (no per-candidate .item())
            k_this = top_k_list[depth]
            topk_s, topk_i = new_lp.topk(k_this, dim=-1)           # [n_leaves, k]

            # Parent scores → tensor once (single H2D for n_leaves floats)
            parent_scores_t = torch.tensor(
                [node_scores[i] for i in leaf_indices],
                device=device, dtype=torch.float32)                # [n_leaves]
            cum_scores = parent_scores_t.unsqueeze(1) + topk_s     # [n_leaves, k]

            flat_scores = cum_scores.reshape(-1)                   # [n_leaves * k]
            flat_tokens = topk_i.reshape(-1)
            leaf_idx_t = torch.tensor(
                leaf_indices, device=device, dtype=torch.long)
            flat_parents = leaf_idx_t.unsqueeze(1).expand(-1, k_this).reshape(-1)

            remaining = budget - len(node_tokens)
            keep = min(remaining, flat_scores.numel())
            if keep <= 0:
                depth_counts.append(0)
                break

            top_vals, top_idx = flat_scores.topk(keep)
            sel_tokens = flat_tokens[top_idx]
            sel_parents = flat_parents[top_idx]

            # One sync to extract the full set.
            sel_t = sel_tokens.tolist()
            sel_p = sel_parents.tolist()
            sel_s = top_vals.tolist()
            for i in range(keep):
                node_tokens.append(sel_t[i])
                node_parents.append(sel_p[i])
                node_depths.append(depth)
                node_scores.append(sel_s[i])
            depth_counts.append(keep)

    draft_time = time.time() - t0

    tree = DraftTree(
        tokens=torch.tensor(node_tokens, device=device),
        parents=torch.tensor(node_parents, device=device),
        depths=torch.tensor(node_depths, device=device),
        scores=torch.tensor(node_scores, device=device, dtype=torch.float32),
        depth_counts=depth_counts,
        device=device,
    )
    return tree, draft_time


# ======================================================================
# Multi-exit tree construction
# ======================================================================

@torch.inference_mode()
def build_multi_exit_tree(
    model,
    input_ids: torch.Tensor,
    exit_layers: List[int],
    num_layers: int,
    gamma: int = 7,
    top_k_per_exit: int = 5,
    budget: int = 63,
) -> Tuple[DraftTree, float]:
    """Build a draft tree combining candidates from multiple exit layers.

    Runs the FULL model (all layers) at each depth, then extracts hidden
    states at each exit layer → logits → top-k. Candidates from all exits
    are merged, deduplicated (same parent+token → keep highest score), and
    pruned to budget.

    No early exit speedup for the draft (all layers run), but wider tree
    may improve acceptance.
    """
    device = input_ids.device
    prefix_len = input_ids.shape[1]
    norm = model.model.norm
    lm_head = model.lm_head

    node_tokens = []
    node_parents = []
    node_depths = []
    node_scores = []
    depth_counts = []

    t0 = time.time()

    # Prefix forward (full model, with hidden states)
    out = model.model(input_ids, use_cache=True, output_hidden_states=True,
                      return_dict=True)
    cache = out.past_key_values

    # Depth 0: top-k per exit, merge
    candidates_d0 = []
    for e in exit_layers:
        h = out.hidden_states[e][:, -1, :]  # [1, H]
        if e < num_layers:
            h = norm(h)
        logits = lm_head(h).float()  # [1, V]
        lp = F.log_softmax(logits, dim=-1)
        topk_s, topk_i = lp.topk(top_k_per_exit, dim=-1)
        for ki in range(top_k_per_exit):
            candidates_d0.append((
                topk_s[0, ki].item(), topk_i[0, ki].item(), -1, 0))

    # Deduplicate (same token → keep highest score)
    seen = {}
    for score, tok, par, d in candidates_d0:
        if tok not in seen or score > seen[tok][0]:
            seen[tok] = (score, tok, par, d)
    candidates_d0 = list(seen.values())
    candidates_d0.sort(key=lambda c: c[0], reverse=True)

    n_roots = min(budget, len(candidates_d0))
    for score, tok, par, d in candidates_d0[:n_roots]:
        node_tokens.append(tok)
        node_parents.append(par)
        node_depths.append(d)
        node_scores.append(score)
    depth_counts.append(n_roots)

    # Depth 1..gamma-1
    for depth in range(1, gamma):
        n_leaves = depth_counts[-1]
        if n_leaves == 0:
            break
        leaf_start = len(node_tokens) - n_leaves
        leaf_indices = list(range(leaf_start, leaf_start + n_leaves))

        leaf_toks = torch.tensor(
            [[node_tokens[i] for i in leaf_indices]], device=device)
        pos_ids = torch.full((1, n_leaves), prefix_len + depth - 1,
                             device=device, dtype=torch.long)

        # 4D mask (current leaves not yet in cache)
        nodes_in_cache = len(node_tokens) - n_leaves
        cache_len = prefix_len + nodes_in_cache
        total_kv = cache_len + n_leaves
        mdtype = next(model.parameters()).dtype
        min_val = torch.finfo(mdtype).min
        mask = torch.full((1, 1, n_leaves, total_kv), min_val,
                          device=device, dtype=mdtype)
        for li, leaf_idx in enumerate(leaf_indices):
            mask[0, 0, li, :prefix_len] = 0.0
            j = leaf_idx
            while j >= 0:
                mask[0, 0, li, prefix_len + j] = 0.0
                j = node_parents[j]
            mask[0, 0, li, cache_len + li] = 0.0

        # Forward FULL model (need all exits' hidden states)
        fout = model.model(
            input_ids=leaf_toks,
            attention_mask=mask,
            position_ids=pos_ids,
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        cache = fout.past_key_values
        all_hs = fout.hidden_states

        # Per-exit top-k → merge candidates
        all_candidates = []
        for e in exit_layers:
            h_e = all_hs[e][0].float()  # [n_leaves, H]
            if e < num_layers:
                h_e = norm(h_e.to(norm.weight.dtype)).float()
            logits_e = lm_head(h_e.to(lm_head.weight.dtype)).float()
            lp_e = F.log_softmax(logits_e, dim=-1)
            topk_s, topk_i = lp_e.topk(top_k_per_exit, dim=-1)

            for li, leaf_idx in enumerate(leaf_indices):
                ps = node_scores[leaf_idx]
                for ki in range(top_k_per_exit):
                    all_candidates.append((
                        ps + topk_s[li, ki].item(),
                        topk_i[li, ki].item(),
                        leaf_idx,
                        depth,
                    ))

        # Deduplicate (same parent + same token → keep highest)
        seen = {}
        for score, tok, par, d in all_candidates:
            key = (par, tok)
            if key not in seen or score > seen[key][0]:
                seen[key] = (score, tok, par, d)
        deduped = list(seen.values())
        deduped.sort(key=lambda c: c[0], reverse=True)

        remaining = budget - len(node_tokens)
        keep = min(remaining, len(deduped))
        if keep <= 0:
            depth_counts.append(0)
            break
        for score, tok, par, d in deduped[:keep]:
            node_tokens.append(tok)
            node_parents.append(par)
            node_depths.append(d)
            node_scores.append(score)
        depth_counts.append(keep)

    draft_time = time.time() - t0

    tree = DraftTree(
        tokens=torch.tensor(node_tokens, device=device),
        parents=torch.tensor(node_parents, device=device),
        depths=torch.tensor(node_depths, device=device),
        scores=torch.tensor(node_scores, device=device, dtype=torch.float32),
        depth_counts=depth_counts,
        device=device,
    )
    return tree, draft_time


# ======================================================================
# Per-exit independent trees (no dedup across exits)
# ======================================================================

@torch.inference_mode()
def build_per_exit_trees(
    model,
    input_ids: torch.Tensor,
    exit_layers: List[int],
    num_layers: int,
    gamma: int = 7,
    top_k_per_exit: int = 3,
    budget_per_exit: int = 15,
) -> Tuple[DraftTree, float]:
    """Build N INDEPENDENT trees, one per exit layer. No dedup across exits.

    All exits share ONE draft model forward per depth (shared backbone), but
    each exit independently builds its own tree from its own layer's logits.
    Nodes from different exits never share parents — each tree is an isolated
    subgraph.

    At top_k_per_exit=1, budget_per_exit=gamma: each exit produces a pure
    CHAIN of gamma tokens. N exits → N independent chains → target verifies
    all N chains in ONE forward, picks the longest accepted.

    Tree layout in the flat representation:
      Nodes [0..n_0-1]         = exit_0's tree
      Nodes [n_0..n_0+n_1-1]   = exit_1's tree
      ...
    Parent indices are in GLOBAL index space (offset by each tree's start).
    A node's ancestor chain stays within its own exit's tree (by construction).
    """
    device = input_ids.device
    prefix_len = input_ids.shape[1]
    norm = model.model.norm
    lm_head = model.lm_head
    n_exits = len(exit_layers)

    # Per-exit node storage (local parent indices within each exit's tree)
    per_exit_tokens = [[] for _ in range(n_exits)]
    per_exit_parents = [[] for _ in range(n_exits)]   # local, -1 for root
    per_exit_depths = [[] for _ in range(n_exits)]
    per_exit_scores = [[] for _ in range(n_exits)]

    t0 = time.time()

    # Prefix forward (shared across all exits)
    out = model.model(input_ids, use_cache=True, output_hidden_states=True,
                      return_dict=True)
    cache = out.past_key_values
    hs_last = out.hidden_states

    # Depth 0: each exit picks its OWN top-k (no dedup)
    for e_idx, e in enumerate(exit_layers):
        h = hs_last[e][:, -1, :]
        if e < num_layers:
            h = norm(h)
        logits = lm_head(h).float()
        lp = F.log_softmax(logits, dim=-1)
        topk_s, topk_i = lp.topk(min(top_k_per_exit, budget_per_exit), dim=-1)
        for ki in range(topk_i.shape[1]):
            per_exit_tokens[e_idx].append(topk_i[0, ki].item())
            per_exit_parents[e_idx].append(-1)
            per_exit_depths[e_idx].append(0)
            per_exit_scores[e_idx].append(topk_s[0, ki].item())

    # Depth 1..gamma-1
    for depth in range(1, gamma):
        # Collect leaves from all exits: each leaf tracked by (exit_idx, local_idx)
        leaf_exit_idx = []
        leaf_local_idx = []
        leaf_tokens_list = []
        leaf_parent_tok_positions = []  # global position in the shared cache
        # We need a SHARED global index for cache positions across exits.
        # Layout: exit_0's all nodes first, then exit_1's, etc.

        # Compute starting offset for each exit in the flat cache layout
        exit_node_counts = [len(per_exit_tokens[i]) for i in range(n_exits)]
        exit_offsets = [0]
        for i in range(n_exits):
            exit_offsets.append(exit_offsets[-1] + exit_node_counts[i])
        total_nodes = exit_offsets[-1]

        for e_idx in range(n_exits):
            for li, d in enumerate(per_exit_depths[e_idx]):
                if d == depth - 1:
                    leaf_exit_idx.append(e_idx)
                    leaf_local_idx.append(li)
                    leaf_tokens_list.append(per_exit_tokens[e_idx][li])

        n_leaves_total = len(leaf_tokens_list)
        if n_leaves_total == 0:
            break

        # Forward: batch all leaves from all exits
        leaf_toks = torch.tensor([leaf_tokens_list], device=device)  # [1, n_leaves_total]
        pos_ids = torch.full((1, n_leaves_total), prefix_len + depth - 1,
                             device=device, dtype=torch.long)

        # 4D mask: each leaf attends to prefix + its OWN exit's ancestor chain.
        # Cache currently holds: prefix + previously forwarded nodes from all exits
        # (ordered by exit: exit_0_nodes first, then exit_1_nodes, etc.)
        # BUT at depth=1, no nodes have been forwarded yet; depth-0 nodes are the
        # "leaves" being forwarded now.

        # nodes_in_cache = all nodes from depths 0..depth-2 across all exits
        nodes_in_cache = 0
        for e_idx in range(n_exits):
            for d in per_exit_depths[e_idx]:
                if d < depth - 1:
                    nodes_in_cache += 1
        cache_len = prefix_len + nodes_in_cache
        total_kv = cache_len + n_leaves_total
        mdtype = next(model.parameters()).dtype
        min_val = torch.finfo(mdtype).min
        mask = torch.full((1, 1, n_leaves_total, total_kv), min_val,
                          device=device, dtype=mdtype)

        # For each leaf, set attendance
        # Each exit's nodes in the global flat layout:
        #   exit e_idx's nodes start at global index `exit_offsets[e_idx]`
        #   within that, at cache position prefix_len + exit_offsets[e_idx] + local_idx
        #   (assuming nodes are added to cache in exit-major, insertion order)
        # Actually: cache order reflects FORWARD order. At depth d, we forward all
        # leaves across exits in order. So cache position = prefix_len + forward_order.
        # For simplicity, use global index = exit_offsets[e_idx] + local_idx.

        for li, (e_idx, local_idx) in enumerate(zip(leaf_exit_idx, leaf_local_idx)):
            mask[0, 0, li, :prefix_len] = 0.0
            # Trace ancestors within this exit's tree
            j = local_idx
            while j >= 0:
                global_idx = exit_offsets[e_idx] + j
                # Ancestor is in cache only if its depth < current depth - 1 actually
                # Actually if it was forwarded in a previous iteration, yes.
                # All ancestors have depth < leaf's depth, so they're in cache.
                if global_idx < nodes_in_cache:
                    mask[0, 0, li, prefix_len + global_idx] = 0.0
                j = per_exit_parents[e_idx][j]
            mask[0, 0, li, cache_len + li] = 0.0  # self

        fout = model.model(
            input_ids=leaf_toks,
            attention_mask=mask,
            position_ids=pos_ids,
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        cache = fout.past_key_values
        all_hs = fout.hidden_states

        # For each leaf, use ITS exit's hidden state to get top-k children
        # Group leaves by exit for efficient extraction
        new_nodes_per_exit = [[] for _ in range(n_exits)]

        for li in range(n_leaves_total):
            e_idx = leaf_exit_idx[li]
            e = exit_layers[e_idx]
            local_idx = leaf_local_idx[li]

            h = all_hs[e][0, li, :]  # [H]
            if e < num_layers:
                h = norm(h.to(norm.weight.dtype)).float()
            else:
                h = h.float()
            logits_e = lm_head(h.to(lm_head.weight.dtype)).float()  # [V]
            lp_e = F.log_softmax(logits_e, dim=-1)
            topk_s, topk_i = lp_e.topk(top_k_per_exit)

            parent_score = per_exit_scores[e_idx][local_idx]
            for ki in range(top_k_per_exit):
                new_nodes_per_exit[e_idx].append((
                    parent_score + topk_s[ki].item(),
                    topk_i[ki].item(),
                    local_idx,   # parent (local to this exit)
                    depth,
                ))

        # Prune each exit's new candidates to per-exit budget
        for e_idx in range(n_exits):
            candidates = new_nodes_per_exit[e_idx]
            candidates.sort(key=lambda c: c[0], reverse=True)
            current_tree_size = len(per_exit_tokens[e_idx])
            remaining = budget_per_exit - current_tree_size
            keep = min(remaining, len(candidates))
            for score, tok, par, d in candidates[:keep]:
                per_exit_tokens[e_idx].append(tok)
                per_exit_parents[e_idx].append(par)
                per_exit_depths[e_idx].append(d)
                per_exit_scores[e_idx].append(score)

    # Flatten: concatenate all exits' trees, offset parent indices
    flat_tokens = []
    flat_parents = []
    flat_depths = []
    flat_scores = []

    offset = 0
    for e_idx in range(n_exits):
        for i in range(len(per_exit_tokens[e_idx])):
            flat_tokens.append(per_exit_tokens[e_idx][i])
            par_local = per_exit_parents[e_idx][i]
            flat_parents.append(-1 if par_local < 0 else offset + par_local)
            flat_depths.append(per_exit_depths[e_idx][i])
            flat_scores.append(per_exit_scores[e_idx][i])
        offset += len(per_exit_tokens[e_idx])

    draft_time = time.time() - t0

    tree = DraftTree(
        tokens=torch.tensor(flat_tokens, device=device),
        parents=torch.tensor(flat_parents, device=device),
        depths=torch.tensor(flat_depths, device=device),
        scores=torch.tensor(flat_scores, device=device, dtype=torch.float32),
        depth_counts=[],
        device=device,
    )
    return tree, draft_time


# ======================================================================
# Target verification
# ======================================================================

@torch.inference_mode()
def verify_tree(
    target_model,
    input_ids: torch.Tensor,
    tree: DraftTree,
) -> Tuple[torch.Tensor, int, float]:
    """Target verifies the draft tree in one forward pass.

    Args:
        target_model: the large target LLM
        input_ids: [1, prefix_len] the prompt
        tree: DraftTree with n_nodes nodes

    Returns:
        (accepted_tokens, n_accepted, target_time_seconds)
        accepted_tokens: [1, n_accepted + 1] (accepted draft tokens + 1 correction)
        n_accepted: number of draft tokens accepted (= acceptance length)
    """
    device = input_ids.device
    prefix_len = input_ids.shape[1]
    n = tree.n_nodes

    if n == 0:
        # Empty tree → just do target forward on prefix, predict 1 token
        t0 = time.time()
        out = target_model(input_ids)
        correction = out.logits[:, -1, :].argmax(-1, keepdim=True)
        target_time = time.time() - t0
        return correction, 0, target_time

    t0 = time.time()

    tdtype = next(target_model.parameters()).dtype
    tree_pos = build_tree_position_ids(tree, prefix_len)

    # Pack: [prefix_tokens, tree_tokens]
    tree_input = torch.cat([input_ids, tree.tokens.unsqueeze(0)], dim=1)

    # Build full 4D attention mask [1, 1, total_len, total_len] in ONE pass.
    # Prefix rows use standard causal; tree rows use tree ancestry.
    total_len = prefix_len + n
    min_val = torch.finfo(tdtype).min

    # Prefix causal (vectorized): row i attends to cols [0..i] in prefix,
    # nothing in tree columns.
    arange = torch.arange(total_len, device=device)
    # causal[i, j] = True iff j <= i (attend)
    causal = arange.unsqueeze(0) <= arange.unsqueeze(1)  # [total_len, total_len]

    # Tree ancestry (CPU-side, avoid GPU sync): node i attends to prefix
    # + its ancestor chain in the tree. Build a [n, prefix_len + n] bool matrix.
    tree_attend = torch.zeros(n, total_len, dtype=torch.bool, device=device)
    tree_attend[:, :prefix_len] = True  # all tree nodes attend to full prefix
    parents_cpu = tree.parents.cpu().tolist()
    for i in range(n):
        j = i
        while j >= 0:
            tree_attend[i, prefix_len + j] = True
            j = parents_cpu[j]

    # Compose: full_attend[i, j] = prefix causal if i < prefix_len else tree_attend
    full_attend = causal.clone()
    full_attend[prefix_len:, :] = tree_attend

    # Convert bool → float mask (0 for attend, min_val for block)
    full_mask = torch.where(full_attend,
                            torch.zeros((), dtype=tdtype, device=device),
                            torch.full((), min_val, dtype=tdtype, device=device))
    full_mask = full_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, total_len, total_len]

    # Position IDs
    prefix_pos = torch.arange(prefix_len, device=device).unsqueeze(0)
    full_pos = torch.cat([prefix_pos, tree_pos], dim=1)

    # Target forward (use sdpa for 4D mask support)
    out = target_model(
        input_ids=tree_input,
        attention_mask=full_mask,
        position_ids=full_pos,
    )
    logits = out.logits[0].float()  # [total_len, V]

    target_time = time.time() - t0

    # Find longest accepted path
    # Target prediction at position (prefix_len - 1) → what should be at pos prefix_len
    # This is compared against depth-0 roots.
    # Target prediction at tree node i → what should follow node i's position
    # This is compared against node i's children.

    target_argmax = logits.argmax(dim=-1)  # [total_len]

    # Check all root-to-leaf paths, find the longest accepted one
    paths = tree.get_leaf_paths()
    best_accepted = 0
    best_path = []

    for path in paths:
        accepted = 0
        # Check depth 0: target at prefix_end predicts root
        root_idx = path[0]
        if target_argmax[prefix_len - 1].item() == tree.tokens[root_idx].item():
            accepted = 1
            # Check deeper depths
            for pi in range(1, len(path)):
                prev_idx = path[pi - 1]
                curr_idx = path[pi]
                # Target at prev_idx's position predicts curr token
                if target_argmax[prefix_len + prev_idx].item() == tree.tokens[curr_idx].item():
                    accepted += 1
                else:
                    break
        if accepted > best_accepted:
            best_accepted = accepted
            best_path = path[:accepted]

    # Accepted tokens = tree tokens along best path
    if best_accepted > 0:
        accepted_toks = tree.tokens[best_path].unsqueeze(0)  # [1, best_accepted]
    else:
        accepted_toks = torch.empty(1, 0, dtype=torch.long, device=device)

    # Correction token: target's prediction at the last accepted position
    if best_accepted > 0:
        last_accepted_idx = best_path[-1]
        correction = target_argmax[prefix_len + last_accepted_idx].unsqueeze(0).unsqueeze(0)
    else:
        correction = target_argmax[prefix_len - 1].unsqueeze(0).unsqueeze(0)

    result = torch.cat([accepted_toks, correction], dim=1)  # [1, accepted+1]
    return result, best_accepted, target_time


# ======================================================================
# Full speculative decoding loop
# ======================================================================

@torch.inference_mode()
def spec_decode_tree(
    target_model,
    draft_model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    exit_layer: int,
    gamma: int = 7,
    top_k: int = 10,
    budget: int = 63,
    eos_token_id=None,
) -> dict:
    """Run tree speculative decoding with a single-exit draft.

    Returns dict with timing and acceptance stats.
    """
    device = input_ids.device
    cur_ids = input_ids.clone()
    total_tokens = 0
    total_rounds = 0
    total_accepted = 0
    total_draft_time = 0.0
    total_target_time = 0.0

    while total_tokens < max_new_tokens:
        tree, dt = build_draft_tree(
            draft_model, cur_ids, exit_layer, gamma, top_k, budget)
        total_draft_time += dt

        new_tokens, n_acc, tt = verify_tree(target_model, cur_ids, tree)
        total_target_time += tt

        cur_ids = torch.cat([cur_ids, new_tokens], dim=1)
        total_tokens += new_tokens.shape[1]
        total_rounds += 1
        total_accepted += n_acc

        # EOS check
        if eos_token_id is not None:
            eos_list = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
            if any(t in new_tokens[0].tolist() for t in eos_list):
                break

    total_time = total_draft_time + total_target_time
    return {
        "total_tokens": total_tokens,
        "total_rounds": total_rounds,
        "total_accepted": total_accepted,
        "mean_alpha": total_accepted / max(total_rounds, 1),
        "tokens_per_sec": total_tokens / max(total_time, 1e-6),
        "total_time": total_time,
        "draft_time": total_draft_time,
        "target_time": total_target_time,
        "draft_fraction": total_draft_time / max(total_time, 1e-6),
    }


@torch.inference_mode()
def spec_decode_tree_per_exit(
    target_model,
    draft_model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    exit_layers: List[int],
    num_layers: int,
    gamma: int = 7,
    top_k_per_exit: int = 3,
    budget_per_exit: int = 15,
    eos_token_id=None,
) -> dict:
    """Tree spec decoding with N INDEPENDENT trees (one per exit).
    Target verifies all N trees in one forward, picks the longest accepted path.

    With top_k_per_exit=1, budget_per_exit=gamma: N independent chains.
    """
    device = input_ids.device
    cur_ids = input_ids.clone()
    total_tokens = 0
    total_rounds = 0
    total_accepted = 0
    total_draft_time = 0.0
    total_target_time = 0.0

    while total_tokens < max_new_tokens:
        tree, dt = build_per_exit_trees(
            draft_model, cur_ids, exit_layers, num_layers,
            gamma, top_k_per_exit, budget_per_exit)
        total_draft_time += dt

        new_tokens, n_acc, tt = verify_tree(target_model, cur_ids, tree)
        total_target_time += tt

        cur_ids = torch.cat([cur_ids, new_tokens], dim=1)
        total_tokens += new_tokens.shape[1]
        total_rounds += 1
        total_accepted += n_acc

        if eos_token_id is not None:
            eos_list = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
            if any(t in new_tokens[0].tolist() for t in eos_list):
                break

    total_time = total_draft_time + total_target_time
    return {
        "total_tokens": total_tokens,
        "total_rounds": total_rounds,
        "total_accepted": total_accepted,
        "mean_alpha": total_accepted / max(total_rounds, 1),
        "tokens_per_sec": total_tokens / max(total_time, 1e-6),
        "total_time": total_time,
        "draft_time": total_draft_time,
        "target_time": total_target_time,
        "draft_fraction": total_draft_time / max(total_time, 1e-6),
    }


@torch.inference_mode()
def spec_decode_tree_multi_exit(
    target_model,
    draft_model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    exit_layers: List[int],
    num_layers: int,
    gamma: int = 7,
    top_k_per_exit: int = 5,
    budget: int = 63,
    eos_token_id=None,
) -> dict:
    """Run tree speculative decoding with multi-exit draft."""
    device = input_ids.device
    cur_ids = input_ids.clone()
    total_tokens = 0
    total_rounds = 0
    total_accepted = 0
    total_draft_time = 0.0
    total_target_time = 0.0

    while total_tokens < max_new_tokens:
        tree, dt = build_multi_exit_tree(
            draft_model, cur_ids, exit_layers, num_layers,
            gamma, top_k_per_exit, budget)
        total_draft_time += dt

        new_tokens, n_acc, tt = verify_tree(target_model, cur_ids, tree)
        total_target_time += tt

        cur_ids = torch.cat([cur_ids, new_tokens], dim=1)
        total_tokens += new_tokens.shape[1]
        total_rounds += 1
        total_accepted += n_acc

        if eos_token_id is not None:
            eos_list = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
            if any(t in new_tokens[0].tolist() for t in eos_list):
                break

    total_time = total_draft_time + total_target_time
    return {
        "total_tokens": total_tokens,
        "total_rounds": total_rounds,
        "total_accepted": total_accepted,
        "mean_alpha": total_accepted / max(total_rounds, 1),
        "tokens_per_sec": total_tokens / max(total_time, 1e-6),
        "total_time": total_time,
        "draft_time": total_draft_time,
        "target_time": total_target_time,
        "draft_fraction": total_draft_time / max(total_time, 1e-6),
    }


# ======================================================================
# Incremental KV verification (persistent cache across rounds)
# ======================================================================

def _cache_reorder(cache, sel_idx: torch.Tensor):
    """In-place: keep only positions in sel_idx along the seq dim of every layer.

    Handles both old DynamicCache API (key_cache/value_cache lists) and the
    newer layers[...].keys/values attributes in transformers 5.x.
    """
    if hasattr(cache, 'layers') and cache.layers is not None and len(cache.layers) > 0 \
            and hasattr(cache.layers[0], 'keys'):
        for layer in cache.layers:
            layer.keys = layer.keys.index_select(-2, sel_idx)
            layer.values = layer.values.index_select(-2, sel_idx)
    elif hasattr(cache, 'key_cache') and len(cache.key_cache) > 0:
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = cache.key_cache[i].index_select(-2, sel_idx)
            cache.value_cache[i] = cache.value_cache[i].index_select(-2, sel_idx)
    else:
        raise RuntimeError(f"Cannot reorder cache of type {type(cache)}")
    new_len = int(sel_idx.numel())
    if hasattr(cache, '_seen_tokens'):
        cache._seen_tokens = new_len


@dataclass
class VerifyState:
    """Persistent state for incremental KV decoding (target or draft)."""
    cache: object
    cache_len: int
    pending_token: torch.Tensor     # [1, 1]


@torch.inference_mode()
def prefill_minus_one(model, input_ids: torch.Tensor) -> VerifyState:
    """Forward prefix[:-1], return state with pending = prefix[-1]."""
    prefix = input_ids[:, :-1]
    if prefix.shape[1] > 0:
        out = model(prefix, use_cache=True)
        cache = out.past_key_values
        cache_len = prefix.shape[1]
    else:
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache()
        cache_len = 0
    return VerifyState(
        cache=cache,
        cache_len=cache_len,
        pending_token=input_ids[:, -1:].contiguous(),
    )


target_prefill = prefill_minus_one  # alias for clarity in eval scripts


@torch.inference_mode()
def build_draft_tree_incremental(
    model,
    state: VerifyState,
    exit_layer: int,
    gamma: int = 7,
    top_k=10,
    budget: int = 63,
    draft_mode: str = 'argmax',
    temperature: float = 1.0,
    need_probs: bool = False,
    draft_top_k_vocab: int = 0,
) -> Tuple[DraftTree, float]:
    """Build a draft tree with PERSISTENT draft KV cache.

    draft_mode:
      'argmax' — top-k branching at each depth (deterministic, default)
      'sample' — multinomial sample of k children per leaf, no replacement.
                 Uses temperature for softmax scaling.
    need_probs:
      If True, also records p_draft(token | parent) per node into tree.probs.
      Required for ratio verify ('ratio' verify_mode).
    """
    """Build a draft tree with PERSISTENT draft KV cache.

    Preconditions:
      state.cache has `state.cache_len` positions.
      state.pending_token is the next token (not yet in cache).

    Postconditions (mutates state in place):
      state.cache has `state.cache_len + 1 + tree.n_nodes` positions, laid out as
        [prior cache ...][pending][tree node 0][tree node 1]...[tree node n-1]
      Tree node i is at cache position `L0 + 1 + i` where L0 is the pre-call cache_len.
      state.cache_len is updated accordingly.
      state.pending_token is now stale (the caller should replace it with the
      correction after target verification).

    Returns: (tree, draft_time)
    """
    # Normalize top_k to list of length gamma
    if isinstance(top_k, int):
        top_k_list = [top_k] * gamma
    else:
        top_k_list = list(top_k)
        if len(top_k_list) < gamma:
            top_k_list = top_k_list + [top_k_list[-1]] * (gamma - len(top_k_list))

    device = state.pending_token.device
    mdtype = next(model.parameters()).dtype
    min_val = torch.finfo(mdtype).min

    L0 = state.cache_len            # original cache_len (before tree build)
    prefix_len = L0 + 1             # cache position AFTER pending has been forwarded

    node_tokens = []
    node_parents = []
    node_depths = []
    node_scores = []
    node_probs = []   # p_draft per node (1.0 placeholder when not needed)
    depth_counts = []
    T_draft = max(temperature, 1e-6)

    t0 = time.time()

    with early_exit_context(model, exit_layer):
        # Step 1: forward pending → cache grows by 1; logits give depth-0 roots.
        pos_p = torch.tensor([[L0]], device=device, dtype=torch.long)
        out = model(
            input_ids=state.pending_token,
            position_ids=pos_p,
            past_key_values=state.cache,
            use_cache=True,
        )
        state.cache = out.past_key_values
        state.cache_len = prefix_len

        last_logits = out.logits[:, -1, :].float()
        n_roots = min(top_k_list[0], budget)
        if draft_mode == 'sample':
            probs_root = F.softmax(last_logits / T_draft, dim=-1)        # [1, V]
            if draft_top_k_vocab > 0:
                K_v = min(draft_top_k_vocab, probs_root.shape[-1])
                tkv_vals, tkv_idx = probs_root.topk(K_v, dim=-1)          # [1, K_v]
                tkv_norm = tkv_vals / tkv_vals.sum(-1, keepdim=True).clamp(min=1e-20)
                n_r = min(n_roots, K_v)
                idx_in_k = torch.multinomial(tkv_norm[0], n_r,
                                             replacement=False)           # [n_r]
                sampled = tkv_idx[0, idx_in_k]                            # [n_r]
                p_at = tkv_norm[0, idx_in_k].tolist()
                roots_sc = tkv_norm.clamp(min=1e-20).log()[0, idx_in_k].tolist()
                n_roots = n_r
            else:
                sampled = torch.multinomial(probs_root[0], n_roots,
                                            replacement=False)            # [n_roots]
                p_at = probs_root[0, sampled].tolist()
                log_probs_full = F.log_softmax(last_logits / T_draft, dim=-1)
                roots_sc = log_probs_full[0, sampled].tolist()
            roots_tok = sampled.tolist()
        else:
            log_probs = F.log_softmax(last_logits, dim=-1)
            topk_s, topk_i = log_probs.topk(n_roots, dim=-1)
            roots_tok = topk_i[0].tolist()
            roots_sc = topk_s[0].tolist()
            if need_probs:
                probs_root_T = F.softmax(last_logits / T_draft, dim=-1)
                p_at = probs_root_T[0, topk_i[0]].tolist()
            else:
                p_at = [1.0] * n_roots
        for r in range(n_roots):
            node_tokens.append(roots_tok[r])
            node_parents.append(-1)
            node_depths.append(0)
            node_scores.append(roots_sc[r])
            node_probs.append(p_at[r])
        depth_counts.append(n_roots)

        # Depth 1..gamma: forward depth-(d-1) leaves. At depth == gamma we only
        # place the final-depth leaves into cache (no new candidates to add).
        for depth in range(1, gamma + 1):
            n_leaves = depth_counts[-1]
            if n_leaves == 0:
                break

            leaf_start = len(node_tokens) - n_leaves
            leaf_indices = list(range(leaf_start, leaf_start + n_leaves))

            leaf_toks = torch.tensor(
                [node_tokens[i] for i in leaf_indices],
                device=device, dtype=torch.long).unsqueeze(0)
            pos_ids = torch.full(
                (1, n_leaves), prefix_len + depth - 1,
                device=device, dtype=torch.long)

            cache_len = state.cache_len
            total_kv = cache_len + n_leaves

            # 4D mask: attend prefix [0..prefix_len-1] + own ancestor chain + self.
            mask_bool = torch.zeros(n_leaves, total_kv, dtype=torch.bool)
            mask_bool[:, :prefix_len] = True
            for li, leaf_idx in enumerate(leaf_indices):
                j = leaf_idx
                while j >= 0:
                    mask_bool[li, prefix_len + j] = True
                    j = node_parents[j]
                mask_bool[li, cache_len + li] = True
            mask = torch.where(
                mask_bool.to(device, non_blocking=True),
                torch.zeros((), dtype=mdtype, device=device),
                torch.full((), min_val, dtype=mdtype, device=device),
            ).unsqueeze(0).unsqueeze(0)

            fout = model(
                input_ids=leaf_toks,
                attention_mask=mask,
                position_ids=pos_ids,
                past_key_values=state.cache,
                use_cache=True,
            )
            state.cache = fout.past_key_values
            state.cache_len = cache_len + n_leaves

            if depth >= gamma:
                # Final-depth leaves are now in cache; no more candidates needed.
                break

            new_logits = fout.logits[0].float()
            new_lp = F.log_softmax(new_logits, dim=-1)
            k_this = top_k_list[depth]

            if draft_mode == 'sample':
                probs_leaves = F.softmax(new_logits / T_draft, dim=-1)   # [n_leaves, V]
                if draft_top_k_vocab > 0:
                    K_v = min(draft_top_k_vocab, probs_leaves.shape[-1])
                    tkv_vals, tkv_idx = probs_leaves.topk(K_v, dim=-1)   # [n_leaves, K_v]
                    tkv_norm = tkv_vals / tkv_vals.sum(-1, keepdim=True).clamp(min=1e-20)
                    k_eff = min(k_this, K_v)
                    idx_in_k = torch.multinomial(tkv_norm, k_eff,
                                                 replacement=False)      # [n_leaves, k_eff]
                    sampled = tkv_idx.gather(1, idx_in_k)                # actual tokens
                    cand_probs = tkv_norm.gather(1, idx_in_k)
                    topk_s = tkv_norm.clamp(min=1e-20).log().gather(1, idx_in_k)
                    topk_i = sampled
                else:
                    sampled = torch.multinomial(probs_leaves, k_this,
                                                replacement=False)        # [n_leaves, k]
                    topk_i = sampled
                    topk_s = (new_lp.gather(1, sampled))
                    cand_probs = probs_leaves.gather(1, sampled)
            else:
                topk_s, topk_i = new_lp.topk(k_this, dim=-1)
                if need_probs:
                    probs_T = F.softmax(new_logits / T_draft, dim=-1)
                    cand_probs = probs_T.gather(1, topk_i)
                else:
                    cand_probs = torch.ones_like(topk_s)

            parent_scores_t = torch.tensor(
                [node_scores[i] for i in leaf_indices],
                device=device, dtype=torch.float32)
            cum_scores = parent_scores_t.unsqueeze(1) + topk_s
            flat_scores = cum_scores.reshape(-1)
            flat_tokens = topk_i.reshape(-1)
            flat_probs = cand_probs.reshape(-1)
            leaf_idx_t = torch.tensor(
                leaf_indices, device=device, dtype=torch.long)
            flat_parents = leaf_idx_t.unsqueeze(1).expand(-1, k_this).reshape(-1)

            remaining = budget - len(node_tokens)
            keep = min(remaining, flat_scores.numel())
            if keep <= 0:
                depth_counts.append(0)
                break

            top_vals, top_idx = flat_scores.topk(keep)
            sel_tokens = flat_tokens[top_idx]
            sel_parents = flat_parents[top_idx]
            sel_probs = flat_probs[top_idx]
            sel_t = sel_tokens.tolist()
            sel_p = sel_parents.tolist()
            sel_s = top_vals.tolist()
            sel_pr = sel_probs.tolist()
            for i in range(keep):
                node_tokens.append(sel_t[i])
                node_parents.append(sel_p[i])
                node_depths.append(depth)
                node_scores.append(sel_s[i])
                node_probs.append(sel_pr[i])
            depth_counts.append(keep)

    draft_time = time.time() - t0

    probs_t = None
    if need_probs or draft_mode == 'sample':
        probs_t = torch.tensor(node_probs, device=device, dtype=torch.float32)
    tree = DraftTree(
        tokens=torch.tensor(node_tokens, device=device),
        parents=torch.tensor(node_parents, device=device),
        depths=torch.tensor(node_depths, device=device),
        scores=torch.tensor(node_scores, device=device, dtype=torch.float32),
        probs=probs_t,
        depth_counts=depth_counts,
        device=device,
    )
    return tree, draft_time


@torch.inference_mode()
def verify_tree_step(
    target_model, state: VerifyState, tree: DraftTree,
    temperature: float = 0.0,
    verify_mode: str = 'auto',
    target_top_k: int = 0,
) -> Tuple[torch.Tensor, int, float, VerifyState, List[int]]:
    """Incremental tree verification.

    verify_mode:
      'auto'   — greedy if T=0, simple sample otherwise (legacy)
      'greedy' — argmax(target_logits) == tree token
      'simple' — r ≤ p_target(token)
      'ratio'  — r ≤ p_target(token) / p_draft(token)  (exact rejection sampling).
                 Requires tree.probs (build with need_probs=True or draft_mode='sample').

    Input  = [pending, tree.tokens], length n+1.
    Output new_tokens = [accepted_path] ++ [correction], length n_acc+1.
    Cache is truncated to [prefix][pending][accepted_path]; correction
    becomes the new pending (not yet in cache).

    Also returns best_path (list of tree node indices, length n_acc) so the
    caller can apply the same truncation to the draft cache when using
    build_draft_tree_incremental.

    temperature=0.0 → greedy argmax-match verify (default).
    temperature>0  → EAGLE3-style sample verify: accept iff r ≤ p_target(tok)
                     with p_target = softmax(logits/T); correction sampled
                     from softmax(target/T) at the prediction slot.
    """
    device = tree.device
    n = tree.n_nodes
    cache_len = state.cache_len
    tdtype = next(target_model.parameters()).dtype
    min_val = torch.finfo(tdtype).min

    if n == 0:
        pos = torch.tensor([[cache_len]], device=device, dtype=torch.long)
        t0 = time.time()
        out = target_model(
            input_ids=state.pending_token,
            position_ids=pos,
            past_key_values=state.cache,
            use_cache=True,
        )
        target_time = time.time() - t0
        eff_v0 = verify_mode
        if eff_v0 == 'auto':
            eff_v0 = 'greedy' if temperature == 0.0 else 'simple'
        if eff_v0 == 'greedy':
            correction = out.logits[:, -1, :].argmax(-1, keepdim=True)
        else:
            T_v = max(temperature, 1e-6)
            probs = torch.softmax(out.logits[:, -1, :].float() / T_v, dim=-1)
            if target_top_k > 0:
                tv, ti = probs.topk(target_top_k, dim=-1)
                tn = tv / tv.sum(-1, keepdim=True).clamp(min=1e-20)
                probs = torch.zeros_like(probs).scatter_(-1, ti, tn)
            correction = torch.multinomial(probs[0], 1).view(1, 1)
        new_state = VerifyState(
            cache=out.past_key_values,
            cache_len=cache_len + 1,
            pending_token=correction,
        )
        return correction, 0, target_time, new_state, []

    input_toks = torch.cat([state.pending_token, tree.tokens.unsqueeze(0)], dim=1)

    pos = torch.empty((1, n + 1), dtype=torch.long, device=device)
    pos[0, 0] = cache_len
    pos[0, 1:] = cache_len + 1 + tree.depths.long()

    parents_cpu = tree.parents.cpu().tolist()
    total = cache_len + n + 1
    mask_bool = torch.zeros(n + 1, total, dtype=torch.bool)
    mask_bool[0, :cache_len + 1] = True
    mask_bool[1:, :cache_len + 1] = True
    for i in range(n):
        j = i
        while j >= 0:
            mask_bool[i + 1, cache_len + 1 + j] = True
            j = parents_cpu[j]

    attn_mask = torch.where(
        mask_bool.to(device, non_blocking=True),
        torch.zeros((), dtype=tdtype, device=device),
        torch.full((), min_val, dtype=tdtype, device=device),
    ).unsqueeze(0).unsqueeze(0)

    t0 = time.time()
    out = target_model(
        input_ids=input_toks,
        attention_mask=attn_mask,
        position_ids=pos,
        past_key_values=state.cache,
        use_cache=True,
    )
    cache = out.past_key_values
    logits = out.logits[0].float()
    argmax = logits.argmax(dim=-1)
    target_time = time.time() - t0

    argmax_cpu = argmax.cpu().tolist()
    tree_tokens_cpu = tree.tokens.cpu().tolist()

    children = [[] for _ in range(n)]
    for i in range(n):
        p = parents_cpu[i]
        if p >= 0:
            children[p].append(i)
    leaves_idx = [i for i in range(n) if len(children[i]) == 0]
    paths = []
    for leaf in leaves_idx:
        path = []
        j = leaf
        while j >= 0:
            path.append(j)
            j = parents_cpu[j]
        path.reverse()
        paths.append(path)

    eff_verify = verify_mode
    if eff_verify == 'auto':
        eff_verify = 'greedy' if temperature == 0.0 else 'simple'

    if eff_verify in ('simple', 'ratio', 'ratio_fix'):
        T_v = max(temperature, 1e-6)
        probs = torch.softmax(logits / T_v, dim=-1)
        if target_top_k > 0:
            tv, ti = probs.topk(target_top_k, dim=-1)
            tn = tv / tv.sum(-1, keepdim=True).clamp(min=1e-20)
            probs = torch.zeros_like(probs).scatter_(-1, ti, tn)
        probs_cpu = probs.cpu()
        is_ratio = eff_verify in ('ratio', 'ratio_fix')
        tree_probs_cpu = tree.probs.cpu().tolist() if (is_ratio and tree.probs is not None) else None
        accept_node = [False] * n
        for i in range(n):
            p = parents_cpu[i]
            pos_for_check = 0 if p < 0 else (1 + p)
            token = tree_tokens_cpu[i]
            p_t = probs_cpu[pos_for_check, token].item()
            if eff_verify == 'simple':
                accept_node[i] = (random.random() <= p_t)
            else:  # 'ratio' or 'ratio_fix'
                p_d = tree_probs_cpu[i] if tree_probs_cpu else 1.0
                ratio = min(1.0, p_t / max(p_d, 1e-20))
                accept_node[i] = (random.random() <= ratio)
        best_accepted = 0
        best_path = []
        for path in paths:
            accepted = 0
            for node_idx in path:
                if accept_node[node_idx]:
                    accepted += 1
                else:
                    break
            if accepted > best_accepted:
                best_accepted = accepted
                best_path = path[:accepted]
        # Determine correction slot & compute correction distribution
        if best_accepted > 0:
            last_idx = best_path[-1]
            correction_slot = 1 + last_idx
            accepted_toks = tree.tokens[best_path].unsqueeze(0)
        else:
            correction_slot = 0
            accepted_toks = torch.empty(1, 0, dtype=torch.long, device=device)
        if eff_verify == 'ratio_fix' and tree_probs_cpu is not None:
            # Approximate p_draft distribution at rejection slot as: sibling tokens
            # with their sampled p_draft, others = 0. Then correction ~ normalize(
            # max(0, p_target - p_draft_approx)). If reject with no accepted, the
            # rejection slot's parent is "root"; we approximate with all root-level
            # sampled tokens as siblings.
            if best_accepted > 0:
                parent_of_rejected = best_path[-1]  # last accepted node = parent of rejection
                sibling_nodes = [i for i in range(n) if parents_cpu[i] == parent_of_rejected]
            else:
                sibling_nodes = [i for i in range(n) if parents_cpu[i] < 0]
            draft_approx = torch.zeros_like(probs[correction_slot])
            for sib in sibling_nodes:
                tok = tree_tokens_cpu[sib]
                draft_approx[tok] += tree_probs_cpu[sib]
            p_diff = torch.clamp(probs[correction_slot] - draft_approx, min=0.0)
            denom = p_diff.sum()
            if denom > 1e-20:
                correction = torch.multinomial(p_diff / denom, 1).view(1, 1)
            else:
                correction = torch.multinomial(probs[correction_slot], 1).view(1, 1)
        else:
            correction = torch.multinomial(probs[correction_slot], 1).view(1, 1)
    else:
        best_accepted = 0
        best_path = []
        for path in paths:
            root_idx = path[0]
            if argmax_cpu[0] != tree_tokens_cpu[root_idx]:
                continue
            accepted = 1
            for pi in range(1, len(path)):
                prev_idx = path[pi - 1]
                curr_idx = path[pi]
                if argmax_cpu[1 + prev_idx] == tree_tokens_cpu[curr_idx]:
                    accepted += 1
                else:
                    break
            if accepted > best_accepted:
                best_accepted = accepted
                best_path = path[:accepted]

        if best_accepted > 0:
            last_idx = best_path[-1]
            correction = argmax[1 + last_idx].view(1, 1)
            accepted_toks = tree.tokens[best_path].unsqueeze(0)
        else:
            correction = argmax[0].view(1, 1)
            accepted_toks = torch.empty(1, 0, dtype=torch.long, device=device)
    new_tokens = torch.cat([accepted_toks, correction], dim=1)

    sel = list(range(cache_len + 1))
    sel.extend(cache_len + 1 + idx for idx in best_path)
    sel_idx = torch.tensor(sel, device=device, dtype=torch.long)
    _cache_reorder(cache, sel_idx)

    new_state = VerifyState(
        cache=cache,
        cache_len=cache_len + 1 + best_accepted,
        pending_token=correction,
    )
    return new_tokens, best_accepted, target_time, new_state, best_path
