"""
Optimized tree spec decode for speedup benchmarking.

Differences from tree_spec_decode.py:
  #4 GPU-vectorized 4D mask construction (no CPU build, no H2D copy)
  #5 GPU-vectorized accept-path finding (one parent-walk in tensor ops, no
     Python loop over leaf paths)
  #6 Optional torch.compile(mode="reduce-overhead") wrapping for draft model
     (enables internal CUDA Graph capture); enabled by passing compile=True

Reuses DraftTree, VerifyState, _cache_reorder, prefill_minus_one,
early_exit_context from the base module so this file stays focused.

Not used by training or by the existing eval pipeline — only by
eval_partial_prefix_fast.py.
"""
import time
from typing import List, Tuple

import torch
import torch.nn.functional as F

from tree_spec_decode import (
    DraftTree,
    VerifyState,
    _cache_reorder,
    early_exit_context,
    prefill_minus_one,
)


# ─── #4: GPU mask construction ──────────────────────────────────────────────

def _build_draft_mask_gpu(parents_t, leaf_t, prefix_len, cache_len,
                          total_kv, max_walk_depth, dtype, device):
    """Per-depth draft-tree 4D mask, built directly on GPU.

    Rows = current depth's leaves (n_leaves).
    Cols = total_kv positions = [prefix (prefix_len) | tree cache (cache_len-prefix_len) | new leaves at cache_len..cache_len+n_leaves-1].
    Each row i: attend prefix [0..prefix_len-1] + ancestor chain of leaf_t[i]
                 in tree cache + own new-leaf position cache_len+i.
    """
    n_leaves = leaf_t.shape[0]
    min_val = torch.finfo(dtype).min

    mask_bool = torch.zeros(n_leaves, total_kv, dtype=torch.bool, device=device)
    mask_bool[:, :prefix_len] = True
    row_idx = torch.arange(n_leaves, device=device)
    mask_bool[row_idx, cache_len + row_idx] = True

    current = leaf_t.clone()
    for _ in range(max_walk_depth + 1):
        valid = current >= 0
        if not bool(valid.any()):
            break
        safe = current.clamp(min=0)
        vrows = row_idx[valid]
        vcols = prefix_len + safe[valid]
        mask_bool[vrows, vcols] = True
        current = torch.where(valid, parents_t[safe], current)

    return torch.where(
        mask_bool,
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), min_val, dtype=dtype, device=device),
    ).unsqueeze(0).unsqueeze(0)


def _build_verify_mask_gpu(parents_t, n, cache_len, total,
                           max_walk_depth, dtype, device):
    """Verify-step 4D mask, built directly on GPU.

    Layout (n_q = n+1):
      row 0   = pending token (sees only prefix [0..cache_len-1] + self at cache_len)
      row i+1 = tree node i   (sees prefix + pending + ancestor chain of node i)

    Cols: [prefix (cache_len) | pending (1) | tree nodes (n)] → total = cache_len+1+n.
    """
    min_val = torch.finfo(dtype).min

    mask_bool = torch.zeros(n + 1, total, dtype=torch.bool, device=device)
    mask_bool[:, :cache_len + 1] = True

    node_idx = torch.arange(n, device=device)
    current = node_idx.clone()
    for _ in range(max_walk_depth + 1):
        valid = current >= 0
        if not bool(valid.any()):
            break
        safe = current.clamp(min=0)
        vrows = node_idx[valid] + 1
        vcols = cache_len + 1 + safe[valid]
        mask_bool[vrows, vcols] = True
        current = torch.where(valid, parents_t[safe], current)

    return torch.where(
        mask_bool,
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), min_val, dtype=dtype, device=device),
    ).unsqueeze(0).unsqueeze(0)


# ─── #5: GPU accept-path finding ────────────────────────────────────────────

def _find_best_path_gpu(argmax, tree):
    """Vectorized longest-accepted-prefix path search.

    For each tree node i:
      self_accept[i] = (target argmax at i's parent position) == tree.tokens[i]
        - root (parent==-1): parent position is the pending logit, i.e. argmax[0]
        - non-root: argmax[1 + parent_idx]
      cum_accept[i]  = self_accept[i] AND cum_accept[parent_of_i]
      path_length[i] = depth[i] + 1 if cum_accept[i] else 0

    Best leaf = argmax over path_length. Walk parents back from that node to
    reconstruct the path (Python loop bounded by max depth ≤ γ).
    """
    n = tree.n_nodes
    device = tree.device
    parents = tree.parents
    tokens = tree.tokens
    depths = tree.depths

    parent_pos = torch.where(parents >= 0, 1 + parents, torch.zeros_like(parents))
    self_accept = argmax[parent_pos] == tokens

    cum_accept = self_accept.clone()
    max_d = int(depths.max().item())
    for d in range(1, max_d + 1):
        mask_d = depths == d
        if not bool(mask_d.any()):
            continue
        idx_d = mask_d.nonzero(as_tuple=True)[0]
        par_idx = parents[idx_d]
        cum_accept[idx_d] = cum_accept[idx_d] & cum_accept[par_idx]

    path_length = torch.where(
        cum_accept, depths + 1, torch.zeros_like(depths))
    best_len = int(path_length.max().item())
    if best_len == 0:
        return 0, []

    best_node = int(path_length.argmax().item())
    parents_cpu = parents.cpu().tolist()
    path = []
    j = best_node
    while j >= 0:
        path.append(j)
        j = parents_cpu[j]
    path.reverse()
    return best_len, path


# ─── Optimized build_draft_tree (uses GPU mask) ─────────────────────────────

def build_draft_tree_incremental_fast(
    model, state: VerifyState, exit_layer: int,
    gamma: int = 7, top_k=10, budget: int = 63,
) -> Tuple[DraftTree, float]:
    if isinstance(top_k, int):
        top_k_list = [top_k] * gamma
    else:
        top_k_list = list(top_k)
        if len(top_k_list) < gamma:
            top_k_list = top_k_list + [top_k_list[-1]] * (gamma - len(top_k_list))

    device = state.pending_token.device
    mdtype = next(model.parameters()).dtype

    L0 = state.cache_len
    prefix_len = L0 + 1

    node_tokens: List[int] = []
    node_parents: List[int] = []
    node_depths: List[int] = []
    node_scores: List[float] = []
    depth_counts: List[int] = []

    t0 = time.time()

    with early_exit_context(model, exit_layer):
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
        log_probs = F.log_softmax(last_logits, dim=-1)
        n_roots = min(top_k_list[0], budget)
        topk_s, topk_i = log_probs.topk(n_roots, dim=-1)
        roots_tok = topk_i[0].tolist()
        roots_sc = topk_s[0].tolist()
        for r in range(n_roots):
            node_tokens.append(roots_tok[r])
            node_parents.append(-1)
            node_depths.append(0)
            node_scores.append(roots_sc[r])
        depth_counts.append(n_roots)

        for depth in range(1, gamma + 1):
            n_leaves = depth_counts[-1]
            if n_leaves == 0:
                break

            leaf_start = len(node_tokens) - n_leaves
            leaf_t = torch.arange(
                leaf_start, leaf_start + n_leaves,
                device=device, dtype=torch.long)

            leaf_toks = torch.tensor(
                node_tokens[leaf_start:leaf_start + n_leaves],
                device=device, dtype=torch.long).unsqueeze(0)
            pos_ids = torch.full(
                (1, n_leaves), prefix_len + depth - 1,
                device=device, dtype=torch.long)

            cache_len = state.cache_len
            total_kv = cache_len + n_leaves

            parents_t = torch.tensor(
                node_parents, device=device, dtype=torch.long)
            mask = _build_draft_mask_gpu(
                parents_t, leaf_t, prefix_len, cache_len, total_kv,
                max_walk_depth=depth, dtype=mdtype, device=device)

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
                break

            new_logits = fout.logits[0].float()
            new_lp = F.log_softmax(new_logits, dim=-1)

            k_this = top_k_list[depth]
            topk_s, topk_i = new_lp.topk(k_this, dim=-1)

            parent_scores_t = torch.tensor(
                [node_scores[i]
                 for i in range(leaf_start, leaf_start + n_leaves)],
                device=device, dtype=torch.float32)
            cum_scores = parent_scores_t.unsqueeze(1) + topk_s
            flat_scores = cum_scores.reshape(-1)
            flat_tokens = topk_i.reshape(-1)
            leaf_idx_t = torch.arange(
                leaf_start, leaf_start + n_leaves,
                device=device, dtype=torch.long)
            flat_parents = leaf_idx_t.unsqueeze(1).expand(-1, k_this).reshape(-1)

            remaining = budget - len(node_tokens)
            keep = min(remaining, flat_scores.numel())
            if keep <= 0:
                depth_counts.append(0)
                break

            top_vals, top_idx = flat_scores.topk(keep)
            sel_tokens = flat_tokens[top_idx]
            sel_parents = flat_parents[top_idx]
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


# ─── Optimized verify_tree_step (GPU mask + GPU accept-path) ────────────────

@torch.inference_mode()
def verify_tree_step_fast(target_model, state: VerifyState, tree: DraftTree):
    device = tree.device
    n = tree.n_nodes
    cache_len = state.cache_len
    tdtype = next(target_model.parameters()).dtype

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
        correction = out.logits[:, -1, :].argmax(-1, keepdim=True)
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

    total = cache_len + n + 1
    max_d = int(tree.depths.max().item())
    attn_mask = _build_verify_mask_gpu(
        tree.parents, n, cache_len, total,
        max_walk_depth=max_d, dtype=tdtype, device=device)

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

    best_accepted, best_path = _find_best_path_gpu(argmax, tree)

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


# ─── Outer decode loop (fast variant) ──────────────────────────────────────

@torch.inference_mode()
def spec_decode_tree_smalllm_fast(
    target_model, draft_model, input_ids, max_new_tokens,
    exit_layer, gamma, top_k, budget, eos_token_id=None,
):
    """Drop-in replacement for spec_decode_tree_smalllm, using fast tree+verify."""
    device = input_ids.device
    cur_ids = input_ids.clone()
    total_tokens = 0
    total_rounds = 0
    total_accepted = 0
    total_draft_time = 0.0
    total_target_time = 0.0

    t0 = time.time()
    draft_state = prefill_minus_one(draft_model, cur_ids)
    total_draft_time += time.time() - t0
    t0 = time.time()
    target_state = prefill_minus_one(target_model, cur_ids)
    total_target_time += time.time() - t0

    while total_tokens < max_new_tokens:
        pre_len = draft_state.cache_len
        tree, dt = build_draft_tree_incremental_fast(
            draft_model, draft_state, exit_layer, gamma, top_k, budget)
        total_draft_time += dt

        new_tokens, n_acc, tt, target_state, best_path = verify_tree_step_fast(
            target_model, target_state, tree)
        total_target_time += tt

        sel = list(range(pre_len + 1))
        sel.extend(pre_len + 1 + idx for idx in best_path)
        sel_idx = torch.tensor(sel, device=device, dtype=torch.long)
        _cache_reorder(draft_state.cache, sel_idx)
        draft_state.cache_len = pre_len + 1 + n_acc
        draft_state.pending_token = new_tokens[:, -1:].contiguous()

        cur_ids = torch.cat([cur_ids, new_tokens], dim=1)
        total_tokens += new_tokens.shape[1]
        total_rounds += 1
        total_accepted += n_acc

        if eos_token_id is not None:
            eos_list = (eos_token_id if isinstance(eos_token_id, list)
                        else [eos_token_id])
            if any(t in new_tokens[0].tolist() for t in eos_list):
                break

    total_time = total_draft_time + total_target_time
    prompt_len = input_ids.shape[1]
    return {
        "total_tokens": total_tokens,
        "total_rounds": total_rounds,
        "total_accepted": total_accepted,
        "mean_alpha": total_accepted / max(total_rounds, 1),
        "tokens_per_sec": total_tokens / max(total_time, 1e-6),
        "total_time": total_time,
        "draft_time": total_draft_time,
        "target_time": total_target_time,
        "output_ids": cur_ids[0, prompt_len:].tolist(),
    }


# ─── #6: torch.compile wrapper (CUDA Graph via mode="reduce-overhead") ─────

def maybe_compile(model, enable: bool):
    """Optionally wrap model with torch.compile to enable CUDA Graph capture.

    `mode="reduce-overhead"` triggers cudagraph trees inside inductor:
      - First call per unique input shape: compile + capture (slow)
      - Subsequent calls with same shape: replay (fast)

    Caveats for HF models:
      - Shape changes (variable n_leaves per depth) trigger recompiles
      - DynamicCache mutation may cause graph break — use StaticCache for
        best results (requires more wiring; not done here)
      - First few rounds will be slower due to warmup compiles

    Use only with stable shapes. For variable-budget trees, may net negative.
    """
    if not enable:
        return model
    return torch.compile(model, mode="reduce-overhead", fullgraph=False)
