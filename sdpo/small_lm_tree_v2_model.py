"""Small-LM tree on-policy rollout training (v2: arbitrary top_k_per_depth).

Matches the production EVAL tree shape (see sdpo/tree_spec_decode.py):
  - gamma = len(top_k_per_depth)
  - top_k_per_depth[d] = branching factor at depth d
  - No budget pruning during training (full tree).

Example (production eval config):
  top_k_per_depth = [4, 3, 2, 1, 1, 1, 1]
  → depth_counts = [4, 12, 24, 24, 24, 24, 24], N = 136 nodes, 24 paths

Per training step, per sample:
  1. Target prefix forward (frozen, KV cache, no grad).
  2. Draft prefix forward (KV cache).
  3. Pick anchor position (random valid in loss_mask).
  4. Draft tree rollout, gamma sequential forwards:
     Step 0: read prefix logits → top-K[0] roots
     Step d ≥ 1: forward depth-(d-1) tokens (with tree-attn mask) → top-K[d]
       children each.
  5. Target tree verify (single forward, tree attention mask).
  6. Loss, by --eal_mode:
     sample_minpq (current Tree EAL): the draft SAMPLES k distinct children per
       node. alpha_v = min(p_v, q_v) with p = frozen target prob of the drafted
       token and q = draft prob (carries the gradient where q < p).
       E[L] = sum over nodes of prod_{v on path} alpha_v (siblings are mutually
       exclusive under one target draw per slot). loss = -E[L].
       Eval must use --draft-mode sample --verify-mode ratio.
     soft_topk: greedy top-k tree, -E[L] on the Top-K path union with forward
       alpha = p_target(token) and a p - q straight-through backward.
     budget / path_topk: anchor_kl_coef * KL - eal_aux_coef * E[L_tree], where
       E[L_tree] uses the independent-coin recursion of verify_tree_step simple.
"""

import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None
try:
    from transformers.cache_utils import DynamicLayer
except ImportError:
    DynamicLayer = object


def _build_tree_topology(top_k_per_depth):
    """Build tree topology for arbitrary top_k_per_depth list."""
    gamma = len(top_k_per_depth)
    depth_counts = [top_k_per_depth[0]]
    for d in range(1, gamma):
        depth_counts.append(depth_counts[d-1] * top_k_per_depth[d])
    N = sum(depth_counts)
    parents = [-1] * N
    depths = [0] * N
    depth_starts = [0] * gamma
    for d in range(1, gamma):
        depth_starts[d] = depth_starts[d-1] + depth_counts[d-1]
    for d in range(1, gamma):
        for offset in range(depth_counts[d]):
            i = depth_starts[d] + offset
            depths[i] = d
            parent_offset = offset // top_k_per_depth[d]
            parents[i] = depth_starts[d-1] + parent_offset
    n_paths = depth_counts[-1]
    paths = []
    for k in range(n_paths):
        cur = depth_starts[gamma-1] + k
        path = [cur]
        while parents[cur] >= 0:
            cur = parents[cur]
            path.append(cur)
        path.reverse()
        paths.append(tuple(path))
    return tuple(parents), tuple(depths), tuple(paths), tuple(depth_counts), tuple(depth_starts), N


def _extract_kv_lists(past_kv):
    if past_kv is None:
        return [], []
    if hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache"):
        return list(past_kv.key_cache), list(past_kv.value_cache)
    if hasattr(past_kv, "layers"):
        keys, values = [], []
        for layer in past_kv.layers:
            if hasattr(layer, "keys"):
                keys.append(layer.keys); values.append(layer.values)
            elif hasattr(layer, "key_cache"):
                keys.append(layer.key_cache); values.append(layer.value_cache)
            else:
                raise RuntimeError(f"Unknown LayerCache: {type(layer)}")
        return keys, values
    if isinstance(past_kv, (tuple, list)):
        return [l[0] for l in past_kv], [l[1] for l in past_kv]
    raise RuntimeError(f"Unknown past_key_values type: {type(past_kv)}")


def _slice_cache_to_bs1(keys, values, b, kv_len):
    new_cache = DynamicCache()
    for layer_idx in range(len(keys)):
        k_sliced = keys[layer_idx][b:b+1, :, :kv_len, :].contiguous()
        v_sliced = values[layer_idx][b:b+1, :, :kv_len, :].contiguous()
        new_cache.update(k_sliced, v_sliced, layer_idx)
    return new_cache


def _pad_cache_batch(keys, values, batch_idx, kv_lens):
    """Slice each sample's prefix cache to its anchor and right-pad to max_kv.

    Real keys stay at positions 0..kv_len-1 with the RoPE they were written
    with. Slots kv_len..max_kv-1 keep that sample's later keys instead of
    zeros; the tree attention mask hides them, so the output is unchanged.
    """
    max_kv = max(kv_lens)
    b_t = torch.tensor(batch_idx, device=keys[0].device, dtype=torch.long)
    new_cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(zip(keys, values)):
        new_cache.update(k[:, :, :max_kv].index_select(0, b_t),
                         v[:, :, :max_kv].index_select(0, b_t), layer_idx)
    return new_cache, max_kv


class _PreallocLayer(DynamicLayer):
    """Cache layer with room for `extra` tokens after the prefix: update
    writes in place and returns views, where DynamicLayer re-concatenates the
    whole cache on every call."""

    def __init__(self, keys, values, length):
        super().__init__()
        self.buf_k, self.buf_v, self.cur = keys, values, length
        self.dtype, self.device = keys.dtype, keys.device
        self.keys, self.values = keys[:, :, :length], values[:, :, :length]
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        n = key_states.size(-2)
        self.buf_k[:, :, self.cur:self.cur + n] = key_states
        self.buf_v[:, :, self.cur:self.cur + n] = value_states
        self.cur += n
        self.keys = self.buf_k[:, :, :self.cur]
        self.values = self.buf_v[:, :, :self.cur]
        return self.keys, self.values

    def get_seq_length(self):
        return self.cur


def _prealloc_cache_batch(keys, values, batch_idx, kv_lens, extra):
    """_pad_cache_batch with room for `extra` appended tokens per layer."""
    max_kv = max(kv_lens)
    b_t = torch.tensor(batch_idx, device=keys[0].device, dtype=torch.long)
    cache = DynamicCache()
    layers = []
    for k, v in zip(keys, values):
        shape = (len(batch_idx), k.size(1), max_kv + extra, k.size(3))
        bk, bv = k.new_empty(shape), v.new_empty(shape)
        torch.index_select(k[:, :, :max_kv], 0, b_t, out=bk[:, :, :max_kv])
        torch.index_select(v[:, :, :max_kv], 0, b_t, out=bv[:, :, :max_kv])
        layers.append(_PreallocLayer(bk, bv, max_kv))
    cache.layers = layers
    return cache, max_kv


def _gqa_fold_sdpa(module, query, key, value, attention_mask, dropout=0.0,
                   scaling=None, is_causal=None, **kwargs):
    """SDPA for grouped-query attention without repeating K/V: the g query
    heads sharing a KV head are folded into the query length. Same output
    layout as transformers' sdpa_attention_forward."""
    B, Hq, q, D = query.shape
    g = Hq // key.size(1)
    qf = query.reshape(B, key.size(1), g * q, D)
    mask = attention_mask
    if mask is not None and g > 1:
        mask = mask.unsqueeze(2).expand(-1, -1, g, -1, -1).reshape(
            mask.size(0), mask.size(1), g * q, mask.size(-1))
    out = F.scaled_dot_product_attention(qf, key, value, attn_mask=mask,
                                         dropout_p=0.0, scale=scaling)
    return out.reshape(B, Hq, q, D).transpose(1, 2).contiguous(), None


class _fold_gqa_attention:
    """Routes 'sdpa' to _gqa_fold_sdpa inside the block."""

    def __enter__(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        self.mapping = ALL_ATTENTION_FUNCTIONS._global_mapping
        self.prev = self.mapping["sdpa"]
        self.mapping["sdpa"] = _gqa_fold_sdpa

    def __exit__(self, *a):
        self.mapping["sdpa"] = self.prev


def _prefix_allowed(kv_lens, max_kv, device):
    """[B, max_kv] bool, True on each sample's real prefix positions."""
    kv_t = torch.tensor(kv_lens, device=device, dtype=torch.long)
    return torch.arange(max_kv, device=device).view(1, -1) < kv_t.view(-1, 1)


def soft_topk_score(logits, token_ids, k, tau):
    """Soft score that token_ids sit in the top-k of logits.

    The threshold for each token is the k-th largest logit among the others,
    so a token that is exactly rank k is not compared with itself.
    logits: [B, G, V], token_ids: [B, G, C].
    """
    kk = max(int(k), 1)
    vals, idx = logits.topk(kk + 1, dim=-1)
    in_top = (token_ids.unsqueeze(-1) == idx[..., :kk].unsqueeze(-2)).any(-1)
    boundary = vals[..., kk - 1]
    nxt = vals[..., kk]
    thr = torch.where(in_top, nxt.unsqueeze(-1), boundary.unsqueeze(-1))
    z = logits.gather(-1, token_ids)
    return torch.sigmoid((z - thr) / max(float(tau), 1e-6))


def tree_expected_accepted_length(alpha, parents):
    """E[L_tree] for this repo's verify_tree_step simple rule.

    alpha:   [B, N] P(coin of node accepts) = softmax(target / T)[token]
    parents: [B, N] long, -1 for roots. Sibling order does not affect E[L].

    P(M_v <= 0) = 1 - alpha_v
    P(M_v <= t) = (1 - alpha_v) + alpha_v * prod_children P(M_c <= t-1)
    empty product = 1, so a leaf has P(M <= t) = 1 for every t >= 1.
    P(L >= d) = 1 - prod_roots P(M_r <= d-1)
    E[L] = sum_d P(L >= d)
    The +1 correction token is not part of L.
    """
    B, N = alpha.shape
    if N == 0:
        return alpha.new_zeros(B)
    device = alpha.device
    alpha = alpha.clamp(0.0, 1.0)

    node_depth = torch.zeros(B, N, dtype=torch.long, device=device)
    seen_parent = parents.clone()
    for _ in range(N):
        has_p = seen_parent >= 0
        if not bool(has_p.any()):
            break
        node_depth = node_depth + has_p.long()
        seen_parent = torch.where(
            has_p,
            parents.gather(1, seen_parent.clamp(min=0)),
            seen_parent,
        )
    max_depth = int(node_depth.max().item())
    gamma = max_depth + 1

    cdf = alpha.new_empty(gamma, B, N)
    cdf[0] = 1.0 - alpha
    node_ids = torch.arange(N, device=device)
    is_child = parents.unsqueeze(1) == node_ids.view(1, N, 1)
    for t in range(1, gamma):
        child_cdf = cdf[t - 1].unsqueeze(1).expand(B, N, N)
        prod = torch.where(is_child, child_cdf, torch.ones_like(child_cdf)).prod(dim=-1)
        cdf[t] = (1.0 - alpha) + alpha * prod

    root = parents < 0
    eal = alpha.new_zeros(B)
    for d in range(1, gamma + 1):
        vals = torch.where(root, cdf[d - 1], torch.ones_like(cdf[d - 1]))
        eal = eal + (1.0 - vals.prod(dim=-1))
    return eal


def _center_by_group(x, group):
    """Standardise x within each group of draws sharing one anchor.

    Groups with a single draw get zero: there is nothing to compare them to, so
    they contribute only the value term.
    """
    if group is None:
        group = list(range(x.numel()))
    g = torch.as_tensor(group, device=x.device, dtype=torch.long)
    n_g = int(g.max().item()) + 1 if g.numel() else 0
    cnt = x.new_zeros(n_g).index_add_(0, g, torch.ones_like(x))
    mean = x.new_zeros(n_g).index_add_(0, g, x) / cnt.clamp(min=1.0)
    dev = x - mean[g]
    var = x.new_zeros(n_g).index_add_(0, g, dev * dev) / (cnt - 1).clamp(min=1.0)
    out = dev / var[g].sqrt().clamp(min=1e-3)
    return torch.where(cnt[g] > 1, out, torch.zeros_like(out))


def tree_path_sum_eal(alpha, parents):
    """E[L] = sum over nodes of prod of alpha along the root-to-node path.

    This is sum_d sum_{P in paths_d} prod_{v in P} alpha_v. It is exact when
    sibling acceptances are mutually exclusive (one target draw per slot), as
    with sampled drafting and alpha_v = min(p_v, q_v). Nodes must be ordered
    so every parent index is smaller than its children (depth order).
    """
    B, N = alpha.shape
    if N == 0:
        return alpha.new_zeros(B)
    cum = alpha.new_zeros(B, N)
    is_root = parents < 0
    # Depth-order processing: a node's parent is always to its left.
    # Iterate until every node has been assigned (gamma iterations).
    assigned = torch.zeros(B, N, dtype=torch.bool, device=alpha.device)
    for _ in range(N):
        parent_cum = cum.gather(1, parents.clamp(min=0))
        parent_ok = assigned.gather(1, parents.clamp(min=0)) | is_root
        new = parent_ok & ~assigned
        if not bool(new.any()):
            break
        val = torch.where(is_root, alpha, alpha * parent_cum)
        cum = torch.where(new, val, cum)
        assigned = assigned | new
    return cum.sum(dim=-1)


def tree_path_mean_eal(alpha, parents):
    """Mean over leaves of sum_d prod_{nodes on the path up to d} alpha.

    alpha[node] is the factor for entering that node. When siblings share a
    slot they carry the same alpha, and averaging leaves counts that shared
    prefix once per path instead of once per sibling.
    """
    B, N = alpha.shape
    if N == 0:
        return alpha.new_zeros(B)
    cum = alpha.new_zeros(B, N)
    total = alpha.new_zeros(B, N)
    is_root = parents < 0
    assigned = torch.zeros(B, N, dtype=torch.bool, device=alpha.device)
    for _ in range(N):
        parent_cum = cum.gather(1, parents.clamp(min=0))
        parent_total = total.gather(1, parents.clamp(min=0))
        parent_ok = assigned.gather(1, parents.clamp(min=0)) | is_root
        new = parent_ok & ~assigned
        if not bool(new.any()):
            break
        this = torch.where(is_root, alpha, alpha * parent_cum)
        cum = torch.where(new, this, cum)
        total = torch.where(
            new, torch.where(is_root, this, parent_total + this), total)
        assigned = assigned | new
    referred = torch.zeros(B, N, dtype=torch.bool, device=alpha.device)
    valid_parent = parents >= 0
    referred = referred.scatter(1, parents.clamp(min=0), valid_parent)
    is_leaf = ~referred & assigned
    denom = is_leaf.float().sum(dim=-1).clamp(min=1.0)
    return (total * is_leaf.float()).sum(dim=-1) / denom


def longest_path_from_accept(accept, parents):
    """Longest path whose nodes are all accepted. accept: [B, M] bool."""
    B, M = accept.shape
    lengths = accept.new_zeros(B, dtype=torch.float)
    for b in range(B):
        reached = [0] * M
        best = 0
        for i in range(M):
            parent_id = int(parents[b, i].item())
            ok = bool(accept[b, i].item())
            if parent_id >= 0:
                ok = ok and reached[parent_id] > 0
                reached[i] = reached[parent_id] + 1 if ok else 0
            else:
                reached[i] = 1 if ok else 0
            if reached[i] > best:
                best = reached[i]
        lengths[b] = best
    return lengths.mean()


@torch.no_grad()
def longest_path_by_depth(accept, parents, depth_starts, depth_counts):
    """longest_path_from_accept on a full tree whose nodes are stored depth by
    depth, one GPU op per depth instead of a Python loop over nodes."""
    reached = torch.zeros(accept.shape, device=accept.device, dtype=torch.long)
    for start, cnt in zip(depth_starts, depth_counts):
        end = start + cnt
        par = parents[:, start:end]
        prev = reached.gather(1, par.clamp(min=0))
        ok = accept[:, start:end] & ((par < 0) | (prev > 0))
        reached[:, start:end] = torch.where(
            ok, torch.where(par < 0, torch.ones_like(prev), prev + 1),
            torch.zeros_like(prev))
    return reached.max(dim=1).values.float().mean()


def longest_accepted_path(tokens, parents, target_logits):
    """Length of the longest path whose tokens match the target argmax.

    A node counts only when every ancestor was also accepted. This is the
    path eval keeps, not the tree-wide E[L].
    """
    accept = target_logits.detach().argmax(dim=-1).eq(tokens)
    B, M = tokens.shape
    lengths = tokens.new_zeros(B, dtype=torch.float)
    for b in range(B):
        reached = [0] * M
        best = 0
        for i in range(M):
            parent_id = int(parents[b, i].item())
            ok = bool(accept[b, i].item())
            if parent_id >= 0:
                ok = ok and reached[parent_id] > 0
                reached[i] = reached[parent_id] + 1 if ok else 0
            else:
                reached[i] = 1 if ok else 0
            if reached[i] > best:
                best = reached[i]
        lengths[b] = best
    return lengths.mean()


def induce_path_topk_union(full_parents, paths, path_scores, path_topk):
    """Top-K complete paths and the parent tensor of their unique-prefix union.

    paths:       [P, gamma] node ids, root to leaf, full unpruned topology.
    path_scores: [B, P] cumulative draft log-prob, one score per complete path.
    Returns parents [B, M] and gather_index [B, M]. gather_index is the original
    node id, or -1 for padding. Padding columns are extra roots; the caller sets
    their acceptance probability to 0 so they do not change E[L_tree].
    Shared prefixes appear once.
    """
    _B, n_paths = path_scores.shape
    k = min(int(path_topk), int(n_paths))
    if k < 1:
        raise ValueError(f"path_topk must be >= 1, got {path_topk}")
    device = path_scores.device
    sel = path_scores.topk(k, dim=-1).indices
    chains = paths.to(device)[sel]
    unions = [torch.unique(chains[b].reshape(-1), sorted=True) for b in range(chains.size(0))]
    max_n = max(int(u.numel()) for u in unions)
    parents_out = torch.full((chains.size(0), max_n), -1, device=device, dtype=torch.long)
    gather_index = torch.full((chains.size(0), max_n), -1, device=device, dtype=torch.long)
    parent_t = torch.tensor(full_parents, device=device, dtype=torch.long)
    for b, uniq in enumerate(unions):
        n = int(uniq.numel())
        gather_index[b, :n] = uniq
        old_parent = parent_t[uniq]
        mapped = torch.searchsorted(uniq, old_parent.clamp(min=0))
        parents_out[b, :n] = torch.where(
            old_parent >= 0, mapped, torch.full_like(mapped, -1))
    return parents_out, gather_index, sel


def _expected_acceptance(draft_logits, target_logits, temperature):
    """P(target accepts a token drawn from the draft) at each node."""
    T = max(float(temperature), 1e-6)
    q = torch.softmax(draft_logits / T, dim=-1)
    p = torch.softmax(target_logits / T, dim=-1)
    return (q * p).sum(-1)


def budget_columns_for_union(gather_index, n_prefix, last_sel):
    """Map full-tree node ids onto the budget-pruned logit tensor.

    Prefix columns keep their full-tree ids. Last-depth columns were reordered
    by last_sel. Padding ids stay clamped; the caller masks them.
    """
    idx = gather_index.clamp(min=0)
    if last_sel is None:
        return idx
    leaf_offset = (gather_index - n_prefix).clamp(min=0)
    match = last_sel.unsqueeze(1) == leaf_offset.unsqueeze(-1)
    k_index = match.to(dtype=torch.long).argmax(dim=-1)
    return torch.where(gather_index >= n_prefix, n_prefix + k_index, idx)


def acceptance_weighted_topk_ce(draft_logits, target_logits, alpha, parents,
                                temperature, k, valid):
    """Uniform cross-entropy onto the target's top-k at each drafted node.

    k is the branching factor at each node, an int or a [B, N] long tensor.
    A depth that drafts one token is trained on the target's top-1 only.
    Every valid node has the same weight. dE[L]/d alpha puts about half the
    loss on the four roots and about a tenth on the single-token tail, so the
    positions that make the accept length longer barely move. Chain KL trains
    every position equally; this matches that on the tree.
    alpha and parents are kept so the call site stays the logged tree.
    """
    del alpha, parents
    if valid.shape[:2] != draft_logits.shape[:2]:
        raise ValueError(
            "valid mask must align with draft logits, "
            f"got valid {tuple(valid.shape)} logits {tuple(draft_logits.shape)}")
    vocab = int(draft_logits.size(-1))
    if torch.is_tensor(k):
        if k.shape != draft_logits.shape[:2]:
            raise ValueError(
                "per-node k must align with draft logits, "
                f"got k {tuple(k.shape)} logits {tuple(draft_logits.shape[:2])}")
        node_k = k.to(device=draft_logits.device, dtype=torch.long).clamp(1, vocab)
    else:
        node_k = torch.full(draft_logits.shape[:2], min(int(k), vocab),
                            device=draft_logits.device, dtype=torch.long)
        if int(node_k.reshape(-1)[0]) < 1:
            raise ValueError(f"top-k must be >= 1, got {k}")
    k_max = int(node_k.max().item())
    T = max(float(temperature), 1e-6)
    log_q = torch.log_softmax(draft_logits / T, dim=-1)
    top_tokens = target_logits.detach().topk(k_max, dim=-1).indices
    log_top = log_q.gather(-1, top_tokens)
    keep = torch.arange(k_max, device=draft_logits.device).view(1, 1, -1) < node_k.unsqueeze(-1)
    ce = -(log_top.masked_fill(~keep, 0.0).sum(-1) / node_k.clamp(min=1).float())
    weights = valid.to(dtype=ce.dtype)
    return (weights * ce).sum() / weights.sum().clamp(min=1.0)


class _LogQSurrogate(torch.autograd.Function):
    """Forward returns q(token). Backward applies d(log q), recomputed from logits."""

    @staticmethod
    def forward(ctx, logits, token_ids):
        logits_f = logits.float()
        chosen = logits_f.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
        log_q = chosen - torch.logsumexp(logits_f, dim=-1)
        ctx.save_for_backward(logits, token_ids)
        return log_q.exp()

    @staticmethod
    def backward(ctx, grad_out):
        logits, token_ids = ctx.saved_tensors
        probs = torch.softmax(logits.float(), dim=-1)
        grad = probs * (-grad_out.unsqueeze(-1))
        grad.scatter_add_(-1, token_ids.unsqueeze(-1), grad_out.unsqueeze(-1))
        return grad, None


def accept_length_surrogate(draft_logits, target_logits, token_ids, temperature):
    """Forward α is the real acceptance p_target(drafted token).

    Backward sends p - q into the draft logits. The discrete token has no
    gradient, so without this the accept length cannot train the draft.
    Descent therefore raises words the target assigns more probability to.
    """
    T = max(float(temperature), 1e-6)
    log_q = torch.log_softmax(draft_logits.float() / T, dim=-1)
    p = torch.softmax(target_logits.detach().float() / T, dim=-1)
    hard = p.gather(-1, token_ids.long().unsqueeze(-1)).squeeze(-1)
    directed = (p * log_q).sum(dim=-1)
    return hard.detach() + directed - directed.detach()


def log_q_surrogate(logits, token_ids, temperature):
    """Forward value is q(token). Backward is d(log q).

    d(q)/dz = q(1-q) vanishes when the draft is confidently wrong. d(log q)/dz
    is one_hot - q, so the same -E[L] still pushes that token into the top-k.
    """
    T = max(float(temperature), 1e-6)
    token_ids = token_ids.long()
    if abs(T - 1.0) < 1e-8:
        return _LogQSurrogate.apply(logits, token_ids)
    return _LogQSurrogate.apply(logits.float() / T, token_ids)


def dense_position_eal(draft_logits, target_token, loss_mask, temperature,
                       chunk=64):
    """-q(target argmax) at every supervised position, with the log-q backward.

    A one-step chain has E[L] = q. The prefix forward already has these logits,
    so this covers every assistant token instead of one random anchor.
    """
    mask = loss_mask.bool()
    if mask.shape[:2] != draft_logits.shape[:2]:
        raise ValueError(
            "loss_mask and draft logits must share [B, L], "
            f"got {tuple(mask.shape)} {tuple(draft_logits.shape)}")
    _B, L, _vocab = draft_logits.shape
    picked = []
    for start in range(0, L, chunk):
        end = min(L, start + chunk)
        sl_mask = mask[:, start:end]
        if not bool(sl_mask.any()):
            continue
        alpha = log_q_surrogate(
            draft_logits[:, start:end], target_token[:, start:end], temperature)
        picked.append(alpha[sl_mask])
    if not picked:
        return draft_logits.new_zeros(())
    return -torch.cat(picked).mean()


def _dense_tree_chunk(drf_h, tgt_h, drf_head, tgt_head, k_max, temp=1.0):
    with torch.no_grad():
        tl = tgt_head(tgt_h).float()
        u = tl.argmax(dim=-1)
        logp = F.log_softmax(tl, dim=-1)
        del tl
    dl = drf_head(drf_h).float()
    logq = F.log_softmax(dl, dim=-1)
    kl = (logp.exp() * (logp - logq)).sum(-1)
    del logp, logq
    z_u = dl.gather(-1, u.unsqueeze(-1)).squeeze(-1)
    lse_t = torch.logsumexp(dl / temp, dim=-1)
    kth = dl.scatter(-1, u.unsqueeze(-1), torch.finfo(dl.dtype).min).topk(
        k_max, dim=-1).values
    return kl, z_u, kth, lse_t


@torch.no_grad()
def _renewal_start_prob(surv, mask):
    """[B, L] probability that a window starts at each position.
    surv[b, r, d-1] = P(L_r >= d), d = 1..gamma, non-increasing in d.
    rho is 1 at the first supervised position and
    rho_n = sum_r rho_r P(L_r = n - r - 1)  (one extra target token per window)."""
    import numpy as np
    B, L, gamma = surv.shape
    s_np = surv.double().cpu().numpy()
    mask_np = mask.cpu().numpy()
    rho = np.zeros((B, L))
    for b in range(B):
        pos = np.flatnonzero(mask_np[b])
        if pos.size == 0:
            continue
        s_full = np.concatenate([np.ones((L, 1)), s_np[b], np.zeros((L, 1))], 1)
        p_len = s_full[:, :gamma + 1] - s_full[:, 1:]           # [L, gamma+1]
        r_b = rho[b]
        r_b[pos[0]] = 1.0
        for r in range(pos[0], L):
            if r_b[r] == 0.0 or not mask_np[b, r]:
                continue
            for ell in range(gamma + 1):
                n = r + ell + 1
                if n < L:
                    r_b[n] += r_b[r] * p_len[r, ell]
    return torch.from_numpy(rho * mask_np).to(surv.device, torch.float32)


@torch.no_grad()
def _tree_window_starts(hit, mask, gamma):
    """Windows greedy tree decoding opens. hit[b, t, j]: the target mode at t
    is inside the draft top-k_j, i.e. depth j of a window covering t accepts.
    A window at s accepting n tokens is followed by one at s + n + 1."""
    hit_c = hit.cpu().numpy()
    mask_c = mask.cpu().numpy()
    L = mask_c.shape[1]
    start = torch.zeros(mask.shape, dtype=torch.bool)
    for b in range(mask_c.shape[0]):
        valid = mask_c[b].nonzero()[0]
        if valid.size == 0:
            continue
        s, end = int(valid[0]), int(valid[-1]) + 1
        while s < end:
            if mask_c[b, s]:
                start[b, s] = True
            n = 0
            while n < gamma and s + n < L and mask_c[b, s + n] and hit_c[b, s + n, n]:
                n += 1
            s += n + 1
    return start.to(mask.device)


@torch.no_grad()
def _top4path_accept(rank, lp, mask, top_k, path_topk):
    """[B, L] teacher-forced accept length when verification keeps only the
    path_topk paths with the highest product of draft probabilities. Paths
    are scored with the draft's sorted log-probs at the target-sequence
    contexts, i.e. an off-target branch is assumed to continue the way the
    target context does; on the validation set this lands within ~0.07 of
    the measured top4path accept length, where the plain top-k_j rule is
    ~0.45 too long."""
    B, L = rank.shape
    gamma = len(top_k)
    dev = rank.device
    ranks = torch.cartesian_prod(*[torch.arange(k, device=dev) for k in top_k])
    if ranks.dim() == 1:
        ranks = ranks.unsqueeze(-1)
    pos = (torch.arange(L, device=dev).view(-1, 1)
           + torch.arange(gamma, device=dev).view(1, -1))
    inside = pos < L
    pos = pos.clamp(max=L - 1)
    out = torch.zeros(B, L, device=dev)
    for b in range(B):
        lp_w = lp[b][pos]                                           # [L, gamma, K]
        score = lp_w.gather(
            -1, ranks.T.unsqueeze(0).expand(L, -1, -1)).sum(1)      # [L, P]
        keep = score.topk(min(path_topk, ranks.size(0)), dim=-1).indices
        valid = inside & mask[b][pos]
        match = (ranks.unsqueeze(0) == rank[b][pos].unsqueeze(1)) & valid.unsqueeze(1)
        depth = match.long().cumprod(-1).sum(-1)                    # [L, P]
        out[b] = depth.gather(1, keep).max(-1).values.float()
    return out


@torch.no_grad()
def _walk_windows(n, mask):
    """Window starts given the accept length n[b, s] of a window opened at s."""
    n_c = n.long().cpu().numpy()
    mask_c = mask.cpu().numpy()
    start = torch.zeros(mask.shape, dtype=torch.bool)
    for b in range(mask_c.shape[0]):
        valid = mask_c[b].nonzero()[0]
        if valid.size == 0:
            continue
        s, end = int(valid[0]), int(valid[-1]) + 1
        while s < end:
            if mask_c[b, s]:
                start[b, s] = True
            s += int(n_c[b, s]) + 1
    return start.to(mask.device)


@torch.no_grad()
def decode_window_starts(drf_hidden, tgt_hidden, drf_head, tgt_head, loss_mask,
                         top_k, chunk=256, path_topk=0):
    """[B, L-1] bool: hidden positions where greedy tree decoding with the
    current draft opens a window. With path_topk == 0, depth j accepts iff the
    target mode is in the draft top-k_j (the full tree); with path_topk > 0
    only the path_topk most probable paths are verified."""
    gamma, k_max = len(top_k), max(top_k)
    mask = loss_mask[:, 1:].bool()
    k_idx = torch.tensor([k - 1 for k in top_k], device=drf_hidden.device)
    hits, ranks, lps = [], [], []
    for s in range(0, drf_hidden.size(1) - 1, chunk):
        e = min(drf_hidden.size(1) - 1, s + chunk)
        u = tgt_head(tgt_hidden[:, s:e]).argmax(dim=-1)
        dl = drf_head(drf_hidden[:, s:e]).float()
        z_u = dl.gather(-1, u.unsqueeze(-1))
        if path_topk > 0:
            ranks.append((dl > z_u).sum(-1))
            lps.append(dl.log_softmax(-1).topk(k_max, dim=-1).values)
        else:
            kth = dl.scatter(-1, u.unsqueeze(-1), torch.finfo(dl.dtype).min).topk(
                k_max, dim=-1).values.index_select(-1, k_idx)
            hits.append(z_u > kth)
        del dl
    if path_topk > 0:
        n = _top4path_accept(torch.cat(ranks, 1), torch.cat(lps, 1), mask,
                             list(top_k), int(path_topk))
        return _walk_windows(n, mask)
    return _tree_window_starts(torch.cat(hits, 1), mask, gamma)


def dense_tree_loss(drf_hidden, tgt_hidden, drf_head, tgt_head, loss_mask,
                    top_k, tau, var_coef=0.0, start_only=False, chunk=256,
                    starts=None, path_n=None, temp=1.0, prob_starts=False,
                    path_margin=None, prob_region=None, eal_region=None):
    """Teacher-forced greedy tree loss over every supervised position.

    Greedy tree verification follows the target mode u, so depth j of a window
    at s accepts iff u_{s+j} is in the draft top-k_j, which the prefix logits
    decide exactly. Its surrogate is c_j = sigmoid((z(u) - z_(k_j)) / tau),
    z_(k) the k-th largest other logit, and E[L]_s = sum_d prod_{j<=d} c_j.
    Returns dense KL(p||q), the mean window E[L] (only decoding window starts
    when start_only), and the per-sequence variance of window E[L]s.
    With path_n ([B, L-1, k] tokens accepted along each of the k kept paths of
    the window at every position) and starts, E[L]_s is instead the mean over
    the kept paths of their chain EAL sum_d prod_{j<=d} q_j, q_j =
    softmax(z / temp)(u_{s+j}), as in top4_kl_eal. Terms past a path's first
    mismatch are dropped: their contexts leave the sequence and the path has
    stopped accepting. The variance is then across the kept paths.
    prob_starts replaces the hard starts with the probability that a window
    starts at each position (_renewal_start_prob, detached), with
    P(L_r >= d) = prod_{j<d} c_{r+j} * sigmoid(margin_d / tau): the gold
    prefix of length d enters the tree and a path extending it is among the
    kept ones. path_margin[b, r, d-1] is the draft log-score of the best path
    extending that prefix minus the path_topk-th best path not extending it,
    NaN when the prefix is not in the tree (the last defined margin is used).
    At tau -> 0 this is the exact accept length. With prob_region the
    recursion starts (rho = 1) at the region's first position and stays
    inside it; eal_region then restricts the weighted E[L] to a sub-range.
    Chunks are checkpointed so [B, L, V] logits never coexist with the graph.
    """
    from torch.utils.checkpoint import checkpoint
    gamma = len(top_k)
    k_max = max(top_k)
    mask = loss_mask[:, 1:].bool()
    dh, th = drf_hidden[:, :-1], tgt_hidden[:, :-1]
    L = dh.size(1)
    kls, zus, kths, lses = [], [], [], []
    for s in range(0, L, chunk):
        e = min(L, s + chunk)
        kl, z_u, kth, lse_t = checkpoint(
            _dense_tree_chunk, dh[:, s:e], th[:, s:e], drf_head, tgt_head, k_max,
            temp, use_reentrant=False)
        kls.append(kl)
        zus.append(z_u)
        kths.append(kth)
        lses.append(lse_t)
    kl = torch.cat(kls, 1)
    z_u = torch.cat(zus, 1)
    kth = torch.cat(kths, 1)
    lse_t = torch.cat(lses, 1)
    mf = mask.float()
    nv = mf.sum()
    kl_mean = (kl * mf).sum() / (nv + 1e-8)

    if path_n is not None:
        usable = L - gamma + 1
        if prob_starts:
            k_idx = torch.tensor([k - 1 for k in top_k], device=kth.device)
            c_d = torch.sigmoid((z_u.detach().unsqueeze(-1)
                                 - kth.detach().index_select(-1, k_idx)) / tau)
            c_d = F.pad(c_d * mf.unsqueeze(-1), (0, 0, 0, gamma))
            rank = torch.stack([c_d[:, j:j + L, j] for j in range(gamma)], -1)
            margin = path_margin.clone()
            margin[:, :, 0] = margin[:, :, 0].nan_to_num(nan=float("inf"))
            for d in range(1, gamma):
                margin[:, :, d] = torch.where(margin[:, :, d].isnan(),
                                              margin[:, :, d - 1], margin[:, :, d])
            keep = torch.sigmoid(margin / tau).cummin(-1).values
            region = mask if prob_region is None else prob_region
            starts = _renewal_start_prob(torch.cumprod(rank, -1) * keep, region)
            if eal_region is not None:
                starts = starts * eal_region.float()
        q = torch.exp(z_u / temp - lse_t) * mf
        q_win = torch.stack([q[:, j:j + usable] for j in range(gamma)], dim=-1)
        cp = torch.cumprod(q_win, dim=-1)                               # [B, U, gamma]
        m = path_n[:, :usable]                                          # [B, U, k]
        live = (torch.arange(gamma, device=m.device).view(1, 1, 1, -1)
                <= m.unsqueeze(-1)).float()                             # [B, U, k, gamma]
        chain = (cp.unsqueeze(2) * live).sum(-1)                        # [B, U, k]
        w = starts[:, :usable].float()
        ws = w.sum() + 1e-8
        dense_tree_loss.last_windows_per_seq = float(w.sum()) / w.size(0)
        eal_mean = (chain.mean(-1) * w).sum() / ws
        var_mean = (chain.var(dim=-1, unbiased=False) * w).sum() / ws
        hard_mean = (m.max(-1).values.float() * w).sum() / ws
        return kl_mean, eal_mean, var_mean, hard_mean

    k_idx = torch.tensor([k - 1 for k in top_k], device=kth.device)
    thr = kth.index_select(-1, k_idx)                       # [B, L, gamma]
    c = torch.sigmoid((z_u.unsqueeze(-1) - thr) / tau) * mf.unsqueeze(-1)
    usable = L - gamma + 1
    if usable <= 0:
        z = kl_mean * 0.0
        return kl_mean, z, z, z
    c_win = torch.stack([c[:, j:j + usable, j] for j in range(gamma)], dim=-1)
    eal_win = torch.cumprod(c_win, dim=-1).sum(-1)          # [B, U]
    if starts is not None:
        w = starts[:, :usable].float()
    elif start_only:
        hit = z_u.detach().unsqueeze(-1) > thr.detach()
        w = _tree_window_starts(hit, mask, gamma)[:, :usable].float()
    else:
        w = mf[:, :usable]
    nw = w.sum(-1)
    eal_mean = (eal_win * w).sum() / (w.sum() + 1e-8)
    with torch.no_grad():
        hit_h = (z_u.unsqueeze(-1) > thr) & mask.unsqueeze(-1)
        h_win = torch.stack([hit_h[:, j:j + usable, j] for j in range(gamma)], dim=-1)
        hard = torch.cumprod(h_win.long(), dim=-1).sum(-1).float()
        hard_mean = (hard * w).sum() / (w.sum() + 1e-8)
    if var_coef:
        mu = (eal_win * w).sum(-1) / nw.clamp(min=1.0)
        var = (((eal_win - mu.unsqueeze(-1)) ** 2) * w).sum(-1) / nw.clamp(min=1.0)
        ok = (nw >= 2).float()
        var_mean = (var * ok).sum() / ok.sum().clamp(min=1.0)
    else:
        var_mean = eal_mean.detach() * 0.0
    return kl_mean, eal_mean, var_mean, hard_mean


def chain_style_tree_eal(draft_logits, target_logits, tokens, parents, valid,
                         temperature):
    """loss = -E[L] along the single target-token chain.

    Forward α = q(target argmax). Backward differentiates log q, so a
    confidently wrong draft still gets a gradient. The next context is only
    the child whose drafted token is that argmax. Sibling branches are not
    continuations of this acceptance. If the target token was not drafted,
    the chain stops after the current α.
    """
    if (draft_logits.shape[:2] != parents.shape or valid.shape != parents.shape
            or tokens.shape != parents.shape):
        raise ValueError(
            "draft logits, tokens, parents, and valid must share [B, M], "
            f"got {tuple(draft_logits.shape)} {tuple(tokens.shape)} "
            f"{tuple(parents.shape)} {tuple(valid.shape)}")
    B, M, _vocab = draft_logits.shape
    target_token = target_logits.detach().argmax(dim=-1)
    alpha_row = log_q_surrogate(draft_logits, target_token, temperature)

    eals = []
    for b in range(B):
        children = {}
        for i in range(M):
            if not bool(valid[b, i]):
                continue
            parent_id = int(parents[b, i].item())
            children.setdefault(parent_id, []).append(i)
        alphas = []
        context = -1
        for _ in range(M):
            kids = children.get(context, [])
            if not kids:
                break
            alphas.append(alpha_row[b, kids[0]])
            wanted = int(target_token[b, kids[0]].item())
            match = next(
                (k for k in kids if int(tokens[b, k].item()) == wanted), None)
            if match is None:
                break
            context = match
        if not alphas:
            eals.append(draft_logits.new_zeros(()))
            continue
        alpha = torch.stack(alphas).view(1, -1)
        parent_idx = [-1] + list(range(len(alphas) - 1))
        parent_row = torch.tensor(
            parent_idx, device=draft_logits.device, dtype=torch.long).view(1, -1)
        eals.append(tree_expected_accepted_length(alpha, parent_row).squeeze(0))
    eal = torch.stack(eals).mean()
    return -eal, eal


def soft_topk_tree_eal(alpha, parents, paths, path_scores, path_topk, tau=1.0,
                       alpha_soft=None):
    """Soft Top-K surrogate of E[L](TopK(T), q).

    pi = softmax(s / tau)
    mu_p = 1 - (1 - pi_p)^K
    beta_v = 1 - prod_{p containing v} (1 - mu_p)
    E_soft = TreeEAL(beta * alpha) on the full topology.
    Shared prefixes are one node. Parallel branches use the survival recursion.
    Returns (loss, eal_soft_mean, eal_hard_mean). loss = -E_soft.
    eal_hard is the exact discrete Top-K union and is detached.

    The forward value of alpha stays the hard target probability of the drafted
    token. Path scores alone only rescale those already-chosen tokens, so the
    greedy tree and its accept length do not move. alpha_soft is the expected
    acceptance q·p at the same nodes; it is added with a straight-through
    estimator so the draft distribution is trained toward tokens the target
    would accept.
    """
    if path_scores.dim() != 2:
        raise ValueError(f"path_scores must be [B, P], got {tuple(path_scores.shape)}")
    B, n_paths = path_scores.shape
    k = min(int(path_topk), int(n_paths))
    if k < 1:
        raise ValueError(f"path_topk must be >= 1, got {path_topk}")
    device = path_scores.device
    alpha_hard = alpha.detach()
    if alpha_soft is None:
        alpha = alpha_hard
    else:
        # Forward value is the hard acceptance. Gradient is d(q·p).
        alpha = alpha_hard + (alpha_soft - alpha_soft.detach())
    if parents.dim() == 1:
        parent_rows = parents.to(device=device, dtype=torch.long).view(1, -1).expand(B, -1)
        parent_list = parents.tolist()
    else:
        parent_rows = parents.to(device=device, dtype=torch.long)
        parent_list = parents[0].tolist()
    paths = paths.to(device=device, dtype=torch.long)
    tau = max(float(tau), 1e-6)
    pi = torch.softmax(path_scores / tau, dim=-1).clamp(max=1.0 - 1e-6)
    log_one_minus_mu = k * torch.log1p(-pi)
    member = path_scores.new_zeros(n_paths, alpha.size(1))
    member[torch.arange(n_paths, device=device).unsqueeze(1), paths] = 1.0
    beta = 1.0 - torch.exp(log_one_minus_mu @ member)
    eal_soft = tree_expected_accepted_length(beta * alpha, parent_rows)

    union_parents, gather_index, _sel = induce_path_topk_union(
        parent_list, paths, path_scores.detach(), k)
    valid = gather_index >= 0
    picked = alpha_hard.gather(1, gather_index.clamp(min=0))
    alpha_u = torch.where(valid, picked, torch.zeros_like(picked))
    eal_hard = tree_expected_accepted_length(alpha_u, union_parents)
    loss = -eal_soft.mean()
    return loss, eal_soft.mean(), eal_hard.mean()


def node_marginal_eal_loss(alpha, parents, logp_draft, valid):
    """L_EAL = -sum_v sg(dE[L]/d alpha_v) log p_draft(v | parent).

    alpha:      [B, M] target acceptance on T_K, pads included
    parents:    [B, M] long
    logp_draft: [B, M] draft log-prob of the drafted token, carries grad
    valid:      [B, M] bool, False on padding columns
    Returns (loss, analytic E[L] averaged over the batch).
    """
    with torch.enable_grad():
        alpha_live = alpha.detach().requires_grad_(True)
        eal_vec = tree_expected_accepted_length(alpha_live, parents)
        weights = torch.autograd.grad(eal_vec.sum(), alpha_live)[0]
    weights = torch.where(valid, weights, torch.zeros_like(weights)).detach()
    logp = torch.where(valid, logp_draft, torch.zeros_like(logp_draft))
    loss = -(weights * logp).sum() / alpha.shape[0]
    return loss, eal_vec.detach().mean()


class SmallLMTreeV2Model(nn.Module):

    def __init__(self, target_path, draft_path, top_k_per_depth,
                 dtype=torch.float16, tree_budget=None):
        super().__init__()
        self.top_k_per_depth = tuple(int(x) for x in top_k_per_depth)
        self.gamma = len(self.top_k_per_depth)
        assert self.gamma >= 1
        parents, depths, paths, depth_counts, depth_starts, N = \
            _build_tree_topology(self.top_k_per_depth)
        self.tree_parents = parents
        self.tree_depths = depths
        self.tree_paths = paths
        self.tree_depth_counts = depth_counts
        self.tree_depth_starts = depth_starts
        self.tree_n_nodes = N
        self.tree_budget = int(N if tree_budget is None else tree_budget)
        if self.tree_budget < 1:
            raise ValueError(f"tree_budget must be >= 1, got {self.tree_budget}")

        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_path, torch_dtype=dtype, attn_implementation="sdpa")
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_path, torch_dtype=dtype, attn_implementation="sdpa")
        for p in self.target_model.parameters():
            p.requires_grad = False
        self.target_model.eval()

        self.register_buffer(
            "paths", torch.tensor(paths, dtype=torch.long), persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.target_model.eval()
        self.draft_model.train(mode)
        return self

    def _target_tree_mask(self, prefix_len, device, dtype):
        N = self.tree_n_nodes
        parents = self.tree_parents
        min_val = torch.finfo(dtype).min
        mask = torch.full((1, 1, N, prefix_len + N), min_val,
                          device=device, dtype=dtype)
        mask[0, 0, :, :prefix_len] = 0.0
        for i in range(N):
            j = i
            while j >= 0:
                mask[0, 0, i, prefix_len + j] = 0.0
                j = parents[j]
        return mask

    def _draft_step_mask(self, d_step, prefix_len, device, dtype):
        """4D mask for draft step d_step (d_step ≥ 1).
        Input: depth_counts[d_step-1] tokens at depth d_step-1.
        Cache before: prefix + sum(depth_counts[0:d_step-1]) nodes.
        Each new token sees prefix + ancestor chain + self.
        """
        min_val = torch.finfo(dtype).min
        depth_idx = d_step - 1
        depth_counts = self.tree_depth_counts
        depth_starts = self.tree_depth_starts
        top_k = self.top_k_per_depth
        q_len = depth_counts[depth_idx]
        cache_before = depth_starts[depth_idx]   # sum(depth_counts[0:d_step-1])
        kv_len = prefix_len + cache_before + q_len
        mask = torch.full((1, 1, q_len, kv_len), min_val,
                          device=device, dtype=dtype)
        mask[0, 0, :, :prefix_len] = 0.0
        for q in range(q_len):
            # Ancestor at depth k (0 ≤ k ≤ depth_idx - 1):
            # offset = q // prod(top_k[k+1 : depth_idx+1])
            cur = q
            for k_back in range(depth_idx - 1, -1, -1):
                cur = cur // top_k[k_back + 1]
                slot = prefix_len + depth_starts[k_back] + cur
                mask[0, 0, q, slot] = 0.0
            # Self in new tokens region
            mask[0, 0, q, prefix_len + cache_before + q] = 0.0
        return mask

    def _draft_step_tree_allowed(self, d_step, device):
        """[q_len, cache_before + q_len] bool: ancestors and self of each new node."""
        cache = self.__dict__.setdefault("_draft_tree_allowed", {})
        key = (d_step, str(device))
        if key not in cache:
            depth_idx = d_step - 1
            depth_starts = self.tree_depth_starts
            top_k = self.top_k_per_depth
            q_len = self.tree_depth_counts[depth_idx]
            cache_before = depth_starts[depth_idx]
            allowed = torch.zeros(q_len, cache_before + q_len, dtype=torch.bool)
            for q in range(q_len):
                cur = q
                for k_back in range(depth_idx - 1, -1, -1):
                    cur = cur // top_k[k_back + 1]
                    allowed[q, depth_starts[k_back] + cur] = True
                allowed[q, cache_before + q] = True
            cache[key] = allowed.to(device)
        return cache[key]

    def _draft_step_mask_batch(self, d_step, kv_lens, max_kv, device, dtype):
        """Batched 4D mask. Tree nodes are appended after the right-padded prefix."""
        tree = self._draft_step_tree_allowed(d_step, device)
        B, q_len = len(kv_lens), tree.size(0)
        prefix = _prefix_allowed(kv_lens, max_kv, device)
        allowed = torch.cat([prefix.unsqueeze(1).expand(B, q_len, max_kv),
                             tree.unsqueeze(0).expand(B, -1, -1)], dim=-1)
        mask = torch.zeros(allowed.shape, device=device, dtype=dtype)
        return mask.masked_fill_(~allowed, torch.finfo(dtype).min).unsqueeze(1)

    def _target_mask_batch(self, kv_lens, max_kv, parents, device, dtype):
        """Per-sample ancestor mask for a budget-pruned tree. parents: [B, N]."""
        B, N = parents.shape
        anc = torch.zeros(B, N, N, dtype=torch.bool, device=device)
        cur = torch.arange(N, device=device).view(1, N).expand(B, N)
        for _ in range(self.gamma):
            hit = F.one_hot(cur.clamp(min=0), N).bool() & (cur >= 0).unsqueeze(-1)
            anc |= hit
            cur = torch.where(cur >= 0, parents.gather(1, cur.clamp(min=0)), cur)
        prefix = _prefix_allowed(kv_lens, max_kv, device)
        allowed = torch.cat([prefix.unsqueeze(1).expand(B, N, max_kv), anc], dim=-1)
        mask = torch.zeros(allowed.shape, device=device, dtype=dtype)
        return mask.masked_fill_(~allowed, torch.finfo(dtype).min).unsqueeze(1)

    def _rollout_batch(self, batch_idx, anchor_pos,
                       drf_anchor_logits, tgt_anchor_logits,
                       drf_keys, drf_values, tgt_keys, tgt_values,
                       temperature, eal_mode="budget", path_topk=16,
                       eal_score_coef=0.1, anchor_group=None,
                       anchor_kl_coef=0.0, path_eal_weights=None,
                       path_eal_mean=False, path_eal_var_coef=0.0,
                       tree_kl_w=None, tree_eal_w=None):
        """Greedy top-k tree, global budget prune, batched across the microbatch.

        Returns mean KL over kept nodes and mean Tree-EAL. KL carries the draft
        gradient. Tree-EAL uses target acceptance probabilities.
        """
        device = drf_anchor_logits.device
        B = len(batch_idx)
        gamma = self.gamma
        top_k = self.top_k_per_depth
        depth_counts = self.tree_depth_counts
        depth_starts = self.tree_depth_starts
        kv_lens = [ap + 1 for ap in anchor_pos]
        T = max(float(temperature), 1e-6)

        a_t = torch.tensor(anchor_pos, device=device, dtype=torch.long)

        drf_cache, max_kv = _pad_cache_batch(drf_keys, drf_values, batch_idx, kv_lens)

        # Already the anchor rows, shape [B, V]. Casting the full prefix
        # logits to float32 does not fit on an 80GB H100.
        drf_root = drf_anchor_logits.float()
        tgt_root = tgt_anchor_logits.float()
        root_lp = F.log_softmax(drf_root, dim=-1)
        # sample_minpq: the draft SAMPLES k distinct children per node at T,
        # matching eval --draft-mode sample. Otherwise greedy top-k.
        sample_tree = eal_mode in ("sample_minpq", "sample_anyk", "sample_cover")
        if sample_tree:
            with torch.no_grad():
                root_tok = torch.multinomial(
                    torch.softmax(drf_root / T, dim=-1), top_k[0], replacement=False)
            root_scores = root_lp.gather(-1, root_tok)
        else:
            root_scores, root_tok = root_lp.topk(top_k[0], dim=-1)

        tokens_per_depth = [root_tok]
        scores_per_depth = [root_scores]
        pred_per_depth = []
        # soft_topk keeps the draft distribution so the accept length can train it.
        skip_full_logits = eal_mode == "node_marginal"
        if not skip_full_logits:
            pred_per_depth.append(
                drf_root.unsqueeze(1).expand(B, top_k[0], -1).contiguous())
        last_sel = None
        full_leaf_tok = root_tok if gamma == 1 else None
        full_leaf_sc = root_scores if gamma == 1 else None
        full_leaf_logp = root_scores if gamma == 1 else None
        full_leaf_pred = pred_per_depth[0] if gamma == 1 and pred_per_depth else None
        prefix_logp = [root_scores]

        for d_step in range(1, gamma):
            n_in = tokens_per_depth[-1].size(1)
            if n_in != depth_counts[d_step - 1]:
                raise RuntimeError(
                    "budget prune before the last depth is not supported "
                    f"(depth {d_step - 1}: {n_in} != {depth_counts[d_step - 1]})")
            mask = self._draft_step_mask_batch(
                d_step, kv_lens, max_kv, device, self.draft_model.dtype)
            pos = (a_t + d_step).view(B, 1).expand(B, n_in)
            step_out = self.draft_model(
                input_ids=tokens_per_depth[-1],
                attention_mask=mask,
                position_ids=pos,
                past_key_values=drf_cache,
                use_cache=(d_step < gamma - 1),
                return_dict=True,
            )
            depth_pred = step_out.logits.float()
            child_lp = F.log_softmax(depth_pred, dim=-1)
            k = top_k[d_step]
            if sample_tree:
                with torch.no_grad():
                    flat_p = torch.softmax(depth_pred / T, dim=-1).reshape(B * n_in, -1)
                    child_tok = torch.multinomial(
                        flat_p, k, replacement=False).view(B, n_in, k)
                child_sc = child_lp.gather(-1, child_tok)
            else:
                child_sc, child_tok = child_lp.topk(k, dim=-1)
            cum = scores_per_depth[-1].unsqueeze(-1) + child_sc
            flat_tok = child_tok.reshape(B, -1)
            flat_sc = cum.reshape(B, -1)
            flat_logp = child_sc.reshape(B, -1)
            flat_pred = (None if skip_full_logits
                         else depth_pred.repeat_interleave(k, dim=1))
            if d_step == gamma - 1:
                # Snapshot every complete path before the node-budget prune.
                full_leaf_tok = flat_tok
                full_leaf_sc = flat_sc
                full_leaf_logp = flat_logp
                full_leaf_pred = flat_pred
            else:
                prefix_logp.append(flat_logp)
            n_cand = flat_tok.size(1)
            placed = sum(t.size(1) for t in tokens_per_depth)
            keep = min(self.tree_budget - placed, n_cand)
            if keep <= 0:
                raise RuntimeError(
                    f"tree_budget {self.tree_budget} exhausted before depth {d_step}")
            if keep < n_cand:
                if d_step < gamma - 1:
                    raise RuntimeError(
                        "budget prune before the last depth is not supported "
                        f"(depth {d_step}: keep {keep} of {n_cand})")
                _, sel = flat_sc.topk(keep, dim=-1)
                flat_tok = flat_tok.gather(1, sel)
                flat_sc = flat_sc.gather(1, sel)
                last_sel = sel
            tokens_per_depth.append(flat_tok)
            scores_per_depth.append(flat_sc)
            if not skip_full_logits:
                if keep < n_cand:
                    flat_pred = flat_pred.gather(
                        1, sel.unsqueeze(-1).expand(B, keep, flat_pred.size(-1)))
                pred_per_depth.append(flat_pred)
            if d_step < gamma - 1:
                drf_cache = step_out.past_key_values

        tokens = torch.cat(tokens_per_depth, dim=1)
        drf_node = None if skip_full_logits else torch.cat(pred_per_depth, dim=1)
        N = tokens.size(1)

        if last_sel is None:
            parents = torch.tensor(
                self.tree_parents, device=device, dtype=torch.long)
            parents = parents.view(1, -1).expand(B, -1).contiguous()
            depths = torch.tensor(self.tree_depths, device=device, dtype=torch.long)
        else:
            n_prefix = depth_starts[-1]
            base = torch.tensor(
                self.tree_parents[:n_prefix], device=device, dtype=torch.long)
            base = base.view(1, -1).expand(B, -1)
            full_last_parents = torch.tensor(
                [self.tree_parents[n_prefix + j] for j in range(depth_counts[-1])],
                device=device, dtype=torch.long)
            leaf_par = full_last_parents.view(1, -1).expand(B, -1).gather(1, last_sel)
            parents = torch.cat([base, leaf_par], dim=1)
            depths = torch.tensor(
                list(self.tree_depths[:n_prefix]) + [gamma - 1] * last_sel.size(1),
                device=device, dtype=torch.long)

        tgt_cache, max_kv_t = _pad_cache_batch(tgt_keys, tgt_values, batch_idx, kv_lens)
        assert max_kv_t == max_kv
        pos = a_t.view(B, 1) + 1 + depths.view(1, -1)
        tree_mask = self._target_mask_batch(
            kv_lens, max_kv, parents, device, self.target_model.dtype)
        with torch.no_grad():
            tgt_out = self.target_model(
                input_ids=tokens,
                attention_mask=tree_mask,
                position_ids=pos,
                past_key_values=tgt_cache,
                use_cache=False,
                return_dict=True,
            )
        tgt_tree = tgt_out.logits.float()
        del tgt_out
        has_parent = parents >= 0
        gathered = tgt_tree.gather(
            1, parents.clamp(min=0).unsqueeze(-1).expand(B, N, tgt_tree.size(-1)))
        tgt_node = torch.where(
            has_parent.unsqueeze(-1),
            gathered,
            tgt_root.unsqueeze(1).expand(B, N, -1),
        )

        if skip_full_logits or eal_mode in (
                "soft_topk", "sample_minpq", "sample_anyk", "sample_cover",
                "wsoft", "greedy_topk", "dense_tree", "top4_kl_eal"):
            kl = drf_root.new_zeros(())
        else:
            tgt_logp = F.log_softmax(tgt_node, dim=-1)
            tgt_p = tgt_logp.exp()
            drf_logp = F.log_softmax(drf_node, dim=-1)
            kl = (tgt_p * (tgt_logp - drf_logp)).sum(-1).clamp(min=0.0).mean()
        if eal_mode == "dense_tree":
            # KL over the nodes of the path_topk paths eval keeps (ranked by
            # the product of draft node probabilities); E[L] is trained densely
            # in forward. Logs the greedy accepted length on those paths.
            if last_sel is not None or full_leaf_sc is None:
                raise RuntimeError(
                    "dense_tree ranks every complete path, so tree_budget "
                    "must cover the full tree")
            k_paths = min(int(path_topk), int(full_leaf_sc.size(1)))
            chosen = self.paths[full_leaf_sc.detach().topk(k_paths, dim=-1).indices]
            on_path = torch.zeros(B, N, device=device, dtype=torch.bool)
            on_path.scatter_(1, chosen.reshape(B, -1), True)
            tgt_logp = F.log_softmax(tgt_node, dim=-1)
            kl_node = (tgt_logp.exp() * (tgt_logp - F.log_softmax(drf_node, dim=-1))
                       ).sum(-1).clamp(min=0.0)
            del tgt_logp
            kl = (kl_node * on_path).sum() / on_path.sum().clamp(min=1)
            with torch.no_grad():
                accept = tgt_node.argmax(dim=-1).eq(tokens) & on_path
                eal = longest_path_by_depth(accept, parents, depth_starts, depth_counts)
            eal_loss = None
            n_report = float(on_path.sum(dim=-1).float().mean().item())
        elif eal_mode == "top4_kl_eal":
            # Only the path_topk paths eval keeps (ranked by the product of
            # draft node probabilities). KL is averaged over the draft
            # distributions that produced those nodes, one per parent (siblings
            # share one). E[L] is the mean of the paths' chain EALs,
            # P_j = softmax(z_j / T)[argmax p_j]; eal_loss = -(mean - var_coef
            # * variance across the paths).
            if last_sel is not None or full_leaf_sc is None:
                raise RuntimeError(
                    "top4_kl_eal ranks every complete path, so tree_budget "
                    "must cover the full tree")
            k_paths = min(int(path_topk), int(full_leaf_sc.size(1)))
            chosen = self.paths[full_leaf_sc.detach().topk(k_paths, dim=-1).indices]
            on_path = torch.zeros(B, N, device=device, dtype=torch.bool)
            on_path.scatter_(1, chosen.reshape(B, -1), True)
            slot = parents + 1
            n_sib = torch.zeros(B, N + 1, device=device).scatter_add_(
                1, slot, on_path.float())
            w_kl = on_path.float() / n_sib.gather(1, slot).clamp(min=1.0)
            if tree_kl_w is not None:
                w_kl = w_kl * tree_kl_w.unsqueeze(-1)
            tgt_logp = F.log_softmax(tgt_node, dim=-1)
            kl_node = (tgt_logp.exp() * (tgt_logp - F.log_softmax(drf_node, dim=-1))
                       ).sum(-1).clamp(min=0.0)
            del tgt_logp
            kl = (kl_node * w_kl).sum() / w_kl.sum().clamp(min=1.0)
            q_u = torch.softmax(drf_node / T, dim=-1).gather(
                -1, tgt_node.argmax(dim=-1, keepdim=True)).squeeze(-1)
            factors = q_u.gather(1, chosen.reshape(B, -1)).view(B, k_paths, gamma)
            chain_eal = torch.cumprod(factors, dim=-1).sum(dim=-1)
            path_var = chain_eal.var(dim=-1, unbiased=False)
            tree_score = chain_eal.mean(dim=-1) - float(path_eal_var_coef) * path_var
            if tree_eal_w is None:
                eal_loss = -tree_score.mean()
            else:
                eal_loss = -(tree_score * tree_eal_w).sum() / tree_eal_w.sum().clamp(min=1.0)
            path_std = path_var.detach().sqrt().mean()
            with torch.no_grad():
                accept = tgt_node.argmax(dim=-1).eq(tokens) & on_path
                eal = longest_path_by_depth(accept, parents, depth_starts, depth_counts)
                hard_al = accept.gather(1, chosen.reshape(B, -1)).view(
                    B, k_paths, gamma).float().cumprod(dim=-1).sum(dim=-1)
                path_stats = torch.cat([
                    chain_eal.detach().mean(dim=0), hard_al.mean(dim=0),
                    torch.stack([path_var.detach().mean(),
                                 hard_al.var(dim=-1, unbiased=False).mean()])])
            n_report = float(on_path.sum(dim=-1).float().mean().item())
        elif eal_mode == "budget":
            alpha = torch.softmax(tgt_node / T, dim=-1).gather(
                -1, tokens.long().unsqueeze(-1)).squeeze(-1)
            eal = tree_expected_accepted_length(alpha, parents).mean()
            n_report = int(N)
            eal_loss = None
        elif eal_mode == "sample_anyk":
            # Sampled tree scored with the rule eval actually runs: every node
            # is accepted independently with min(1, p/q) and the longest fully
            # accepted prefix is kept, so a slot advances when ANY sibling is
            # accepted. tree_expected_accepted_length is that survival
            # recursion, so E[L | tree] is exact. A chain has one sibling per
            # slot and cannot express this term.
            tok_idx = tokens.long().unsqueeze(-1)
            p_node = torch.softmax(tgt_node / T, dim=-1).gather(-1, tok_idx).squeeze(-1)
            log_q = torch.log_softmax(drf_node / T, dim=-1).gather(-1, tok_idx).squeeze(-1)
            alpha = (p_node.detach() / log_q.exp().clamp(min=1e-9)).clamp(max=1.0)
            eal_cond = tree_expected_accepted_length(alpha, parents)
            # alpha falls as q rises, so the value term alone would shrink q to
            # inflate p/q. Which tokens enter the tree also depends on q, and
            # that path is the score-function term below; together they are the
            # gradient of E_{tree~q}[L]. log q of the sampled set is averaged
            # per node, because the raw sum over 128 nodes reaches -1500 on tail
            # tokens and under grad clipping leaves nothing of the value term.
            # The weight is E[L | tree] (same expectation as one draw, less
            # noise) centred over the draws at the same anchor.
            score = log_q.mean(dim=1)
            adv = _center_by_group(eal_cond.detach(), anchor_group)
            eal_loss = -(eal_cond.mean()
                         + eal_score_coef * (adv * score).mean())
            with torch.no_grad():
                a = alpha.detach()
                draws = [longest_path_from_accept(torch.rand_like(a) <= a, parents)
                         for _ in range(16)]
                eal = torch.stack(draws).mean()
            n_report = int(N)
        elif eal_mode == "sample_minpq":
            # Sampled tree. alpha_v = min(p_v, q_v): the chance the draft
            # samples v and the ratio test accepts it. Siblings are mutually
            # exclusive, so E[L] = sum over nodes of prod alpha along the path.
            # q carries the draft gradient wherever q < p.
            tok_idx = tokens.long().unsqueeze(-1)
            p_node = torch.softmax(tgt_node / T, dim=-1).gather(-1, tok_idx).squeeze(-1)
            q_node = torch.softmax(drf_node / T, dim=-1).gather(-1, tok_idx).squeeze(-1)
            alpha = torch.minimum(p_node.detach(), q_node)
            eal_loss = -tree_path_sum_eal(alpha, parents).mean()
            with torch.no_grad():
                # One realized accept length under the eval rule:
                # independent r <= min(1, p/q), longest fully accepted path.
                ratio = (p_node / q_node.clamp(min=1e-20)).clamp(max=1.0)
                accept = torch.rand_like(ratio) <= ratio
                eal = longest_path_from_accept(accept, parents)
            n_report = int(N)
        elif eal_mode == "sample_cover":
            # One target draw per slot, siblings mutually exclusive. The cover
            # of a slot is c = 1 - sum_v p(v) (1-q(v))^k with k equal to that
            # slot's child count only. A root (k=4) is trained to spread over
            # p's head and a deep node (k=1) is trained to spike, and those
            # gradients land on different prefixes. c is the factor for
            # entering each node; siblings share their parent's c. E[L] is the
            # mean over leaves of the tail sum along the path.
            c = drf_node.new_empty(B, N)
            accept = torch.zeros(B, N, dtype=torch.bool, device=device)
            for d, cnt in enumerate(depth_counts):
                start = depth_starts[d]
                end = min(start + cnt, N)
                if start >= end:
                    break
                k = max(int(top_k[d]), 1)
                p_sl = torch.softmax(tgt_node[:, start:end] / T, dim=-1)
                q_sl = torch.softmax(drf_node[:, start:end] / T, dim=-1)
                miss = (1.0 - q_sl).clamp(min=0.0)
                if k != 1:
                    miss = miss.pow(k)
                c[:, start:end] = (1.0 - (p_sl * miss).sum(dim=-1)).clamp(0.0, 1.0)
                n_here = end - start
                tok = tokens[:, start:end]
                with torch.no_grad():
                    p_draw = p_sl.detach()
                    if k == 1 or n_here % k != 0:
                        draw = torch.multinomial(
                            p_draw.reshape(B * n_here, -1), 1).view(B, n_here)
                    else:
                        n_groups = n_here // k
                        draw = torch.multinomial(
                            p_draw[:, ::k].reshape(B * n_groups, -1), 1).view(B, n_groups)
                        draw = draw.repeat_interleave(k, dim=1)
                    accept[:, start:end] = tok.long() == draw
                del p_sl, q_sl, miss
            eal_loss = -tree_path_mean_eal(c, parents).mean()
            with torch.no_grad():
                eal = longest_path_from_accept(accept, parents)
            n_report = int(N)
        elif eal_mode == "wsoft":
            # Weighted soft top-k Tree EAL for verify-mode simple: several
            # children of one slot can pass, so
            #   S_h(d) = 1 - prod_v (1 - p(v) s_v S_v(d-1))
            # with s_v = sigmoid((z_q(v) - z_(k)^{-v}) / tau). tree_expected_
            # accepted_length is that recursion. Candidates are the drafted
            # top-k, which have a real subtree, plus the target's top-m tokens
            # that the draft left out. Those extras are leaves: they receive a
            # gradient for entering the top-k, and their continuation is the
            # depth-1 approximation. Loss = -E[L] + kl_coef * KL(p || q).
            tau = float(getattr(self, "greedy_tau", 1.0))
            mix_m = max(int(getattr(self, "soft_target_m", 4)), 1)
            kl_coef = float(anchor_kl_coef)
            alpha = drf_node.new_zeros(B, N)
            p_draft = drf_node.new_zeros(B, N)
            extra_alpha = []
            extra_parents = []
            kl_sum = drf_node.new_zeros(())
            kl_count = 0
            for d, cnt in enumerate(depth_counts):
                start = depth_starts[d]
                end = min(start + cnt, N)
                if start >= end:
                    break
                k = max(int(top_k[d]), 1)
                n_here = end - start
                if n_here % k != 0:
                    raise RuntimeError(
                        f"wsoft depth {d} has {n_here} nodes, not a multiple of k={k}")
                groups = n_here // k
                drf_g = drf_node[:, start:end].contiguous().view(
                    B, groups, k, -1)[:, :, 0, :]
                tgt_g = tgt_node[:, start:end].contiguous().view(
                    B, groups, k, -1)[:, :, 0, :]
                tok = tokens[:, start:end].contiguous().view(B, groups, k).long()
                log_q = F.log_softmax(drf_g, dim=-1)
                log_p = F.log_softmax(tgt_g, dim=-1)
                kl_g = (log_p.exp() * (log_p - log_q)).sum(-1).clamp(min=0.0)
                kl_sum = kl_sum + kl_g.sum()
                kl_count += int(kl_g.numel())
                s_tok = soft_topk_score(drf_g, tok, k, tau)
                p_tok = log_p.gather(-1, tok).exp()
                alpha[:, start:end] = (p_tok.detach() * s_tok).reshape(B, n_here)
                p_draft[:, start:end] = p_tok.detach().reshape(B, n_here)
                tgt_idx = tgt_g.detach().topk(mix_m, dim=-1).indices
                in_draft = (tgt_idx.unsqueeze(-1) == tok.unsqueeze(-2)).any(-1)
                s_extra = soft_topk_score(drf_g, tgt_idx, k, tau)
                p_extra = log_p.gather(-1, tgt_idx).exp()
                a_extra = torch.where(
                    in_draft, torch.zeros_like(p_extra), p_extra.detach() * s_extra)
                extra_alpha.append(a_extra.reshape(B, groups * mix_m))
                if d == 0:
                    par = torch.full(
                        (B, groups * mix_m), -1, device=device, dtype=torch.long)
                else:
                    par = parents[:, start:end:k].repeat_interleave(mix_m, dim=1)
                extra_parents.append(par)
                del log_q, log_p, drf_g, tgt_g
            kl = kl_sum / max(kl_count, 1)
            alpha_ext = torch.cat([alpha] + extra_alpha, dim=1)
            parents_ext = torch.cat([parents] + extra_parents, dim=1)
            eal_soft = tree_expected_accepted_length(alpha_ext, parents_ext)
            eal_loss = -eal_soft.mean() + kl_coef * kl
            with torch.no_grad():
                accept = torch.rand_like(p_draft) <= p_draft
                eal = longest_path_from_accept(accept, parents)
            n_report = int(alpha_ext.size(1))
        elif eal_mode == "greedy_topk":
            # Greedy top-k tree. Rank complete paths by the product of the
            # draft probabilities of their nodes (full_leaf_sc is the sum of
            # those log-probs) and keep path_topk of them.
            # On each kept path the chain factor is the draft probability of
            # the target argmax at that node. Path ranking above stays at
            # temperature 1. Only this lookup is sharpened:
            #   P_j = softmax(z_j / T)[argmax p_j]
            #   chain EAL = Σ_d Π_{j≤d} P_j
            # T < 1 pulls P toward the greedy 0/1 decision. The trained tree
            # EAL is a weighted sum of those chain EALs. Empty weights mean
            # every kept path has weight 1. path_eal_weights follows topk
            # order: index 0 is the highest draft-probability path.
            # The logged accept length is the max, unweighted.
            if last_sel is not None or full_leaf_sc is None:
                raise RuntimeError(
                    "greedy_topk scores every complete path, so tree_budget "
                    "must cover the full tree")
            if full_leaf_sc.size(1) != self.paths.size(0):
                raise RuntimeError(
                    f"path scores {full_leaf_sc.size(1)} != paths {self.paths.size(0)}")
            q_node = torch.softmax(drf_node / T, dim=-1)
            target_mode = tgt_node.argmax(dim=-1)
            factor = q_node.gather(-1, target_mode.unsqueeze(-1)).squeeze(-1)
            k_paths = min(int(path_topk), int(full_leaf_sc.size(1)))
            chosen = self.paths[full_leaf_sc.detach().topk(k_paths, dim=-1).indices]
            factors = factor.gather(1, chosen.reshape(B, -1)).view(B, k_paths, gamma)
            chain_eal = torch.cumprod(factors, dim=-1).sum(dim=-1)
            if path_eal_weights:
                if len(path_eal_weights) != k_paths:
                    raise RuntimeError(
                        f"path_eal_weights has {len(path_eal_weights)} values, "
                        f"but {k_paths} paths were kept")
                weights = chain_eal.new_tensor(list(path_eal_weights))
                trained = (chain_eal * weights.view(1, -1)).sum(dim=-1)
            elif path_eal_mean:
                trained = chain_eal.mean(dim=-1)
            else:
                trained = chain_eal.sum(dim=-1)
            eal_loss = -trained.mean()
            # Spread of the kept paths' EALs (population variance over the
            # path_topk paths of one tree); penalising it trains all of them
            # together instead of letting the best path carry the tree.
            path_var = chain_eal.var(dim=-1, unbiased=False)
            if path_eal_var_coef:
                eal_loss = eal_loss + float(path_eal_var_coef) * path_var.mean()
            path_std = path_var.detach().sqrt().mean()
            with torch.no_grad():
                eal = chain_eal.max(dim=-1).values.mean()
            n_report = k_paths
            del q_node
        elif eal_mode in ("path_topk", "node_marginal", "soft_topk"):
            n_paths = depth_counts[-1]
            if full_leaf_sc is None or full_leaf_sc.size(1) != n_paths:
                raise RuntimeError(
                    "path_topk Tree-EAL needs every complete path scored "
                    "before the node-budget prune")
            n_prefix = depth_starts[-1]
            if gamma == 1:
                alpha_full = torch.softmax(tgt_root / T, dim=-1).gather(
                    -1, full_leaf_tok.long().unsqueeze(-1)).squeeze(-1)
            else:
                prefix_tokens = torch.cat(tokens_per_depth[:-1], dim=1)
                alpha_prefix = torch.softmax(tgt_node[:, :n_prefix] / T, dim=-1).gather(
                    -1, prefix_tokens.long().unsqueeze(-1)).squeeze(-1)
                leaf_parents = torch.tensor(
                    [self.tree_parents[n_prefix + j] for j in range(n_paths)],
                    device=device, dtype=torch.long)
                leaf_logits = tgt_tree.index_select(1, leaf_parents)
                alpha_leaves = torch.softmax(leaf_logits / T, dim=-1).gather(
                    -1, full_leaf_tok.long().unsqueeze(-1)).squeeze(-1)
                alpha_full = torch.cat([alpha_prefix, alpha_leaves], dim=1)
            union_parents, gather_index, _sel = induce_path_topk_union(
                self.tree_parents, self.paths, full_leaf_sc.detach(), path_topk)
            valid = gather_index >= 0
            picked = alpha_full.gather(1, gather_index.clamp(min=0))
            alpha_u = torch.where(valid, picked, torch.zeros_like(picked))
            n_report = float(valid.sum(dim=-1).float().mean().item())
            if eal_mode == "node_marginal":
                if gamma == 1:
                    full_logp = full_leaf_logp
                else:
                    full_logp = torch.cat(prefix_logp + [full_leaf_logp], dim=1)
                logp_u = full_logp.gather(1, gather_index.clamp(min=0))
                eal_loss, eal = node_marginal_eal_loss(
                    alpha_u, union_parents, logp_u, valid)
            elif eal_mode == "soft_topk":
                # Loss = -E[L] on the Top-K path union.
                # Logged accept length is the longest hard path on the budget tree.
                # Forward α = p_target(drafted token). Backward is p - q.
                if gamma == 1:
                    full_drf = full_leaf_pred
                    full_tgt = tgt_root.unsqueeze(1).expand_as(full_drf)
                    full_tok = full_leaf_tok
                else:
                    if full_leaf_pred is None:
                        raise RuntimeError(
                            "soft_topk needs draft logits for every leaf "
                            "before the node-budget prune")
                    full_drf = torch.cat(
                        [drf_node[:, :n_prefix], full_leaf_pred], dim=1)
                    full_tgt = torch.cat(
                        [tgt_node[:, :n_prefix], leaf_logits], dim=1)
                    full_tok = torch.cat([prefix_tokens, full_leaf_tok], dim=1)
                alpha_train = accept_length_surrogate(full_drf, full_tgt, full_tok, T)
                picked_s = alpha_train.gather(1, gather_index.clamp(min=0))
                alpha_s = torch.where(valid, picked_s, torch.zeros_like(picked_s))
                eal_loss = -tree_expected_accepted_length(
                    alpha_s, union_parents).mean()
                eal = longest_accepted_path(tokens, parents, tgt_node)
            else:
                eal = tree_expected_accepted_length(alpha_u, union_parents).mean()
                eal_loss = None
        else:
            raise ValueError(
                "eal_mode must be 'budget', 'path_topk', 'node_marginal', "
                f"'soft_topk', 'sample_minpq', 'sample_anyk', "
                f"'sample_cover', or 'wsoft', got {eal_mode}")
        return {
            "kl": kl,
            "eal": eal,
            "eal_loss": eal_loss,
            "n_nodes": n_report,
            "path_std": (path_std if eal_mode in ("greedy_topk", "top4_kl_eal")
                         else None),
            "path_stats": path_stats if eal_mode == "top4_kl_eal" else None,
        }

    @torch.no_grad()
    def _expand_positions(self, rows, cols, drf_hidden, drf_keys, drf_values, u,
                          mask, path_topk, chunk, n, n_path, path_margin):
        """Expand the draft tree at (rows[i], cols[i]) and write the greedy
        top-`path_topk`-path accept length, the accepted length of each kept
        path and the keep margins (see dense_tree_loss) into n, n_path and
        path_margin."""
        gamma, top_k = self.gamma, self.top_k_per_depth
        dev = drf_hidden.device
        Lm = mask.size(1)
        offs = torch.arange(gamma, device=dev)
        rows_t = torch.as_tensor(rows, device=dev, dtype=torch.long)
        cols_t = torch.as_tensor(cols, device=dev, dtype=torch.long)
        fast = bool(getattr(self, "fast_expand", True))
        n_tree = sum(math.prod(top_k[:d + 1]) for d in range(gamma - 1))
        with (_fold_gqa_attention() if fast else contextlib.nullcontext()):
            self._expand_chunks(rows, cols, rows_t, cols_t, drf_hidden, drf_keys,
                                drf_values, u, mask, path_topk, chunk, n, n_path,
                                path_margin, fast, n_tree, offs, Lm)

    def _expand_chunks(self, rows, cols, rows_t, cols_t, drf_hidden, drf_keys,
                       drf_values, u, mask, path_topk, chunk, n, n_path,
                       path_margin, fast, n_tree, offs, Lm):
        gamma, top_k = self.gamma, self.top_k_per_depth
        dev = drf_hidden.device
        for c in range(0, len(rows), chunk):
            b_t, a_t = rows_t[c:c + chunk], cols_t[c:c + chunk]
            C = a_t.numel()
            kv_lens = [a + 1 for a in cols[c:c + chunk]]
            if fast:
                cache, max_kv = _prealloc_cache_batch(
                    drf_keys, drf_values, rows[c:c + chunk], kv_lens, n_tree)
            else:
                cache, max_kv = _pad_cache_batch(
                    drf_keys, drf_values, rows[c:c + chunk], kv_lens)
            lp = F.log_softmax(self.draft_model.lm_head(drf_hidden[b_t, a_t]).float(), -1)
            sc, tok = lp.topk(top_k[0], dim=-1)
            toks = [tok]
            for d in range(1, gamma):
                n_in = toks[-1].size(1)
                out = self.draft_model(
                    input_ids=toks[-1],
                    attention_mask=self._draft_step_mask_batch(
                        d, kv_lens, max_kv, dev, self.draft_model.dtype),
                    position_ids=(a_t + d).view(C, 1).expand(C, n_in),
                    past_key_values=cache, use_cache=(d < gamma - 1),
                    return_dict=True)
                csc, ctok = F.log_softmax(out.logits.float(), -1).topk(top_k[d], dim=-1)
                sc = (sc.unsqueeze(-1) + csc).reshape(C, -1)
                toks.append(ctok.reshape(C, -1))
                if d < gamma - 1:
                    cache = out.past_key_values
                del out
            nodes = torch.cat(toks, dim=1)
            k_paths = min(int(path_topk), sc.size(1))
            chosen = self.paths[sc.topk(k_paths, dim=-1).indices]      # [C, k, gamma]
            path_tok = nodes.gather(1, chosen.reshape(C, -1)).view(C, k_paths, gamma)
            p = a_t.view(-1, 1) + offs.view(1, -1)
            inside = p < Lm
            p = p.clamp(max=Lm - 1)
            bb = b_t.view(-1, 1).expand_as(p)
            gold = u[bb, p]
            valid = inside & mask[bb, p]
            match = (path_tok == gold.unsqueeze(1)) & valid.unsqueeze(1)
            m_k = match.long().cumprod(-1).sum(-1)                   # [C, k]
            n_path[b_t, a_t, :k_paths] = m_k
            all_tok = nodes[:, self.paths.reshape(-1)].view(C, -1, gamma)
            m_all = ((all_tok == gold.unsqueeze(1)) & valid.unsqueeze(1)
                     ).long().cumprod(-1).sum(-1)                    # [C, P]
            ext = m_all.unsqueeze(-1) > offs.view(1, 1, -1)          # [C, P, gamma]
            sc3 = sc.unsqueeze(-1).expand_as(ext)
            best = sc3.masked_fill(~ext, float("-inf")).max(1).values
            rest = sc3.masked_fill(ext, float("-inf"))
            thr = rest.topk(k_paths, dim=1).values[:, -1]
            path_margin[b_t, a_t] = torch.where(
                ext.any(1), best - thr, torch.full_like(best, float("nan")))
            n[b_t, a_t] = m_k.max(-1).values.float()
            del cache

    @torch.no_grad()
    def _exact_window_starts(self, drf_hidden, drf_keys, drf_values, tgt_hidden,
                             loss_mask, path_topk, chunk=64, all_positions=True,
                             segment_frac=0.0, burn=32, random_frac=0.0):
        """[B, L-1] bool: window starts of greedy top-`path_topk`-path decoding,
        found by running the verification: the draft expands its full tree,
        the path_topk paths with the highest draft probability product are
        kept and the accept length is the longest prefix of the target's
        greedy tokens one of them follows. The target needs no tree forward:
        along accepted nodes the context is the sequence itself, so its greedy
        token at depth j of a window at t is the argmax of its prefix logits
        at t + j.
        all_positions expands every supervised position. Otherwise only the
        starts are expanded: the windows the approximate rule
        (_top4path_accept) predicts are expanded in one batch, then the exact
        walk is followed and, wherever it reaches an unexpanded position s,
        the predicted chain from s up to the first expanded position is
        expanded next, together with up to 3 predicted windows from every
        possible next start s + n + 1 so that few rounds are needed. The
        starts are the same either way, up to bf16 near-ties in the path
        ranking that depend on how positions are batched.
        segment_frac > 0 expands only a random contiguous segment of that
        fraction of each sequence's supervised positions plus `burn` positions
        before it, walks from the start of the burn-in and returns the starts
        inside the segment; window chains from different origins merge within
        a few windows, so these are the full walk's starts. The segment and
        the expanded region are kept in _exact_segment and _exact_known.
        random_frac > 0 expands that fraction of each sequence's supervised
        positions, drawn uniformly, and returns them in place of the starts
        (unweighted E[L] at random positions)."""
        import numpy as np
        mask = loss_mask[:, 1:].bool()
        B, Lm = mask.shape
        gamma, top_k = self.gamma, self.top_k_per_depth
        dev = drf_hidden.device
        u = torch.cat([self.target_model.lm_head(tgt_hidden[:, s:s + 256, :]).argmax(-1)
                       for s in range(0, Lm, 256)], dim=1)[:, :Lm]
        n = torch.zeros(B, Lm, device=dev)
        n_path = torch.zeros(B, Lm, int(path_topk), device=dev, dtype=torch.long)
        path_margin = torch.full((B, Lm, gamma), float("nan"), device=dev)
        args = (drf_hidden, drf_keys, drf_values, u, mask, path_topk, chunk,
                n, n_path, path_margin)
        self._exact_rounds = 1
        self._exact_segment = None
        if random_frac > 0:
            pick = torch.zeros(B, Lm, dtype=torch.bool, device=dev)
            for b in range(B):
                v = mask[b].nonzero(as_tuple=True)[0]
                if v.numel():
                    k = max(1, int(math.ceil(random_frac * v.numel())))
                    pick[b, v[torch.randperm(v.numel(), device=dev)[:k]]] = True
            rows, cols = pick.nonzero(as_tuple=True)
            self._expand_positions(rows.tolist(), cols.tolist(), *args)
            self._exact_known = pick
            self._exact_path_n = n_path
            self._exact_path_margin = path_margin
            return pick, n
        if segment_frac > 0:
            mask_c = mask.cpu().numpy()
            seg = np.zeros((B, Lm), dtype=bool)
            region = np.zeros((B, Lm), dtype=bool)
            for b in range(B):
                v = mask_c[b].nonzero()[0]
                if v.size == 0:
                    continue
                p0, p1 = int(v[0]), int(v[-1]) + 1
                seg_len = max(1, int(math.ceil(segment_frac * (p1 - p0))))
                s0 = p0 + int(np.random.randint(0, p1 - p0 - seg_len + 1))
                seg[b, s0:s0 + seg_len] = True
                region[b, max(p0, s0 - burn):s0 + seg_len] = True
            seg &= mask_c
            region &= mask_c
            rows, cols = region.nonzero()
            self._expand_positions(rows.tolist(), cols.tolist(), *args)
            n_c = n.long().cpu().numpy()
            start = torch.zeros(B, Lm, dtype=torch.bool)
            for b in range(B):
                r = region[b].nonzero()[0]
                if r.size == 0:
                    continue
                s, end = int(r[0]), int(r[-1]) + 1
                while s < end:
                    if seg[b, s]:
                        start[b, s] = True
                    s += int(n_c[b, s]) + 1
            self._exact_known = torch.from_numpy(region).to(dev)
            self._exact_segment = torch.from_numpy(seg).to(dev)
            self._exact_path_n = n_path
            self._exact_path_margin = path_margin
            return start.to(dev), n
        if all_positions:
            rows, cols = mask.nonzero(as_tuple=True)
            self._expand_positions(rows.tolist(), cols.tolist(), *args)
            self._exact_known = mask.clone()
        else:
            k_max = max(top_k)
            ranks, lps = [], []
            for s in range(0, Lm, 256):
                dl = self.draft_model.lm_head(drf_hidden[:, s:min(Lm, s + 256)]).float()
                z_u = dl.gather(-1, u[:, s:s + dl.size(1)].unsqueeze(-1))
                ranks.append((dl > z_u).sum(-1))
                lps.append(dl.log_softmax(-1).topk(k_max, dim=-1).values)
                del dl
            n_appr = _top4path_accept(torch.cat(ranks, 1), torch.cat(lps, 1), mask,
                                      list(top_k), int(path_topk)).long().cpu().numpy()
            mask_c = mask.cpu().numpy()
            known = ~mask_c
            n_c = np.zeros((B, Lm), dtype=np.int64)
            span = [(int(v[0]), int(v[-1]) + 1) if v.size else (0, 0)
                    for v in (mask_c[b].nonzero()[0] for b in range(B))]
            first = True
            while True:
                todo = set()
                for b in range(B):
                    s, end = span[b]
                    while s < end and known[b, s]:
                        s += int(n_c[b, s]) + 1
                    if s >= end:
                        continue
                    heads = [s] if first else [s] + [s + ell + 1 for ell in range(gamma + 1)]
                    for h, t in enumerate(heads):
                        steps = 0
                        while t < end and not known[b, t] and (h == 0 or steps < 3):
                            todo.add((b, t))
                            t += int(n_appr[b, t]) + 1
                            steps += 1
                first = False
                if not todo:
                    break
                rows, cols = map(list, zip(*sorted(todo)))
                self._expand_positions(rows, cols, *args)
                known[rows, cols] = True
                n_c[rows, cols] = n[rows, cols].long().cpu().numpy()
                self._exact_rounds += 1
            self._exact_rounds -= 1
            self._exact_known = torch.from_numpy(known & mask_c).to(dev)
        self._exact_path_n = n_path
        self._exact_path_margin = path_margin
        return _walk_windows(n, mask), n

    def _eal_gradnorm_coef(self, kl_part, eal_loss, ratio):
        """EAL coefficient that makes its gradient `ratio` times as long as the
        KL gradient. The norms are measured every `eal_gn_every` calls on the
        draft parameters and the coefficient is an EMA over measurements."""
        n = getattr(self, "_eal_gn_calls", 0)
        self._eal_gn_calls = n + 1
        if n % int(getattr(self, "eal_gn_every", 10)) == 0:
            params = [p for p in self.draft_model.parameters() if p.requires_grad]

            def gnorm(loss):
                gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                return torch.sqrt(sum(g.float().pow(2).sum() for g in gs if g is not None))

            n_kl, n_eal = torch.stack([gnorm(kl_part), gnorm(eal_loss)]).tolist()
            coef = ratio * n_kl / max(n_eal, 1e-8)
            prev = getattr(self, "_eal_gn_coef", None)
            self._eal_gn_coef = coef if prev is None else 0.9 * prev + 0.1 * coef
            self._eal_gn_norms = (n_kl, n_eal)
        return self._eal_gn_coef

    def forward(self, input_ids, attention_mask, loss_mask,
                anchor_kl_coef=1.0, eal_aux_coef=0.1, temperature=1.0,
                eal_mode="budget", path_topk=16, n_anchors=1, n_draws=1,
                eal_score_coef=0.1, path_eal_weights=None,
                path_eal_mean=False, path_eal_var_coef=0.0, **kwargs):
        B, L = input_ids.shape
        device = input_ids.device

        # n_anchors distinct trees per sequence. One tree per sequence covers
        # a single position out of ~900 supervised ones, which is far less
        # signal per step than the chain losses get.
        # n_draws independent trees per anchor. The score-function baseline has
        # to compare draws at the same position: positions differ in difficulty
        # much more than two draws at one position do, so a baseline taken
        # across positions measures "was this position easy" and inflates the
        # draft entropy instead of steering the sampling.
        n_anchors = max(int(n_anchors), 1)
        n_draws = max(int(n_draws), 1)
        batch_idx = []
        anchor_pos = []
        anchor_group = []
        n_groups = 0
        for b in range(B):
            valid = (loss_mask[b] == 1).nonzero(as_tuple=True)[0]
            if len(valid) == 0:
                continue
            k = min(n_anchors, int(valid.numel()))
            sel = torch.randperm(int(valid.numel()), device=valid.device)[:k]
            for s in sel.tolist():
                for _ in range(n_draws):
                    batch_idx.append(b)
                    anchor_pos.append(int(valid[s].item()))
                    anchor_group.append(n_groups)
                n_groups += 1

        # Only the anchor rows need logits, so the LM head runs on those rows
        # instead of materializing [B, L, V].
        def _prefix(lm):
            out = lm.model(input_ids=input_ids, attention_mask=attention_mask,
                           use_cache=True, return_dict=True)
            keys, values = _extract_kv_lists(out.past_key_values)
            return out.last_hidden_state, keys, values

        def _rows(lm, hidden):
            b_t = torch.tensor(batch_idx, device=hidden.device, dtype=torch.long)
            a_t = torch.tensor(anchor_pos, device=hidden.device, dtype=torch.long)
            return lm.lm_head(hidden[b_t, a_t]).float()

        start_anchors = (eal_mode == "top4_kl_eal"
                         and bool(getattr(self, "dense_eal_start", False)))
        top4_dense_kl = (eal_mode == "top4_kl_eal"
                         and bool(getattr(self, "dense_kl", False)))
        keep_tgt = eal_mode == "dense_tree" or start_anchors or top4_dense_kl
        split = start_anchors and bool(getattr(self, "eal_start_split", False))
        tree_kl_w = tree_eal_w = None
        with torch.no_grad():
            tgt_hidden, tgt_keys, tgt_values = _prefix(self.target_model)
            if not start_anchors:
                tgt_anchor = _rows(self.target_model, tgt_hidden) if batch_idx else None
            if not keep_tgt:
                del tgt_hidden

        drf_hidden, drf_keys, drf_values = _prefix(self.draft_model)

        if start_anchors:
            # Anchors only where greedy tree decoding opens a window.
            if getattr(self, "exact_starts", False):
                starts, _ = self._exact_window_starts(
                    drf_hidden.detach(), drf_keys, drf_values, tgt_hidden,
                    loss_mask, path_topk,
                    chunk=int(getattr(self, "exact_starts_chunk", 64)),
                    all_positions=False)
            else:
                starts = decode_window_starts(
                    drf_hidden.detach(), tgt_hidden, self.draft_model.lm_head,
                    self.target_model.lm_head, loss_mask, list(self.top_k_per_depth),
                    path_topk=(int(path_topk) if getattr(self, "top4path_starts", False)
                               else 0))
            if split:
                # Keep the uniform trees for KL; the window-start trees below
                # carry the EAL.
                n_uniform = len(batch_idx)
            else:
                batch_idx, anchor_pos, anchor_group = [], [], []
                n_groups = 0
            for b in range(B):
                valid = starts[b].nonzero(as_tuple=True)[0]
                if len(valid) == 0:
                    continue
                k = min(n_anchors, int(valid.numel()))
                sel = torch.randperm(int(valid.numel()), device=valid.device)[:k]
                for s in sel.tolist():
                    for _ in range(n_draws):
                        batch_idx.append(b)
                        anchor_pos.append(int(valid[s].item()))
                        anchor_group.append(n_groups)
                    n_groups += 1
            if split:
                is_start = torch.zeros(len(batch_idx), device=device)
                is_start[n_uniform:] = 1.0
                tree_kl_w, tree_eal_w = 1.0 - is_start, is_start
            with torch.no_grad():
                tgt_anchor = _rows(self.target_model, tgt_hidden) if batch_idx else None
            if not top4_dense_kl:
                del tgt_hidden

        if not batch_idx:
            z = drf_hidden.sum() * 0.0
            del drf_hidden
            return z, z.detach(), {
                "total_loss": 0.0, "kl_loss": 0.0, "aux_loss": 0.0,
                "eal_mean": 0.0, "num_valid": 0.0, "num_anchors": 0,
                "mean_tau": 0.0, "eagle_loss": 0.0, "tree_nodes": 0,
                "eal_loss": 0.0, "dense_loss": 0.0,
            }

        dense_loss = None
        drf_anchor = _rows(self.draft_model, drf_hidden)
        if eal_mode == "dense_tree":
            exact = None
            if getattr(self, "exact_starts", False):
                exact, exact_n = self._exact_window_starts(
                    drf_hidden.detach(), drf_keys, drf_values, tgt_hidden,
                    loss_mask, path_topk,
                    chunk=int(getattr(self, "exact_starts_chunk", 64)),
                    all_positions=bool(getattr(self, "prob_starts", False)),
                    segment_frac=float(getattr(self, "exact_segment_frac", 0.0)),
                    burn=int(getattr(self, "exact_burn", 32)),
                    random_frac=float(getattr(self, "exact_random_frac", 0.0)))
                exact_mask = self._exact_known
                exact_stats = torch.stack([
                    exact.float().sum() / B,
                    exact_n[exact].mean() if exact.any() else exact_n.new_zeros(()),
                    exact_n[exact_mask].mean(),
                    exact_mask.float().sum() / loss_mask[:, 1:].float().sum(),
                    exact_n.new_tensor(float(self._exact_rounds))])
            top4_start = exact is not None and getattr(self, "top4_start_eal", False)
            d_kl, d_eal, d_var, d_hard = dense_tree_loss(
                drf_hidden, tgt_hidden, self.draft_model.lm_head,
                self.target_model.lm_head, loss_mask, list(self.top_k_per_depth),
                tau=float(getattr(self, "greedy_tau", 1.0)),
                var_coef=path_eal_var_coef,
                start_only=bool(getattr(self, "dense_eal_start", False)),
                starts=exact,
                path_n=self._exact_path_n if top4_start else None,
                temp=temperature,
                prob_starts=top4_start and getattr(self, "prob_starts", False),
                path_margin=self._exact_path_margin if top4_start else None,
                prob_region=(self._exact_known if top4_start
                             and self._exact_segment is not None else None),
                eal_region=self._exact_segment if top4_start else None)
            del tgt_hidden
        elif top4_dense_kl:
            d_kl = dense_tree_loss(
                drf_hidden, tgt_hidden, self.draft_model.lm_head,
                self.target_model.lm_head, loss_mask, list(self.top_k_per_depth),
                tau=1.0)[0]
            del tgt_hidden
        del drf_hidden

        out = self._rollout_batch(
            batch_idx, anchor_pos, drf_anchor, tgt_anchor,
            drf_keys, drf_values, tgt_keys, tgt_values, temperature,
            eal_mode=eal_mode, path_topk=path_topk,
            eal_score_coef=eal_score_coef, anchor_group=anchor_group,
            anchor_kl_coef=anchor_kl_coef, path_eal_weights=path_eal_weights,
            path_eal_mean=path_eal_mean, path_eal_var_coef=path_eal_var_coef,
            tree_kl_w=tree_kl_w, tree_eal_w=tree_eal_w)

        if eal_mode in ("node_marginal", "soft_topk", "sample_minpq",
                        "sample_anyk", "sample_cover", "wsoft", "greedy_topk"):
            total_loss = out["eal_loss"]
            if dense_loss is not None:
                total_loss = total_loss + dense_loss
        elif eal_mode == "top4_kl_eal":
            kl_part = anchor_kl_coef * out["kl"]
            if top4_dense_kl:
                dense_loss = d_kl
                kl_part = kl_part + d_kl
            eal_coef = eal_aux_coef
            gn_ratio = float(getattr(self, "eal_gn_ratio", 0.0))
            if gn_ratio > 0 and self.training and torch.is_grad_enabled():
                eal_coef = self._eal_gradnorm_coef(kl_part, out["eal_loss"], gn_ratio)
            total_loss = kl_part + eal_coef * out["eal_loss"]
            if getattr(self, "keep_parts", False):
                self.last_parts = {"top4_kl": out["kl"], "eal_loss": out["eal_loss"],
                                   "dense_kl": d_kl if top4_dense_kl else None}
        elif eal_mode == "dense_tree":
            # dense KL + anchor_kl_coef * rollout-node KL
            #   - eal_aux_coef * (dense tree E[L] - var_coef * Var_windows E[L])
            out["eal_loss"] = -(d_eal - float(path_eal_var_coef) * d_var)
            dense_loss = d_kl
            kl_part = d_kl + anchor_kl_coef * out["kl"]
            eal_coef = eal_aux_coef
            gn_ratio = float(getattr(self, "eal_gn_ratio", 0.0))
            if gn_ratio > 0 and self.training and torch.is_grad_enabled():
                eal_coef = self._eal_gradnorm_coef(kl_part, out["eal_loss"], gn_ratio)
            total_loss = kl_part + eal_coef * out["eal_loss"]
        else:
            total_loss = anchor_kl_coef * out["kl"] - eal_aux_coef * out["eal"]
        eal_loss_value = (out["eal_loss"] if out["eal_loss"] is not None
                          else out["eal"].new_zeros(()))
        dense_value = (dense_loss if dense_loss is not None
                       else total_loss.new_zeros(()))

        total_v, kl_v, eal_v, eal_loss_v, dense_v = torch.stack([
            total_loss.detach().float(), out["kl"].detach().float(),
            out["eal"].detach().float(), eal_loss_value.detach().float(),
            dense_value.detach().float()]).tolist()
        metrics = {
            "total_loss": total_v,
            "kl_loss": kl_v,
            "aux_loss": -eal_v,
            "eal_mean": eal_v,
            "eal_loss": eal_loss_v,
            "dense_loss": dense_v,
            "num_anchors": len(batch_idx),
            "num_valid": float(len(batch_idx)),
            "mean_tau": eal_v,
            "eagle_loss": kl_v,
            "tree_nodes": out["n_nodes"],
        }
        if out.get("path_std") is not None:
            metrics["path_eal_std"] = float(out["path_std"])
        if out.get("path_stats") is not None:
            stats = out["path_stats"].float().tolist()
            k = (len(stats) - 2) // 2
            for i in range(k):
                metrics[f"path{i + 1}_eal"] = stats[i]
                metrics[f"path{i + 1}_al"] = stats[k + i]
            metrics["path_eal_var"], metrics["path_al_var"] = stats[-2:]
        if eal_mode == "top4_kl_eal" and getattr(self, "_eal_gn_coef", None) is not None:
            metrics["eal_gn_coef"] = self._eal_gn_coef
            metrics["gnorm_kl"], metrics["gnorm_eal"] = self._eal_gn_norms
        if eal_mode == "dense_tree":
            metrics["dense_eal"], metrics["dense_var"], metrics["dense_hard_al"] = (
                torch.stack([d_eal.detach(), d_var.detach(), d_hard]).float().tolist())
            if exact is not None:
                (metrics["windows_per_seq"], metrics["exact_al_at_starts"],
                 metrics["exact_al_all_pos"], metrics["exact_expanded_frac"],
                 metrics["exact_rounds"]) = exact_stats.float().tolist()
                if getattr(self, "prob_starts", False):
                    metrics["prob_windows_per_seq"] = dense_tree_loss.last_windows_per_seq
            if getattr(self, "_eal_gn_coef", None) is not None:
                metrics["eal_gn_coef"] = self._eal_gn_coef
                metrics["gnorm_kl"], metrics["gnorm_eal"] = self._eal_gn_norms
        return total_loss, total_loss.detach(), metrics
