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

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None


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
    with. Padding is masked out by the tree attention mask.
    """
    max_kv = max(kv_lens)
    B = len(batch_idx)
    new_cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(zip(keys, values)):
        _, n_kv, _, dim = k.shape
        k_out = k.new_zeros(B, n_kv, max_kv, dim)
        v_out = v.new_zeros(B, n_kv, max_kv, dim)
        for i, b in enumerate(batch_idx):
            n = kv_lens[i]
            k_out[i, :, :n] = k[b, :, :n]
            v_out[i, :, :n] = v[b, :, :n]
        new_cache.update(k_out, v_out, layer_idx)
    return new_cache, max_kv


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

    def _draft_step_mask_batch(self, d_step, kv_lens, max_kv, device, dtype):
        """Batched 4D mask. Tree nodes are appended after the right-padded prefix."""
        min_val = torch.finfo(dtype).min
        depth_idx = d_step - 1
        depth_counts = self.tree_depth_counts
        depth_starts = self.tree_depth_starts
        top_k = self.top_k_per_depth
        q_len = depth_counts[depth_idx]
        cache_before = depth_starts[depth_idx]
        kv_len = max_kv + cache_before + q_len
        B = len(kv_lens)
        mask = torch.full((B, 1, q_len, kv_len), min_val, device=device, dtype=dtype)
        for b, kv in enumerate(kv_lens):
            mask[b, 0, :, :kv] = 0.0
        for q in range(q_len):
            cur = q
            for k_back in range(depth_idx - 1, -1, -1):
                cur = cur // top_k[k_back + 1]
                slot = max_kv + depth_starts[k_back] + cur
                mask[:, 0, q, slot] = 0.0
            mask[:, 0, q, max_kv + cache_before + q] = 0.0
        return mask

    def _target_mask_batch(self, kv_lens, max_kv, parents, device, dtype):
        """Per-sample ancestor mask for a budget-pruned tree. parents: [B, N]."""
        B, N = parents.shape
        min_val = torch.finfo(dtype).min
        kv_total = max_kv + N
        mask = torch.full((B, 1, N, kv_total), min_val, device=device, dtype=dtype)
        for b, kv in enumerate(kv_lens):
            mask[b, 0, :, :kv] = 0.0
        b_idx = torch.arange(B, device=device)[:, None].expand(B, N)
        q_idx = torch.arange(N, device=device)[None, :].expand(B, N)
        cur = q_idx.clone()
        alive = torch.ones(B, N, dtype=torch.bool, device=device)
        for _ in range(self.gamma):
            if not bool(alive.any()):
                break
            mask[b_idx[alive], 0, q_idx[alive], max_kv + cur[alive]] = 0.0
            nxt = parents.gather(1, cur.clamp(min=0))
            cur = nxt
            alive = alive & (cur >= 0)
        return mask

    def _rollout_batch(self, batch_idx, anchor_pos,
                       drf_anchor_logits, tgt_anchor_logits,
                       drf_keys, drf_values, tgt_keys, tgt_values,
                       temperature, eal_mode="budget", path_topk=16,
                       eal_score_coef=0.1, anchor_group=None,
                       anchor_kl_coef=0.0):
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
                "wsoft"):
            kl = drf_root.new_zeros(())
        else:
            tgt_logp = F.log_softmax(tgt_node, dim=-1)
            tgt_p = tgt_logp.exp()
            drf_logp = F.log_softmax(drf_node, dim=-1)
            kl = (tgt_p * (tgt_logp - drf_logp)).sum(-1).clamp(min=0.0).mean()
        if eal_mode == "budget":
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
        }

    def forward(self, input_ids, attention_mask, loss_mask,
                anchor_kl_coef=1.0, eal_aux_coef=0.1, temperature=1.0,
                eal_mode="budget", path_topk=16, n_anchors=1, n_draws=1,
                eal_score_coef=0.1, **kwargs):
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

        def _rows(logits):
            b_t = torch.tensor(batch_idx, device=logits.device, dtype=torch.long)
            a_t = torch.tensor(anchor_pos, device=logits.device, dtype=torch.long)
            return logits[b_t, a_t].float()

        with torch.no_grad():
            tgt_out = self.target_model(
                input_ids=input_ids, attention_mask=attention_mask,
                use_cache=True, return_dict=True)
            tgt_keys, tgt_values = _extract_kv_lists(tgt_out.past_key_values)
            tgt_anchor = _rows(tgt_out.logits) if batch_idx else None
            del tgt_out
            torch.cuda.empty_cache()

        drf_out = self.draft_model(
            input_ids=input_ids, attention_mask=attention_mask,
            use_cache=True, return_dict=True)
        drf_keys, drf_values = _extract_kv_lists(drf_out.past_key_values)

        if not batch_idx:
            z = drf_out.logits.sum() * 0.0
            del drf_out
            return z, z.detach(), {
                "total_loss": 0.0, "kl_loss": 0.0, "aux_loss": 0.0,
                "eal_mean": 0.0, "num_valid": 0.0, "num_anchors": 0,
                "mean_tau": 0.0, "eagle_loss": 0.0, "tree_nodes": 0,
                "eal_loss": 0.0, "dense_loss": 0.0,
            }

        dense_loss = None
        drf_anchor = _rows(drf_out.logits)
        del drf_out

        out = self._rollout_batch(
            batch_idx, anchor_pos, drf_anchor, tgt_anchor,
            drf_keys, drf_values, tgt_keys, tgt_values, temperature,
            eal_mode=eal_mode, path_topk=path_topk,
            eal_score_coef=eal_score_coef, anchor_group=anchor_group,
            anchor_kl_coef=anchor_kl_coef)

        if eal_mode in ("node_marginal", "soft_topk", "sample_minpq",
                        "sample_anyk", "sample_cover", "wsoft"):
            total_loss = out["eal_loss"]
            if dense_loss is not None:
                total_loss = total_loss + dense_loss
        else:
            total_loss = anchor_kl_coef * out["kl"] - eal_aux_coef * out["eal"]
        eal_loss_value = (out["eal_loss"] if out["eal_loss"] is not None
                          else out["eal"].new_zeros(()))
        dense_value = (dense_loss if dense_loss is not None
                       else total_loss.new_zeros(()))

        metrics = {
            "total_loss": total_loss.item(),
            "kl_loss": out["kl"].item(),
            "aux_loss": (-out["eal"]).item(),
            "eal_mean": out["eal"].item(),
            "eal_loss": eal_loss_value.item(),
            "dense_loss": dense_value.item(),
            "num_anchors": len(batch_idx),
            "num_valid": float(len(batch_idx)),
            "mean_tau": out["eal"].item(),
            "eagle_loss": out["kl"].item(),
            "tree_nodes": out["n_nodes"],
        }
        return total_loss, total_loss.detach(), metrics
