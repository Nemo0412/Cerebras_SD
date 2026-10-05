"""Actual verify_tree_step accepted length vs the Tree-EAL recursion.

Each setting builds a tree whose sibling probabilities are a valid softmax
(they share one parent distribution, so they sum to at most 1), then:

  computed = tree_expected_accepted_length
  actual   = mean of verify_tree_step(..., verify_mode='simple', T=1)

A second block compares the training Top-K union against the budget-128
inference tree. A third block compares the soft plug-in to the mean E[L]
of K-with-replacement path subsets.
"""
import math
import os
import random
import sys
from collections import defaultdict
from multiprocessing import get_context

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from small_lm_tree_v2_model import (
    _build_tree_topology,
    induce_path_topk_union,
    soft_topk_tree_eal,
    tree_expected_accepted_length,
)
from tree_spec_decode import DraftTree, VerifyState, verify_tree_step

N_TRIALS = 2000
PROD = (4, 3, 2, 1, 1, 1, 1)


class _Out:
    def __init__(self, logits, cache):
        self.logits = logits
        self.past_key_values = cache


class _Cache:
    def __init__(self, seq):
        self.key_cache = [torch.zeros(1, 1, seq, 1)]
        self.value_cache = [torch.zeros(1, 1, seq, 1)]


class _Stub:
    """Target whose softmax at each parent slot equals the prescribed alphas."""

    def __init__(self, logits):
        self.logits = logits
        self._p = torch.zeros(1)

    def parameters(self):
        return iter([self._p])

    def __call__(self, input_ids, attention_mask=None, position_ids=None,
                 past_key_values=None, use_cache=True):
        prefix = 0
        if past_key_values is not None and past_key_values.key_cache:
            prefix = past_key_values.key_cache[0].shape[-2]
        return _Out(self.logits, _Cache(prefix + input_ids.shape[1]))


def depths_of(parents):
    depth = [0] * len(parents)
    for i, p in enumerate(parents):
        if p >= 0:
            depth[i] = depth[p] + 1
    return depth


def make_alpha(parents, mass_of_depth, split, seed):
    """Per-slot categorical: children of one parent sum to mass <= 1."""
    g = torch.Generator().manual_seed(seed)
    parents = [int(p) for p in parents]
    depth = depths_of(parents)
    groups = defaultdict(list)
    for i, p in enumerate(parents):
        groups[0 if p < 0 else 1 + p].append(i)
    alpha = torch.zeros(len(parents))
    for slot, nodes in groups.items():
        d = 0 if slot == 0 else depth[slot - 1] + 1
        mass = float(mass_of_depth(d))
        if not 0.0 < mass <= 1.0:
            raise ValueError(f"slot mass {mass} at depth {d}")
        m = len(nodes)
        if split == "equal":
            w = torch.full((m,), 1.0 / m)
        elif split == "skew":
            if m == 1:
                w = torch.ones(1)
            else:
                w = torch.tensor([0.7] + [0.3 / (m - 1)] * (m - 1))
        elif split == "random":
            w = torch.rand(m, generator=g).clamp(min=1e-3)
            w = w / w.sum()
        else:
            raise ValueError(split)
        alpha[nodes] = w * mass
    return alpha


def logits_for(alpha, parents):
    parents = [int(p) for p in parents]
    n = len(parents)
    vocab = n + 2
    dump = n + 1
    tokens = list(range(n))
    logits = torch.full((1, n + 1, vocab), -1e9)
    groups = defaultdict(list)
    for i, p in enumerate(parents):
        groups[0 if p < 0 else 1 + p].append(i)
    for slot, nodes in groups.items():
        mass = float(sum(alpha[i] for i in nodes))
        if mass > 1.0 + 1e-5:
            raise ValueError(f"slot {slot} mass {mass} > 1")
        row = logits[0, slot]
        for i in nodes:
            row[tokens[i]] = math.log(max(float(alpha[i]), 1e-12))
        row[dump] = math.log(max(1.0 - mass, 1e-12))
    return logits, tokens


def analytic(alpha, parents):
    return tree_expected_accepted_length(
        alpha.view(1, -1),
        torch.tensor(parents, dtype=torch.long).view(1, -1),
    ).item()


def _verify_chunk(payload):
    alpha, parents, n_trials, seed = payload
    torch.set_num_threads(1)
    logits, tokens = logits_for(alpha, parents)
    depth = depths_of(parents)
    tree = DraftTree(
        tokens=torch.tensor(tokens, dtype=torch.long),
        parents=torch.tensor(parents, dtype=torch.long),
        depths=torch.tensor(depth, dtype=torch.long),
        scores=torch.zeros(len(parents)),
        depth_counts=[],
        device=torch.device("cpu"),
    )
    model = _Stub(logits)
    random.seed(seed)
    total = 0
    total_sq = 0.0
    for _ in range(n_trials):
        state = VerifyState(
            cache=_Cache(1),
            cache_len=1,
            pending_token=torch.tensor([[0]]),
        )
        _toks, n_acc, _t, _st, _path = verify_tree_step(
            model, state, tree, temperature=1.0, verify_mode="simple")
        total += n_acc
        total_sq += float(n_acc * n_acc)
    return total, total_sq


def actual_mean(alpha, parents, n_trials, seed):
    parents = [int(p) for p in parents]
    alpha = alpha.detach().float().reshape(-1).tolist()
    n_workers = min(8, n_trials)
    base, extra = divmod(n_trials, n_workers)
    payloads = []
    cursor = 0
    for w in range(n_workers):
        count = base + (1 if w < extra else 0)
        if count <= 0:
            continue
        payloads.append((alpha, parents, count, seed + 1009 * w))
        cursor += count
    ctx = get_context("fork")
    with ctx.Pool(len(payloads)) as pool:
        parts = pool.map(_verify_chunk, payloads)
    total = sum(p[0] for p in parts)
    total_sq = sum(p[1] for p in parts)
    mean = total / n_trials
    var = max(total_sq / n_trials - mean * mean, 0.0) * n_trials / (n_trials - 1)
    return mean, math.sqrt(var)


def report(name, alpha, parents, seed):
    comp = analytic(alpha, parents)
    act, std = actual_mean(alpha, parents, N_TRIALS, seed)
    se = std / math.sqrt(N_TRIALS)
    gap = act - comp
    z = gap / se if se > 0 else 0.0
    flag = "MATCH" if abs(z) < 4 else "MISMATCH"
    print(
        f"{name:22s}  computed={comp:8.4f}  actual={act:8.4f}  "
        f"gap={gap:+8.4f}  se={se:.4f}  z={z:+5.2f}  {flag}  "
        f"nodes={len(parents)} mean_a={float(alpha.mean()):.3f}",
        flush=True,
    )
    return flag


def remap(keep, parents, alpha):
    old_to_new = {old: i for i, old in enumerate(keep)}
    new_parents = []
    for old in keep:
        p = int(parents[old])
        new_parents.append(-1 if p < 0 else old_to_new[p])
    return alpha[keep].clone(), new_parents


def budget_keep(parents, scores, budget):
    """Prefix nodes, then highest-score leaves, until `budget` nodes."""
    n = len(parents)
    children = [[] for _ in range(n)]
    for i, p in enumerate(parents):
        if p >= 0:
            children[p].append(i)
    leaves = [i for i in range(n) if not children[i]]
    prefix = [i for i in range(n) if children[i]]
    room = budget - len(prefix)
    ranked = sorted(leaves, key=lambda i: float(scores[i]), reverse=True)
    return prefix + ranked[:max(room, 0)]


def main():
    print(f"trials={N_TRIALS} verify_mode=simple T=1", flush=True)
    print("--- same tree: recursion vs verify_tree_step ---", flush=True)
    flags = []

    flags.append(report(
        "depth2_textbook",
        torch.tensor([0.50, 0.40, 0.30, 0.20, 0.60, 0.10]),
        [-1, -1, 0, 0, 1, 1],
        seed=1,
    ))
    chain_p = [-1] + list(range(6))
    flags.append(report(
        "chain7_a0.5",
        torch.full((7,), 0.5),
        chain_p,
        seed=2,
    ))
    flags.append(report(
        "chain7_a0.9",
        torch.full((7,), 0.9),
        chain_p,
        seed=3,
    ))
    flags.append(report(
        "two_roots",
        torch.tensor([0.50, 0.40]),
        [-1, -1],
        seed=4,
    ))

    shapes = {
        "wide8": (8,),
        "branch_2222": (2, 2, 2, 2),
        "prod_4332111": PROD,
        "shallow_432": (4, 3, 2),
    }
    built = {}
    for name, spec in shapes.items():
        parents, _d, paths, counts, starts, n = _build_tree_topology(spec)
        built[name] = (parents, paths, counts, starts, n)

    regimes = [
        ("low", lambda d: 0.25, "equal"),
        ("mid", lambda d: 0.55, "equal"),
        ("high", lambda d: 0.90, "equal"),
        ("decay", lambda d: 0.85 * (0.75 ** d), "equal"),
        ("skew", lambda d: 0.80, "skew"),
        ("random", lambda d: 0.70, "random"),
    ]
    seed = 10
    for shape, (parents, _paths, _c, _s, _n) in built.items():
        for reg, mass, split in regimes:
            if shape != "prod_4332111" and reg not in ("mid", "skew", "random"):
                continue
            alpha = make_alpha(parents, mass, split, seed)
            flags.append(report(f"{shape}_{reg}", alpha, parents, seed))
            seed += 1

    print("--- Top-K union, same tree on both sides ---", flush=True)
    parents, paths, counts, starts, n = built["prod_4332111"]
    alpha = make_alpha(parents, lambda d: 0.55, "equal", seed=100)
    # Higher index = higher score, so path 0 is outside Top-16.
    path_scores = torch.arange(counts[-1], dtype=torch.float32)
    paths_t = torch.tensor(paths, dtype=torch.long)
    for k in (1, 4, 8, 16, 24):
        union_p, gather, _sel = induce_path_topk_union(
            parents, paths_t, path_scores.view(1, -1), k)
        keep = [int(x) for x in gather[0].tolist() if x >= 0]
        a_u, p_u = remap(keep, parents, alpha)
        flags.append(report(f"union_k{k}", a_u, p_u, seed=200 + k))

    print("--- computed union vs actual budget-128 tree ---", flush=True)
    n_prefix = starts[-1]
    leaf_scores = torch.zeros(n)
    for j, sc in enumerate(path_scores.tolist()):
        leaf_scores[n_prefix + j] = sc
    for label, mass, split, sc in (
        ("mid", lambda d: 0.55, "equal", leaf_scores),
        ("skew", lambda d: 0.80, "skew", leaf_scores),
        ("private_high", lambda d: 0.05, "equal", leaf_scores),
    ):
        a = make_alpha(parents, mass, split, seed=300)
        if label == "private_high":
            # Path 0 has the lowest score, so Top-16 drops it. Make that
            # path certain and every other node unlikely.
            a = a.clone()
            a[:] = 0.02
            for node in paths[0]:
                a[node] = 0.98
            # Renormalize siblings so the stub softmax stays valid.
            groups = defaultdict(list)
            for i, p in enumerate(parents):
                groups[0 if p < 0 else 1 + p].append(i)
            for nodes in groups.values():
                s = float(a[nodes].sum())
                if s > 1:
                    a[nodes] = a[nodes] / s
        union_p, gather, _sel = induce_path_topk_union(
            parents, paths_t, path_scores.view(1, -1), 16)
        keep_u = [int(x) for x in gather[0].tolist() if x >= 0]
        a_u, p_u = remap(keep_u, parents, a)
        comp = analytic(a_u, p_u)
        keep_b = budget_keep(parents, sc, 128)
        a_b, p_b = remap(keep_b, parents, a)
        act, std = actual_mean(a_b, p_b, N_TRIALS, seed=400)
        se = std / math.sqrt(N_TRIALS)
        gap = act - comp
        z = gap / se if se > 0 else 0.0
        # This gap is expected when the two trees differ. Report it, do not
        # treat it as a recursion failure.
        print(
            f"{'union_vs_budget_' + label:22s}  computed_union={comp:8.4f}  "
            f"actual_budget={act:8.4f}  gap={gap:+8.4f}  se={se:.4f}  z={z:+5.2f}  "
            f"union_nodes={len(p_u)} budget_nodes={len(p_b)}",
            flush=True,
        )
        # Sanity: recursion on the budget tree itself must match the verifier.
        flags.append(report(f"budget128_{label}", a_b, p_b, seed=500))

    print("--- soft plug-in vs mean E[L] of random subsets ---", flush=True)
    alpha_row = alpha.view(1, -1)
    parent_t = torch.tensor(parents, dtype=torch.long)
    for tag, scores, k, tau in (
        ("peaked_k16_t1", torch.linspace(0, -12, 24), 16, 1.0),
        ("flat_k16_t1", torch.zeros(24), 16, 1.0),
        ("mild_k16_t4", torch.linspace(0, -6, 24), 16, 4.0),
        ("peaked_k1_t1", torch.linspace(0, -12, 24), 1, 1.0),
        ("peaked_k8_t1", torch.linspace(0, -12, 24), 8, 1.0),
    ):
        s = scores.view(1, -1)
        _loss, e_soft, e_hard = soft_topk_tree_eal(
            alpha_row, parent_t, paths_t, s, k, tau=tau)
        pi = torch.softmax(s / tau, dim=-1)[0]
        n_sub = 1500
        g = torch.Generator().manual_seed(7)
        draws = torch.multinomial(
            pi.expand(n_sub, -1), num_samples=k, replacement=True, generator=g)
        drawn = paths_t[draws].reshape(n_sub, -1)
        mask = torch.zeros(n_sub, n)
        mask.scatter_(1, drawn, 1.0)
        alpha_s = alpha_row.expand(n_sub, -1) * mask
        parents_s = parent_t.view(1, -1).expand(n_sub, -1)
        chunks = []
        for lo in range(0, n_sub, 250):
            chunks.append(tree_expected_accepted_length(
                alpha_s[lo:lo + 250], parents_s[lo:lo + 250]))
        e_sub = torch.cat(chunks)
        sub_mean = e_sub.mean().item()
        sub_se = e_sub.std(unbiased=True).item() / math.sqrt(n_sub)
        gap = e_soft.item() - sub_mean
        z = gap / sub_se if sub_se > 0 else 0.0
        flag = "MATCH" if abs(z) < 4 else "MISMATCH"
        print(
            f"{tag:22s}  soft={e_soft.item():8.4f}  subset={sub_mean:8.4f}  "
            f"gap={gap:+8.4f}  se={sub_se:.4f}  z={z:+5.2f}  {flag}  "
            f"hard={e_hard.item():.4f}",
            flush=True,
        )

    n_bad = sum(f == "MISMATCH" for f in flags)
    print(f"SAME_TREE_MISMATCHES={n_bad}/{len(flags)}", flush=True)


if __name__ == "__main__":
    main()
