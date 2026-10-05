"""Sanity checks for path-Top-K Tree-EAL.

A. Analytic E[L] vs verify_tree_step simple-mode Monte Carlo.
B. Alphas on private nodes of non-selected paths do not change E[L].
C. Gradient of E[L_tree] w.r.t. draft logits.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from small_lm_tree_v2_model import (
    _build_tree_topology,
    induce_path_topk_union,
    tree_expected_accepted_length,
)


def leaf_paths(parents):
    """Same path order as verify_tree_step: increasing leaf index."""
    parents = [int(p) for p in parents]
    n = len(parents)
    children = [[] for _ in range(n)]
    for i, p in enumerate(parents):
        if p >= 0:
            children[p].append(i)
    paths = []
    for leaf in range(n):
        if children[leaf]:
            continue
        path = []
        j = leaf
        while j >= 0:
            path.append(j)
            j = parents[j]
        path.reverse()
        paths.append(path)
    return paths


def monte_carlo_mean_length(alpha, parents, n_trials, seed):
    """Independent Bernoulli coins, then max consecutive accepted prefix.

    Matches verify_tree_step simple mode. Ties do not change the length.
    """
    alpha = alpha.detach().reshape(-1).float()
    parents_row = parents.detach().reshape(-1).tolist()
    n = alpha.numel()
    g = torch.Generator().manual_seed(seed)
    coins = torch.rand(n_trials, n, generator=g) <= alpha.view(1, n)
    lengths = torch.zeros(n_trials)
    for path in leaf_paths(parents_row):
        alive = torch.ones(n_trials, dtype=torch.bool)
        plen = torch.zeros(n_trials)
        for node in path:
            alive = alive & coins[:, node]
            plen = plen + alive.float()
        lengths = torch.maximum(lengths, plen)
    return lengths.mean().item(), lengths.std(unbiased=True).item()


def check_monte_carlo():
    alpha = torch.tensor([[0.50, 0.40, 0.30, 0.20, 0.60, 0.10]])
    parents = torch.tensor([[-1, -1, 0, 0, 1, 1]])
    analytic = tree_expected_accepted_length(alpha, parents).item()
    assert abs(analytic - 1.11968) < 1e-5, analytic
    empirical, std = monte_carlo_mean_length(alpha, parents, 100_000, seed=0)
    se = std / (100_000 ** 0.5)
    gap = empirical - analytic
    print(f"[A depth2] analytic={analytic:.6f} mc={empirical:.6f} "
          f"gap={gap:.6f} se={se:.6f} z={gap / se:.2f}")
    assert abs(gap) < 4 * se

    full_parents, _depths, paths, _counts, _starts, n = _build_tree_topology(
        [4, 3, 2, 1, 1, 1, 1])
    paths_t = torch.tensor(paths, dtype=torch.long)
    scores = torch.arange(24, dtype=torch.float32).view(1, -1)
    union_parents, gather, sel = induce_path_topk_union(
        full_parents, paths_t, scores, 16)
    alpha_full = (((torch.arange(n) % 7) + 1).float() / 10.0).view(1, -1)
    alpha_u = torch.where(
        gather >= 0,
        alpha_full.gather(1, gather.clamp(min=0)),
        torch.zeros_like(gather, dtype=torch.float32),
    )
    analytic_k = tree_expected_accepted_length(alpha_u, union_parents).item()
    empirical_k, std_k = monte_carlo_mean_length(
        alpha_u, union_parents, 100_000, seed=1)
    se_k = std_k / (100_000 ** 0.5)
    gap_k = empirical_k - analytic_k
    print(f"[A top16] analytic={analytic_k:.6f} mc={empirical_k:.6f} "
          f"gap={gap_k:.6f} se={se_k:.6f} z={gap_k / se_k:.2f} "
          f"union={(gather >= 0).sum().item()} paths={sel.numel()}")
    assert abs(gap_k) < 4 * se_k


def check_isolation():
    full_parents, depths, paths, _counts, starts, n = _build_tree_topology(
        [4, 3, 2, 1, 1, 1, 1])
    paths_t = torch.tensor(paths, dtype=torch.long)
    scores = torch.arange(24, dtype=torch.float32).view(1, -1)
    union_parents, gather, sel = induce_path_topk_union(
        full_parents, paths_t, scores, 16)
    selected = set(sel[0].tolist())
    kept = set(int(x) for x in gather[0].tolist() if x >= 0)
    private = []
    for k, path in enumerate(paths):
        if k in selected:
            continue
        for node in path:
            if depths[node] >= 2:
                private.append(node)
                assert node not in kept
    alpha = (((torch.arange(n) % 7) + 1).float() / 10.0).view(1, -1)
    def eal(values):
        picked = torch.where(
            gather >= 0,
            values.gather(1, gather.clamp(min=0)),
            torch.zeros_like(gather, dtype=torch.float32),
        )
        return tree_expected_accepted_length(picked, union_parents).item()
    base = eal(alpha)
    flipped = alpha.clone()
    flipped[0, private] = 1.0 - flipped[0, private]
    changed = eal(flipped)
    print(f"[B] base={base:.8f} after flipping {len(private)} private alphas"
          f"={changed:.8f} absdiff={abs(base - changed):.3e}")
    assert base == changed


def check_gradient():
    torch.manual_seed(0)
    full_parents, _depths, paths, _counts, _starts, n = _build_tree_topology(
        [4, 3, 2, 1, 1, 1, 1])
    paths_t = torch.tensor(paths, dtype=torch.long)
    vocab = 16
    draft_logits = torch.randn(n, vocab, requires_grad=True)
    target_logits = torch.randn(n, vocab)  # frozen target
    tokens = draft_logits.detach().argmax(dim=-1)
    alpha = torch.softmax(target_logits, dim=-1).gather(
        -1, tokens.unsqueeze(-1)).squeeze(-1).view(1, -1)
    path_scores = F.log_softmax(draft_logits, dim=-1).gather(
        -1, tokens.unsqueeze(-1)).squeeze(-1)
    # One score per complete path: sum of node log-probs. Detach, as in training.
    path_score = torch.stack([
        path_scores.index_select(0, paths_t[k]).sum() for k in range(24)
    ]).view(1, 24)
    union_parents, gather, _sel = induce_path_topk_union(
        full_parents, paths_t, path_score.detach(), 16)
    alpha_u = torch.where(
        gather >= 0,
        alpha.gather(1, gather.clamp(min=0)),
        torch.zeros_like(gather, dtype=torch.float32),
    )
    eal = tree_expected_accepted_length(alpha_u, union_parents)
    print(f"[C] E[L].requires_grad={eal.requires_grad} grad_fn={eal.grad_fn}")
    assert eal.requires_grad is False
    assert draft_logits.grad is None

    # The recursion itself propagates grad into alpha. The cut is upstream.
    alpha_live = alpha.detach().clone().requires_grad_(True)
    alpha_live_u = torch.where(
        gather >= 0,
        alpha_live.gather(1, gather.clamp(min=0)),
        torch.zeros_like(gather, dtype=torch.float32),
    )
    eal_live = tree_expected_accepted_length(alpha_live_u, union_parents)
    eal_live.backward()
    grad_norm = alpha_live.grad.norm().item()
    print(f"[C] dE[L]/d alpha norm={grad_norm:.6f} "
          f"(recursion is differentiable in alpha)")
    assert grad_norm > 0

    def kl_and_eal(logits):
        log_q = F.log_softmax(target_logits, dim=-1)
        q = log_q.exp()
        log_p = F.log_softmax(logits, dim=-1)
        kl = (q * (log_q - log_p)).sum(-1).mean()
        tok = logits.detach().argmax(dim=-1)
        a = torch.softmax(target_logits, dim=-1).gather(
            -1, tok.unsqueeze(-1)).squeeze(-1).view(1, -1)
        ps = F.log_softmax(logits, dim=-1).gather(
            -1, tok.unsqueeze(-1)).squeeze(-1).detach()
        pscore = torch.stack([
            ps.index_select(0, paths_t[k]).sum() for k in range(24)
        ]).view(1, 24)
        up, gidx, _ = induce_path_topk_union(full_parents, paths_t, pscore, 16)
        au = torch.where(
            gidx >= 0,
            a.gather(1, gidx.clamp(min=0)),
            torch.zeros_like(gidx, dtype=torch.float32),
        )
        length = tree_expected_accepted_length(au, up)
        return kl, length

    base = draft_logits.detach().clone()
    d1 = base.clone().requires_grad_(True)
    kl1, eal1 = kl_and_eal(d1)
    (kl1 - eal1).backward()
    d2 = base.clone().requires_grad_(True)
    kl2, _eal2 = kl_and_eal(d2)
    kl2.backward()
    same = torch.allclose(d1.grad, d2.grad)
    print(f"[C] KL-E[L] grad equals KL-only grad: {same} "
          f"max_abs_diff={(d1.grad - d2.grad).abs().max().item():.3e} "
          f"eal.requires_grad={eal1.requires_grad}")
    assert same


def check_node_marginal_loss():
    from small_lm_tree_v2_model import node_marginal_eal_loss
    # Chain: E = a1 + a1*a2, dE/da1 = 1+a2, dE/da2 = a1.
    alpha = torch.tensor([[0.5, 0.4]])
    parents = torch.tensor([[-1, 0]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    logits = torch.tensor([[2.0, 0.0], [0.0, 1.0]], requires_grad=True)
    tokens = torch.tensor([0, 1])
    logp = F.log_softmax(logits, dim=-1)[torch.arange(2), tokens].view(1, 2)
    loss, eal = node_marginal_eal_loss(alpha, parents, logp, valid)
    assert abs(eal.item() - 0.7) < 1e-5, eal.item()
    alpha_live = alpha.clone().requires_grad_(True)
    ref = tree_expected_accepted_length(alpha_live, parents)
    ref.backward()
    w = alpha_live.grad
    assert torch.allclose(w, torch.tensor([[1.4, 0.5]]), atol=1e-5), w
    manual = -(w[0, 0] * logp[0, 0] + w[0, 1] * logp[0, 1])
    assert torch.allclose(loss, manual), (loss, manual)
    loss.backward()
    assert logits.grad.abs().sum().item() > 0
    # Two leaves: dE/daA = 1-aB. Redundant sibling gets a smaller weight.
    a2 = torch.tensor([[0.5, 0.3]], requires_grad=True)
    p2 = torch.tensor([[-1, -1]])
    tree_expected_accepted_length(a2, p2).backward()
    assert torch.allclose(a2.grad, torch.tensor([[0.7, 0.5]]), atol=1e-5), a2.grad
    # Same value under the eval no_grad wrapper. Pad columns stay out of the sum.
    with torch.no_grad():
        loss_ng, eal_ng = node_marginal_eal_loss(
            alpha, parents, logp.detach(), valid)
    assert abs(eal_ng.item() - 0.7) < 1e-5
    assert abs(loss_ng.item() - manual.detach().item()) < 1e-5
    alpha_pad = torch.tensor([[0.5, 0.4, 0.0]])
    parents_pad = torch.tensor([[-1, 0, -1]])
    valid_pad = torch.tensor([[True, True, False]])
    logits_pad = logits.detach().clone().requires_grad_(True)
    logp_pad = F.log_softmax(logits_pad, dim=-1)[torch.arange(2), tokens]
    logp_pad = torch.cat([logp_pad, logp_pad[:1]], dim=0).view(1, 3)
    loss_pad, eal_pad = node_marginal_eal_loss(
        alpha_pad, parents_pad, logp_pad, valid_pad)
    assert abs(eal_pad.item() - 0.7) < 1e-5, eal_pad.item()
    assert torch.allclose(loss_pad, manual.detach()), (loss_pad, manual)
    print(f"[D] chain E[L]={eal.item():.4f} weights={w.view(-1).tolist()} "
          f"draft_grad_norm={logits.grad.norm().item():.4f}")


def main():
    check_node_marginal_loss()
    check_monte_carlo()
    check_isolation()
    check_gradient()
    print("ALL SANITY CHECKS PASSED")


if __name__ == "__main__":
    main()
