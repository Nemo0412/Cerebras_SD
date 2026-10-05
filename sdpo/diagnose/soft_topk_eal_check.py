"""Check soft Top-K Tree-EAL against the hard union and Bernoulli samples."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from small_lm_tree_v2_model import (
    _build_tree_topology,
    soft_topk_tree_eal,
    tree_expected_accepted_length,
)

TOP_K = (4, 3, 2, 1, 1, 1, 1)
K = 16
TAU = 1.0


def bernoulli_lengths(alpha, paths, n_mc, seed):
    g = torch.Generator().manual_seed(seed)
    coins = torch.rand(n_mc, alpha.numel(), generator=g) < alpha.view(1, -1)
    prefix = coins[:, paths].cumprod(dim=-1)
    return prefix.sum(dim=-1).max(dim=-1).values.float()


def main():
    parents, depths, paths, counts, starts, n_nodes = _build_tree_topology(TOP_K)
    paths_t = torch.tensor(paths, dtype=torch.long)
    parent_t = torch.tensor(parents, dtype=torch.long)
    assert n_nodes == 136 and paths_t.shape == (24, 7)

    # Closed form from the depth-2 verify example: E[L] = 1.11968.
    a2 = torch.tensor([[0.50, 0.40, 0.30, 0.20, 0.60, 0.10]])
    p2 = torch.tensor([[-1, -1, 0, 0, 1, 1]])
    e2 = tree_expected_accepted_length(a2, p2).item()
    assert abs(e2 - 1.11968) < 1e-4, e2

    g = torch.Generator().manual_seed(0)
    alpha = torch.rand(1, n_nodes, generator=g).clamp(0.05, 0.95)
    # Peaked scores, same shape as a trained draft's path scores.
    scores = torch.linspace(-2.5, -14.0, 24).view(1, 24)
    scores = scores + 0.05 * torch.randn(1, 24, generator=g)
    scores = scores.detach().requires_grad_(True)

    loss, e_soft, e_hard = soft_topk_tree_eal(
        alpha, parent_t, paths_t, scores, K, tau=TAU)
    print(f"soft={e_soft.item():.4f} hard={e_hard.item():.4f}")

    # Hard 0/1 inclusion must reproduce the discrete union.
    sel = scores.detach().topk(K, dim=-1).indices[0]
    chosen = paths_t[sel].reshape(-1).unique()
    beta = torch.zeros(1, n_nodes)
    beta[0, chosen] = 1.0
    e_beta = tree_expected_accepted_length(beta * alpha, parent_t.view(1, -1)).item()
    assert abs(e_beta - e_hard.item()) < 1e-5, (e_beta, e_hard.item())

    # Actual Bernoulli length on that fixed union.
    alpha_u = alpha[0].clone()
    alpha_u[beta[0] == 0] = 0.0
    # Paths that are not entirely inside the union cannot be the surviving path
    # beyond their included prefix; zeroing alpha outside the union does that.
    mc = bernoulli_lengths(alpha_u, paths_t, 100_000, seed=1)
    mc_mean = mc.mean().item()
    mc_se = mc.std(unbiased=True).item() / (100_000 ** 0.5)
    gap = e_hard.item() - mc_mean
    z = gap / mc_se
    print(f"hard={e_hard.item():.4f} bernoulli={mc_mean:.4f} "
          f"gap={gap:.4f} se={mc_se:.4f} z={z:.2f}")
    assert abs(z) < 4.0, z

    # Actual expected E[L] of K-with-replacement path draws vs the soft plug-in.
    pi = torch.softmax(scores.detach() / TAU, dim=-1)[0]
    n_sub = 4000
    draws = torch.multinomial(
        pi.expand(n_sub, -1), num_samples=K, replacement=True, generator=g)
    drawn_nodes = paths_t[draws].reshape(n_sub, -1)
    mask = torch.zeros(n_sub, n_nodes)
    mask.scatter_(1, drawn_nodes, 1.0)
    alpha_s = alpha.expand(n_sub, -1) * mask
    e_sub = tree_expected_accepted_length(alpha_s, parent_t.view(1, -1).expand(n_sub, -1))
    sub_mean = e_sub.mean().item()
    sub_se = e_sub.std(unbiased=True).item() / (n_sub ** 0.5)
    soft_gap = e_soft.item() - sub_mean
    soft_z = soft_gap / sub_se
    print(f"soft={e_soft.item():.4f} subset_mc={sub_mean:.4f} "
          f"gap={soft_gap:.4f} se={sub_se:.4f} z={soft_z:.2f}")

    # Gradient must reach path scores, and a higher-alpha path must be preferred.
    loss.backward()
    grad = scores.grad[0]
    assert grad.abs().sum().item() > 0, grad
    print(f"score_grad_norm={grad.norm().item():.4f} "
          f"best_path_grad={grad[int(scores.detach().argmax())].item():.4f}")

    # Two paths, one high-alpha and one low-alpha, equal scores.
    # Raising the high-alpha path score must increase E_soft.
    tiny_alpha = torch.tensor([[0.9, 0.8, 0.1, 0.1]])
    tiny_parents = torch.tensor([-1, 0, -1, 2])
    tiny_paths = torch.tensor([[0, 1], [2, 3]])
    tiny_scores = torch.zeros(1, 2, requires_grad=True)
    loss_t, e_t, _ = soft_topk_tree_eal(
        tiny_alpha, tiny_parents, tiny_paths, tiny_scores, path_topk=1, tau=1.0)
    loss_t.backward()
    # path 0 is the high-alpha chain. d(loss)/ds_0 = -dE/ds_0 should be negative.
    print(f"tiny E={e_t.item():.4f} dL/ds={tiny_scores.grad.view(-1).tolist()}")
    assert tiny_scores.grad[0, 0].item() < 0, tiny_scores.grad
    assert tiny_scores.grad[0, 1].item() > 0, tiny_scores.grad

    ok_soft = abs(soft_z) < 4.0
    print("SOFT_MATCH" if ok_soft else "SOFT_MISMATCH")
    print("HARD_MATCH")


if __name__ == "__main__":
    main()
