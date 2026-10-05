"""
Offline analysis: S_estimate-based adaptive tree pruning.

For each position, simulates a chain draft (teacher-forced, γ steps) and records:
- max_prob at each step (draft model's confidence = acceptance estimate)
- actual acceptance (argmax match with target)

Then analyzes: at different S_estimate thresholds, how many steps would survive
pruning, and how does τ change?

Key insight: S_estimate(depth d) = Π_{i≤d} max_prob_i estimates the probability
that the entire chain up to depth d is accepted. If S_estimate drops below a
threshold, the remaining depths are unlikely to contribute to τ.

Usage:
    CUDA_VISIBLE_DEVICES=0 python evaluation/eval_tree_pruning.py \
        --basepath meta-llama/Llama-3.1-8B-Instruct \
        --draftpath /path/to/EAGLE3 \
        --testpath sdpo/data/test.jsonl \
        --gamma 7
"""

import argparse
import json
import os
import shutil
import sys

import torch
import torch.nn.functional as F
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(script_dir, '..')
sys.path.insert(0, project_dir)
sys.path.insert(0, os.path.join(project_dir, 'sdpo'))
sys.path.insert(0, os.path.join(project_dir, 'traineagle3'))

from huggingface_hub import snapshot_download
from traineagle3.configs import EConfig
from traineagle3.cnets import Model
from transformers import AutoTokenizer
from types import SimpleNamespace

from eval_test_loss import load_samples


@torch.no_grad()
def eval_one_sample_detailed(model, sample, gamma, device, full2draft, t2d, d2t):
    """Like eval_one_sample but returns per-position, per-step max_prob and accept."""
    input_ids = sample['input_ids'].unsqueeze(0).to(device)
    attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
    loss_mask = sample['loss_mask'].unsqueeze(0).to(device)

    hidden_states, target, loss_mask_3d, input_ids_shifted = model.dataprepare(
        input_ids, attention_mask, loss_mask)
    loss_mask_1d = loss_mask_3d.squeeze(-1).squeeze(0).bool()
    B, L, _ = hidden_states.shape

    hs_projected = model.fc(hidden_states.to(model.fc.weight.dtype))
    attn_mask = model._prepare_decoder_attention_mask(
        attention_mask, (B, L), hs_projected, 0)
    position_ids = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0)
    target_greedy = target.argmax(dim=-1)

    all_max_prob = []  # max probability at each step (draft confidence)
    all_accept = []  # actual acceptance
    all_q_target = []  # q(y*) for S_θ computation
    cache_hidden = [[], []]
    cur_ids = input_ids_shifted
    cur_hs = hs_projected
    cur_tgt = target_greedy.clone()

    for idx in range(gamma):
        embeds = model.embed_tokens(cur_ids).to(cur_hs.dtype)
        layer_out, cache_hidden = model.midlayer(
            input_emb=embeds, hidden_states=cur_hs, cache_hidden=cache_hidden,
            attention_mask=attn_mask, position_ids=position_ids,
            past_key_value=None, output_attentions=False, use_cache=True)
        cur_hs = layer_out[0]
        logits = model.lm_head(model.norm(cur_hs)).float()

        probs = F.softmax(logits, dim=-1)
        max_prob = probs.max(dim=-1).values.squeeze(0)  # [L]

        draft_d = logits.argmax(dim=-1)
        draft_full = draft_d + d2t[draft_d]
        tgt_in = t2d[cur_tgt]
        accept = (draft_full == cur_tgt) & tgt_in

        tgt_d = full2draft[cur_tgt].clamp(min=0)
        q_target = probs.gather(-1, tgt_d.unsqueeze(-1)).squeeze(-1).squeeze(0)
        q_target = q_target * tgt_in.float().squeeze(0)

        all_max_prob.append(max_prob)
        all_accept.append(accept.squeeze(0))
        all_q_target.append(q_target)

        if idx < gamma - 1:
            cur_ids = cur_tgt
            cur_tgt = torch.cat([cur_tgt[:, 1:], torch.zeros_like(cur_tgt[:, :1])], dim=1)

    max_prob_all = torch.stack(all_max_prob, dim=1)  # [L, gamma]
    accept_all = torch.stack(all_accept, dim=1)  # [L, gamma]
    q_target_all = torch.stack(all_q_target, dim=1)  # [L, gamma]

    return {
        'max_prob': max_prob_all[loss_mask_1d].cpu().numpy(),  # [N, gamma]
        'accept': accept_all[loss_mask_1d].cpu().numpy(),  # [N, gamma]
        'q_target': q_target_all[loss_mask_1d].cpu().numpy(),  # [N, gamma]
        'n_positions': loss_mask_1d.sum().item(),
    }


def analyze_pruning(all_max_prob, all_accept, gamma, thresholds):
    """Simulate pruning at different S_estimate thresholds."""
    N = all_max_prob.shape[0]

    # Compute S_estimate = cumulative product of max_prob
    S_estimate = np.cumprod(all_max_prob, axis=1)  # [N, gamma]

    # Actual tau (no pruning)
    valid_chain = np.cumprod(all_accept.astype(float), axis=1)
    tau_actual = valid_chain.sum(axis=1)  # [N]

    results = []
    for thresh in thresholds:
        # For each position, find the last depth where S_estimate >= threshold
        # All deeper depths would be pruned
        survives = S_estimate >= thresh  # [N, gamma]
        # Pruned tau: only count accepted steps that survive pruning
        pruned_valid = valid_chain * survives
        tau_pruned = pruned_valid.sum(axis=1)

        # Average depths kept
        depths_kept = survives.sum(axis=1)  # [N]

        results.append({
            'threshold': thresh,
            'mean_tau_original': tau_actual.mean(),
            'mean_tau_pruned': tau_pruned.mean(),
            'tau_retention': tau_pruned.mean() / (tau_actual.mean() + 1e-8),
            'mean_depths_kept': depths_kept.mean(),
            'depth_reduction': 1.0 - depths_kept.mean() / gamma,
            'compute_saved': 1.0 - depths_kept.mean() / gamma,
        })

    return results


def analyze_s_estimate_accuracy(all_max_prob, all_accept, gamma):
    """How well does S_estimate predict actual acceptance at each depth?"""
    S_estimate = np.cumprod(all_max_prob, axis=1)
    actual_accept_at_depth = np.cumprod(all_accept.astype(float), axis=1)

    print(f"\n  S_estimate vs actual acceptance (per depth):")
    print(f"  {'depth':<8} {'mean_S_est':<12} {'mean_actual':<12} {'ratio':<10} {'corr':<10}")
    print(f"  {'-'*52}")
    for d in range(gamma):
        s_est = S_estimate[:, d]
        actual = actual_accept_at_depth[:, d]
        ratio = s_est.mean() / (actual.mean() + 1e-8)
        corr = np.corrcoef(s_est, actual)[0, 1] if s_est.std() > 0 and actual.std() > 0 else 0
        print(f"  {d:<8} {s_est.mean():<12.4f} {actual.mean():<12.4f} {ratio:<10.2f} {corr:<10.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--basepath', required=True)
    parser.add_argument('--draftpath', required=True)
    parser.add_argument('--testpath', required=True)
    parser.add_argument('--gamma', type=int, default=7)
    parser.add_argument('--max_len', type=int, default=2048)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--config_path', type=str,
                        default=os.path.join(project_dir, 'sdpo', 'config.json'))
    args = parser.parse_args()

    device = torch.device('cuda:0')
    if not os.path.isdir(args.basepath):
        args.basepath = snapshot_download(args.basepath)
    if not os.path.isdir(args.draftpath):
        args.draftpath = snapshot_download(args.draftpath)

    tokenizer = AutoTokenizer.from_pretrained(args.basepath)
    draft_cfg = os.path.join(args.draftpath, "config.json")
    config = EConfig.from_pretrained(draft_cfg if os.path.exists(draft_cfg) else args.config_path)
    config.gradient_checkpointing = False

    train_ns = SimpleNamespace(
        bs=1, num_epochs=1, num_workers=0, max_len=args.max_len,
        config_path=args.config_path, gradient_checkpointing=False,
        eagle_coef=0, kl_coef=0, gamma=args.gamma, baseline=None,
    )
    ds_config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {"stage": 2},
    }

    model = Model(config, ds_config, train_ns, path=args.basepath,
                  load_emb=True, load_head=True)
    sf_path = os.path.join(args.draftpath, "model.safetensors")
    bin_path = os.path.join(args.draftpath, "pytorch_model.bin")
    if os.path.exists(sf_path):
        from safetensors.torch import load_file as sf_load
        state = sf_load(sf_path, device="cpu")
    elif os.path.exists(bin_path):
        state = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No weights in {args.draftpath}")
    for vk in ("d2t", "t2d"):
        state.pop(vk, None)
    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)

    draft_cache = os.path.join(args.draftpath, "cache.pt")
    if os.path.exists(draft_cache):
        shutil.copy(draft_cache, "cache.pt")
    model.scandata(args.testpath, args.basepath)
    model = model.to(device)
    model.eval()
    model.length = args.gamma

    d2t = model.d2t.to(device)
    t2d = model.t2d.to(device)
    draft_ids = torch.arange(len(d2t), device=device)
    full_ids = draft_ids + d2t
    full2draft = torch.full((t2d.shape[0],), -1, dtype=torch.long, device=device)
    full2draft[full_ids] = draft_ids

    dataset = load_samples(tokenizer, args.testpath, args.max_len, args.max_samples)
    print(f"Loaded {len(dataset)} test samples")

    # Collect per-position data
    all_max_prob = []
    all_accept = []
    total_positions = 0

    for i, sample in enumerate(dataset):
        if i % 20 == 0:
            print(f"  {i}/{len(dataset)}...")
        r = eval_one_sample_detailed(model, sample, args.gamma, device, full2draft, t2d, d2t)
        all_max_prob.append(r['max_prob'])
        all_accept.append(r['accept'])
        total_positions += r['n_positions']

    all_max_prob = np.concatenate(all_max_prob, axis=0)  # [total_N, gamma]
    all_accept = np.concatenate(all_accept, axis=0)  # [total_N, gamma]
    print(f"\nTotal positions: {total_positions}")

    # 1. S_estimate accuracy analysis
    print(f"\n{'='*60}")
    print("Part 1: S_estimate accuracy (does it predict acceptance?)")
    print(f"{'='*60}")
    analyze_s_estimate_accuracy(all_max_prob, all_accept, args.gamma)

    # 2. Pruning simulation
    print(f"\n{'='*60}")
    print("Part 2: Pruning simulation (threshold sweep)")
    print(f"{'='*60}")
    thresholds = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    pruning_results = analyze_pruning(all_max_prob, all_accept, args.gamma, thresholds)

    print(f"\n  {'threshold':<12} {'τ_original':<12} {'τ_pruned':<12} {'τ_retain%':<12} "
          f"{'avg_depth':<12} {'compute_saved%':<14}")
    print(f"  {'-'*74}")
    for r in pruning_results:
        print(f"  {r['threshold']:<12.2f} {r['mean_tau_original']:<12.4f} "
              f"{r['mean_tau_pruned']:<12.4f} {r['tau_retention']*100:<12.1f} "
              f"{r['mean_depths_kept']:<12.2f} {r['compute_saved']*100:<14.1f}")

    # 3. Sweet spot analysis
    print(f"\n{'='*60}")
    print("Part 3: Sweet spot (best τ_retained / compute_saved tradeoff)")
    print(f"{'='*60}")
    # Find threshold where τ retention > 95% with maximum compute saving
    sweet = None
    for r in pruning_results:
        if r['tau_retention'] >= 0.95:
            if sweet is None or r['compute_saved'] > sweet['compute_saved']:
                sweet = r
    if sweet:
        print(f"  Best 95%+ retention: threshold={sweet['threshold']:.2f}, "
              f"τ_retain={sweet['tau_retention']*100:.1f}%, "
              f"compute_saved={sweet['compute_saved']*100:.1f}%, "
              f"avg_depth={sweet['mean_depths_kept']:.2f}/{args.gamma}")
    else:
        print(f"  No threshold achieves 95% τ retention — acceptance too fragile for pruning")

    # 4. Per-step max_prob distribution
    print(f"\n{'='*60}")
    print("Part 4: Per-step max_prob distribution")
    print(f"{'='*60}")
    print(f"  {'step':<8} {'mean':<10} {'p25':<10} {'p50':<10} {'p75':<10} {'p90':<10}")
    print(f"  {'-'*58}")
    for d in range(args.gamma):
        mp = all_max_prob[:, d]
        print(f"  {d:<8} {mp.mean():<10.4f} {np.percentile(mp, 25):<10.4f} "
              f"{np.percentile(mp, 50):<10.4f} {np.percentile(mp, 75):<10.4f} "
              f"{np.percentile(mp, 90):<10.4f}")

    # Save
    out_path = os.path.join(script_dir, 'tree_pruning_analysis.json')
    with open(out_path, 'w') as f:
        json.dump({
            'n_positions': total_positions,
            'n_samples': len(dataset),
            'gamma': args.gamma,
            'pruning_results': pruning_results,
        }, f, indent=2, default=float)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
