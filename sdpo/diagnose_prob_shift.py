"""Compare baseline (untrained) draft against 3 trained drafts on:
  P(target_argmax | draft)  —  draft's probability on the "right" token
  max P(· | draft)           —  draft's top-1 probability

Split positions by whether the BASELINE draft was originally correct
(argmax matched target) or wrong, then report how the 3 trained drafts
shift those two probabilities vs baseline.

Usage:
    python sdpo/diagnose_prob_shift.py --num-samples 30
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_draft(path):
    for f in glob.glob(os.path.join(path, "*.bin")):
        state = torch.load(f, map_location="cpu")
        changed = False
        for prefix in ("draft.base.", "draft_model."):
            if any(k.startswith(prefix) for k in state):
                state = {k.removeprefix(prefix): v for k, v in state.items()}
                changed = True
        if changed:
            torch.save(state, f)
    m = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16,
        attn_implementation="sdpa").cuda()
    m.eval()
    return m


@torch.no_grad()
def forward_probs(model, input_ids):
    """Return softmax probs at each position. Shape [L-1, V]."""
    out = model(input_ids=input_ids, return_dict=True).logits[:, :-1, :].float()
    return F.softmax(out, dim=-1).squeeze(0)   # [L-1, V]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target', default='Qwen/Qwen3-8B')
    p.add_argument('--base-draft', default='Qwen/Qwen3-0.6B')
    p.add_argument('--drafts', nargs='+', default=[
        ('kl',    '/scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/q8_q06_kl_regen/state_2'),
        ('klv4',  '/scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/q8_q06_klv4_regen/state_2'),
        ('kleal', '/scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/q8_q06_kleal_pow_pow_regen_2k/state_2'),
    ], help='Format: name=path pairs, or just paths with auto names.')
    p.add_argument('--data-path',
                   default='/scratch/yf3005/gto_data/sharegpt_qwen3_8b_regen.jsonl')
    p.add_argument('--num-samples', type=int, default=30)
    p.add_argument('--max-len', type=int, default=1024)
    args = p.parse_args()

    # Parse --drafts if provided as strings
    parsed = []
    for item in args.drafts:
        if isinstance(item, tuple):
            parsed.append(item)
        elif '=' in item:
            parts = item.split('=', 1)
            parsed.append((parts[0], parts[1]))
        else:
            parsed.append((os.path.basename(item.rstrip('/')), item))
    draft_specs = parsed

    print(f"Loading target: {args.target}")
    target = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.float16,
        attn_implementation="sdpa").cuda()
    target.eval()
    tok = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)

    print(f"Loading base draft: {args.base_draft}")
    base_draft = AutoModelForCausalLM.from_pretrained(
        args.base_draft, torch_dtype=torch.float16,
        attn_implementation="sdpa").cuda()
    base_draft.eval()

    trained = {}
    for name, path in draft_specs:
        print(f"Loading trained draft [{name}]: {path}")
        trained[name] = load_draft(path)

    print(f"\nLoading {args.num_samples} samples from {args.data_path}")
    ds = load_dataset('json', data_files=args.data_path)['train']
    ds = ds.shuffle(seed=42).select(range(args.num_samples * 3))

    # Accumulators:
    # correct_mass[name] = list of P(target_argmax) per-position across all samples
    # correct_top1[name] = list of max P per-position
    # same for wrong
    # *_correct[name] = bool list, whether trained model's argmax matches target at that position
    data = {
        'correct_mass_base': [],
        'correct_top1_base': [],
        'wrong_mass_base': [],
        'wrong_top1_base': [],
    }
    for name in trained:
        data[f'correct_mass_{name}'] = []
        data[f'correct_top1_{name}'] = []
        data[f'correct_correct_{name}'] = []   # argmax==target at base-correct positions
        data[f'wrong_mass_{name}'] = []
        data[f'wrong_top1_{name}'] = []
        data[f'wrong_correct_{name}'] = []     # argmax==target at base-wrong positions (= flip)

    num_used = 0
    for ex in ds:
        src = ex.get('conversations', [])
        if not src or src[0].get('from') != 'human':
            continue
        msgs = []
        for s in src[:4]:
            role = 'user' if s['from'] == 'human' else 'assistant'
            msgs.append({'role': role, 'content': s['value']})
        try:
            text = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=False,
                                           enable_thinking=True)
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=False)
        ids = tok(text, return_tensors="pt",
                  add_special_tokens=False).input_ids
        if ids.shape[1] < 20 or ids.shape[1] > args.max_len:
            continue
        ids = ids.cuda()

        with torch.no_grad():
            tgt_lg = target(input_ids=ids, return_dict=True).logits[:, :-1, :].float()
            tgt_argmax = tgt_lg.argmax(-1).squeeze(0)        # [L-1]

        base_probs = forward_probs(base_draft, ids)           # [L-1, V]
        base_argmax = base_probs.argmax(-1)
        base_correct = (base_argmax == tgt_argmax)            # [L-1]

        # Prob on target token (baseline)
        base_mass = base_probs.gather(-1, tgt_argmax.unsqueeze(-1)).squeeze(-1)
        base_top1 = base_probs.max(-1).values

        trained_stats = {}
        for name, model in trained.items():
            probs = forward_probs(model, ids)
            mass = probs.gather(-1, tgt_argmax.unsqueeze(-1)).squeeze(-1)
            top1 = probs.max(-1).values
            argmax_match = (probs.argmax(-1) == tgt_argmax)
            trained_stats[name] = (mass, top1, argmax_match)

        # Split by baseline correctness
        for mask_name, mask in [('correct', base_correct),
                                 ('wrong', ~base_correct)]:
            if mask.any():
                data[f'{mask_name}_mass_base'].extend(base_mass[mask].cpu().tolist())
                data[f'{mask_name}_top1_base'].extend(base_top1[mask].cpu().tolist())
                for name, (mass, top1, am) in trained_stats.items():
                    data[f'{mask_name}_mass_{name}'].extend(mass[mask].cpu().tolist())
                    data[f'{mask_name}_top1_{name}'].extend(top1[mask].cpu().tolist())
                    data[f'{mask_name}_correct_{name}'].extend(am[mask].cpu().tolist())

        num_used += 1
        if num_used >= args.num_samples:
            break

    # Report
    print(f"\n{'='*85}")
    print(f"Analyzed {num_used} samples")
    n_correct = len(data['correct_mass_base'])
    n_wrong = len(data['wrong_mass_base'])
    print(f"Baseline correct positions: {n_correct}  |  wrong positions: {n_wrong}")
    print(f"Baseline accuracy: {n_correct / (n_correct + n_wrong):.3%}")

    names = ['base'] + list(trained.keys())
    for split, n_pos in [('correct', n_correct), ('wrong', n_wrong)]:
        print(f"\n{'─'*85}")
        print(f"── Baseline {split.upper()} positions (N={n_pos}) ──")
        print(f"{'draft':<10} | {'P(target) mean':<16} | {'P(target) median':<18} | "
              f"{'top1 mean':<12} | {'top1 median':<12}")
        print('-' * 85)
        for name in names:
            mass = np.array(data[f'{split}_mass_{name}'])
            top1 = np.array(data[f'{split}_top1_{name}'])
            if len(mass) == 0: continue
            print(f"{name:<10} | {mass.mean():<16.4f} | {np.median(mass):<18.4f} | "
                  f"{top1.mean():<12.4f} | {np.median(top1):<12.4f}")

        # Δ vs baseline
        print(f"\n  Δ vs base on the same {split} positions:")
        print(f"  {'draft':<10} | {'Δ P(target)':<14} | {'Δ top1':<14} | "
              f"{'% pos Δmass>0':<14}")
        print('  ' + '-' * 60)
        base_mass = np.array(data[f'{split}_mass_base'])
        base_top1 = np.array(data[f'{split}_top1_base'])
        for name in trained:
            tm = np.array(data[f'{split}_mass_{name}'])
            tt = np.array(data[f'{split}_top1_{name}'])
            dm = tm - base_mass
            dt = tt - base_top1
            up = (dm > 0).mean()
            print(f"  {name:<10} | {dm.mean():+14.4f} | {dt.mean():+14.4f} | {up:14.2%}")

    # ─────────────────────────────────────────────────────────────────────────
    # New diagnostics
    # ─────────────────────────────────────────────────────────────────────────

    # Diag 1: Argmax flip rate on base-WRONG positions.
    # This is the metric that maps directly to acceptance — only flips matter.
    print(f"\n{'='*85}")
    print(f"── DIAG 1: Argmax flip rate on base-WRONG positions (N={n_wrong}) ──")
    print("  (fraction of wrong positions where trained draft's argmax == target)")
    print(f"  {'draft':<10} | {'flip rate':<10} | {'#flipped':<10}")
    print('  ' + '-' * 40)
    for name in trained:
        flips = np.array(data[f'wrong_correct_{name}'])
        rate = flips.mean()
        print(f"  {name:<10} | {rate:>9.2%} | {int(flips.sum()):>10d}")

    # Also: argmax stay-correct rate on base-CORRECT positions (regression check)
    print(f"\n── DIAG 1b: Argmax stay-correct on base-CORRECT positions (N={n_correct}) ──")
    print("  (fraction of correct positions trained draft DIDN'T break)")
    print(f"  {'draft':<10} | {'stay rate':<10} | {'#broken':<10}")
    print('  ' + '-' * 40)
    for name in trained:
        stay = np.array(data[f'correct_correct_{name}'])
        rate = stay.mean()
        broken = int((1 - stay).sum())
        print(f"  {name:<10} | {rate:>9.2%} | {broken:>10d}")

    # Diag 2: Difficulty-bucket flip rate on base-WRONG positions.
    # Bucket by base P(target): how easy was the position to fix?
    print(f"\n{'─'*85}")
    print("── DIAG 2: Flip rate by base-difficulty bucket (base wrong positions) ──")
    base_wrong_mass = np.array(data['wrong_mass_base'])
    buckets = [
        ('hard   [0,0.05)',   (base_wrong_mass <  0.05)),
        ('hard-m [0.05,0.15)', (base_wrong_mass >= 0.05) & (base_wrong_mass < 0.15)),
        ('medium [0.15,0.30)', (base_wrong_mass >= 0.15) & (base_wrong_mass < 0.30)),
        ('easy   [0.30,0.50)', (base_wrong_mass >= 0.30) & (base_wrong_mass < 0.50)),
        ('v_easy [0.50, ..)',  (base_wrong_mass >= 0.50)),
    ]
    header = f"  {'bucket':<22} | {'N':>5} |"
    for name in trained:
        header += f" {name:>8} |"
    print(header)
    print('  ' + '-' * (22 + 8 + len(trained) * 11))
    for bname, bmask in buckets:
        n_b = int(bmask.sum())
        if n_b == 0:
            continue
        row = f"  {bname:<22} | {n_b:>5d} |"
        for name in trained:
            flips = np.array(data[f'wrong_correct_{name}'])[bmask]
            rate = flips.mean() if len(flips) > 0 else 0.0
            row += f" {rate:>7.2%} |"
        print(row)

    # Diag 3: ΔP(target) distribution on base-WRONG positions (KLV4 vs KL focus).
    # If V4 helps mostly via long-tail (a few big improvements), median ≈ KL but q90 > KL.
    # If V4 helps via uniform shift, all quantiles shift up.
    print(f"\n{'─'*85}")
    print("── DIAG 3: ΔP(target) quantiles on base-WRONG positions ──")
    print("  (each model vs base; pairwise vs first trained draft also shown)")
    print(f"  {'draft':<10} | {'q10':>7} {'q25':>7} {'q50':>7} {'q75':>7} {'q90':>7} | {'mean':>7}")
    print('  ' + '-' * 70)
    base_mass = np.array(data['wrong_mass_base'])
    deltas_vs_base = {}
    for name in trained:
        tm = np.array(data[f'wrong_mass_{name}'])
        d = tm - base_mass
        deltas_vs_base[name] = d
        qs = np.percentile(d, [10, 25, 50, 75, 90])
        print(f"  {name:<10} | {qs[0]:+7.4f} {qs[1]:+7.4f} {qs[2]:+7.4f} "
              f"{qs[3]:+7.4f} {qs[4]:+7.4f} | {d.mean():+7.4f}")

    # Pairwise: compare every other trained draft against the first one (treat as anchor).
    if len(trained) >= 2:
        anchor = list(trained.keys())[0]
        anchor_mass = np.array(data[f'wrong_mass_{anchor}'])
        print(f"\n  Pairwise ΔP(target) vs '{anchor}' (positive = better than {anchor}):")
        print(f"  {'draft':<10} | {'q10':>7} {'q25':>7} {'q50':>7} {'q75':>7} {'q90':>7} | {'mean':>7} | {'% pos>{anchor}':>10}")
        print('  ' + '-' * 80)
        for name in list(trained.keys())[1:]:
            tm = np.array(data[f'wrong_mass_{name}'])
            d = tm - anchor_mass
            qs = np.percentile(d, [10, 25, 50, 75, 90])
            up = (d > 0).mean()
            print(f"  {name:<10} | {qs[0]:+7.4f} {qs[1]:+7.4f} {qs[2]:+7.4f} "
                  f"{qs[3]:+7.4f} {qs[4]:+7.4f} | {d.mean():+7.4f} | {up:>9.2%}")

        # Pairwise flip-disagreement: how many wrong positions does name flip but anchor doesn't?
        print(f"\n  Pairwise flip disagreement (base-wrong positions, anchor='{anchor}'):")
        print(f"  {'draft':<10} | {'flip & !anchor':<15} | {'!flip & anchor':<15} | {'net flips':<10}")
        print('  ' + '-' * 60)
        anchor_flips = np.array(data[f'wrong_correct_{anchor}']).astype(bool)
        for name in list(trained.keys())[1:]:
            flips = np.array(data[f'wrong_correct_{name}']).astype(bool)
            only_name = (flips & ~anchor_flips).sum()
            only_anchor = (~flips & anchor_flips).sum()
            net = only_name - only_anchor
            print(f"  {name:<10} | {int(only_name):>15d} | {int(only_anchor):>15d} | {net:>+10d}")


if __name__ == "__main__":
    main()
