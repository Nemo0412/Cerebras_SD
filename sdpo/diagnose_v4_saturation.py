"""V4 sigmoid-gap saturation analysis.

Hypothesis: after KL training, P(target) on wrong positions sits ~0.21
(max P ~0.47) → logit gap ~ -0.8. With T=0.1, σ(gap/T) ~ σ(-8) ~ 3e-4.
V4's gradient coefficient σ(1-σ)/T then drops to ~3e-3, far weaker than KL's
(1 - P_target) ~ 0.79. This explains the observed KLV4 ≈ KL behavior.

Reports per-model on base-WRONG positions:
  - logit_gap = z[target] - max(z) distribution (q10/q25/q50/q75/q90, mean)
  - σ(gap/T) at T=0.1 and T=1.0 — V4 loss value
  - σ(1-σ)/T — V4 gradient coefficient on z[target]
  - (1 - P_target) — proxy for KL gradient coefficient
  - gradient ratio V4/KL

Usage:
    python sdpo/diagnose_v4_saturation.py --num-samples 30
"""
import argparse
import glob
import os

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
def forward_logits(model, input_ids):
    """Return logits at each position in float32. Shape [L-1, V]."""
    return model(input_ids=input_ids,
                 return_dict=True).logits[:, :-1, :].float().squeeze(0)


def quantiles(x, qs=(10, 25, 50, 75, 90)):
    return np.percentile(x, qs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target', default='Qwen/Qwen3-8B')
    p.add_argument('--base-draft', default='Qwen/Qwen3-0.6B')
    p.add_argument('--drafts', nargs='+', default=[
        'kl=/scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/q8_q06_kl_regen/state_2',
        'klv4=/scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/q8_q06_klv4_regen/state_2',
    ])
    p.add_argument('--data-path',
                   default='/scratch/yf3005/gto_data/sharegpt_qwen3_8b_regen.jsonl')
    p.add_argument('--num-samples', type=int, default=30)
    p.add_argument('--max-len', type=int, default=1024)
    p.add_argument('--temperatures', nargs='+', type=float, default=[0.1, 1.0])
    args = p.parse_args()

    parsed = []
    for item in args.drafts:
        if '=' in item:
            k, v = item.split('=', 1)
            parsed.append((k, v))
        else:
            parsed.append((os.path.basename(item.rstrip('/')), item))

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
    drafts = {'base': base_draft}
    for name, path in parsed:
        print(f"Loading {name}: {path}")
        drafts[name] = load_draft(path)

    print(f"\nLoading {args.num_samples} samples")
    ds = load_dataset('json', data_files=args.data_path)['train']
    ds = ds.shuffle(seed=42).select(range(args.num_samples * 3))

    # Per-model accumulators on base-WRONG positions:
    #   gap[name]   = list of (z[target] - max(z)) at wrong positions
    #   p_tgt[name] = list of P(target)
    stats = {n: {'gap': [], 'p_tgt': []} for n in drafts}

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

        # base draft to determine wrong mask
        base_lg = forward_logits(base_draft, ids)
        base_argmax = base_lg.argmax(-1)
        wrong = (base_argmax != tgt_argmax)
        if not wrong.any():
            continue

        for name, model in drafts.items():
            lg = forward_logits(model, ids) if name != 'base' else base_lg
            # logit gap = z[target] - max(z)  (negative on wrong)
            z_target = lg.gather(-1, tgt_argmax.unsqueeze(-1)).squeeze(-1)
            z_max = lg.max(-1).values
            gap = z_target - z_max
            # P(target)
            p = F.softmax(lg, dim=-1)
            p_target = p.gather(-1, tgt_argmax.unsqueeze(-1)).squeeze(-1)

            stats[name]['gap'].extend(gap[wrong].cpu().tolist())
            stats[name]['p_tgt'].extend(p_target[wrong].cpu().tolist())

        num_used += 1
        if num_used >= args.num_samples:
            break

    n_pos = len(stats['base']['gap'])
    print(f"\n{'='*95}")
    print(f"Analyzed {num_used} samples, {n_pos} base-WRONG positions")

    # ── Section 1: logit gap distribution ──
    print(f"\n── Logit gap z[target] - max(z) on base-WRONG positions ──")
    print(f"  {'model':<8} | {'q10':>7} {'q25':>7} {'q50':>7} {'q75':>7} {'q90':>7} | {'mean':>7}")
    print('  ' + '-' * 70)
    for name in drafts:
        g = np.array(stats[name]['gap'])
        qs = quantiles(g)
        print(f"  {name:<8} | {qs[0]:+7.3f} {qs[1]:+7.3f} {qs[2]:+7.3f} "
              f"{qs[3]:+7.3f} {qs[4]:+7.3f} | {g.mean():+7.3f}")

    print(f"\n── P(target) on base-WRONG positions ──")
    print(f"  {'model':<8} | {'q10':>6} {'q25':>6} {'q50':>6} {'q75':>6} {'q90':>6} | {'mean':>6}")
    print('  ' + '-' * 60)
    for name in drafts:
        p = np.array(stats[name]['p_tgt'])
        qs = quantiles(p)
        print(f"  {name:<8} | {qs[0]:>6.3f} {qs[1]:>6.3f} {qs[2]:>6.3f} "
              f"{qs[3]:>6.3f} {qs[4]:>6.3f} | {p.mean():>6.3f}")

    # ── Section 2: σ(gap/T) value (V4 loss) at different T ──
    print(f"\n── σ(gap/T): V4 loss value (smaller → loss saturated near 0) ──")
    print(f"  T={args.temperatures}")
    print(f"  {'model':<8} | T    | {'q10':>9} {'q25':>9} {'q50':>9} {'q75':>9} {'q90':>9} | {'mean':>9} | {'frac<0.01':>10}")
    print('  ' + '-' * 95)
    for T in args.temperatures:
        for name in drafts:
            g = np.array(stats[name]['gap'])
            sig = 1.0 / (1.0 + np.exp(-g / T))
            qs = quantiles(sig)
            saturated = (sig < 0.01).mean()
            print(f"  {name:<8} | {T:>4.2f} | {qs[0]:>9.2e} {qs[1]:>9.2e} {qs[2]:>9.2e} "
                  f"{qs[3]:>9.2e} {qs[4]:>9.2e} | {sig.mean():>9.2e} | {saturated:>9.2%}")

    # ── Section 3: σ(1-σ)/T — V4 gradient coefficient on z[target] ──
    print(f"\n── σ(1-σ)/T: V4 gradient coefficient on z[target] (per position) ──")
    print(f"  {'model':<8} | T    | {'q10':>9} {'q25':>9} {'q50':>9} {'q75':>9} {'q90':>9} | {'mean':>9}")
    print('  ' + '-' * 85)
    for T in args.temperatures:
        for name in drafts:
            g = np.array(stats[name]['gap'])
            sig = 1.0 / (1.0 + np.exp(-g / T))
            grad = sig * (1.0 - sig) / T
            qs = quantiles(grad)
            print(f"  {name:<8} | {T:>4.2f} | {qs[0]:>9.2e} {qs[1]:>9.2e} {qs[2]:>9.2e} "
                  f"{qs[3]:>9.2e} {qs[4]:>9.2e} | {grad.mean():>9.2e}")

    # ── Section 4: KL gradient coefficient ≈ (1 - P_target) ──
    # KL gradient on z[target] for target with mass-1 on tgt token: ∂L/∂z_target = P_draft(target) - 1
    # so |grad coef| = (1 - P_target)
    print(f"\n── (1 - P_target): KL gradient coefficient on z[target] ──")
    print(f"  {'model':<8} | {'q10':>6} {'q25':>6} {'q50':>6} {'q75':>6} {'q90':>6} | {'mean':>6}")
    print('  ' + '-' * 60)
    for name in drafts:
        p = np.array(stats[name]['p_tgt'])
        c = 1.0 - p
        qs = quantiles(c)
        print(f"  {name:<8} | {qs[0]:>6.3f} {qs[1]:>6.3f} {qs[2]:>6.3f} "
              f"{qs[3]:>6.3f} {qs[4]:>6.3f} | {c.mean():>6.3f}")

    # ── Section 5: V4 / KL gradient ratio (mean, mean) — actually per-position ──
    print(f"\n── V4 vs KL gradient magnitude ratio per position ──")
    print(f"  ratio = (sigmoid_coef · σ(1-σ)/T) / (1 - P_target)")
    print(f"  V4 loss has aux_weight, but here we report the raw ratio per position.")
    print(f"  (multiply by sigmoid_coef=0.1 to get final V4 contribution vs KL anchor)")
    print(f"  {'model':<8} | T    | sigmoid_coef·ratio  q10 / q50 / q90  |  mean")
    print('  ' + '-' * 80)
    for T in args.temperatures:
        for name in drafts:
            g = np.array(stats[name]['gap'])
            p = np.array(stats[name]['p_tgt'])
            sig = 1.0 / (1.0 + np.exp(-g / T))
            v4_grad = sig * (1.0 - sig) / T
            kl_grad = 1.0 - p
            ratio = 0.1 * v4_grad / np.maximum(kl_grad, 1e-9)
            qs = quantiles(ratio)
            print(f"  {name:<8} | {T:>4.2f} | "
                  f"q10={qs[0]:.2e} q50={qs[2]:.2e} q90={qs[4]:.2e}  | mean={ratio.mean():.2e}")

    # ── Section 6: histogram bins of |gap| / T at T=0.1 ──
    T = 0.1
    print(f"\n── Distribution of |gap|/T at T={T}: how saturated is V4? ──")
    print(f"  bins: |gap|/T < 1 (active), 1-3 (weak), 3-5 (faded), >5 (saturated)")
    bins = [(0, 1, 'active   <1   '),
            (1, 3, 'weak     1-3  '),
            (3, 5, 'faded    3-5  '),
            (5, 1e9, 'saturated>5  ')]
    print(f"  {'model':<8} | " + ' | '.join(b[2] for b in bins))
    print('  ' + '-' * 75)
    for name in drafts:
        g = np.abs(np.array(stats[name]['gap'])) / T
        row = f"  {name:<8} |"
        for lo, hi, _ in bins:
            f = ((g >= lo) & (g < hi)).mean()
            row += f"  {f:>9.2%}  |"
        print(row)


if __name__ == "__main__":
    main()
