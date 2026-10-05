"""
Diagnostic: compare ro vs noro step-0 forward path equivalence.

Compares the actual forward calls used in each path:
  noro: draft(input_ids, attention_mask)                   # no use_cache, no attn_impl override
  ro:   draft(input_ids, attention_mask, use_cache=True)   # explicit attn_impl="sdpa"

Cross 3 axes:
  - dtype: bf16 vs fp16
  - attn_implementation: sdpa (ro forces this), eager (potential noro fallback), default (noro behavior)
  - use_cache: True (ro) vs False (noro)

Reports:
  - max/mean |Δlogit| at step 0
  - |Δper_pos| (KL/CE per-position)
  - |Δanchor_loss| (sliding γ-window aggregation)

Usage:
    python sdpo/diagnose_ro_noro_equivalence.py \
        --ckpt /scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/q8_q06_kl_regen_l2k/state_2 \
        --testpath sdpo/data/mixed_val_80.jsonl \
        --basepath Qwen/Qwen3-8B \
        --dtype bf16 fp16 \
        --attn_impls default sdpa eager \
        --num_batches 4
"""
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_pipeline import build_dataset, DataCollator


DTYPE_MAP = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}


@torch.no_grad()
def noro_step0_loss(draft, input_ids, attention_mask, loss_mask, target_p,
                    target_argmax, gamma, anchor_weight='pow08', anchor='kl'):
    """Replicate noro step-0 anchor loss exactly (small_lm_model.py)."""
    out = draft(input_ids=input_ids, attention_mask=attention_mask)
    draft_lg = out.logits[:, :-1, :].float()
    dlogp = F.log_softmax(draft_lg, dim=-1)
    if anchor == 'kl':
        per_pos = -torch.sum(target_p * dlogp, dim=-1)
    else:
        per_pos = -dlogp.gather(-1, target_argmax.unsqueeze(-1)).squeeze(-1)
    return draft_lg, per_pos, _aggregate(per_pos, loss_mask, gamma, anchor_weight)


@torch.no_grad()
def ro_step0_loss(draft, input_ids, attention_mask, loss_mask, target_p,
                  target_argmax, gamma, anchor_weight='pow08', anchor='kl'):
    """Replicate ro step-0 anchor loss exactly (small_lm_rollout_model.py)."""
    out = draft(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    draft_lg = out.logits[:, :-1, :].float()
    dlogp = F.log_softmax(draft_lg, dim=-1)
    if anchor == 'kl':
        per_pos = -torch.sum(target_p * dlogp, dim=-1)
    else:
        per_pos = -dlogp.gather(-1, target_argmax.unsqueeze(-1)).squeeze(-1)
    return draft_lg, per_pos, _aggregate(per_pos, loss_mask, gamma, anchor_weight)


def _aggregate(per_pos, loss_mask, gamma, anchor_weight):
    mask = loss_mask[:, 1:].bool()
    nv = mask.float().sum()
    Lp = mask.shape[1]
    usable = Lp - gamma + 1
    if anchor_weight == 'none' or usable <= 0:
        return (per_pos * mask.float()).sum() / (nv + 1e-8)
    aw = _step_weights(anchor_weight, gamma, per_pos.device, per_pos.dtype)
    stack = torch.stack(
        [per_pos[:, k:k + usable] for k in range(gamma)], dim=-1)
    win = (stack * aw).sum(dim=-1)
    win_mask = mask[:, :usable]
    return (win * win_mask.float()).sum() / (win_mask.float().sum() + 1e-8)


def _step_weights(scheme, n, device, dtype):
    if scheme == 'pow08':
        w = torch.tensor([0.8 ** k for k in range(n)], device=device, dtype=dtype)
    elif scheme == 'uniform':
        w = torch.ones(n, device=device, dtype=dtype)
    elif scheme == 'dec':
        w = torch.tensor([(n - k) / n for k in range(n)], device=device, dtype=dtype)
    else:
        w = torch.ones(n, device=device, dtype=dtype)
    return w


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='Trained draft ckpt dir (state_N)')
    p.add_argument('--basepath', required=True, help='Target model path (for tokenizer)')
    p.add_argument('--testpath', required=True)
    p.add_argument('--dtype', nargs='+', default=['bf16', 'fp16'],
                   choices=list(DTYPE_MAP.keys()))
    p.add_argument('--attn_impls', nargs='+', default=['default', 'sdpa', 'eager'],
                   choices=['default', 'sdpa', 'eager', 'flash_attention_2'])
    p.add_argument('--anchor', default='kl', choices=['kl', 'ce'])
    p.add_argument('--anchor_weight', default='pow08')
    p.add_argument('--gamma', type=int, default=7)
    p.add_argument('--max_len', type=int, default=2048)
    p.add_argument('--num_batches', type=int, default=4)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.basepath)
    ds = build_dataset(tok, args.testpath, args.max_len, max_samples=args.num_batches,
                       gamma=args.gamma)
    coll = DataCollator()

    # Load target model once in fp32 for ground-truth target_p / target_argmax
    target = AutoModelForCausalLM.from_pretrained(
        args.basepath, torch_dtype=torch.bfloat16,
        attn_implementation='sdpa').cuda().eval()

    print(f'\nckpt: {args.ckpt}')
    print(f'anchor={args.anchor} anchor_weight={args.anchor_weight} gamma={args.gamma}')
    print('=' * 100)
    print(f'{"dtype":>6} {"attn_impl":>10} {"resolved":>10} {"batch":>5} | '
          f'{"max|Δlogit|":>12} {"mean|Δlogit|":>12} '
          f'{"max|Δpp|":>10} {"|Δanchor|":>11}  noro / ro')
    print('-' * 100)

    for dt_name in args.dtype:
        dt = DTYPE_MAP[dt_name]
        for ai in args.attn_impls:
            kwargs = dict(torch_dtype=dt)
            if ai != 'default':
                kwargs['attn_implementation'] = ai
            try:
                draft = AutoModelForCausalLM.from_pretrained(
                    args.ckpt, **kwargs).cuda().eval()
            except Exception as e:
                print(f'{dt_name:>6} {ai:>10} (load fail: {type(e).__name__})')
                continue

            resolved = getattr(draft.config, '_attn_implementation', '?')

            for b_i in range(min(args.num_batches, len(ds))):
                batch = coll([ds[b_i]])
                iid = batch['input_ids'].cuda()
                am = batch['attention_mask'].cuda()
                lm = batch['loss_mask'].cuda()

                with torch.no_grad():
                    t_lg = target(input_ids=iid, attention_mask=am
                                  ).logits[:, :-1, :].float()
                    t_p = F.softmax(t_lg, dim=-1)
                    t_arg = t_lg.argmax(dim=-1)

                n_lg, n_pp, n_loss = noro_step0_loss(
                    draft, iid, am, lm, t_p, t_arg,
                    args.gamma, args.anchor_weight, args.anchor)
                r_lg, r_pp, r_loss = ro_step0_loss(
                    draft, iid, am, lm, t_p, t_arg,
                    args.gamma, args.anchor_weight, args.anchor)

                d_lg = (n_lg - r_lg).abs()
                d_pp = (n_pp - r_pp).abs()
                d_loss = (n_loss - r_loss).abs().item()

                print(f'{dt_name:>6} {ai:>10} {resolved:>10} {b_i:>5} | '
                      f'{d_lg.max().item():>12.3e} {d_lg.mean().item():>12.3e} '
                      f'{d_pp.max().item():>10.3e} {d_loss:>11.3e}  '
                      f'{n_loss.item():.6f} / {r_loss.item():.6f}')

            del draft
            torch.cuda.empty_cache()

    print('=' * 100)
    print('Interpretation:')
    print('  - resolved column shows what attn impl was actually selected')
    print('  - max|Δlogit| ~ 0 same dtype same attn -> use_cache=True is bit-equiv to noro')
    print('  - max|Δlogit| > 1e-4 within same attn  -> use_cache flips a kernel branch')
    print('  - sdpa vs eager Δlogit >> use_cache Δlogit -> attn_impl is the dominant axis')
    print('  - bf16 vs fp16 Δlogit >> attn_impl Δlogit  -> dtype dominates')
    print('  - if Δanchor across all axes << 0.016 -> training-dynamics (loss_scale,')
    print('    optimizer state precision) is the dominant gap, not single-step forward.')


if __name__ == '__main__':
    main()
