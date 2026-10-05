"""Per-step hard-accept rate for an ro-trained ckpt.

Reports for each γ-step k=0..γ-1:
  - accept rate (P(draft argmax == target argmax | reached step k))
  - cumulative accept rate (P(all step 0..k accepted))

Usage:
    python sdpo/diagnose_ro_step_acc.py \\
        --draft /scratch/.../q8_q06_klv4_regen_ro_l2k_g2/state_2 \\
        --gamma 7 --num-samples 30
"""
import argparse
import glob
import os
import sys

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
        path, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").cuda().eval()
    return m


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target', default='Qwen/Qwen3-8B')
    p.add_argument('--draft', required=True,
                   help='Path to draft ckpt dir')
    p.add_argument('--data-path',
                   default='/scratch/tx856/spec_reason/accept_length/'
                           'SDPO-Speculative-Decoding-Policy-Optimization/'
                           'sdpo/data/mixed_val_80.jsonl')
    p.add_argument('--gamma', type=int, default=7)
    p.add_argument('--num-samples', type=int, default=30)
    p.add_argument('--max-len', type=int, default=1024)
    p.add_argument('--max-anchors-per-sample', type=int, default=64,
                   help='Random subset of valid anchor positions per sample')
    args = p.parse_args()

    print(f"Loading target: {args.target}")
    target = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").cuda().eval()
    tok = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)

    print(f"Loading draft: {args.draft}")
    draft = load_draft(args.draft)

    print(f"Loading {args.num_samples} samples from {args.data_path}")
    ds = load_dataset('json', data_files=args.data_path)['train']
    ds = ds.select(range(min(args.num_samples * 3, len(ds))))

    # Counters per step: count of "step k reached" and "step k accepted"
    gamma = args.gamma
    reached = [0] * gamma
    accepted = [0] * gamma

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
            text = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False,
                enable_thinking=True)
        except TypeError:
            text = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False)
        ids = tok(text, return_tensors="pt",
                  add_special_tokens=False).input_ids
        if ids.shape[1] < 30 or ids.shape[1] > args.max_len:
            continue
        ids = ids.cuda()
        L = ids.shape[1]

        # Target argmax (clean) at every position
        tgt_lg = target(input_ids=ids).logits[:, :-1, :].float()
        tgt_argmax = tgt_lg.argmax(-1).squeeze(0)         # [L-1]

        # Draft step 0 (clean prefix forward)
        prefix_out = draft(input_ids=ids, use_cache=True, return_dict=True)
        dlg0 = prefix_out.logits[:, :-1, :].float()
        d_step0_argmax = dlg0.argmax(-1).squeeze(0)       # [L-1]
        cache = prefix_out.past_key_values

        # Pick valid anchors: positions where we can run γ-step rollout
        # i.e., anchor_pos s.t. anchor_pos + γ - 1 <= L-2
        valid_anchor = torch.arange(L - 1 - (gamma - 1), device=ids.device)
        if valid_anchor.numel() < 2:
            continue
        # Sample subset
        n_pick = min(args.max_anchors_per_sample, valid_anchor.numel())
        perm = torch.randperm(valid_anchor.numel(), device=ids.device)[:n_pick]
        anchor_pos = valid_anchor[perm].sort().values

        # Step 0 hard_accept (clean prefix prediction at anchor_pos)
        step0_accept = (d_step0_argmax[anchor_pos] ==
                        tgt_argmax[anchor_pos]).cpu()
        K = anchor_pos.numel()
        reached[0] += K
        accepted[0] += int(step0_accept.sum())

        # Rollout step 1..γ-1: build chain per anchor (sequential, naive)
        # For each anchor, feed prev-step argmax → draft → next prediction.
        # Only continue rollout if previous step accepted (mimics inference).
        # NOTE: this naive approach ignores KV cache reuse for simplicity;
        # the per-step result reflects the same draft model outputs.
        for ai in range(K):
            ap = int(anchor_pos[ai].item())
            if not bool(step0_accept[ai]):
                continue   # didn't reach step 1
            # Step 1+: use prev-step argmax as input at the next position.
            cur_token = d_step0_argmax[ap].view(1, 1)
            cur_pos = ap + 1
            # Build a fresh small input for sequential rollout from anchor.
            # input includes prefix [0..ap] + current token at ap+1
            # For step k (k=1..γ-1): we need to forward draft on
            # (clean prefix [0..ap], step k-1's predicted token at ap+k).
            # For simplicity, re-encode each step from scratch (slow but correct).
            for k in range(1, gamma):
                if cur_pos > L - 2:
                    break
                # Build current input: clean prefix + chain so far
                # We track "chain_tokens" appended after clean prefix.
                # Simplest: re-forward draft on full ids[0:ap+1] + chain tokens
                # then use the last logit.
                if k == 1:
                    chain_so_far = cur_token   # [1, 1]
                else:
                    chain_so_far = torch.cat([chain_so_far, cur_token], dim=1)
                full_input = torch.cat(
                    [ids[:, :ap + 1], chain_so_far], dim=1)
                out = draft(input_ids=full_input).logits.float()
                next_logit = out[0, -1]
                next_argmax = int(next_logit.argmax().item())
                # Compare with target_argmax at position ap+k
                target_token = int(tgt_argmax[ap + k].item())
                reached[k] += 1
                if next_argmax == target_token:
                    accepted[k] += 1
                    cur_token = torch.tensor(
                        [[next_argmax]], device=ids.device, dtype=ids.dtype)
                    cur_pos = ap + k + 1
                else:
                    break

        num_used += 1
        if num_used >= args.num_samples:
            break

    print(f"\n{'='*70}")
    print(f"Analyzed {num_used} samples, draft: {args.draft}")
    print(f"\n{'step':<6} {'reached':>10} {'accepted':>10} "
          f"{'accept rate':>14} {'cum accept':>12}")
    print('-' * 60)
    cum = 1.0
    for k in range(gamma):
        if reached[k] == 0:
            print(f"{k:<6} {0:>10} {0:>10} {'—':>14} {'—':>12}")
            continue
        rate = accepted[k] / reached[k]
        # cum: probability of reaching AND accepting step k from start
        cum_reached_rate = reached[k] / max(reached[0], 1)
        cum_accept = accepted[k] / max(reached[0], 1)
        print(f"{k:<6} {reached[k]:>10} {accepted[k]:>10} "
              f"{rate:>13.3%} {cum_accept:>11.3%}")

    print(f"\nstep 0 accept rate ≈ {accepted[0]/max(reached[0],1):.3%}")
    print(f"step 0 reject rate ≈ {(1-accepted[0]/max(reached[0],1)):.3%}")


if __name__ == "__main__":
    main()
