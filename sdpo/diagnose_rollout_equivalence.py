"""
Diagnostic: verify rollout chain logits match standard autoregressive generation.

For a chosen anchor position p, run:
  Method A: standard AR — at each step append draft's argmax, fresh full forward.
  Method B: rollout code — 4D mask + past_key_values, K-batched (here K=1).

If A == B, the rollout machinery (mask + cache + position_ids) is correct;
any training degradation is from the dirty-context objective, not the code.

Usage:
    python sdpo/diagnose_rollout_equivalence.py \
        --draft Qwen/Qwen3-0.6B \
        --testpath sdpo/data/mixed_val_80.jsonl \
        --gamma 7 --num_samples 3
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_rollout_mask(anchor_pos, step_k, prefix_len, K, device, dtype):
    """Copy of SmallLMRolloutModel._build_rollout_mask, dtype-explicit."""
    total_kv = prefix_len + step_k * K
    min_val = torch.finfo(dtype).min
    mask = torch.full((1, 1, K, total_kv), min_val, device=device, dtype=dtype)
    col = torch.arange(prefix_len, device=device)
    attend_prefix = col.unsqueeze(0) <= anchor_pos.unsqueeze(1)
    mask[0, 0, :, :prefix_len][attend_prefix] = 0.0
    anchor_idx = torch.arange(K, device=device)
    for j in range(step_k):
        kv_pos = prefix_len + j * K + anchor_idx
        mask[0, 0, anchor_idx, kv_pos] = 0.0
    return mask


@torch.inference_mode()
def standard_ar_chain(draft, input_ids, anchor_pos, gamma):
    chain = []
    cur = input_ids[:, :anchor_pos + 1].clone()
    for k in range(gamma):
        out = draft(cur, use_cache=False, return_dict=True)
        logits_k = out.logits[0, -1, :].float()
        argmax_k = logits_k.argmax(-1)
        chain.append((argmax_k.item(), logits_k))
        cur = torch.cat([cur, argmax_k.view(1, 1)], dim=1)
    return chain


@torch.inference_mode()
def rollout_chain_via_code(draft, input_ids, anchor_pos, gamma):
    L = input_ids.shape[1]
    device = input_ids.device
    K = 1
    anchor = torch.tensor([anchor_pos], device=device, dtype=torch.long)

    prefix = input_ids[:, :anchor_pos + 1]
    prefix_out = draft(prefix, use_cache=True, return_dict=True)
    rollout_cache = prefix_out.past_key_values

    dlg0 = prefix_out.logits[0, -1, :].float()
    chain = [(dlg0.argmax(-1).item(), dlg0)]

    current_tokens = dlg0.argmax(-1).view(1, 1)
    prefix_len = anchor_pos + 1

    for step in range(1, gamma):
        pos_ids = torch.tensor([[anchor_pos + step]], device=device, dtype=torch.long)
        attn_mask_4d = build_rollout_mask(
            anchor, step, prefix_len, K, device, draft.dtype)
        out = draft(
            input_ids=current_tokens,
            attention_mask=attn_mask_4d,
            position_ids=pos_ids,
            past_key_values=rollout_cache,
            use_cache=True,
            return_dict=True,
        )
        rollout_cache = out.past_key_values
        logits_k = out.logits[0, 0, :].float()
        argmax_k = logits_k.argmax(-1)
        chain.append((argmax_k.item(), logits_k))
        current_tokens = argmax_k.view(1, 1)

    return chain


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--draft", default="Qwen/Qwen3-0.6B")
    p.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    p.add_argument("--testpath", required=True)
    p.add_argument("--gamma", type=int, default=7)
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--anchor_offset", type=int, default=50)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    print(f"Loading draft from {args.draft} ...")
    draft = AutoModelForCausalLM.from_pretrained(
        args.draft, torch_dtype=torch.float16,
        attn_implementation="sdpa").cuda().eval()

    with open(args.testpath) as f:
        rows = [json.loads(l) for l in f][:args.num_samples]

    print(f"\nComparing standard AR vs rollout code @ γ={args.gamma}")
    print("=" * 80)
    for ri, row in enumerate(rows):
        if "conversations" in row:
            role_map = {"human": "user", "gpt": "assistant", "system": "system"}
            msgs = [{"role": role_map.get(c["from"], c["from"]),
                     "content": c["value"]} for c in row["conversations"]]
            text = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=False)
        elif "turns" in row:
            text = tok.apply_chat_template(
                [{"role": "user", "content": row["turns"][0]}],
                tokenize=False, add_generation_prompt=True)
        else:
            continue

        ids = tok(text, return_tensors="pt",
                  add_special_tokens=False).input_ids.cuda()
        L = ids.shape[1]
        if L < args.anchor_offset + args.gamma + 5:
            print(f"sample {ri}: too short ({L} tokens), skipping")
            continue

        anchor_pos = min(args.anchor_offset, L - args.gamma - 5)

        chain_a = standard_ar_chain(draft, ids, anchor_pos, args.gamma)
        chain_b = rollout_chain_via_code(draft, ids, anchor_pos, args.gamma)

        print(f"\nSample {ri}, L={L}, anchor={anchor_pos}")
        print(f"  {'step':>4}  {'arg_A':>8}  {'arg_B':>8}  match  "
              f"{'max|Δlogit|':>12}  {'mean|Δlogit|':>12}")
        for k in range(args.gamma):
            arg_a, lg_a = chain_a[k]
            arg_b, lg_b = chain_b[k]
            d = (lg_a - lg_b).abs()
            ok = "✓" if arg_a == arg_b else "✗"
            print(f"  {k:>4}  {arg_a:>8}  {arg_b:>8}  {ok}    "
                  f"{d.max().item():>12.3e}  {d.mean().item():>12.3e}")

    print("\n" + "=" * 80)
    print("All ✓ + max|Δ| < 1e-3 -> rollout code numerically correct.")
    print("Any ✗ or max|Δ| > 1e-3 -> bug in 4D mask / KV cache / pos_ids.")


if __name__ == "__main__":
    main()
