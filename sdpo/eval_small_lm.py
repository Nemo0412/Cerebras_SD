"""
Eval for Small LM Draft Speculative Decoding.

Runs actual speculative decoding: draft generates γ tokens, target verifies.
Measures mean alpha (tokens accepted per round).

Usage:
  python sdpo/eval_small_lm.py \
    --base-model-path Qwen/Qwen3-8B \
    --draft-model-path /path/to/checkpoint \
    --tag my_run --bench-name mt_bench
"""

import argparse
import json
import os
import sys
import time

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.inference_mode()
def speculative_decode_step(target_model, draft_model, input_ids, gamma):
    """One round of speculative decoding.

    Returns:
        new_tokens: accepted tokens + 1 correction token
        n_accepted: number of draft tokens accepted
    """
    device = input_ids.device

    # Draft: generate γ tokens autoregressively
    draft_ids = input_ids.clone()
    draft_tokens = []
    for _ in range(gamma):
        draft_logits = draft_model(draft_ids).logits[:, -1, :]
        next_token = draft_logits.argmax(dim=-1, keepdim=True)
        draft_tokens.append(next_token)
        draft_ids = torch.cat([draft_ids, next_token], dim=1)

    draft_tokens = torch.cat(draft_tokens, dim=1)  # [1, γ]

    # Target: verify all γ tokens in one forward pass
    candidate = torch.cat([input_ids, draft_tokens], dim=1)
    target_logits = target_model(candidate).logits

    # Check acceptance: compare target's prediction at each position
    n_accepted = 0
    for i in range(gamma):
        pos = input_ids.shape[1] + i - 1  # target logits at this position predict next token
        target_token = target_logits[:, pos, :].argmax(dim=-1)
        if target_token.item() == draft_tokens[:, i].item():
            n_accepted += 1
        else:
            break

    # Correction: target's prediction at the first rejected position
    correction_pos = input_ids.shape[1] + n_accepted - 1
    correction_token = target_logits[:, correction_pos, :].argmax(dim=-1, keepdim=True)

    # New tokens = accepted draft tokens + correction
    new_tokens = torch.cat([draft_tokens[:, :n_accepted], correction_token], dim=1)

    return new_tokens, n_accepted


@torch.inference_mode()
def generate_with_spec_decode(target_model, draft_model, input_ids, max_new_tokens, gamma):
    """Generate tokens using speculative decoding."""
    total_tokens = 0
    total_rounds = 0
    total_accepted = 0
    cur_ids = input_ids

    while total_tokens < max_new_tokens:
        new_tokens, n_accepted = speculative_decode_step(
            target_model, draft_model, cur_ids, gamma)

        cur_ids = torch.cat([cur_ids, new_tokens], dim=1)
        total_tokens += new_tokens.shape[1]
        total_rounds += 1
        total_accepted += n_accepted

        # Check for EOS
        if hasattr(target_model.config, 'eos_token_id'):
            eos = target_model.config.eos_token_id
            if isinstance(eos, list):
                if any(t in new_tokens[0].tolist() for t in eos):
                    break
            elif eos in new_tokens[0].tolist():
                break

    mean_alpha = total_accepted / total_rounds if total_rounds > 0 else 0
    return cur_ids, total_tokens, total_rounds, mean_alpha


def load_questions(bench_name, data_dir="data"):
    path = os.path.join(data_dir, bench_name, "question.jsonl")
    if not os.path.exists(path):
        print(f"  Bench {bench_name} not found at {path}, skipping")
        return []
    questions = []
    with open(path) as f:
        for line in f:
            questions.append(json.loads(line))
    return questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model-path', required=True)
    parser.add_argument('--draft-model-path', required=True)
    parser.add_argument('--tag', default='smalllm')
    parser.add_argument('--bench-name', default='mt_bench')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--gamma', type=int, default=7)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--output-dir', default='smalllm_eval_results')
    args = parser.parse_args()

    if args.bench_name == 'all':
        benches = ['mt_bench', 'gsm8k', 'humaneval', 'qa', 'sum', 'alpaca', 'aime']
    else:
        benches = [b.strip() for b in args.bench_name.split(',') if b.strip()]

    print(f"[EVAL] Loading target: {args.base_model_path}")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, torch_dtype=torch.float16, device_map="auto")
    target_model.eval()

    print(f"[EVAL] Loading draft: {args.draft_model_path}")
    # Fix DeepSpeed 'draft_model.' prefix in-place before from_pretrained.
    # Uses from_pretrained (not from_config) to support nested configs (Qwen3.5).
    import glob
    ckpt_files = glob.glob(os.path.join(args.draft_model_path, "*.bin"))
    for f in ckpt_files:
        state = torch.load(f, map_location="cpu")
        if any(k.startswith("draft_model.") for k in state):
            print(f"[EVAL] Stripping 'draft_model.' prefix from {f}")
            cleaned = {k.removeprefix("draft_model."): v for k, v in state.items()}
            torch.save(cleaned, f)
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model_path, torch_dtype=torch.float16, device_map="auto")
    draft_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}
    for bench in benches:
        questions = load_questions(bench)
        if not questions:
            continue

        print(f"[EVAL] Running {bench} ({len(questions)} questions)")

        bench_tokens = 0
        bench_rounds = 0
        bench_accepted = 0
        t0 = time.time()

        for q in tqdm(questions, desc=f"{args.tag}/{bench}"):
            prompt = q.get("turns", [q.get("prompt", "")])[0] if "turns" in q else q.get("prompt", "")
            msgs = [{"role": "user", "content": prompt}]
            try:
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=True)
            except TypeError:
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
            input_ids = input_ids.to(target_model.device)

            _, n_tokens, n_rounds, _ = generate_with_spec_decode(
                target_model, draft_model, input_ids, args.max_new_tokens, args.gamma)

            bench_tokens += n_tokens
            bench_rounds += n_rounds
            # Recompute accepted from tokens and rounds
            # Each round produces (n_accepted + 1) tokens, total = Σ(n_accepted_i + 1) = total_accepted + rounds
            bench_accepted += (n_tokens - n_rounds)

        elapsed = time.time() - t0
        mean_alpha = bench_accepted / bench_rounds if bench_rounds > 0 else 0

        print(f"\n{'='*50}")
        print(f"  Tag:             {args.tag}")
        print(f"  Bench:           {bench}")
        print(f"  Mean Alpha:      {mean_alpha:.3f}")
        print(f"  Tokens/sec:      {bench_tokens/elapsed:.1f}")
        print(f"  Total tokens:    {bench_tokens}")
        print(f"  Total rounds:    {bench_rounds}")
        print(f"  Total time:      {elapsed:.1f}s")
        print(f"{'='*50}\n")

        all_results[bench] = {
            "mean_alpha": mean_alpha,
            "tokens_per_sec": bench_tokens / elapsed,
            "total_tokens": bench_tokens,
            "total_rounds": bench_rounds,
        }

    # Summary
    if len(all_results) > 1:
        total_accepted = sum(r["total_tokens"] - r["total_rounds"] for r in all_results.values())
        total_rounds = sum(r["total_rounds"] for r in all_results.values())
        overall_alpha = total_accepted / total_rounds if total_rounds > 0 else 0
        print(f"{'='*60}")
        print(f"  SUMMARY for {args.tag}")
        print(f"  {'Benchmark':<15} {'Alpha':>8}  {'Tok/s':>8}  {'Questions':>10}")
        print(f"  {'-'*45}")
        for bench, r in all_results.items():
            print(f"  {bench:<15} {r['mean_alpha']:>8.3f}  {r['tokens_per_sec']:>8.1f}  {len(load_questions(bench)):>10}")
        print(f"  {'-'*45}")
        print(f"  {'OVERALL':<15} {overall_alpha:>8.3f}")
        print(f"{'='*60}")

    # Save
    out_path = os.path.join(args.output_dir, f"{args.tag}.json")
    with open(out_path, 'w') as f:
        json.dump({"tag": args.tag, "results": all_results}, f, indent=2)
    print(f"[EVAL] Saved to {out_path}")


if __name__ == "__main__":
    main()
