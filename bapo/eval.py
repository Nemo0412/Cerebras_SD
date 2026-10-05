"""
BAPO Evaluation: real speculative decoding on MT-bench.

Loads a trained BAPO/distill checkpoint into the EAGLE3 inference engine,
runs actual speculative decoding on MT-bench questions, and measures:
  - Alpha (mean tokens accepted per verification round)
  - Wall-clock time per question
  - Speedup vs autoregressive baseline (if baseline file provided)

Usage:
  # Evaluate a single checkpoint
  python bapo/eval.py \
      --base-model-path /scratch/xt2251/models/Llama-3.1-8B-Instruct \
      --ea-model-path bapo_sweep/default/state_2 \
      --tag default

  # Evaluate all sweep checkpoints
  bash bapo/slurm/submit_eval.sh

  # Evaluate the original EAGLE3 (no BAPO training) as baseline
  python bapo/eval.py \
      --base-model-path /scratch/xt2251/models/Llama-3.1-8B-Instruct \
      --ea-model-path /scratch/xt2251/models/EAGLE3-LLaMA3.1-Instruct-8B \
      --tag eagle3_original
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer


def _seed_all(s=0):
    """Seed every RNG source touched by EAGLE3 inference."""
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    random.seed(s)
    np.random.seed(s)

# Add parent dir for model imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.ea_model import EaModel
from model.utils import *


@torch.inference_mode()
def evaluate(args):
    print(f"[EVAL] Loading model: {args.ea_model_path}")
    print(f"[EVAL] Base model: {args.base_model_path}")

    model = EaModel.from_pretrained(
        base_model_path=args.base_model_path,
        ea_model_path=args.ea_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_eagle3=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    model.eval()

    # Load questions
    question_file = os.path.join(
        os.path.dirname(__file__), '..', 'data', args.bench_name, 'question.jsonl')
    questions = []
    with open(question_file) as f:
        for line in f:
            questions.append(json.loads(line))
    if args.num_questions is not None:
        questions = questions[:args.num_questions]
    print(f"[EVAL] Loaded {len(questions)} questions from {args.bench_name}")

    if args.temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=args.temperature)
    else:
        logits_processor = None

    system_msg = (
        "You are a helpful, respectful and honest assistant. Always answer as "
        "helpfully as possible, while being safe.  Your answers should not "
        "include any harmful, unethical, racist, sexist, toxic, dangerous, or "
        "illegal content. Please ensure that your responses are socially "
        "unbiased and positive in nature.\n\nIf a question does not make any "
        "sense, or is not factually coherent, explain why instead of answering "
        "something not correct. If you don't know the answer to a question, "
        "please don't share false information."
    )

    # Warmup
    print("[EVAL] Warming up...")
    q0 = questions[0]
    for _ in range(3):
        _seed_all(args.seed)
        msgs = [{"role": "system", "content": system_msg},
                {"role": "user", "content": q0["turns"][0]}]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True)
        input_ids = tokenizer([prompt], add_special_tokens=False).input_ids
        model.eagenerate(
            torch.as_tensor(input_ids).cuda(),
            temperature=args.temperature,
            max_length=4096, log=True, is_llama3=True)

    # Evaluate
    results = []
    total_tokens = 0
    total_rounds = 0
    total_time = 0.0

    for question in tqdm(questions, desc=f"[EVAL] {args.tag}"):
        _seed_all(args.seed * 1000 + question["question_id"])
        messages = [{"role": "system", "content": system_msg}]
        q_results = {"question_id": question["question_id"], "turns": []}

        for j, qs in enumerate(question["turns"]):
            messages.append({"role": "user", "content": qs})
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer([prompt], add_special_tokens=False).input_ids

            torch.cuda.synchronize()
            t0 = time.time()
            output_ids, new_token, idx = model.eagenerate(
                torch.as_tensor(input_ids).cuda(),
                temperature=args.temperature,
                log=True, is_llama3=True)
            torch.cuda.synchronize()
            elapsed = time.time() - t0

            output_ids = output_ids[0][len(input_ids[0]):]
            stop_ids = [tokenizer.eos_token_id,
                        tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            stop_idx = [i for i, tok in enumerate(output_ids) if tok in stop_ids]
            if stop_idx:
                output_ids = output_ids[:stop_idx[0]]

            output = tokenizer.decode(output_ids, spaces_between_special_tokens=False)
            for st in tokenizer.special_tokens_map.values():
                if isinstance(st, list):
                    for s in st:
                        output = output.replace(s, "")
                else:
                    output = output.replace(st, "")
            output = output.strip()

            new_token = int(new_token)
            idx = int(idx)
            alpha = new_token / (idx + 1) if idx > 0 else 0
            total_tokens += new_token
            total_rounds += idx + 1
            total_time += elapsed

            q_results["turns"].append({
                "output": output,
                "new_tokens": new_token,
                "rounds": idx + 1,
                "alpha": round(alpha, 3),
                "wall_time": round(elapsed, 3),
                "tokens_per_sec": round(new_token / elapsed, 1) if elapsed > 0 else 0,
            })

            messages.append({"role": "assistant", "content": output})

        results.append(q_results)

    # Summary
    mean_alpha = total_tokens / total_rounds if total_rounds > 0 else 0
    mean_tps = total_tokens / total_time if total_time > 0 else 0

    summary = {
        "tag": args.tag,
        "ea_model_path": args.ea_model_path,
        "bench_name": args.bench_name,
        "num_questions": len(questions),
        "total_tokens": total_tokens,
        "total_rounds": total_rounds,
        "total_time_sec": round(total_time, 2),
        "mean_alpha": round(mean_alpha, 3),
        "mean_tokens_per_sec": round(mean_tps, 1),
    }

    print(f"\n{'='*50}")
    print(f"  Tag:             {args.tag}")
    print(f"  Mean Alpha:      {mean_alpha:.3f}")
    print(f"  Tokens/sec:      {mean_tps:.1f}")
    print(f"  Total tokens:    {total_tokens}")
    print(f"  Total rounds:    {total_rounds}")
    print(f"  Total time:      {total_time:.1f}s")
    print(f"{'='*50}")

    # Save (per-bench to avoid overwrite when running multiple benches under one tag)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.tag}__{args.bench_name}.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"[EVAL] Saved to {out_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str,
                        default="/scratch/xt2251/models/Llama-3.1-8B-Instruct")
    parser.add_argument("--ea-model-path", type=str, required=True,
                        help="Path to EAGLE3 checkpoint (original or BAPO-trained)")
    parser.add_argument("--tag", type=str, default="eval",
                        help="Name for this eval run")
    parser.add_argument("--bench-name", type=str, default="all",
                        help="Benchmark name or 'all' to run mt_bench,gsm8k,humaneval,qa,sum,alpaca,aime")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--total-token", type=int, default=60)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="bapo_eval_results")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base RNG seed. Per-question uses seed*1000 + question_id.")
    parser.add_argument("--num-questions", type=int, default=None,
                        help="Limit to first N questions (default: all).")
    args = parser.parse_args()

    ALL_BENCHMARKS = ["mt_bench", "gsm8k", "humaneval", "qa", "sum", "alpaca", "aime"]

    if args.bench_name == "all":
        benchmarks = ALL_BENCHMARKS
    else:
        benchmarks = [b.strip() for b in args.bench_name.split(",")]

    all_summaries = []
    for bench in benchmarks:
        qfile = os.path.join(os.path.dirname(__file__), '..', 'data', bench, 'question.jsonl')
        if not os.path.exists(qfile):
            print(f"[EVAL] SKIP {bench} (no question file)")
            continue
        args.bench_name = bench
        summary = evaluate(args)
        all_summaries.append(summary)

    # Print comparison table
    if len(all_summaries) > 1:
        print(f"\n{'='*60}")
        print(f"  SUMMARY for {args.tag}")
        print(f"  {'Benchmark':<12} {'Alpha':>8} {'Tok/s':>8} {'Questions':>10}")
        print(f"  {'-'*40}")
        total_tok, total_rnd, total_t = 0, 0, 0.0
        for s in all_summaries:
            print(f"  {s['bench_name']:<12} {s['mean_alpha']:>8.3f} "
                  f"{s['mean_tokens_per_sec']:>8.1f} {s['num_questions']:>10}")
            total_tok += s['total_tokens']
            total_rnd += s['total_rounds']
            total_t += s['total_time_sec']
        overall_alpha = total_tok / total_rnd if total_rnd > 0 else 0
        overall_tps = total_tok / total_t if total_t > 0 else 0
        print(f"  {'-'*40}")
        print(f"  {'OVERALL':<12} {overall_alpha:>8.3f} {overall_tps:>8.1f} "
              f"{sum(s['num_questions'] for s in all_summaries):>10}")
        print(f"{'='*60}")
