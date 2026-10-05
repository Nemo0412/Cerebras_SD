"""Evaluation script for Gemma 4 chain spec decode.

Usage:
  python sdpo/eval_gemma4_chain.py \
    --target-model google/gemma-4-31B-it \
    --draft-model  google/gemma-4-E2B-it \
    --bench-name   mt_bench,gsm8k,humaneval \
    --gamma 7 --max-new-tokens 256 --num-samples 5 \
    --temperature 0.0 --tag gemma4_chain_smoke

Runs chain speculative decode (γ-step draft + parallel target verify) over each
bench's questions and records mean acceptance length.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from sdpo.spec_decode_gemma4 import load_text_model, chain_decode, tree_decode


def _seed_all(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def load_questions(bench: str, data_dir: str = "data"):
    path = os.path.join(data_dir, bench, "question.jsonl")
    if not os.path.exists(path):
        print(f"  Bench {bench} not found at {path}, skipping")
        return []
    with open(path) as f:
        return [json.loads(l) for l in f]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-model", default="google/gemma-4-31B-it")
    p.add_argument("--draft-model", default="google/gemma-4-E2B-it")
    p.add_argument("--bench-name", default="mt_bench")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--mode", default="chain", choices=["chain", "tree"])
    p.add_argument("--gamma", type=int, default=7)
    p.add_argument("--top-k", default="4,3,2,1,1,1,1",
                   help="Per-depth top_k for tree mode (comma list)")
    p.add_argument("--budget", type=int, default=63,
                   help="Max total nodes in tree mode")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--draft-mode", default="argmax",
                   choices=["argmax", "sample"])
    p.add_argument("--verify-mode", default="auto",
                   choices=["auto", "greedy", "simple", "ratio"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", required=True)
    p.add_argument("--output-dir", default="smalllm_tree_eval_results")
    p.add_argument("--print-samples", type=int, default=0,
                   help="Print first N generated samples for inspection.")
    p.add_argument("--enable-thinking", action="store_true", default=True)
    p.add_argument("--no-thinking", dest="enable_thinking", action="store_false")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load models ──
    # 31B target on cuda:0, E2B draft on cuda:1. Single-device per model avoids
    # device_map='auto' splitting embeddings across GPUs.
    n_gpu = torch.cuda.device_count()
    assert n_gpu >= 2, f"Need ≥2 GPUs (got {n_gpu}). Use --gres=gpu:2."
    print(f"[GEMMA4-EVAL] Loading target {args.target_model} → cuda:0")
    target = load_text_model(args.target_model, device="cuda:0",
                              dtype=torch.bfloat16)
    target.eval()
    print(f"[GEMMA4-EVAL] Loading draft {args.draft_model} → cuda:1")
    draft = load_text_model(args.draft_model, device="cuda:1",
                             dtype=torch.bfloat16)
    draft.eval()
    print(f"[GEMMA4-EVAL] Loading tokenizer {args.target_model}")
    tok = AutoTokenizer.from_pretrained(args.target_model)
    eos = tok.eos_token_id

    benches = ([b.strip() for b in args.bench_name.split(",") if b.strip()]
               if args.bench_name != "all" else
               ["mt_bench", "gsm8k", "humaneval", "longwriter_top50", "osr2_top50"])

    all_results = {}
    for bench in benches:
        qs = load_questions(bench, args.data_dir)
        if not qs: continue
        if args.num_samples:
            qs = qs[:args.num_samples]
        print(f"\n[GEMMA4-EVAL] Limited to first {len(qs)} samples")
        print(f"[GEMMA4-EVAL] {bench} chain decode (γ={args.gamma}, T={args.temperature}, "
              f"draft_mode={args.draft_mode}, verify_mode={args.verify_mode})")

        agg = {"rounds": 0, "accepted": 0, "generated": 0, "elapsed": 0.0}
        n_printed = 0
        top_k_list = [int(x) for x in args.top_k.split(",")] if args.mode == "tree" else None
        for qi, q in enumerate(tqdm(qs, desc=f"{args.mode}/{bench}")):
            _seed_all(args.seed * 1000 + qi)
            prompt = (q.get("turns", [q.get("prompt", "")])[0]
                      if "turns" in q else q.get("prompt", ""))
            msgs = [{"role": "user", "content": prompt}]
            try:
                text = tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=args.enable_thinking)
            except TypeError:
                text = tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt",
                      add_special_tokens=False).input_ids   # CPU; spec_decode moves per-device

            if args.mode == "tree":
                r = tree_decode(
                    target, draft, ids,
                    max_new_tokens=args.max_new_tokens,
                    gamma=args.gamma,
                    top_k=top_k_list,
                    budget=args.budget,
                    temperature=args.temperature,
                    draft_mode=args.draft_mode,
                    verify_mode=args.verify_mode,
                    eos_token_id=eos)
            else:
                r = chain_decode(
                    target, draft, ids,
                    max_new_tokens=args.max_new_tokens,
                    gamma=args.gamma,
                    temperature=args.temperature,
                    draft_mode=args.draft_mode,
                    verify_mode=args.verify_mode,
                    eos_token_id=eos)
            agg["rounds"] += r.n_rounds
            agg["accepted"] += r.total_accepted
            agg["generated"] += r.n_generated
            agg["elapsed"] += r.elapsed

            if n_printed < args.print_samples:
                n_printed += 1
                gen_text = tok.decode(
                    r.output_ids[0, ids.shape[1]:].cpu().tolist(),
                    skip_special_tokens=True)
                print(f"\n  --- sample {qi} (α={r.mean_alpha:.3f}, "
                      f"{r.n_generated} tok in {r.elapsed:.1f}s) ---")
                print(f"  PROMPT: {prompt[:200]}")
                print(f"  GEN: {gen_text[:400]}")

        if agg["rounds"]:
            mean_alpha = agg["accepted"] / agg["rounds"]
            tps = agg["generated"] / max(agg["elapsed"], 1e-6)
            print(f"\n[GEMMA4-EVAL] {bench}  α={mean_alpha:.4f}  "
                  f"tok/s={tps:.2f}  rounds={agg['rounds']}  "
                  f"generated={agg['generated']}")
            all_results[bench] = {
                "mean_alpha": mean_alpha,
                "tokens_per_sec": tps,
                "total_time": agg["elapsed"],
                "rounds": agg["rounds"],
                "generated": agg["generated"],
                "accepted": agg["accepted"],
                "gamma": args.gamma,
            }

    # ── Save ──
    out_path = os.path.join(args.output_dir, f"{args.tag}.json")
    with open(out_path, "w") as f:
        json.dump({"tag": args.tag,
                   "results": {b: {args.mode: v} for b, v in all_results.items()}},
                  f, indent=2)
    print(f"\n[GEMMA4-EVAL] Saved to {out_path}")


if __name__ == "__main__":
    main()
