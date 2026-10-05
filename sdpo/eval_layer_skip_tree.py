"""
Eval for Layer-Skip Tree Speculative Decoding.

Two settings:
  (A) Single-exit tree: draft uses one exit layer (real early exit for speed),
      generates a tree, target verifies. Compare speedup vs full model.
  (B) Multi-exit tree: draft uses multiple exits to generate a wider tree.
      More diverse candidates → potentially higher acceptance.

Measures: mean α (acceptance length), tokens/sec, wall-clock speedup,
draft vs target time breakdown.

Usage:
  # Setting A: single exit
  python sdpo/eval_layer_skip_tree.py \\
    --base-model-path Qwen/Qwen3-32B \\
    --draft-model-path Qwen/Qwen3-0.6B \\
    --mode single \\
    --exit-layers 20,24,26,28 \\
    --bench-name mt_bench \\
    --top-k 10 --budget 63

  # Setting B: multi-exit
  python sdpo/eval_layer_skip_tree.py \\
    --base-model-path Qwen/Qwen3-32B \\
    --draft-model-path Qwen/Qwen3-0.6B \\
    --mode multi \\
    --exit-layers 20,24,26,28 \\
    --bench-name mt_bench \\
    --top-k-per-exit 5 --budget 63

  # Also runs baseline: full-model chain spec decode (no tree, no early exit)
"""

import argparse
import glob
import json
import os
import sys
import time

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from tree_spec_decode import (
    build_draft_tree,
    build_multi_exit_tree,
    build_per_exit_trees,
    early_exit_context,
    spec_decode_tree,
    spec_decode_tree_multi_exit,
    spec_decode_tree_per_exit,
    verify_tree,
)


def load_questions(bench_name, data_dir="data"):
    path = os.path.join(data_dir, bench_name, "question.jsonl")
    if not os.path.exists(path):
        print(f"  Bench {bench_name} not found at {path}, skipping")
        return []
    with open(path) as f:
        return [json.loads(l) for l in f]


@torch.inference_mode()
def baseline_chain_decode(target_model, draft_model, input_ids,
                          max_new_tokens, gamma=7, force_sdpa_math=True):
    """Baseline: chain speculative decoding (no tree, full model, no early exit).

    If `force_sdpa_math=True`, passes a 4D causal mask to the target forward so
    SDPA falls back to the math backend (same path as tree verify). This makes
    the baseline vs tree comparison isolate the algorithmic effect only (both
    pay the same attention overhead). Set to False to let target use Flash
    Attention (faster but different attention path from tree).
    """
    device = input_ids.device
    cur_ids = input_ids.clone()
    total_tokens = 0
    total_rounds = 0
    total_accepted = 0
    t0 = time.time()

    while total_tokens < max_new_tokens:
        # Draft: generate γ tokens greedily
        draft_ids = cur_ids.clone()
        draft_tokens = []
        for _ in range(gamma):
            out = draft_model(draft_ids)
            next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
            draft_tokens.append(next_tok)
            draft_ids = torch.cat([draft_ids, next_tok], dim=1)
        draft_tokens = torch.cat(draft_tokens, dim=1)

        # Target: verify
        candidate = torch.cat([cur_ids, draft_tokens], dim=1)
        if force_sdpa_math:
            # Build 4D causal mask to force SDPA math backend (same as tree)
            tdtype = next(target_model.parameters()).dtype
            seq_len = candidate.shape[1]
            min_val = torch.finfo(tdtype).min
            arange = torch.arange(seq_len, device=device)
            causal = arange.unsqueeze(0) <= arange.unsqueeze(1)
            attn_mask = torch.where(
                causal,
                torch.zeros((), dtype=tdtype, device=device),
                torch.full((), min_val, dtype=tdtype, device=device),
            ).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            target_logits = target_model(candidate, attention_mask=attn_mask).logits
        else:
            target_logits = target_model(candidate).logits

        n_accepted = 0
        for i in range(gamma):
            pos = cur_ids.shape[1] + i - 1
            if target_logits[:, pos, :].argmax(-1).item() == draft_tokens[:, i].item():
                n_accepted += 1
            else:
                break

        correction_pos = cur_ids.shape[1] + n_accepted - 1
        correction = target_logits[:, correction_pos, :].argmax(-1, keepdim=True)
        new_tokens = torch.cat([draft_tokens[:, :n_accepted], correction], dim=1)

        cur_ids = torch.cat([cur_ids, new_tokens], dim=1)
        total_tokens += new_tokens.shape[1]
        total_rounds += 1
        total_accepted += n_accepted

        if hasattr(target_model.config, 'eos_token_id'):
            eos = target_model.config.eos_token_id
            eos_list = eos if isinstance(eos, list) else [eos]
            if any(t in new_tokens[0].tolist() for t in eos_list):
                break

    elapsed = time.time() - t0
    return {
        "total_tokens": total_tokens,
        "total_rounds": total_rounds,
        "total_accepted": total_accepted,
        "mean_alpha": total_accepted / max(total_rounds, 1),
        "tokens_per_sec": total_tokens / max(elapsed, 1e-6),
        "total_time": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model-path', required=True, help='Target LLM')
    parser.add_argument('--draft-model-path', required=True, help='Draft model or checkpoint')
    parser.add_argument('--mode', default='single',
                        choices=['single', 'multi', 'per_exit', 'both', 'all'],
                        help='single=one exit tree; multi=merged multi-exit tree (dedup); '
                             'per_exit=N independent trees, target verifies all; '
                             'both=single+multi; all=single+multi+per_exit')
    parser.add_argument('--exit-layers', required=True,
                        help='comma-separated exit layers (1-indexed)')
    parser.add_argument('--bench-name', default='mt_bench')
    parser.add_argument('--gamma', type=int, default=7, help='Tree depth')
    parser.add_argument('--top-k', type=int, default=10,
                        help='Branching factor per node (single-exit mode)')
    parser.add_argument('--top-k-per-exit', type=int, default=5,
                        help='Top-k per exit per node (multi-exit / per-exit modes)')
    parser.add_argument('--budget-per-exit', type=int, default=15,
                        help='Per-exit budget (per_exit mode). Total budget ≈ N × this.')
    parser.add_argument('--budget', type=int, default=63, help='Max tree nodes')
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Limit to first N samples per bench (default: all)')
    parser.add_argument('--tag', default='tree_eval')
    parser.add_argument('--output-dir', default='layerskip_tree_eval_results')
    parser.add_argument('--run-baseline', action='store_true',
                        help='Also run chain spec decode baseline for speed comparison')
    args = parser.parse_args()

    exit_layers = [int(e) for e in args.exit_layers.split(',') if e.strip()]

    benches = ['mt_bench', 'gsm8k', 'humaneval', 'qa', 'sum', 'alpaca', 'aime'] \
        if args.bench_name == 'all' else \
        [b.strip() for b in args.bench_name.split(',') if b.strip()]

    # Load models
    print(f"[TREE-EVAL] Target: {args.base_model_path}")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, torch_dtype=torch.float16,
        device_map="auto", attn_implementation="sdpa")
    target_model.eval()

    print(f"[TREE-EVAL] Draft: {args.draft_model_path}")
    # Fix checkpoint prefixes
    ckpt_files = glob.glob(os.path.join(args.draft_model_path, "*.bin"))
    for f in ckpt_files:
        state = torch.load(f, map_location="cpu")
        changed = False
        for prefix in ("draft.base.", "draft_model."):
            if any(k.startswith(prefix) for k in state):
                print(f"[TREE-EVAL] Stripping '{prefix}' from {f}")
                state = {k.removeprefix(prefix): v for k, v in state.items()}
                changed = True
        if changed:
            torch.save(state, f)

    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model_path, torch_dtype=torch.float16,
        attn_implementation="sdpa").cuda()
    draft_model.eval()

    num_layers = len(draft_model.model.layers)
    print(f"[TREE-EVAL] Draft has {num_layers} layers, exits={exit_layers}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path, trust_remote_code=True)

    eos_id = getattr(target_model.config, 'eos_token_id', None)

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    for bench in benches:
        questions = load_questions(bench)
        if not questions:
            continue
        if args.num_samples is not None:
            questions = questions[:args.num_samples]
            print(f"[TREE-EVAL] Limited to first {len(questions)} samples for {bench}")

        all_results[bench] = {}

        # ── Baseline: chain spec decode (full model, no tree) ──
        if args.run_baseline:
            print(f"\n[TREE-EVAL] {bench} baseline chain (full model)")
            bl_stats = {"total_tokens": 0, "total_rounds": 0, "total_accepted": 0,
                        "total_time": 0.0}
            for q in tqdm(questions, desc=f"baseline/{bench}"):
                prompt = q.get("turns", [q.get("prompt", "")])[0] \
                    if "turns" in q else q.get("prompt", "")
                msgs = [{"role": "user", "content": prompt}]
                try:
                    text = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True,
                        enable_thinking=True)
                except TypeError:
                    text = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True)
                input_ids = tokenizer(text, return_tensors="pt",
                                      add_special_tokens=False).input_ids.to(
                    target_model.device)

                r = baseline_chain_decode(
                    target_model, draft_model, input_ids,
                    args.max_new_tokens, args.gamma)
                for k in bl_stats:
                    bl_stats[k] += r[k]

            bl_alpha = bl_stats["total_accepted"] / max(bl_stats["total_rounds"], 1)
            bl_tps = bl_stats["total_tokens"] / max(bl_stats["total_time"], 1e-6)
            all_results[bench]["baseline_chain"] = {
                "mean_alpha": bl_alpha,
                "tokens_per_sec": bl_tps,
                "total_time": bl_stats["total_time"],
            }
            print(f"  baseline: α={bl_alpha:.3f}  tok/s={bl_tps:.1f}")

        # ── Setting A: single-exit tree ──
        if args.mode in ('single', 'both', 'all'):
            for exit_layer in exit_layers:
                key = f"single_exit_{exit_layer}"
                print(f"\n[TREE-EVAL] {bench} × single_exit={exit_layer} "
                      f"(top_k={args.top_k}, budget={args.budget})")

                agg = {"total_tokens": 0, "total_rounds": 0, "total_accepted": 0,
                       "total_time": 0.0, "draft_time": 0.0, "target_time": 0.0}

                for q in tqdm(questions, desc=f"single_e{exit_layer}/{bench}"):
                    prompt = q.get("turns", [q.get("prompt", "")])[0] \
                        if "turns" in q else q.get("prompt", "")
                    msgs = [{"role": "user", "content": prompt}]
                    try:
                        text = tokenizer.apply_chat_template(
                            msgs, tokenize=False, add_generation_prompt=True,
                            enable_thinking=True)
                    except TypeError:
                        text = tokenizer.apply_chat_template(
                            msgs, tokenize=False, add_generation_prompt=True)
                    input_ids = tokenizer(text, return_tensors="pt",
                                          add_special_tokens=False).input_ids.to(
                        target_model.device)

                    r = spec_decode_tree(
                        target_model, draft_model, input_ids,
                        args.max_new_tokens, exit_layer,
                        args.gamma, args.top_k, args.budget,
                        eos_token_id=eos_id)
                    for k in agg:
                        agg[k] += r[k]

                alpha = agg["total_accepted"] / max(agg["total_rounds"], 1)
                tps = agg["total_tokens"] / max(agg["total_time"], 1e-6)
                df = agg["draft_time"] / max(agg["total_time"], 1e-6)

                all_results[bench][key] = {
                    "mean_alpha": alpha,
                    "tokens_per_sec": tps,
                    "total_time": agg["total_time"],
                    "draft_time": agg["draft_time"],
                    "target_time": agg["target_time"],
                    "draft_fraction": df,
                    "exit_layer": exit_layer,
                    "layers_used": exit_layer,
                    "layers_total": num_layers,
                    "layer_ratio": exit_layer / num_layers,
                }
                print(f"  exit={exit_layer}: α={alpha:.3f}  tok/s={tps:.1f}  "
                      f"draft={df:.1%}  layers={exit_layer}/{num_layers}")

        # ── Setting C: per-exit independent trees ──
        if args.mode in ('per_exit', 'all'):
            key = f"per_exit_{'_'.join(map(str, exit_layers))}"
            print(f"\n[TREE-EVAL] {bench} × per_exit={exit_layers} "
                  f"(top_k_per_exit={args.top_k_per_exit}, "
                  f"budget_per_exit={args.budget_per_exit})")

            agg = {"total_tokens": 0, "total_rounds": 0, "total_accepted": 0,
                   "total_time": 0.0, "draft_time": 0.0, "target_time": 0.0}

            for q in tqdm(questions, desc=f"per_exit/{bench}"):
                prompt = q.get("turns", [q.get("prompt", "")])[0] \
                    if "turns" in q else q.get("prompt", "")
                msgs = [{"role": "user", "content": prompt}]
                try:
                    text = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True,
                        enable_thinking=True)
                except TypeError:
                    text = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True)
                input_ids = tokenizer(text, return_tensors="pt",
                                      add_special_tokens=False).input_ids.to(
                    target_model.device)

                r = spec_decode_tree_per_exit(
                    target_model, draft_model, input_ids,
                    args.max_new_tokens, exit_layers, num_layers,
                    args.gamma, args.top_k_per_exit, args.budget_per_exit,
                    eos_token_id=eos_id)
                for k in agg:
                    agg[k] += r[k]

            alpha = agg["total_accepted"] / max(agg["total_rounds"], 1)
            tps = agg["total_tokens"] / max(agg["total_time"], 1e-6)
            df = agg["draft_time"] / max(agg["total_time"], 1e-6)

            all_results[bench][key] = {
                "mean_alpha": alpha,
                "tokens_per_sec": tps,
                "total_time": agg["total_time"],
                "draft_time": agg["draft_time"],
                "target_time": agg["target_time"],
                "draft_fraction": df,
                "exit_layers": exit_layers,
                "top_k_per_exit": args.top_k_per_exit,
                "budget_per_exit": args.budget_per_exit,
            }
            print(f"  per_exit: α={alpha:.3f}  tok/s={tps:.1f}  draft={df:.1%}")

        # ── Setting B: multi-exit tree ──
        if args.mode in ('multi', 'both', 'all'):
            key = f"multi_exit_{'_'.join(map(str, exit_layers))}"
            print(f"\n[TREE-EVAL] {bench} × multi_exit={exit_layers} "
                  f"(top_k_per_exit={args.top_k_per_exit}, budget={args.budget})")

            agg = {"total_tokens": 0, "total_rounds": 0, "total_accepted": 0,
                   "total_time": 0.0, "draft_time": 0.0, "target_time": 0.0}

            for q in tqdm(questions, desc=f"multi/{bench}"):
                prompt = q.get("turns", [q.get("prompt", "")])[0] \
                    if "turns" in q else q.get("prompt", "")
                msgs = [{"role": "user", "content": prompt}]
                try:
                    text = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True,
                        enable_thinking=True)
                except TypeError:
                    text = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True)
                input_ids = tokenizer(text, return_tensors="pt",
                                      add_special_tokens=False).input_ids.to(
                    target_model.device)

                r = spec_decode_tree_multi_exit(
                    target_model, draft_model, input_ids,
                    args.max_new_tokens, exit_layers, num_layers,
                    args.gamma, args.top_k_per_exit, args.budget,
                    eos_token_id=eos_id)
                for k in agg:
                    agg[k] += r[k]

            alpha = agg["total_accepted"] / max(agg["total_rounds"], 1)
            tps = agg["total_tokens"] / max(agg["total_time"], 1e-6)
            df = agg["draft_time"] / max(agg["total_time"], 1e-6)

            all_results[bench][key] = {
                "mean_alpha": alpha,
                "tokens_per_sec": tps,
                "total_time": agg["total_time"],
                "draft_time": agg["draft_time"],
                "target_time": agg["target_time"],
                "draft_fraction": df,
                "exit_layers": exit_layers,
            }
            print(f"  multi_exit: α={alpha:.3f}  tok/s={tps:.1f}  draft={df:.1%}")

        # Save incrementally
        out_path = os.path.join(args.output_dir, f"{args.tag}.json")
        with open(out_path, 'w') as f:
            json.dump({"tag": args.tag, "exit_layers": exit_layers,
                       "config": {"gamma": args.gamma, "top_k": args.top_k,
                                  "top_k_per_exit": args.top_k_per_exit,
                                  "budget": args.budget,
                                  "max_new_tokens": args.max_new_tokens},
                       "results": all_results}, f, indent=2)

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  TREE SPEC DECODE SUMMARY: {args.tag}")
    for bench, per_mode in all_results.items():
        print(f"\n  {bench}:")
        for mode_key, stats in per_mode.items():
            alpha = stats['mean_alpha']
            tps = stats['tokens_per_sec']
            extra = ""
            if 'draft_fraction' in stats:
                extra = f"  draft={stats['draft_fraction']:.1%}"
            if 'layer_ratio' in stats:
                extra += f"  layers={stats['layers_used']}/{stats['layers_total']}"
            print(f"    {mode_key:<35} α={alpha:.3f}  tok/s={tps:.1f}{extra}")
    print(f"{'='*70}")
    print(f"[TREE-EVAL] Saved to {out_path}")


if __name__ == "__main__":
    main()
