"""
Speedup-benchmark inference for tree spec decode with optimized kernels.

Imports the fast tree-spec-decode (GPU mask + GPU accept-path + optional
torch.compile) and compares against:
  - vanilla autoregressive (target only, no SD)
  - chain baseline SD (existing implementation)
  - tree SD (existing, unoptimized)
  - tree SD FAST (optimized)

Use case: measure speedup of #4 + #5 + #6 vs vanilla AR on partial-prefix
AIME data (long-context continuation).

Usage:
    python sdpo/eval_partial_prefix_fast.py \\
        --partial-path /scratch/yf3005/lcrkv/longreason_partial/qwen3-8b_aime2025/partial.jsonl \\
        --base-model-path Qwen/Qwen3-8B \\
        --draft-model-path /scratch/tx856/.../grpo_smp/state_2 \\
        --max-new-tokens 4096 --num-samples 1 \\
        --gamma 7 --top-k 4,3,2,1,1,1,1 --budget 128 \\
        --run-vanilla --run-tree --run-tree-fast
        # add --compile to enable torch.compile (#6)
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_small_lm_tree import (
    baseline_chain_decode,
    spec_decode_tree_smalllm,
)
from eval_partial_prefix import (
    vanilla_ar_decode,
    load_partial,
    build_prefix_ids,
)
from tree_spec_decode_fast import (
    spec_decode_tree_smalllm_fast,
    maybe_compile,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--partial-path", required=True)
    p.add_argument("--base-model-path", required=True)
    p.add_argument("--draft-model-path", required=True)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--max-partial-tokens", type=int, default=None,
                   help="Cap partial_output_ids length to shorten prefix.")
    p.add_argument("--gamma", type=int, default=7)
    p.add_argument("--top-k", default="4,3,2,1,1,1,1")
    p.add_argument("--budget", type=int, default=128)
    p.add_argument("--baseline-gamma", type=int, default=7)
    p.add_argument("--run-vanilla", action="store_true")
    p.add_argument("--run-baseline", action="store_true")
    p.add_argument("--run-tree", action="store_true",
                   help="Original (unoptimized) tree spec decode.")
    p.add_argument("--run-tree-fast", action="store_true",
                   help="Optimized tree spec decode (#4 GPU mask + #5 GPU accept-path).")
    p.add_argument("--compile", action="store_true",
                   help="(#6) Wrap draft model with torch.compile(mode='reduce-overhead') "
                        "to enable CUDA Graph capture inside inductor.")
    p.add_argument("--warmup-rounds", type=int, default=0,
                   help="Run N warmup decoding rounds before timed runs "
                        "(useful with --compile to skip first-call compile cost).")
    p.add_argument("--tag", default="partial_prefix_fast")
    p.add_argument("--output-dir", default="smalllm_tree_eval_results")
    p.add_argument("--print-samples", type=int, default=0)
    args = p.parse_args()

    if not (args.run_vanilla or args.run_baseline or args.run_tree
            or args.run_tree_fast):
        args.run_tree_fast = True

    if "," in args.top_k:
        args.top_k = [int(k) for k in args.top_k.split(",") if k.strip()]
    else:
        args.top_k = int(args.top_k)

    print(f"[FAST-EVAL] Target: {args.base_model_path}")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, torch_dtype=torch.float16,
        attn_implementation="sdpa").cuda().eval()

    print(f"[FAST-EVAL] Draft:  {args.draft_model_path}")
    for f in glob.glob(os.path.join(args.draft_model_path, "*.bin")):
        state = torch.load(f, map_location="cpu")
        changed = False
        for prefix in ("draft.base.", "draft_model."):
            if any(k.startswith(prefix) for k in state):
                print(f"[FAST-EVAL] Stripping '{prefix}' from {f}")
                state = {k.removeprefix(prefix): v for k, v in state.items()}
                changed = True
        if changed:
            torch.save(state, f)

    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model_path, torch_dtype=torch.float16,
        attn_implementation="sdpa").cuda().eval()

    if args.compile:
        print(f"[FAST-EVAL] torch.compile mode='reduce-overhead' on draft + target ...")
        draft_model = maybe_compile(draft_model, True)
        target_model = maybe_compile(target_model, True)

    num_layers = len(getattr(draft_model, "model",
                             draft_model).model.layers) \
        if hasattr(getattr(draft_model, "model", draft_model), "model") \
        else len(draft_model.model.layers)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path, trust_remote_code=True)
    eos_id = getattr(getattr(target_model, "_orig_mod", target_model).config,
                     "eos_token_id", None)

    rows = load_partial(args.partial_path)
    if args.num_samples is not None:
        rows = rows[:args.num_samples]
    print(f"[FAST-EVAL] {len(rows)} rows from {args.partial_path}")

    device = next(target_model.parameters()).device
    results = {}
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Warmup (if compile or just for fair timing) ──
    if args.warmup_rounds > 0 and rows:
        print(f"[FAST-EVAL] {args.warmup_rounds} warmup round(s) ...")
        warm_prefix, _, _ = build_prefix_ids(
            tokenizer, rows[0], device, args.max_partial_tokens)
        for _ in range(args.warmup_rounds):
            spec_decode_tree_smalllm_fast(
                target_model, draft_model, warm_prefix,
                max_new_tokens=32, exit_layer=num_layers,
                gamma=args.gamma, top_k=args.top_k, budget=args.budget,
                eos_token_id=None)
        torch.cuda.synchronize()

    def _run_mode(name, run_fn, agg_keys):
        print(f"\n[FAST-EVAL] {name}")
        agg = {k: 0.0 for k in agg_keys}
        for ri, row in enumerate(tqdm(rows, desc=name)):
            prefix_ids, pl, partial_l = build_prefix_ids(
                tokenizer, row, device, args.max_partial_tokens)
            r = run_fn(prefix_ids)
            for k in agg_keys:
                agg[k] += r[k]
            if ri < args.print_samples:
                gen_txt = tokenizer.decode(r["output_ids"],
                                           skip_special_tokens=False)
                print(f"\n  --- {name} sample {ri} id={row['id']} ---")
                print(f"  prefix tokens: {pl}+{partial_l}={pl+partial_l}")
                print(f"  new_tok={len(r['output_ids'])} "
                      f"time={r['total_time']:.1f}s")
                if 'mean_alpha' in r:
                    print(f"  α={r['mean_alpha']:.3f}")
                print(f"  CONTINUATION:\n{gen_txt[:1500]}{'...' if len(gen_txt) > 1500 else ''}\n")
        return agg

    if args.run_vanilla:
        agg = _run_mode("vanilla",
                        lambda ids: vanilla_ar_decode(
                            target_model, ids, args.max_new_tokens,
                            eos_token_id=eos_id),
                        ["total_tokens", "total_time"])
        tps = agg["total_tokens"] / max(agg["total_time"], 1e-6)
        results["vanilla_ar"] = {
            "tokens_per_sec": tps, "total_time": agg["total_time"],
            "total_tokens": agg["total_tokens"],
        }
        print(f"  vanilla: tok/s={tps:.1f}")

    if args.run_baseline:
        agg = _run_mode("baseline",
                        lambda ids: baseline_chain_decode(
                            target_model, draft_model, ids,
                            args.max_new_tokens, args.baseline_gamma),
                        ["total_tokens", "total_rounds",
                         "total_accepted", "total_time"])
        alpha = agg["total_accepted"] / max(agg["total_rounds"], 1)
        tps = agg["total_tokens"] / max(agg["total_time"], 1e-6)
        results["baseline_chain"] = {
            "mean_alpha": alpha, "tokens_per_sec": tps,
            "total_time": agg["total_time"],
        }
        print(f"  baseline: α={alpha:.3f}  tok/s={tps:.1f}")

    if args.run_tree:
        agg = _run_mode("tree (orig)",
                        lambda ids: spec_decode_tree_smalllm(
                            target_model, draft_model, ids,
                            args.max_new_tokens, num_layers,
                            args.gamma, args.top_k, args.budget,
                            eos_token_id=eos_id),
                        ["total_tokens", "total_rounds", "total_accepted",
                         "total_time", "draft_time", "target_time"])
        alpha = agg["total_accepted"] / max(agg["total_rounds"], 1)
        tps = agg["total_tokens"] / max(agg["total_time"], 1e-6)
        df = agg["draft_time"] / max(agg["total_time"], 1e-6)
        results["tree_orig"] = {
            "mean_alpha": alpha, "tokens_per_sec": tps,
            "total_time": agg["total_time"],
            "draft_fraction": df,
            "gamma": args.gamma, "top_k": args.top_k, "budget": args.budget,
        }
        print(f"  tree (orig): α={alpha:.3f}  tok/s={tps:.1f}  draft={df:.1%}")

    if args.run_tree_fast:
        agg = _run_mode("tree (fast #4+#5)",
                        lambda ids: spec_decode_tree_smalllm_fast(
                            target_model, draft_model, ids,
                            args.max_new_tokens, num_layers,
                            args.gamma, args.top_k, args.budget,
                            eos_token_id=eos_id),
                        ["total_tokens", "total_rounds", "total_accepted",
                         "total_time", "draft_time", "target_time"])
        alpha = agg["total_accepted"] / max(agg["total_rounds"], 1)
        tps = agg["total_tokens"] / max(agg["total_time"], 1e-6)
        df = agg["draft_time"] / max(agg["total_time"], 1e-6)
        results["tree_fast"] = {
            "mean_alpha": alpha, "tokens_per_sec": tps,
            "total_time": agg["total_time"],
            "draft_fraction": df,
            "gamma": args.gamma, "top_k": args.top_k, "budget": args.budget,
            "compiled": bool(args.compile),
        }
        print(f"  tree (fast): α={alpha:.3f}  tok/s={tps:.1f}  draft={df:.1%}")

    # ── Speedup summary ──
    if "vanilla_ar" in results:
        v_tps = results["vanilla_ar"]["tokens_per_sec"]
        for mode_key in ("baseline_chain", "tree_orig", "tree_fast"):
            if mode_key in results:
                s_tps = results[mode_key]["tokens_per_sec"]
                results[mode_key]["speedup_vs_vanilla"] = s_tps / v_tps
                print(f"  speedup {mode_key:>14s} vs vanilla: "
                      f"{s_tps/v_tps:.2f}x")
    if "tree_orig" in results and "tree_fast" in results:
        ratio = (results["tree_fast"]["tokens_per_sec"]
                 / results["tree_orig"]["tokens_per_sec"])
        results["tree_fast"]["speedup_vs_tree_orig"] = ratio
        print(f"  speedup tree_fast vs tree_orig: {ratio:.2f}x")

    out_path = os.path.join(args.output_dir, f"{args.tag}.json")
    with open(out_path, "w") as f:
        json.dump({
            "tag": args.tag,
            "partial_path": args.partial_path,
            "n_rows": len(rows),
            "config": {
                "gamma": args.gamma, "top_k": args.top_k,
                "budget": args.budget, "max_new_tokens": args.max_new_tokens,
                "max_partial_tokens": args.max_partial_tokens,
                "compile": bool(args.compile),
            },
            "results": results,
        }, f, indent=2)
    print(f"\n[FAST-EVAL] Saved to {out_path}")


if __name__ == "__main__":
    main()
