"""
Eval spec decoding accept length on long-context partial-prefix continuations.

Use case: each row of partial.jsonl contains a prompt + 10000-token partial
output (mid-thinking on AIME). We prefill BOTH target (Qwen3-8B) and draft
(0.6B ckpt) with prompt + partial_output, then spec-decode the continuation.
Measure mean acceptance length τ over the next max_new_tokens.

Usage:
    python sdpo/eval_partial_prefix.py \
        --partial-path /scratch/yf3005/lcrkv/longreason_partial/qwen3-8b_aime2025/partial.jsonl \
        --base-model-path Qwen/Qwen3-8B \
        --draft-model-path /scratch/tx856/.../q8_q06_klv4_l2k_grpo_smp/state_2 \
        --max-new-tokens 4096 \
        --tag aime2025_partial_grpo_smp \
        --run-baseline --run-tree
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


@torch.inference_mode()
def vanilla_ar_decode(target_model, input_ids, max_new_tokens, eos_token_id=None):
    """Plain greedy autoregressive decode with target model only — no SD.
    One forward pass per output token. Returns same dict shape as other decoders
    so the aggregator works. mean_alpha is reported as 1.0 (one token per round)
    purely for schema consistency; the meaningful number is tokens_per_sec.
    """
    device = input_ids.device
    cur_ids = input_ids.clone()
    prompt_len = cur_ids.shape[1]
    eos_list = (eos_token_id if isinstance(eos_token_id, list)
                else ([eos_token_id] if eos_token_id is not None else []))

    t0 = time.time()
    # Prefill once
    out = target_model(input_ids=cur_ids, use_cache=True, return_dict=True)
    past = out.past_key_values
    next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)

    generated = [next_tok.item()]
    cur_ids = torch.cat([cur_ids, next_tok], dim=1)

    for _ in range(max_new_tokens - 1):
        if generated[-1] in eos_list:
            break
        out = target_model(
            input_ids=next_tok, past_key_values=past,
            use_cache=True, return_dict=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
        generated.append(next_tok.item())
        cur_ids = torch.cat([cur_ids, next_tok], dim=1)

    elapsed = time.time() - t0
    return {
        "total_tokens": len(generated),
        "total_rounds": len(generated),
        "total_accepted": len(generated),
        "mean_alpha": 1.0,
        "tokens_per_sec": len(generated) / max(elapsed, 1e-6),
        "total_time": elapsed,
        "output_ids": generated,
    }


def load_partial(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def build_prefix_ids(tokenizer, row, device, max_partial_tokens=None):
    """Return [1, L] tensor of prompt_chat_text + partial_output_ids[:max_partial_tokens]."""
    prompt_ids = tokenizer(
        row["prompt_chat_text"],
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0].tolist()
    partial_ids = list(row["partial_output_ids"])
    if max_partial_tokens is not None:
        partial_ids = partial_ids[:max_partial_tokens]
    full = prompt_ids + partial_ids
    return torch.tensor([full], device=device, dtype=torch.long), \
        len(prompt_ids), len(partial_ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--partial-path", required=True)
    p.add_argument("--base-model-path", required=True)
    p.add_argument("--draft-model-path", required=True)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--max-partial-tokens", type=int, default=None,
                   help="Cap partial_output_ids length (e.g. 500 to make prefix short).")
    p.add_argument("--gamma", type=int, default=7,
                   help="Tree gamma (also baseline chain gamma if --baseline-gamma omitted)")
    p.add_argument("--top-k", default="4,3,2,1,1,1,1")
    p.add_argument("--budget", type=int, default=128)
    p.add_argument("--baseline-gamma", type=int, default=7)
    p.add_argument("--run-baseline", action="store_true")
    p.add_argument("--run-tree", action="store_true")
    p.add_argument("--run-vanilla", action="store_true",
                   help="Plain autoregressive decode with target only (no SD) — slowest, for speedup reference.")
    p.add_argument("--tag", default="partial_prefix")
    p.add_argument("--output-dir", default="smalllm_tree_eval_results")
    p.add_argument("--print-samples", type=int, default=0,
                   help="Print prefix metadata + decoded continuation for first N rows.")
    args = p.parse_args()

    if not (args.run_baseline or args.run_tree or args.run_vanilla):
        args.run_tree = True  # default behavior: tree only

    if "," in args.top_k:
        args.top_k = [int(k) for k in args.top_k.split(",") if k.strip()]
    else:
        args.top_k = int(args.top_k)

    print(f"[PARTIAL-EVAL] Target: {args.base_model_path}")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, torch_dtype=torch.float16,
        attn_implementation="sdpa").cuda().eval()

    print(f"[PARTIAL-EVAL] Draft:  {args.draft_model_path}")
    # Strip stale prefixes (same as eval_small_lm_tree)
    for f in glob.glob(os.path.join(args.draft_model_path, "*.bin")):
        state = torch.load(f, map_location="cpu")
        changed = False
        for prefix in ("draft.base.", "draft_model."):
            if any(k.startswith(prefix) for k in state):
                print(f"[PARTIAL-EVAL] Stripping '{prefix}' from {f}")
                state = {k.removeprefix(prefix): v for k, v in state.items()}
                changed = True
        if changed:
            torch.save(state, f)

    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model_path, torch_dtype=torch.float16,
        attn_implementation="sdpa").cuda().eval()

    num_layers = len(draft_model.model.layers)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path, trust_remote_code=True)

    eos_id = getattr(target_model.config, "eos_token_id", None)

    rows = load_partial(args.partial_path)
    if args.num_samples is not None:
        rows = rows[:args.num_samples]
    print(f"[PARTIAL-EVAL] {len(rows)} rows from {args.partial_path}")

    device = target_model.device
    results = {}
    os.makedirs(args.output_dir, exist_ok=True)

    if args.run_baseline:
        print(f"\n[PARTIAL-EVAL] baseline chain (gamma={args.baseline_gamma})")
        agg = {"total_tokens": 0, "total_rounds": 0,
               "total_accepted": 0, "total_time": 0.0}
        for ri, row in enumerate(tqdm(rows, desc="baseline")):
            prefix_ids, pl, partial_l = build_prefix_ids(tokenizer, row, device, args.max_partial_tokens)
            r = baseline_chain_decode(
                target_model, draft_model, prefix_ids,
                args.max_new_tokens, args.baseline_gamma)
            for k in agg:
                agg[k] += r[k]
            if ri < args.print_samples:
                gen_txt = tokenizer.decode(r["output_ids"], skip_special_tokens=False)
                print(f"\n  --- baseline sample {ri} id={row['id']} ---")
                print(f"  prefix tokens: {pl} (prompt) + {partial_l} (partial) = {pl+partial_l}")
                print(f"  α={r['mean_alpha']:.3f}  new_tok={len(r['output_ids'])}")
                print(f"  CONTINUATION:\n{gen_txt}\n")
        alpha = agg["total_accepted"] / max(agg["total_rounds"], 1)
        tps = agg["total_tokens"] / max(agg["total_time"], 1e-6)
        results["baseline_chain"] = {
            "mean_alpha": alpha, "tokens_per_sec": tps,
            "total_time": agg["total_time"],
        }
        print(f"  baseline: α={alpha:.3f}  tok/s={tps:.1f}")

    if args.run_vanilla:
        print(f"\n[PARTIAL-EVAL] vanilla AR (target-only, no SD)")
        agg = {"total_tokens": 0, "total_time": 0.0}
        for ri, row in enumerate(tqdm(rows, desc="vanilla")):
            prefix_ids, pl, partial_l = build_prefix_ids(tokenizer, row, device, args.max_partial_tokens)
            r = vanilla_ar_decode(
                target_model, prefix_ids, args.max_new_tokens,
                eos_token_id=eos_id)
            agg["total_tokens"] += r["total_tokens"]
            agg["total_time"] += r["total_time"]
            if ri < args.print_samples:
                gen_txt = tokenizer.decode(r["output_ids"], skip_special_tokens=False)
                print(f"\n  --- vanilla sample {ri} id={row['id']} ---")
                print(f"  prefix tokens: {pl} (prompt) + {partial_l} (partial) = {pl+partial_l}")
                print(f"  new_tok={len(r['output_ids'])} time={r['total_time']:.1f}s")
                print(f"  CONTINUATION:\n{gen_txt}\n")
        tps = agg["total_tokens"] / max(agg["total_time"], 1e-6)
        results["vanilla_ar"] = {
            "tokens_per_sec": tps,
            "total_time": agg["total_time"],
            "total_tokens": agg["total_tokens"],
        }
        print(f"  vanilla: tok/s={tps:.1f}  total_time={agg['total_time']:.1f}s")

    if args.run_tree:
        print(f"\n[PARTIAL-EVAL] tree decode "
              f"(γ={args.gamma}, top_k={args.top_k}, budget={args.budget})")
        agg = {"total_tokens": 0, "total_rounds": 0, "total_accepted": 0,
               "total_time": 0.0, "draft_time": 0.0, "target_time": 0.0}
        for ri, row in enumerate(tqdm(rows, desc="tree")):
            prefix_ids, pl, partial_l = build_prefix_ids(tokenizer, row, device, args.max_partial_tokens)
            r = spec_decode_tree_smalllm(
                target_model, draft_model, prefix_ids,
                args.max_new_tokens, num_layers,
                args.gamma, args.top_k, args.budget,
                eos_token_id=eos_id)
            for k in agg:
                agg[k] += r[k]
            if ri < args.print_samples:
                gen_txt = tokenizer.decode(r["output_ids"], skip_special_tokens=False)
                print(f"\n  --- tree sample {ri} id={row['id']} ---")
                print(f"  prefix tokens: {pl} (prompt) + {partial_l} (partial) = {pl+partial_l}")
                print(f"  α={r['mean_alpha']:.3f}  new_tok={len(r['output_ids'])}")
                print(f"  CONTINUATION:\n{gen_txt}\n")
        alpha = agg["total_accepted"] / max(agg["total_rounds"], 1)
        tps = agg["total_tokens"] / max(agg["total_time"], 1e-6)
        df = agg["draft_time"] / max(agg["total_time"], 1e-6)
        results["tree"] = {
            "mean_alpha": alpha, "tokens_per_sec": tps,
            "total_time": agg["total_time"],
            "draft_time": agg["draft_time"],
            "target_time": agg["target_time"],
            "draft_fraction": df,
            "gamma": args.gamma, "top_k": args.top_k, "budget": args.budget,
        }
        print(f"  tree: α={alpha:.3f}  tok/s={tps:.1f}  draft={df:.1%}")

    # Speedup summary if both vanilla and an SD mode were run
    if "vanilla_ar" in results:
        v_tps = results["vanilla_ar"]["tokens_per_sec"]
        for mode in ("baseline_chain", "tree"):
            if mode in results:
                s_tps = results[mode]["tokens_per_sec"]
                results[mode]["speedup_vs_vanilla"] = s_tps / v_tps
                print(f"  speedup {mode:>14s} vs vanilla: {s_tps/v_tps:.2f}x")

    out_path = os.path.join(args.output_dir, f"{args.tag}.json")
    with open(out_path, "w") as f:
        json.dump({"tag": args.tag, "partial_path": args.partial_path,
                   "n_rows": len(rows), "results": results}, f, indent=2)
    print(f"\n[PARTIAL-EVAL] Saved to {out_path}")


if __name__ == "__main__":
    main()
