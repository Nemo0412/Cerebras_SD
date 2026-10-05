"""
Microbenchmark for tree spec-decode upper-bound throughput.

Strips all overhead (mask construction, accept-path, Python control flow) and
times only the model forwards with FIXED shapes:
  - Vanilla AR: single target forward with shape [1, 1] per token
  - Tree round: γ × draft forward with shape [1, K] + 1 × target verify with
                shape [1, N+1]

All shapes are constant across rounds → torch.compile / CUDA Graph friendly.

Correctness is irrelevant — inputs are zeros, outputs ignored. Only measures
how fast the forwards can run in the best case.

Usage:
    python sdpo/bench_tree_static.py \\
        --target Qwen/Qwen3-8B --draft Qwen/Qwen3-0.6B \\
        --prefix_len 10000 --gamma 7 --k_per_depth 16 --tree_size 128 \\
        --n_rounds 50 [--compile]
"""
import argparse
import time

import torch
from transformers import AutoModelForCausalLM


def cuda_time(fn):
    torch.cuda.synchronize()
    t0 = time.time()
    fn()
    torch.cuda.synchronize()
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--prefix_len", type=int, default=10000)
    ap.add_argument("--gamma", type=int, default=7)
    ap.add_argument("--k_per_depth", type=int, default=16,
                    help="Fixed K-batched leaves per draft step "
                         "(constant across all γ depths for static shape).")
    ap.add_argument("--tree_size", type=int, default=128,
                    help="Fixed total tree nodes for verify step.")
    ap.add_argument("--n_rounds", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--alphas", default="4,5,6,7",
                    help="Comma-separated assumed accept rates for tok/s calc.")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile(mode='reduce-overhead') both models. "
                         "Enables internal CUDA Graph capture (fixed shapes only).")
    args = ap.parse_args()

    print(f"[bench] target={args.target}  draft={args.draft}")
    print(f"[bench] prefix_len={args.prefix_len}  γ={args.gamma}  "
          f"K={args.k_per_depth}  tree_size={args.tree_size}")

    target = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.float16,
        attn_implementation="sdpa").cuda().eval()
    draft = AutoModelForCausalLM.from_pretrained(
        args.draft, torch_dtype=torch.float16,
        attn_implementation="sdpa").cuda().eval()

    if args.compile:
        print("[bench] torch.compile(mode='reduce-overhead') ...")
        target = torch.compile(target, mode="reduce-overhead", fullgraph=False)
        draft = torch.compile(draft, mode="reduce-overhead", fullgraph=False)

    # ─── Prefill caches ───────────────────────────────────────────────
    prefix = torch.randint(0, 30000, (1, args.prefix_len), device='cuda')
    with torch.inference_mode():
        target_cache_v = target(prefix, use_cache=True,
                                return_dict=True).past_key_values
        target_cache_t = target(prefix, use_cache=True,
                                return_dict=True).past_key_values
        draft_cache = draft(prefix, use_cache=True,
                            return_dict=True).past_key_values
    print(f"[bench] prefill done, cache_len = {args.prefix_len}")

    K = args.k_per_depth
    G = args.gamma
    T = args.tree_size

    # ─── Static input buffers ─────────────────────────────────────────
    in_1 = torch.zeros(1, 1, dtype=torch.long, device='cuda')
    in_K = torch.zeros(1, K, dtype=torch.long, device='cuda')
    in_T = torch.zeros(1, T + 1, dtype=torch.long, device='cuda')

    # ─── Bench 1: vanilla AR (1-token target forward repeated) ────────
    print("\n[bench] vanilla AR warmup ...")
    with torch.inference_mode():
        for _ in range(args.warmup):
            o = target(input_ids=in_1, past_key_values=target_cache_v,
                       use_cache=True, return_dict=True)
            target_cache_v = o.past_key_values

    def run_vanilla():
        nonlocal target_cache_v
        with torch.inference_mode():
            for _ in range(args.n_rounds):
                o = target(input_ids=in_1, past_key_values=target_cache_v,
                           use_cache=True, return_dict=True)
                target_cache_v = o.past_key_values

    t_vanilla = cuda_time(run_vanilla)
    vanilla_step = t_vanilla / args.n_rounds
    vanilla_tps = 1.0 / vanilla_step
    print(f"[bench] vanilla AR: {vanilla_step * 1000:.2f} ms/step → "
          f"{vanilla_tps:.1f} tok/s")

    # ─── Bench 2: draft K-batched forward (alone) ─────────────────────
    print("\n[bench] draft K-batched warmup ...")
    with torch.inference_mode():
        for _ in range(args.warmup):
            od = draft(input_ids=in_K, past_key_values=draft_cache,
                       use_cache=True, return_dict=True)
            draft_cache = od.past_key_values

    def run_draft_only():
        nonlocal draft_cache
        with torch.inference_mode():
            for _ in range(args.n_rounds * G):  # G forwards per round
                od = draft(input_ids=in_K, past_key_values=draft_cache,
                           use_cache=True, return_dict=True)
                draft_cache = od.past_key_values

    t_draft = cuda_time(run_draft_only)
    draft_step = t_draft / (args.n_rounds * G)
    print(f"[bench] draft K-batched: {draft_step * 1000:.2f} ms/step "
          f"({G} steps/round)")

    # ─── Bench 3: target verify (TREE+1 tokens) ───────────────────────
    print("\n[bench] target verify warmup ...")
    with torch.inference_mode():
        for _ in range(args.warmup):
            ot = target(input_ids=in_T, past_key_values=target_cache_t,
                        use_cache=True, return_dict=True)
            target_cache_t = ot.past_key_values

    def run_target_verify():
        nonlocal target_cache_t
        with torch.inference_mode():
            for _ in range(args.n_rounds):
                ot = target(input_ids=in_T, past_key_values=target_cache_t,
                            use_cache=True, return_dict=True)
                target_cache_t = ot.past_key_values

    t_target = cuda_time(run_target_verify)
    target_verify_step = t_target / args.n_rounds
    print(f"[bench] target verify (N+1={T + 1}): "
          f"{target_verify_step * 1000:.2f} ms/round")

    # ─── Round time ──────────────────────────────────────────────────
    round_time = G * draft_step + target_verify_step
    print(f"\n[bench] tree round = γ × draft + verify = "
          f"{G} × {draft_step * 1000:.2f} + {target_verify_step * 1000:.2f} "
          f"= {round_time * 1000:.2f} ms/round")
    draft_frac = (G * draft_step) / round_time
    print(f"[bench] draft fraction = {draft_frac:.1%}, "
          f"target fraction = {1 - draft_frac:.1%}")

    # ─── Tok/s table for assumed α ───────────────────────────────────
    print(f"\n[bench] theoretical tok/s @ various α:")
    print(f"  {'α':>4}  {'tree tok/s':>11}  {'speedup vs vanilla':>20}")
    print(f"  {'-' * 4}  {'-' * 11}  {'-' * 20}")
    for alpha_str in args.alphas.split(","):
        alpha = float(alpha_str)
        tps_tree = alpha / round_time
        sp = tps_tree / vanilla_tps
        print(f"  {alpha:>4.1f}  {tps_tree:>11.1f}  {sp:>17.2f}x")

    print(f"\n[bench] caveats:")
    print("  - No attention_mask (default causal). Real tree decode adds 4D mask.")
    print("  - No KV-cache truncation. Real decode does index_select per round.")
    print("  - Inputs are zeros (no accept logic). Real decode walks tree paths.")
    print("  - These overheads are ~5-15% on top of the numbers shown.")


if __name__ == "__main__":
    main()
