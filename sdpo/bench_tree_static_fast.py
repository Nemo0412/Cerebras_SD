"""
Upper-bound microbench using StaticCache (no DynamicCache.cat overhead).

Replaces DynamicCache (which torch.cat's a new tensor every layer per forward)
with HF transformers' StaticCache (pre-allocated, in-place writes via
cache_position). Forward shapes stay fixed across rounds.

This isolates the DynamicCache overhead from the SDPA kernel overhead. Compare
output to bench_tree_static.py (DynamicCache) to see the gap.

Optional --attn-impl flash_attention_2 (requires `pip install flash-attn`).
Default sdpa.

Usage:
    python sdpo/bench_tree_static_fast.py \\
        --target Qwen/Qwen3-8B --draft Qwen/Qwen3-0.6B \\
        --prefix_len 10000 --gamma 7 --k_per_depth 16 --tree_size 128 \\
        --n_rounds 30 --warmup 5 [--attn-impl flash_attention_2]
"""
import argparse
import time

import torch
from transformers import AutoModelForCausalLM, StaticCache


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
    ap.add_argument("--k_per_depth", type=int, default=16)
    ap.add_argument("--tree_size", type=int, default=128)
    ap.add_argument("--n_rounds", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--alphas", default="4,5,6,7")
    ap.add_argument("--vanilla_k", type=int, default=None,
                    help="Batch size B for vanilla AR baseline (B independent "
                         "sequences in parallel, each generates 1 token/forward). "
                         "Default = k_per_depth. Per-sequence per-token latency "
                         "= forward time; this is what's used to compute SPEEDUP "
                         "(matches what one sequence sees in a B-batched server).")
    ap.add_argument("--attn_impl", default="sdpa",
                    choices=["sdpa", "flash_attention_2", "eager"])
    args = ap.parse_args()

    PL = args.prefix_len
    G = args.gamma
    K = args.k_per_depth
    T = args.tree_size
    NR = args.n_rounds
    VK = args.vanilla_k if args.vanilla_k is not None else K

    print(f"[bench-fast] target={args.target}  draft={args.draft}")
    print(f"[bench-fast] prefix_len={PL}  γ={G}  K={K}  tree_size={T}")
    print(f"[bench-fast] attn_impl={args.attn_impl}")

    # Pre-compute max cache sizes (cache grows across rounds in this bench)
    max_kv_vanilla = PL + (NR + args.warmup) * VK + 1024
    max_kv_draft = PL + (NR + args.warmup) * G * K + 1024
    max_kv_target_t = PL + (NR + args.warmup) * (T + 1) + 1024

    target = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.float16,
        attn_implementation=args.attn_impl).cuda().eval()
    draft = AutoModelForCausalLM.from_pretrained(
        args.draft, torch_dtype=torch.float16,
        attn_implementation=args.attn_impl).cuda().eval()

    print(f"[bench-fast] allocating StaticCaches: "
          f"vanilla_target={max_kv_vanilla}, draft={max_kv_draft}, "
          f"verify_target={max_kv_target_t}")

    target_cache_v = StaticCache(
        config=target.config, max_batch_size=VK, max_cache_len=max_kv_vanilla,
        device="cuda", dtype=torch.float16)
    target_cache_t = StaticCache(
        config=target.config, max_batch_size=1, max_cache_len=max_kv_target_t,
        device="cuda", dtype=torch.float16)
    draft_cache = StaticCache(
        config=draft.config, max_batch_size=1, max_cache_len=max_kv_draft,
        device="cuda", dtype=torch.float16)

    # ─── Prefill once into each cache ──────────────────────────────────
    prefix = torch.randint(0, 30000, (1, PL), device="cuda")
    prefix_batched = prefix.expand(VK, -1).contiguous()  # [VK, PL] for vanilla
    cache_pos_prefix = torch.arange(PL, device="cuda", dtype=torch.long)
    with torch.inference_mode():
        target(input_ids=prefix_batched, past_key_values=target_cache_v,
               cache_position=cache_pos_prefix, use_cache=True)
        target(input_ids=prefix, past_key_values=target_cache_t,
               cache_position=cache_pos_prefix, use_cache=True)
        draft(input_ids=prefix, past_key_values=draft_cache,
              cache_position=cache_pos_prefix, use_cache=True)
    cur_v = PL
    cur_t = PL
    cur_d = PL
    print(f"[bench-fast] prefill done, cache_len={PL}")

    # Static input + position buffers
    in_vanilla_batched = torch.zeros(VK, 1, dtype=torch.long, device="cuda")
    in_K = torch.zeros(1, K, dtype=torch.long, device="cuda")
    in_T = torch.zeros(1, T + 1, dtype=torch.long, device="cuda")
    pos_off_K = torch.arange(K, device="cuda", dtype=torch.long)
    pos_off_T = torch.arange(T + 1, device="cuda", dtype=torch.long)

    # ─── Vanilla AR baseline (batch_size=VK, seq=1 — VK independent sequences) ──
    def step_vanilla():
        nonlocal cur_v
        pos = torch.tensor([cur_v], device="cuda", dtype=torch.long)
        with torch.inference_mode():
            target(input_ids=in_vanilla_batched, past_key_values=target_cache_v,
                   cache_position=pos, use_cache=True)
        cur_v += 1

    print(f"\n[bench-fast] vanilla AR (batch={VK}, seq=1) warmup ...")
    for _ in range(args.warmup):
        step_vanilla()

    def run_vanilla():
        for _ in range(NR):
            step_vanilla()

    t_v = cuda_time(run_vanilla)
    vanilla_step = t_v / NR
    # Per-sequence sees 1 token per forward. Per-sequence per-token latency = step time.
    # System-wide throughput = VK / step_time.
    vanilla_per_seq_tps = 1.0 / vanilla_step
    vanilla_system_tps = VK / vanilla_step
    print(f"[bench-fast] vanilla AR (batch={VK}, seq=1): {vanilla_step * 1000:.2f} ms/forward")
    print(f"  → per-sequence: {vanilla_per_seq_tps:.1f} tok/s/seq "
          f"({vanilla_step * 1000:.2f} ms/tok)")
    print(f"  → system total: {vanilla_system_tps:.1f} tok/s ({VK} sequences in parallel)")

    # ─── Draft K-batched ──────────────────────────────────────────────
    def step_draft():
        nonlocal cur_d
        pos = pos_off_K + cur_d
        with torch.inference_mode():
            draft(input_ids=in_K, past_key_values=draft_cache,
                  cache_position=pos, use_cache=True)
        cur_d += K

    print("\n[bench-fast] draft K-batched warmup ...")
    for _ in range(args.warmup):
        step_draft()

    def run_draft():
        for _ in range(NR * G):
            step_draft()

    t_d = cuda_time(run_draft)
    draft_step = t_d / (NR * G)
    print(f"[bench-fast] draft K-batched: {draft_step * 1000:.2f} ms/step "
          f"({G} steps/round)")

    # ─── Target verify (TREE+1 tokens) ────────────────────────────────
    def step_target():
        nonlocal cur_t
        pos = pos_off_T + cur_t
        with torch.inference_mode():
            target(input_ids=in_T, past_key_values=target_cache_t,
                   cache_position=pos, use_cache=True)
        cur_t += T + 1

    print("\n[bench-fast] target verify warmup ...")
    for _ in range(args.warmup):
        step_target()

    def run_target():
        for _ in range(NR):
            step_target()

    t_t = cuda_time(run_target)
    target_step = t_t / NR
    print(f"[bench-fast] target verify (N+1={T + 1}): "
          f"{target_step * 1000:.2f} ms/round")

    # ─── Round time + tok/s ───────────────────────────────────────────
    round_time = G * draft_step + target_step
    draft_frac = G * draft_step / round_time
    print(f"\n[bench-fast] tree round = {G} × {draft_step * 1000:.2f} + "
          f"{target_step * 1000:.2f} = {round_time * 1000:.2f} ms/round")
    print(f"[bench-fast] draft fraction = {draft_frac:.1%}, "
          f"target fraction = {1 - draft_frac:.1%}")

    print(f"\n[bench-fast] tree per-sequence latency = round_time / α")
    print(f"[bench-fast] vanilla per-sequence latency = {vanilla_step * 1000:.2f} ms/tok "
          f"(batch={VK}, each sequence sees 1 forward per token)")
    print(f"\n[bench-fast] theoretical per-sequence latency @ various α:")
    print(f"  {'α':>4}  {'tree ms/tok':>11}  {'tree tok/s':>11}  "
          f"{'speedup vs vanilla':>20}")
    print(f"  {'-' * 4}  {'-' * 11}  {'-' * 11}  {'-' * 20}")
    for alpha_str in args.alphas.split(","):
        alpha = float(alpha_str)
        tree_per_tok = round_time / alpha
        tps_tree = alpha / round_time
        sp = vanilla_step / tree_per_tok
        print(f"  {alpha:>4.1f}  {tree_per_tok * 1000:>9.2f}    {tps_tree:>11.1f}  "
              f"{sp:>17.2f}x")


if __name__ == "__main__":
    main()
