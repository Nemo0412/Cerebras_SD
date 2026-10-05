"""Generate Qwen3-8B regen on LongWriter-6k prompts → ShareGPT JSONL.

For each sample in zai-org/LongWriter-6k:
  1. Extract the first user-role message as prompt.
  2. Apply Qwen3 chat template with enable_thinking=True.
  3. vLLM generate (greedy, no max_new_tokens cap — uses model's max context).
  4. Save {"id": ..., "conversations": [human, gpt]} to JSONL.

Output matches the format of sdpo/data/sharegpt_qwen3_8b_regen.jsonl.

Usage:
  python sdpo/data/prepare_longwriter_regen.py \\
    --model Qwen/Qwen3-8B \\
    --output sdpo/data/longwriter_qwen3_8b_regen.jsonl
"""
import argparse
import json
import os
import time

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--dataset", default="zai-org/LongWriter-6k")
    p.add_argument("--split", default="train")
    p.add_argument("--output", required=True)
    p.add_argument("--max-model-len", type=int, default=32768,
                   help="Max prompt+generate length (Qwen3-8B context = 32k or 128k).")
    p.add_argument("--max-tokens", type=int, default=32000,
                   help="Max generation tokens. Will run until EOS or this cap.")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 = greedy (matches existing regen dataset).")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--gpu-mem-util", type=float, default=0.92)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--max-num-seqs", type=int, default=64,
                   help="vLLM concurrent sequence cap; reduce for long outputs.")
    p.add_argument("--num-samples", type=int, default=None,
                   help="Subsample for testing; None = all.")
    p.add_argument("--enable-thinking", action="store_true", default=True,
                   help="Qwen3 thinking mode (default True for regen consistency).")
    p.add_argument("--no-thinking", dest="enable_thinking", action="store_false")
    p.add_argument("--resume", action="store_true",
                   help="Skip prompts whose id is already in output (resume).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".",
                exist_ok=True)

    # ── Load dataset ──
    print(f"[REGEN] Loading dataset {args.dataset} split={args.split}")
    ds = load_dataset(args.dataset, split=args.split)
    if args.num_samples:
        ds = ds.select(range(min(args.num_samples, len(ds))))
    print(f"[REGEN] {len(ds)} samples")

    # ── Build prompts ──
    print(f"[REGEN] Loading tokenizer for {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    prompts = []
    ids = []
    raw_messages = []
    for i, row in enumerate(ds):
        msgs = row.get("messages") or []
        # Keep only the FIRST user turn as the single-turn prompt
        user_msg = next((m for m in msgs if m.get("role") == "user"), None)
        if user_msg is None:
            continue
        single_turn = [{"role": "user", "content": user_msg["content"]}]
        try:
            text = tokenizer.apply_chat_template(
                single_turn, tokenize=False, add_generation_prompt=True,
                enable_thinking=args.enable_thinking)
        except TypeError:
            text = tokenizer.apply_chat_template(
                single_turn, tokenize=False, add_generation_prompt=True)
        prompts.append(text)
        ids.append(f"longwriter_{i:05d}")
        raw_messages.append(user_msg["content"])
    print(f"[REGEN] Built {len(prompts)} prompts (thinking={args.enable_thinking})")

    # ── Resume support: skip already-done ids ──
    done_ids = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
        print(f"[REGEN] Resume mode: {len(done_ids)} ids already in {args.output}")

    keep_idx = [i for i, x in enumerate(ids) if x not in done_ids]
    if not keep_idx:
        print(f"[REGEN] All samples already done, exiting.")
        return
    prompts_todo = [prompts[i] for i in keep_idx]
    ids_todo = [ids[i] for i in keep_idx]
    raw_todo = [raw_messages[i] for i in keep_idx]
    print(f"[REGEN] {len(prompts_todo)} samples to generate")

    # ── Init vLLM ──
    print(f"[REGEN] Loading vLLM ({args.model}, "
          f"tp={args.tensor_parallel_size}, max_model_len={args.max_model_len})")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=True,
    )

    # ── Sampling params ──
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # ── Generate (streaming write — flush per chunk so partial output is safe) ──
    CHUNK = 64
    t0 = time.time()
    written = 0
    with open(args.output, "a") as fout:
        for s in range(0, len(prompts_todo), CHUNK):
            e = min(s + CHUNK, len(prompts_todo))
            batch_prompts = prompts_todo[s:e]
            batch_ids = ids_todo[s:e]
            batch_raw = raw_todo[s:e]
            outputs = llm.generate(batch_prompts, sampling)
            for out, oid, raw_user in zip(outputs, batch_ids, batch_raw):
                gen = out.outputs[0].text
                record = {
                    "id": oid,
                    "conversations": [
                        {"from": "human", "value": raw_user},
                        {"from": "gpt", "value": gen},
                    ],
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
            fout.flush()
            elapsed = time.time() - t0
            print(f"[REGEN] {written}/{len(prompts_todo)} done  "
                  f"elapsed={elapsed:.0f}s  "
                  f"rate={written/max(elapsed,1):.1f} samples/s")

    print(f"[REGEN] Wrote {args.output} ({written} new records)")


if __name__ == "__main__":
    main()
