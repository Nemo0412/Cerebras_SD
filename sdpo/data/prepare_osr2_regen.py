"""Generate Qwen3-8B regen on nvidia/OpenScienceReasoning-2 prompts → ShareGPT JSONL.

For each sample:
  1. Use `input` field as prompt (multiple-choice / open-ended science question).
  2. Apply Qwen3 chat template (thinking on).
  3. vLLM generate (greedy by default).
  4. Save {"id": ..., "conversations": [human, gpt]} to JSONL.

Streaming load (avoids downloading the full 15GB parquet when only 10k needed).

Usage:
  python sdpo/data/prepare_osr2_regen.py \\
    --model Qwen/Qwen3-8B \\
    --num-samples 10000 \\
    --output sdpo/data/osr2_qwen3_8b_regen.jsonl
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
    p.add_argument("--dataset", default="nvidia/OpenScienceReasoning-2")
    p.add_argument("--split", default="train")
    p.add_argument("--prompt-field", default="input")
    p.add_argument("--output", required=True)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--max-tokens", type=int, default=32000)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--gpu-mem-util", type=float, default=0.92)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--max-num-seqs", type=int, default=16)
    p.add_argument("--num-samples", type=int, default=10000)
    p.add_argument("--enable-thinking", action="store_true", default=True)
    p.add_argument("--no-thinking", dest="enable_thinking", action="store_false")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".",
                exist_ok=True)

    # ── Streaming load (first N rows only) ──
    print(f"[REGEN] Streaming {args.dataset} split={args.split}, fetching {args.num_samples}")
    stream = load_dataset(args.dataset, split=args.split, streaming=True)

    print(f"[REGEN] Loading tokenizer for {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    prompts = []
    ids = []
    raw_prompts = []
    it = iter(stream)
    for i in range(args.num_samples):
        try:
            row = next(it)
        except StopIteration:
            break
        text_raw = row.get(args.prompt_field)
        if not text_raw or not isinstance(text_raw, str):
            continue
        msg = [{"role": "user", "content": text_raw}]
        try:
            text = tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True,
                enable_thinking=args.enable_thinking)
        except TypeError:
            text = tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True)
        prompts.append(text)
        ids.append(f"osr2_{i:06d}")
        raw_prompts.append(text_raw)
    print(f"[REGEN] Built {len(prompts)} prompts (thinking={args.enable_thinking})")

    # ── Resume support ──
    done_ids = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
        print(f"[REGEN] Resume: {len(done_ids)} done already")

    keep_idx = [i for i, x in enumerate(ids) if x not in done_ids]
    if not keep_idx:
        print(f"[REGEN] All done, exiting.")
        return
    prompts_todo = [prompts[i] for i in keep_idx]
    ids_todo = [ids[i] for i in keep_idx]
    raw_todo = [raw_prompts[i] for i in keep_idx]
    print(f"[REGEN] {len(prompts_todo)} samples to generate")

    # ── vLLM init ──
    print(f"[REGEN] Loading vLLM ({args.model})")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

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
                  f"rate={written/max(elapsed,1):.2f} samples/s  "
                  f"eta={(len(prompts_todo)-written)/max(written/max(elapsed,1),1e-3):.0f}s")

    print(f"[REGEN] Wrote {args.output} ({written} new records)")


if __name__ == "__main__":
    main()
