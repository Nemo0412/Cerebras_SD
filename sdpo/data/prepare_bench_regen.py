"""Regen GSM8K train / HumanEval (non-test) with a target via vLLM.

Mirrors prepare_longwriter_regen.py: loads HF dataset, applies Qwen3
chat template with enable_thinking=True (default), runs vLLM, writes
ShareGPT-format JSONL with resume support.

HumanEval: excludes problems whose entry_point matches any function in
data/humaneval/question.jsonl (eval set), so train and test are disjoint.

Usage:
  python sdpo/data/prepare_bench_regen.py \
    --bench gsm8k --model Qwen/Qwen3-8B \
    --output sdpo/data/gsm8k_qwen3_8b_regen.jsonl
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", required=True, choices=["gsm8k", "humaneval"])
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--humaneval-test-jsonl",
                   default="data/humaneval/question.jsonl",
                   help="Used to exclude eval entry_points from HumanEval train.")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-tokens", type=int, default=4096,
                   help="Thinking + answer can be long for math/code.")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Match regen_sharegpt.py (0.7).")
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--gpu-mem-util", type=float, default=0.92)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--enable-thinking", action="store_true", default=True)
    p.add_argument("--no-thinking", dest="enable_thinking", action="store_false")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def load_gsm8k_train():
    """Returns list of (id, user_prompt) tuples from GSM8K train split."""
    ds = load_dataset("openai/gsm8k", "main", split="train")
    items = []
    for i, row in enumerate(ds):
        items.append((f"gsm8k_train_{i:05d}", row["question"]))
    return items


def load_humaneval_nontest(test_jsonl):
    """Returns list of (id, user_prompt) excluding entry_points found in
    test_jsonl (data/humaneval/question.jsonl)."""
    test_fn_names = set()
    p = Path(test_jsonl)
    if p.exists():
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                prompt = r["turns"][0] if "turns" in r else r.get("prompt", "")
                m = re.search(r"def (\w+)\(", prompt)
                if m:
                    test_fn_names.add(m.group(1))
        print(f"[REGEN] HumanEval excluding {len(test_fn_names)} test entry_points")
    else:
        print(f"[REGEN] WARN: no test file at {p}, no exclusion applied")
    ds = load_dataset("openai/openai_humaneval", split="test")
    items = []
    for row in ds:
        if row["entry_point"] in test_fn_names:
            continue
        # ShareGPT-style: ask the model to complete the function.
        user_msg = f"Complete the code I provided.\n\n{row['prompt']}"
        items.append((row["task_id"].replace("/", "_"), user_msg))
    return items


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    # ── Load benchmark ──
    if args.bench == "gsm8k":
        items = load_gsm8k_train()
    else:
        items = load_humaneval_nontest(args.humaneval_test_jsonl)
    if args.num_samples:
        items = items[: args.num_samples]
    print(f"[REGEN] {args.bench}: {len(items)} samples")

    # ── Build prompts ──
    print(f"[REGEN] Loading tokenizer for {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    prompts, ids, raw_users = [], [], []
    for oid, raw_user in items:
        single_turn = [{"role": "user", "content": raw_user}]
        try:
            text = tokenizer.apply_chat_template(
                single_turn, tokenize=False, add_generation_prompt=True,
                enable_thinking=args.enable_thinking)
        except TypeError:
            text = tokenizer.apply_chat_template(
                single_turn, tokenize=False, add_generation_prompt=True)
        prompts.append(text)
        ids.append(oid)
        raw_users.append(raw_user)
    print(f"[REGEN] Built {len(prompts)} prompts (thinking={args.enable_thinking})")

    # ── Resume ──
    done_ids = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
        print(f"[REGEN] Resume: {len(done_ids)} ids already done")
    keep = [i for i, x in enumerate(ids) if x not in done_ids]
    if not keep:
        print("[REGEN] All done, exiting.")
        return
    prompts = [prompts[i] for i in keep]
    ids = [ids[i] for i in keep]
    raw_users = [raw_users[i] for i in keep]
    print(f"[REGEN] {len(prompts)} todo")

    # ── vLLM ──
    print(f"[REGEN] Loading vLLM ({args.model}, tp={args.tensor_parallel_size}, "
          f"max_model_len={args.max_model_len})")
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

    CHUNK = 128
    t0 = time.time()
    written = 0
    with open(args.output, "a") as fout:
        for s in range(0, len(prompts), CHUNK):
            e = min(s + CHUNK, len(prompts))
            outs = llm.generate(prompts[s:e], sampling)
            for out, oid, raw_user in zip(outs, ids[s:e], raw_users[s:e]):
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
            print(f"[REGEN] {written}/{len(prompts)} done  "
                  f"elapsed={elapsed:.0f}s  rate={written/max(elapsed,1):.1f}/s")

    print(f"[REGEN] Wrote {args.output} ({written} new records)")


if __name__ == "__main__":
    main()
