"""Qwen3-8B replies for Alpaca prompts, written in the ShareGPT jsonl layout.

Prompts come from tatsu-lab/alpaca (instruction + optional input). Replies are
sampled from Qwen3-8B at T=1 with thinking enabled, the same chat template the
tree eval uses. Only replies that finished (</think> present, finish_reason
stop) are kept, because the Qwen3 template rewrites truncated think blocks.
"""
import argparse
import glob
import json
import os

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--shard", type=int, required=True)
parser.add_argument("--num_shards", type=int, default=4)
parser.add_argument("--out", required=True)
parser.add_argument("--max_tokens", type=int, default=1800)
parser.add_argument("--max_model_len", type=int, default=2560)
parser.add_argument("--limit", type=int, default=0)
args = parser.parse_args()

hf_home = os.environ.get("HF_HOME", "/mnt/ssd_ext/lls/cerebras_sd/hf")
pq = glob.glob(f"{hf_home}/hub/datasets--tatsu-lab--alpaca/snapshots/*/data/*.parquet")[0]
df = pd.read_parquet(pq)

prompts = []
for i, row in enumerate(df.itertuples(index=False)):
    text = row.instruction.strip()
    if isinstance(row.input, str) and row.input.strip():
        text = text + "\n\n" + row.input.strip()
    prompts.append((f"alpaca_{i}", text))

seen = set()
dedup = []
for pid, text in prompts:
    if text in seen:
        continue
    seen.add(text)
    dedup.append((pid, text))
shard = dedup[args.shard::args.num_shards]
if args.limit:
    shard = shard[:args.limit]
print(f"[gen] shard {args.shard}/{args.num_shards}: {len(shard)} prompts "
      f"(total unique {len(dedup)})", flush=True)

tok = AutoTokenizer.from_pretrained(args.model)
chat = []
for _pid, text in shard:
    msgs = [{"role": "user", "content": text}]
    chat.append(tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True))

llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
          gpu_memory_utilization=0.90, seed=1000 + args.shard,
          enable_prefix_caching=True)
sp = SamplingParams(temperature=1.0, top_p=1.0, top_k=-1,
                    max_tokens=args.max_tokens, seed=1000 + args.shard)

outs = llm.generate(chat, sp)

kept = dropped = 0
with open(args.out, "w") as f:
    for (pid, text), o in zip(shard, outs):
        c = o.outputs[0]
        reply = c.text
        if c.finish_reason != "stop" or "</think>" not in reply:
            dropped += 1
            continue
        rec = {
            "id": pid,
            "conversations": [
                {"from": "human", "value": text},
                {"from": "gpt", "value": reply},
            ],
            "meta": {"prompt_tokens": len(o.prompt_token_ids),
                     "reply_tokens": len(c.token_ids)},
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        kept += 1
print(f"[gen] shard {args.shard} kept={kept} dropped={dropped} -> {args.out}", flush=True)
