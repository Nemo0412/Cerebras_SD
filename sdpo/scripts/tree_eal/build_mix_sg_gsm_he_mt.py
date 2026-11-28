#!/usr/bin/env python3
"""Build a small ShareGPT + GSM8K + HumanEval + MT-Bench mix for fast KL=0 ablations.

Default size (~4.5k) targets Tree EAL (sample_rsd, n_anchors=8) × 3 epochs in <8h
on 2×A100/H200. MT-Bench turn-1 answers are regenerated with the target LM unless
--mt-regen already exists.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
from pathlib import Path


def reservoir_sharegpt(path: str, n: int, rng: random.Random):
    sg, n_seen = [], 0
    with open(path) as f:
        for line in f:
            o = json.loads(line)
            i = str(o.get("id", ""))
            if i.startswith("gsm8k") or "HumanEval" in i:
                continue
            n_seen += 1
            if len(sg) < n:
                sg.append(o)
            else:
                j = rng.randint(0, n_seen - 1)
                if j < n:
                    sg[j] = o
    return sg, n_seen


def regen_mt_bench(mtq_path: str, base: str, out_path: Path, max_new_tokens: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    questions = [json.loads(l) for l in open(mtq_path)]
    rows = []
    print(f"[mt] generating {len(questions)} turn-1 answers with {base}", flush=True)
    for o in questions:
        prompt = o["turns"][0]
        msgs = [{"role": "user", "content": prompt}]
        try:
            text = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        except TypeError:
            text = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **ids, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        rows.append({
            "id": f"mt_bench_{o['question_id']}",
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": gen},
            ],
            "source": "mt_bench",
        })
        print(f"  done {o['question_id']} gen_chars={len(gen)}", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[mt] wrote {out_path}", flush=True)
    del model
    torch.cuda.empty_cache()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sharegpt", required=True)
    p.add_argument("--gsm8k", required=True)
    p.add_argument("--humaneval", required=True)
    p.add_argument("--mt-bench-questions", required=True,
                   help="data/mt_bench/question.jsonl")
    p.add_argument("--base-model", default="",
                   help="Target LM for MT-Bench regen (required unless --mt-regen exists)")
    p.add_argument("--mt-regen", default="",
                   help="Optional existing MT-Bench regen JSONL; skip generation if set")
    p.add_argument("--n-sharegpt", type=int, default=3600)
    p.add_argument("--n-gsm8k", type=int, default=600)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mix-name", default="mix_sg_gsm_he_mt_4k5.jsonl")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mix_path = out_dir / args.mix_name
    meta_path = out_dir / (Path(args.mix_name).stem + ".meta.json")
    mt_regen = Path(args.mt_regen) if args.mt_regen else out_dir / "mt_bench_qwen3_8b_regen.jsonl"
    rng = random.Random(args.seed)

    if mt_regen.is_file():
        mt_rows = [json.loads(l) for l in open(mt_regen)]
        for o in mt_rows:
            o.setdefault("source", "mt_bench")
        print(f"[mt] reuse {mt_regen} n={len(mt_rows)}", flush=True)
    else:
        if not args.base_model:
            raise SystemExit("--base-model required when --mt-regen is missing")
        mt_rows = regen_mt_bench(
            args.mt_bench_questions, args.base_model, mt_regen, args.max_new_tokens)

    sg, n_seen = reservoir_sharegpt(args.sharegpt, args.n_sharegpt, rng)
    print(f"[sg] reservoir {len(sg)} / seen {n_seen}", flush=True)
    gsm = [json.loads(l) for l in open(args.gsm8k)]
    rng.shuffle(gsm)
    gsm = gsm[: args.n_gsm8k]
    he = [json.loads(l) for l in open(args.humaneval)]
    for o in sg:
        o["source"] = "sharegpt"
    for o in gsm:
        o["source"] = "gsm8k"
    for o in he:
        o["source"] = "humaneval"

    mix = sg + gsm + he + mt_rows
    rng.shuffle(mix)
    with open(mix_path, "w") as f:
        for o in mix:
            f.write(json.dumps({
                "id": o["id"],
                "conversations": o["conversations"],
                "source": o.get("source", ""),
            }, ensure_ascii=False) + "\n")
    ctr = collections.Counter(o.get("source") for o in mix)
    meta = {
        "path": str(mix_path),
        "n": len(mix),
        "sources": dict(ctr),
        "mt_regen": str(mt_regen),
        "seed": args.seed,
        "n_sharegpt": args.n_sharegpt,
        "n_gsm8k": args.n_gsm8k,
    }
    json.dump(meta, open(meta_path, "w"), indent=2)
    print("[mix]", meta, flush=True)


if __name__ == "__main__":
    main()
