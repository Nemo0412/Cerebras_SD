#!/usr/bin/env python3
"""
prepare_data.py

Builds mixed_train.jsonl + mixed_val.jsonl from four sources:
  ShareGPT (~74%), GSM8K train split (all 7473), HumanEval non-test (~84), Alpaca.

The 80 HumanEval test problems in data/humaneval/question.jsonl are excluded by
matching entry_point function names. After building, checks for overlap with
mt_bench_test.jsonl and humaneval/question.jsonl.

Output uses from/value ShareGPT format, compatible with sdpo/main.py build_dataset.
Defaults: 68000 train + ~1000 val.

Usage:
  python sdpo/data/prepare_data.py --output-dir sdpo/data
"""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from datasets import load_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sharegpt(row_id, human, assistant):
    return {
        "id": row_id,
        "conversations": [
            {"from": "human", "value": human},
            {"from": "gpt", "value": assistant},
        ],
    }


def _md5(s):
    return hashlib.md5(s.encode()).hexdigest()


def _split(items, n_train, n_val, rng):
    items = list(items)
    rng.shuffle(items)
    return items[n_val:n_val + n_train], items[:n_val]


def save_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"    → {path}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_sharegpt(n_train, n_val, seed):
    ds = list(load_dataset("Aeala/ShareGPT_Vicuna_unfiltered")["train"])
    valid = []
    for r in ds:
        convs = r.get("conversations", [])
        filtered = [c for c in convs if c.get("from") in ("human", "gpt")]
        if not filtered or filtered[0].get("from") != "human":
            continue
        row_id = r.get("id") or _md5(str(convs))
        valid.append({"id": row_id, "conversations": filtered})
    train, val = _split(valid, n_train, n_val, random.Random(seed))
    return train, val


def load_gsm8k(n_train, n_val, seed):
    ds = load_dataset("openai/gsm8k", "main")
    train_src = list(ds["train"])
    test_src = list(ds["test"])
    rng = random.Random(seed)
    rng.shuffle(train_src)
    rng.shuffle(test_src)

    def to_row(r):
        return _sharegpt(_md5(r["question"] + r["answer"]), r["question"], r["answer"])

    return [to_row(r) for r in train_src[:n_train]], [to_row(r) for r in test_src[:n_val]]


def load_humaneval(test_question_jsonl, seed):
    # Read function names from the existing test file; exclude matching problems from train.
    test_fn_names = set()
    path = Path(test_question_jsonl)
    if path.exists():
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                prompt = r["turns"][0] if "turns" in r else r.get("prompt", "")
                m = re.search(r"def (\w+)\(", prompt)
                if m:
                    test_fn_names.add(m.group(1))
        print(f"  HumanEval: excluding {len(test_fn_names)} test functions")
    else:
        print(f"  HumanEval: test file not found at {path}, no exclusion applied")

    all_items = list(load_dataset("openai/openai_humaneval")["test"])
    train_items = [r for r in all_items if r["entry_point"] not in test_fn_names]
    print(f"  HumanEval: {len(all_items)} total, {len(all_items)-len(train_items)} excluded, {len(train_items)} for train")

    def to_row(item):
        return _sharegpt(item["task_id"].replace("/", "_"), item["prompt"], item["canonical_solution"])

    rng = random.Random(seed)
    rng.shuffle(train_items)
    return [to_row(r) for r in train_items], []


def load_alpaca(n_train, n_val, seed):
    all_items = [r for r in load_dataset("tatsu-lab/alpaca")["train"] if r["output"].strip()]
    train, val = _split(all_items, n_train, n_val, random.Random(seed))

    def to_row(r):
        human = r["instruction"]
        if r["input"].strip():
            human = f"{human}\n\n{r['input']}"
        return _sharegpt(_md5(human + r["output"]), human, r["output"])

    return [to_row(r) for r in train], [to_row(r) for r in val]


# ---------------------------------------------------------------------------
# Contamination check
# ---------------------------------------------------------------------------

def _norm(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def check_contamination(train_path, test_files):
    # Normalize and hash every human turn in train, then check each test file for hits.
    print("\nRunning contamination check...")
    train_questions = set()
    with open(train_path) as f:
        for line in f:
            r = json.loads(line)
            convs = r.get("conversations", [])
            if convs:
                train_questions.add(_norm(convs[0]["value"]))

    total_hits = 0
    for test_path, key_fn in test_files:
        p = Path(test_path)
        if not p.exists():
            print(f"  SKIP {test_path} (not found)")
            continue
        hits = 0
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                q = key_fn(r)
                if q and _norm(q) in train_questions:
                    hits += 1
        status = "CLEAN" if hits == 0 else f"WARNING: {hits} overlaps"
        print(f"  {status:<30}  ← {p.name}")
        total_hits += hits

    if total_hits == 0:
        print("  No contamination detected.")
    else:
        print(f"  TOTAL {total_hits} contaminated samples — remove before training!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir",           default="sdpo/data")
    parser.add_argument("--humaneval-test-jsonl",  default="../../data/humaneval/question.jsonl")
    parser.add_argument("--sharegpt-train",  type=int, default=50000)
    parser.add_argument("--sharegpt-val",    type=int, default=500)
    parser.add_argument("--gsm8k-train",     type=int, default=7473)
    parser.add_argument("--gsm8k-val",       type=int, default=300)
    parser.add_argument("--alpaca-train",    type=int, default=10445)
    parser.add_argument("--alpaca-val",      type=int, default=200)
    parser.add_argument("--seed",            type=int, default=42)
    args = parser.parse_args()

    print("Loading ShareGPT...")
    sg_train, sg_val = load_sharegpt(args.sharegpt_train, args.sharegpt_val, args.seed)
    print(f"  {len(sg_train)} train, {len(sg_val)} val")

    print("Loading GSM8K...")
    gsm_train, gsm_val = load_gsm8k(args.gsm8k_train, args.gsm8k_val, args.seed)
    print(f"  {len(gsm_train)} train, {len(gsm_val)} val")

    print("Loading HumanEval...")
    he_train, _ = load_humaneval(args.humaneval_test_jsonl, args.seed)

    print("Loading Alpaca...")
    alp_train, alp_val = load_alpaca(args.alpaca_train, args.alpaca_val, args.seed)
    print(f"  {len(alp_train)} train, {len(alp_val)} val")

    rng = random.Random(args.seed)
    all_train = sg_train + gsm_train + he_train + alp_train
    all_val = sg_val + gsm_val + alp_val
    rng.shuffle(all_train)
    rng.shuffle(all_val)

    out_dir = Path(args.output_dir)
    print("\nSaving...")
    save_jsonl(all_train, out_dir / "mixed_train.jsonl")
    save_jsonl(all_val, out_dir / "mixed_val.jsonl")

    print(f"\n{'Dataset':<12} {'Train':>7} {'Val':>6}")
    print("-" * 28)
    for name, tr, vl in [
        ("ShareGPT",  len(sg_train),  len(sg_val)),
        ("GSM8K",     len(gsm_train), len(gsm_val)),
        ("HumanEval", len(he_train),  0),
        ("Alpaca",    len(alp_train), len(alp_val)),
    ]:
        print(f"  {name:<10} {tr:>7} {vl:>6}")
    print("-" * 28)
    print(f"  {'Total':<10} {len(all_train):>7} {len(all_val):>6}")
    print(f"\n  --trainpath {out_dir / 'mixed_train.jsonl'}")
    print(f"  --testpath  {out_dir / 'mixed_val.jsonl'}")

    check_contamination(
        out_dir / "mixed_train.jsonl",
        [
            (f"{out_dir}/mt_bench_test.jsonl",
             lambda r: r["turns"][0] if r.get("turns") else None),
            (args.humaneval_test_jsonl,
             lambda r: r["turns"][0] if r.get("turns") else r.get("prompt")),
        ],
    )


if __name__ == "__main__":
    main()
