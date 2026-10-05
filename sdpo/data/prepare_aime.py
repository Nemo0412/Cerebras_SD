"""
Prepare AIME dataset for training and evaluation.

Source: AI-MO/aimo-validation-aime (90 problems from AIME 2022-2024)
  - Train: problems from 2022 + 2023 (~60 problems)
  - Eval: problems from 2024 (~30 problems)

Outputs:
  - sdpo/data/aime_train.jsonl      (ShareGPT format for training)
  - data/aime/question.jsonl         (benchmark format for eval)

Also creates augmented training files:
  - sdpo/data/mixed_train_aime.jsonl     (mixed_train + aime_train)
  - sdpo/data/mixed_train_10K_aime.jsonl (mixed_train_10K + aime_train)

Usage:
  python sdpo/data/prepare_aime.py
"""

import json
import os
import re
from pathlib import Path

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
AIME_TRAIN_OUT = REPO_ROOT / "sdpo/data/aime_train.jsonl"
AIME_EVAL_OUT = REPO_ROOT / "data/aime/question.jsonl"
MIXED_FULL = REPO_ROOT / "sdpo/data/mixed_train.jsonl"
MIXED_10K = REPO_ROOT / "sdpo/data/mixed_train_10K.jsonl"
MIXED_FULL_AIME = REPO_ROOT / "sdpo/data/mixed_train_aime.jsonl"
MIXED_10K_AIME = REPO_ROOT / "sdpo/data/mixed_train_10K_aime.jsonl"


def to_sharegpt(row_id, problem, solution):
    return {
        "id": str(row_id),
        "conversations": [
            {"from": "human", "value": problem},
            {"from": "gpt", "value": solution},
        ],
    }


def main():
    print("Loading AI-MO/aimo-validation-aime...")
    ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
    print(f"  Total: {len(ds)} problems")

    train_rows = []
    eval_rows = []

    for i, r in enumerate(ds):
        year = re.search(r'(\d{4})_AIME', r["url"])
        year = year.group(1) if year else "unknown"

        if year == "2024":
            eval_rows.append({
                "question_id": int(r["id"]),
                "category": "aime",
                "turns": [r["problem"]],
                "answer": r["answer"],
            })
        else:
            train_rows.append(to_sharegpt(f"aime_{r['id']}", r["problem"], r["solution"]))

    print(f"  Train (2022-2023): {len(train_rows)}")
    print(f"  Eval  (2024):       {len(eval_rows)}")

    # Save AIME train (ShareGPT format)
    AIME_TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(AIME_TRAIN_OUT, "w") as f:
        for row in train_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote {AIME_TRAIN_OUT}  ({len(train_rows)} rows)")

    # Save AIME eval (benchmark format)
    AIME_EVAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(AIME_EVAL_OUT, "w") as f:
        for row in eval_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {AIME_EVAL_OUT}  ({len(eval_rows)} rows)")

    # Create augmented training files by concatenating
    def augment(base_path, out_path):
        if not base_path.exists():
            print(f"  SKIP augment: {base_path} not found")
            return
        n_base = 0
        with open(out_path, "w") as fout:
            with open(base_path) as fin:
                for line in fin:
                    fout.write(line)
                    n_base += 1
            for row in train_rows:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {out_path}  ({n_base} base + {len(train_rows)} aime = {n_base + len(train_rows)})")

    augment(MIXED_FULL, MIXED_FULL_AIME)
    augment(MIXED_10K, MIXED_10K_AIME)

    print("\nDone.")


if __name__ == "__main__":
    main()
