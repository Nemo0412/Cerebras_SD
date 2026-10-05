"""
Prepare extended eval question.jsonl from HuggingFace test sets.

The default data/<bench>/question.jsonl files are 80-sample EAGLE3/Spec-Bench
subsets. This pulls full HF test splits to lower eval noise. Output goes to
data/<bench>_full/ to keep the 80-sample baselines intact.

Sources:
  - gsm8k_full      :  openai/gsm8k       'main' test split   (1319 problems)
  - humaneval_full  :  openai/openai_humaneval test split     ( 164 problems)
  - aime_full       :  HuggingFaceH4/aime_2024 + opencompass/AIME2025 train splits
                       (30 + 30 = 60 problems)

Output format matches eval_small_lm_tree.py's load_questions():
  {"question_id": int, "category": str, "turns": [str], "reference": [str]}

Usage:
  python sdpo/data/prepare_eval_extended.py                      # full size
  python sdpo/data/prepare_eval_extended.py --num-samples 500    # cap each bench
  python sdpo/data/prepare_eval_extended.py --benches gsm8k      # one bench only
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]


def write_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {path}  ({len(rows)} rows)")


def prepare_gsm8k(out_path, num_samples=None):
    print("Loading openai/gsm8k 'main' test split...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if num_samples is not None:
        ds = ds.select(range(min(num_samples, len(ds))))
    rows = [{
        "question_id": i,
        "category": "math",
        "turns": [ex["question"]],
        "reference": [ex["answer"]],
    } for i, ex in enumerate(ds)]
    write_jsonl(rows, out_path)
    return len(rows)


def prepare_humaneval(out_path, num_samples=None):
    print("Loading openai/openai_humaneval test split...")
    ds = load_dataset("openai/openai_humaneval", split="test")
    if num_samples is not None:
        ds = ds.select(range(min(num_samples, len(ds))))
    rows = [{
        "question_id": i,
        "category": "code",
        "turns": ["Complete the code I provided.\n\n" + ex["prompt"]],
        "reference": [ex["canonical_solution"]],
    } for i, ex in enumerate(ds)]
    write_jsonl(rows, out_path)
    return len(rows)


def prepare_aime(out_path, num_samples=None):
    print("Loading HuggingFaceH4/aime_2024 + opencompass/AIME2025 ...")
    rows = []
    qid = 0
    sources = [
        ("HuggingFaceH4/aime_2024", None, "train", 2024),
        ("opencompass/AIME2025", "AIME2025-I", "test", 2025),
        ("opencompass/AIME2025", "AIME2025-II", "test", 2025),
    ]
    for repo, config, split, year in sources:
        try:
            ds = (load_dataset(repo, config, split=split) if config
                  else load_dataset(repo, split=split))
        except Exception as e:
            print(f"  skip {repo} {config or ''}: {e}")
            continue
        for ex in ds:
            q = ex.get("problem") or ex.get("question") or ex.get("Problem")
            a = (ex.get("answer") or ex.get("Answer")
                 or ex.get("solution") or ex.get("Solution") or "")
            if q is None:
                continue
            rows.append({
                "question_id": qid,
                "category": f"math_aime_{year}",
                "turns": [str(q)],
                "reference": [str(a)],
            })
            qid += 1
    if num_samples is not None:
        rows = rows[:num_samples]
    write_jsonl(rows, out_path)
    return len(rows)


PREPARERS = {
    "gsm8k": prepare_gsm8k,
    "humaneval": prepare_humaneval,
    "aime": prepare_aime,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benches", nargs="+",
                        default=list(PREPARERS.keys()),
                        choices=list(PREPARERS.keys()))
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Cap each bench (default: full HF test split).")
    parser.add_argument("--suffix", default="_full",
                        help="Output dir suffix, e.g. gsm8k_full.")
    args = parser.parse_args()

    print(f"Suffix: {args.suffix}")
    print(f"Cap:    {args.num_samples or 'full set'}\n")

    for bench in args.benches:
        out_path = REPO_ROOT / "data" / (bench + args.suffix) / "question.jsonl"
        PREPARERS[bench](out_path, args.num_samples)

    print("\nDone. To eval against the full set:")
    for bench in args.benches:
        print(f"  --bench-name {bench}{args.suffix}  --num-samples <N>")


if __name__ == "__main__":
    main()
