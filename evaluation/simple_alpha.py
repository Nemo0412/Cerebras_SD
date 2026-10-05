"""Compute average acceptance length (alpha) from EAGLE generation jsonl files.

Alpha = total new tokens / number of EAGLE rounds per turn.
This includes the mandatory +1 sampled token per round, so alpha >= 1.

Usage:
    python3 simple_alpha.py --file path/to/output.jsonl
    python3 simple_alpha.py --file a.jsonl b.jsonl c.jsonl
"""
import argparse
import json
import numpy as np


def compute_alpha(jsonl_file):
    data = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    alphas = []
    for d in data:
        for choice in d["choices"]:
            for new_tok, idx in zip(choice["new_tokens"], choice["idxs"]):
                alphas.append(new_tok / (idx + 1))

    return np.array(alphas)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, nargs="+", required=True,
                        help="One or more jsonl files to compute alpha for.")
    args = parser.parse_args()

    for path in args.file:
        alphas = compute_alpha(path)
        print(f"{path}")
        print(f"  mean alpha : {alphas.mean():.3f}")
        print(f"  std        : {alphas.std():.3f}")
        print(f"  min / max  : {alphas.min():.3f} / {alphas.max():.3f}")
        print(f"  n turns    : {len(alphas)}")
        print()
