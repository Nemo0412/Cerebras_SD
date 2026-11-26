#!/usr/bin/env python3
"""Rebuild demo-test train/val/OOD JSONL splits (A/B/C/D).

Matches the construction used in the eng-wl3181 demo:
  Train A: seed=0 shuffle alpaca-regen → first 2048
  Val  A:  seed=0 shuffle mixed_val_80 → first 16
  Test B:  alpaca after the 2048 cut → first 32
  Test C:  mixed_val remainder → first 32
  Test D:  sharegpt cleaned, ids not in mixed_val, seed=1 → first 32
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    turns = [len(r.get("conversations") or []) for r in rows]
    print(f"wrote {path} n={len(rows)} "
          f"turns[min/med/max]={min(turns)}/{sorted(turns)[len(turns)//2]}/{max(turns)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpaca", required=True, type=Path)
    ap.add_argument("--mixed-val", required=True, type=Path)
    ap.add_argument("--sharegpt", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "data")
    ap.add_argument("--train-n", type=int, default=2048)
    ap.add_argument("--val-n", type=int, default=16)
    ap.add_argument("--ood-n", type=int, default=32)
    args = ap.parse_args()

    rng = random.Random(0)
    alpaca = load_jsonl(args.alpaca)
    rng.shuffle(alpaca)
    train = alpaca[: args.train_n]
    holdout = alpaca[args.train_n:]

    mixed = load_jsonl(args.mixed_val)
    rng_v = random.Random(0)
    rng_v.shuffle(mixed)
    val = mixed[: args.val_n]
    val_ids = {r["id"] for r in val}
    mixed_rest = [r for r in mixed if r["id"] not in val_ids]
    mixed_all_ids = {r["id"] for r in mixed}

    sg = []
    with args.sharegpt.open() as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in mixed_all_ids:
                continue
            if not r.get("conversations") or len(r["conversations"]) < 2:
                continue
            sg.append(r)
            if len(sg) >= 5000:
                break
    rng_s = random.Random(1)
    rng_s.shuffle(sg)

    out = args.out_dir
    write_jsonl(out / "quick_train_2048.jsonl", train)
    write_jsonl(out / "quick_val_16.jsonl", val)
    write_jsonl(out / "quick_ood_A_train_val16.jsonl", val)
    write_jsonl(out / "quick_ood_B_alpaca_holdout.jsonl", holdout[: args.ood_n])
    write_jsonl(out / "quick_ood_C_mixed_sharegpt.jsonl", mixed_rest[: args.ood_n])
    write_jsonl(out / "quick_ood_D_sharegpt_fresh.jsonl", sg[: args.ood_n])

    train_ids = {r["id"] for r in train}
    for name in ("quick_ood_B_alpaca_holdout.jsonl",
                 "quick_ood_C_mixed_sharegpt.jsonl",
                 "quick_ood_D_sharegpt_fresh.jsonl"):
        ids = {json.loads(l)["id"] for l in open(out / name) if l.strip()}
        assert not (train_ids & ids), f"leak into {name}"
    print("ok: no train-id overlap in B/C/D")


if __name__ == "__main__":
    main()
