"""Summarize per-H_p-bucket comparison of surrogate values + accept length
across trained ckpts vs baseline. Reads diagnose_ent_{ckpt}_{setting}.json files.

For each (mode, T) setting and each entropy bucket (low/mid/high H_p):
  Tabulate hard_τ, AL_TV, AL_KL, EAL, V4, WKL means for every ckpt
  Show Δ from baseline per ckpt.

Buckets are computed via FIXED ENTROPY EDGES (derived from baseline) so that
all ckpts are bucketed the same way → values are comparable across ckpts.

Usage:
  python sdpo/diagnose/summarize_al_entropy.py
"""
import argparse
import glob
import json
import os
import re

import numpy as np


def load_rounds(path):
    d = json.load(open(path))
    return d["rounds"], d["meta"]


def parse_filename(fname):
    """diagnose_ent_{ckpt}_{setting}.json"""
    base = os.path.basename(fname).replace(".json", "")
    m = re.match(r"diagnose_ent_(.+)_(greedy|sampT05|sampT1|sampT2)$", base)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", default="diagnose_ent_*.json")
    p.add_argument("--baseline-tag", default="baseline")
    p.add_argument("--metrics", nargs="+",
                   default=["hard_tau", "AL_TV", "AL_KL", "EAL", "V4", "WKL"])
    args = p.parse_args()

    # ── Discover files ──
    files = sorted(glob.glob(args.pattern))
    runs = {}    # runs[setting][ckpt_tag] = rounds_list
    for f in files:
        ck, st = parse_filename(f)
        if ck is None:
            continue
        rounds, _ = load_rounds(f)
        runs.setdefault(st, {})[ck] = rounds
    if not runs:
        print(f"No matching files for pattern: {args.pattern}")
        return

    SETTINGS_ORDER = ["greedy", "sampT05", "sampT1", "sampT2"]
    SETTINGS = [s for s in SETTINGS_ORDER if s in runs]
    SETTING_LABEL = {"greedy": "greedy",
                     "sampT05": "sample T=0.5",
                     "sampT1": "sample T=1.0",
                     "sampT2": "sample T=2.0"}

    # ── Compute H_p tertile edges per setting using BASELINE data ──
    setting_edges = {}
    for st in SETTINGS:
        if args.baseline_tag not in runs[st]:
            print(f"[WARN] baseline {args.baseline_tag} missing for {st}")
            continue
        base_rounds = runs[st][args.baseline_tag]
        h_p = np.array([r["mean_H_p"] for r in base_rounds])
        edges = np.quantile(h_p, [1/3, 2/3])
        setting_edges[st] = edges

    def bucket_of(h):
        return "low_H" if h <= edges[0] else ("mid_H" if h <= edges[1] else "high_H")

    # ── For each setting, table: ckpt × {bucket × metric} ──
    for st in SETTINGS:
        if st not in setting_edges:
            continue
        edges = setting_edges[st]
        print("\n" + "="*120)
        print(f"  Setting: {SETTING_LABEL[st]}    H_p tertile edges (from baseline): "
              f"low ≤ {edges[0]:.2f} < mid ≤ {edges[1]:.2f} < high")
        print("="*120)

        # Collect per-ckpt per-bucket means
        ckpts = sorted(runs[st].keys(), key=lambda c: (c != args.baseline_tag, c))
        table = {}    # table[ckpt][bucket][metric] = mean
        counts = {}
        for ck in ckpts:
            table[ck] = {b: {} for b in ("low_H", "mid_H", "high_H")}
            counts[ck] = {b: 0 for b in ("low_H", "mid_H", "high_H")}
            for r in runs[st][ck]:
                b = bucket_of(r["mean_H_p"])
                counts[ck][b] += 1
                for m in args.metrics:
                    table[ck][b].setdefault(m, []).append(r[m])
            for b in ("low_H", "mid_H", "high_H"):
                for m in args.metrics:
                    vals = table[ck][b].get(m, [])
                    table[ck][b][m] = float(np.mean(vals)) if vals else float("nan")

        # Print: one block per bucket
        for b in ("low_H", "mid_H", "high_H"):
            print(f"\n  ── {b} bucket ──")
            head = f"  {'ckpt':<22}{'N':>5}"
            for m in args.metrics:
                head += f"{m:>11}"
            head += "    " + "  ".join([f"Δ{m}" for m in args.metrics if m != "hard_tau"])
            print(head)
            base_vals = table.get(args.baseline_tag, {}).get(b, {})
            for ck in ckpts:
                row = f"  {ck:<22}{counts[ck][b]:>5}"
                for m in args.metrics:
                    v = table[ck][b][m]
                    row += f"{v:>11.3f}"
                if ck != args.baseline_tag and base_vals:
                    row += "   "
                    for m in args.metrics:
                        if m == "hard_tau":
                            continue
                        bv = base_vals.get(m, float("nan"))
                        d = table[ck][b][m] - bv
                        sign = "+" if d >= 0 else ""
                        row += f"  {sign}{d:.3f}"
                print(row)

    # ── Cross-temperature comparison: one table per metric ──
    # For each metric, rows = ckpts, cols = settings, cells = mean (Δ vs baseline).
    print("\n" + "="*120)
    print("  Cross-temperature comparison (all rounds, no bucketing)")
    print("  Each cell shows: mean (Δ vs baseline)")
    print("="*120)

    ckpts_all = sorted(set().union(*[set(runs[s].keys()) for s in SETTINGS]))
    if args.baseline_tag in ckpts_all:
        ckpts_all.remove(args.baseline_tag)
        ckpts_all = [args.baseline_tag] + ckpts_all

    for m in args.metrics:
        print(f"\n  --- {m} ---")
        head = f"  {'ckpt':<22}"
        for st in SETTINGS:
            head += f"{SETTING_LABEL[st]:>22}"
        print(head)
        # baseline values per setting
        base_vals = {}
        for st in SETTINGS:
            if args.baseline_tag in runs[st]:
                base_vals[st] = float(np.mean(
                    [r[m] for r in runs[st][args.baseline_tag]]))
            else:
                base_vals[st] = float("nan")
        for ck in ckpts_all:
            row = f"  {ck:<22}"
            for st in SETTINGS:
                if ck not in runs.get(st, {}):
                    row += f"{'-':>22}"
                    continue
                v = float(np.mean([r[m] for r in runs[st][ck]]))
                bv = base_vals.get(st, float("nan"))
                if ck == args.baseline_tag:
                    cell = f"{v:.3f}"
                elif not np.isnan(bv):
                    d = v - bv
                    sign = "+" if d >= 0 else ""
                    cell = f"{v:.3f} ({sign}{d:.3f})"
                else:
                    cell = f"{v:.3f}"
                row += f"{cell:>22}"
            print(row)

    # ── Δhard_τ × (setting × bucket) cross-table (existing) ──
    print("\n" + "="*120)
    print("  Δhard_τ vs baseline, by setting × bucket")
    print("="*120)
    head = f"  {'ckpt':<22}"
    for st in SETTINGS:
        for b in ("low", "mid", "high"):
            head += f"{SETTING_LABEL[st][:8]+'/'+b:>14}"
    print(head)
    for st in SETTINGS:
        if args.baseline_tag not in runs[st]:
            continue
    ckpts_all = sorted(set().union(*[set(runs[s].keys()) for s in SETTINGS]))
    for ck in ckpts_all:
        if ck == args.baseline_tag:
            continue
        row = f"  {ck:<22}"
        for st in SETTINGS:
            if st not in setting_edges:
                row += "         -    "*3
                continue
            edges = setting_edges[st]
            base_per_bucket = {b: [] for b in ("low_H", "mid_H", "high_H")}
            ck_per_bucket = {b: [] for b in ("low_H", "mid_H", "high_H")}
            for r in runs[st].get(args.baseline_tag, []):
                base_per_bucket[bucket_of(r["mean_H_p"])].append(r["hard_tau"])
            for r in runs[st].get(ck, []):
                ck_per_bucket[bucket_of(r["mean_H_p"])].append(r["hard_tau"])
            for b in ("low_H", "mid_H", "high_H"):
                if base_per_bucket[b] and ck_per_bucket[b]:
                    d = np.mean(ck_per_bucket[b]) - np.mean(base_per_bucket[b])
                    sign = "+" if d >= 0 else ""
                    row += f"      {sign}{d:>6.3f}"
                else:
                    row += "         -    "
        print(row)


if __name__ == "__main__":
    main()
