"""Show absolute values of all surrogates + hard_tau per ckpt, per setting.

Layout: one table per setting, rows = metrics, cols = ckpts.
Also dumps a wide CSV.
"""
import csv
import glob
import json
import os
import re

import numpy as np


SETTINGS_ORDER = ["greedy", "sampT05", "sampT1", "sampT2"]
SETTING_LABEL = {"greedy": "greedy",
                 "sampT05": "sample T=0.5",
                 "sampT1": "sample T=1.0",
                 "sampT2": "sample T=2.0"}
METRICS = ["hard_tau", "AL_TV", "AL_KL", "EAL", "V4", "WKL"]
BASELINE_TAG = "baseline"

# Discover files
files = sorted(glob.glob("diagnose_ent_*.json"))
runs = {}
for f in files:
    base = os.path.basename(f).replace(".json", "")
    m = re.match(r"diagnose_ent_(.+)_(greedy|sampT05|sampT1|sampT2)$", base)
    if not m:
        continue
    ck, st = m.group(1), m.group(2)
    runs.setdefault(st, {})[ck] = json.load(open(f))["rounds"]

# Compute means per ckpt per setting per metric
means = {}    # means[setting][ckpt][metric] = mean
for st in SETTINGS_ORDER:
    if st not in runs:
        continue
    means[st] = {}
    for ck, rounds in runs[st].items():
        means[st][ck] = {}
        for m in METRICS:
            means[st][ck][m] = float(np.mean([r[m] for r in rounds]))

# Order ckpts: baseline first
all_ckpts = sorted(set().union(*[set(d.keys()) for d in runs.values()]))
if BASELINE_TAG in all_ckpts:
    all_ckpts.remove(BASELINE_TAG)
    all_ckpts = [BASELINE_TAG] + all_ckpts

# Print per-setting table (metric × ckpt)
for st in SETTINGS_ORDER:
    if st not in means:
        continue
    print("\n" + "="*110)
    print(f"  Setting: {SETTING_LABEL[st]}")
    print("="*110)
    header = f"  {'metric':<10}"
    for ck in all_ckpts:
        header += f"{ck:>16}"
    print(header)
    print("  " + "-"*108)
    for m in METRICS:
        row = f"  {m:<10}"
        for ck in all_ckpts:
            v = means[st].get(ck, {}).get(m, float("nan"))
            row += f"{v:>16.3f}"
        print(row)

# Also print: same data but ckpt × (setting × metric) wide table
print("\n" + "="*110)
print("  Wide view: ckpt × (setting × metric), absolute values")
print("="*110)
print(f"\n  Settings: {' | '.join(SETTING_LABEL[s] for s in SETTINGS_ORDER if s in means)}")
print(f"  Metrics per setting: {METRICS}\n")

# Wide CSV
rows_csv = []
fieldnames = ["ckpt"]
for st in SETTINGS_ORDER:
    if st not in means:
        continue
    for m in METRICS:
        fieldnames.append(f"{st}_{m}")

for ck in all_ckpts:
    row = {"ckpt": ck}
    for st in SETTINGS_ORDER:
        if st not in means:
            continue
        for m in METRICS:
            row[f"{st}_{m}"] = round(means[st].get(ck, {}).get(m, float("nan")), 4)
    rows_csv.append(row)

with open("loss_and_tau_values.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows_csv:
        w.writerow(r)

# Print delta-from-baseline summary
if BASELINE_TAG in all_ckpts:
    print("="*110)
    print(f"  Δ vs baseline (positive = trained pushed metric UP)")
    print("="*110)
    base = {st: means[st].get(BASELINE_TAG, {}) for st in SETTINGS_ORDER if st in means}
    for st in SETTINGS_ORDER:
        if st not in means:
            continue
        print(f"\n  -- {SETTING_LABEL[st]} --")
        head = f"  {'metric':<10}"
        for ck in all_ckpts:
            if ck == BASELINE_TAG:
                continue
            head += f"{ck:>16}"
        print(head)
        for m in METRICS:
            row = f"  {m:<10}"
            bv = base[st].get(m, float("nan"))
            for ck in all_ckpts:
                if ck == BASELINE_TAG:
                    continue
                v = means[st].get(ck, {}).get(m, float("nan"))
                d = v - bv
                row += f"{d:>+16.3f}"
            print(row)

print(f"\n[OK] CSV: loss_and_tau_values.csv ({len(rows_csv)} rows)")
