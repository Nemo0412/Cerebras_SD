"""Pivot eval results into wide CSV: rows = ckpts, cols = setting × bench.

Output format matches user's spreadsheet layout:
  data, config, greedy_mt, greedy_gsm, greedy_he, greedy_avg,
                t05_mt, t05_gsm, t05_he, t05_avg,
                t1_mt, t1_gsm, t1_he, t1_avg

Settings:
  greedy → eval at temperature 0
  t05    → sample T=0.5
  t1     → sample T=1.0

`--decode-mode chain|tree` controls which decoder's α is reported.

Usage:
  python sdpo/diagnose/csv_eval_pivot.py --decode-mode tree --output eval_pivot_tree.csv
  python sdpo/diagnose/csv_eval_pivot.py --decode-mode chain --output eval_pivot_chain.csv
"""
import argparse
import csv
import json
import os


CKPTS = [
    # (ckpt_tag,                       config_label,                  greedy_state)
    ("baseline",                       "0.6B baseline (no train)",    None),
    ("kl_regen_l2k_6ep",               "6ep KL anchor",               "state5"),
    ("kltv_regen_l2k_6ep",             "6ep KL + TV",                 "state5"),
    ("klv4_regen_l2k_6ep",             "6ep KL + V4",                 "state5"),
    ("klv4_l2k_grpo_base_6ep",         "6ep KLV4 + GRPO_base",        "state5"),
    ("klv4_l2k_grpo_smp",              "KLV4 + GRPO_smp",             "state2"),
    ("klv8_regen_l2k_6ep",             "6ep KL + V8",                 "state5"),
    ("klv8_l2k_grpo",                  "KLV8 + GRPO",                 "state2"),
    ("alkl_only_regen_l2k_6ep",        "6ep AL_KL only",              "state5"),
    ("alkl_grpo_6ep",                  "AL_KL + GRPO",                "state5"),
    ("altv_only_regen_l2k_6ep",        "6ep AL_TV only",              "state5"),
    ("altv_grpo_6ep",                  "AL_TV + GRPO",                "state5"),
    ("wkl_only_regen_l2k_6ep",         "6ep WKL only",                "state5"),
    ("wkl_grpo_6ep",                   "WKL + GRPO",                  "state5"),
]

BENCHES = ["mt_bench", "gsm8k", "humaneval"]
BENCH_SHORT = {"mt_bench": "mt", "gsm8k": "gsm", "humaneval": "he"}

SETTINGS = [
    ("greedy", lambda gs: gs if gs else None),
    ("t05",    lambda gs: "samp_t05"),
    ("t1",     lambda gs: "samp_t1"),
]


def load_alpha(eval_dir, ckpt_tag, suffix, bench, decode_mode):
    if suffix is None:
        return None
    if ckpt_tag == "baseline":
        path = os.path.join(eval_dir, f"q8_q06_baseline_{suffix}.json")
    else:
        path = os.path.join(eval_dir, f"q8_q06_{ckpt_tag}_{suffix}.json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    r = d.get("results", {}).get(bench, {})
    key = "baseline_chain" if decode_mode == "chain" else "tree"
    return r.get(key, {}).get("mean_alpha")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", default="smalllm_tree_eval_results")
    p.add_argument("--decode-mode", choices=["chain", "tree"], default="tree")
    p.add_argument("--data", default="mixed")
    p.add_argument("--output", default="eval_pivot.csv")
    args = p.parse_args()

    # Build columns: data, config, then for each setting → 3 benches + avg
    fieldnames = ["data", "config"]
    for setting, _ in SETTINGS:
        for b in BENCHES:
            fieldnames.append(f"{setting}_{BENCH_SHORT[b]}")
        fieldnames.append(f"{setting}_avg")

    rows = []
    for ckpt_tag, label, greedy_state in CKPTS:
        row = {"data": args.data, "config": label}
        for setting, suffix_fn in SETTINGS:
            suffix = suffix_fn(greedy_state)
            vals = []
            for b in BENCHES:
                a = load_alpha(args.eval_dir, ckpt_tag, suffix, b, args.decode_mode)
                col = f"{setting}_{BENCH_SHORT[b]}"
                if a is not None:
                    row[col] = round(a, 4)
                    vals.append(a)
                else:
                    row[col] = ""
            row[f"{setting}_avg"] = round(sum(vals)/len(vals), 4) if vals else ""
        rows.append(row)

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[CSV] decode_mode={args.decode_mode}  wrote {args.output}  ({len(rows)} rows)")

    # Also print to console
    col_widths = {"data": 8, "config": 30}
    for c in fieldnames[2:]:
        col_widths[c] = 9
    print()
    print(" ".join(f"{c:<{col_widths[c]}}" for c in fieldnames))
    for r in rows:
        print(" ".join(
            f"{str(r.get(c, '')):<{col_widths[c]}}" for c in fieldnames))


if __name__ == "__main__":
    main()
