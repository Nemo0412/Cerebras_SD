"""Aggregate eval results across (ckpt × temperature × bench × decode_mode) → CSV.

Reads smalllm_tree_eval_results/q8_q06_{ckpt}_{suffix}.json.
Suffixes:
  state5 / state2 (greedy)         → temp=greedy
  samp_t05                          → temp=0.5
  samp_t1                           → temp=1.0
Decode modes:
  baseline_chain  → "chain"
  tree            → "tree"
Benches: mt_bench, gsm8k, humaneval, ...

Output columns:
  ckpt, loss_family, temp, bench, decode_mode, mean_alpha, tokens_per_sec,
  draft_fraction, gamma, top_k, budget

Usage:
  python sdpo/diagnose/csv_eval_summary.py --output eval_summary.csv
"""
import argparse
import csv
import json
import os
import re


CKPTS = [
    # (ckpt_tag,                                  loss_family,    state_for_greedy)
    ("baseline",                                  "baseline",     None),
    ("kl_regen_l2k_6ep",                          "KL_only",      "state5"),
    ("kltv_regen_l2k_6ep",                        "KL+TV",        "state5"),
    ("klv4_regen_l2k_6ep",                        "KLV4",         "state5"),
    ("klv4_l2k_grpo_smp",                         "KLV4+GRPO_smp", "state2"),
    ("klv4_l2k_grpo_base_6ep",                    "KLV4+GRPO_base","state5"),
    ("klv8_regen_l2k_6ep",                        "KLV8",         "state5"),
    ("klv8_l2k_grpo",                             "KLV8+GRPO",    "state2"),
    ("alkl_only_regen_l2k_6ep",                   "AL_KL_only",   "state5"),
    ("alkl_grpo_6ep",                             "AL_KL+GRPO",   "state5"),
    ("altv_only_regen_l2k_6ep",                   "AL_TV_only",   "state5"),
    ("altv_grpo_6ep",                             "AL_TV+GRPO",   "state5"),
    ("wkl_only_regen_l2k_6ep",                    "WKL_only",     "state5"),
    ("wkl_grpo_6ep",                              "WKL+GRPO",     "state5"),
]

SUFFIX_TO_TEMP = {
    "samp_t05": "sample_T=0.5",
    "samp_t1": "sample_T=1.0",
}

MODE_LABEL = {
    "baseline_chain": "chain",
    "tree": "tree",
}


def find_json(ckpt_tag, suffix, eval_dir):
    """Look for q8_q06_{ckpt_tag}_{suffix}.json"""
    if ckpt_tag == "baseline":
        path = os.path.join(eval_dir, f"q8_q06_baseline_{suffix}.json")
    else:
        path = os.path.join(eval_dir, f"q8_q06_{ckpt_tag}_{suffix}.json")
    return path if os.path.exists(path) else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", default="smalllm_tree_eval_results")
    p.add_argument("--output", default="eval_summary.csv")
    args = p.parse_args()

    rows = []
    missing = []
    for ckpt_tag, family, greedy_state in CKPTS:
        # Build the list of (temp, suffix) to try
        suffixes = []
        if greedy_state:
            suffixes.append(("greedy", greedy_state))
        else:
            # baseline: no greedy state suffix; nothing to look up for greedy
            pass
        suffixes.append(("sample_T=0.5", "samp_t05"))
        suffixes.append(("sample_T=1.0", "samp_t1"))

        for temp, suffix in suffixes:
            path = find_json(ckpt_tag, suffix, args.eval_dir)
            if not path:
                missing.append(f"{ckpt_tag}/{suffix}")
                continue
            d = json.load(open(path))
            results = d.get("results", {})
            for bench, modes in results.items():
                for mode_key, stats in modes.items():
                    mode = MODE_LABEL.get(mode_key, mode_key)
                    if not isinstance(stats, dict):
                        continue
                    rows.append({
                        "ckpt": ckpt_tag,
                        "loss_family": family,
                        "temp": temp,
                        "bench": bench,
                        "decode_mode": mode,
                        "mean_alpha": stats.get("mean_alpha"),
                        "tokens_per_sec": stats.get("tokens_per_sec"),
                        "draft_fraction": stats.get("draft_fraction"),
                        "gamma": stats.get("gamma"),
                        "top_k": stats.get("top_k"),
                        "budget": stats.get("budget"),
                    })

    # Sort: family > ckpt > temp > bench > decode_mode
    temp_order = {"greedy": 0, "sample_T=0.5": 1, "sample_T=1.0": 2}
    rows.sort(key=lambda r: (r["loss_family"], r["ckpt"],
                              temp_order.get(r["temp"], 99),
                              r["bench"], r["decode_mode"]))

    fieldnames = ["ckpt", "loss_family", "temp", "bench", "decode_mode",
                  "mean_alpha", "tokens_per_sec", "draft_fraction",
                  "gamma", "top_k", "budget"]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[CSV] Wrote {args.output}  ({len(rows)} rows)")
    if missing:
        print(f"[CSV] Missing JSONs ({len(missing)}):")
        for m in missing:
            print(f"    - {m}")

    # Also print a pivot summary: rows = ckpt+family, cols = temp×bench×mode → mean_alpha
    print("\n[PIVOT SUMMARY: mean_alpha by ckpt × temp × bench × decode_mode]")
    benches = sorted({r["bench"] for r in rows})
    by_ckpt = {}
    for r in rows:
        by_ckpt.setdefault((r["loss_family"], r["ckpt"]), {})[
            (r["temp"], r["bench"], r["decode_mode"])] = r["mean_alpha"]

    # Header
    cols = []
    for t in ("greedy", "sample_T=0.5", "sample_T=1.0"):
        for b in benches:
            for m in ("chain", "tree"):
                cols.append((t, b, m))
    print(f"{'family':<18}{'ckpt':<32}", end="")
    for (t, b, m) in cols:
        label = f"{t[0]}/{b[:4]}/{m[0]}"
        print(f"{label:>10}", end="")
    print()
    for (fam, ck), cells in sorted(by_ckpt.items()):
        print(f"{fam:<18}{ck:<32}", end="")
        for col in cols:
            v = cells.get(col)
            print(f"{v:>10.3f}" if v is not None else f"{'-':>10}", end="")
        print()


if __name__ == "__main__":
    main()
