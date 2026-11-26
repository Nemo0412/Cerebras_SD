#!/usr/bin/env python3
"""Plot Sample Chain EAL vs dense Sample Tree EAL profiling curves.

Losses live on different scales (chain al_tv vs tree dense residual TV + KL),
so they get separate y-axes / panels. Also plots analytic accept length and
the tree run's top-4 path soft / hard lengths (every step, markers every 50).
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path: str) -> list[dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def series(rows, key, world_size=1, x_mode="samples"):
    """x_mode=samples uses step * world_size so 1-GPU chain and 3-GPU tree align."""
    xs, ys = [], []
    for r in rows:
        if key not in r or r[key] is None:
            continue
        try:
            y = float(r[key])
        except (TypeError, ValueError):
            continue
        if y != y:  # NaN
            continue
        step = int(r.get("step", len(xs) + 1))
        xs.append(step * world_size if x_mode == "samples" else step)
        ys.append(y)
    return np.asarray(xs), np.asarray(ys)


def smooth(y, w=25):
    if y.size == 0:
        return y
    w = max(1, min(int(w), y.size))
    ker = np.ones(w) / w
    return np.convolve(y, ker, mode="same")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/ssd_ext/lls/cerebras_sd/checkpoints")
    ap.add_argument("--chain-tag", default="profile_sample_chain_eal")
    ap.add_argument("--tree-tag", default="profile_sample_tree_eal")
    ap.add_argument("--out", default="/mnt/ssd_ext/lls/cerebras_sd/profile_chain_vs_tree.png")
    ap.add_argument("--smooth", type=int, default=25)
    ap.add_argument("--chain-gpus", type=int, default=1)
    ap.add_argument("--tree-gpus", type=int, default=3)
    ap.add_argument("--path-compare", action="store_true",
                    help="Plot chain vs tree top-4 path hard AL under tree "
                         "sample_rsd eval (chain needs tree_path_probe logs "
                         "or tree_path_probe.jsonl).")
    ap.add_argument("--xmax", type=float, default=0,
                    help="If >0, limit x-axis to this samples-seen value "
                         "(e.g. 10000).")
    ap.add_argument("--chain-path-tag", default="",
                    help="Optional extra savedir tag whose train_steps/"
                         "profile_every_50 hold chain tree-path probe curves "
                         "(used when --chain-tag itself has no path* keys).")
    args = ap.parse_args()
    cg, tg = args.chain_gpus, args.tree_gpus

    chain = load_jsonl(os.path.join(args.root, args.chain_tag, "train_steps.jsonl"))
    tree = load_jsonl(os.path.join(args.root, args.tree_tag, "train_steps.jsonl"))
    chain50 = load_jsonl(os.path.join(args.root, args.chain_tag, "profile_every_50.jsonl"))
    tree50 = load_jsonl(os.path.join(args.root, args.tree_tag, "profile_every_50.jsonl"))
    chain_probe = load_jsonl(os.path.join(
        args.root, args.chain_tag, "tree_path_probe.jsonl"))
    chain_path_extra = []
    chain_path_extra50 = []
    if args.chain_path_tag:
        chain_path_extra = load_jsonl(os.path.join(
            args.root, args.chain_path_tag, "train_steps.jsonl"))
        chain_path_extra50 = load_jsonl(os.path.join(
            args.root, args.chain_path_tag, "profile_every_50.jsonl"))

    if args.path_compare:
        def rolling(x, y, w):
            x = np.asarray(x, float)
            y = np.asarray(y, float)
            if y.size < w:
                return x, y
            c = np.cumsum(np.insert(y, 0, 0.0))
            return x[w - 1:], (c[w:] - c[:-w]) / w

        def clip_xy(xs, ys):
            if not args.xmax or xs.size == 0:
                return xs, ys
            m = xs <= args.xmax
            return xs[m], ys[m]

        fig, ax = plt.subplots(figsize=(11, 6))
        w = max(args.smooth, 50)
        colors = ["#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
        # Tree training path hard ALs
        for i, c in enumerate(colors, 1):
            xs, ys = series(tree, f"path{i}_al", tg)
            xs, ys = clip_xy(xs, ys)
            if xs.size:
                rx, ry = rolling(xs, ys, w)
                ax.plot(rx, ry, color=c, lw=2, label=f"tree path{i} hard AL")
        # Chain online probe (every N steps) if present
        chain_path_rows = [r for r in chain if "path1_al" in r]
        if not chain_path_rows:
            chain_path_rows = [r for r in chain50 if "path1_al" in r]
        if not chain_path_rows:
            chain_path_rows = [r for r in chain_path_extra if "path1_al" in r]
        if not chain_path_rows:
            chain_path_rows = [r for r in chain_path_extra50 if "path1_al" in r]
        for i, c in enumerate(colors, 1):
            xs, ys = series(chain_path_rows, f"path{i}_al", cg)
            xs, ys = clip_xy(xs, ys)
            if xs.size:
                # online probes are already sparse (every 50); light smooth only
                rw = max(3, min(11, max(xs.size // 10, 1)))
                rx, ry = rolling(xs, ys, rw) if xs.size >= rw else (xs, ys)
                ax.plot(rx, ry, color=c, lw=2, ls="--",
                        label=f"chain path{i} hard AL (tree probe)")
        # Offline probes with explicit step → connect as a sparse dashed curve
        stepped = [pr for pr in chain_probe
                   if pr.get("step") is not None and "path1_al" in pr]
        if stepped:
            stepped = sorted(stepped, key=lambda r: float(r["step"]))
            for i, c in enumerate(colors, 1):
                xs = np.asarray([float(r["step"]) * cg for r in stepped])
                ys = np.asarray([float(r[f"path{i}_al"]) for r in stepped])
                xs, ys = clip_xy(xs, ys)
                if xs.size:
                    ax.plot(xs, ys, color=c, lw=1.5, ls="--", marker="o",
                            markersize=5, alpha=0.85,
                            label=f"chain path{i} offline probe")
        elif chain_probe and not chain_path_rows:
            # fallback: endpoint hlines only when no series exists
            for pr in chain_probe:
                for i, c in enumerate(colors, 1):
                    if f"path{i}_al" in pr:
                        ax.axhline(pr[f"path{i}_al"], color=c, lw=1,
                                   alpha=0.35, ls=":")
        if args.xmax:
            ax.set_xlim(0, args.xmax)
        ax.set_xlabel("samples seen (step × n_gpus)")
        ax.set_ylabel("path hard AL (tree sample_rsd)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8, ncol=2)
        ax.set_title(
            "Top-4 path hard AL — tree train vs chain under tree-eval probe\n"
            "solid=tree train; dashed=chain tree-path probe")
        fig.tight_layout()
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        fig.savefig(args.out, dpi=140)
        print(f"wrote {args.out} (path-compare) "
              f"tree={len(tree)} chain_probe_steps={len(chain_path_rows)} "
              f"offline_probes={len(chain_probe)} xmax={args.xmax or 'none'}")
        return

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    fig.suptitle(
        "Sample Chain EAL (al_tv) vs dense Sample Tree EAL (sample_rsd)\n"
        "x = samples seen (step × n_gpus); losses not directly comparable",
        fontsize=12)

    # --- panel 1: losses ---
    ax = axes[0]
    cx, cy = series(chain, "loss", cg)
    tx, ty = series(tree, "loss", tg)
    te_x, te_y = series(tree, "eal_loss", tg)
    if cx.size:
        ax.plot(cx, smooth(cy, args.smooth), color="#1f77b4", lw=2,
                label="chain loss (= aux / −E[L] al_tv)")
        if chain50:
            sx, sy = series(chain50, "loss", cg)
            ax.scatter(sx, sy, color="#1f77b4", s=28, zorder=3, label="chain @50")
    ax.set_ylabel("chain loss", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    if tx.size:
        ax2.plot(tx, smooth(ty, args.smooth), color="#d62728", lw=2,
                 label="tree total loss")
        if te_x.size:
            ax2.plot(te_x, smooth(te_y, args.smooth), color="#ff7f0e", lw=1.5,
                     ls="--", label="tree eal_loss (−dense TV)")
        if tree50:
            sx, sy = series(tree50, "loss", tg)
            ax2.scatter(sx, sy, color="#d62728", s=28, zorder=3, label="tree @50")
    ax2.set_ylabel("tree loss", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    ax.set_title("Training loss (dual axis)")

    # --- panel 2: analytic accept length ---
    ax = axes[1]
    cx, cy = series(chain, "analytic_accepted_length", cg)
    tx, ty = series(tree, "analytic_accepted_length", tg)
    if cx.size:
        ax.plot(cx, smooth(cy, args.smooth), color="#1f77b4", lw=2,
                label="chain analytic E[L]")
        if chain50:
            sx, sy = series(chain50, "analytic_accepted_length", cg)
            ax.scatter(sx, sy, color="#1f77b4", s=28, zorder=3)
    if tx.size:
        ax.plot(tx, smooth(ty, args.smooth), color="#d62728", lw=2,
                label="tree analytic RSD E[L]")
        if tree50:
            sx, sy = series(tree50, "analytic_accepted_length", tg)
            ax.scatter(sx, sy, color="#d62728", s=28, zorder=3)
    ax.set_ylabel("analytic accept length")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Logged analytic accept length")

    # --- panel 3: top-4 path lengths (tree only) ---
    ax = axes[2]
    colors = ["#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    drew = False
    for i, c in enumerate(colors, start=1):
        x_e, y_e = series(tree, f"path{i}_eal", tg)
        x_a, y_a = series(tree, f"path{i}_al", tg)
        if x_e.size:
            ax.plot(x_e, smooth(y_e, args.smooth), color=c, lw=2,
                    label=f"path{i} soft RSD mass")
            drew = True
        if x_a.size:
            ax.plot(x_a, smooth(y_a, args.smooth), color=c, lw=1.2, ls=":",
                    label=f"path{i} hard AL")
            drew = True
        if tree50:
            sx, sy = series(tree50, f"path{i}_eal", tg)
            if sx.size:
                ax.scatter(sx, sy, color=c, s=22, zorder=3)
    if not drew:
        ax.text(0.5, 0.5, "no path1..path4_* yet in tree train_steps.jsonl",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("samples seen (step × n_gpus)")
    ax.set_ylabel("top-4 path length")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.set_title("Tree top-4 draft paths (path1 = highest draft-prob product)")

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}  chain_steps={len(chain)} tree_steps={len(tree)} "
          f"chain50={len(chain50)} tree50={len(tree50)}")


if __name__ == "__main__":
    main()
