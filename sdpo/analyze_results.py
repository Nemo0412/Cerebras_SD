"""Load per-sample kl_vs_klv4 results and slice by bench / category / diff.

Usage:
    python sdpo/analyze_results.py                            # default summary + top/bot
    python sdpo/analyze_results.py --bench mt_bench --mode tree
    python sdpo/analyze_results.py --by-category              # mt_bench split by category
    python sdpo/analyze_results.py --show-prompt --top 5      # print full prompts
    python sdpo/analyze_results.py --csv > out.csv            # flat CSV
"""
import argparse
import json
import os
import sys
from collections import defaultdict

DATA_DIR = 'data'
DEFAULT_JSON = 'smalllm_tree_eval_results/kl_vs_klv4_per_sample.json'


def load_questions(bench):
    path = os.path.join(DATA_DIR, bench, "question.jsonl")
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as f:
        for i, line in enumerate(f):
            q = json.loads(line)
            out[i] = q
    return out


def fmt_row(r, mode, show_prompt_len):
    return (f"idx={r['idx']:3d}  "
            f"Δ={r[f'diff_{mode}']:+.3f}  "
            f"klv4={r[f'klv4_{mode}_alpha']:.2f} "
            f"kl={r[f'kl_{mode}_alpha']:.2f}  "
            f"| {r.get('_cat', '?'):<12} | {r['prompt'][:show_prompt_len]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--json', default=DEFAULT_JSON)
    p.add_argument('--bench', default=None,
                   help='mt_bench / gsm8k / humaneval. Default: all')
    p.add_argument('--mode', default='tree',
                   choices=['chain', 'tree', 'both'])
    p.add_argument('--top', type=int, default=5)
    p.add_argument('--by-category', action='store_true',
                   help='For mt_bench, break down by category')
    p.add_argument('--show-prompt', action='store_true',
                   help='Print full prompt for top/bottom examples')
    p.add_argument('--csv', action='store_true',
                   help='Dump flat CSV to stdout (no summary)')
    args = p.parse_args()

    if not os.path.exists(args.json):
        print(f"JSON not found: {args.json}", file=sys.stderr)
        sys.exit(1)

    data = json.load(open(args.json))
    benches = [args.bench] if args.bench else list(data.keys())

    # Join category from original question files
    for bench in benches:
        if bench not in data:
            continue
        qs = load_questions(bench)
        for r in data[bench]:
            q = qs.get(r['idx'], {})
            r['_cat'] = q.get('category', bench)

    if args.csv:
        keys = ['bench', 'idx', '_cat',
                'kl_chain_alpha', 'klv4_chain_alpha', 'diff_chain',
                'kl_tree_alpha', 'klv4_tree_alpha', 'diff_tree',
                'prompt']
        print(','.join(keys))
        for bench in benches:
            for r in data.get(bench, []):
                vals = [bench, str(r['idx']), r['_cat']]
                for k in keys[3:-1]:
                    vals.append(f"{r[k]:.4f}")
                prompt = r['prompt'].replace('"', "'").replace('\n', ' ')[:200]
                vals.append(f'"{prompt}"')
                print(','.join(vals))
        return

    plen = 300 if args.show_prompt else 80
    modes = ['chain', 'tree'] if args.mode == 'both' else [args.mode]

    for bench in benches:
        rows = data.get(bench, [])
        if not rows:
            print(f"\n{bench}: (no data)")
            continue

        print(f"\n{'='*80}\n{bench}  ({len(rows)} samples)")
        for mode in modes:
            diffs = [r[f'diff_{mode}'] for r in rows]
            kls = [r[f'kl_{mode}_alpha'] for r in rows]
            klv4s = [r[f'klv4_{mode}_alpha'] for r in rows]
            n = len(diffs)
            mean_d = sum(diffs) / n
            import statistics
            std_d = statistics.stdev(diffs) if n > 1 else 0.0
            se_d = std_d / (n ** 0.5) if n > 1 else 0.0
            wins = sum(1 for d in diffs if d > 0.1)
            losses = sum(1 for d in diffs if d < -0.1)
            ties = n - wins - losses
            print(f"\n  [{mode}]  mean: kl={sum(kls)/n:.3f}  klv4={sum(klv4s)/n:.3f}  "
                  f"Δ={mean_d:+.4f} (SE={se_d:.4f}, 95%CI=±{1.96*se_d:.4f})")
            print(f"         klv4_win (Δ>0.1)={wins}  tie={ties}  kl_win (Δ<-0.1)={losses}")

            if args.by_category and bench == 'mt_bench':
                by_cat = defaultdict(list)
                for r in rows:
                    by_cat[r['_cat']].append(r[f'diff_{mode}'])
                print(f"         by category:")
                for cat, ds in sorted(by_cat.items()):
                    m = sum(ds) / len(ds)
                    print(f"           {cat:<14} n={len(ds):2d}  Δ={m:+.4f}")

            print(f"\n  TOP {args.top} KLV4 wins ({mode}):")
            top = sorted(rows, key=lambda r: -r[f'diff_{mode}'])[:args.top]
            for r in top:
                print(f"    {fmt_row(r, mode, plen)}")

            print(f"\n  TOP {args.top} KL wins ({mode}):")
            bot = sorted(rows, key=lambda r: r[f'diff_{mode}'])[:args.top]
            for r in bot:
                print(f"    {fmt_row(r, mode, plen)}")


if __name__ == "__main__":
    main()
