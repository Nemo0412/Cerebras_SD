"""Profile target model's per-position top-K probability distribution.
Answers: what's the actual probability mass at rank 1/5/10/20/50? How much
mass sits beyond top-20? Where is the cliff (if any)?
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_bench(bench, data_dir="data"):
    path = os.path.join(data_dir, bench, "question.jsonl")
    with open(path) as f:
        return [json.loads(l) for l in f]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target-model', required=True)
    p.add_argument('--bench-name', default='mt_bench,gsm8k,humaneval')
    p.add_argument('--num-samples', type=int, default=5)
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--data-dir', default='data')
    p.add_argument('--output', default='target_topk_profile.json')
    args = p.parse_args()

    torch.manual_seed(0)
    print(f"[LOAD] {args.target_model}")
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=torch.bfloat16,
        device_map='auto', attn_implementation='sdpa')
    target.eval()
    tok = AutoTokenizer.from_pretrained(args.target_model)

    # Aggregate stats across all positions
    all_stats = {b: [] for b in args.bench_name.split(',')}

    for bench in args.bench_name.split(','):
        questions = load_bench(bench, args.data_dir)[:args.num_samples]
        print(f"\n[{bench}] {len(questions)} samples")
        for qi, q in enumerate(questions):
            prompt = q.get("turns", [q.get("prompt", "")])[0] \
                if "turns" in q else q.get("prompt", "")
            msgs = [{"role": "user", "content": prompt}]
            try:
                text = tok.apply_chat_template(msgs, tokenize=False,
                                                add_generation_prompt=True,
                                                enable_thinking=True)
            except TypeError:
                text = tok.apply_chat_template(msgs, tokenize=False,
                                                add_generation_prompt=True)
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(target.device)

            with torch.inference_mode():
                out = target.generate(
                    ids, max_new_tokens=args.max_new_tokens,
                    do_sample=True, temperature=args.temperature,
                    top_p=1.0, top_k=0,
                    output_scores=True, return_dict_in_generate=True,
                    pad_token_id=tok.eos_token_id if tok.pad_token_id is None else tok.pad_token_id,
                )

            # out.scores is tuple of [1, V] tensors, one per generated position
            for step_scores in out.scores:
                logits = step_scores[0].float() / args.temperature
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                sorted_p = np.sort(probs)[::-1]   # descending

                # per-position stats
                cum = np.cumsum(sorted_p)
                s = {
                    'top1': float(sorted_p[0]),
                    'top5_mass': float(sorted_p[:5].sum()),
                    'top10_mass': float(sorted_p[:10].sum()),
                    'top20_mass': float(sorted_p[:20].sum()),
                    'top50_mass': float(sorted_p[:50].sum()),
                    'top100_mass': float(sorted_p[:100].sum()),
                    'p_at_rank20': float(sorted_p[19]),
                    'p_at_rank50': float(sorted_p[49]),
                    'p_at_rank100': float(sorted_p[99]),
                    # rank where cum mass crosses thresholds (K under top-P)
                    'rank_at_90pct': int(np.searchsorted(cum, 0.90) + 1),
                    'rank_at_95pct': int(np.searchsorted(cum, 0.95) + 1),
                    'rank_at_99pct': int(np.searchsorted(cum, 0.99) + 1),
                    'rank_at_995pct': int(np.searchsorted(cum, 0.995) + 1),
                    'rank_at_999pct': int(np.searchsorted(cum, 0.999) + 1),
                    'rank_at_9999pct': int(np.searchsorted(cum, 0.9999) + 1),
                    # entropy in nats
                    'entropy': float(-(probs * np.log(np.clip(probs, 1e-30, 1))).sum()),
                    # ratio to top1 (for "dynamic threshold" idea)
                    'p_at_rank20_over_top1': float(sorted_p[19] / max(sorted_p[0], 1e-30)),
                    'p_at_rank50_over_top1': float(sorted_p[49] / max(sorted_p[0], 1e-30)),
                }
                all_stats[bench].append(s)
            print(f"  q{qi}: {len(out.scores)} positions profiled, "
                  f"top1_avg={np.mean([s['top1'] for s in all_stats[bench][-len(out.scores):]]):.3f}, "
                  f"top20_mass_avg={np.mean([s['top20_mass'] for s in all_stats[bench][-len(out.scores):]]):.3f}")

    # Summary: expanded percentiles + histogram bins
    summary = {}
    keys = ['top1', 'top5_mass', 'top10_mass', 'top20_mass', 'top50_mass', 'top100_mass',
            'p_at_rank20', 'p_at_rank50', 'p_at_rank100',
            'rank_at_90pct', 'rank_at_95pct', 'rank_at_99pct',
            'rank_at_995pct', 'rank_at_999pct', 'rank_at_9999pct',
            'entropy', 'p_at_rank20_over_top1', 'p_at_rank50_over_top1']
    all_flat = []
    for b in all_stats: all_flat.extend(all_stats[b])
    per_bench_and_all = {**all_stats, 'ALL': all_flat}
    hist_edges = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999, 1.0]
    for k in keys:
        summary[k] = {}
        for b, lst in per_bench_and_all.items():
            vs = np.array([s[k] for s in lst], dtype=np.float64)
            summary[k][b] = {
                'mean': float(vs.mean()),
                'p1': float(np.percentile(vs, 1)),
                'p5': float(np.percentile(vs, 5)),
                'p10': float(np.percentile(vs, 10)),
                'p25': float(np.percentile(vs, 25)),
                'p50': float(np.percentile(vs, 50)),
                'p75': float(np.percentile(vs, 75)),
                'p90': float(np.percentile(vs, 90)),
                'p95': float(np.percentile(vs, 95)),
                'p99': float(np.percentile(vs, 99)),
                'n': int(len(vs)),
            }
            if k in ('top1','top5_mass','top10_mass','top20_mass','top50_mass','top100_mass'):
                hist, _ = np.histogram(vs, bins=hist_edges)
                summary[k][b]['hist_edges'] = hist_edges
                summary[k][b]['hist_counts'] = [int(x) for x in hist]
                summary[k][b]['hist_frac'] = [float(x)/len(vs) for x in hist]
            if k.startswith('rank_at_'):
                # K histogram bins for top-P: 1, 2, 3, 5, 10, 20, 50, 100, 500, +inf
                k_edges = [0.5, 1.5, 2.5, 3.5, 5.5, 10.5, 20.5, 50.5, 100.5, 500.5, np.inf]
                khist, _ = np.histogram(vs, bins=k_edges)
                summary[k][b]['k_hist_edges'] = k_edges
                summary[k][b]['k_hist_counts'] = [int(x) for x in khist]
                summary[k][b]['k_hist_frac'] = [float(x)/len(vs) for x in khist]

    with open(args.output, 'w') as f:
        json.dump({'model': args.target_model, 'summary': summary,
                   'per_bench_stats_len': {b: len(all_stats[b]) for b in all_stats}},
                   f, indent=2)
    print(f"\n[SAVE] {args.output}")


if __name__ == '__main__':
    main()
