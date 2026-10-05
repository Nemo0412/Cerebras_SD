"""Per-sample α comparison: KL baseline vs KLV4 on 3 benches × {chain, tree}.

Saves per-sample JSON and prints top-K where KLV4 wins / loses vs KL, so we
can eyeball which prompts benefit from the extra aux loss.
"""
import argparse
import glob
import json
import os
import sys

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_small_lm_tree import (
    baseline_chain_decode,
    load_questions,
    spec_decode_tree_smalllm,
)


def load_draft(path):
    # Strip wrapper prefixes in checkpoint state_dict if present.
    for f in glob.glob(os.path.join(path, "*.bin")):
        state = torch.load(f, map_location="cpu")
        changed = False
        for prefix in ("draft.base.", "draft_model."):
            if any(k.startswith(prefix) for k in state):
                state = {k.removeprefix(prefix): v for k, v in state.items()}
                changed = True
        if changed:
            torch.save(state, f)
    m = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16,
        attn_implementation="sdpa").cuda()
    m.eval()
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target', default='Qwen/Qwen3-8B')
    p.add_argument('--kl-draft',
                   default='/scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/q8_q06_kl_regen/state_2')
    p.add_argument('--klv4-draft',
                   default='/scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/q8_q06_klv4_regen/state_2')
    p.add_argument('--benches', default='mt_bench,gsm8k,humaneval')
    p.add_argument('--num-samples', type=int,
                   default=int(os.environ.get('NUM_SAMPLES', 80)))
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--gamma', type=int, default=7)
    p.add_argument('--top-k', default='4,3,2,1,1,1,1')
    p.add_argument('--budget', type=int, default=128)
    p.add_argument('--output',
                   default='smalllm_tree_eval_results/kl_vs_klv4_per_sample.json')
    args = p.parse_args()

    top_k = [int(x) for x in args.top_k.split(',')]

    print(f"Loading target: {args.target}")
    target = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.float16,
        attn_implementation="sdpa").cuda()
    target.eval()
    tok = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    eos_id = getattr(target.config, 'eos_token_id', None)

    print(f"Loading KL draft: {args.kl_draft}")
    print(f"Loading KLV4 draft: {args.klv4_draft}")
    drafts = {
        'kl': load_draft(args.kl_draft),
        'klv4': load_draft(args.klv4_draft),
    }
    n_layers = len(drafts['kl'].model.layers)
    print(f"Drafts loaded, n_layers={n_layers}")

    def tokenize_prompt(q):
        prompt = (q.get("turns", [q.get("prompt", "")])[0]
                  if "turns" in q else q.get("prompt", ""))
        msgs = [{"role": "user", "content": prompt}]
        try:
            text = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=True)
        except TypeError:
            text = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt",
                  add_special_tokens=False).input_ids.to(target.device)
        return ids, prompt

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    benches = [b.strip() for b in args.benches.split(',')]
    all_data = {}

    for bench in benches:
        print(f"\n=== {bench} ===")
        qs = load_questions(bench)
        if not qs:
            print(f"  (no questions loaded for {bench}, skipping)")
            continue
        qs = qs[:args.num_samples]
        per_sample = []
        for i, q in enumerate(tqdm(qs, desc=bench)):
            ids, prompt = tokenize_prompt(q)
            row = {"idx": i, "prompt": prompt[:500]}
            for mname, draft in drafts.items():
                rc = baseline_chain_decode(
                    target, draft, ids, args.max_new_tokens, args.gamma)
                row[f'{mname}_chain_alpha'] = rc['mean_alpha']
                row[f'{mname}_chain_rounds'] = rc['total_rounds']
                row[f'{mname}_chain_tokens'] = rc['total_tokens']

                rt = spec_decode_tree_smalllm(
                    target, draft, ids, args.max_new_tokens, n_layers,
                    args.gamma, top_k, args.budget, eos_token_id=eos_id)
                row[f'{mname}_tree_alpha'] = rt['mean_alpha']
                row[f'{mname}_tree_rounds'] = rt['total_rounds']
                row[f'{mname}_tree_tokens'] = rt['total_tokens']
            row['diff_chain'] = row['klv4_chain_alpha'] - row['kl_chain_alpha']
            row['diff_tree'] = row['klv4_tree_alpha'] - row['kl_tree_alpha']
            per_sample.append(row)

            all_data[bench] = per_sample
            with open(args.output, 'w') as f:
                json.dump(all_data, f, indent=2)

    print(f"\n{'='*80}\nPER-SAMPLE SUMMARY (saved to {args.output})")
    for bench, rows in all_data.items():
        print(f"\n{bench} ({len(rows)} samples):")
        for mode in ['chain', 'tree']:
            diffs = [r[f'diff_{mode}'] for r in rows]
            mean_d = sum(diffs) / len(diffs) if diffs else 0.0
            pos = sum(1 for d in diffs if d > 0.1)
            neg = sum(1 for d in diffs if d < -0.1)
            tie = len(diffs) - pos - neg
            print(f"  {mode}: mean Δ={mean_d:+.4f}  "
                  f"klv4_win={pos}  tie={tie}  kl_win={neg}")

            top = sorted(rows, key=lambda r: -r[f'diff_{mode}'])[:3]
            bot = sorted(rows, key=lambda r: r[f'diff_{mode}'])[:3]
            print(f"  TOP KLV4 wins:")
            for r in top:
                print(f"    idx={r['idx']:3d} Δ={r[f'diff_{mode}']:+.3f}  "
                      f"klv4={r[f'klv4_{mode}_alpha']:.2f} "
                      f"kl={r[f'kl_{mode}_alpha']:.2f}  "
                      f"| {r['prompt'][:80]}")
            print(f"  TOP KL wins:")
            for r in bot:
                print(f"    idx={r['idx']:3d} Δ={r[f'diff_{mode}']:+.3f}  "
                      f"klv4={r[f'klv4_{mode}_alpha']:.2f} "
                      f"kl={r[f'kl_{mode}_alpha']:.2f}  "
                      f"| {r['prompt'][:80]}")


if __name__ == "__main__":
    main()
