"""
Eval for Layer-Skip Draft Speculative Decoding.

For each requested exit layer in a single small-LM checkpoint (e.g.,
Qwen3-0.6B), runs an independent chain speculative decoding loop against
a large target (e.g., Qwen3-32B). Reports mean alpha per (bench, exit).

Purpose:
  Baseline exploration — find which intermediate layers of the pretrained
  small LM already yield "decent" acceptance α when used as draft. The
  best exit layer(s) then become the training targets for main_layer_skip.

NOTE: This eval runs the FULL small-LM forward pass and reads intermediate
hidden states via `output_hidden_states=True`. It measures acceptance α, not
wall-clock speedup. Actual inference speedup comes from truncating the
small LM to the chosen exit — this script does NOT truncate, because every
exit must be evaluated on the same backbone.

Usage:
  python sdpo/eval_layer_skip.py \\
    --base-model-path Qwen/Qwen3-32B \\
    --draft-model-path Qwen/Qwen3-0.6B \\
    --exit-layers 4,8,12,16,20,24,28 \\
    --tag q32_0p6b_layerskip \\
    --bench-name mt_bench,gsm8k,humaneval
"""

import argparse
import json
import os
import sys
import time

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from layer_skip_model import LayerSkipBackbone


@torch.inference_mode()
def speculative_decode_step(target_model, draft_model, input_ids, gamma, exit_layer):
    draft_ids = input_ids.clone()
    draft_tokens = []
    for _ in range(gamma):
        logits_per_exit = draft_model.forward_with_exits(draft_ids)
        draft_logits = logits_per_exit[exit_layer][:, -1, :]
        next_token = draft_logits.argmax(dim=-1, keepdim=True)
        draft_tokens.append(next_token)
        draft_ids = torch.cat([draft_ids, next_token], dim=1)
    draft_tokens = torch.cat(draft_tokens, dim=1)

    candidate = torch.cat([input_ids, draft_tokens], dim=1)
    target_logits = target_model(candidate).logits

    n_accepted = 0
    for i in range(gamma):
        pos = input_ids.shape[1] + i - 1
        target_token = target_logits[:, pos, :].argmax(dim=-1)
        if target_token.item() == draft_tokens[:, i].item():
            n_accepted += 1
        else:
            break

    correction_pos = input_ids.shape[1] + n_accepted - 1
    correction_token = target_logits[:, correction_pos, :].argmax(dim=-1, keepdim=True)
    new_tokens = torch.cat([draft_tokens[:, :n_accepted], correction_token], dim=1)
    return new_tokens, n_accepted


@torch.inference_mode()
def generate_with_spec_decode(target_model, draft_model, input_ids,
                              max_new_tokens, gamma, exit_layer):
    total_tokens = 0
    total_rounds = 0
    total_accepted = 0
    cur_ids = input_ids

    while total_tokens < max_new_tokens:
        new_tokens, n_accepted = speculative_decode_step(
            target_model, draft_model, cur_ids, gamma, exit_layer)
        cur_ids = torch.cat([cur_ids, new_tokens], dim=1)
        total_tokens += new_tokens.shape[1]
        total_rounds += 1
        total_accepted += n_accepted

        if hasattr(target_model.config, 'eos_token_id'):
            eos = target_model.config.eos_token_id
            if isinstance(eos, list):
                if any(t in new_tokens[0].tolist() for t in eos):
                    break
            elif eos in new_tokens[0].tolist():
                break

    return total_tokens, total_rounds, total_accepted


def load_questions(bench_name, data_dir="data"):
    path = os.path.join(data_dir, bench_name, "question.jsonl")
    if not os.path.exists(path):
        print(f"  Bench {bench_name} not found at {path}, skipping")
        return []
    with open(path) as f:
        return [json.loads(l) for l in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model-path', required=True)
    parser.add_argument('--draft-model-path', required=True)
    parser.add_argument('--exit-layers', required=True,
                        help='comma-separated 1-indexed exit layer list')
    parser.add_argument('--tag', default='layerskip')
    parser.add_argument('--bench-name', default='mt_bench')
    parser.add_argument('--gamma', type=int, default=7)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Limit to first N samples per bench (default: all)')
    parser.add_argument('--output-dir', default='layerskip_eval_results')
    args = parser.parse_args()

    exit_layers = [int(e) for e in args.exit_layers.split(',') if e.strip()]

    if args.bench_name == 'all':
        benches = ['mt_bench', 'gsm8k', 'humaneval', 'qa', 'sum', 'alpaca', 'aime']
    else:
        benches = [b.strip() for b in args.bench_name.split(',') if b.strip()]

    print(f"[EVAL] Target: {args.base_model_path}")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, torch_dtype=torch.float16, device_map="auto")
    target_model.eval()

    print(f"[EVAL] Draft (layer-skip): {args.draft_model_path}, exits={exit_layers}")
    # Fix any 'draft_model.' / 'draft.base.' prefix left over from training saves.
    import glob
    ckpt_files = glob.glob(os.path.join(args.draft_model_path, "*.bin"))
    for f in ckpt_files:
        state = torch.load(f, map_location="cpu")
        changed = False
        for prefix in ("draft.base.", "draft_model."):
            if any(k.startswith(prefix) for k in state):
                print(f"[EVAL] Stripping '{prefix}' prefix from {f}")
                state = {k.removeprefix(prefix): v for k, v in state.items()}
                changed = True
        if changed:
            torch.save(state, f)

    draft_model = LayerSkipBackbone(
        args.draft_model_path, exit_layers=exit_layers,
        dtype=torch.float16).cuda()
    draft_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path, trust_remote_code=True)

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    for bench in benches:
        questions = load_questions(bench)
        if not questions:
            continue
        if args.num_samples is not None:
            questions = questions[:args.num_samples]

        all_results[bench] = {}
        for exit_layer in exit_layers:
            print(f"[EVAL] {bench} × exit={exit_layer} ({len(questions)} questions)")

            bench_tokens = 0
            bench_rounds = 0
            bench_accepted = 0
            t0 = time.time()

            for q in tqdm(questions, desc=f"{args.tag}/{bench}/L{exit_layer}"):
                prompt = (q.get("turns", [q.get("prompt", "")])[0]
                          if "turns" in q else q.get("prompt", ""))
                msgs = [{"role": "user", "content": prompt}]
                try:
                    text = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True,
                        enable_thinking=True)
                except TypeError:
                    text = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True)
                input_ids = tokenizer(
                    text, return_tensors="pt",
                    add_special_tokens=False).input_ids.to(target_model.device)

                n_tok, n_rnd, n_acc = generate_with_spec_decode(
                    target_model, draft_model, input_ids,
                    args.max_new_tokens, args.gamma, exit_layer)
                bench_tokens += n_tok
                bench_rounds += n_rnd
                bench_accepted += n_acc

            elapsed = time.time() - t0
            mean_alpha = bench_accepted / bench_rounds if bench_rounds > 0 else 0

            print(f"\n  {bench}/exit={exit_layer}: "
                  f"α={mean_alpha:.3f}  tok/s={bench_tokens/elapsed:.1f}  "
                  f"time={elapsed:.1f}s\n")

            all_results[bench][str(exit_layer)] = {
                "mean_alpha": mean_alpha,
                "tokens_per_sec": bench_tokens / elapsed,
                "total_tokens": bench_tokens,
                "total_rounds": bench_rounds,
            }

            # Save incrementally so partial progress survives walltime kill
            out_path = os.path.join(args.output_dir, f"{args.tag}.json")
            with open(out_path, 'w') as f:
                json.dump({"tag": args.tag, "exit_layers": exit_layers,
                           "results": all_results}, f, indent=2)

    # Final summary table
    print(f"\n{'='*70}")
    print(f"  LAYER-SKIP SUMMARY: {args.tag}")
    header = f"  {'bench':<12}" + "".join(f"  L{e:>3}" for e in exit_layers)
    print(header)
    for bench, per_exit in all_results.items():
        row = "  " + f"{bench:<12}" + "".join(
            f"  {per_exit.get(str(e), {}).get('mean_alpha', 0):>4.2f}"
            for e in exit_layers)
        print(row)
    print(f"{'='*70}\n")
    print(f"[EVAL] Saved to {out_path}")


if __name__ == "__main__":
    main()
