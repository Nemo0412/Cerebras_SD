"""
AL_KL surrogate vs actual accept-length correlation via hooks on the EAGLE-3
inference engine (bapo/eval.py-style flow).

NO changes to bapo/eval.py, model/ea_model.py, model/cnets.py, model/utils.py.
We monkey-patch evaluate_posterior + register forward hook on ea_layer.lm_head
to capture per-round (target_logits along accepted path, draft_logits, τ).

Usage:
  python sdpo/diagnose/al_kl_inference_hook.py \
    --base-model-path Qwen/Qwen3-8B \
    --ea-model-path /scratch/.../q8_eagle3_alkl_only_6ep/state_5 \
    --bench-name mt_bench --num-questions 10 \
    --output diagnose_al_kl_hook_alkl.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--base-model-path', required=True)
    p.add_argument('--ea-model-path', required=True)
    p.add_argument('--bench-name', default='mt_bench')
    p.add_argument('--num-questions', type=int, default=10)
    p.add_argument('--top-k', type=int, default=10)
    p.add_argument('--depth', type=int, default=5)
    p.add_argument('--total-token', type=int, default=60)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--output', default='diagnose_al_kl_hook.json')
    return p.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    from transformers import AutoTokenizer
    from model.ea_model import EaModel
    import model.utils as mu

    print(f"[HOOK] Loading EaModel (top_k={args.top_k}, depth={args.depth})")
    model = EaModel.from_pretrained(
        base_model_path=args.base_model_path,
        ea_model_path=args.ea_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map='auto',
        use_eagle3=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path,
                                              trust_remote_code=True)

    # ── HOOK 1: capture draft lm_head outputs (= per-step draft logits) ──
    # Each topK_genrate call invokes lm_head γ+1 times (1 root + γ tree depth).
    # We collect all calls into a flat list, then reset per round.
    draft_logits_buffer = []

    def lm_head_hook(module, input, output):
        # output: [B, S, V_draft] — draft's logits at draft positions
        draft_logits_buffer.append(output.detach())

    lm_head_handle = model.ea_layer.lm_head.register_forward_hook(lm_head_hook)

    # ── HOOK 2: wrap evaluate_posterior to capture target logits + path ──
    target_log_buffer = []  # per-round: (logits, candidates, best_candidate, accept_length)
    orig_eval = mu.evaluate_posterior

    def patched_eval(logits, candidates, logits_processor):
        result = orig_eval(logits, candidates, logits_processor)
        best_candidate, accept_length, sample_p = result
        target_log_buffer.append({
            'logits': logits.detach().clone(),  # [n_paths, γ+1, V_target]
            'candidates': candidates.detach().clone(),  # [n_paths, γ+1]
            'best': int(best_candidate),
            'accept_length': int(accept_length),
            'n_draft_calls': len(draft_logits_buffer),  # how many lm_head invocations seen
        })
        return result

    mu.evaluate_posterior = patched_eval
    # Also need to patch the imported reference inside ea_model
    import model.ea_model as em
    em.evaluate_posterior = patched_eval

    qfile = os.path.join(_ROOT, 'data', args.bench_name, 'question.jsonl')
    questions = [json.loads(l) for l in open(qfile)][:args.num_questions]
    print(f"[HOOK] {len(questions)} questions from {args.bench_name}")

    system_msg = "You are a helpful, respectful and honest assistant."
    all_rounds = []  # (τ, AL_KL, AL_TV)

    for qi, q in enumerate(questions):
        msgs = [{"role": "system", "content": system_msg},
                {"role": "user", "content": q['turns'][0]}]
        try:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer([text], add_special_tokens=False,
                              return_tensors='pt').input_ids.cuda()

        # Reset buffers
        draft_logits_buffer.clear()
        target_log_buffer.clear()

        # Run eagenerate (same as bapo/eval.py)
        t0 = time.time()
        output_ids, new_token, idx = model.eagenerate(
            input_ids, temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            log=True, is_llama3=True)
        elapsed = time.time() - t0

        # Process per-round data
        # draft_logits_buffer is a flat list of lm_head outputs.
        # Each round's topK_genrate produces some number of calls.
        # target_log_buffer[r]['n_draft_calls'] is cumulative count after round r.
        prev_n_draft = 0
        for r_idx, tl in enumerate(target_log_buffer):
            n_draft = tl['n_draft_calls']
            round_draft_logits = draft_logits_buffer[prev_n_draft:n_draft]
            prev_n_draft = n_draft
            # round_draft_logits: list of γ+1 tensors (raw logits before logsoftmax)
            # Each shape [1, S, V_draft] where S varies; we want per-step distribution

            # Build accepted path's draft + target distributions
            best = tl['best']
            n_acc = tl['accept_length']
            target_logits = tl['logits']  # [n_paths, max_depth+1, V_target]
            candidates = tl['candidates']  # [n_paths, max_depth+1]

            if n_acc == 0 or len(round_draft_logits) == 0:
                # No accepted tokens this round; skip or log τ=0
                all_rounds.append([0, 0.0, 0.0])
                continue

            # Target logits at the accepted path positions
            # target_logits[best, 0..n_acc-1] = target's predictions of tokens
            # at positions 1..n_acc along the chosen path
            tgt_lg_path = target_logits[best, :n_acc].float()    # [n_acc, V_target]
            tgt_path_tokens = candidates[best, :n_acc + 1]       # accepted token IDs

            # Map target distribution to draft vocab via t2d
            t2d_bool = model.ea_layer.t2d.bool()
            tgt_logp_d = F.log_softmax(tgt_lg_path[:, t2d_bool], dim=-1)
            tgt_p_d = tgt_logp_d.exp()

            # Draft logits along accepted path: we need to find them in the
            # flat draft_logits buffer. Skip — too complex; instead approximate
            # by assuming draft's per-step distribution is the topK_genrate's
            # last_headout at the corresponding tree depth.
            # For simplicity: collect ALL draft lm_head outputs this round,
            # take the first n_acc that correspond to the accepted path depths.
            # In chain mode (top_k=1, n_paths=1), there's 1 draft logit per depth.
            # For general tree, n_paths > 1 and depths share.
            # Pragmatic: use the FIRST n_acc draft logits (rough approximation for tree).
            #
            # Better: in chain mode (top_k=1), draft_logits[d] is the prediction at depth d.
            # In tree mode, draft_logits has multiple per depth (one per parent).
            # For correlation purposes, use the first n_acc which correspond to the accepted-prefix-path predictions.

            # Each lm_head call in topK_genrate produces 2-D output [N, V_draft]:
            #   - initial step (line 700): N=1
            #   - depth loop (line 734): N=top_k (chain: 1, tree: top_k)
            # For chain (top_k=1), each call gives [1, V_draft] → take it directly.
            # For tree, each call gives [top_k, V_draft] → take first one (root of path).
            draft_last_logits = []
            for dlg in round_draft_logits:
                if dlg.dim() == 2:
                    draft_last_logits.append(dlg[:1])    # [1, V_draft]
                elif dlg.dim() == 3:
                    draft_last_logits.append(dlg[:, -1, :])
            if len(draft_last_logits) == 0:
                all_rounds.append([n_acc, 0.0, 0.0])
                continue
            if len(draft_last_logits) < n_acc:
                # not enough draft logits captured (tree internal state)
                # fall back: use what we have
                pass
            draft_lg_path = torch.cat(draft_last_logits[:n_acc], dim=0).float()  # [≤n_acc, V_draft]
            # If shorter than n_acc, truncate target to match
            n_eff = draft_lg_path.shape[0]
            if n_eff == 0:
                all_rounds.append([n_acc, 0.0, 0.0])
                continue
            tgt_logp_d = tgt_logp_d[:n_eff]
            tgt_p_d = tgt_p_d[:n_eff]

            draft_logp = F.log_softmax(draft_lg_path, dim=-1)
            draft_q = draft_logp.exp()

            kl = (tgt_p_d * (tgt_logp_d - draft_logp)).sum(-1).clamp(min=0)
            beta_KL = (0.5 * torch.exp(-kl)).clamp(min=0, max=1)
            beta_TV = torch.sum(torch.min(tgt_p_d, draft_q), dim=-1)

            AL_KL = beta_KL.cumprod(0).sum().item()
            AL_TV = beta_TV.cumprod(0).sum().item()
            all_rounds.append([n_acc, AL_KL, AL_TV])

        print(f"  q{qi}: {len(target_log_buffer)} rounds, "
              f"α={new_token/(idx+1):.2f}, {elapsed:.1f}s")

    # Cleanup hooks
    lm_head_handle.remove()
    mu.evaluate_posterior = orig_eval
    em.evaluate_posterior = orig_eval

    M = np.array(all_rounds) if all_rounds else np.zeros((0, 3))
    print(f"\n[HOOK] Total {M.shape[0]} rounds")
    cols = ['hard_τ', 'AL_KL', 'AL_TV']
    for i, c in enumerate(cols):
        print(f"  {c:<10}mean={M[:,i].mean():.3f}  std={M[:,i].std():.3f}  "
              f"min={M[:,i].min():.2f}  max={M[:,i].max():.2f}")

    if M.shape[0] >= 2:
        print(f"\n  Correlations vs AL_KL:")
        print(f"  {'truth':<10}{'pearson':>10}{'spearman':>11}")
        for i, c in [(0, 'hard_τ'), (2, 'AL_TV')]:
            pe = pearsonr(M[:, 1], M[:, i])
            sp = spearmanr(M[:, 1], M[:, i])
            print(f"  {c:<10}{pe.statistic:>10.4f}{sp.statistic:>11.4f}")

    results = {
        'meta': {'ea_ckpt': args.ea_model_path, 'top_k': args.top_k,
                 'depth': args.depth, 'n_rounds': int(M.shape[0])},
        'stats': {c: {'mean': float(M[:,i].mean()), 'std': float(M[:,i].std())}
                  for i, c in enumerate(cols)} if M.shape[0] else {},
        'correlations': {
            f'AL_KL_vs_{c}': {
                'pearson': float(pearsonr(M[:, 1], M[:, i]).statistic),
                'spearman': float(spearmanr(M[:, 1], M[:, i]).statistic),
            } for i, c in [(0, 'hard_tau'), (2, 'AL_TV')]
        } if M.shape[0] >= 2 else {},
    }
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[HOOK] Wrote {args.output}")


if __name__ == '__main__':
    main()
