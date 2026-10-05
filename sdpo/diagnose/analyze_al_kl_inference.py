"""
TRUE-inference AL_KL vs accept-length correlation on EAGLE-3 chain SD.

Run real chain-mode SD (γ-step linear chain) with EAGLE-3 draft + Qwen3-8B
target, log per-round (hard_τ, AL_KL, AL_TV). Both target & draft use
proper KV caches advanced per round; target re-runs on draft's proposed
sequence (no teacher-forcing).

Usage:
  python sdpo/diagnose/analyze_al_kl_inference.py \
    --base-model-path Qwen/Qwen3-8B \
    --ea-model-path /scratch/.../q8_eagle3_kl_regen_l2k_6ep/state_5 \
    --bench-name mt_bench --num-questions 10 --gamma 6 \
    --output diagnose_al_kl_inf_kl.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--base-model-path', required=True)
    p.add_argument('--ea-model-path', required=True)
    p.add_argument('--bench-name', default='mt_bench')
    p.add_argument('--num-questions', type=int, default=10)
    p.add_argument('--gamma', type=int, default=6, help='Chain depth.')
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--output', default='diagnose_al_kl_inf.json')
    return p.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    from transformers import AutoTokenizer
    from model.ea_model import EaModel

    print(f"[INF-ANALYZE] Loading EaModel (chain mode, γ={args.gamma})")
    model = EaModel.from_pretrained(
        base_model_path=args.base_model_path,
        ea_model_path=args.ea_model_path,
        total_token=args.gamma + 1,
        depth=args.gamma,
        top_k=1,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map='auto',
        use_eagle3=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path,
                                              trust_remote_code=True)

    qfile = os.path.join(_ROOT, 'data', args.bench_name, 'question.jsonl')
    questions = [json.loads(l) for l in open(qfile)]
    questions = questions[:args.num_questions]
    print(f"[INF-ANALYZE] Loaded {len(questions)} questions from {args.bench_name}")

    system_msg = "You are a helpful, respectful and honest assistant."
    rounds_data = []

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
        n = _run_chain_sd(model, input_ids, args.gamma,
                          args.max_new_tokens, rounds_data, tokenizer)
        print(f"  q{qi}: {n} rounds, last τ={rounds_data[-1][0] if rounds_data else '-'}")

    M = np.array(rounds_data)
    print(f"\n[INF-ANALYZE] Total {M.shape[0]} rounds")
    cols = ['hard_τ', 'AL_KL', 'AL_TV']
    print(f"  {'metric':<10}{'mean':>9}{'std':>9}{'min':>7}{'max':>7}")
    for i, c in enumerate(cols):
        print(f"  {c:<10}{M[:,i].mean():>9.3f}{M[:,i].std():>9.3f}"
              f"{M[:,i].min():>7.2f}{M[:,i].max():>7.2f}")

    print(f"\n  Correlations vs AL_KL (col 1):")
    print(f"  {'truth':<10}{'pearson':>10}{'spearman':>11}")
    for i, c in [(0, 'hard_τ'), (2, 'AL_TV')]:
        pe = pearsonr(M[:, 1], M[:, i])
        sp = spearmanr(M[:, 1], M[:, i])
        print(f"  {c:<10}{pe.statistic:>10.4f}{sp.statistic:>11.4f}")

    results = {
        'meta': {'ea_ckpt': args.ea_model_path, 'gamma': args.gamma,
                 'n_rounds': int(M.shape[0])},
        'stats': {c: {'mean': float(M[:,i].mean()), 'std': float(M[:,i].std())}
                  for i, c in enumerate(cols)},
        'correlations': {
            f'AL_KL_vs_{c}': {
                'pearson': float(pearsonr(M[:, 1], M[:, i]).statistic),
                'spearman': float(spearmanr(M[:, 1], M[:, i]).statistic),
            } for i, c in [(0, 'hard_tau'), (2, 'AL_TV')]
        },
        'data': M.tolist(),
    }
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[INF-ANALYZE] Wrote {args.output}")


def _target_forward_with_hidden(target, input_ids, past_key_values=None,
                                 position_ids=None):
    """Target forward returning (logits, kv, eagle3_hidden_concat)."""
    out = target(input_ids=input_ids, past_key_values=past_key_values,
                 position_ids=position_ids, use_cache=True,
                 output_hidden_states=True)
    hs = out.hidden_states
    N = len(hs) - 1  # excluding embeddings
    # EAGLE-3 AngelSlim Qwen3-8B uses [2, N//2, N-3]
    concat = torch.cat([hs[2], hs[N // 2], hs[N - 3]], dim=-1)
    return out.logits, out.past_key_values, concat


def _trunc_kv(kv, keep_len):
    """Truncate KV cache tuple to keep_len positions."""
    return tuple((k[..., :keep_len, :], v[..., :keep_len, :]) for k, v in kv)


def _run_chain_sd(model, input_ids, gamma, max_new_tokens, rounds_data, tokenizer):
    """One question. Run chain SD with γ-step draft autoregressive, target
    one-shot verify per round. Append (τ, AL_KL, AL_TV) per round."""
    target = model.base_model
    draft = model.ea_layer
    device = input_ids.device

    eos = getattr(model.base_model.config, 'eos_token_id', None)
    eos_set = set(eos) if isinstance(eos, list) else ({eos} if eos else set())

    L = input_ids.shape[1]

    # ── Prefill target on full prompt ────────────────────────────────────
    t_logits, t_kv, t_hidden_concat = _target_forward_with_hidden(
        target, input_ids)
    # Last logit (= target's prediction of token L); needed for first round's
    # "what would target predict from pending"
    # Actually: pending = input_ids[:, -1:]; target's next-token prediction
    # comes from running target with pending in cache. Already done; t_logits[-1] is it.
    cache_len = t_kv[0][0].shape[2]  # = L

    # ── Prefill draft on prompt[:-1] (predicts at each position p the (p+1)-th token) ──
    # Following topK_genrate convention: input_ids drops first token, hidden_states
    # is full sequence; draft predicts what comes after.
    # For our prefill, we want draft's KV up to position L-1, last hidden output usable
    # as the "previous out_hidden" for round 1 step 0.
    draft_input_ids = input_ids[:, 1:]                       # length L-1, predicts pos 1..L-1
    draft_hidden_in = t_hidden_concat[:, :-1, :]             # length L-1
    d_out_hidden, d_kv = draft(draft_hidden_in, draft_input_ids,
                                use_cache=True)
    # draft's prediction at last position (= prediction of token L, which is pending)
    # Save it for round logic.
    n_generated = 0
    n_rounds_q = 0

    while n_generated < max_new_tokens:
        # ── Draft autoregressive γ steps from pending ──
        # Round step 0 input: pending token + last target hidden at pending position.
        # The "pending hidden" is t_hidden_concat[:, -1:, :] (target's hidden at pos L-1
        # given prefix predicts pending = input_ids[L-1]).
        pending = input_ids[:, -1:]               # the token whose next we want to predict
        pending_hidden = t_hidden_concat[:, -1:, :]  # target hidden at pending position

        draft_logits_list = []
        draft_tokens_list = []
        prev_out_hidden = pending_hidden
        cur_token = pending
        cur_kv = d_kv

        for step in range(gamma):
            # Draft forward: prev hidden (1 token) + current token → next prediction
            cur_pos = torch.tensor(
                [[cur_kv[0][0].shape[2]]], device=device, dtype=torch.long)
            out_hidden, cur_kv = draft(
                prev_out_hidden, cur_token,
                past_key_values=cur_kv,
                position_ids=cur_pos,
                use_cache=True)
            logits = draft.lm_head(draft.norm(out_hidden[:, -1])).float()
            draft_argmax_d = logits.argmax(-1)                   # draft vocab
            draft_argmax_full = draft_argmax_d + draft.d2t[draft_argmax_d]
            draft_logits_list.append(logits)                     # [1, V_draft]
            draft_tokens_list.append(draft_argmax_full)
            # Next step inputs
            prev_out_hidden = out_hidden[:, -1:, :]
            cur_token = draft_argmax_full.view(1, 1)

        draft_tokens = torch.stack(draft_tokens_list, dim=-1).view(1, -1)  # [1, γ]
        draft_logits_d = torch.stack(draft_logits_list, dim=1)             # [1, γ, V_draft]

        # ── Target one-shot verify on [pending, d1..dγ] ──
        verify_input = torch.cat([pending, draft_tokens], dim=1)            # [1, γ+1]
        verify_pos = torch.arange(cache_len, cache_len + gamma + 1,
                                  device=device).unsqueeze(0)
        # But pending is ALREADY in t_kv (it was the last token of prefill).
        # We need target's next prediction GIVEN pending in cache. To do this
        # cleanly, we re-run with verify input but the cache already has prefix
        # up to pending position. Slot positions: cache_len..cache_len+γ.
        # The first verify position is "pending again" — that's wrong.
        # Fix: drop pending from verify_input; cache already includes it.
        # Then verify_input = draft_tokens, positions cache_len..cache_len+γ-1.
        # Target logits at position cache_len+j-1 = target's prediction of token at
        # absolute position cache_len+j, which is the (j+1)-th draft token.
        # We want target_logits at each pred position (γ predictions for γ draft tokens):
        #   pred at j: given prefix + d_1..d_j, predict next
        # For accept check: compare d_{j+1} to target_argmax[j]
        # Wait, we need GROUND-TRUTH (= target's argmax) for each of the γ draft tokens.
        # The first draft token d_1 is the prediction of "what comes after pending".
        # target's prediction of "what comes after pending" comes from the LAST logit
        # of prefill = t_logits[:, -1, :]. So we already have it.
        # For d_2: target sees [pending, d_1], predicts next. Need to run target.
        # For γ-step accept, we need γ target predictions, one per draft position.
        # Run target on draft_tokens (γ tokens), positions cache_len..cache_len+γ-1.
        verify_input = draft_tokens
        verify_pos = torch.arange(cache_len, cache_len + gamma,
                                  device=device).unsqueeze(0)
        t_verify_logits, t_kv_new, t_hidden_new = _target_forward_with_hidden(
            target, verify_input, past_key_values=t_kv, position_ids=verify_pos)
        # Target's prediction at position j (0..γ-1) given prefix + d_1..d_j
        # Concretely: t_verify_logits[0, j] is target's logit for token (cache_len+j+1)
        # given KV cache + d_{1..j+1}.
        # Wait actually: target sees verify_input = [d_1, d_2, ..., d_γ]. Position j
        # in the forward sees prefix + d_{1..j+1}. So t_verify_logits[0, j] predicts
        # token at position cache_len + j + 1, given prefix + d_{1..j+1}.
        # For accept check at step j+1: compare d_{j+1} to target's predicted (j+1)-th
        # token. target's predicted (j+1)-th token from the prefill last logit
        # (for j=0) and from t_verify_logits[0, j-1] for j>=1.

        # Stack target's per-step predictions:
        #   For j = 0..γ-1, the prediction of d_{j+1} is:
        #     if j == 0: t_logits[:, -1, :]  (prefill last logit)
        #     else: t_verify_logits[:, j-1, :]
        # Easier: concat [t_logits[:, -1, :], t_verify_logits[0, :-1, :]] → [γ, V_target]
        target_preds = torch.cat(
            [t_logits[:, -1:, :], t_verify_logits[:, :-1, :]], dim=1)[0].float()  # [γ, V_target]
        t_argmax = target_preds.argmax(-1)
        t_argmax_cpu = t_argmax.cpu().tolist()
        draft_cpu = draft_tokens[0].cpu().tolist()

        n_acc = 0
        for j in range(gamma):
            if t_argmax_cpu[j] == draft_cpu[j]:
                n_acc += 1
            else:
                break

        # ── Compute AL_KL, AL_TV using target distribution at draft-proposed context ──
        t2d_bool = draft.t2d.bool()
        tgt_logp_d = F.log_softmax(target_preds[:, t2d_bool], dim=-1)        # [γ, V_draft]
        tgt_p_d = tgt_logp_d.exp()
        draft_logp = F.log_softmax(draft_logits_d[0].float(), dim=-1)        # [γ, V_draft]
        draft_q = draft_logp.exp()

        kl = (tgt_p_d * (tgt_logp_d - draft_logp)).sum(-1)
        beta_KL = (0.5 * torch.exp(-kl)).clamp(min=0, max=1)
        beta_TV = torch.sum(torch.min(tgt_p_d, draft_q), dim=-1)

        AL_KL = beta_KL.cumprod(0).sum().item()
        AL_TV = beta_TV.cumprod(0).sum().item()

        rounds_data.append([n_acc, AL_KL, AL_TV])
        n_rounds_q += 1

        # ── Advance: accept n_acc draft tokens + correction (target_argmax at first reject) ──
        accepted = draft_tokens[:, :n_acc]
        # Correction = target_argmax at position n_acc (the first reject's correct token)
        correction = t_argmax[n_acc:n_acc + 1].view(1, 1)
        new_tokens = torch.cat([accepted, correction], dim=1)
        n_generated += new_tokens.shape[1]

        # Stop on EOS
        if any(int(t.item()) in eos_set for t in new_tokens[0]):
            break

        # ── Update target KV cache: keep prefix + n_acc accepted ──
        # t_kv_new has prefix + all γ draft tokens. We keep prefix + first n_acc.
        keep_len = cache_len + n_acc
        t_kv = _trunc_kv(t_kv_new, keep_len)
        # Append correction by running target on [correction] to extend cache to keep_len+1
        corr_pos = torch.tensor([[keep_len]], device=device, dtype=torch.long)
        t_logits, t_kv, t_hidden_concat_corr = _target_forward_with_hidden(
            target, correction, past_key_values=t_kv, position_ids=corr_pos)
        cache_len = keep_len + 1

        # ── Update draft KV cache: keep its prefill + n_acc accepted positions ──
        d_kv = _trunc_kv(cur_kv, d_kv[0][0].shape[2] + n_acc)

        # Update input_ids tail (logically: we appended accepted+correction tokens)
        input_ids = torch.cat([input_ids, new_tokens], dim=1)

        # ── Run draft on correction token to keep KV in sync ──
        # The "next pending" for the new round is correction.
        # Draft needs to see (concat_hidden of corr position) + correction token to
        # produce its next prediction. We feed corr token into draft's KV with
        # corr's target_hidden.
        d_corr_pos = torch.tensor(
            [[d_kv[0][0].shape[2]]], device=device, dtype=torch.long)
        d_out_hidden, d_kv = draft(
            t_hidden_concat_corr, correction,
            past_key_values=d_kv, position_ids=d_corr_pos,
            use_cache=True)
        # Now t_hidden_concat for next round's pending_hidden:
        t_hidden_concat = t_hidden_concat_corr

    return n_rounds_q


if __name__ == '__main__':
    main()
