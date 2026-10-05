"""Per-SD-window AL surrogate vs actual accept length on real Qwen3-8B/0.6B SD.

Adapted from sdpo/eval_small_lm.py: runs actual greedy speculative decoding,
captures draft_logits (γ steps) + target_logits at the verify forward, then per
SD round computes:
  - hard_tau : actual accept length n_accepted (ground truth)
  - V4       : Σ_k ∏_{j<k} 2σ((z_draft[target_argmax_j] - max(z_draft_j))/T)
  - EAL      : Σ_k ∏_{j<k} P_target(draft_argmax_j)              (soft EAL, qx=1.0)
  - AL_KL    : Σ_k ∏_{j<k} 0.5·exp(-KL(p_target_j || p_draft_j))
  - AL_TV    : Σ_k ∏_{j<k} Σ_v min(p_target_j[v], p_draft_j[v])  (= 1 - TV)

Per-round Pearson / Spearman correlation with hard_tau printed at end.

Usage:
  python sdpo/diagnose/al_correlation_smalllm_inference.py \\
    --base-model-path Qwen/Qwen3-8B \\
    --draft-model-path Qwen/Qwen3-0.6B \\
    --num-samples 5 --gamma 7 --max-new-tokens 256
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.inference_mode()
def speculative_decode_step_with_hook(target_model, draft_model, input_ids,
                                       gamma, T_v4=1.0):
    """One SD round, returns new_tokens, n_accepted, and surrogate dict."""
    device = input_ids.device
    prefix_len = input_ids.shape[1]

    # Draft: γ autoregressive forwards, collect last-position logits per step
    draft_ids = input_ids.clone()
    draft_logits_chain = []   # [γ] tensors of [V_draft]
    draft_tokens_list = []
    for _ in range(gamma):
        out = draft_model(draft_ids)
        drf_lg = out.logits[0, -1, :].float()   # [V]
        nxt = drf_lg.argmax(-1, keepdim=True)
        draft_logits_chain.append(drf_lg)
        draft_tokens_list.append(nxt)
        draft_ids = torch.cat([draft_ids, nxt.unsqueeze(0)], dim=1)
    draft_tokens = torch.cat(draft_tokens_list, dim=0)   # [γ]

    # Target: one verify forward on prefix + draft chain
    target_logits = target_model(draft_ids).logits[0].float()   # [prefix+γ, V]

    # Accept loop (greedy)
    n_accepted = 0
    for i in range(gamma):
        pos = prefix_len + i - 1
        tgt_argmax = target_logits[pos].argmax(-1)
        if tgt_argmax.item() == draft_tokens[i].item():
            n_accepted += 1
        else:
            break

    # Correction token
    correction_pos = prefix_len + n_accepted - 1
    correction_token = target_logits[correction_pos].argmax(-1, keepdim=True)
    new_tokens = torch.cat(
        [draft_tokens[:n_accepted].unsqueeze(0), correction_token.unsqueeze(0)],
        dim=1)

    # Per-step β values, all over the SAME Qwen3 vocab
    beta_v4 = torch.zeros(gamma, device=device)
    beta_eal = torch.zeros(gamma, device=device)
    beta_kl = torch.zeros(gamma, device=device)
    beta_tv = torch.zeros(gamma, device=device)

    for j in range(gamma):
        drf_lg = draft_logits_chain[j]                       # [V]
        tgt_lg = target_logits[prefix_len + j - 1]            # [V] (target's pred at this step)
        drf_p = F.softmax(drf_lg, dim=-1)
        tgt_p = F.softmax(tgt_lg, dim=-1)
        tgt_logp = F.log_softmax(tgt_lg, dim=-1)
        drf_logp = F.log_softmax(drf_lg, dim=-1)

        drf_argmax = drf_lg.argmax(-1)
        tgt_argmax = tgt_lg.argmax(-1)

        # V4: 2σ((z_d[t*] - max(z_d))/T)
        z_d_tstar = drf_lg[tgt_argmax]
        z_d_max = drf_lg.max()
        gap = (z_d_tstar - z_d_max) / T_v4
        beta_v4[j] = (2.0 * torch.sigmoid(gap)).clamp(0, 1)

        # EAL (soft): P_target(draft_argmax)
        beta_eal[j] = tgt_p[drf_argmax]

        # AL_KL: 0.5 · exp(-KL(p_target || p_draft))
        kl = (tgt_p * (tgt_logp - drf_logp)).sum().clamp(min=0)
        beta_kl[j] = (0.5 * torch.exp(-kl)).clamp(0, 1)

        # AL_TV: Σ min(p_t, p_d)
        beta_tv[j] = torch.min(tgt_p, drf_p).sum()

    surrogates = {
        "V4":    torch.cumprod(beta_v4, 0).sum().item(),
        "EAL":   torch.cumprod(beta_eal, 0).sum().item(),
        "AL_KL": torch.cumprod(beta_kl, 0).sum().item(),
        "AL_TV": torch.cumprod(beta_tv, 0).sum().item(),
    }
    return new_tokens, n_accepted, surrogates


@torch.inference_mode()
def generate_with_hook(target_model, draft_model, input_ids, max_new_tokens,
                       gamma, T_v4):
    rounds = []   # list of {"hard_tau": k, "V4":..., "EAL":..., ...}
    total_tokens = 0
    cur_ids = input_ids
    eos = target_model.config.eos_token_id
    if not isinstance(eos, list):
        eos = [eos] if eos is not None else []

    while total_tokens < max_new_tokens:
        new_tokens, n_acc, surr = speculative_decode_step_with_hook(
            target_model, draft_model, cur_ids, gamma, T_v4=T_v4)
        cur_ids = torch.cat([cur_ids, new_tokens], dim=1)
        total_tokens += new_tokens.shape[1]
        surr["hard_tau"] = n_acc
        rounds.append(surr)
        if any(t in new_tokens[0].tolist() for t in eos):
            break
    return cur_ids, rounds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model-path", default="Qwen/Qwen3-8B")
    p.add_argument("--draft-model-path", default="Qwen/Qwen3-0.6B")
    p.add_argument("--bench-name", default="mt_bench")
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--gamma", type=int, default=7)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--T-v4", type=float, default=1.0,
                   help="Temperature for V4 sigmoid gap")
    p.add_argument("--output", default="al_correlation_smalllm.json")
    args = p.parse_args()

    print(f"[AL-CORR] target={args.base_model_path}  draft={args.draft_model_path}")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, torch_dtype=torch.float16, device_map="auto")
    target_model.eval()
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model_path, torch_dtype=torch.float16, device_map="auto")
    draft_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path,
                                              trust_remote_code=True)

    qpath = f"data/{args.bench_name}/question.jsonl"
    with open(qpath) as f:
        questions = [json.loads(l) for l in f][:args.num_samples]
    print(f"[AL-CORR] {len(questions)} questions from {args.bench_name}")

    all_rounds = []
    for qi, q in enumerate(tqdm(questions)):
        prompt = q["turns"][0]
        msgs = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(text, return_tensors="pt",
                              add_special_tokens=False).input_ids
        input_ids = input_ids.to(target_model.device)

        t0 = time.time()
        _, rounds = generate_with_hook(
            target_model, draft_model, input_ids, args.max_new_tokens,
            args.gamma, T_v4=args.T_v4)
        elapsed = time.time() - t0
        a = np.mean([r["hard_tau"] for r in rounds]) if rounds else 0.0
        print(f"  q{qi}: {len(rounds)} rounds  α={a:.2f}  {elapsed:.1f}s")
        all_rounds.extend(rounds)

    if not all_rounds:
        print("[AL-CORR] No rounds collected.")
        return

    # Aggregate
    cols = ["hard_tau", "V4", "EAL", "AL_KL", "AL_TV"]
    M = np.array([[r[c] for c in cols] for r in all_rounds])
    print(f"\n[AL-CORR] Total {M.shape[0]} rounds")
    for i, c in enumerate(cols):
        print(f"  {c:<8}  mean={M[:,i].mean():.3f}  std={M[:,i].std():.3f}  "
              f"min={M[:,i].min():.2f}  max={M[:,i].max():.2f}")

    print(f"\n  Correlations vs hard_tau (ground truth):")
    print(f"  {'metric':<8} {'pearson':>10} {'spearman':>11}")
    correlations = {}
    if M.shape[0] >= 2:
        for i, c in enumerate(cols[1:], start=1):
            pe = pearsonr(M[:, 0], M[:, i])
            sp = spearmanr(M[:, 0], M[:, i])
            print(f"  {c:<8} {pe.statistic:>10.4f} {sp.statistic:>11.4f}")
            correlations[c] = {"pearson": float(pe.statistic),
                               "spearman": float(sp.statistic)}

    result = {
        "meta": {"target": args.base_model_path,
                 "draft": args.draft_model_path,
                 "gamma": args.gamma, "T_v4": args.T_v4,
                 "n_rounds": int(M.shape[0])},
        "stats": {c: {"mean": float(M[:, i].mean()),
                      "std": float(M[:, i].std())} for i, c in enumerate(cols)},
        "correlations_vs_hard_tau": correlations,
        "rounds": all_rounds,
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[AL-CORR] Wrote {args.output}")


if __name__ == "__main__":
    main()
