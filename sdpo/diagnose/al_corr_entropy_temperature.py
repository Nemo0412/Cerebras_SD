"""Per-SD-window surrogate vs hard_tau, stratified by target/draft ENTROPY and
DECODING MODE (greedy / stochastic-sampling) at given TEMPERATURE T.

Extension of al_correlation_smalllm_inference.py. For each SD window:
  - Captures draft + target logits at each of γ steps.
  - Computes per-step H(p_target^T), H(q_draft^T), KL(p ‖ q).
  - Computes window-level surrogates:
      AL_TV  = Σ_k cumprod(1 − TV(p^T_j, q^T_j))                                — sampling-mode E[τ]
      AL_KL  = Σ_k cumprod(0.5 · exp(−KL(p^T_j ‖ q^T_j)))                       — BH lower bound
      WKL    = Σ_j (γ − j) · KL(p^T_j ‖ q^T_j)                                  — linear-decay weighted KL (training loss)
      EAL    = Σ_k cumprod(p^T_target(draft_token_j))                            — soft greedy accept (qx=1)
  - Decoding:
      mode=greedy: T=0, draft picks argmax, target accept iff draft_argmax = target_argmax.
      mode=sample: T>0, draft samples from q^T, target accepts with prob min(1, p^T/q^T).
  - hard_tau = actual #accepted draft tokens in this window.

Analysis at end:
  - Overall mean and correlation (pearson/spearman) of each surrogate vs hard_tau.
  - Per-entropy-bucket correlation (rounds bucketed by mean H(p_target) over γ steps).

Usage:
  python sdpo/diagnose/al_corr_entropy_temperature.py \\
    --base-model-path Qwen/Qwen3-8B \\
    --draft-model-path Qwen/Qwen3-0.6B \\
    --num-samples 5 --gamma 7 --max-new-tokens 256 \\
    --mode sample --temperature 1.0 \\
    --output diagnose_ent_T1_sample.json
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def _temp_softmax(lg, T):
    """softmax(lg / T) with T=0 → one-hot at argmax."""
    if T <= 0:
        idx = lg.argmax(-1, keepdim=True)
        out = torch.zeros_like(lg)
        out.scatter_(-1, idx, 1.0)
        return out
    return F.softmax(lg / T, dim=-1)


@torch.inference_mode()
def speculative_decode_step(target_model, draft_model, input_ids, gamma,
                            mode="greedy", T=0.0):
    """One SD round under given mode/temperature.

    Returns new_tokens [1, n_accepted+1], n_accepted, per-window metrics dict.
    """
    device = input_ids.device
    prefix_len = input_ids.shape[1]

    # Draft: γ autoregressive forwards
    draft_ids = input_ids.clone()
    draft_logits_chain = []
    draft_tokens_list = []
    for _ in range(gamma):
        out = draft_model(draft_ids)
        drf_lg = out.logits[0, -1, :].float()
        if mode == "greedy" or T <= 0:
            nxt = drf_lg.argmax(-1)
        else:
            q = _temp_softmax(drf_lg, T)
            nxt = torch.multinomial(q, 1).squeeze(-1)
        draft_logits_chain.append(drf_lg)
        draft_tokens_list.append(nxt)
        draft_ids = torch.cat([draft_ids, nxt.view(1, 1)], dim=1)
    draft_tokens = torch.stack(draft_tokens_list, dim=0)    # [γ]

    # Target verify: one forward
    target_logits = target_model(draft_ids).logits[0].float()    # [prefix+γ, V]

    # Accept loop
    n_accepted = 0
    for i in range(gamma):
        pos = prefix_len + i - 1
        tgt_lg = target_logits[pos]
        drf_lg = draft_logits_chain[i]
        drafted = draft_tokens[i]
        if mode == "greedy" or T <= 0:
            if tgt_lg.argmax(-1).item() == drafted.item():
                n_accepted += 1
            else:
                break
        else:
            p_T = _temp_softmax(tgt_lg, T)
            q_T = _temp_softmax(drf_lg, T)
            ratio = (p_T[drafted] / (q_T[drafted] + 1e-12)).clamp(max=1.0)
            u = torch.rand((), device=device)
            if u < ratio:
                n_accepted += 1
            else:
                break

    # Correction
    correction_pos = prefix_len + n_accepted - 1
    tgt_lg_corr = target_logits[correction_pos]
    if mode == "greedy" or T <= 0 or n_accepted == gamma:
        correction = tgt_lg_corr.argmax(-1)
    else:
        # residual sampling: p_res ∝ max(p^T - q^T, 0)
        drf_lg_corr = draft_logits_chain[n_accepted]
        p_T = _temp_softmax(tgt_lg_corr, T)
        q_T = _temp_softmax(drf_lg_corr, T)
        p_res = (p_T - q_T).clamp(min=0)
        p_res_sum = p_res.sum()
        if p_res_sum < 1e-8:
            correction = p_T.argmax(-1)
        else:
            correction = torch.multinomial(p_res / p_res_sum, 1).squeeze(-1)

    new_tokens = torch.cat([draft_tokens[:n_accepted], correction.view(1)], dim=0)

    # Per-step metrics (always at the SAME temperature for surrogate consistency)
    T_eff = max(T, 1e-6) if mode == "sample" else 1.0   # entropy/KL at T=1 baseline for greedy
    h_p_list = []
    h_q_list = []
    kl_list = []
    beta_tv = torch.zeros(gamma, device=device)
    beta_kl = torch.zeros(gamma, device=device)
    beta_eal = torch.zeros(gamma, device=device)
    beta_v4 = torch.zeros(gamma, device=device)
    T_v4 = 1.0   # V4 sigmoid gap temperature (training-loss default)
    for j in range(gamma):
        drf_lg = draft_logits_chain[j]
        tgt_lg = target_logits[prefix_len + j - 1]
        p = _temp_softmax(tgt_lg, T_eff)
        q = _temp_softmax(drf_lg, T_eff)
        logp = torch.log(p.clamp(min=1e-30))
        logq = torch.log(q.clamp(min=1e-30))
        h_p = -(p * logp).sum().item()
        h_q = -(q * logq).sum().item()
        kl = (p * (logp - logq)).sum().clamp(min=0).item()
        h_p_list.append(h_p)
        h_q_list.append(h_q)
        kl_list.append(kl)
        beta_tv[j] = torch.min(p, q).sum()
        beta_kl[j] = (0.5 * math.exp(-kl))
        # EAL = P_target(draft_argmax) — uses TARGET-temperature distribution,
        # token = draft's sampled choice
        beta_eal[j] = p[draft_tokens[j]]
        # V4 = 2σ((z_draft[t*] - max z_draft) / T_v4)  where t* = target argmax
        tgt_argmax = tgt_lg.argmax(-1)
        z_d_tstar = drf_lg[tgt_argmax]
        z_d_max = drf_lg.max()
        beta_v4[j] = (2.0 * torch.sigmoid((z_d_tstar - z_d_max) / T_v4)).clamp(0, 1)

    AL_TV = torch.cumprod(beta_tv, 0).sum().item()
    AL_KL = torch.cumprod(beta_kl, 0).sum().item()
    EAL = torch.cumprod(beta_eal, 0).sum().item()
    V4 = torch.cumprod(beta_v4, 0).sum().item()
    # WKL = Σ_j (γ-j) · KL_j
    weights = [gamma - j for j in range(gamma)]
    WKL = sum(w * k for w, k in zip(weights, kl_list))

    metrics = {
        "hard_tau": n_accepted,
        "AL_TV": AL_TV, "AL_KL": AL_KL, "EAL": EAL, "V4": V4, "WKL": WKL,
        "mean_H_p": float(np.mean(h_p_list)),
        "mean_H_q": float(np.mean(h_q_list)),
        "mean_KL": float(np.mean(kl_list)),
    }
    return new_tokens, n_accepted, metrics


@torch.inference_mode()
def generate_with_hook(target_model, draft_model, input_ids, max_new_tokens,
                       gamma, mode, T):
    rounds = []
    cur_ids = input_ids
    total = 0
    eos = target_model.config.eos_token_id
    if not isinstance(eos, list):
        eos = [eos] if eos is not None else []
    while total < max_new_tokens:
        new_tokens, _, m = speculative_decode_step(
            target_model, draft_model, cur_ids, gamma, mode=mode, T=T)
        cur_ids = torch.cat([cur_ids, new_tokens.unsqueeze(0)], dim=1)
        total += new_tokens.shape[0]
        rounds.append(m)
        if any(t in new_tokens.tolist() for t in eos):
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
    p.add_argument("--mode", choices=["greedy", "sample"], default="greedy")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Sampling temperature (only used if mode=sample). "
                        "Always used to define p^T/q^T for AL_TV/AL_KL/WKL/H computation.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="al_corr_ent_temp.json")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[ENT-T] target={args.base_model_path}  draft={args.draft_model_path}")
    print(f"[ENT-T] mode={args.mode}  T={args.temperature}  γ={args.gamma}")
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
            args.gamma, mode=args.mode, T=args.temperature)
        elapsed = time.time() - t0
        a = np.mean([r["hard_tau"] for r in rounds]) if rounds else 0.0
        print(f"  q{qi}: {len(rounds)} rds  α={a:.2f}  H_p={np.mean([r['mean_H_p'] for r in rounds]):.2f}  {elapsed:.1f}s")
        all_rounds.extend(rounds)

    if not all_rounds:
        print("[ENT-T] No rounds."); return

    cols = ["hard_tau", "AL_TV", "AL_KL", "EAL", "V4", "WKL", "mean_H_p", "mean_H_q", "mean_KL"]
    M = np.array([[r[c] for c in cols] for r in all_rounds])
    print(f"\n[ENT-T] Total {M.shape[0]} rounds, mode={args.mode} T={args.temperature}")
    for i, c in enumerate(cols):
        print(f"  {c:<12} mean={M[:,i].mean():7.3f} std={M[:,i].std():7.3f} "
              f"min={M[:,i].min():7.2f} max={M[:,i].max():7.2f}")

    # Overall correlation
    print(f"\n  Overall correlation vs hard_tau (N={M.shape[0]}):")
    print(f"  {'metric':<10}{'pearson':>10}{'spearman':>11}")
    overall_corr = {}
    for c in ("AL_TV", "AL_KL", "EAL", "V4", "WKL"):
        i = cols.index(c)
        pe = pearsonr(M[:, 0], M[:, i])
        sp = spearmanr(M[:, 0], M[:, i])
        print(f"  {c:<10}{pe.statistic:>10.3f}{sp.statistic:>11.3f}")
        overall_corr[c] = {"pearson": float(pe.statistic),
                           "spearman": float(sp.statistic)}

    # Bucket by mean_H_p (tertiles)
    h_p = M[:, cols.index("mean_H_p")]
    edges = np.quantile(h_p, [1/3, 2/3])
    buckets = {
        "low_H":  h_p <= edges[0],
        "mid_H":  (h_p > edges[0]) & (h_p <= edges[1]),
        "high_H": h_p > edges[1],
    }
    bucket_corr = {}
    print(f"\n  Per-bucket correlation by mean_H_p (tertiles, edges={edges.round(2)}):")
    print(f"  {'bucket':<8}{'N':>5}{'<H_p>':>8}{'<τ>':>6}"
          f"{'AL_TV P/S':>14}{'AL_KL P/S':>14}{'EAL P/S':>14}{'V4 P/S':>14}{'WKL P/S':>14}")
    for bname, bmask in buckets.items():
        N = int(bmask.sum())
        if N < 5:
            print(f"  {bname:<8}{N:>5} (too few)")
            continue
        h_avg = h_p[bmask].mean()
        t_avg = M[bmask, 0].mean()
        line = f"  {bname:<8}{N:>5}{h_avg:>8.2f}{t_avg:>6.2f}"
        bucket_corr[bname] = {"N": N, "mean_H_p": float(h_avg), "mean_tau": float(t_avg)}
        for c in ("AL_TV", "AL_KL", "EAL", "V4", "WKL"):
            i = cols.index(c)
            pe = pearsonr(M[bmask, 0], M[bmask, i]).statistic
            sp = spearmanr(M[bmask, 0], M[bmask, i]).statistic
            line += f"  {pe:5.2f}/{sp:5.2f}"
            bucket_corr[bname][c] = {"pearson": float(pe), "spearman": float(sp)}
        print(line)

    result = {
        "meta": {"target": args.base_model_path, "draft": args.draft_model_path,
                 "mode": args.mode, "temperature": args.temperature,
                 "gamma": args.gamma, "n_rounds": int(M.shape[0])},
        "stats": {c: {"mean": float(M[:, i].mean()),
                      "std": float(M[:, i].std())} for i, c in enumerate(cols)},
        "overall_correlation": overall_corr,
        "bucket_correlation": bucket_corr,
        "rounds": all_rounds,
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[ENT-T] Wrote {args.output}")


if __name__ == "__main__":
    main()
