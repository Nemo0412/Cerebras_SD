"""
Verify how well AL_KL surrogate correlates with actual expected accepted length
on a trained EAGLE-3 draft model.

Per valid position t in held-out data, compute four quantities over γ-rollout:
  - AL_KL surrogate:  S_KL(t)   = Σ_{k=1..γ} Π_{j<k} 0.5 · exp(-KL(p̃_{t+j} || q̃_{t+j}))
  - AL_TV (SpS-EAL):  S_TV(t)   = Σ_{k=1..γ} Π_{j<k} (1 − TV(p̃_{t+j}, q̃_{t+j}))
                                = Σ_{k=1..γ} Π_{j<k} Σ_z min(p_z, q_z)
  - hard τ:           τ_hard(t) = Σ_{k=1..γ} Π_{j<k} 1[draft_argmax_{t+j} == target_argmax_{t+j}]
  - soft τ:           τ_soft(t) = Σ_{k=1..γ} Π_{j<k} P_target_{t+j}[draft_argmax_{t+j}]

Then compute Pearson / Spearman correlation between AL_KL and each "truth"
candidate, plus scatter plot data.

Usage:
  python sdpo/diagnose/analyze_al_kl_correlation.py \
    --base-model-path Qwen/Qwen3-8B \
    --draft-model-path /scratch/.../q8_eagle3_kl_regen_l2k_6ep/state_5 \
    --testpath sdpo/data/mixed_val_80.jsonl \
    --gamma 7 --max-samples 50 \
    --output diagnose_al_kl_corr.json
"""
import argparse
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

# Project root (two levels up from sdpo/diagnose/) for traineagle3 import
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'sdpo'))   # for rejection_model

from traineagle3.configs import EConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--base-model-path', required=True)
    p.add_argument('--draft-model-path', required=True,
                   help='Path to trained EAGLE draft state_N directory.')
    p.add_argument('--testpath', default='sdpo/data/mixed_val_80.jsonl')
    p.add_argument('--config-path', default='sdpo/config_qwen3.json')
    p.add_argument('--gamma', type=int, default=7)
    p.add_argument('--max-samples', type=int, default=50,
                   help='Number of validation samples to process.')
    p.add_argument('--max-len', type=int, default=2048)
    p.add_argument('--output', default='diagnose_al_kl_corr.json')
    return p.parse_args()


def _load_draft_weights(model, draftpath):
    """Copy of sdpo.main.load_draft_weights (avoids importing main.py argparse)."""
    from safetensors.torch import load_file as sf_load
    sf_path = os.path.join(draftpath, "model.safetensors")
    bin_path = os.path.join(draftpath, "pytorch_model.bin")
    if os.path.exists(sf_path):
        state = sf_load(sf_path, device="cpu")
    elif os.path.exists(bin_path):
        state = torch.load(bin_path, map_location="cpu", weights_only=False)
    else:
        raise FileNotFoundError(f"No draft weights in {draftpath}")
    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    d2t = state.pop("d2t", None)
    t2d = state.pop("t2d", None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys: {unexpected}")
    if d2t is not None and t2d is not None:
        model.register_buffer("d2t", d2t)
        model.register_buffer("t2d", t2d)


def load_eagle_model(args):
    """Load RejectionModel + target model, mirroring sdpo/main.py setup."""
    from huggingface_hub import snapshot_download
    from rejection_model import RejectionModel

    # Resolve HF hub id → local path for base
    if os.path.isdir(args.base_model_path):
        basepath = args.base_model_path
    else:
        print(f"  Downloading base: {args.base_model_path}")
        basepath = snapshot_download(args.base_model_path)
    draftpath = args.draft_model_path
    if not os.path.isdir(draftpath):
        print(f"  Downloading draft: {draftpath}")
        draftpath = snapshot_download(draftpath)

    # Load draft config
    with open(args.config_path) as f:
        cfg_dict = json.load(f)
    cfg = EConfig.from_dict(cfg_dict)
    cfg.gradient_checkpointing = False
    if hasattr(cfg, 'rope_theta'):
        print(f"  rope_theta = {cfg.rope_theta}")

    training_config = SimpleNamespace(
        bs=1, num_epochs=1, num_workers=0, max_len=args.max_len,
        config_path=args.config_path, gradient_checkpointing=False,
        eagle_coef=0, kl_coef=0, gamma=args.gamma, baseline=None,
    )
    model = RejectionModel(
        cfg, None, training_config, basepath,
        load_emb=True, load_head=True,
    )
    _load_draft_weights(model, draftpath)
    print(f"  Loaded draft weights from {draftpath}")
    model = model.cuda().eval()
    return model


def load_val_samples(args, tokenizer, max_samples):
    """Load a few mixed_val samples and tokenize with ShareGPT chat template."""
    items = []
    with open(args.testpath) as f:
        for line in f:
            d = json.loads(line)
            items.append(d)
            if len(items) >= max_samples:
                break
    print(f"  Loaded {len(items)} val samples")

    batches = []
    for d in items:
        convs = d['conversations']
        msgs = []
        for c in convs:
            role = 'user' if c['from'] in ('human', 'user') else 'assistant'
            msgs.append({'role': role, 'content': c['value']})
        try:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(msgs, tokenize=False)
        enc = tokenizer(text, return_tensors='pt', truncation=True,
                        max_length=args.max_len)
        batches.append(enc['input_ids'])
    return batches


@torch.no_grad()
def compute_metrics_for_batch(model, input_ids, gamma):
    """Run γ-rollout on one batch, return per-position [N, 4] metrics:
       [AL_KL, AL_TV, hard_tau, soft_tau], filtered to valid positions.
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.ones_like(input_ids)
    loss_mask[:, 0] = 0  # don't predict first

    # Mirror RejectionModel.forward setup
    hidden_states, target, loss_mask_3d, input_ids_shifted = \
        model.dataprepare(input_ids, attention_mask, loss_mask)
    loss_mask_2d = loss_mask_3d.squeeze(-1).bool()
    B, L, _ = hidden_states.shape

    hs_projected = model.fc(hidden_states.to(model.fc.weight.dtype))
    attn_mask = model._prepare_decoder_attention_mask(
        attention_mask, (B, L), hs_projected, 0)
    position_ids = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0)

    target_greedy = target.argmax(dim=-1)
    t2d = model.t2d.to(device)
    d2t = model.d2t.to(device)
    draft_ids = torch.arange(len(d2t), device=device)
    full2draft = torch.full((t2d.shape[0],), -1, dtype=torch.long, device=device)
    full2draft[draft_ids + d2t] = draft_ids

    # γ-rollout (collect draft logits per step, no grad)
    all_logits, all_tgt_d, all_tgt_in = [], [], []
    all_target_logits = []  # save target_logits at each γ step for KL/TV
    cache_hidden = [[], []]
    cur_ids = input_ids_shifted
    cur_hs = hs_projected
    cur_tgt = target_greedy.clone()
    cur_tgt_lg = target.clone()  # full target logits, shifted same way
    cur_mask = loss_mask_2d.clone()

    def shift(t):
        return torch.cat([t[:, 1:], torch.zeros_like(t[:, :1])], dim=1)

    for idx in range(gamma):
        embeds = model.embed_tokens(cur_ids).to(cur_hs.dtype)
        layer_out, cache_hidden = model.midlayer(
            input_emb=embeds, hidden_states=cur_hs,
            cache_hidden=cache_hidden, attention_mask=attn_mask,
            position_ids=position_ids, past_key_value=None,
            output_attentions=False, use_cache=True)
        cur_hs = layer_out[0]
        logits = model.lm_head(model.norm(cur_hs)).float()  # [B, L, V_draft]
        all_logits.append(logits)
        tgt_d = full2draft[cur_tgt].clamp(min=0)
        tgt_in = t2d[cur_tgt]
        all_tgt_d.append(tgt_d)
        all_tgt_in.append(tgt_in)
        all_target_logits.append(cur_tgt_lg.clone())
        if idx < gamma - 1:
            # AUTOREGRESSIVE: draft uses its own argmax as next input
            # (instead of ground-truth shift). cur_ids[b, p] becomes draft's
            # predicted next token at position p (in target vocab).
            draft_argmax_d = logits.argmax(dim=-1)        # [B, L] draft vocab
            draft_argmax_full = draft_argmax_d + d2t[draft_argmax_d]
            cur_ids = draft_argmax_full
            # Target side: still shift ground truth (used for KL/TV reference
            # — exact for positions before first divergence, approx after).
            cur_tgt = shift(cur_tgt)
            cur_tgt_lg = torch.cat(
                [cur_tgt_lg[:, 1:], torch.zeros_like(cur_tgt_lg[:, :1])], dim=1)
            cur_mask = shift(cur_mask)

    # Now compute per-position β values for each metric, then cumprod, sum
    # β_KL[t, j] = 0.5 · exp(-KL(p̃, q̃))
    # β_TV[t, j] = Σ min(p̃, q̃)
    # β_hard[t, j] = 1[draft_argmax == target_argmax]
    # β_soft[t, j] = p̃[draft_argmax]

    valid_init = loss_mask_2d  # [B, L] — only positions in original sequence count
    # We need each step's mask (positions shifted out at end go invalid)
    step_masks = []
    m = loss_mask_2d.clone()
    for j in range(gamma):
        step_masks.append(m.clone())
        if j < gamma - 1:
            m = torch.cat([m[:, 1:], torch.zeros_like(m[:, :1])], dim=1)

    # Build [γ] tensors of β for each metric (shape [B, L, γ])
    beta_KL = torch.zeros(B, L, gamma, device=device)
    beta_TV = torch.zeros(B, L, gamma, device=device)
    beta_hard = torch.zeros(B, L, gamma, device=device)
    beta_soft = torch.zeros(B, L, gamma, device=device)

    for j in range(gamma):
        draft_lg = all_logits[j]                       # [B, L, V_draft]
        tgt_lg = all_target_logits[j]                  # [B, L, V_full]
        tgt_d = all_tgt_d[j]                           # [B, L]
        tgt_in = all_tgt_in[j]                         # [B, L]
        mj = step_masks[j].float()                     # [B, L]

        # p̃ restricted to draft vocab, q̃ = softmax(draft_lg)
        tgt_lg_d = tgt_lg[..., t2d].float()
        tgt_logp = F.log_softmax(tgt_lg_d, dim=-1)
        tgt_p = tgt_logp.exp()
        draft_logp = F.log_softmax(draft_lg, dim=-1)
        draft_q = draft_logp.exp()

        # KL(p || q)
        kl = (tgt_p * (tgt_logp - draft_logp)).sum(-1)      # [B, L]
        beta_KL[..., j] = (0.5 * torch.exp(-kl)) * (tgt_in & step_masks[j]).float() \
                        + 1.0 * (~(tgt_in & step_masks[j])).float() * 0  # invalid → 0 (won't count in stats below)

        # TV
        overlap = torch.sum(torch.min(tgt_p, draft_q), dim=-1)
        beta_TV[..., j] = overlap * (tgt_in & step_masks[j]).float()

        # hard accept
        draft_argmax = draft_lg.argmax(dim=-1)          # in draft vocab
        draft_full = draft_argmax + d2t[draft_argmax]
        target_argmax = tgt_lg.argmax(dim=-1)
        hard = (draft_full == target_argmax) & tgt_in & step_masks[j]
        beta_hard[..., j] = hard.float()

        # soft: p̃[draft_argmax_in_draft_vocab]
        soft = draft_q.gather(-1, draft_argmax.unsqueeze(-1)).squeeze(-1)
        # actually for soft EAL we want p_target[draft_token], not p_draft
        # P_target at draft_argmax in target's full distribution:
        tgt_p_full = F.softmax(tgt_lg.float(), dim=-1)
        soft_p_t = tgt_p_full.gather(-1, draft_full.unsqueeze(-1)).squeeze(-1)
        beta_soft[..., j] = soft_p_t * (tgt_in & step_masks[j]).float()

    # cumprod and sum over γ
    # For β_KL: invalid positions have β=0 → cumprod hits 0 → not informative.
    # Replace invalid β with 1 to keep cumprod neutral.
    for beta in (beta_KL, beta_TV, beta_hard, beta_soft):
        # mask: where step_mask is 0 OR tgt_in is 0, set β=1 (neutral for cumprod)
        for j in range(gamma):
            invalid = ~(all_tgt_in[j] & step_masks[j])
            beta[..., j] = torch.where(invalid, torch.ones_like(beta[..., j]), beta[..., j])

    al_KL = torch.cumprod(beta_KL, dim=-1).sum(-1)       # [B, L]
    al_TV = torch.cumprod(beta_TV, dim=-1).sum(-1)
    al_hard = torch.cumprod(beta_hard, dim=-1).sum(-1)
    al_soft = torch.cumprod(beta_soft, dim=-1).sum(-1)

    # Filter to ORIGINAL valid positions (loss_mask_2d at base position)
    al_KL_np = al_KL.cpu().numpy()
    al_TV_np = al_TV.cpu().numpy()
    al_hard_np = al_hard.cpu().numpy()
    al_soft_np = al_soft.cpu().numpy()
    mask_np = loss_mask_2d.cpu().numpy().astype(bool)

    # Per-position (overlapping windows): every valid position
    valid = mask_np.flatten()
    per_pos = np.stack([
        al_KL_np.flatten()[valid],
        al_TV_np.flatten()[valid],
        al_hard_np.flatten()[valid],
        al_soft_np.flatten()[valid],
    ], axis=-1)

    # Per-round (non-overlapping, inference-style):
    #   start at first valid p; record (AL_KL[p], AL_TV[p], hard_τ[p]);
    #   advance p by (hard_τ[p] + 1); continue until end of valid region.
    per_round_list = []
    for b in range(B):
        valid_positions = np.where(mask_np[b])[0]
        if len(valid_positions) == 0:
            continue
        start = int(valid_positions[0])
        end = int(valid_positions[-1])
        p = start
        while p <= end:
            if not mask_np[b, p]:
                p += 1
                continue
            hard_t = int(round(float(al_hard_np[b, p])))
            per_round_list.append([
                float(al_KL_np[b, p]),
                float(al_TV_np[b, p]),
                float(al_hard_np[b, p]),
                float(al_soft_np[b, p]),
            ])
            # Advance by (hard_τ + 1): τ accepted + 1 correction
            p += hard_t + 1
    per_round = np.array(per_round_list) if per_round_list else np.zeros((0, 4))
    return per_pos, per_round


def main():
    args = parse_args()
    from transformers import AutoTokenizer

    print(f"[ANALYZE] Loading tokenizer + model")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path,
                                               trust_remote_code=True)
    model = load_eagle_model(args)
    samples = load_val_samples(args, tokenizer, args.max_samples)

    per_pos_all, per_round_all = [], []
    print(f"[ANALYZE] Processing {len(samples)} samples (γ={args.gamma})")
    for i, ids in enumerate(samples):
        try:
            per_pos, per_round = compute_metrics_for_batch(model, ids, args.gamma)
            per_pos_all.append(per_pos)
            per_round_all.append(per_round)
        except Exception as e:
            print(f"  sample {i} skipped: {e}")
            continue
    Mpos = np.concatenate(per_pos_all, axis=0)
    Mrnd = np.concatenate(per_round_all, axis=0) if per_round_all else np.zeros((0, 4))
    print(f"[ANALYZE] {Mpos.shape[0]} positions / {Mrnd.shape[0]} non-overlapping rounds")

    cols = ['AL_KL', 'AL_TV', 'hard_tau', 'soft_tau']
    results = {'meta': {
        'draft_ckpt': args.draft_model_path,
        'gamma': args.gamma,
        'columns': cols,
        'n_positions': int(Mpos.shape[0]),
        'n_rounds': int(Mrnd.shape[0]),
    }}

    for label, M in [('per_position (overlapping)', Mpos),
                     ('per_round (inference-style, non-overlap)', Mrnd)]:
        print(f"\n  ─── {label} ───")
        if M.shape[0] < 2:
            print("    too few data points")
            continue
        print(f"    {'metric':<10}{'mean':>9}{'std':>9}{'min':>7}{'max':>7}")
        stats = {}
        for c, idx in zip(cols, range(4)):
            stats[c] = {
                'mean': float(M[:,idx].mean()), 'std': float(M[:,idx].std()),
                'min': float(M[:,idx].min()), 'max': float(M[:,idx].max()),
            }
            print(f"    {c:<10}{M[:,idx].mean():>9.3f}{M[:,idx].std():>9.3f}"
                  f"{M[:,idx].min():>7.2f}{M[:,idx].max():>7.2f}")
        corrs = {}
        print(f"    correlations vs AL_KL:")
        print(f"    {'truth':<12}{'pearson':>10}{'spearman':>11}")
        for c, idx in zip(cols[1:], range(1, 4)):
            pe = pearsonr(M[:, 0], M[:, idx])
            sp = spearmanr(M[:, 0], M[:, idx])
            print(f"    {c:<12}{pe.statistic:>10.4f}{sp.statistic:>11.4f}")
            corrs[f'AL_KL_vs_{c}'] = {
                'pearson': float(pe.statistic),
                'spearman': float(sp.statistic),
            }
        key = 'per_position' if 'overlap' in label and 'non' not in label else 'per_round'
        results[key] = {'stats': stats, 'correlations': corrs}

    # save subsample for scatter
    n_save = min(2000, Mrnd.shape[0])
    if n_save > 0:
        idx_sub = np.random.choice(Mrnd.shape[0], n_save, replace=False)
        results['scatter_sample_rounds'] = Mrnd[idx_sub].tolist()

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[ANALYZE] Wrote {args.output}")


if __name__ == '__main__':
    main()
