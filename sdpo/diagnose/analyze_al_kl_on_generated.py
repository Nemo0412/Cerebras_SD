"""
AL_KL vs accept-length correlation on Qwen3-self-generated responses.

Step 1: Target Qwen3-8B generates greedy responses on prompts (= what real
        SD inference would produce, since SD greedy verify = target argmax).
Step 2: Load RejectionModel + tokenized (prompt + generated response).
Step 3: dataprepare() + γ-rollout (training-style autoregressive draft).
Step 4: Per non-overlap round (advance by τ+1), compute (hard_τ, AL_KL, AL_TV).
Step 5: Correlation.

Because the test sequence IS Qwen3-argmax, teacher-forced target_argmax
matches real inference target_argmax → τ measurement is inference-realistic.

Usage:
  python sdpo/diagnose/analyze_al_kl_on_generated.py \
    --base-model-path Qwen/Qwen3-8B \
    --draft-model-path /scratch/.../q8_eagle3_alkl_only_6ep/state_5 \
    --bench-name mt_bench --num-questions 10 --gamma 7 \
    --gen-tokens 256 \
    --output diagnose_al_kl_gen_alkl.json
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

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'sdpo'))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--base-model-path', required=True)
    p.add_argument('--draft-model-path', required=True)
    p.add_argument('--bench-name', default='mt_bench')
    p.add_argument('--num-questions', type=int, default=10)
    p.add_argument('--gen-tokens', type=int, default=256,
                   help='New tokens to generate per prompt for analysis.')
    p.add_argument('--config-path', default='sdpo/config_qwen3.json')
    p.add_argument('--gamma', type=int, default=7)
    p.add_argument('--max-len', type=int, default=2048)
    p.add_argument('--output', default='diagnose_al_kl_gen.json')
    return p.parse_args()


def _load_draft_weights(model, draftpath):
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


def load_model(args):
    from huggingface_hub import snapshot_download
    from rejection_model import RejectionModel
    from traineagle3.configs import EConfig
    if os.path.isdir(args.base_model_path):
        basepath = args.base_model_path
    else:
        print(f"  Downloading base: {args.base_model_path}")
        basepath = snapshot_download(args.base_model_path)
    draftpath = args.draft_model_path
    if not os.path.isdir(draftpath):
        draftpath = snapshot_download(draftpath)
    with open(args.config_path) as f:
        cfg = EConfig.from_dict(json.load(f))
    cfg.gradient_checkpointing = False
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
    model = model.cuda().eval()
    return model


def generate_qwen3_response(target_model, tokenizer, prompt_ids, max_new):
    """Greedy generation with target model. Returns full input_ids
    (prompt + generated)."""
    with torch.no_grad():
        out = target_model.generate(
            prompt_ids,
            max_new_tokens=max_new,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    return out


@torch.no_grad()
def compute_round_metrics(model, input_ids, prompt_len, gamma):
    """Run γ-rollout on (prompt + generated) sequence. Compute per-position
    AL_KL/AL_TV/hard_τ, then iterate non-overlap rounds advancing by τ+1
    starting from prompt_len (= first generated token position).
    Returns array [N_rounds, 4]: AL_KL, AL_TV, hard_τ, soft_τ."""
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.zeros_like(input_ids)
    # Loss mask only on generated tokens (prompt_len..end)
    loss_mask[:, prompt_len:] = 1

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

    # γ-rollout (autoregressive)
    all_logits, all_tgt_d, all_tgt_in, all_tgt_lg = [], [], [], []
    cache_hidden = [[], []]
    cur_ids = input_ids_shifted
    cur_hs = hs_projected
    cur_tgt = target_greedy.clone()
    cur_tgt_lg = target.clone()
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
        logits = model.lm_head(model.norm(cur_hs)).float()
        all_logits.append(logits)
        all_tgt_d.append(full2draft[cur_tgt].clamp(min=0))
        all_tgt_in.append(t2d[cur_tgt])
        all_tgt_lg.append(cur_tgt_lg.clone())
        if idx < gamma - 1:
            # autoregressive: draft sees its own argmax as next input
            draft_argmax_d = logits.argmax(dim=-1)
            cur_ids = draft_argmax_d + d2t[draft_argmax_d]
            cur_tgt = shift(cur_tgt)
            cur_tgt_lg = torch.cat(
                [cur_tgt_lg[:, 1:], torch.zeros_like(cur_tgt_lg[:, :1])], dim=1)
            cur_mask = shift(cur_mask)

    # Per-step β values
    step_masks = []
    m = loss_mask_2d.clone()
    for j in range(gamma):
        step_masks.append(m.clone())
        if j < gamma - 1:
            m = torch.cat([m[:, 1:], torch.zeros_like(m[:, :1])], dim=1)

    beta_KL = torch.zeros(B, L, gamma, device=device)
    beta_TV = torch.zeros(B, L, gamma, device=device)
    beta_hard = torch.zeros(B, L, gamma, device=device)
    beta_soft = torch.zeros(B, L, gamma, device=device)

    for j in range(gamma):
        draft_lg = all_logits[j]
        tgt_lg = all_tgt_lg[j]
        tgt_in = all_tgt_in[j]
        mj = step_masks[j].bool()
        valid = tgt_in & mj

        tgt_lg_d = tgt_lg[..., t2d].float()
        tgt_logp = F.log_softmax(tgt_lg_d, dim=-1)
        tgt_p = tgt_logp.exp()
        draft_logp = F.log_softmax(draft_lg, dim=-1)
        draft_q = draft_logp.exp()

        kl = (tgt_p * (tgt_logp - draft_logp)).sum(-1)
        beta_KL[..., j] = torch.where(valid, 0.5 * torch.exp(-kl),
                                       torch.ones_like(kl))
        overlap = torch.sum(torch.min(tgt_p, draft_q), dim=-1)
        beta_TV[..., j] = torch.where(valid, overlap, torch.ones_like(overlap))

        draft_argmax = draft_lg.argmax(dim=-1)
        draft_full = draft_argmax + d2t[draft_argmax]
        target_argmax = tgt_lg.argmax(dim=-1)
        hard = (draft_full == target_argmax) & valid
        beta_hard[..., j] = torch.where(valid, hard.float(),
                                          torch.ones_like(hard.float()))

        tgt_p_full = F.softmax(tgt_lg.float(), dim=-1)
        soft = tgt_p_full.gather(-1, draft_full.unsqueeze(-1)).squeeze(-1)
        beta_soft[..., j] = torch.where(valid, soft, torch.ones_like(soft))

    al_KL = torch.cumprod(beta_KL, dim=-1).sum(-1)
    al_TV = torch.cumprod(beta_TV, dim=-1).sum(-1)
    al_hard = torch.cumprod(beta_hard, dim=-1).sum(-1)
    al_soft = torch.cumprod(beta_soft, dim=-1).sum(-1)

    # Per-round iteration: start at prompt_len, advance by hard_τ + 1
    rounds = []
    valid_mask = loss_mask_2d[0].cpu().numpy().astype(bool)
    al_KL_np = al_KL[0].cpu().numpy()
    al_TV_np = al_TV[0].cpu().numpy()
    al_hard_np = al_hard[0].cpu().numpy()
    al_soft_np = al_soft[0].cpu().numpy()

    valid_positions = np.where(valid_mask)[0]
    if len(valid_positions) == 0:
        return np.zeros((0, 4))
    p = int(valid_positions[0])
    end = int(valid_positions[-1])
    while p <= end:
        if not valid_mask[p]:
            p += 1
            continue
        hard_t = int(round(float(al_hard_np[p])))
        rounds.append([
            float(al_KL_np[p]),
            float(al_TV_np[p]),
            float(al_hard_np[p]),
            float(al_soft_np[p]),
        ])
        p += hard_t + 1
    return np.array(rounds)


def main():
    args = parse_args()
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"[GEN-ANALYZE] Loading models")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path,
                                              trust_remote_code=True)
    target_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, torch_dtype=torch.float16,
        attn_implementation='sdpa').cuda().eval()
    model = load_model(args)  # RejectionModel (has draft + same target)
    # Use RejectionModel's internal target instead of separate load
    # Actually we can reuse model's base for generation
    # But RejectionModel's target is stored internally; check structure
    # For simplicity use the separate target_model loaded above

    qfile = os.path.join(_ROOT, 'data', args.bench_name, 'question.jsonl')
    questions = [json.loads(l) for l in open(qfile)]
    questions = questions[:args.num_questions]
    print(f"[GEN-ANALYZE] {len(questions)} questions from {args.bench_name}")

    system_msg = "You are a helpful, respectful and honest assistant."
    all_rounds = []
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
        prompt_ids = tokenizer([text], add_special_tokens=False,
                                return_tensors='pt').input_ids.cuda()
        prompt_len = prompt_ids.shape[1]

        # Generate with target (greedy)
        full_ids = generate_qwen3_response(
            target_model, tokenizer, prompt_ids, args.gen_tokens)
        gen_len = full_ids.shape[1] - prompt_len

        # Analyze γ-rollout on full sequence, only counting positions in generated region
        rounds = compute_round_metrics(model, full_ids, prompt_len, args.gamma)
        all_rounds.append(rounds)
        print(f"  q{qi}: prompt={prompt_len}, gen={gen_len}, rounds={rounds.shape[0]}, "
              f"mean τ={rounds[:,2].mean():.2f}" if rounds.shape[0] else f"  q{qi}: 0 rounds")

    M = np.concatenate(all_rounds, axis=0) if all_rounds else np.zeros((0, 4))
    print(f"\n[GEN-ANALYZE] Total {M.shape[0]} rounds on Qwen3-self-generated text")
    cols = ['AL_KL', 'AL_TV', 'hard_τ', 'soft_τ']
    for i, c in enumerate(cols):
        print(f"  {c:<10}mean={M[:,i].mean():.3f}  std={M[:,i].std():.3f}")

    print(f"\n  Correlations vs AL_KL:")
    print(f"  {'truth':<10}{'pearson':>10}{'spearman':>11}")
    for i, c in [(1, 'AL_TV'), (2, 'hard_τ'), (3, 'soft_τ')]:
        pe = pearsonr(M[:, 0], M[:, i])
        sp = spearmanr(M[:, 0], M[:, i])
        print(f"  {c:<10}{pe.statistic:>10.4f}{sp.statistic:>11.4f}")

    results = {
        'meta': {'draft_ckpt': args.draft_model_path, 'gamma': args.gamma,
                 'n_rounds': int(M.shape[0]), 'n_questions': len(questions)},
        'stats': {c: {'mean': float(M[:,i].mean()), 'std': float(M[:,i].std())}
                  for i, c in enumerate(cols)},
        'correlations': {
            f'AL_KL_vs_{c}': {
                'pearson': float(pearsonr(M[:, 0], M[:, i]).statistic),
                'spearman': float(spearmanr(M[:, 0], M[:, i]).statistic),
            } for i, c in [(1, 'AL_TV'), (2, 'hard_tau'), (3, 'soft_tau')]
        },
    }
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[GEN-ANALYZE] Wrote {args.output}")


if __name__ == '__main__':
    main()
