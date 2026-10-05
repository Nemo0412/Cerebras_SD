"""
Small LM Draft Model for Speculative Decoding Training.

Teacher-forced training. Both target (frozen) and draft (trainable) share
the same tokenizer/vocab (Qwen3 family).

Loss is a combination of an anchor loss and an optional auxiliary loss:

    L_total = L_anchor + sigmoid_coef * L_aux

anchor (--anchor):
    'kl'  — soft-target KL (EAGLE-style)
    'ce'  — hard-label cross-entropy on target argmax

aux_loss (--aux_loss):
    'none' — anchor only
    'v2'   — acceptance_length_v2: -Σ_k 0.8^k · ∏_{j=0..k} β(t+j)
             where β = σ((z_target - max z)/T) + 0.5, clamped ≤ 1
             (β floor = 0.5; softer cumprod attenuation on deep rejects)
    'v4'   — acceptance_length_v4: same structure as v2 but β = 2·σ(gap/T)
             (β floor = 0; honest cumprod — a reject fully kills deeper terms)
    'v5'   — truncated-at-first-reject: same β and cum_prod as v4, but the
             weighted sum over the window stops at (and includes) the first
             hard reject. Beyond first reject the cum_prod terms are dropped.
    'v6'   — peaked-at-first-reject: same β and cum_prod as v4, but uses a
             dynamic per-window weight profile: small_w (0.2) for positions
             before the first reject, peak_w (1.0) AT the first reject,
             0 after. All-accept windows get small_w uniform. Overrides the
             aux_weight scheme (the profile is intrinsic).
    'tv'   — total variation: Σ_k 0.8^k · (1 - min-overlap) over sliding γ window
    'eal'  — Expected Accept Length: Σ_k cum_prod(P)[k] where P_t = draft
             softmax probability at target argmax. Mathematically equivalent to
             E[τ] = Σ_k k·Pr[τ=k] under the assumption that per-position
             acceptance probability is the draft's marginal prob on target token.

Supported combinations: {kl, ce} × {none, v2, v4, v5, v6, tv, eal} = 14 configs.

Legacy flag `baseline` is kept for backwards compat:
    eagle_only → anchor=kl, aux_loss=none
    ce_only    → anchor=ce, aux_loss=none
"""

import copy
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM


def _load_lm(name: str, dtype=torch.float16):
    """Polymorphic LM loader.

    Standard models: AutoModelForCausalLM.
    Gemma 4: load via _load_gemma4_as_causal — bypasses the multimodal
    Gemma4ForConditionalGeneration class and produces a clean
    Gemma4ForCausalLM with proper softcap + correct tied-embed semantics
    under DeepSpeed ZeRO-3. (The multimodal wrapper triggers a ZeRO-3
    gather bug on the tied lm_head ↔ embed_tokens parameter — target
    logits saturate at ~1600 vs ~120 outside DS, mean_tau collapses to 0.)
    """
    cfg = AutoConfig.from_pretrained(name)
    arch = cfg.architectures[0] if cfg.architectures else ""
    if "Gemma4" in arch:
        return _load_gemma4_as_causal(name, dtype)
    return AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)


_GEMMA4_CAUSAL_CACHE_ROOT = "/scratch/tx856/.cache/gemma4_causal"


def _load_gemma4_as_causal(name: str, dtype):
    """Load Gemma 4 as Gemma4ForCausalLM from a pre-remapped local cache.

    HF checkpoints ship Gemma4ForConditionalGeneration with keys like
    `model.language_model.*` (+ vision/audio/projector). Loading those into
    Gemma4ForCausalLM directly produces random init for the language model
    (all language keys MISSING). Loading the multimodal class and wrapping
    triggers a DeepSpeed ZeRO-3 bug on the tied lm_head ↔ embed_tokens
    parameter — target logits saturate at ~1600 vs ~120 outside DS,
    mean_tau collapses to 0.

    Fix: do a ONE-TIME on-disk remap with `sdpo/data/remap_gemma4_to_causal.py`
    that strips `model.language_model.` → `model.`, drops vision/audio
    towers, and rewrites the config as `Gemma4ForCausalLM`. The result is a
    clean Gemma4ForCausalLM checkpoint that loads with from_pretrained (no
    state_dict construction at training time, no per-rank CPU state_dict).

    For trained-draft ckpts (passed via DRAFTPATH=…/state_N), the saved keys
    already use `model.X` prefix — load directly without consulting cache.
    """
    from transformers import Gemma4ForCausalLM
    import os

    if os.path.isdir(name):
        return Gemma4ForCausalLM.from_pretrained(
            name, dtype=dtype, attn_implementation="sdpa")

    safe = name.replace("/", "_")
    cache_dir = os.path.join(_GEMMA4_CAUSAL_CACHE_ROOT, safe)
    if not os.path.exists(os.path.join(cache_dir, "config.json")):
        raise FileNotFoundError(
            f"Gemma 4 causal cache missing at {cache_dir}. Run once:\n"
            f"  python sdpo/data/remap_gemma4_to_causal.py --model {name}")
    return Gemma4ForCausalLM.from_pretrained(
        cache_dir, dtype=dtype, attn_implementation="sdpa")


def _get_step_weights(scheme, n, device, dtype):
    """Per-step weights of length n for loss aggregation.

    Schemes (NOT normalized):
      uniform:  [1, 1, ..., 1]                 sum = n
      pow08  :  [1, 0.8, 0.64, ...]            sum ≈ (1-0.8^n)/0.2
      dec    :  [n, n-1, ..., 1]               sum = n(n+1)/2 (early heavy)
      inc    :  [1, 2, ..., n]                 sum = n(n+1)/2 (late heavy)

    When switching schemes, the absolute loss magnitude changes; compensate
    sigmoid_coef / lr / anchor scale if needed.
    """
    if scheme == 'uniform':
        return torch.ones(n, device=device, dtype=dtype)
    if scheme == 'pow08':
        return torch.tensor([0.8 ** k for k in range(n)],
                            device=device, dtype=dtype)
    if scheme == 'dec':
        return torch.arange(n, 0, -1, device=device, dtype=dtype)
    if scheme == 'inc':
        return torch.arange(1, n + 1, device=device, dtype=dtype)
    raise ValueError(f"Unknown weight scheme: {scheme}")


class SmallLMDraftModel(nn.Module):

    def __init__(self, target_path, draft_path, gamma=7, dtype=torch.float16,
                 enable_grpo=False):
        super().__init__()
        self.gamma = gamma
        self.length = gamma

        # Target (frozen)
        self.target_model = _load_lm(target_path, dtype=dtype)
        self.target_model.eval()
        for p in self.target_model.parameters():
            p.requires_grad = False

        # Draft (trainable)
        self.draft_model = _load_lm(draft_path, dtype=dtype)
        # Gradient checkpointing for Gemma 4 draft. Saves ~half the activation
        # memory on 31B target + E2B draft (2x H200 ZeRO-2 setup is ~80 GB/rank;
        # without checkpointing the activation peak OOMs at max_len=2048).
        # Other model families: opt-in via env override only.
        import os as _os
        # Default OFF: gradient_checkpointing_enable(use_reentrant=False) on
        # Gemma 4 draft accumulates bf16 numerical noise in recompute → NaN
        # after ~140 steps (verified 2026-06-29). H200 143 GB fits max_len=512
        # without checkpointing (~140 GB/rank). Enable via GRADIENT_CHECKPOINTING=1
        # if you accept the NaN risk (e.g., longer max_len + small grad steps).
        if _os.environ.get("GRADIENT_CHECKPOINTING", "0") == "1":
            if hasattr(self.draft_model, "gradient_checkpointing_enable"):
                self.draft_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})

        # Ref draft (frozen initial policy for GRPO). Materialized lazily by
        # main script via _init_ref_draft_model_from_path AFTER
        # deepspeed.initialize, when the HfDeepSpeedConfig sentinel has been
        # released and AutoModel.from_pretrained no longer auto-partitions.
        # We bypass nn.Module.__setattr__ so DeepSpeed's ZeRO-3 partition
        # coordinator never tracks the frozen ref (avoids NOT_AVAILABLE
        # autograd-fetch failure). Costs ~1.2 GB bf16 per GPU for 0.6B draft.
        self._enable_grpo = enable_grpo
        object.__setattr__(self, 'ref_draft_model', None)

    def _init_ref_draft_model_from_path(self, path, dtype):
        """Called from main script AFTER deepspeed.initialize and AFTER the
        HfDeepSpeedConfig sentinel has been released. Loads a fresh frozen
        ref model from disk (full copy, not ZeRO-3 partitioned)."""
        ref = _load_lm(path, dtype=dtype).cuda().eval()
        for p in ref.parameters():
            p.requires_grad = False
        object.__setattr__(self, 'ref_draft_model', ref)

    def train(self, mode=True):
        super().train(mode)
        self.target_model.eval()
        self.draft_model.train(mode)
        if self.ref_draft_model is not None:
            self.ref_draft_model.eval()
        return self

    def forward(self, input_ids, attention_mask, loss_mask,
                sigmoid_coef=0.1, temperature=1.0,
                prob_temperature=1.0,
                draft_temperature=1.0,
                anchor='kl', aux_loss='v2',
                anchor_weight='none', aux_weight='pow08',
                grpo_coef=0.0, grpo_k_groups=8, grpo_m=4,
                grpo_eps=0.2, grpo_mode='window', grpo_sample_temp=1.0,
                grpo_reward='hard',
                baseline=None):
        # prob_temperature: T applied to p̃ = softmax(z/T) for AL_TV / AL_KL / WKL.
        # draft_temperature: T applied to draft log_softmax in KL anchor only.
        #   draft_logp = log_softmax(draft_lg / T_d). Default 1.0 = no scaling.
        #   Higher T softens draft prob → KL pushes wider logit gap to match target_p.
        # ── Normalize flags (legacy compat) ──
        if baseline == 'eagle_only':
            anchor, aux_loss = 'kl', 'none'
        elif baseline == 'ce_only':
            anchor, aux_loss = 'ce', 'none'
        if aux_loss == 'acceptance_length_v2':
            aux_loss = 'v2'
        if aux_loss == 'acceptance_length_v4':
            aux_loss = 'v4'
        assert anchor in ('kl', 'ce', 'eal', 'none', 'kl_rkl'), \
            f"anchor must be kl|ce|eal|none, got {anchor}"
        assert aux_loss in ('none', 'v2', 'v4', 'v4_w1', 'v4eal', 'v5', 'v6', 'v7', 'v8',
                            'tv', 'eal', 'eal_w1', 'peal',
                            'al_tv', 'al_kl', 'wkl',
                            'al_tv_intersect_topk', 'al_tv_target_topk',
                            'al_tv_target_topp',
                            'al_tv_target_topk_topp',
                            'al_tv_renewal',
                            'al_tv_target_topk_renewal',
                            'tree_v4', 'tree_altv', 'tree_ce', 'tree_altv_wk',
                            'tree_ce_hard', 'tree_ce_soft', 'altv_p_tree_altv',
                            'topk_softce'), \
            f"aux_loss invalid: {aux_loss}"
        assert anchor_weight in ('none', 'uniform', 'pow08', 'dec', 'inc', 'v6'), \
            f"anchor_weight must be none|uniform|pow08|dec|inc|v6, got {anchor_weight}"
        assert aux_weight in ('uniform', 'pow08', 'dec', 'inc', 'v6'), \
            f"aux_weight must be uniform|pow08|dec|inc|v6, got {aux_weight}"

        B, L = input_ids.shape
        device = input_ids.device
        gamma = self.gamma
        mask = loss_mask[:, 1:].bool()

        # ── Target forward (frozen) ──
        with torch.no_grad():
            target_lg = self.target_model(
                input_ids=input_ids,
                attention_mask=attention_mask).logits[:, :-1, :].float()
            target_argmax = target_lg.argmax(dim=-1)
            target_p = F.softmax(target_lg, dim=-1)

        # ── Draft forward (trainable) ──
        draft_lg = self.draft_model(
            input_ids=input_ids,
            attention_mask=attention_mask).logits[:, :-1, :].float()
        draft_logp = F.log_softmax(draft_lg, dim=-1)

        nv = mask.float().sum()

        Lp_anchor = mask.shape[1]
        usable_anchor = Lp_anchor - gamma + 1

        # ── No anchor (GRPO-only / pure aux training) ──
        if anchor == 'none':
            # Attach zero to compute graph so backward works when GRPO skips
            # all groups (e.g., K_groups too large for short sequences in batch).
            anchor_loss = draft_lg.sum() * 0.0
            _skip_perpos_anchor = True
        # ── EAL anchor (window-level, not per-position) ──
        # Replaces per-position KL/CE entirely. Loss = -mean over windows of
        # Σ_k aw[k] · cum_prod(P)[k]  where P_t = softmax(draft)[target_argmax_t].
        # anchor_weight controls the per-cum_prod-term weighting (uniform = exact EAL).
        elif anchor == 'eal':
            if usable_anchor <= 0:
                anchor_loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                draft_q = F.softmax(draft_lg, dim=-1)
                P = draft_q.gather(
                    -1, target_argmax.unsqueeze(-1)).squeeze(-1)         # [B, L-1]
                P_stack = torch.stack(
                    [P[:, k:k + usable_anchor] for k in range(gamma)], dim=-1)
                cum_prod_eal = torch.cumprod(P_stack, dim=-1)            # [B, U, γ]
                if anchor_weight == 'v6':
                    with torch.no_grad():
                        ha = (draft_lg.argmax(dim=-1) == target_argmax)
                        acc_st = torch.stack(
                            [ha[:, k:k + usable_anchor]
                             for k in range(gamma)], dim=-1)
                        cumrej = (~acc_st).int().cumsum(-1)
                        aw_eal = (((cumrej == 0).to(P.dtype)) * 1.0
                                  + (((cumrej == 1) & ~acc_st).to(P.dtype)) * 1.5)
                else:
                    sched = anchor_weight if anchor_weight != 'none' else 'uniform'
                    aw_eal = _get_step_weights(
                        sched, gamma, device, P.dtype)                   # [γ]
                eal_per_win = (cum_prod_eal * aw_eal).sum(dim=-1)        # [B, U]
                anchor_win_mask = mask[:, :usable_anchor]
                nv_anchor_win = anchor_win_mask.float().sum()
                anchor_loss = -(eal_per_win * anchor_win_mask.float()
                                ).sum() / (nv_anchor_win + 1e-8)
            # Skip the per-position KL/CE branch below
            _skip_perpos_anchor = True
        else:
            _skip_perpos_anchor = False

        # ── Anchor per-position loss (KL or CE) ──
        if not _skip_perpos_anchor:
            if anchor == 'kl':
                # Apply draft_temperature: scale draft logits before log_softmax.
                # Default T=1 reuses precomputed draft_logp; otherwise recompute.
                if draft_temperature != 1.0:
                    draft_logp_T = F.log_softmax(draft_lg / draft_temperature, dim=-1)
                else:
                    draft_logp_T = draft_logp
                per_pos = -torch.sum(target_p * draft_logp_T, dim=-1)
            elif anchor == 'kl_rkl':
                # Forward KL on accepted positions, Reverse KL on rejected.
                # Accept mask (detached): draft_argmax == target_argmax.
                # Forward KL: -Σ p_target · log q_draft (mode-covering)
                # Reverse KL: Σ q_draft · (log q_draft − log p_target) (mode-seeking)
                if draft_temperature != 1.0:
                    draft_logp_T = F.log_softmax(draft_lg / draft_temperature, dim=-1)
                    draft_q_T = F.softmax(draft_lg / draft_temperature, dim=-1)
                else:
                    draft_logp_T = draft_logp
                    draft_q_T = torch.exp(draft_logp)
                per_pos_fkl = -torch.sum(target_p * draft_logp_T, dim=-1)
                target_logp = torch.log(target_p.clamp(min=1e-30))
                per_pos_rkl = torch.sum(
                    draft_q_T * (draft_logp_T - target_logp), dim=-1)
                with torch.no_grad():
                    accept_mask = (draft_lg.argmax(dim=-1) == target_argmax)
                per_pos = torch.where(accept_mask, per_pos_fkl, per_pos_rkl)
            else:  # 'ce'
                per_pos = -draft_logp.gather(
                    -1, target_argmax.unsqueeze(-1)).squeeze(-1)

            # 'none'   = per-position uniform avg over all valid tokens
            # static   = sliding γ-window with fixed scheme (uniform/pow08/dec/inc)
            # 'v6'     = dynamic per-window weight from hard_accept:
            #              before first reject → 1.0, first reject → 1.5, after → 0
            if anchor_weight == 'none' or usable_anchor <= 0:
                anchor_loss = (per_pos * mask.float()).sum() / (nv + 1e-8)
            else:
                per_pos_stack = torch.stack(
                    [per_pos[:, k:k + usable_anchor] for k in range(gamma)],
                    dim=-1)

                if anchor_weight == 'v6':
                    # Dynamic per-window weights from hard_accept pattern.
                    with torch.no_grad():
                        hard_accept_anchor = (
                            draft_lg.argmax(dim=-1) == target_argmax)
                        accept_stack = torch.stack(
                            [hard_accept_anchor[:, k:k + usable_anchor]
                             for k in range(gamma)], dim=-1)
                        reject_stack = ~accept_stack
                        cumrej = reject_stack.int().cumsum(-1)
                        is_first_reject = (cumrej == 1) & reject_stack
                        is_before_first = cumrej == 0
                        aw = (is_before_first.to(per_pos.dtype) * 1.0
                              + is_first_reject.to(per_pos.dtype) * 1.5)
                    anchor_win = (per_pos_stack * aw).sum(dim=-1)
                else:
                    aw = _get_step_weights(
                        anchor_weight, gamma, device, per_pos.dtype)
                    anchor_win = (per_pos_stack * aw).sum(dim=-1)

                anchor_win_mask = mask[:, :usable_anchor]
                nv_anchor_win = anchor_win_mask.float().sum()
                anchor_loss = (anchor_win * anchor_win_mask.float()).sum() \
                    / (nv_anchor_win + 1e-8)

        zero = torch.tensor(0.0, device=device)

        # ── GRPO loss (computed before aux=none early return so it applies
        #    to pure anchor configs as well) ──
        grpo_loss = torch.tensor(0.0, device=device)
        if grpo_coef > 0.0 and self.ref_draft_model is not None:
            if grpo_mode == 'sample':
                grpo_loss = self._compute_grpo_sample_loss(
                    input_ids, attention_mask, mask,
                    draft_lg, target_argmax, target_lg,
                    grpo_k_groups, grpo_m, grpo_eps, grpo_sample_temp,
                    grpo_reward)
            else:
                grpo_loss = self._compute_grpo_loss(
                    input_ids, attention_mask, mask,
                    draft_lg, target_argmax,
                    grpo_k_groups, grpo_m, grpo_eps, grpo_reward)

        # ── Early return: aux=none ──
        if aux_loss == 'none':
            # Compute mean_tau / step_acc / tau_hist for visibility even with
            # aux=none (KL-only / CE-only baseline). Without these metrics the
            # logged mean_tau is misleadingly 0 — looks like a training
            # collapse when training is actually fine (eagle_loss decreases).
            with torch.no_grad():
                hard_accept = (draft_lg.argmax(dim=-1) == target_argmax)
                Lp_m = mask.shape[1]
                usable_m = Lp_m - gamma + 1
                if usable_m > 0:
                    accept_stack_m = torch.stack(
                        [hard_accept[:, k:k + usable_m] for k in range(gamma)],
                        dim=-1)
                    win_mask_m = mask[:, :usable_m]
                    nv_win_m = win_mask_m.float().sum()
                    # tau = longest run of consecutive accepts from k=0
                    cum_m = torch.cumprod(accept_stack_m.long(), dim=-1)
                    tau_per_win = cum_m.sum(dim=-1).float()
                    mean_tau_val = ((tau_per_win * win_mask_m.float()).sum()
                                    / (nv_win_m + 1e-8)).item()
                    step_acc_val = [
                        ((accept_stack_m[..., k].float() * win_mask_m.float()).sum()
                         / (nv_win_m + 1e-8)).item()
                        for k in range(gamma)
                    ]
                    tau_int = tau_per_win.long()
                    tau_hist_val = [
                        int(((tau_int == t) & win_mask_m).sum().item())
                        for t in range(gamma + 1)
                    ]
                else:
                    mean_tau_val = 0.0
                    step_acc_val = [0.0] * gamma
                    tau_hist_val = [0] * (gamma + 1)
            return anchor_loss + grpo_coef * grpo_loss, zero, {
                "aux_loss": 0.0,
                "eagle_loss": anchor_loss.item(),
                "grpo_loss": grpo_loss.item(),
                "mean_tau": mean_tau_val,
                "num_valid": nv.item(),
                "step_acc": step_acc_val,
                "tau_hist": tau_hist_val,
            }

        # ── Prep sliding γ-window for aux + metrics ──
        with torch.no_grad():
            hard_accept = (draft_lg.argmax(dim=-1) == target_argmax)

        Lp = mask.shape[1]
        usable = Lp - gamma + 1
        if usable <= 0:
            return anchor_loss + grpo_coef * grpo_loss, zero, {
                "aux_loss": 0.0,
                "eagle_loss": anchor_loss.item(),
                "grpo_loss": grpo_loss.item(),
                "mean_tau": 0.0,
                "num_valid": 0.0,
                "step_acc": [0.0] * gamma,
                "tau_hist": [0] * (gamma + 1),
            }

        accept_stack = torch.stack(
            [hard_accept[:, k:k + usable] for k in range(gamma)], dim=-1)
        win_mask = mask[:, :usable]
        nv_win = win_mask.float().sum()

        # aux_weight: static [γ] vector, OR 'v6' dynamic per-window [B, U, γ]
        if aux_weight == 'v6':
            with torch.no_grad():
                rej_st = ~accept_stack
                cumrej_aw = rej_st.int().cumsum(-1)
                weights = ((cumrej_aw == 0).to(torch.float32) * 1.0
                           + ((cumrej_aw == 1) & rej_st).to(torch.float32) * 1.5)
        else:
            weights = _get_step_weights(aux_weight, gamma, device, torch.float32)

        # ── Aux loss ──
        if aux_loss == 'v4eal':
            # Option A: V4 + EAL parallel sum.
            #   β_v4 = 2σ((z_target - z_max) / T)   (logit rank signal)
            #   β_eal = softmax(z_d)[target_argmax]  (probability margin signal)
            #   L = -mean[ cumprod(β_v4)·w + eal_weight · cumprod(β_eal)·w ]
            # V4 covers rejected positions (rank gradient), EAL covers boundary
            # margin (push gap further once V4 saturates).
            z_target = draft_lg.gather(
                -1, target_argmax.unsqueeze(-1)).squeeze(-1)
            z_max = draft_lg.max(dim=-1).values
            gap = (z_target - z_max) / temperature
            beta_v4 = 2.0 * torch.sigmoid(gap)
            draft_q = F.softmax(draft_lg, dim=-1)
            beta_eal = draft_q.gather(
                -1, target_argmax.unsqueeze(-1)).squeeze(-1)

            v4_stack = torch.stack(
                [beta_v4[:, k:k + usable] for k in range(gamma)], dim=-1)
            eal_stack = torch.stack(
                [beta_eal[:, k:k + usable] for k in range(gamma)], dim=-1)
            v4_cum = torch.cumprod(v4_stack, dim=-1)
            eal_cum = torch.cumprod(eal_stack, dim=-1)

            v4_acc = (v4_cum * weights).sum(dim=-1)
            eal_acc = (eal_cum * weights).sum(dim=-1)
            # eal_weight defaults to 0.5; controlled via class attr (set in main).
            ew = getattr(self, 'eal_weight', 0.5)
            combined = v4_acc + ew * eal_acc
            aux = -(combined * win_mask.float()).sum() / (nv_win + 1e-8)

        elif aux_loss in ('v2', 'v4'):
            z_target = draft_lg.gather(
                -1, target_argmax.unsqueeze(-1)).squeeze(-1)
            z_max = draft_lg.max(dim=-1).values
            gap = (z_target - z_max) / temperature
            if aux_loss == 'v2':
                # β = σ(gap/T) + 0.5, clamped ≤ 1 → β ∈ [0.5, 1]
                beta = (torch.sigmoid(gap) + 0.5).clamp(max=1.0)
            else:  # 'v4'
                # β = 2σ(gap/T) → β ∈ [0, 1]; 2σ(0)=1 so no clamp needed
                beta = 2.0 * torch.sigmoid(gap)

            beta_stack = torch.stack(
                [beta[:, k:k + usable] for k in range(gamma)], dim=-1)
            cum_prod = torch.cumprod(beta_stack, dim=-1)
            acc_length = (cum_prod * weights).sum(dim=-1)
            aux = -(acc_length * win_mask.float()).sum() / (nv_win + 1e-8)

        elif aux_loss == 'v5':
            # β = 2σ(gap/T), cum_prod over window like v4, but truncate the
            # weighted sum at (including) the first hard reject per window.
            z_target = draft_lg.gather(
                -1, target_argmax.unsqueeze(-1)).squeeze(-1)
            z_max = draft_lg.max(dim=-1).values
            gap = (z_target - z_max) / temperature
            beta = 2.0 * torch.sigmoid(gap)
            beta_stack = torch.stack(
                [beta[:, k:k + usable] for k in range(gamma)], dim=-1)
            cum_prod = torch.cumprod(beta_stack, dim=-1)
            # Mask keeps cum_prod terms up to & including the first reject.
            with torch.no_grad():
                cumrej = (~accept_stack).int().cumsum(-1)
                trunc_mask = (cumrej <= 1).to(beta.dtype)
            acc_length = (cum_prod * trunc_mask * weights).sum(dim=-1)
            aux = -(acc_length * win_mask.float()).sum() / (nv_win + 1e-8)

        elif aux_loss == 'v8':
            # V8: gap to 2nd-largest draft logit (V4 but compare to top-2 not top-1).
            #   gap = (z[target] - z[2nd-max]) / T   ( > 0 if target=argmax,
            #                                          = 0 if target=2nd,
            #                                          < 0 otherwise)
            #   β = (2σ(gap)).clamp(max=1.0)
            # cum_prod over γ-window then sum (same framework as V4).
            z_target = draft_lg.gather(
                -1, target_argmax.unsqueeze(-1)).squeeze(-1)
            top2 = draft_lg.topk(2, dim=-1)
            z_2nd = top2.values[..., 1]
            gap = (z_target - z_2nd) / temperature
            beta = (2.0 * torch.sigmoid(gap)).clamp(max=1.0)
            beta_stack = torch.stack(
                [beta[:, k:k + usable] for k in range(gamma)], dim=-1)
            cum_prod = torch.cumprod(beta_stack, dim=-1)
            acc_length = (cum_prod * weights).sum(dim=-1)
            aux = -(acc_length * win_mask.float()).sum() / (nv_win + 1e-8)

        elif aux_loss == 'v7':
            # V7: top-2 aware sigmoid gap. Per position:
            #   gap1 = (z[target] - z[max])     / T   (V4-style, ≤ 0)
            #   gap2 = (z[target] - z[2nd-max]) / T   ( > 0 if target=argmax,
            #                                          = 0 if target=2nd,
            #                                          < 0 otherwise)
            #   β = (2σ(gap1) + 2σ(gap2)) / 2 = σ(gap1) + σ(gap2)
            # cum_prod over γ-window then sum (same as V4 framework).
            z_target = draft_lg.gather(
                -1, target_argmax.unsqueeze(-1)).squeeze(-1)
            top2 = draft_lg.topk(2, dim=-1)
            z_max = top2.values[..., 0]
            z_2nd = top2.values[..., 1]
            gap1 = (z_target - z_max) / temperature
            gap2 = (z_target - z_2nd) / temperature
            beta = torch.sigmoid(gap1) + torch.sigmoid(gap2)
            beta_stack = torch.stack(
                [beta[:, k:k + usable] for k in range(gamma)], dim=-1)
            cum_prod = torch.cumprod(beta_stack, dim=-1)
            acc_length = (cum_prod * weights).sum(dim=-1)
            aux = -(acc_length * win_mask.float()).sum() / (nv_win + 1e-8)

        elif aux_loss == 'v6':
            # β = 2σ(gap/T), cum_prod over window like v4.
            # Per-window dynamic weights: small_w before first reject,
            # peak_w AT first reject, 0 after. All-accept windows: small_w
            # uniform. aux_weight scheme is ignored (V6 uses intrinsic profile).
            z_target = draft_lg.gather(
                -1, target_argmax.unsqueeze(-1)).squeeze(-1)
            z_max = draft_lg.max(dim=-1).values
            gap = (z_target - z_max) / temperature
            beta = 2.0 * torch.sigmoid(gap)
            beta_stack = torch.stack(
                [beta[:, k:k + usable] for k in range(gamma)], dim=-1)
            cum_prod = torch.cumprod(beta_stack, dim=-1)
            small_w, peak_w = 0.2, 1.0
            with torch.no_grad():
                reject_stack = ~accept_stack
                cumrej = reject_stack.int().cumsum(-1)
                is_first_reject = (cumrej == 1) & reject_stack
                is_before_first = cumrej == 0
                v6_w = (is_first_reject.to(beta.dtype) * peak_w
                        + is_before_first.to(beta.dtype) * small_w)
            acc_length = (cum_prod * v6_w).sum(dim=-1)
            aux = -(acc_length * win_mask.float()).sum() / (nv_win + 1e-8)

        elif aux_loss == 'eal':
            # Expected Accept Length:
            #   P_t = softmax(draft_logit_t)[target_argmax_t]    ∈ [0, 1]
            #   E[τ] = Σ_{k=1..γ-1} k · cum[k-1]·(1-P[k]) + γ · cum[γ-1]
            #        = Σ_{k=0..γ-1} cum_prod[k]                  (algebraic identity)
            # We weight each cum_prod term by aux_weight scheme (uniform = exact EAL).
            draft_q = F.softmax(draft_lg, dim=-1)
            P = draft_q.gather(
                -1, target_argmax.unsqueeze(-1)).squeeze(-1)         # [B, L-1]
            P_stack = torch.stack(
                [P[:, k:k + usable] for k in range(gamma)], dim=-1)
            cum_prod = torch.cumprod(P_stack, dim=-1)                # [B, U, γ]
            eal = (cum_prod * weights).sum(dim=-1)                   # [B, U]
            aux = -(eal * win_mask.float()).sum() / (nv_win + 1e-8)

        elif aux_loss == 'v4_w1':
            # V4 with window size 1 (no sliding window, no cumprod):
            #   β = 2σ((z_target - z_max) / T)   only at the anchor position.
            #   L = -mean(β) over valid positions.
            z_target = draft_lg.gather(
                -1, target_argmax.unsqueeze(-1)).squeeze(-1)        # [B, L-1]
            z_max = draft_lg.max(dim=-1).values
            gap = (z_target - z_max) / temperature
            beta = 2.0 * torch.sigmoid(gap)
            aux = -(beta * mask.float()).sum() / (nv + 1e-8)

        elif aux_loss == 'eal_w1':
            # EAL with window size 1:
            #   P = softmax(draft_logit)[target_argmax]   only at anchor position.
            #   L = -mean(P) over valid positions.
            draft_q = F.softmax(draft_lg, dim=-1)
            P = draft_q.gather(
                -1, target_argmax.unsqueeze(-1)).squeeze(-1)        # [B, L-1]
            aux = -(P * mask.float()).sum() / (nv + 1e-8)

        elif aux_loss == 'peal':
            # Penalty-augmented EAL:
            #   score_t = P_target_t - Σ_{j: P_j > P_target_t} P_j
            # Same EAL framework but with distractor penalty per position.
            # When target is argmax, penalty=0 → score = P_target (matches EAL).
            # When target is rank-K, score becomes negative → cum_prod sign flips,
            # strongly punishing positions where target is dominated.
            draft_q = F.softmax(draft_lg, dim=-1)
            P_target = draft_q.gather(
                -1, target_argmax.unsqueeze(-1)).squeeze(-1)            # [B, L-1]
            distractor_mask = (draft_q > P_target.unsqueeze(-1)).float()  # [B, L-1, V]
            penalty = (draft_q * distractor_mask).sum(dim=-1)           # [B, L-1]
            P_score = P_target - penalty                                # [B, L-1], can be < 0
            P_stack = torch.stack(
                [P_score[:, k:k + usable] for k in range(gamma)], dim=-1)
            cum_prod = torch.cumprod(P_stack, dim=-1)                   # [B, U, γ]
            peal = (cum_prod * weights).sum(dim=-1)                     # [B, U]
            aux = -(peal * win_mask.float()).sum() / (nv_win + 1e-8)

        elif aux_loss == 'al_tv':
            # E[L] with β_j = 1 − TV(p̃,q̃) = Σ min(p̃,q̃) per position.
            # p̃ = softmax(z/T), q̃ = softmax(z_d/T).  cum_prod over γ-window then sum.
            T = prob_temperature
            target_p_T = F.softmax(target_lg / T, dim=-1) if T != 1.0 else target_p
            draft_q = F.softmax(draft_lg / T, dim=-1)
            beta = torch.sum(torch.min(target_p_T, draft_q), dim=-1)      # [B, L-1]
            beta_stack = torch.stack(
                [beta[:, k:k + usable] for k in range(gamma)], dim=-1)
            cum_prod = torch.cumprod(beta_stack, dim=-1)                # [B, U, γ]
            acc_length = (cum_prod * weights).sum(dim=-1)               # [B, U]
            focal_alpha = float(getattr(self, 'altv_focal_alpha', 0.0))
            focal_mode = getattr(self, 'altv_focal_mode', 'etau')
            if focal_alpha > 0:
                if focal_mode == 'beta':
                    avg_beta = beta_stack.mean(dim=-1).detach()          # [B, U]
                    difficulty = (1.0 - avg_beta).clamp(0.0, 1.0)
                else:  # 'etau'
                    difficulty = ((gamma - acc_length.detach()) / gamma).clamp(0.0, 1.0)
                w_anchor = difficulty ** focal_alpha                     # [B, U]
                w_masked = w_anchor * win_mask.float()
                aux = -(acc_length * w_masked).sum() / w_masked.sum().clamp(min=1e-8)
            else:
                aux = -(acc_length * win_mask.float()).sum() / (nv_win + 1e-8)

        elif aux_loss == 'al_tv_intersect_topk':
            # Variant 1: draft keeps only intersect(topK_draft, topK_target), zeros elsewhere.
            # target keeps only topK_target. Both renormalize, then TV overlap.
            T = prob_temperature
            K = int(getattr(self, 'altv_topk', 20))
            target_p_T = F.softmax(target_lg / T, dim=-1)
            draft_q = F.softmax(draft_lg / T, dim=-1)
            _, tk_p = torch.topk(target_p_T, K, dim=-1)
            _, tk_q = torch.topk(draft_q, K, dim=-1)
            mask_p = torch.zeros_like(target_p_T).scatter_(-1, tk_p, 1.0)
            mask_q = torch.zeros_like(draft_q).scatter_(-1, tk_q, 1.0)
            target_masked = target_p_T * mask_p
            target_norm = target_masked / target_masked.sum(-1, keepdim=True).clamp(min=1e-20)
            draft_masked = draft_q * mask_p * mask_q
            draft_norm = draft_masked / draft_masked.sum(-1, keepdim=True).clamp(min=1e-20)
            beta = torch.sum(torch.min(target_norm, draft_norm), dim=-1)
            beta_stack = torch.stack(
                [beta[:, k:k + usable] for k in range(gamma)], dim=-1)
            cum_prod = torch.cumprod(beta_stack, dim=-1)
            acc_length = (cum_prod * weights).sum(dim=-1)
            aux = -(acc_length * win_mask.float()).sum() / (nv_win + 1e-8)

        elif aux_loss == 'al_tv_target_topk':
            # Variant 2: target keeps only topK_target (renormalized), draft keeps full vocab.
            # TV overlap: only positions in target's topK contribute (others → target=0 → min=0).
            T = prob_temperature
            K = int(getattr(self, 'altv_topk', 20))
            target_p_T = F.softmax(target_lg / T, dim=-1)
            draft_q = F.softmax(draft_lg / T, dim=-1)
            top_p_vals, tk_p = torch.topk(target_p_T, K, dim=-1)             # [B, L-1, K]
            target_norm_top = top_p_vals / top_p_vals.sum(-1, keepdim=True).clamp(min=1e-20)
            draft_at_top = draft_q.gather(-1, tk_p)                          # [B, L-1, K]
            beta = torch.sum(torch.min(target_norm_top, draft_at_top), dim=-1)
            beta_stack = torch.stack(
                [beta[:, k:k + usable] for k in range(gamma)], dim=-1)
            cum_prod = torch.cumprod(beta_stack, dim=-1)
            acc_length = (cum_prod * weights).sum(dim=-1)
            aux = -(acc_length * win_mask.float()).sum() / (nv_win + 1e-8)

        elif aux_loss == 'al_tv_target_topp':
            # Variant: target keeps smallest nucleus s.t. cumulative prob ≥ P.
            # per-position dynamic K; sharp positions collapse to K=1, flat positions extend.
            # NOTE: draft is masked but NOT renormalized (matches al_tv_target_topk).
            # Renormalizing draft would make β ≡ 1 at K=1 positions, killing the gradient.
            P = float(getattr(self, 'altv_topp', 0.99))
            T = prob_temperature
            target_p_T = F.softmax(target_lg / T, dim=-1)
            draft_q = F.softmax(draft_lg / T, dim=-1)
            sorted_p, sort_idx = torch.sort(target_p_T, descending=True, dim=-1)
            cum_p = torch.cumsum(sorted_p, dim=-1)
            keep_sorted = (cum_p - sorted_p) < P
            keep_sorted[..., 0] = True   # always keep top-1
            mask = torch.zeros_like(target_p_T).scatter_(-1, sort_idx, keep_sorted.float())
            target_masked = target_p_T * mask
            target_norm = target_masked / target_masked.sum(-1, keepdim=True).clamp(min=1e-20)
            draft_masked = draft_q * mask   # raw draft mass on nucleus, no renorm
            beta = torch.sum(torch.min(target_norm, draft_masked), dim=-1)
            beta_stack = torch.stack(
                [beta[:, k:k + usable] for k in range(gamma)], dim=-1)
            cum_prod = torch.cumprod(beta_stack, dim=-1)
            acc_length = (cum_prod * weights).sum(dim=-1)
            aux = -(acc_length * win_mask.float()).sum() / (nv_win + 1e-8)

        elif aux_loss == 'al_tv_target_topk_topp':
            # Hybrid: keep union of target's top-K and top-P nucleus (both start at
            # rank 0 contiguously, so the union = max(K, K_p) tokens per position).
            # target renormalized within the kept set; draft raw (matches topk).
            K = int(getattr(self, 'altv_topk', 20))
            P = float(getattr(self, 'altv_topp', 0.99))
            T = prob_temperature
            target_p_T = F.softmax(target_lg / T, dim=-1)
            draft_q = F.softmax(draft_lg / T, dim=-1)
            sorted_p, sort_idx = torch.sort(target_p_T, descending=True, dim=-1)
            cum_p = torch.cumsum(sorted_p, dim=-1)
            keep_p_sorted = (cum_p - sorted_p) < P
            keep_p_sorted[..., 0] = True
            rank_ids = torch.arange(sorted_p.shape[-1], device=sorted_p.device)
            keep_k_sorted = (rank_ids < K).expand_as(sorted_p)
            keep_sorted = keep_p_sorted | keep_k_sorted
            mask = torch.zeros_like(target_p_T).scatter_(-1, sort_idx, keep_sorted.float())
            target_masked = target_p_T * mask
            target_norm = target_masked / target_masked.sum(-1, keepdim=True).clamp(min=1e-20)
            draft_masked = draft_q * mask   # raw, no renorm
            beta = torch.sum(torch.min(target_norm, draft_masked), dim=-1)
            beta_stack = torch.stack(
                [beta[:, k:k + usable] for k in range(gamma)], dim=-1)
            cum_prod = torch.cumprod(beta_stack, dim=-1)
            acc_length = (cum_prod * weights).sum(dim=-1)
            aux = -(acc_length * win_mask.float()).sum() / (nv_win + 1e-8)

        elif aux_loss == 'al_tv_renewal':
            # Renewal-weighted TV objective (paper §4.1-4.2):
            #   L = -Σ_t ρ̄_t · cumprod(β_{t..t+γ-1}),
            # where ρ_t is the probability a window starts at position t under the
            # forward renewal recursion (segment restart with ρ_0 = 1).  ρ is
            # detached so gradient flows only through the acc_length term.
            T = prob_temperature
            target_p_T = F.softmax(target_lg / T, dim=-1) if T != 1.0 else target_p
            draft_q = F.softmax(draft_lg / T, dim=-1)
            beta = torch.sum(torch.min(target_p_T, draft_q), dim=-1)   # [B, L-1]
            beta_stack = torch.stack(
                [beta[:, k:k + usable] for k in range(gamma)], dim=-1)  # [B, U, γ]
            cum_prod = torch.cumprod(beta_stack, dim=-1)                # [B, U, γ]
            acc_length = (cum_prod * weights).sum(dim=-1)               # [B, U]
            with torch.no_grad():
                beta_d = beta_stack.detach()                            # [B, U, γ]
                cum_d = cum_prod.detach()                               # [B, U, γ]
                # P(L_t = k) for k in [0, γ]:
                #   k=0..γ-1: (Π_{j<k} β_{t+j}) · (1 − β_{t+k})
                #   k=γ    : Π_{j<γ} β_{t+j}
                one = torch.ones_like(beta_d[..., :1])
                prefix = torch.cat([one, cum_d[..., :-1]], dim=-1)      # [B, U, γ]: Π_{j<k} β
                pk_short = prefix * (1.0 - beta_d)                       # [B, U, γ]: k=0..γ-1
                pk_full = cum_d[..., -1]                                 # [B, U]:   k=γ
                B_size, U = cum_d.shape[0], cum_d.shape[1]
                rho = torch.zeros(B_size, U, device=beta.device, dtype=beta.dtype)
                rho[:, 0] = 1.0
                # ρ_t = Σ_{k=0..γ} ρ_{t-k-1} · P(L_{t-k-1} = k), jump k+1.
                for tt in range(1, U):
                    acc_rho = torch.zeros(B_size, device=beta.device, dtype=beta.dtype)
                    kmax = min(gamma - 1, tt - 1)   # k in [0, γ-1] for pk_short
                    for k in range(kmax + 1):
                        src = tt - k - 1
                        acc_rho = acc_rho + rho[:, src] * pk_short[:, src, k]
                    src_full = tt - gamma - 1        # k = γ case (pk_full)
                    if src_full >= 0:
                        acc_rho = acc_rho + rho[:, src_full] * pk_full[:, src_full]
                    rho[:, tt] = acc_rho
                rho_masked = rho * win_mask.float()
                rho_norm = rho_masked / rho_masked.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            aux = -(acc_length * rho_norm).sum(dim=-1).mean()

        elif aux_loss == 'al_tv_target_topk_renewal':
            # V2 (target top-K β) + renewal (ρ̄ anchor weighting).
            # β is computed on target's top-K distribution (renormalized) — matches
            # target-topK inference verify.  ρ_t is the forward renewal recursion
            # over that SAME β (detached), giving anchor weights that concentrate
            # right after rejections, consistent with the topK loss.
            T = prob_temperature
            K = int(getattr(self, 'altv_topk', 20))
            target_p_T = F.softmax(target_lg / T, dim=-1)
            draft_q = F.softmax(draft_lg / T, dim=-1)
            top_p_vals, tk_p = torch.topk(target_p_T, K, dim=-1)             # [B, L-1, K]
            target_norm_top = top_p_vals / top_p_vals.sum(-1, keepdim=True).clamp(min=1e-20)
            draft_at_top = draft_q.gather(-1, tk_p)                          # [B, L-1, K]
            beta = torch.sum(torch.min(target_norm_top, draft_at_top), dim=-1)  # [B, L-1]
            beta_stack = torch.stack(
                [beta[:, k:k + usable] for k in range(gamma)], dim=-1)       # [B, U, γ]
            cum_prod = torch.cumprod(beta_stack, dim=-1)                     # [B, U, γ]
            acc_length = (cum_prod * weights).sum(dim=-1)                    # [B, U]
            with torch.no_grad():
                beta_d = beta_stack.detach()
                cum_d = cum_prod.detach()
                one = torch.ones_like(beta_d[..., :1])
                prefix = torch.cat([one, cum_d[..., :-1]], dim=-1)
                pk_short = prefix * (1.0 - beta_d)
                pk_full = cum_d[..., -1]
                B_size, U = cum_d.shape[0], cum_d.shape[1]
                rho = torch.zeros(B_size, U, device=beta.device, dtype=beta.dtype)
                rho[:, 0] = 1.0
                for tt in range(1, U):
                    acc_rho = torch.zeros(B_size, device=beta.device, dtype=beta.dtype)
                    kmax = min(gamma - 1, tt - 1)
                    for k in range(kmax + 1):
                        src = tt - k - 1
                        acc_rho = acc_rho + rho[:, src] * pk_short[:, src, k]
                    src_full = tt - gamma - 1
                    if src_full >= 0:
                        acc_rho = acc_rho + rho[:, src_full] * pk_full[:, src_full]
                    rho[:, tt] = acc_rho
                rho_masked = rho * win_mask.float()
                rho_norm = rho_masked / rho_masked.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            aux = -(acc_length * rho_norm).sum(dim=-1).mean()

        elif aux_loss == 'al_kl':
            # E[L] with β_j = 0.5 · exp(−KL(p̃‖q̃)) per position.
            # p̃ = softmax(z/T), q̃ = softmax(z_d/T).  log β = log(0.5) − KL.
            T = prob_temperature
            target_p_T = F.softmax(target_lg / T, dim=-1) if T != 1.0 else target_p
            target_logp_T = F.log_softmax(target_lg / T, dim=-1) if T != 1.0 \
                            else torch.log(target_p.clamp(min=1e-30))
            draft_logp_T = F.log_softmax(draft_lg / T, dim=-1) if T != 1.0 else draft_logp
            kl = (target_p_T * (target_logp_T - draft_logp_T)).sum(dim=-1)    # [B, L-1]
            log_beta = math.log(0.5) - kl
            log_beta_stack = torch.stack(
                [log_beta[:, k:k + usable] for k in range(gamma)], dim=-1)
            cum_log = torch.cumsum(log_beta_stack, dim=-1)              # [B, U, γ]
            acc_length = (torch.exp(cum_log) * weights).sum(dim=-1)     # [B, U]
            aux = -(acc_length * win_mask.float()).sum() / (nv_win + 1e-8)

        elif aux_loss == 'wkl':
            # L̃_EAL = Σ_{j=0..γ-1} (γ − j) · KL(p̃ ‖ q̃) per position-step.
            # p̃ = softmax(z/T), q̃ = softmax(z_d/T).  Linear-decay weighted per-step KL.
            T = prob_temperature
            target_p_T = F.softmax(target_lg / T, dim=-1) if T != 1.0 else target_p
            target_logp_T = F.log_softmax(target_lg / T, dim=-1) if T != 1.0 \
                            else torch.log(target_p.clamp(min=1e-30))
            draft_logp_T = F.log_softmax(draft_lg / T, dim=-1) if T != 1.0 else draft_logp
            kl = (target_p_T * (target_logp_T - draft_logp_T)).sum(dim=-1)    # [B, L-1]
            kl_stack = torch.stack(
                [kl[:, k:k + usable] for k in range(gamma)], dim=-1)
            lin_w = torch.tensor([gamma - j for j in range(gamma)],
                                 device=kl.device, dtype=kl.dtype)
            weighted_kl = (kl_stack * lin_w).sum(dim=-1)                # [B, U]
            aux = (weighted_kl * win_mask.float()).sum() / (nv_win + 1e-8)

        elif aux_loss == 'tree_v4':
            # Tree V4: sigmoid gap to K-th highest draft logit per depth d.
            # β_d = 2σ((z_draft[y*_target] − z_draft[K_d-th]) / T) ∈ [0, 1]
            # Cumulative product over γ depths, weighted sum as accept length.
            # top_k_per_depth: list[int] of length γ (set via self.top_k_per_depth).
            #   K=1 → equivalent to chain v4.
            #   K=[4,3,2,1,1,1,1] → matches production tree eval.
            tk = getattr(self, 'top_k_per_depth', [1] * gamma)
            assert len(tk) == gamma, \
                f"top_k_per_depth must have length gamma={gamma}, got {len(tk)}"
            T = temperature
            betas = []
            for d in range(gamma):
                K = tk[d]
                z_slice = draft_lg[:, d:d + usable]                       # [B, U, V]
                y_slice = target_argmax[:, d:d + usable]                  # [B, U]
                z_y = z_slice.gather(-1, y_slice.unsqueeze(-1)).squeeze(-1)  # [B, U]
                top_k_vals, _ = torch.topk(z_slice, K, dim=-1)            # [B, U, K]
                z_K_th = top_k_vals[..., -1]                              # [B, U]
                gap = (z_y - z_K_th) / T
                beta_d = 2.0 * torch.sigmoid(gap)
                betas.append(beta_d)
            beta_stack = torch.stack(betas, dim=-1)                        # [B, U, γ]
            cum_prod = torch.cumprod(beta_stack, dim=-1)
            acc_length = (cum_prod * weights).sum(dim=-1)                  # [B, U]
            aux = -(acc_length * win_mask.float()).sum() / (nv_win + 1e-8)

        elif aux_loss == 'altv_p_tree_altv':
            # ALTV + tree_altv_mix · tree_ALTV.
            # aux = L_altv + tree_altv_mix · L_tree_altv (both negative acc-length terms)
            # anchor is applied separately (e.g., anchor=kl, sigmoid_coef=0.1 gives
            # total = L_KL + 0.1 · (L_altv + mix · L_tree_altv)).
            tk = getattr(self, 'top_k_per_depth', [1] * gamma)
            assert len(tk) == gamma, \
                f"top_k_per_depth must have length gamma={gamma}, got {len(tk)}"
            mix = getattr(self, 'tree_altv_mix', 0.1)
            T = prob_temperature
            target_p_T = F.softmax(target_lg / T, dim=-1) if T != 1.0 else target_p
            draft_q = F.softmax(draft_lg / T, dim=-1)
            # ── Full ALTV (aux=al_tv) ──
            beta_full = torch.sum(torch.min(target_p_T, draft_q), dim=-1)  # [B, L-1]
            beta_full_stack = torch.stack(
                [beta_full[:, k:k + usable] for k in range(gamma)], dim=-1)
            cum_full = torch.cumprod(beta_full_stack, dim=-1)
            acc_full = (cum_full * weights).sum(dim=-1)                    # [B, U]
            L_altv = -(acc_full * win_mask.float()).sum() / (nv_win + 1e-8)
            # ── Tree ALTV (aux=tree_altv) ──
            beta_tree_list = []
            for d in range(gamma):
                K = tk[d]
                q_slice = draft_q[:, d:d + usable]
                p_slice = target_p_T[:, d:d + usable]
                _, top_k_idx = torch.topk(q_slice, K, dim=-1)
                top_k_mask = torch.zeros_like(q_slice).scatter_(
                    -1, top_k_idx, 1.0)
                beta_d = (top_k_mask * torch.minimum(q_slice, p_slice)).sum(-1)
                beta_tree_list.append(beta_d)
            beta_tree_stack = torch.stack(beta_tree_list, dim=-1)          # [B, U, γ]
            cum_tree = torch.cumprod(beta_tree_stack, dim=-1)
            acc_tree = (cum_tree * weights).sum(dim=-1)                    # [B, U]
            L_tree_altv = -(acc_tree * win_mask.float()).sum() / (nv_win + 1e-8)
            aux = L_altv + mix * L_tree_altv

        elif aux_loss == 'tree_altv':
            # Tree ALTV: top-K-restricted min-overlap.
            # β_d = Σ_{v ∈ top-K_d(q_draft)} min(q_draft(v), p_target(v))
            # Same cum-prod structure as al_tv, but sum limited to draft top-K set.
            tk = getattr(self, 'top_k_per_depth', [1] * gamma)
            assert len(tk) == gamma, \
                f"top_k_per_depth must have length gamma={gamma}, got {len(tk)}"
            T = prob_temperature
            target_p_T = F.softmax(target_lg / T, dim=-1) if T != 1.0 else target_p
            draft_q = F.softmax(draft_lg / T, dim=-1)
            betas = []
            for d in range(gamma):
                K = tk[d]
                q_slice = draft_q[:, d:d + usable]                        # [B, U, V]
                p_slice = target_p_T[:, d:d + usable]
                _, top_k_idx = torch.topk(q_slice, K, dim=-1)             # [B, U, K]
                top_k_mask = torch.zeros_like(q_slice).scatter_(
                    -1, top_k_idx, 1.0)                                     # [B, U, V]
                beta_d = (top_k_mask * torch.minimum(q_slice, p_slice)).sum(-1)
                betas.append(beta_d)
            beta_stack = torch.stack(betas, dim=-1)                        # [B, U, γ]
            cum_prod = torch.cumprod(beta_stack, dim=-1)
            acc_length = (cum_prod * weights).sum(dim=-1)                  # [B, U]
            aux = -(acc_length * win_mask.float()).sum() / (nv_win + 1e-8)

        elif aux_loss == 'tree_altv_wk':
            # Tree ALTV weighted-K: full-vocab min-overlap with extra weight on
            # draft top-K positions.
            # β_d = Σ_v min(q_draft(v), p_target(v)) · (1 + wk · 1[v ∈ top-K_d(q_draft)])
            # wk=0 → identical to al_tv (full ALTV). wk→∞ (relative) → identical to
            # tree_altv (top-K restricted). wk=1 → top-K contribution doubled.
            tk = getattr(self, 'top_k_per_depth', [1] * gamma)
            assert len(tk) == gamma, \
                f"top_k_per_depth must have length gamma={gamma}, got {len(tk)}"
            wk = getattr(self, 'wk_bonus', 1.0)
            T = prob_temperature
            target_p_T = F.softmax(target_lg / T, dim=-1) if T != 1.0 else target_p
            draft_q = F.softmax(draft_lg / T, dim=-1)
            betas = []
            for d in range(gamma):
                K = tk[d]
                q_slice = draft_q[:, d:d + usable]                        # [B, U, V]
                p_slice = target_p_T[:, d:d + usable]
                overlap = torch.minimum(q_slice, p_slice)                 # [B, U, V]
                _, top_k_idx = torch.topk(q_slice, K, dim=-1)             # [B, U, K]
                top_k_mask = torch.zeros_like(q_slice).scatter_(
                    -1, top_k_idx, 1.0)                                     # [B, U, V]
                weight_v = 1.0 + wk * top_k_mask                          # [B, U, V]
                beta_d = (weight_v * overlap).sum(-1)
                betas.append(beta_d)
            beta_stack = torch.stack(betas, dim=-1)                        # [B, U, γ]
            cum_prod = torch.cumprod(beta_stack, dim=-1)
            acc_length = (cum_prod * weights).sum(dim=-1)                  # [B, U]
            aux = -(acc_length * win_mask.float()).sum() / (nv_win + 1e-8)

        elif aux_loss == 'topk_softce':
            # Top-K restricted soft-CE (target-weighted CE on draft's top-K set).
            # Per depth d, position offset d:
            #   L_d = -Σ_v 1[v ∈ draft_top_K_d] · p_target(v) · log q_draft(v)
            # Sum over depths with pow08 position weights, avg over windows.
            # Related to KL anchor (-Σ_v p_target · log q_draft) but restricted to
            # draft's top-K candidates at each depth (matches tree verify topology).
            tk = getattr(self, 'top_k_per_depth', [1] * gamma)
            assert len(tk) == gamma, \
                f"top_k_per_depth must have length gamma={gamma}, got {len(tk)}"
            draft_q = F.softmax(draft_lg, dim=-1)
            Ls = []
            for d in range(gamma):
                K = tk[d]
                q_slice = draft_q[:, d:d + usable]                        # [B, U, V]
                p_slice = target_p[:, d:d + usable]                       # [B, U, V]
                logq_slice = draft_logp[:, d:d + usable]                  # [B, U, V]
                _, top_k_idx = torch.topk(q_slice, K, dim=-1)             # [B, U, K]
                top_k_mask = torch.zeros_like(q_slice).scatter_(
                    -1, top_k_idx, 1.0)                                     # [B, U, V]
                L_d = -(top_k_mask * p_slice * logq_slice).sum(-1)         # [B, U]
                Ls.append(L_d)
            L_stack = torch.stack(Ls, dim=-1)                              # [B, U, γ]
            L_win = (L_stack * weights).sum(dim=-1)                        # [B, U]
            aux = (L_win * win_mask.float()).sum() / (nv_win + 1e-8)

        elif aux_loss == 'tree_ce_hard':
            # Tree-gated hard-label CE (Version A).
            # gate_{w,d} = Π_{j=0..d} 1[y*_{w+j} ∈ draft_top_K_j]  (hard mask, detached)
            # L = Σ_w Σ_d weight[d] · gate_{w,d} · (-log q_draft(y*_{w+d}))
            # Real CE shape (uses -log q, not q); gate acts as a per-position weight
            # that mimics "target argmax stays inside draft top-K along the tree path".
            tk = getattr(self, 'top_k_per_depth', [1] * gamma)
            assert len(tk) == gamma, \
                f"top_k_per_depth must have length gamma={gamma}, got {len(tk)}"
            draft_q = F.softmax(draft_lg, dim=-1)
            log_q_y = draft_logp.gather(
                -1, target_argmax.unsqueeze(-1)).squeeze(-1)              # [B, L-1]
            in_topk_list, ce_list = [], []
            for d in range(gamma):
                K = tk[d]
                q_slice = draft_q[:, d:d + usable]                        # [B, U, V]
                y_slice = target_argmax[:, d:d + usable]                  # [B, U]
                q_y = q_slice.gather(-1, y_slice.unsqueeze(-1)).squeeze(-1)
                top_k_vals, _ = torch.topk(q_slice, K, dim=-1)
                threshold = top_k_vals[..., -1]
                in_topk_list.append((q_y >= threshold).float().detach())  # [B, U]
                ce_list.append(-log_q_y[:, d:d + usable])                 # [B, U]
            in_topk_stack = torch.stack(in_topk_list, dim=-1)              # [B, U, γ]
            ce_stack = torch.stack(ce_list, dim=-1)                        # [B, U, γ]
            gate = torch.cumprod(in_topk_stack, dim=-1)                    # [B, U, γ]
            L_win = (gate * ce_stack * weights).sum(dim=-1)                # [B, U]
            aux = (L_win * win_mask.float()).sum() / (nv_win + 1e-8)

        elif aux_loss == 'tree_ce_soft':
            # EAL-weighted CE (Version B).
            # β_d = q_draft(y*_d), gate_{w,d} = Π_{j=0..d} β_j  (soft cumulative accept prob)
            # L = Σ_w Σ_d weight[d] · gate_{w,d}.detach() · (-log q_draft(y*_{w+d}))
            # Real CE shape; gate is soft (EAL cumulative prob) but detached so only
            # the -log q term contributes gradient. Position weight scales by the
            # empirical accept prob so deep unlikely positions get less CE pressure.
            draft_q = F.softmax(draft_lg, dim=-1)
            log_q_y = draft_logp.gather(
                -1, target_argmax.unsqueeze(-1)).squeeze(-1)              # [B, L-1]
            beta_list, ce_list = [], []
            for d in range(gamma):
                y_slice = target_argmax[:, d:d + usable]
                q_slice = draft_q[:, d:d + usable]
                q_y = q_slice.gather(-1, y_slice.unsqueeze(-1)).squeeze(-1)
                beta_list.append(q_y)                                     # [B, U]
                ce_list.append(-log_q_y[:, d:d + usable])                 # [B, U]
            beta_stack = torch.stack(beta_list, dim=-1)                    # [B, U, γ]
            ce_stack = torch.stack(ce_list, dim=-1)                        # [B, U, γ]
            gate = torch.cumprod(beta_stack, dim=-1).detach()              # [B, U, γ]
            L_win = (gate * ce_stack * weights).sum(dim=-1)                # [B, U]
            aux = (L_win * win_mask.float()).sum() / (nv_win + 1e-8)

        elif aux_loss == 'tree_ce':
            # Tree CE: hard-label CE with tree top-K coverage constraint.
            # β_d = q_draft(y*_target) if y*_target ∈ top-K_d else 0
            # Plus coverage penalty: relu(q_draft(K-th) − q_draft(y*)) — pushes
            # y* into top-K when it's out. `coverage_lambda` (default 0.5) sets
            # the trade-off between EAL reward and coverage penalty.
            tk = getattr(self, 'top_k_per_depth', [1] * gamma)
            assert len(tk) == gamma, \
                f"top_k_per_depth must have length gamma={gamma}, got {len(tk)}"
            coverage_lambda = getattr(self, 'coverage_lambda', 0.5)
            draft_q = F.softmax(draft_lg, dim=-1)
            betas, coverages = [], []
            for d in range(gamma):
                K = tk[d]
                q_slice = draft_q[:, d:d + usable]                        # [B, U, V]
                y_slice = target_argmax[:, d:d + usable]                  # [B, U]
                q_y = q_slice.gather(-1, y_slice.unsqueeze(-1)).squeeze(-1)  # [B, U]
                top_k_vals, _ = torch.topk(q_slice, K, dim=-1)            # [B, U, K]
                threshold = top_k_vals[..., -1]                           # [B, U]  K-th highest prob
                # β_d = q(y*) if y* in top-K else 0 (mask detached)
                in_top_k = (q_y >= threshold).float().detach()
                beta_d = q_y * in_top_k
                betas.append(beta_d)
                # Coverage penalty: relu(threshold − q(y*)) — provides gradient
                # to push q(y*) above K-th when y* is out of top-K.
                coverages.append(F.relu(threshold - q_y))
            beta_stack = torch.stack(betas, dim=-1)                        # [B, U, γ]
            cum_prod = torch.cumprod(beta_stack, dim=-1)
            acc_length = (cum_prod * weights).sum(dim=-1)                  # [B, U]
            cov_stack = torch.stack(coverages, dim=-1)                     # [B, U, γ]
            cov_weighted = (cov_stack * weights).sum(dim=-1)               # [B, U]
            eal_reward = (acc_length * win_mask.float()).sum() / (nv_win + 1e-8)
            cov_penalty = (cov_weighted * win_mask.float()).sum() / (nv_win + 1e-8)
            aux = -eal_reward + coverage_lambda * cov_penalty

        else:  # 'tv'
            draft_q = F.softmax(draft_lg, dim=-1)
            overlap = torch.sum(torch.min(target_p, draft_q), dim=-1)  # [B,L-1]
            tv_per_pos = 1.0 - overlap
            tv_stack = torch.stack(
                [tv_per_pos[:, k:k + usable] for k in range(gamma)], dim=-1)
            weighted_tv = (tv_stack * weights).sum(dim=-1)
            aux = (weighted_tv * win_mask.float()).sum() / (nv_win + 1e-8)

        total_loss = anchor_loss + sigmoid_coef * aux + grpo_coef * grpo_loss

        # ── Metrics ──
        with torch.no_grad():
            hard_cp = torch.cumprod(accept_stack.float(), dim=-1)
            tau = hard_cp.sum(dim=-1)
            mean_tau_val = (tau * win_mask.float()).sum() / (nv_win + 1e-8)
            step_acc = [
                accept_stack[:, :, k][win_mask].float().mean().item()
                if win_mask.any() else 0.0
                for k in range(gamma)
            ]
            tau_int = tau[win_mask].long().clamp(max=gamma)
            tau_hist = [(tau_int == v).sum().item() for v in range(gamma + 1)]

        return total_loss, aux, {
            "aux_loss": aux.item(),
            "eagle_loss": anchor_loss.item(),
            "grpo_loss": grpo_loss.item(),
            "mean_tau": mean_tau_val.item(),
            "num_valid": nv_win.item(),
            "step_acc": step_acc,
            "tau_hist": tau_hist,
        }

    def _compute_grpo_loss(self, input_ids, attention_mask, mask,
                            draft_lg, target_argmax,
                            K_groups, m_win, eps, reward_type='hard'):
        """GRPO on K_groups × m_win consecutive sliding γ-windows.

        Reward per sample (window at position i):
            R_i = τ_current − τ_ref                       (debiased, eq. 9)
        Advantage per group:
            A_i = (R_i − mean(R)) / (std(R) + δ)          (standardize, eq. 10)
        Ratio (per-token geometric mean on current greedy trajectory):
            ρ_i = exp((log π(S_i) − log π_ref(S_i)) / γ)  (eq. 11)
        PPO-clipped loss:
            L = −(1/m) Σ min(ρ_i A_i, clip(ρ_i, 1±ε) A_i)  (eq. 12)
        """
        device = input_ids.device
        B = input_ids.shape[0]
        gamma = self.gamma
        span = m_win + gamma - 1

        with torch.no_grad():
            ref_lg_full = self.ref_draft_model(
                input_ids=input_ids,
                attention_mask=attention_mask).logits[:, :-1, :].float()

        grpo_losses = []
        for b in range(B):
            mask_b = mask[b].int()
            Lp = mask_b.shape[0]
            if Lp < span:
                continue
            cum = torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                             torch.cumsum(mask_b, 0)])
            valid = (cum[span:] - cum[:Lp - span + 1]) == span
            valid_starts = valid.nonzero(as_tuple=True)[0].tolist()
            if not valid_starts:
                continue

            random.shuffle(valid_starts)
            selected = []
            for s in valid_starts:
                if all(abs(s - u) >= span for u in selected):
                    selected.append(s)
                    if len(selected) == K_groups:
                        break
            # Use as many groups as fit; short sequences contribute fewer groups
            # rather than being skipped entirely.
            if not selected:
                continue

            for g_start in selected:
                rewards = []
                log_ratios = []
                for offset in range(m_win):
                    i = g_start + offset
                    draft_lg_win = draft_lg[b, i:i + gamma]           # [γ, V]
                    ref_lg_win = ref_lg_full[b, i:i + gamma]          # [γ, V]
                    tgt_arg_win = target_argmax[b, i:i + gamma]       # [γ]

                    S_curr = draft_lg_win.argmax(-1)                  # [γ]
                    if reward_type == 'eal':
                        # EAL: cumprod(softmax(z)[target]) summed
                        P_curr = F.softmax(draft_lg_win, dim=-1).gather(
                            -1, tgt_arg_win.unsqueeze(-1)).squeeze(-1)
                        tau_curr = P_curr.cumprod(-1).sum()
                        with torch.no_grad():
                            P_ref = F.softmax(ref_lg_win, dim=-1).gather(
                                -1, tgt_arg_win.unsqueeze(-1)).squeeze(-1)
                            tau_ref = P_ref.cumprod(-1).sum()
                    else:
                        hit_curr = (S_curr == tgt_arg_win).float()
                        tau_curr = hit_curr.cumprod(-1).sum()
                        with torch.no_grad():
                            S_ref = ref_lg_win.argmax(-1)
                            hit_ref = (S_ref == tgt_arg_win).float()
                            tau_ref = hit_ref.cumprod(-1).sum()

                    rewards.append(tau_curr - tau_ref)

                    lp_draft = F.log_softmax(draft_lg_win, -1).gather(
                        -1, S_curr.unsqueeze(-1)).squeeze(-1).sum() / gamma
                    with torch.no_grad():
                        lp_ref = F.log_softmax(ref_lg_win, -1).gather(
                            -1, S_curr.unsqueeze(-1)).squeeze(-1).sum() / gamma
                    log_ratios.append(lp_draft - lp_ref)

                rewards_t = torch.stack(rewards)
                advantages = (rewards_t - rewards_t.mean()) \
                    / (rewards_t.std() + 1e-6)
                advantages = advantages.clamp(-5.0, 5.0).detach()

                log_ratios_t = torch.stack(log_ratios).clamp(-10.0, 10.0)
                ratio = log_ratios_t.exp().clamp(0.1, 10.0)
                unclipped = ratio * advantages
                clipped = ratio.clamp(1 - eps, 1 + eps) * advantages
                grpo_losses.append(-torch.min(unclipped, clipped).mean())

        if grpo_losses:
            return torch.stack(grpo_losses).mean()
        return torch.tensor(0.0, device=device)

    def _compute_grpo_sample_loss(self, input_ids, attention_mask, mask,
                                    draft_lg, target_argmax, target_lg,
                                    K_groups, m_win, eps, sample_temp,
                                    reward_type='hard'):
        """GRPO with multinomial sampling at each γ-step (shared anchor prefix).

        Group structure:
            K_groups anchors per sequence (non-overlapping γ-token spans).
            For each anchor, m_win independent samples drawn pointwise from
            draft's prob distribution at each of γ positions [anchor:anchor+γ].
            All m_win traces share the same prefix (input is teacher-forced
            on target greedy up to anchor); they differ only in the sampled
            tokens at anchor..anchor+γ-1.

        Reward (debiased):
            R_i = τ_current_i − τ_ref          (τ_ref = ref greedy on same window)
        Advantage:
            A_i = (R_i − mean(R)) / (std(R) + δ)
        Ratio (per-token geometric mean log-prob on the sampled trace S_i):
            ρ_i = exp((log π(S_i) − log π_ref(S_i)) / γ)
        """
        device = input_ids.device
        B = input_ids.shape[0]
        gamma = self.gamma

        with torch.no_grad():
            ref_lg_full = self.ref_draft_model(
                input_ids=input_ids,
                attention_mask=attention_mask).logits[:, :-1, :].float()

        grpo_losses = []
        for b in range(B):
            mask_b = mask[b].int()
            Lp = mask_b.shape[0]
            if Lp < gamma:
                continue
            cum = torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                             torch.cumsum(mask_b, 0)])
            valid = (cum[gamma:] - cum[:Lp - gamma + 1]) == gamma
            valid_starts = valid.nonzero(as_tuple=True)[0].tolist()
            if not valid_starts:
                continue

            random.shuffle(valid_starts)
            selected = []
            for s in valid_starts:
                if all(abs(s - u) >= gamma for u in selected):
                    selected.append(s)
                    if len(selected) == K_groups:
                        break
            if not selected:
                continue

            for anchor in selected:
                draft_lg_win = draft_lg[b, anchor:anchor + gamma]      # [γ, V]
                ref_lg_win = ref_lg_full[b, anchor:anchor + gamma]     # [γ, V]
                tgt_arg_win = target_argmax[b, anchor:anchor + gamma]  # [γ]
                # target_lg_win used by reward_type='hard_dist' (R_dist).
                target_lg_win = target_lg[b, anchor:anchor + gamma]    # [γ, V]

                lp_draft_full = F.log_softmax(draft_lg_win, dim=-1)    # [γ, V]
                with torch.no_grad():
                    lp_ref_full = F.log_softmax(ref_lg_win, dim=-1)
                    sample_probs = F.softmax(
                        draft_lg_win.detach() / sample_temp, dim=-1)
                    # Reference reward (shared across m_win samples)
                    if reward_type == 'eal':
                        P_ref = F.softmax(ref_lg_win, dim=-1).gather(
                            -1, tgt_arg_win.unsqueeze(-1)).squeeze(-1)
                        tau_ref = P_ref.cumprod(-1).sum()
                    else:
                        S_ref = ref_lg_win.argmax(-1)
                        hit_ref = (S_ref == tgt_arg_win).float()
                        tau_ref = hit_ref.cumprod(-1).sum()
                    # Distribution-Based Proximity Reward precompute (paper §4.3.2).
                    # logp_target(y_t | x, y_{<t}) — argmax token's target log-prob.
                    # APPROXIMATION: also use teacher-forced target_lg for
                    # logp_target(ŷ_t | x, ŷ_{<t}); paper conditions on ŷ_{<t} but
                    # re-running target per sample is ~64× cost. The approximation
                    # preserves "Δ small ⇔ draft tokens have high target log-prob"
                    # which is the signal R_dist aims to capture.
                    if reward_type == 'hard_dist':
                        tgt_logp_win = F.log_softmax(target_lg_win, dim=-1)
                        logp_y = tgt_logp_win.gather(
                            -1, tgt_arg_win.unsqueeze(-1)).squeeze(-1)  # [γ]

                rewards = []
                log_ratios = []
                for _ in range(m_win):
                    with torch.no_grad():
                        S_curr = torch.multinomial(
                            sample_probs, num_samples=1).squeeze(-1)   # [γ]
                    if reward_type == 'eal':
                        # EAL on the SAMPLED trace: cumprod(P[target]) along sample
                        # path (P[target] is the draft prob at each step regardless
                        # of what was sampled, so this collapses to deterministic EAL
                        # — keeps semantic but reward is identical across m_win samples).
                        P_curr = F.softmax(draft_lg_win, dim=-1).gather(
                            -1, tgt_arg_win.unsqueeze(-1)).squeeze(-1)
                        tau_curr = P_curr.cumprod(-1).sum()
                        R = tau_curr - tau_ref
                    else:
                        hit_curr = (S_curr == tgt_arg_win).float()
                        tau_curr = hit_curr.cumprod(-1).sum()
                        R = tau_curr - tau_ref
                        # R_dist: active only when tau_curr == 0 (k=0 in paper).
                        if reward_type == 'hard_dist' and tau_curr.item() == 0.0:
                            with torch.no_grad():
                                logp_yhat = tgt_logp_win.gather(
                                    -1, S_curr.unsqueeze(-1)).squeeze(-1)
                                delta = (logp_y - logp_yhat).sum()
                                r_dist = (self.grpo_reward_eta
                                          * (delta < self.grpo_reward_eps).float())
                            R = R + r_dist
                    rewards.append(R)

                    lp_draft = lp_draft_full.gather(
                        -1, S_curr.unsqueeze(-1)).squeeze(-1).sum() / gamma
                    with torch.no_grad():
                        lp_ref = lp_ref_full.gather(
                            -1, S_curr.unsqueeze(-1)).squeeze(-1).sum() / gamma
                    log_ratios.append(lp_draft - lp_ref)

                rewards_t = torch.stack(rewards)
                advantages = (rewards_t - rewards_t.mean()) \
                    / (rewards_t.std() + 1e-6)
                advantages = advantages.clamp(-5.0, 5.0).detach()

                log_ratios_t = torch.stack(log_ratios).clamp(-10.0, 10.0)
                ratio = log_ratios_t.exp().clamp(0.1, 10.0)
                unclipped = ratio * advantages
                clipped = ratio.clamp(1 - eps, 1 + eps) * advantages
                grpo_losses.append(-torch.min(unclipped, clipped).mean())

        if grpo_losses:
            return torch.stack(grpo_losses).mean()
        return torch.tensor(0.0, device=device)
