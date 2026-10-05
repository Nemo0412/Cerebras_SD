"""
Direct S_θ optimization model: maximize E[τ] without RL.

Teacher-forced on target greedy tokens, no rollouts, no advantage.
L_total = L_direct_S + eagle_coef * L_EAGLE + kl_coef * L_KL

The key difference from L_EAGLE:
- L_EAGLE: per-position KL(p_target || q_draft), equal weight across positions
- L_direct_S: cumulative product objective, autograd naturally gives w_j = Σ_{t≥j} S_t,
  so early positions get stronger gradient (they appear in more S_t terms)

The key difference from SDPO (RL):
- No sampling, no rollouts, no advantage normalization
- Deterministic: single teacher-forced pass
- Same position-aware gradient weighting, but no variance from rollout sampling
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'traineagle3'))
from traineagle3.cnets import Model


class DirectSModel(Model):
    """Direct S_θ maximization without RL. forward() routes to direct_s_forward()."""

    def __init__(self, config, ds_config, training_config, path,
                 load_emb=True, load_head=True, ref_model=None):
        super().__init__(config, ds_config, training_config,
                         load_emb=load_emb, load_head=load_head, path=path)
        self.eagle_config = config
        self.ref_model = ref_model
        if ref_model is not None:
            for param in ref_model.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, loss_mask,
                eagle_coef: float = 0.003, kl_coef: float = 0.1,
                eps: float = 1e-8, baseline: str = None):
        return self.direct_s_forward(input_ids, attention_mask, loss_mask,
                                     eagle_coef=eagle_coef, kl_coef=kl_coef,
                                     eps=eps, baseline=baseline)

    def direct_s_forward(self, input_ids, attention_mask, loss_mask,
                         eagle_coef: float = 0.003, kl_coef: float = 0.1,
                         eps: float = 1e-8, baseline: str = None):
        """
        One training step: direct S_θ maximization.
        Returns (total_loss, direct_s_loss, kl_loss, metrics).
        """
        hidden_states, target, loss_mask_3d, input_ids_shifted = self.dataprepare(
            input_ids, attention_mask, loss_mask
        )
        loss_mask_2d = loss_mask_3d.squeeze(-1).bool()
        batch_size, seq_length, _ = hidden_states.shape
        device = hidden_states.device
        gamma = self.length

        hidden_states = hidden_states.to(self.fc.weight.dtype)
        hs_projected = self.fc(hidden_states)

        if self.training and self.gradient_checkpointing and not hs_projected.requires_grad:
            hs_projected.requires_grad_(True)

        attn_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hs_projected, 0
        )
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        with torch.no_grad():
            target_greedy = target.argmax(dim=-1)

        # Eagle-only baseline: skip S optimization, only L_EAGLE.
        if baseline == 'eagle_only':
            eagle_loss = self._compute_eagle_loss(
                hs_projected, input_ids_shifted, target,
                attn_mask, position_ids, loss_mask_2d, eps=eps,
            )
            num_valid = loss_mask_2d.float().sum()
            metrics = {
                "direct_s_loss": 0.0,
                "eagle_loss": eagle_loss.item(),
                "kl_loss": 0.0,
                "mean_S": 0.0,
                "mean_tau": 0.0,
                "step_acc": [0.0] * gamma,
                "num_valid": num_valid.item(),
            }
            zero = torch.tensor(0.0, device=device)
            return eagle_loss, zero, zero, metrics

        # --- Direct S_θ: teacher-forced on target greedy ---
        t2d = self.t2d.to(device)
        d2t = self.d2t.to(device)

        # Build correct full_vocab → draft_vocab reverse mapping.
        # d2t[draft_id] = offset, full_id = draft_id + d2t[draft_id].
        # Reverse: for each draft_id, compute full_id, then reverse_map[full_id] = draft_id.
        draft_ids = torch.arange(len(d2t), device=device)
        full_ids = draft_ids + d2t
        full2draft = torch.full((t2d.shape[0],), -1, dtype=torch.long, device=device)
        full2draft[full_ids] = draft_ids

        # --- Debug: verify d2t/t2d consistency (only first call) ---
        if not hasattr(self, '_debug_checked'):
            self._debug_checked = True
            # full2draft should agree with t2d
            in_vocab_via_full2draft = (full2draft >= 0)
            mismatch = (in_vocab_via_full2draft != t2d).sum().item()
            print(f"[DirectS DEBUG] full2draft vs t2d mismatch: {mismatch} / {t2d.shape[0]}")
            print(f"[DirectS DEBUG] draft_vocab_size={len(d2t)}, t2d.sum()={t2d.sum().item()}")
            # Spot-check: pick 5 random in-vocab tokens, verify round-trip
            sample_draft_ids = torch.randint(0, len(d2t), (5,), device=device)
            for di in sample_draft_ids:
                fi = di + d2t[di]
                di_back = full2draft[fi]
                print(f"  draft_id={di.item()} → full_id={fi.item()} → back={di_back.item()} "
                      f"(match={di.item() == di_back.item()})")

        all_logprobs = []  # HAS grad, for S_θ
        all_accepts = []  # bool, for valid mask (detached)
        all_draft_tokens_d = []  # for KL
        _debug_step_logprobs = []  # for first-batch diagnostics

        cache_hidden = [[], []]
        current_input_ids = input_ids_shifted
        current_hs = hs_projected
        current_tgt_greedy = target_greedy.clone()

        for idx in range(gamma):
            last = (idx == gamma - 1)

            inputs_embeds = self.embed_tokens(current_input_ids)
            if self.training and self.gradient_checkpointing \
                    and not inputs_embeds.requires_grad:
                inputs_embeds.requires_grad_(True)
            inputs_embeds = inputs_embeds.to(current_hs.dtype)

            if self.gradient_checkpointing and self.training:
                def _make_fwd(module):
                    def _fwd(*inputs):
                        return module(*inputs, None, False)
                    return _fwd
                layer_outputs, cache_hidden = \
                    torch.utils.checkpoint.checkpoint(
                        _make_fwd(self.midlayer),
                        inputs_embeds, current_hs, cache_hidden, attn_mask, position_ids,
                    )
            else:
                layer_outputs, cache_hidden = self.midlayer(
                    input_emb=inputs_embeds,
                    hidden_states=current_hs,
                    cache_hidden=cache_hidden,
                    attention_mask=attn_mask,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=True,
                )

            hs_out = layer_outputs[0]
            current_hs = hs_out
            logits = self.lm_head(self.norm(hs_out)).float()

            with torch.no_grad():
                # Draft greedy token (for acceptance check)
                draft_greedy_d = logits.detach().argmax(dim=-1)
                if self.eagle_config.vocab_size == self.eagle_config.draft_vocab_size:
                    draft_greedy_full = draft_greedy_d
                else:
                    draft_greedy_full = draft_greedy_d + d2t[draft_greedy_d]

                tgt_in_dv = t2d[current_tgt_greedy]
                accept = (draft_greedy_full == current_tgt_greedy) & tgt_in_dv

                # Map target greedy (full vocab) to draft vocab index
                tgt_draft_idx = full2draft[current_tgt_greedy]
                # Clamp -1 (OOV) to 0 for safe gather; OOV logprobs get zeroed below
                tgt_draft_idx = tgt_draft_idx.clamp(min=0)

                all_accepts.append(accept)
                all_draft_tokens_d.append(tgt_draft_idx)

            # Log prob of target greedy token under draft model (WITH grad)
            log_softmax_out = F.log_softmax(logits, dim=-1)
            logprob_tgt = log_softmax_out \
                            .gather(-1, tgt_draft_idx.detach().unsqueeze(-1)).squeeze(-1)
            # OOV positions: set logprob to -20 (≈ prob 2e-9) so cumulative product
            # naturally decays through OOV steps. Using 0 (=log(1)) would be wrong
            # as it implies probability 1 at OOV positions.
            logprob_tgt = torch.where(tgt_in_dv.detach(), logprob_tgt,
                                      torch.full_like(logprob_tgt, -20.0))
            all_logprobs.append(logprob_tgt)

            # --- Debug: per-step diagnostics (detached, on valid positions only) ---
            with torch.no_grad():
                valid_mask = loss_mask_2d & tgt_in_dv
                if valid_mask.any():
                    prob_tgt = logprob_tgt[valid_mask].exp().mean().item()
                    acc_rate = accept[valid_mask].float().mean().item()
                    # Verify: draft argmax probability vs target token probability
                    draft_argmax_prob = log_softmax_out.max(dim=-1).values[valid_mask].exp().mean().item()
                    _debug_step_logprobs.append({
                        'step': idx, 'q_target': prob_tgt, 'accept_rate': acc_rate,
                        'q_argmax': draft_argmax_prob,
                    })

            if not last:
                # Teacher-forced: next step always uses target greedy token
                current_input_ids = current_tgt_greedy.detach()
                current_tgt_greedy = torch.cat(
                    [current_tgt_greedy[:, 1:],
                     torch.zeros_like(current_tgt_greedy[:, :1])],
                    dim=1
                )

        logprobs = torch.stack(all_logprobs, dim=2)  # [B, L, gamma] HAS grad
        accepts = torch.stack(all_accepts, dim=2).detach()  # [B, L, gamma]
        draft_tokens_d = torch.stack(all_draft_tokens_d, dim=2).detach()

        # S_θ with grad: exp(cumsum(logprobs)), NO valid mask.
        # Teacher-forced: all steps have correct context, so every S_t is meaningful.
        # OOV positions have logprob=-20, making exp(cumsum) ≈ 0 after OOV naturally.
        S_t_theta = torch.exp(torch.cumsum(logprobs, dim=2))
        S_theta = S_t_theta.sum(dim=2) / gamma  # [B, L]

        # valid mask is only for MONITORING mean_tau, not for the loss
        valid = torch.cumprod(accepts.float(), dim=2).detach()

        num_valid = loss_mask_2d.float().sum()
        direct_s_loss = -(S_theta * loss_mask_2d.float()).sum() / (num_valid + eps)

        # --- Debug: first-batch summary (printed once per step, only rank 0) ---
        if not hasattr(self, '_debug_step_count'):
            self._debug_step_count = 0
        self._debug_step_count += 1
        if self._debug_step_count <= 3:  # first 3 batches only
            with torch.no_grad():
                print(f"\n[DirectS DEBUG batch {self._debug_step_count}]")
                print(f"  S_theta: mean={S_theta[loss_mask_2d].mean().item():.4f}, "
                      f"min={S_theta[loss_mask_2d].min().item():.4f}, "
                      f"max={S_theta[loss_mask_2d].max().item():.4f}")
                print(f"  mean_tau={valid.sum(dim=2)[loss_mask_2d].mean().item():.4f}")
                print(f"  num_valid={num_valid.item():.0f}, "
                      f"OOV_frac={(~accepts[:,:,0][loss_mask_2d] & ~self.t2d.to(device)[target_greedy][loss_mask_2d]).float().mean().item():.4f}")
                for d in _debug_step_logprobs:
                    print(f"  step {d['step']}: q(target)={d['q_target']:.4f}, "
                          f"accept_rate={d['accept_rate']:.4f}, "
                          f"q(argmax)={d['q_argmax']:.4f}")
                # Check logprobs are reasonable
                lp_valid = logprobs[:,:,0][loss_mask_2d]
                print(f"  logprobs step0: mean={lp_valid.mean().item():.4f}, "
                      f"min={lp_valid.min().item():.4f}, max={lp_valid.max().item():.4f}")
                if logprobs.requires_grad:
                    print(f"  logprobs has grad: True")
                else:
                    print(f"  WARNING: logprobs has NO grad!")

        # L_KL against frozen ref model
        kl_loss = torch.tensor(0.0, device=device)
        if self.ref_model is not None and kl_coef > 0:
            kl_loss = self._compute_kl_direct(
                hidden_states, input_ids_shifted, target_greedy,
                attn_mask, position_ids,
                logprobs, draft_tokens_d,
                loss_mask_2d, eps=eps,
            )

        # L_EAGLE
        eagle_loss = self._compute_eagle_loss(
            hs_projected, input_ids_shifted, target,
            attn_mask, position_ids, loss_mask_2d, eps=eps,
        )

        total_loss = direct_s_loss + eagle_coef * eagle_loss + kl_coef * kl_loss

        # Metrics
        with torch.no_grad():
            mean_tau = (valid.sum(dim=2) * loss_mask_2d.float()).sum() / (num_valid + eps)
            mean_S = S_theta[loss_mask_2d].mean()
            step_acc = [
                accepts[:, :, k][loss_mask_2d].float().mean().item()
                for k in range(gamma)
            ]

        metrics = {
            "direct_s_loss": direct_s_loss.item(),
            "eagle_loss": eagle_loss.item(),
            "kl_loss": kl_loss.item(),
            "mean_S": mean_S.item(),
            "mean_tau": mean_tau.item(),
            "step_acc": step_acc,
            "num_valid": num_valid.item(),
        }

        return total_loss, direct_s_loss, kl_loss, metrics

    def _compute_kl_direct(
        self,
        hidden_states_raw: torch.Tensor,
        input_ids: torch.Tensor,
        target_greedy: torch.Tensor,
        attn_mask: torch.Tensor,
        position_ids: torch.Tensor,
        logprobs_theta: torch.Tensor,
        draft_tokens_d: torch.Tensor,
        loss_mask_2d: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """KL regularizer for direct S mode: single-token log ratio on target greedy tokens."""
        with torch.no_grad():
            ref_logprobs = self._ref_model_logprobs_direct(
                hidden_states_raw, input_ids, target_greedy,
                attn_mask, position_ids, draft_tokens_d,
            )
        if ref_logprobs is None:
            return torch.tensor(0.0, device=hidden_states_raw.device)

        mask = loss_mask_2d.float().unsqueeze(2).expand_as(logprobs_theta)
        kl = ((logprobs_theta - ref_logprobs) * mask).sum()
        count = mask.sum()
        return kl / (count + eps)

    @torch.no_grad()
    def _ref_model_logprobs_direct(
        self,
        hidden_states_raw: torch.Tensor,
        input_ids: torch.Tensor,
        target_greedy: torch.Tensor,
        attn_mask: torch.Tensor,
        position_ids: torch.Tensor,
        draft_tokens_d: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Get ref model log probs for target greedy tokens (teacher-forced)."""
        if self.ref_model is None:
            return None

        gamma = self.length
        device = hidden_states_raw.device
        ref_logprobs = []
        cache_hidden = [[], []]
        current_input_ids = input_ids
        current_tgt_greedy = target_greedy.clone()

        _ref_d2t_src = self.ref_model.d2t if hasattr(self.ref_model, "d2t") else self.d2t
        ref_d2t = _ref_d2t_src.to(device)
        t2d = self.t2d.to(device)
        # Build correct full→draft reverse mapping
        draft_ids = torch.arange(len(ref_d2t), device=device)
        full_ids = draft_ids + ref_d2t
        full2draft = torch.full((t2d.shape[0],), -1, dtype=torch.long, device=device)
        full2draft[full_ids] = draft_ids

        current_hs = self.ref_model.fc(hidden_states_raw.to(self.ref_model.fc.weight.dtype))

        for idx in range(gamma):
            last = (idx == gamma - 1)

            inputs_embeds = self.ref_model.embed_tokens(current_input_ids).to(current_hs.dtype)
            layer_outputs, cache_hidden = self.ref_model.midlayer(
                input_emb=inputs_embeds,
                hidden_states=current_hs,
                cache_hidden=cache_hidden,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=True,
            )

            hs_out = layer_outputs[0]
            current_hs = hs_out
            logits = self.ref_model.lm_head(self.ref_model.norm(hs_out)).float()

            tgt_draft_idx = full2draft[current_tgt_greedy].clamp(min=0)
            lp = F.log_softmax(logits, dim=-1) \
                   .gather(-1, tgt_draft_idx.unsqueeze(-1)).squeeze(-1)
            tgt_in_dv = t2d[current_tgt_greedy]
            lp = torch.where(tgt_in_dv, lp, torch.full_like(lp, -20.0))
            ref_logprobs.append(lp)

            if not last:
                current_input_ids = current_tgt_greedy
                current_tgt_greedy = torch.cat(
                    [current_tgt_greedy[:, 1:],
                     torch.zeros_like(current_tgt_greedy[:, :1])],
                    dim=1
                )

        return torch.stack(ref_logprobs, dim=2)

    # _compute_eagle_loss is inherited from SDPOModel's parent
    # But we need our own copy since we inherit from Model directly.
    def _compute_eagle_loss(
        self,
        hs_projected: torch.Tensor,
        input_ids_shifted: torch.Tensor,
        target_logits: torch.Tensor,
        attn_mask: torch.Tensor,
        position_ids: torch.Tensor,
        loss_mask_2d: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Soft-target KL loss matching EAGLE3 training (cnets.py:790-868)."""
        device = hs_projected.device
        gamma = self.length
        t2d_dev = self.t2d.to(device)

        losses = []
        cache_hidden = [[], []]
        current_hidden = hs_projected
        current_input_ids = input_ids_shifted
        current_target = target_logits
        current_loss_mask = loss_mask_2d.float().unsqueeze(-1)

        for idx in range(gamma):
            last = (idx == gamma - 1)

            inputs_embeds = self.embed_tokens(current_input_ids).to(current_hidden.dtype)
            if self.gradient_checkpointing and self.training:
                if not inputs_embeds.requires_grad:
                    inputs_embeds.requires_grad_(True)
                def _make_fwd(module):
                    def _fwd(*inputs):
                        return module(*inputs, None, False)
                    return _fwd
                layer_out, cache_hidden = torch.utils.checkpoint.checkpoint(
                    _make_fwd(self.midlayer),
                    inputs_embeds, current_hidden, cache_hidden, attn_mask, position_ids,
                )
            else:
                layer_out, cache_hidden = self.midlayer(
                    input_emb=inputs_embeds,
                    hidden_states=current_hidden,
                    cache_hidden=cache_hidden,
                    attention_mask=attn_mask,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=True,
                )

            hs_out = layer_out[0]
            current_hidden = hs_out

            eagle_logits = self.lm_head(self.norm(hs_out)).float()

            with torch.no_grad():
                target_max_token = current_target.argmax(dim=-1)
                target_mask = t2d_dev[target_max_token].unsqueeze(-1).float()
                position_mask = target_mask * current_loss_mask
                target_draft_logits = current_target[..., t2d_dev].float()
                target_p = torch.softmax(target_draft_logits, dim=-1).detach()

            out_logp = torch.log_softmax(eagle_logits, dim=-1)
            plogp = target_p * out_logp
            loss = -torch.sum(position_mask * plogp, dim=-1).mean()
            losses.append(loss)

            if not last:
                with torch.no_grad():
                    current_input_ids = torch.cat(
                        [current_input_ids[:, 1:],
                         torch.zeros_like(current_input_ids[:, :1])], dim=1)
                    current_target = torch.cat(
                        [current_target[:, 1:],
                         torch.zeros_like(current_target[:, :1])], dim=1)
                    current_loss_mask = torch.cat(
                        [current_loss_mask[:, 1:],
                         torch.zeros_like(current_loss_mask[:, :1])], dim=1)

        return sum(losses)
