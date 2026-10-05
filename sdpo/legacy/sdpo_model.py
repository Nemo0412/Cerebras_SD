"""
SDPO model: extends EAGLE3 training model with on-policy RL that directly maximizes E[tau].

Loss = L_SDPO + eagle_coef * L_EAGLE + kl_coef * L_KL
Advantage is normalized within each position's G rollouts — never across positions,
because context difficulty varies per position.
"""

import sys
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

# Must import from traineagle3/cnets.py (the TRAINING model), NOT model/cnets.py.
# traineagle3 Model has: self.target_model, dataprepare(), the 7-step training loop.
# model/cnets.py is the INFERENCE model (topK_genrate, no target model inside).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'traineagle3'))
from traineagle3.cnets import Model


class SDPOModel(Model):
    """Wraps EAGLE3 Model with SDPO training. forward() routes to sdpo_forward()."""

    def __init__(self, config, ds_config, training_config, path,
                 load_emb=True, load_head=True, ref_model=None):
        super().__init__(config, ds_config, training_config,
                         load_emb=load_emb, load_head=load_head, path=path)
        self.eagle_config = config   # cnets.Model doesn't store config; needed for vocab checks
        self.ref_model = ref_model
        if ref_model is not None:
            for param in ref_model.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, loss_mask,
                G: int = 4, eagle_coef: float = 0.05, kl_coef: float = 0.1,
                temperature: float = 1.0, eps: float = 1e-8,
                mode: str = 'eager', baseline: str = None):
        return self.sdpo_forward(input_ids, attention_mask, loss_mask,
                                 G=G, eagle_coef=eagle_coef, kl_coef=kl_coef,
                                 temperature=temperature, eps=eps, mode=mode,
                                 baseline=baseline)

    def sdpo_forward(self, input_ids, attention_mask, loss_mask,
                     G: int = 4, eagle_coef: float = 0.05, kl_coef: float = 0.1,
                     temperature: float = 1.0, eps: float = 1e-8,
                     mode: str = 'eager', baseline: str = None):
        """
        One SDPO training step. Returns (total_loss, sdpo_loss, kl_loss, metrics).

        temperature=0 collapses all G rollouts to the same greedy trajectory, giving
        zero advantage variance and no gradient signal. Keep temperature > 0.
        """
        # dataprepare() runs frozen target model + left-shift.
        # At draft step k from position p, the draft predicts token p+k+2.
        hidden_states, target, loss_mask_3d, input_ids_shifted = self.dataprepare(
            input_ids, attention_mask, loss_mask
        )
        loss_mask_2d = loss_mask_3d.squeeze(-1).bool()   # [B, L]
        batch_size, seq_length, _ = hidden_states.shape
        device = hidden_states.device
        gamma = self.length

        # Target may run fp32; draft fc.weight is fp16 — align before projection.
        hidden_states = hidden_states.to(self.fc.weight.dtype)
        hs_projected = self.fc(hidden_states)            # [B, L, H], shared across rollouts

        if self.training and self.gradient_checkpointing and not hs_projected.requires_grad:
            hs_projected.requires_grad_(True)

        attn_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hs_projected, 0
        )
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)  # [B, L]

        with torch.no_grad():
            target_greedy = target.argmax(dim=-1)        # [B, L], full vocab

        # Eagle-only baseline: skip rollouts, only L_EAGLE supervised loss.
        if baseline == 'eagle_only':
            eagle_loss = self._compute_eagle_loss(
                hs_projected, input_ids_shifted, target,
                attn_mask, position_ids, loss_mask_2d, eps=eps,
            )
            num_valid = loss_mask_2d.float().sum()
            metrics = {
                "sdpo_loss": 0.0,
                "eagle_loss": eagle_loss.item(),
                "kl_loss": 0.0,
                "mean_S": 0.0,
                "mean_tau": 0.0,
                "step_acc": [0.0] * gamma,
                "num_valid": num_valid.item(),
                "adv_degenerate_frac": 0.0,
            }
            zero = torch.tensor(0.0, device=device)
            return eagle_loss, zero, zero, metrics

        all_S = []   # [B,L] rollout reward, for S-fallback advantage
        all_tau = []   # [B,L] acceptance lengths, primary advantage signal
        all_logprobs = []   # [B,L,gamma] WITH grad, for differentiable S_theta
        all_accepts = []   # [B,L,gamma] bool
        all_draft_tokens = []   # [B,L,gamma] draft vocab ids, for KL

        for g in range(G):
            S_g, tau_g, logprobs_g, accepts_g, draft_tokens_g = self._single_rollout(
                hs_projected, input_ids_shifted, target_greedy,
                attn_mask, position_ids, temperature=temperature,
                batch_size=batch_size, seq_length=seq_length,
            )
            all_S.append(S_g)
            all_tau.append(tau_g)
            all_logprobs.append(logprobs_g)
            all_accepts.append(accepts_g)
            all_draft_tokens.append(draft_tokens_g)

        # Advantage — mode-conditional.
        # eager: greedy chain is fixed, so S and tau are strictly monotone.
        #        S-advantage is unbiased; tau-advantage adds nothing.
        # sampling: different tokens can be accepted per rollout, breaking monotonicity.
        #           tau-advantage eliminates the confidence-vs-length conflation bias;
        #           S-advantage is the fallback when std(tau)=0.
        # Never mix positions — context difficulty varies per position.
        S_stack = torch.stack(all_S, dim=0).detach()  # [G, B, L]

        with torch.no_grad():
            if mode == 'eager':
                mean_S = S_stack.mean(dim=0, keepdim=True)
                std_S = S_stack.std(dim=0, keepdim=True)
                A = (S_stack - mean_S) / (std_S + eps)
                adv_degenerate = std_S.squeeze(0) < eps
            else:  # sampling
                tau_stack = torch.stack(all_tau, dim=0).float().detach()  # [G, B, L]
                tau_norm = tau_stack / gamma
                mean_tau_adv = tau_norm.mean(dim=0, keepdim=True)
                std_tau = tau_norm.std(dim=0, keepdim=True)
                A_tau = (tau_norm - mean_tau_adv) / (std_tau + eps)

                mean_S = S_stack.mean(dim=0, keepdim=True)
                std_S = S_stack.std(dim=0, keepdim=True)
                A_S = (S_stack - mean_S) / (std_S + eps)

                tau_degenerate = std_tau.squeeze(0) < eps
                adv_degenerate = tau_degenerate & (std_S.squeeze(0) < eps)
                A = torch.where(tau_degenerate.unsqueeze(0), A_S, A_tau)

            A = A * loss_mask_2d.float().unsqueeze(0)
            A[:, adv_degenerate] = 0.0

        # L_SDPO: gradient flows through exp(cumsum(logprobs)), so w_j (suffix-sum weights)
        # emerge from autograd — do NOT manually compute or detach w_j.
        num_valid = loss_mask_2d.float().sum()
        sdpo_loss = torch.tensor(0.0, device=device)

        for g in range(G):
            A_g = A[g]  # [B, L], constant
            lp_g = all_logprobs[g]  # [B, L, gamma], HAS grad
            acc_g = all_accepts[g]  # [B, L, gamma], bool, no grad

            valid_g = torch.cumprod(acc_g.float(), dim=2)
            # exp(cumsum(logprobs)) is more stable than cumprod(probs)
            S_t_theta = torch.exp(torch.cumsum(lp_g, dim=2)) * valid_g
            S_theta = S_t_theta.sum(dim=2) / gamma
            loss_g = -(A_g * S_theta * loss_mask_2d.float()).sum() / (num_valid + eps)
            sdpo_loss = sdpo_loss + loss_g

        sdpo_loss = sdpo_loss / G

        # KL against frozen initial draft checkpoint, not the target model.
        kl_loss = torch.tensor(0.0, device=device)

        if self.ref_model is not None:
            kl_loss = self._compute_kl_loss(
                hidden_states, input_ids_shifted,
                attn_mask, position_ids,
                all_logprobs, all_accepts, all_draft_tokens,
                loss_mask_2d, eps=eps,
            )

        # L_EAGLE: dense teacher-forced KL loss against target model's distribution.
        eagle_loss = self._compute_eagle_loss(
            hs_projected, input_ids_shifted, target,
            attn_mask, position_ids, loss_mask_2d, eps=eps,
        )

        total_loss = sdpo_loss + eagle_coef * eagle_loss + kl_coef * kl_loss

        with torch.no_grad():
            accepts_stack = torch.stack(all_accepts, dim=0).float()
            valid_stack = torch.cumprod(accepts_stack, dim=3)
            mean_tau = (valid_stack.sum(dim=3) *
                        loss_mask_2d.float().unsqueeze(0)).sum() / (G * num_valid + eps)
            mean_S_scalar = S_stack[:, loss_mask_2d].mean()
            step_acc = [
                accepts_stack[:, :, :, k][:, loss_mask_2d].mean().item()
                for k in range(gamma)
            ]

        with torch.no_grad():
            tau_deg_frac = (adv_degenerate & loss_mask_2d).float().sum() / (num_valid + eps)

        metrics = {
            "sdpo_loss": sdpo_loss.item(),
            "eagle_loss": eagle_loss.item(),
            "kl_loss": kl_loss.item(),
            "mean_S": mean_S_scalar.item(),
            "mean_tau": mean_tau.item(),
            "step_acc": step_acc,
            "num_valid": num_valid.item(),
            "adv_degenerate_frac": tau_deg_frac.item(),
        }

        return total_loss, sdpo_loss, kl_loss, metrics

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
        """Soft-target KL loss matching EAGLE3 training (cnets.py:790-868).
        Unrolls all gamma steps with teacher-forced input, shifting target/input_ids/
        loss_mask left at each step, accumulating cache_hidden across steps."""
        device = hs_projected.device
        gamma = self.length
        t2d_dev = self.t2d.to(device)

        losses = []
        cache_hidden = [[], []]
        current_hidden = hs_projected
        current_input_ids = input_ids_shifted
        current_target = target_logits
        current_loss_mask = loss_mask_2d.float().unsqueeze(-1)  # [B, L, 1]

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

            hs_out = layer_out[0]  # [B, L, H]
            current_hidden = hs_out

            eagle_logits = self.lm_head(self.norm(hs_out)).float()  # [B, L, draft_V]

            with torch.no_grad():
                target_max_token = current_target.argmax(dim=-1)
                target_mask = t2d_dev[target_max_token].unsqueeze(-1).float()  # [B, L, 1]
                position_mask = target_mask * current_loss_mask  # [B, L, 1]
                target_draft_logits = current_target[..., t2d_dev].float()  # [B, L, draft_V]
                target_p = torch.softmax(target_draft_logits, dim=-1).detach()

            out_logp = torch.log_softmax(eagle_logits, dim=-1)
            plogp = target_p * out_logp
            loss = -torch.sum(position_mask * plogp, dim=-1).mean()
            losses.append(loss)

            if not last:
                with torch.no_grad():
                    # Shift left by 1 — matches padding(tensor, left=False) in cnets.py
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

    def _single_rollout(
        self,
        hs_projected: torch.Tensor,
        input_ids: torch.Tensor,
        target_greedy: torch.Tensor,
        attn_mask: torch.Tensor,
        position_ids: torch.Tensor,
        temperature: float,
        batch_size: int,
        seq_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One on-policy draft trajectory of gamma steps.

        Reward probabilities are always under the unscaled policy (temperature=1),
        so S measures true model confidence regardless of the sampling temperature.
        Temperature only determines which token gets selected, not its reward value.

        Returns: S [B,L], tau [B,L], logprobs [B,L,gamma] WITH grad, accepts [B,L,gamma], draft_tokens_d [B,L,gamma]
        """
        gamma = self.length
        device = hs_projected.device
        t2d = self.t2d.to(device)  # [V] bool: which full-vocab tokens are in draft vocab
        d2t = self.d2t.to(device)  # [Vd] int: draft token → full vocab offset

        all_probs = []   # no grad, for S
        all_logprobs = []   # HAS grad, for S_theta in loss
        all_accepts = []   # bool
        all_draft_tokens_d = []   # draft vocab ids, for KL

        cache_hidden = [[], []]
        current_input_ids = input_ids  # [B, L]
        current_hs = hs_projected  # [B, L, H]
        current_tgt_greedy = target_greedy.clone()  # [B, L]

        for idx in range(gamma):
            last = (idx == gamma - 1)

            inputs_embeds = self.embed_tokens(current_input_ids)   # [B, L, H]
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

            hs_out = layer_outputs[0]  # [B, L, H]
            current_hs = hs_out
            logits = self.lm_head(self.norm(hs_out)).float()  # [B, L, draft_V]

            with torch.no_grad():
                if temperature <= 0.0:
                    draft_token_d = logits.detach().argmax(dim=-1)
                else:
                    probs_scaled = F.softmax(logits.detach() / temperature, dim=-1)
                    probs_flat = probs_scaled.view(-1, probs_scaled.shape[-1])
                    draft_token_d = torch.multinomial(probs_flat, num_samples=1) \
                                        .view(batch_size, seq_length)

                if self.eagle_config.vocab_size == self.eagle_config.draft_vocab_size:
                    draft_token_full = draft_token_d
                else:
                    draft_token_full = draft_token_d + d2t[draft_token_d]

                tgt_in_dv = t2d[current_tgt_greedy]
                accept = (draft_token_full == current_tgt_greedy) & tgt_in_dv

                # Reward always under unscaled policy so S reflects true confidence
                prob_sel = F.softmax(logits.detach(), dim=-1) \
                             .gather(-1, draft_token_d.unsqueeze(-1)).squeeze(-1)

                all_probs.append(prob_sel)
                all_accepts.append(accept)
                all_draft_tokens_d.append(draft_token_d)

            # Unscaled log prob for policy gradient — must match token selected above
            logprob_sel = F.log_softmax(logits, dim=-1) \
                            .gather(-1, draft_token_d.detach().unsqueeze(-1)).squeeze(-1)
            all_logprobs.append(logprob_sel)

            if not last:
                current_input_ids = draft_token_full.detach()  # on-policy feedback
                current_tgt_greedy = torch.cat(                  # shift left for next step
                    [current_tgt_greedy[:, 1:],
                     torch.zeros_like(current_tgt_greedy[:, :1])],
                    dim=1
                )

        probs = torch.stack(all_probs, dim=2).detach()
        logprobs = torch.stack(all_logprobs, dim=2)  # HAS grad
        accepts = torch.stack(all_accepts, dim=2).detach()
        draft_tokens_d = torch.stack(all_draft_tokens_d, dim=2).detach()

        with torch.no_grad():
            valid = torch.cumprod(accepts.float(), dim=2)
            tau = valid.sum(dim=2)
            S_t = torch.cumprod(probs, dim=2) * valid
            S = S_t.sum(dim=2) / gamma

        # w_j is not computed here — it emerges from autograd over S_theta in sdpo_forward.
        return S, tau, logprobs, accepts, draft_tokens_d

    def _compute_kl_loss(
        self,
        hidden_states_raw: torch.Tensor,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor,
        position_ids: torch.Tensor,
        all_logprobs: list,
        all_accepts: list,
        all_draft_tokens: list,
        loss_mask_2d: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Single-token log-ratio regularizer: mean(log q_theta(y) - log q_ref(y))
        over sampled tokens at all gamma positions per rollout.
        NOT full-distribution KL — intentionally only constrains the sampled token's
        probability, allowing the rest of the distribution to shift freely (which is
        what SDPO needs to concentrate mass on accepted tokens).
        hidden_states_raw is pre-fc so ref model applies its own fc projection."""
        kl_total = torch.tensor(0.0, device=hidden_states_raw.device)
        count = torch.tensor(0.0, device=hidden_states_raw.device)

        for g in range(len(all_logprobs)):
            logprobs_theta = all_logprobs[g]
            draft_tokens_dg = all_draft_tokens[g]

            with torch.no_grad():
                ref_logprobs_g = self._ref_model_logprobs(
                    hidden_states_raw, input_ids, attn_mask, position_ids, draft_tokens_dg,
                )

            if ref_logprobs_g is None:
                continue

            # KL over all gamma positions — rejected positions can still drift,
            # constraining them prevents instability.
            mask = loss_mask_2d.float().unsqueeze(2).expand_as(logprobs_theta)
            kl_g = ((logprobs_theta - ref_logprobs_g) * mask).sum()
            kl_total = kl_total + kl_g
            count = count + mask.sum()

        return kl_total / (count + eps)

    @torch.no_grad()
    def _ref_model_logprobs(
        self,
        hidden_states_raw: torch.Tensor,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor,
        position_ids: torch.Tensor,
        draft_tokens_d: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Get log probs from the frozen reference model for exact rollout tokens."""
        if self.ref_model is None:
            return None

        gamma = self.length
        device = hidden_states_raw.device
        ref_logprobs = []
        cache_hidden = [[], []]
        current_input_ids = input_ids
        # Fall back to self.d2t if ref_model doesn't have its own (same checkpoint)
        _ref_d2t_src = self.ref_model.d2t if hasattr(self.ref_model, "d2t") else self.d2t
        ref_d2t = _ref_d2t_src.to(device)
        # Apply ref model's own fc, not self.fc
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
            logits_ref = self.ref_model.lm_head(self.ref_model.norm(hs_out)).float()

            token_d = draft_tokens_d[:, :, idx]
            lp_ref = F.log_softmax(logits_ref, dim=-1) \
                        .gather(-1, token_d.unsqueeze(-1)).squeeze(-1)
            ref_logprobs.append(lp_ref)

            if not last:
                if self.eagle_config.vocab_size == self.eagle_config.draft_vocab_size:
                    next_ids = token_d
                else:
                    next_ids = token_d + ref_d2t[token_d]
                current_input_ids = next_ids

        return torch.stack(ref_logprobs, dim=2)   # [B, L, gamma]