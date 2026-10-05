"""
Rejection-Focused E[τ_remaining] model.

L_total = L_SDPO + eagle_coef * L_EAGLE

L_SDPO = -Σ_{t=τ+1}^{γ} w_t,  w_t = Π_{j=τ+1}^{t-1} q(y_j*)
  - τ = first rejection point (detached argmax check)
  - w_{τ+1} = 1: first rejected position always gets full gradient
  - Ratchet: fixing τ+1 → τ increases → loss shifts forward

L_EAGLE = soft-target KL on all γ positions (anchor for accepted positions)
"""

import sys
import os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'traineagle3'))
from traineagle3.cnets import Model


class RejectionModel(Model):

    def __init__(self, config, ds_config, training_config, path,
                 load_emb=True, load_head=True):
        super().__init__(config, ds_config, training_config,
                         load_emb=load_emb, load_head=load_head, path=path)
        self.eagle_config = config

    def forward(self, input_ids, attention_mask, loss_mask,
                eagle_coef=0.003, baseline=None):
        return self.rejection_forward(
            input_ids, attention_mask, loss_mask,
            eagle_coef=eagle_coef, baseline=baseline)

    def rejection_forward(self, input_ids, attention_mask, loss_mask,
                          eagle_coef=0.003, baseline=None):
        hidden_states, target, loss_mask_3d, input_ids_shifted = self.dataprepare(
            input_ids, attention_mask, loss_mask)
        loss_mask_2d = loss_mask_3d.squeeze(-1).bool()
        B, L, _ = hidden_states.shape
        device = hidden_states.device
        gamma = self.length

        hidden_states = hidden_states.to(self.fc.weight.dtype)
        hs_projected = self.fc(hidden_states)
        if self.training and self.gradient_checkpointing and not hs_projected.requires_grad:
            hs_projected.requires_grad_(True)

        attn_mask = self._prepare_decoder_attention_mask(
            attention_mask, (B, L), hs_projected, 0)
        position_ids = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            target_greedy = target.argmax(dim=-1)

        if baseline == 'eagle_only':
            eagle_loss = self._compute_eagle_loss(
                hs_projected, input_ids_shifted, target,
                attn_mask, position_ids, loss_mask_2d)
            return eagle_loss, torch.tensor(0.0, device=device), {
                "sdpo_loss": 0.0, "eagle_loss": eagle_loss.item(),
                "mean_tau": 0.0, "num_valid": loss_mask_2d.float().sum().item(),
                "step_acc": [0.0] * gamma, "tau_hist": [0] * (gamma + 1),
            }

        t2d = self.t2d.to(device)
        d2t = self.d2t.to(device)
        draft_ids = torch.arange(len(d2t), device=device)
        full2draft = torch.full((t2d.shape[0],), -1, dtype=torch.long, device=device)
        full2draft[draft_ids + d2t] = draft_ids

        # --- Teacher-forced γ steps: collect logprobs and accepts ---
        all_logprobs = []
        all_accepts = []
        cache_hidden = [[], []]
        cur_ids = input_ids_shifted
        cur_hs = hs_projected
        cur_tgt = target_greedy.clone()

        for idx in range(gamma):
            embeds = self.embed_tokens(cur_ids).to(cur_hs.dtype)
            if self.training and self.gradient_checkpointing and not embeds.requires_grad:
                embeds.requires_grad_(True)

            if self.gradient_checkpointing and self.training:
                def _fwd(module):
                    def _f(*inputs):
                        return module(*inputs, None, False)
                    return _f
                layer_out, cache_hidden = torch.utils.checkpoint.checkpoint(
                    _fwd(self.midlayer),
                    embeds, cur_hs, cache_hidden, attn_mask, position_ids)
            else:
                layer_out, cache_hidden = self.midlayer(
                    input_emb=embeds, hidden_states=cur_hs,
                    cache_hidden=cache_hidden, attention_mask=attn_mask,
                    position_ids=position_ids, past_key_value=None,
                    output_attentions=False, use_cache=True)

            cur_hs = layer_out[0]
            logits = self.lm_head(self.norm(cur_hs)).float()

            with torch.no_grad():
                draft_d = logits.detach().argmax(dim=-1)
                draft_full = draft_d + d2t[draft_d]
                tgt_in = t2d[cur_tgt]
                accept = (draft_full == cur_tgt) & tgt_in

            # log q(y*) with grad
            tgt_d = full2draft[cur_tgt].clamp(min=0)
            lp = F.log_softmax(logits, dim=-1).gather(-1, tgt_d.unsqueeze(-1)).squeeze(-1)
            lp = torch.where(tgt_in, lp, torch.full_like(lp, -20.0))

            all_logprobs.append(lp)
            all_accepts.append(accept)

            if idx < gamma - 1:
                cur_ids = cur_tgt.detach()
                cur_tgt = torch.cat(
                    [cur_tgt[:, 1:], torch.zeros_like(cur_tgt[:, :1])], dim=1)

        logprobs = torch.stack(all_logprobs, dim=2)  # [B, L, γ], has grad
        accepts = torch.stack(all_accepts, dim=2)  # [B, L, γ], detached

        # --- Compute τ and L_SDPO ---
        sdpo_loss = self._compute_sdpo_loss(logprobs, accepts, loss_mask_2d)

        # --- L_EAGLE ---
        eagle_loss = self._compute_eagle_loss(
            hs_projected, input_ids_shifted, target,
            attn_mask, position_ids, loss_mask_2d)

        total_loss = sdpo_loss + eagle_coef * eagle_loss

        # --- Metrics ---
        with torch.no_grad():
            valid_chain = torch.cumprod(accepts.float(), dim=2)
            tau = valid_chain.sum(dim=2)  # [B, L]
            nv = loss_mask_2d.float().sum()
            mean_tau = (tau * loss_mask_2d.float()).sum() / (nv + 1e-8)
            step_acc = [accepts[:, :, k][loss_mask_2d].float().mean().item()
                        for k in range(gamma)]
            # τ histogram
            tau_int = tau[loss_mask_2d].long().clamp(max=gamma)
            tau_hist = [(tau_int == v).sum().item() for v in range(gamma + 1)]

        metrics = {
            "sdpo_loss": sdpo_loss.item(),
            "eagle_loss": eagle_loss.item(),
            "mean_tau": mean_tau.item(),
            "num_valid": nv.item(),
            "step_acc": step_acc,
            "tau_hist": tau_hist,
        }
        return total_loss, sdpo_loss, metrics

    def _compute_sdpo_loss(self, logprobs, accepts, loss_mask_2d):
        """
        L_SDPO = -Σ_{t=τ+1}^{γ} w_t on rejected suffix.

        w_t = Π_{j=τ+1}^{t-1} q(y_j*), so w_{τ+1} = 1.
        Implementation: mask logprobs to rejected positions, cumsum, shift, exp.
        """
        B, L, gamma = logprobs.shape
        device = logprobs.device

        # τ per position: number of consecutively accepted steps
        with torch.no_grad():
            valid_chain = torch.cumprod(accepts.float(), dim=2)
            tau = valid_chain.sum(dim=2).long()  # [B, L], values 0..γ

        # rejected_mask[b, l, t] = True if step t is rejected (t >= τ)
        step_idx = torch.arange(gamma, device=device)
        rejected_mask = step_idx >= tau.unsqueeze(-1)  # [B, L, γ]

        # Zero out accepted positions' logprobs
        rej_lp = logprobs * rejected_mask.float()  # [B, L, γ]

        # Cumulative sum of rejected logprobs
        cum_rej_lp = torch.cumsum(rej_lp, dim=2)  # [B, L, γ]

        # Shift right by 1: w_t = exp(Σ_{j=τ}^{t-1} log q_j)
        # w at first rejected position = exp(0) = 1
        shifted = torch.cat([
            torch.zeros(B, L, 1, device=device, dtype=cum_rej_lp.dtype),
            cum_rej_lp[:, :, :-1]
        ], dim=2)

        w_t = torch.exp(shifted) * rejected_mask.float()  # [B, L, γ]

        # Loss: negative sum over rejected positions, averaged over valid sequence positions
        nv = loss_mask_2d.float().sum()
        loss = -(w_t * loss_mask_2d.float().unsqueeze(2)).sum() / (nv + 1e-8)
        return loss

    def _compute_eagle_loss(self, hs_projected, input_ids_shifted, target_logits,
                            attn_mask, position_ids, loss_mask_2d):
        """Soft-target KL matching EAGLE3 training (cnets.py:790-868)."""
        device = hs_projected.device
        gamma = self.length
        t2d_dev = self.t2d.to(device)

        losses = []
        cache_hidden = [[], []]
        cur_hs = hs_projected
        cur_ids = input_ids_shifted
        cur_target = target_logits
        cur_mask = loss_mask_2d.float().unsqueeze(-1)

        for idx in range(gamma):
            embeds = self.embed_tokens(cur_ids).to(cur_hs.dtype)
            if self.gradient_checkpointing and self.training:
                if not embeds.requires_grad:
                    embeds.requires_grad_(True)
                def _fwd(module):
                    def _f(*inputs):
                        return module(*inputs, None, False)
                    return _f
                layer_out, cache_hidden = torch.utils.checkpoint.checkpoint(
                    _fwd(self.midlayer),
                    embeds, cur_hs, cache_hidden, attn_mask, position_ids)
            else:
                layer_out, cache_hidden = self.midlayer(
                    input_emb=embeds, hidden_states=cur_hs,
                    cache_hidden=cache_hidden, attention_mask=attn_mask,
                    position_ids=position_ids, past_key_value=None,
                    output_attentions=False, use_cache=True)

            cur_hs = layer_out[0]
            logits = self.lm_head(self.norm(cur_hs)).float()

            with torch.no_grad():
                tgt_max = cur_target.argmax(dim=-1)
                tgt_mask = t2d_dev[tgt_max].unsqueeze(-1).float()
                pos_mask = tgt_mask * cur_mask
                tgt_p = torch.softmax(cur_target[..., t2d_dev].float(), dim=-1)

            out_logp = F.log_softmax(logits, dim=-1)
            loss = -torch.sum(pos_mask * tgt_p * out_logp, dim=-1).mean()
            losses.append(loss)

            if idx < gamma - 1:
                with torch.no_grad():
                    cur_ids = torch.cat([cur_ids[:, 1:], torch.zeros_like(cur_ids[:, :1])], dim=1)
                    cur_target = torch.cat([cur_target[:, 1:], torch.zeros_like(cur_target[:, :1])], dim=1)
                    cur_mask = torch.cat([cur_mask[:, 1:], torch.zeros_like(cur_mask[:, :1])], dim=1)

        return sum(losses)
