"""
Boundary-Aware Accepted Prefix Optimization (BAPO) Model.

Instead of matching token distributions (standard KL distillation) or maximizing
expected acceptance proxies, we train the draft model on the exact verifier-induced
stopping structure, with strong credit assigned to the first rejected token that
terminates the speculative block.

Three training phases:
  Phase 1 (distill): Standard soft-target KL distillation (warm start / from-scratch)
  Phase 2 (bapo):    Boundary-weighted rollout supervision (main contribution)
  Phase 3 (pg):      Optional REINFORCE policy-gradient correction

Phase 2 loss (recommended first prototype):

  L = sum_{t<L-1} w_far * CE(q_t, y*_t)           # early accepted tokens
    + w_near * CE(q_{L-1}, y*_{L-1})               # last accepted token
    + w_fail * 1[L<K] * CE(q_L, y*_L)              # boundary (first rejected)
    + tau * sum_{t=0}^{K-1} KL(p_t || q_t)         # KL stability anchor

where L is the realized accepted length from autoregressive draft rollout
verified against the frozen target model.

Key design principles:
  1. Prefix-aware:      earlier mistakes matter more than later ones
  2. Boundary-focused:  the first rejected token gets special credit
  3. Anti-cheating:     cannot win by becoming degenerate or over-conservative
  4. Stable:            KL anchor prevents optimization collapse
"""

import sys
import os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'traineagle3'))
from traineagle3.cnets import Model


class BAPOModel(Model):

    def __init__(self, config, ds_config, training_config, path,
                 load_emb=True, load_head=True):
        super().__init__(config, ds_config, training_config,
                         load_emb=load_emb, load_head=load_head, path=path)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def forward(self, input_ids, attention_mask, loss_mask,
                phase='bapo', **kwargs):
        if phase == 'distill':
            return self._forward_distill(input_ids, attention_mask, loss_mask)
        elif phase == 'bapo':
            return self._forward_bapo(input_ids, attention_mask, loss_mask, **kwargs)
        elif phase == 'pg':
            return self._forward_pg(input_ids, attention_mask, loss_mask, **kwargs)
        else:
            raise ValueError(f"Unknown phase: {phase}")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _shift_left(x):
        """Shift tensor left by 1 along dim=1, zero-pad on the right."""
        return torch.cat([x[:, 1:], torch.zeros_like(x[:, :1])], dim=1)

    def _midlayer_step(self, embeds, cur_hs, cache_hidden, attn_mask, position_ids):
        """Single midlayer forward with optional gradient checkpointing."""
        if self.training and self.gradient_checkpointing:
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
        return layer_out, cache_hidden

    def _prepare_common(self, input_ids, attention_mask, loss_mask):
        """Run frozen target model, project hidden states, build masks."""
        hidden_states, target_logits, loss_mask_3d, input_ids_shifted = \
            self.dataprepare(input_ids, attention_mask, loss_mask)
        loss_mask_2d = loss_mask_3d.squeeze(-1).bool()
        B, S, _ = hidden_states.shape
        device = hidden_states.device

        hidden_states = hidden_states.to(self.fc.weight.dtype)
        hs_projected = self.fc(hidden_states)
        if self.training and self.gradient_checkpointing and not hs_projected.requires_grad:
            hs_projected.requires_grad_(True)

        attn_mask = self._prepare_decoder_attention_mask(
            attention_mask, (B, S), hs_projected, 0)
        position_ids = torch.arange(S, dtype=torch.long, device=device).unsqueeze(0)

        return (hs_projected, target_logits, loss_mask_2d, input_ids_shifted,
                attn_mask, position_ids, B, S, device)

    def _build_vocab_mapping(self, device):
        """Build draft<->target vocabulary mapping tensors."""
        t2d = self.t2d.to(device)
        d2t = self.d2t.to(device)
        draft_ids = torch.arange(len(d2t), device=device)
        full2draft = torch.full((t2d.shape[0],), -1, dtype=torch.long, device=device)
        full2draft[draft_ids + d2t] = draft_ids
        return t2d, d2t, full2draft

    def _precompute_shifted_masks(self, loss_mask_2d, gamma, B, device):
        """Precompute left-shifted loss masks for each speculative step.

        At step t the draft predicts for position i+t+2, so the mask at
        position i should reflect whether the *target* at that shifted
        offset is still inside the valid range.
        """
        masks = [loss_mask_2d]
        m = loss_mask_2d
        for _ in range(1, gamma):
            m = torch.cat([m[:, 1:],
                           torch.zeros(B, 1, dtype=m.dtype, device=device)], dim=1)
            masks.append(m)
        return masks

    # ==================================================================
    # Phase 1 — Distillation (standard EAGLE3 soft-target KL)
    # ==================================================================
    def _forward_distill(self, input_ids, attention_mask, loss_mask):
        """Teacher-forced soft-target KL distillation (same objective as EAGLE3).

        Used as Phase 1 warm-start for from-scratch training.
        """
        (hs_projected, target_logits, loss_mask_2d, input_ids_shifted,
         attn_mask, position_ids, B, S, device) = \
            self._prepare_common(input_ids, attention_mask, loss_mask)

        gamma = self.length
        t2d_dev = self.t2d.to(device)

        total_loss = torch.tensor(0.0, device=device)
        accs = []
        cache_hidden = [[], []]
        cur_hs = hs_projected
        cur_ids = input_ids_shifted
        cur_target = target_logits
        cur_mask = loss_mask_2d.float().unsqueeze(-1)          # [B, S, 1]

        for idx in range(gamma):
            embeds = self.embed_tokens(cur_ids).to(cur_hs.dtype)
            layer_out, cache_hidden = self._midlayer_step(
                embeds, cur_hs, cache_hidden, attn_mask, position_ids)
            cur_hs = layer_out[0]
            logits = self.lm_head(self.norm(cur_hs)).float()

            with torch.no_grad():
                tgt_max = cur_target.argmax(dim=-1)
                tgt_mask = t2d_dev[tgt_max].unsqueeze(-1).float()
                pos_mask = tgt_mask * cur_mask
                tgt_p = F.softmax(cur_target[..., t2d_dev].float(), dim=-1)
                acc = ((logits.argmax(-1) == tgt_p.argmax(-1))
                       * pos_mask.squeeze(-1)).sum() / (cur_mask.sum() + 1e-6)
                accs.append(acc.item())

            out_logp = F.log_softmax(logits, dim=-1)
            loss = -torch.sum(pos_mask * tgt_p * out_logp, dim=-1).mean()
            total_loss = total_loss + loss

            if idx < gamma - 1:
                with torch.no_grad():
                    # Ground-truth teacher forcing: shift everything left
                    cur_ids = self._shift_left(cur_ids)
                    cur_target = self._shift_left(cur_target)
                    cur_mask = torch.cat(
                        [cur_mask[:, 1:], torch.zeros_like(cur_mask[:, :1])], dim=1)

        metrics = {
            "distill_loss": total_loss.item(),
            "mean_tau": 0.0,
            "step_acc": accs,
            "num_valid": loss_mask_2d.float().sum().item(),
            "tau_hist": [0] * (gamma + 1),
        }
        return total_loss, metrics

    # ==================================================================
    # Phase 2 — BAPO  (boundary-weighted rollout supervision)
    # ==================================================================
    def _forward_bapo(self, input_ids, attention_mask, loss_mask,
                      w_far=1.0, w_near=2.0, w_fail=6.0, w_post=0.5,
                      kl_coef=0.01, sampling='greedy',
                      prod_coef=0.0, prod_tau=1.0):
        """Autoregressive rollout from draft, verified against target.

        Loss focuses supervision on the exact stopping event:
          * accepted tokens get mild CE push  (weight w_far)
          * the last accepted token gets more (weight w_near)
          * the first rejected token (boundary) gets the most (weight w_fail)
          * tokens after boundary get flat CE  (weight w_post)
          * weak KL anchor on all visited states for stability
        """
        (hs_projected, target_logits, loss_mask_2d, input_ids_shifted,
         attn_mask, position_ids, B, S, device) = \
            self._prepare_common(input_ids, attention_mask, loss_mask)

        gamma = self.length
        t2d, d2t, full2draft = self._build_vocab_mapping(device)
        shifted_masks = self._precompute_shifted_masks(
            loss_mask_2d, gamma, B, device)

        # -- Pre-compute target greedy tokens and logits at each step ------
        with torch.no_grad():
            tgt_greedy_steps = []               # gamma x [B, S]
            tgt_logits_steps = []               # gamma x [B, S, V_target]
            g = target_logits.argmax(dim=-1)
            l = target_logits.clone()
            for t in range(gamma):
                tgt_greedy_steps.append(g.clone())
                tgt_logits_steps.append(l.clone())
                if t < gamma - 1:
                    g = self._shift_left(g)
                    l = self._shift_left(l)

        # -- Autoregressive rollout ----------------------------------------
        all_logits  = []                        # gamma x [B, S, V_draft]
        all_accepts = []                        # gamma x [B, S] bool
        cache_hidden = [[], []]
        cur_ids = input_ids_shifted
        cur_hs  = hs_projected

        for t in range(gamma):
            embeds = self.embed_tokens(cur_ids).to(cur_hs.dtype)
            layer_out, cache_hidden = self._midlayer_step(
                embeds, cur_hs, cache_hidden, attn_mask, position_ids)
            cur_hs = layer_out[0]
            logits = self.lm_head(self.norm(cur_hs)).float()
            all_logits.append(logits)

            with torch.no_grad():
                # Draft prediction
                if sampling == 'greedy':
                    draft_d = logits.argmax(dim=-1)             # [B, S]
                else:
                    probs = F.softmax(logits, dim=-1)
                    draft_d = torch.multinomial(
                        probs.view(-1, probs.size(-1)), 1).view(B, S)

                draft_full = draft_d + d2t[draft_d]             # map → full vocab

                # Acceptance: does draft match target at this step?
                tgt_step = tgt_greedy_steps[t]
                tgt_in   = t2d[tgt_step]
                accept   = (draft_full == tgt_step) & tgt_in
                all_accepts.append(accept)


            # Feed draft's own prediction (autoregressive, not teacher-forced)
            if t < gamma - 1:
                cur_ids = draft_full.detach()

        # -- Realized accepted length L ------------------------------------
        accepts = torch.stack(all_accepts, dim=-1)              # [B, S, gamma]
        with torch.no_grad():
            cumprod    = torch.cumprod(accepts.float(), dim=-1)
            L_accepted = cumprod.sum(dim=-1).long()             # [B, S] ∈ {0..gamma}

        # -- Boundary-weighted CE ------------------------------------------
        boundary_loss = self._boundary_loss(
            all_logits, tgt_greedy_steps, L_accepted, shifted_masks,
            w_far, w_near, w_fail, w_post, full2draft, t2d, gamma)

        # -- KL anchor -----------------------------------------------------
        kl_loss = self._kl_anchor(
            all_logits, tgt_logits_steps, shifted_masks, t2d, gamma)

        total_loss = boundary_loss + kl_coef * kl_loss

        # -- Product acceptance loss (differentiable E[L] surrogate) -------
        prod_loss_val = 0.0
        if prod_coef > 0:
            prod_loss = self._product_acceptance_loss(
                all_logits, tgt_greedy_steps, shifted_masks,
                full2draft, t2d, gamma, tau=prod_tau)
            total_loss = total_loss + prod_coef * prod_loss
            prod_loss_val = prod_loss.item()

        # -- Metrics -------------------------------------------------------
        with torch.no_grad():
            nv = loss_mask_2d.float().sum()
            mean_tau = (L_accepted.float() * loss_mask_2d.float()).sum() / (nv + 1e-8)
            step_acc = [accepts[:, :, k][loss_mask_2d].float().mean().item()
                        for k in range(gamma)]
            L_int = L_accepted[loss_mask_2d]
            tau_hist = [(L_int == v).sum().item() for v in range(gamma + 1)]

        metrics = {
            "boundary_loss": boundary_loss.item(),
            "kl_loss":       kl_loss.item(),
            "prod_loss":     prod_loss_val,
            "mean_tau":      mean_tau.item(),
            "num_valid":     nv.item(),
            "step_acc":      step_acc,
            "tau_hist":      tau_hist,
        }
        return total_loss, metrics

    # ---- helpers for Phase 2 ---

    def _boundary_loss(self, all_logits, tgt_greedy_steps, L_accepted,
                       shifted_masks, w_far, w_near, w_fail, w_post,
                       full2draft, t2d, gamma):
        """Boundary-weighted CE loss.

        Per position (b, s) with accepted length L:
          steps t < L-1 :  weight w_far   (early accepted)
          step  t = L-1 :  weight w_near  (last accepted)
          step  t = L   :  weight w_fail  (first rejected, if L < K)
          steps t > L   :  weight w_post  (post-boundary, mild distillation)
        Target token at every step is the verifier's expected / correction token.
        """
        device = all_logits[0].device
        total_ce = torch.tensor(0.0, device=device)
        total_w = torch.tensor(0.0, device=device)

        for t in range(gamma):
            logits = all_logits[t]                              # [B, S, Vd]
            tgt_full = tgt_greedy_steps[t]                      # [B, S]
            tgt_in = t2d[tgt_full]                              # [B, S] bool
            tgt_d = full2draft[tgt_full].clamp(min=0)           # [B, S]

            # Boundary weights
            is_early = (t < L_accepted - 1)                     # [B, S]
            is_last_acc = (t == L_accepted - 1) & (L_accepted > 0)
            is_boundary = (t == L_accepted) & (L_accepted < gamma)
            is_post = (t > L_accepted)

            w = (is_early.float() * w_far
                 + is_last_acc.float() * w_near
                 + is_boundary.float() * w_fail
                 + is_post.float() * w_post)

            # Combine validity mask: original seq, shifted seq, vocab membership
            valid = shifted_masks[t] & shifted_masks[0] & tgt_in
            mask = valid.float() * w

            # CE only at valid positions — avoids NaN from padding/OOV garbage
            log_p = F.log_softmax(logits, dim=-1)
            ce_raw = -log_p.gather(-1, tgt_d.unsqueeze(-1)).squeeze(-1)
            ce = torch.where(valid, ce_raw, torch.zeros_like(ce_raw))

            total_ce = total_ce + (ce * w).sum()
            total_w = total_w + mask.sum()

        return total_ce / (total_w + 1e-8)

    def _kl_anchor(self, all_logits, tgt_logits_steps, shifted_masks,
                   t2d, gamma):
        """KL(target || draft) on all visited states (gradient ≡ CE)."""
        device = all_logits[0].device
        total = torch.tensor(0.0, device=device)
        total_n = torch.tensor(0.0, device=device)

        for t in range(gamma):
            logits    = all_logits[t]
            tgt_lgt   = tgt_logits_steps[t]
            valid     = shifted_masks[t] & shifted_masks[0]

            with torch.no_grad():
                tgt_p = F.softmax(tgt_lgt[..., t2d].float(), dim=-1)

            log_q = F.log_softmax(logits, dim=-1)
            ce_raw = -(tgt_p * log_q).sum(dim=-1)              # [B, S]
            ce = torch.where(valid, ce_raw, torch.zeros_like(ce_raw))

            total   = total   + ce.sum()
            total_n = total_n + valid.float().sum()

        return total / (total_n + 1e-8)

    def _product_acceptance_loss(self, all_logits, tgt_greedy_steps,
                                 shifted_masks, full2draft, t2d, gamma,
                                 tau=1.0):
        """Differentiable surrogate for E[accepted length] via product of soft indicators.

        β̃_i = σ((z[y*_i] - max(z_i)) / τ)
        E[L] ≈ Σ_{j=1}^{K} ∏_{i=1}^{j} β̃_i

        Computed in log-space for numerical stability:
          log β̃_i = log σ(gap_i / τ) = -softplus(-gap_i / τ)
          cumlog_j = Σ_{i=1}^{j} log β̃_i
          E[L] = Σ_j exp(cumlog_j)

        The product structure automatically concentrates gradients on the
        weakest link (the boundary) without hand-tuned position weights.
        """
        device = all_logits[0].device

        log_betas = []
        for t in range(gamma):
            logits   = all_logits[t]                            # [B, S, Vd]
            tgt_full = tgt_greedy_steps[t]                      # [B, S]
            tgt_in   = t2d[tgt_full]                            # [B, S] bool
            tgt_d    = full2draft[tgt_full].clamp(min=0)        # [B, S]

            # Logit gap: 0 when draft agrees, negative otherwise
            z_target = logits.gather(-1, tgt_d.unsqueeze(-1)).squeeze(-1)
            z_max    = logits.max(dim=-1).values
            gap      = (z_target - z_max) / tau                 # [B, S] ≤ 0

            # Reversed-negative sigmoid indicator: β̃ = 2(1 - σ(|gap|))
            #   σ(|gap|) mirrors the negative part upward,
            #   1 - σ(|gap|) flips to a peak at gap=0,
            #   ×2 scales peak to 1.
            # In log-space: log(2(1-σ(|gap|))) = log(2) - softplus(|gap|)
            log_beta = 0.6931471805599453 - F.softplus(gap.abs())  # [B, S]

            # Invalid positions → log_beta=0 (β̃=1), so they don't block the product
            valid = shifted_masks[t] & shifted_masks[0] & tgt_in
            log_beta = torch.where(valid, log_beta, torch.zeros_like(log_beta))

            log_betas.append(log_beta)

        # Stack → cumsum in log-space → exp → sum over steps
        log_betas = torch.stack(log_betas, dim=-1)              # [B, S, gamma]
        cumlog    = torch.cumsum(log_betas, dim=-1)             # [B, S, gamma]
        E_L       = torch.exp(cumlog).sum(dim=-1)               # [B, S]

        # Average over valid positions
        valid_mask = shifted_masks[0]
        nv = valid_mask.float().sum()
        mean_E_L = (E_L * valid_mask.float()).sum() / (nv + 1e-8)

        # Negative because we maximize E[L]
        return -mean_E_L

    # ==================================================================
    # Phase 3 — Policy Gradient  (REINFORCE with accepted-length reward)
    # ==================================================================
    def _forward_pg(self, input_ids, attention_mask, loss_mask,
                    kl_coef=0.01, pg_coef=0.1, baseline=0.0):
        """REINFORCE correction on top of the BAPO-trained draft.

        Reward  R = L  (realized accepted length).
        Loss   = -pg_coef * (R - baseline) * sum_t log q(y_t | s_t)
               + kl_coef * KL(target || draft)

        Sampling is always stochastic in this phase.
        """
        (hs_projected, target_logits, loss_mask_2d, input_ids_shifted,
         attn_mask, position_ids, B, S, device) = \
            self._prepare_common(input_ids, attention_mask, loss_mask)

        gamma = self.length
        t2d, d2t, full2draft = self._build_vocab_mapping(device)
        shifted_masks = self._precompute_shifted_masks(
            loss_mask_2d, gamma, B, device)

        with torch.no_grad():
            tgt_greedy_steps = []
            tgt_logits_steps = []
            g = target_logits.argmax(dim=-1)
            l = target_logits.clone()
            for t in range(gamma):
                tgt_greedy_steps.append(g.clone())
                tgt_logits_steps.append(l.clone())
                if t < gamma - 1:
                    g = self._shift_left(g)
                    l = self._shift_left(l)

        # -- Rollout with sampled tokens -----------------------------------
        all_logits    = []
        all_log_probs = []                                      # log q(y_t | s_t)
        all_accepts   = []
        cache_hidden  = [[], []]
        cur_ids = input_ids_shifted
        cur_hs  = hs_projected

        for t in range(gamma):
            embeds = self.embed_tokens(cur_ids).to(cur_hs.dtype)
            layer_out, cache_hidden = self._midlayer_step(
                embeds, cur_hs, cache_hidden, attn_mask, position_ids)
            cur_hs = layer_out[0]
            logits = self.lm_head(self.norm(cur_hs)).float()
            all_logits.append(logits)

            log_p = F.log_softmax(logits, dim=-1)
            with torch.no_grad():
                probs   = F.softmax(logits, dim=-1)
                draft_d = torch.multinomial(
                    probs.view(-1, probs.size(-1)), 1).view(B, S)

            # Log-prob of the sampled token (gradient flows through log_p)
            token_lp = log_p.gather(-1, draft_d.unsqueeze(-1)).squeeze(-1)
            all_log_probs.append(token_lp)

            with torch.no_grad():
                draft_full = draft_d + d2t[draft_d]
                tgt_step   = tgt_greedy_steps[t]
                tgt_in     = t2d[tgt_step]
                accept     = (draft_full == tgt_step) & tgt_in
                all_accepts.append(accept)

            if t < gamma - 1:
                cur_ids = draft_full.detach()

        # -- Reward --------------------------------------------------------
        accepts = torch.stack(all_accepts, dim=-1)
        with torch.no_grad():
            cumprod    = torch.cumprod(accepts.float(), dim=-1)
            L_accepted = cumprod.sum(dim=-1)                    # [B, S] float
            advantage  = L_accepted - baseline

        # -- PG loss: -(R-b) * sum_t log q(y_t) ----------------------------
        total_lp = torch.stack(all_log_probs, dim=-1).sum(dim=-1)   # [B, S]
        nv = loss_mask_2d.float().sum()
        pg_loss = -(advantage * total_lp * loss_mask_2d.float()).sum() / (nv + 1e-8)

        # -- KL anchor -----------------------------------------------------
        kl_loss = self._kl_anchor(
            all_logits, tgt_logits_steps, shifted_masks, t2d, gamma)

        total_loss = pg_coef * pg_loss + kl_coef * kl_loss

        # -- Metrics -------------------------------------------------------
        with torch.no_grad():
            mean_tau = (L_accepted * loss_mask_2d.float()).sum() / (nv + 1e-8)
            step_acc = [accepts[:, :, k][loss_mask_2d].float().mean().item()
                        for k in range(gamma)]
            L_int = L_accepted[loss_mask_2d].long()
            tau_hist = [(L_int == v).sum().item() for v in range(gamma + 1)]

        metrics = {
            "pg_loss":        pg_loss.item(),
            "kl_loss":        kl_loss.item(),
            "mean_tau":       mean_tau.item(),
            "num_valid":      nv.item(),
            "step_acc":       step_acc,
            "tau_hist":       tau_hist,
            "mean_advantage": advantage[loss_mask_2d].mean().item(),
            "baseline":       baseline,
        }
        return total_loss, metrics
