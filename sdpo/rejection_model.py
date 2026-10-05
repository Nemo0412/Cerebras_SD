"""
SDPO Model: EAGLE (main) + Sigmoid Logit Gap (auxiliary).

L_total = L_EAGLE + sigmoid_coef * L_sigmoid

L_EAGLE = soft-target KL on all γ positions (dense anchor, protects accepted positions)
L_sigmoid = -Σ_{t=1}^{γ} (γ-t+1) * σ((z[y*] - max(z)) / T)
  - Works in logit space (no softmax q(1-q) vanishing)
  - Accepted positions: gap=0 → constant → zero gradient (automatic)
  - Rejected positions: push up z[y*], push down z[argmax]
  - Position weight (γ-t+1): linear decreasing, earlier positions matter more
  - Temperature T: high → wide coverage, low → focus on decision boundary
"""

import sys
import os
import math
import random
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
        # Ref draft (frozen initial policy for GRPO). Materialized lazily by
        # main script via _init_ref_draft_model_from_path AFTER deepspeed.initialize.
        # object.__setattr__ bypasses nn.Module submodule registration so DeepSpeed
        # ZeRO doesn't track the frozen ref.
        object.__setattr__(self, 'ref_draft_model', None)

    def _init_ref_draft_model_from_path(self, path, dtype, draft_path=None):
        """Snapshot current draft components (midlayer/fc/lm_head/norm) as a
        frozen GRPO reference policy. Called from main.py AFTER deepspeed init
        AND after load_draft_weights has loaded the initial AngelSlim weights.
        Target model is shared (already frozen). `path`, `draft_path`, `dtype`
        args kept for API compat with SmallLMModel; only `dtype` used."""
        import copy
        ref_components = {
            '_ref_midlayer': copy.deepcopy(self.midlayer),
            '_ref_fc': copy.deepcopy(self.fc),
            '_ref_lm_head': copy.deepcopy(self.lm_head),
            '_ref_norm': copy.deepcopy(self.norm),
        }
        for name, mod in ref_components.items():
            mod = mod.to(dtype)
            for p in mod.parameters():
                p.requires_grad = False
            mod.eval()
            object.__setattr__(self, name, mod)
        # Snapshot vocab mapping buffers
        object.__setattr__(self, '_ref_d2t', self.d2t.detach().clone())
        object.__setattr__(self, '_ref_t2d', self.t2d.detach().clone())
        # Marker for "ref available"
        object.__setattr__(self, 'ref_draft_model', True)

    def train(self, mode=True):
        super().train(mode)
        # Keep ref draft components frozen in eval mode regardless of train flag.
        for attr in ('_ref_midlayer', '_ref_fc', '_ref_lm_head', '_ref_norm'):
            mod = getattr(self, attr, None)
            if mod is not None:
                mod.eval()
        return self

    def forward(self, input_ids, attention_mask, loss_mask,
                sigmoid_coef=0.1, temperature=3.0, baseline=None,
                aux_loss='sigmoid',
                prob_temperature=1.0,
                grpo_coef=0.0, grpo_k_groups=8, grpo_m=4,
                grpo_eps=0.2, grpo_mode='window', grpo_sample_temp=1.0,
                grpo_reward='hard'):
        # prob_temperature: T applied to softmax(z/T) for al_tv/al_kl/wkl aux
        # losses (default 1.0 = current behavior).
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

        # NOTE: eagle_only / ce_only previously short-circuited here, which
        # silently bypassed GRPO even when grpo_coef > 0. Now handled at the
        # final-loss assembly below so GRPO can coexist with KL/CE anchor.

        # --- Vocab mapping ---
        t2d = self.t2d.to(device)
        d2t = self.d2t.to(device)
        draft_ids = torch.arange(len(d2t), device=device)
        full2draft = torch.full((t2d.shape[0],), -1, dtype=torch.long, device=device)
        full2draft[draft_ids + d2t] = draft_ids

        # --- Teacher-forced γ steps (EAGLE-3 cnets.py style): collect logits ---
        # Match traineagle3/cnets.py forward (line 833-907):
        #   each iter shifts input_ids / target / loss_mask via padding(left=False)
        #   so position p at iter k sees original training token at p+k+1.
        # Input token = ORIGINAL training data (not target.argmax).
        # hidden_states recurrent (draft's own from previous iter).
        all_logits = []
        all_tgt_d = []
        all_tgt_in = []
        all_accepts = []
        cache_hidden = [[], []]
        cur_ids = input_ids_shifted
        cur_hs = hs_projected
        cur_tgt = target_greedy.clone()
        cur_mask = loss_mask_2d.clone()

        def _shift_right(t):
            """padding(t, left=False): drop first, append 0 at end."""
            return torch.cat([t[:, 1:], torch.zeros_like(t[:, :1])], dim=1)

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
            all_logits.append(logits)

            # Target token in draft vocab (at the shifted position)
            tgt_d = full2draft[cur_tgt].clamp(min=0)
            tgt_in = t2d[cur_tgt]
            all_tgt_d.append(tgt_d)
            all_tgt_in.append(tgt_in)

            # Accept check (detached, for metrics)
            with torch.no_grad():
                draft_d = logits.detach().argmax(dim=-1)
                draft_full = draft_d + d2t[draft_d]
                accept = (draft_full == cur_tgt) & tgt_in & cur_mask
            all_accepts.append(accept)

            if idx < gamma - 1:
                # cnets.py-style shift: original training data, not target.argmax
                cur_ids = _shift_right(cur_ids).detach()
                cur_tgt = _shift_right(cur_tgt)
                cur_mask = _shift_right(cur_mask)

        accepts = torch.stack(all_accepts, dim=2)  # [B, L, γ]

        # --- Auxiliary loss ---
        # eagle_only / ce_only baselines skip aux entirely (aux=0, no forward).
        if baseline in ('eagle_only', 'ce_only'):
            aux = torch.tensor(0.0, device=device, requires_grad=True)
        elif aux_loss == 'acceptance_length':
            aux = self._compute_acceptance_length_loss(
                all_logits, all_tgt_d, all_tgt_in, loss_mask_2d, temperature)
        elif aux_loss == 'acceptance_length_v2':
            aux = self._compute_acceptance_length_loss_v2(
                all_logits, all_tgt_d, all_tgt_in, loss_mask_2d, temperature)
        elif aux_loss == 'acceptance_length_v3':
            aux = self._compute_acceptance_length_loss_v3(
                all_logits, all_tgt_d, all_tgt_in, loss_mask_2d, temperature)
        elif aux_loss == 'acceptance_length_v4':
            aux = self._compute_acceptance_length_loss_v4(
                all_logits, all_tgt_d, all_tgt_in, loss_mask_2d, temperature)
        elif aux_loss == 'acceptance_length_v8':
            aux = self._compute_acceptance_length_loss_v8(
                all_logits, all_tgt_d, all_tgt_in, loss_mask_2d, temperature)
        elif aux_loss == 'eal':
            aux = self._compute_eal_loss(
                all_logits, all_tgt_d, all_tgt_in, loss_mask_2d)
        elif aux_loss == 'none':
            aux = torch.tensor(0.0, device=device, requires_grad=True)
        elif aux_loss == 'tv':
            aux = self._compute_tv_loss_full(
                hs_projected, input_ids_shifted, target,
                attn_mask, position_ids, loss_mask_2d)
        elif aux_loss == 'al_tv':
            aux = self._compute_al_tv_loss(
                all_logits, target, loss_mask_2d, t2d, prob_temperature)
        elif aux_loss == 'al_kl':
            aux = self._compute_al_kl_loss(
                all_logits, target, loss_mask_2d, t2d, prob_temperature)
        elif aux_loss == 'wkl':
            aux = self._compute_wkl_loss(
                all_logits, target, loss_mask_2d, t2d, prob_temperature)
        else:
            aux = self._compute_sigmoid_loss(
                all_logits, all_tgt_d, all_tgt_in, loss_mask_2d, temperature)

        # --- GRPO loss (computed once, applies on top of anchor + aux) ---
        grpo_loss = torch.tensor(0.0, device=device)
        if grpo_coef > 0.0 and self.ref_draft_model is not None:
            # Pass the UN-projected concat hidden_states (12288-dim);
            # _compute_ref_all_logits applies its own _ref_fc to go to 4096.
            all_ref_logits = self._compute_ref_all_logits(
                hidden_states, input_ids_shifted, attn_mask, position_ids)
            if grpo_mode == 'sample':
                grpo_loss = self._compute_grpo_sample_loss(
                    all_logits, all_ref_logits, all_tgt_d, all_tgt_in,
                    loss_mask_2d, grpo_k_groups, grpo_m, grpo_eps,
                    grpo_sample_temp, grpo_reward)
            else:
                grpo_loss = self._compute_grpo_loss(
                    all_logits, all_ref_logits, all_tgt_d, all_tgt_in,
                    loss_mask_2d, grpo_k_groups, grpo_m, grpo_eps,
                    grpo_reward, target=target, t2d_dev=t2d)

        # --- Anchor (eagle / ce) + final total_loss ---
        if baseline == 'aux_only':
            # No anchor; aux + GRPO carry the gradient.
            eagle_loss = torch.tensor(0.0, device=device)
            total_loss = aux + grpo_coef * grpo_loss
        elif baseline == 'ce_only':
            eagle_loss = self._compute_ce_loss(
                hs_projected, input_ids_shifted, target,
                attn_mask, position_ids, loss_mask_2d)
            total_loss = eagle_loss + grpo_coef * grpo_loss
        elif baseline == 'eagle_only':
            eagle_loss = self._compute_eagle_loss(
                hs_projected, input_ids_shifted, target,
                attn_mask, position_ids, loss_mask_2d)
            total_loss = eagle_loss + grpo_coef * grpo_loss
        else:
            eagle_loss = self._compute_eagle_loss(
                hs_projected, input_ids_shifted, target,
                attn_mask, position_ids, loss_mask_2d)
            total_loss = eagle_loss + sigmoid_coef * aux + grpo_coef * grpo_loss

        # --- Metrics ---
        with torch.no_grad():
            valid_chain = torch.cumprod(accepts.float(), dim=2)
            tau = valid_chain.sum(dim=2)
            nv = loss_mask_2d.float().sum()
            mean_tau = (tau * loss_mask_2d.float()).sum() / (nv + 1e-8)
            step_acc = [accepts[:, :, k][loss_mask_2d].float().mean().item()
                        for k in range(gamma)]
            tau_int = tau[loss_mask_2d].long().clamp(max=gamma)
            tau_hist = [(tau_int == v).sum().item() for v in range(gamma + 1)]

        metrics = {
            "aux_loss": aux.item(),
            "eagle_loss": eagle_loss.item(),
            "grpo_loss": grpo_loss.item(),
            "mean_tau": mean_tau.item(),
            "num_valid": nv.item(),
            "step_acc": step_acc,
            "tau_hist": tau_hist,
        }
        return total_loss, aux, metrics

    def _compute_acceptance_length_loss_v3(self, all_logits, all_tgt_d, all_tgt_in,
                                            loss_mask_2d, temperature):
        """
        Soft acceptance length loss V3: loss ≈ -E[τ].

        β_soft = σ((gap + margin) / T):
          - gap=0 (accepted): β = σ(margin/T) ≈ 1
          - gap<<0 (rejected): β ≈ 0
        acceptance_length = Σ_{t=0}^{γ-1} ∏_{i=0}^{t} β_i
        Loss = -acceptance_length, directly approximates -E[τ].

        Uniform weight (no 0.8^t decay), so loss value ≈ negative acceptance length.
        """
        gamma = len(all_logits)
        nv = loss_mask_2d.float().sum()
        cur_mask = loss_mask_2d.clone()
        margin = 5.0  # shift so σ(margin/T) ≈ 1 when gap=0

        betas = []
        for t in range(gamma):
            logits = all_logits[t]
            tgt_d = all_tgt_d[t]
            tgt_in = all_tgt_in[t]

            z_target = logits.gather(-1, tgt_d.unsqueeze(-1)).squeeze(-1)
            z_max = logits.max(dim=-1).values
            gap = (z_target - z_max + margin) / temperature

            beta = torch.sigmoid(gap)
            beta = torch.where(tgt_in & cur_mask, beta, torch.ones_like(beta))
            betas.append(beta)

            if t < gamma - 1:
                cur_mask = torch.cat([cur_mask[:, 1:],
                                      torch.zeros_like(cur_mask[:, :1])], dim=1)

        betas = torch.stack(betas, dim=-1)  # [B, L, γ]
        cum_prod = torch.cumprod(betas, dim=-1)  # [B, L, γ]

        # uniform weight, sum over all steps
        acceptance_length = cum_prod.sum(dim=-1)  # [B, L]

        loss = -(acceptance_length * loss_mask_2d.float()).sum() / (nv + 1e-8)
        return loss

    def _compute_acceptance_length_loss_v4(self, all_logits, all_tgt_d, all_tgt_in,
                                            loss_mask_2d, temperature):
        """
        Soft acceptance length loss V4.

        β_soft = 2σ(gap / T):
          - gap=0 (accepted): 2σ(0) = 1.0
          - gap<<0 (rejected): 2σ(-∞) → 0.0
        Sum over steps with 0.8^t decay (matching EAGLE3).
        """
        gamma = len(all_logits)
        nv = loss_mask_2d.float().sum()
        cur_mask = loss_mask_2d.clone()

        betas = []
        for t in range(gamma):
            logits = all_logits[t]
            tgt_d = all_tgt_d[t]
            tgt_in = all_tgt_in[t]

            z_target = logits.gather(-1, tgt_d.unsqueeze(-1)).squeeze(-1)
            z_max = logits.max(dim=-1).values
            gap = (z_target - z_max) / temperature

            beta = 2.0 * torch.sigmoid(gap)
            beta = torch.where(tgt_in & cur_mask, beta, torch.ones_like(beta))
            betas.append(beta)

            if t < gamma - 1:
                cur_mask = torch.cat([cur_mask[:, 1:],
                                      torch.zeros_like(cur_mask[:, :1])], dim=1)

        betas = torch.stack(betas, dim=-1)  # [B, L, γ]
        cum_prod = torch.cumprod(betas, dim=-1)  # [B, L, γ]

        weights = torch.tensor([0.8 ** t for t in range(gamma)],
                               device=betas.device, dtype=betas.dtype)
        weighted = cum_prod * weights

        acceptance_length = weighted.sum(dim=-1)

        loss = -(acceptance_length * loss_mask_2d.float()).sum() / (nv + 1e-8)
        return loss

    def _compute_acceptance_length_loss_v8(self, all_logits, all_tgt_d, all_tgt_in,
                                            loss_mask_2d, temperature):
        """
        V8: gap to 2nd-largest draft logit (instead of max as in V4).
          gap = (z[target] - z[2nd-max]) / T
            > 0 if target is argmax (margin-aware: keep pushing)
            = 0 if target is 2nd
            < 0 if target is rank 3+ (gap to argmax, V4-equivalent)
          β = (2σ(gap)).clamp(max=1.0)
        cum_prod over γ-rollout with 0.8^t decay, then sum.
        """
        gamma = len(all_logits)
        nv = loss_mask_2d.float().sum()
        cur_mask = loss_mask_2d.clone()

        betas = []
        for t in range(gamma):
            logits = all_logits[t]
            tgt_d = all_tgt_d[t]
            tgt_in = all_tgt_in[t]

            z_target = logits.gather(-1, tgt_d.unsqueeze(-1)).squeeze(-1)
            top2 = logits.topk(2, dim=-1)
            z_2nd = top2.values[..., 1]
            gap = (z_target - z_2nd) / temperature

            beta = (2.0 * torch.sigmoid(gap)).clamp(max=1.0)
            beta = torch.where(tgt_in & cur_mask, beta, torch.ones_like(beta))
            betas.append(beta)

            if t < gamma - 1:
                cur_mask = torch.cat([cur_mask[:, 1:],
                                      torch.zeros_like(cur_mask[:, :1])], dim=1)

        betas = torch.stack(betas, dim=-1)
        cum_prod = torch.cumprod(betas, dim=-1)

        weights = torch.tensor([0.8 ** t for t in range(gamma)],
                               device=betas.device, dtype=betas.dtype)
        weighted = cum_prod * weights
        acceptance_length = weighted.sum(dim=-1)

        loss = -(acceptance_length * loss_mask_2d.float()).sum() / (nv + 1e-8)
        return loss

    def _compute_acceptance_length_loss_v2(self, all_logits, all_tgt_d, all_tgt_in,
                                            loss_mask_2d, temperature):
        """
        Soft acceptance length loss with exponential decay weighting.

        L = -Σ_{t=0}^{γ-1} 0.8^t · ∏_{i=0}^{t} β_soft(i)

        β_soft = σ(gap / T) + 0.5, clipped to [0, 1].
        Uses sum (all steps get gradient) with 0.8^t decay (matching EAGLE3).
        """
        gamma = len(all_logits)
        nv = loss_mask_2d.float().sum()
        cur_mask = loss_mask_2d.clone()

        betas = []
        for t in range(gamma):
            logits = all_logits[t]
            tgt_d = all_tgt_d[t]
            tgt_in = all_tgt_in[t]

            z_target = logits.gather(-1, tgt_d.unsqueeze(-1)).squeeze(-1)
            z_max = logits.max(dim=-1).values
            gap = (z_target - z_max) / temperature

            beta = (torch.sigmoid(gap) + 0.5).clamp(max=1.0)
            beta = torch.where(tgt_in & cur_mask, beta, torch.ones_like(beta))
            betas.append(beta)

            if t < gamma - 1:
                cur_mask = torch.cat([cur_mask[:, 1:],
                                      torch.zeros_like(cur_mask[:, :1])], dim=1)

        betas = torch.stack(betas, dim=-1)  # [B, L, γ]
        cum_prod = torch.cumprod(betas, dim=-1)  # [B, L, γ]

        # 0.8^t decay weights, matching EAGLE3
        weights = torch.tensor([0.8 ** t for t in range(gamma)],
                               device=betas.device, dtype=betas.dtype)
        weighted = cum_prod * weights  # [B, L, γ]

        # sum over all steps (all steps get gradient)
        acceptance_length = weighted.sum(dim=-1)  # [B, L]

        loss = -(acceptance_length * loss_mask_2d.float()).sum() / (nv + 1e-8)
        return loss

    def _compute_acceptance_length_loss(self, all_logits, all_tgt_d, all_tgt_in,
                                        loss_mask_2d, temperature):
        """
        Soft E[acceptance length] loss.

        E[τ] = max_j { j · ∏_{i=1}^{j} β_soft(i) }

        β_soft = σ(gap / T) + 0.5, clipped to [0, 1]:
          - gap=0 (accepted) → σ(0)+0.5 = 1.0
          - gap<<0 (rejected) → ≈0.5 → product decays
        OOV positions: β=1 (neutral, don't break the chain).
        Loss = -E[τ], maximizing acceptance length.
        """
        gamma = len(all_logits)
        nv = loss_mask_2d.float().sum()
        cur_mask = loss_mask_2d.clone()

        betas = []
        for t in range(gamma):
            logits = all_logits[t]
            tgt_d = all_tgt_d[t]
            tgt_in = all_tgt_in[t]

            z_target = logits.gather(-1, tgt_d.unsqueeze(-1)).squeeze(-1)
            z_max = logits.max(dim=-1).values
            gap = (z_target - z_max) / temperature

            beta = (torch.sigmoid(gap) + 0.5).clamp(max=1.0)
            beta = torch.where(tgt_in & cur_mask, beta, torch.ones_like(beta))
            betas.append(beta)

            if t < gamma - 1:
                cur_mask = torch.cat([cur_mask[:, 1:],
                                      torch.zeros_like(cur_mask[:, :1])], dim=1)

        betas = torch.stack(betas, dim=-1)  # [B, L, γ]
        cum_prod = torch.cumprod(betas, dim=-1)  # [B, L, γ]

        # j · ∏_{i=1}^{j} β_i  for j = 1, ..., γ
        positions = torch.arange(1, gamma + 1, device=betas.device, dtype=betas.dtype)
        weighted = cum_prod * positions  # [B, L, γ]

        # max_j { j · ∏β }
        acceptance_length = weighted.max(dim=-1).values  # [B, L]

        loss = -(acceptance_length * loss_mask_2d.float()).sum() / (nv + 1e-8)
        return loss

    def _compute_sigmoid_loss(self, all_logits, all_tgt_d, all_tgt_in,
                              loss_mask_2d, temperature):
        """
        L_sigmoid = -Σ_{t=0}^{γ-1} (γ-t) * σ((z[y*] - max(z)) / T)

        Accepted positions: z[y*] = max(z) → gap=0 → σ(0)=0.5 → constant → zero gradient.
        Rejected positions: gap < 0 → pushes z[y*] up and z[argmax] down.
        """
        gamma = len(all_logits)
        nv = loss_mask_2d.float().sum()
        total = torch.tensor(0.0, device=all_logits[0].device)
        cur_mask = loss_mask_2d.clone()

        for t in range(gamma):
            logits = all_logits[t]  # [B, L, V_draft]
            tgt_d = all_tgt_d[t]  # [B, L]
            tgt_in = all_tgt_in[t]  # [B, L] bool

            z_target = logits.gather(-1, tgt_d.unsqueeze(-1)).squeeze(-1)  # [B, L]
            z_max = logits.max(dim=-1).values  # [B, L]
            gap = (z_target - z_max) / temperature

            sig = torch.sigmoid(gap)  # [B, L]

            mask = cur_mask & tgt_in  # [B, L]
            pos_weight = gamma - t  # linear decreasing: γ, γ-1, ..., 1

            total = total - pos_weight * (sig * mask.float()).sum()

            if t < gamma - 1:
                cur_mask = torch.cat([cur_mask[:, 1:],
                                      torch.zeros_like(cur_mask[:, :1])], dim=1)

        return total / (nv + 1e-8)

    def _compute_tv_loss_full(self, hs_projected, input_ids_shifted, target_logits,
                              attn_mask, position_ids, loss_mask_2d):
        """
        TV loss computed during the draft forward (needs both p and q distributions).
        TV(p,q) = 0.5 * Σ|p_i - q_i| = 1 - Σ min(p_i, q_i).
        Aggregated across gamma steps with 0.8^k decay.
        """
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

            # p over draft vocab (target distribution, detached)
            with torch.no_grad():
                tgt_p = torch.softmax(cur_target[..., t2d_dev].float(), dim=-1)

            # q over draft vocab (draft distribution, has gradient)
            draft_q = F.softmax(logits, dim=-1)

            # TV = 1 - alpha = 1 - sum min(p, q)
            alpha = torch.sum(torch.min(tgt_p, draft_q), dim=-1)  # [B, L]
            tv = 1.0 - alpha

            pos_mask_2d = pos_mask.squeeze(-1)
            loss = (pos_mask_2d * tv).sum() / (pos_mask_2d.sum() + 1e-8)
            losses.append(loss)

            if idx < gamma - 1:
                with torch.no_grad():
                    cur_ids = torch.cat([cur_ids[:, 1:], torch.zeros_like(cur_ids[:, :1])], dim=1)
                    cur_target = torch.cat([cur_target[:, 1:], torch.zeros_like(cur_target[:, :1])], dim=1)
                    cur_mask = torch.cat([cur_mask[:, 1:], torch.zeros_like(cur_mask[:, :1])], dim=1)

        weights = [0.8 ** i for i in range(len(losses))]
        return sum(w * l for w, l in zip(weights, losses))

    def _compute_al_tv_loss(self, all_logits, target_logits, loss_mask_2d, t2d_dev,
                            prob_temperature=1.0):
        """
        E[L_t] = Σ_{k=1..γ} Π_{j=0..k-1} (1 − TV(p̃_{t+j}, q̃_{t+j}))
              = Σ_{k=1..γ} Π_{j=0..k-1} Σ_z min(p_z, q_z)

        Per-step β = 1 − TV = Σ min(p, q) (= marginal SpS accept rate).
        With prob_temperature T: p̃ = softmax(z/T), q̃ = softmax(z_d/T).
        Reuses draft logits from train()'s γ-rollout — no extra midlayer pass.
        """
        T = prob_temperature
        gamma = len(all_logits)
        nv = loss_mask_2d.float().sum()
        cur_target = target_logits
        cur_mask = loss_mask_2d.clone()

        betas = []
        for t in range(gamma):
            logits = all_logits[t]  # draft logits, has gradient
            with torch.no_grad():
                tgt_max = cur_target.argmax(dim=-1)
                tgt_in = t2d_dev[tgt_max]
                tgt_p = torch.softmax(
                    cur_target[..., t2d_dev].float() / T, dim=-1)

            draft_q = F.softmax(logits / T, dim=-1)
            beta = torch.sum(torch.min(tgt_p, draft_q), dim=-1)
            beta = torch.where(tgt_in & cur_mask, beta, torch.ones_like(beta))
            betas.append(beta)

            if t < gamma - 1:
                with torch.no_grad():
                    cur_target = torch.cat([cur_target[:, 1:], torch.zeros_like(cur_target[:, :1])], dim=1)
                    cur_mask = torch.cat([cur_mask[:, 1:], torch.zeros_like(cur_mask[:, :1])], dim=1)

        betas = torch.stack(betas, dim=-1)
        cum_prod = torch.cumprod(betas, dim=-1)
        acceptance_length = cum_prod.sum(dim=-1)
        loss = -(acceptance_length * loss_mask_2d.float()).sum() / (nv + 1e-8)
        return loss

    def _compute_al_kl_loss(self, all_logits, target_logits, loss_mask_2d, t2d_dev,
                            prob_temperature=1.0):
        """
        L_EAL^(t) = -Σ_{k=1..γ} exp(Σ_{j=0..k-1} log(0.5 · exp(-KL(p̃_{t+j} || q̃_{t+j}))))

        Log-space cumsum (no underflow) instead of cumprod.
        log β_j = log(0.5) − KL(p̃ || q̃) with p̃ = softmax(z/T), q̃ = softmax(z_d/T).
        Reuses draft logits from train()'s rollout.
        """
        T = prob_temperature
        gamma = len(all_logits)
        nv = loss_mask_2d.float().sum()
        cur_target = target_logits
        cur_mask = loss_mask_2d.clone()

        log_betas_list = []
        for t in range(gamma):
            logits = all_logits[t]

            with torch.no_grad():
                tgt_lg_d = cur_target[..., t2d_dev].float() / T
                tgt_logp = F.log_softmax(tgt_lg_d, dim=-1)
                tgt_p = tgt_logp.exp()
                tgt_max = cur_target.argmax(dim=-1)
                tgt_in = t2d_dev[tgt_max]

            out_logp = F.log_softmax(logits / T, dim=-1)
            kl = (tgt_p * (tgt_logp - out_logp)).sum(dim=-1)

            log_beta = math.log(0.5) - kl
            log_beta = torch.where(tgt_in & cur_mask, log_beta, torch.zeros_like(log_beta))
            log_betas_list.append(log_beta)

            if t < gamma - 1:
                with torch.no_grad():
                    cur_target = torch.cat([cur_target[:, 1:], torch.zeros_like(cur_target[:, :1])], dim=1)
                    cur_mask = torch.cat([cur_mask[:, 1:], torch.zeros_like(cur_mask[:, :1])], dim=1)

        log_betas = torch.stack(log_betas_list, dim=-1)
        cum_log = torch.cumsum(log_betas, dim=-1)
        acceptance_length = torch.exp(cum_log).sum(dim=-1)
        loss = -(acceptance_length * loss_mask_2d.float()).sum() / (nv + 1e-8)
        return loss

    def _compute_wkl_loss(self, all_logits, target_logits, loss_mask_2d, t2d_dev,
                          prob_temperature=1.0):
        """
        L̃_EAL^(t) = Σ_{j=0..γ-1} (γ − j) · KL(p̃_{t+j} || q̃_{t+j})

        Linearly-decaying per-step KL: j=0 has weight γ, j=γ-1 has weight 1.
        With prob_temperature T: p̃ = softmax(z/T), q̃ = softmax(z_d/T).
        Reuses draft logits from train()'s γ-rollout.
        """
        T = prob_temperature
        gamma = len(all_logits)
        cur_target = target_logits
        cur_mask = loss_mask_2d.clone()

        total = torch.tensor(0.0, device=loss_mask_2d.device)
        for t in range(gamma):
            logits = all_logits[t]
            with torch.no_grad():
                tgt_lg_d = cur_target[..., t2d_dev].float() / T
                tgt_logp = F.log_softmax(tgt_lg_d, dim=-1)
                tgt_p = tgt_logp.exp()
                tgt_max = cur_target.argmax(dim=-1)
                tgt_in = t2d_dev[tgt_max]

            out_logp = F.log_softmax(logits / T, dim=-1)
            kl = (tgt_p * (tgt_logp - out_logp)).sum(dim=-1)  # [B, L]

            pos_mask = (tgt_in & cur_mask).float()
            kl_avg = (kl * pos_mask).sum() / (pos_mask.sum() + 1e-8)
            weight = float(gamma - t)
            total = total + weight * kl_avg

            if t < gamma - 1:
                with torch.no_grad():
                    cur_target = torch.cat([cur_target[:, 1:], torch.zeros_like(cur_target[:, :1])], dim=1)
                    cur_mask = torch.cat([cur_mask[:, 1:], torch.zeros_like(cur_mask[:, :1])], dim=1)

        return total

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

        weights = [0.8 ** i for i in range(len(losses))]
        return sum(w * l for w, l in zip(weights, losses))

    def _compute_ce_loss(self, hs_projected, input_ids_shifted, target_logits,
                         attn_mask, position_ids, loss_mask_2d):
        """
        Hard cross-entropy loss: use target argmax as 1-hot label.
        L = -log q(target_argmax) per position, masked.
        Different from soft KL which uses full target distribution.
        Aggregated across γ steps with 0.8^k decay.
        """
        device = hs_projected.device
        gamma = self.length
        t2d_dev = self.t2d.to(device)
        d2t_dev = self.d2t.to(device)
        full2draft = torch.full((t2d_dev.shape[0],), -1, dtype=torch.long, device=device)
        draft_ids = torch.arange(len(d2t_dev), device=device)
        full2draft[draft_ids + d2t_dev] = draft_ids

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
                tgt_max = cur_target.argmax(dim=-1)             # [B, L] target vocab
                tgt_in_draft = t2d_dev[tgt_max]                 # [B, L] bool
                tgt_max_draft = full2draft[tgt_max].clamp(min=0)  # [B, L] draft vocab
                pos_mask = tgt_in_draft.float() * cur_mask.squeeze(-1)  # [B, L]

            out_logp = F.log_softmax(logits, dim=-1)
            ce_per_pos = -out_logp.gather(-1, tgt_max_draft.unsqueeze(-1)).squeeze(-1)  # [B, L]
            loss = (pos_mask * ce_per_pos).mean()
            losses.append(loss)

            if idx < gamma - 1:
                with torch.no_grad():
                    cur_ids = torch.cat([cur_ids[:, 1:], torch.zeros_like(cur_ids[:, :1])], dim=1)
                    cur_target = torch.cat([cur_target[:, 1:], torch.zeros_like(cur_target[:, :1])], dim=1)
                    cur_mask = torch.cat([cur_mask[:, 1:], torch.zeros_like(cur_mask[:, :1])], dim=1)

        weights = [0.8 ** i for i in range(len(losses))]
        return sum(w * l for w, l in zip(weights, losses))

    # ─── EAL aux loss ──────────────────────────────────────────────────
    def _compute_eal_loss(self, all_logits, all_tgt_d, all_tgt_in,
                          loss_mask_2d):
        """Expected Acceptance Length aux loss.

        β_t = P_t = softmax(draft_logits[t])[target_d_t]   (target prob in draft vocab)
        EAL = Σ_t 0.8^t · cum_prod(β_t)
        loss = -mean(EAL on valid positions)
        """
        gamma = len(all_logits)
        nv = loss_mask_2d.float().sum()
        cur_mask = loss_mask_2d.clone()
        betas = []
        for t in range(gamma):
            logits = all_logits[t]
            tgt_d = all_tgt_d[t]
            tgt_in = all_tgt_in[t]
            P = F.softmax(logits, dim=-1).gather(
                -1, tgt_d.unsqueeze(-1)).squeeze(-1)
            beta = torch.where(tgt_in & cur_mask, P, torch.ones_like(P))
            betas.append(beta)
            if t < gamma - 1:
                cur_mask = torch.cat([cur_mask[:, 1:],
                                      torch.zeros_like(cur_mask[:, :1])], dim=1)
        betas = torch.stack(betas, dim=-1)
        cum_prod = torch.cumprod(betas, dim=-1)
        weights = torch.tensor([0.8 ** t for t in range(gamma)],
                               device=betas.device, dtype=betas.dtype)
        eal = (cum_prod * weights).sum(dim=-1)
        loss = -(eal * loss_mask_2d.float()).sum() / (nv + 1e-8)
        return loss

    # ─── Ref draft γ-step rollout for GRPO ─────────────────────────────
    @torch.no_grad()
    def _compute_ref_all_logits(self, hs_projected, input_ids_shifted,
                                  attn_mask, position_ids):
        """Run frozen ref draft's γ-step rollout. Mirrors forward()'s loop
        but uses _ref_midlayer / _ref_fc / _ref_lm_head / _ref_norm.
        Returns list of γ tensors, each [B, L, V_draft]."""
        gamma = self.length
        cache_hidden = [[], []]
        cur_ids = input_ids_shifted.clone()
        cur_hs = self._ref_fc(hs_projected.to(self._ref_fc.weight.dtype))

        ref_logits = []
        for idx in range(gamma):
            embeds = self.embed_tokens(cur_ids).to(cur_hs.dtype)
            layer_out, cache_hidden = self._ref_midlayer(
                input_emb=embeds, hidden_states=cur_hs,
                cache_hidden=cache_hidden, attention_mask=attn_mask,
                position_ids=position_ids, past_key_value=None,
                output_attentions=False, use_cache=True)
            cur_hs = layer_out[0]
            logits = self._ref_lm_head(self._ref_norm(cur_hs)).float()
            ref_logits.append(logits)
            if idx < gamma - 1:
                cur_ids = torch.cat(
                    [cur_ids[:, 1:], torch.zeros_like(cur_ids[:, :1])], dim=1)
        return ref_logits

    # ─── GRPO window mode ──────────────────────────────────────────────
    def _compute_grpo_loss(self, all_logits, all_ref_logits, all_tgt_d,
                           all_tgt_in, loss_mask_2d,
                           K_groups, m_win, eps, reward_type='hard',
                           target=None, t2d_dev=None):
        device = all_logits[0].device
        gamma = len(all_logits)
        # Linear weights for wkl reward: [γ, γ−1, ..., 1]
        lin_w = torch.tensor([gamma - j for j in range(gamma)],
                             device=device, dtype=torch.float32)
        B, L, _ = all_logits[0].shape
        span = m_win + gamma - 1
        grpo_losses = []
        for b in range(B):
            mask_b = loss_mask_2d[b].int()
            if mask_b.shape[0] < span:
                continue
            cum = torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                             torch.cumsum(mask_b, 0)])
            valid = (cum[span:] - cum[:mask_b.shape[0] - span + 1]) == span
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
            if not selected:
                continue

            for g_start in selected:
                rewards, log_ratios = [], []
                for offset in range(m_win):
                    i = g_start + offset
                    draft_lg_win = torch.stack(
                        [all_logits[k][b, i] for k in range(gamma)], dim=0)
                    ref_lg_win = torch.stack(
                        [all_ref_logits[k][b, i] for k in range(gamma)], dim=0)
                    tgt_d_win = torch.stack(
                        [all_tgt_d[k][b, i] for k in range(gamma)], dim=0)
                    tgt_in_win = torch.stack(
                        [all_tgt_in[k][b, i] for k in range(gamma)], dim=0)

                    if reward_type == 'eal':
                        P_curr = F.softmax(draft_lg_win, dim=-1).gather(
                            -1, tgt_d_win.unsqueeze(-1)).squeeze(-1)
                        P_curr = torch.where(tgt_in_win, P_curr,
                                              torch.zeros_like(P_curr))
                        tau_curr = P_curr.cumprod(-1).sum()
                        with torch.no_grad():
                            P_ref = F.softmax(ref_lg_win, dim=-1).gather(
                                -1, tgt_d_win.unsqueeze(-1)).squeeze(-1)
                            P_ref = torch.where(tgt_in_win, P_ref,
                                                 torch.zeros_like(P_ref))
                            tau_ref = P_ref.cumprod(-1).sum()
                    elif reward_type in ('al_tv', 'al_kl', 'wkl'):
                        # Need target distribution restricted to draft vocab
                        # at positions i, i+1, ..., i+γ-1 in original sequence
                        with torch.no_grad():
                            tgt_lg_d_win = target[b, i:i+gamma][:, t2d_dev].float()  # [γ, V_draft]
                            tgt_logp_win = F.log_softmax(tgt_lg_d_win, dim=-1)
                            tgt_p_win = tgt_logp_win.exp()
                        if reward_type == 'al_tv':
                            draft_q_win = F.softmax(draft_lg_win.float(), dim=-1)
                            beta_curr = torch.sum(torch.min(tgt_p_win, draft_q_win), dim=-1)
                            beta_curr = torch.where(tgt_in_win, beta_curr, torch.ones_like(beta_curr))
                            tau_curr = beta_curr.cumprod(-1).sum()
                            with torch.no_grad():
                                ref_q_win = F.softmax(ref_lg_win.float(), dim=-1)
                                beta_ref = torch.sum(torch.min(tgt_p_win, ref_q_win), dim=-1)
                                beta_ref = torch.where(tgt_in_win, beta_ref, torch.ones_like(beta_ref))
                                tau_ref = beta_ref.cumprod(-1).sum()
                        elif reward_type == 'al_kl':
                            draft_logp_win = F.log_softmax(draft_lg_win.float(), dim=-1)
                            kl_curr = (tgt_p_win * (tgt_logp_win - draft_logp_win)).sum(-1)
                            log_beta_curr = math.log(0.5) - kl_curr
                            log_beta_curr = torch.where(tgt_in_win, log_beta_curr, torch.zeros_like(log_beta_curr))
                            tau_curr = torch.exp(log_beta_curr.cumsum(-1)).sum()
                            with torch.no_grad():
                                ref_logp_win = F.log_softmax(ref_lg_win.float(), dim=-1)
                                kl_ref = (tgt_p_win * (tgt_logp_win - ref_logp_win)).sum(-1)
                                log_beta_ref = math.log(0.5) - kl_ref
                                log_beta_ref = torch.where(tgt_in_win, log_beta_ref, torch.zeros_like(log_beta_ref))
                                tau_ref = torch.exp(log_beta_ref.cumsum(-1)).sum()
                        else:  # wkl
                            draft_logp_win = F.log_softmax(draft_lg_win.float(), dim=-1)
                            kl_curr = (tgt_p_win * (tgt_logp_win - draft_logp_win)).sum(-1)
                            kl_curr = torch.where(tgt_in_win, kl_curr, torch.zeros_like(kl_curr))
                            tau_curr = -(kl_curr * lin_w).sum()  # negate (reward = -loss)
                            with torch.no_grad():
                                ref_logp_win = F.log_softmax(ref_lg_win.float(), dim=-1)
                                kl_ref = (tgt_p_win * (tgt_logp_win - ref_logp_win)).sum(-1)
                                kl_ref = torch.where(tgt_in_win, kl_ref, torch.zeros_like(kl_ref))
                                tau_ref = -(kl_ref * lin_w).sum()
                    else:
                        S_curr = draft_lg_win.argmax(-1)
                        hit_curr = ((S_curr == tgt_d_win) & tgt_in_win).float()
                        tau_curr = hit_curr.cumprod(-1).sum()
                        with torch.no_grad():
                            S_ref = ref_lg_win.argmax(-1)
                            hit_ref = ((S_ref == tgt_d_win) & tgt_in_win).float()
                            tau_ref = hit_ref.cumprod(-1).sum()
                    rewards.append(tau_curr - tau_ref)

                    S_curr_d = draft_lg_win.argmax(-1).detach()
                    lp_draft = F.log_softmax(draft_lg_win, -1).gather(
                        -1, S_curr_d.unsqueeze(-1)).squeeze(-1).sum() / gamma
                    with torch.no_grad():
                        lp_ref = F.log_softmax(ref_lg_win, -1).gather(
                            -1, S_curr_d.unsqueeze(-1)).squeeze(-1).sum() / gamma
                    log_ratios.append(lp_draft - lp_ref)

                rewards_t = torch.stack(rewards)
                adv = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-6)
                adv = adv.clamp(-5.0, 5.0).detach()
                lr_t = torch.stack(log_ratios).clamp(-10.0, 10.0)
                ratio = lr_t.exp().clamp(0.1, 10.0)
                unclipped = ratio * adv
                clipped = ratio.clamp(1 - eps, 1 + eps) * adv
                grpo_losses.append(-torch.min(unclipped, clipped).mean())

        if grpo_losses:
            return torch.stack(grpo_losses).mean()
        # Empty-windows fallback: connect graph to all params so DeepSpeed
        # ZeRO grad-buckets stay consistent across ranks (otherwise IndexError).
        return 0.0 * sum(l.sum() for l in all_logits)

    # ─── GRPO sample mode ──────────────────────────────────────────────
    def _compute_grpo_sample_loss(self, all_logits, all_ref_logits, all_tgt_d,
                                    all_tgt_in, loss_mask_2d,
                                    K_groups, m_win, eps, sample_temp,
                                    reward_type='hard'):
        device = all_logits[0].device
        gamma = len(all_logits)
        B, L, _ = all_logits[0].shape
        grpo_losses = []
        for b in range(B):
            mask_b = loss_mask_2d[b].int()
            if mask_b.shape[0] < gamma:
                continue
            cum = torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                             torch.cumsum(mask_b, 0)])
            valid = (cum[gamma:] - cum[:mask_b.shape[0] - gamma + 1]) == gamma
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
                draft_lg_win = torch.stack(
                    [all_logits[k][b, anchor] for k in range(gamma)], dim=0)
                ref_lg_win = torch.stack(
                    [all_ref_logits[k][b, anchor] for k in range(gamma)], dim=0)
                tgt_d_win = torch.stack(
                    [all_tgt_d[k][b, anchor] for k in range(gamma)], dim=0)
                tgt_in_win = torch.stack(
                    [all_tgt_in[k][b, anchor] for k in range(gamma)], dim=0)

                lp_draft_full = F.log_softmax(draft_lg_win, dim=-1)
                with torch.no_grad():
                    lp_ref_full = F.log_softmax(ref_lg_win, dim=-1)
                    sample_probs = F.softmax(
                        draft_lg_win.detach() / sample_temp, dim=-1)
                    if reward_type == 'eal':
                        P_ref = F.softmax(ref_lg_win, dim=-1).gather(
                            -1, tgt_d_win.unsqueeze(-1)).squeeze(-1)
                        P_ref = torch.where(tgt_in_win, P_ref,
                                             torch.zeros_like(P_ref))
                        tau_ref = P_ref.cumprod(-1).sum()
                    else:
                        S_ref = ref_lg_win.argmax(-1)
                        hit_ref = ((S_ref == tgt_d_win) & tgt_in_win).float()
                        tau_ref = hit_ref.cumprod(-1).sum()

                rewards, log_ratios = [], []
                for _ in range(m_win):
                    with torch.no_grad():
                        S_curr = torch.multinomial(
                            sample_probs, num_samples=1).squeeze(-1)
                    if reward_type == 'eal':
                        P_curr = F.softmax(draft_lg_win, dim=-1).gather(
                            -1, tgt_d_win.unsqueeze(-1)).squeeze(-1)
                        P_curr = torch.where(tgt_in_win, P_curr,
                                              torch.zeros_like(P_curr))
                        tau_curr = P_curr.cumprod(-1).sum()
                    else:
                        hit_curr = ((S_curr == tgt_d_win) & tgt_in_win).float()
                        tau_curr = hit_curr.cumprod(-1).sum()
                    rewards.append(tau_curr - tau_ref)

                    lp_draft = lp_draft_full.gather(
                        -1, S_curr.unsqueeze(-1)).squeeze(-1).sum() / gamma
                    with torch.no_grad():
                        lp_ref = lp_ref_full.gather(
                            -1, S_curr.unsqueeze(-1)).squeeze(-1).sum() / gamma
                    log_ratios.append(lp_draft - lp_ref)

                rewards_t = torch.stack(rewards)
                adv = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-6)
                adv = adv.clamp(-5.0, 5.0).detach()
                lr_t = torch.stack(log_ratios).clamp(-10.0, 10.0)
                ratio = lr_t.exp().clamp(0.1, 10.0)
                unclipped = ratio * adv
                clipped = ratio.clamp(1 - eps, 1 + eps) * adv
                grpo_losses.append(-torch.min(unclipped, clipped).mean())

        if grpo_losses:
            return torch.stack(grpo_losses).mean()
        # Empty-windows fallback: maintain gradient flow through all params
        return 0.0 * sum(l.sum() for l in all_logits)
