"""
Small LM Draft Model with ON-POLICY ROLLOUT Training.

Draft (e.g., Qwen3-0.6B) trained against frozen target (e.g., Qwen3-32B)
using multi-step autoregressive rollout from every assistant anchor position:

    step 0: teacher-forced from prefix          → aux_k=0 signal
    step 1: forward on [prefix, d_{t+1} (detached argmax)] → signal
    ...
    step γ-1                                     → signal

Loss:
    L = L_anchor  +  sigmoid_coef · L_aux_rollout

anchor (--anchor):
    'kl' (soft-target KL) or 'ce' (hard-label CE on target argmax).
    Computed teacher-forced at all γ positions from the PREFIX forward.

aux_loss (--aux_loss):
    'none' — anchor only
    'v2'   — -Σ_k 0.8^k · ∏_{j=0..k} β_j  (β = σ((z_target - max z)/T) + 0.5)
             cumulative product benefits from on-policy exposure
    'v4'   — same cumprod structure as v2 but β = 2σ(gap/T)
             β ∈ [0, 1] (honest: a deep reject fully zeroes downstream)
    'tv'   —  Σ_k 0.8^k · (1 - overlap_k)  at each rolled-out step

on_policy_target (--on_policy_target):
    When True, after draft rollout, do ONE additional target forward that
    ingests draft's argmax chain at each anchor (4D mask isolated). Read target's
    argmax / dist at each rolled-out position → use as the DIRTY-CONTEXT target
    for β_k / α_k (k ≥ 1). This fixes the context mismatch of the default
    (where β_k compares draft@dirty-context to target@clean-context) at the cost
    of 1 extra target forward.

Supported: {kl, ce} × {none, v2, v4, tv}.

Architecture: shared prefix KV cache + 4D rollout mask isolates each anchor.
Requires draft loaded with attn_implementation="sdpa".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

try:
    import deepspeed
    _HAS_DS = True
except ImportError:
    _HAS_DS = False


def _get_step_weights(scheme, n, device, dtype):
    """Per-step weights, 4 schemes (un-normalized)."""
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


class SmallLMRolloutModel(nn.Module):

    def __init__(self, target_path, draft_path, gamma=7, dtype=torch.float16):
        super().__init__()
        self.gamma = gamma

        # sdpa is required for the custom 4D mask used both by draft rollout
        # and by the on-policy target re-forward. Flash attention treats a
        # 4D mask as a padding mask and would mix anchors together.
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_path, torch_dtype=dtype, attn_implementation="sdpa")
        self.target_model.eval()
        for p in self.target_model.parameters():
            p.requires_grad = False

        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_path, torch_dtype=dtype, attn_implementation="sdpa")

    def train(self, mode=True):
        super().train(mode)
        self.target_model.eval()
        self.draft_model.train(mode)
        return self

    def _compute_beta(self, logits, target_ids, temperature, variant='v2'):
        """β for V2 or V4 acceptance length loss.

        v2: β = σ(gap/T) + 0.5, clamped ≤ 1 → β ∈ [0.5, 1]
        v4: β = 2σ(gap/T)                   → β ∈ [0,   1]
        """
        z_t = logits.gather(-1, target_ids.unsqueeze(-1).long()).squeeze(-1)
        z_m = logits.max(-1).values
        gap = (z_t - z_m) / temperature
        if variant == 'v4':
            return 2.0 * torch.sigmoid(gap)
        return (torch.sigmoid(gap) + 0.5).clamp(max=1.0)

    def _build_rollout_mask(self, anchor_pos, step_k, prefix_len, K, device):
        """4D mask: each anchor attends to prefix + own rollout chain only."""
        total_kv = prefix_len + step_k * K
        dtype = self.draft_model.dtype
        min_val = torch.finfo(dtype).min
        mask = torch.full((1, 1, K, total_kv), min_val,
                          device=device, dtype=dtype)
        col = torch.arange(prefix_len, device=device)
        attend_prefix = col.unsqueeze(0) <= anchor_pos.unsqueeze(1)
        mask[0, 0, :, :prefix_len][attend_prefix] = 0.0
        anchor_idx = torch.arange(K, device=device)
        for j in range(step_k):
            kv_pos = prefix_len + j * K + anchor_idx
            mask[0, 0, anchor_idx, kv_pos] = 0.0
        return mask

    def _build_verify_mask(self, anchor_pos, num_steps, prefix_len, K, device, dtype):
        """4D mask for target verify on draft's rollout chain (one big forward).

        Layout (step-major flattening), total_q = num_steps * K:
          rows [0..K)            = step 0 tokens (draft_argmax at step 0 per anchor)
          rows [K..2K)           = step 1 tokens
          rows [k*K..(k+1)*K)    = step k tokens
          ...

        KV layout:
          cols [0..prefix_len)                 = prefix K/V (reused from cache)
          cols [prefix_len..prefix_len + K)    = step 0 tokens
          cols [prefix_len + K..prefix_len+2K) = step 1 tokens
          ...

        Row (step k, anchor i) attends to:
          - prefix cols [0..anchor_pos[i]]   (causal within clean prefix)
          - cols (prefix_len + j*K + i) for j=0..k  (own rollout chain up to itself)
        """
        total_kv = prefix_len + num_steps * K
        total_q = num_steps * K
        min_val = torch.finfo(dtype).min
        mask = torch.full((1, 1, total_q, total_kv), min_val,
                          device=device, dtype=dtype)

        col = torch.arange(prefix_len, device=device)
        anchor_idx = torch.arange(K, device=device)
        attend_prefix = col.unsqueeze(0) <= anchor_pos.unsqueeze(1)  # [K, L]

        for k in range(num_steps):
            row_slice = slice(k * K, (k + 1) * K)
            # Prefix (causal per anchor)
            mask[0, 0, row_slice, :prefix_len][attend_prefix] = 0.0
            # Own chain up to step k (inclusive)
            for j in range(k + 1):
                kv_cols = prefix_len + j * K + anchor_idx
                q_rows = k * K + anchor_idx
                mask[0, 0, q_rows, kv_cols] = 0.0
        return mask

    def forward(self, input_ids, attention_mask, loss_mask,
                sigmoid_coef=0.1, temperature=0.1,
                anchor='kl', aux_loss='v2',
                anchor_weight='none', aux_weight='pow08',
                on_policy_target=False,
                soft_rollout=False,
                chain_anchor_coef=1.0,
                rollout_truncate_at_reject=False,
                rollout_loss_mode='chain',
                baseline=None):
        # ── Normalize flags (legacy compat) ──
        if baseline == 'eagle_only':
            anchor, aux_loss = 'kl', 'none'
        elif baseline == 'ce_only':
            anchor, aux_loss = 'ce', 'none'
        if aux_loss == 'acceptance_length_v2':
            aux_loss = 'v2'
        if aux_loss == 'acceptance_length_v4':
            aux_loss = 'v4'
        assert anchor in ('kl', 'ce'), f"anchor must be kl|ce, got {anchor}"
        assert aux_loss in ('none', 'v2', 'v4', 'v5', 'tv', 'al_tv'), \
            f"aux_loss must be none|v2|v4|v5|tv|al_tv, got {aux_loss}"
        assert not (soft_rollout and on_policy_target), \
            "soft_rollout incompatible with on_policy_target (no hard chain to verify)"
        assert anchor_weight in ('none', 'uniform', 'pow08', 'dec', 'inc'), \
            f"anchor_weight must be none|uniform|pow08|dec|inc, got {anchor_weight}"
        assert aux_weight in ('uniform', 'pow08', 'dec', 'inc'), \
            f"aux_weight must be uniform|pow08|dec|inc, got {aux_weight}"
        assert rollout_loss_mode in ('chain', 'hybrid'), \
            f"rollout_loss_mode must be chain|hybrid, got {rollout_loss_mode}"

        B, L = input_ids.shape
        device = input_ids.device
        gamma = self.gamma

        # ── Target prefix forward (frozen) ──
        # Save KV cache when on_policy_target so we can reuse it in target verify.
        with torch.no_grad():
            target_prefix_out = self.target_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=on_policy_target,
                return_dict=True)
            target_lg = target_prefix_out.logits[:, :-1, :].float()
            target_argmax = target_lg.argmax(-1)
            target_p = F.softmax(target_lg, dim=-1)
            target_prefix_cache = (
                target_prefix_out.past_key_values if on_policy_target else None)

        mask = loss_mask[:, 1:].bool()
        nv = mask.float().sum()
        if nv.item() == 0:
            z = torch.tensor(0.0, device=device, requires_grad=True)
            return z, z.detach(), {
                "total_loss": 0.0, "kl_loss": 0.0, "v2_loss": 0.0,
                "num_valid": 0.0, "num_anchors": 0,
                "mean_tau": 0.0, "step_acc": [0.0] * gamma,
                "tau_hist": [0] * (gamma + 1),
                "aux_loss": 0.0, "eagle_loss": 0.0,
            }

        # ── Prefix draft forward (with cache for rollout) ──
        prefix_out = self.draft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        rollout_cache = prefix_out.past_key_values
        step_0_logits = prefix_out.logits

        dlg0 = step_0_logits[:, :-1, :].float()
        dlogp0 = F.log_softmax(dlg0, dim=-1)

        # ── Anchor per-position loss ──
        if anchor == 'kl':
            per_pos = -torch.sum(target_p * dlogp0, dim=-1)
        else:  # 'ce'
            per_pos = -dlogp0.gather(
                -1, target_argmax.unsqueeze(-1)).squeeze(-1)

        # ── Anchor aggregation ──
        # hybrid mode: sliding γ-window on the WHOLE prefix per_pos (legacy).
        # chain mode: anchor_loss is computed later, jointly with rollout chain
        #   (step 0 clean + step 1..γ-1 dirty), with γ-step weights. Sliding
        #   window over prefix is skipped — only the K anchor positions count.
        Lp_anchor = mask.shape[1]
        usable_anchor = Lp_anchor - gamma + 1
        if rollout_loss_mode == 'chain':
            anchor_loss = torch.tensor(0.0, device=device)
        elif anchor_weight == 'none' or usable_anchor <= 0:
            anchor_loss = (per_pos * mask.float()).sum() / (nv + 1e-8)
        else:
            per_pos_stack = torch.stack(
                [per_pos[:, k:k + usable_anchor] for k in range(gamma)], dim=-1)
            aw = _get_step_weights(
                anchor_weight, gamma, device, per_pos.dtype)
            anchor_win = (per_pos_stack * aw).sum(dim=-1)
            anchor_win_mask = mask[:, :usable_anchor]
            nv_anchor_win = anchor_win_mask.float().sum()
            anchor_loss = (anchor_win * anchor_win_mask.float()).sum() \
                / (nv_anchor_win + 1e-8)

        zero = torch.tensor(0.0, device=device)

        # NOTE: previously had an early-return when aux=none AND not ropo/soro,
        # which silently turned plain ro into noro. Removed — main_small_lm_rollout.py
        # always rolls out. ro KL-only now produces chain_anchor on the dirty chain,
        # which is the actual point of "ro" (rollout with clean target greedy as gt).
        step_0_argmax = dlg0.argmax(-1).detach()

        # ── Anchors: every assistant position that can support ≥1 rollout step ──
        anchor_pos = mask[0].nonzero(as_tuple=True)[0]
        anchor_pos = anchor_pos[anchor_pos <= L - 2]
        K = anchor_pos.shape[0]

        if K == 0:
            return anchor_loss, zero, {
                "total_loss": anchor_loss.item(),
                "kl_loss": anchor_loss.item(),
                "v2_loss": 0.0,
                "aux_loss": 0.0,
                "eagle_loss": anchor_loss.item(),
                "num_valid": nv.item(),
                "num_anchors": 0,
                "mean_tau": 0.0,
                "step_acc": [0.0] * gamma,
                "tau_hist": [0] * (gamma + 1),
            }

        rollout_lens = torch.clamp(L - 1 - anchor_pos, max=gamma)
        max_step = int(rollout_lens.max().item())

        # ── Rollout loop: collect draft logits and argmax per step ──
        # draft_argmax_chain[k] = [K] draft argmax at step k (step 0 = pre-loop)
        # draft_logits_chain[k-1] = [K, V] draft logits at step k (k ≥ 1)
        draft_argmax_chain = [step_0_argmax[0, anchor_pos]]
        draft_logits_chain = []

        if soft_rollout:
            # Under ZeRO-3 the embed weight is sharded. Access it only via the
            # layer's forward (which triggers the hook to gather+release), so
            # we use a top-K approximation: soft_emb ≈ Σ (normalized top-K
            # probs) · embed(top-K ids). K=64 captures >99% mass on a peaked
            # softmax; gradient flows through probs (via gather) and embed
            # (via hook). No GatheredParameters needed.
            embed_layer = self.draft_model.get_input_embeddings()
            soft_topk = 64
            draft_dtype = next(self.draft_model.parameters()).dtype

            def _soft_emb(logits_2d):
                p = F.softmax(logits_2d, dim=-1)
                tp, ti = p.topk(soft_topk, dim=-1)            # [K, topk]
                tp = tp / (tp.sum(-1, keepdim=True) + 1e-8)
                te = embed_layer(ti)                          # [K, topk, H]
                return (tp.unsqueeze(-1).to(te.dtype) * te
                        ).sum(1).unsqueeze(0)                 # [1, K, H]

            current_emb = _soft_emb(dlg0[0, anchor_pos]).to(draft_dtype)
            current_tokens = None
        else:
            current_tokens = step_0_argmax[0, anchor_pos].unsqueeze(0)

        for step in range(1, max_step):
            pos_ids = (anchor_pos + step).unsqueeze(0).long()
            attn_mask_4d = self._build_rollout_mask(
                anchor_pos, step, L, K, device)

            if soft_rollout:
                out = self.draft_model(
                    inputs_embeds=current_emb,
                    attention_mask=attn_mask_4d,
                    position_ids=pos_ids,
                    past_key_values=rollout_cache,
                    use_cache=True,
                    return_dict=True,
                )
            else:
                out = self.draft_model(
                    input_ids=current_tokens,
                    attention_mask=attn_mask_4d,
                    position_ids=pos_ids,
                    past_key_values=rollout_cache,
                    use_cache=True,
                    return_dict=True,
                )
            rollout_cache = out.past_key_values
            new_logits = out.logits.squeeze(0).float()           # [K, V]

            draft_logits_chain.append(new_logits)
            new_argmax = new_logits.argmax(-1)
            draft_argmax_chain.append(new_argmax.detach())

            if soft_rollout:
                current_emb = _soft_emb(new_logits).to(draft_dtype)
            else:
                current_tokens = new_argmax.detach().unsqueeze(0)

        # ── Optional: target verify on draft's dirty rollout chain ──
        # Gives target's argmax / distribution at each dirty-context position
        # for β_k / α_k (k ≥ 1). β_0 always uses clean (no dirty token yet).
        target_argmax_dirty = None
        target_p_dirty = None
        if on_policy_target and max_step >= 2:
            num_verify = max_step - 1
            # Stack first num_verify draft argmax: [num_verify, K]; flatten step-major
            verify_input = torch.stack(
                draft_argmax_chain[:num_verify], dim=0).reshape(1, -1).long()
            verify_pos_ids = torch.stack(
                [anchor_pos + (k + 1) for k in range(num_verify)],
                dim=0).reshape(1, -1).long()
            verify_mask = self._build_verify_mask(
                anchor_pos, num_verify, L, K,
                device, self.target_model.dtype)

            with torch.no_grad():
                verify_out = self.target_model(
                    input_ids=verify_input,
                    attention_mask=verify_mask,
                    position_ids=verify_pos_ids,
                    past_key_values=target_prefix_cache,
                    use_cache=False,
                    return_dict=True,
                )
            verify_logits = verify_out.logits.squeeze(0).float()  # [num_verify*K, V]
            target_argmax_dirty = verify_logits.argmax(-1).reshape(num_verify, K)
            # target_p_dirty (full softmax over dirty target) needed for any
            # full-distribution loss on the dirty chain: tv aux, OR KL anchor
            # via chain_anchor (ropo + anchor=kl, regardless of aux).
            # CE anchor and v2/v4/v5 sigmoid aux only need target_argmax_dirty.
            if aux_loss in ('tv', 'al_tv') or anchor == 'kl':
                target_p_dirty = F.softmax(
                    verify_logits, dim=-1).reshape(num_verify, K, -1)

        # ── Hard-accept list (always built; used by metrics + V5 truncation) ──
        hard_accept_list = [
            (step_0_argmax[0, anchor_pos] == target_argmax[0, anchor_pos]).float()
        ]
        for step in range(1, max_step):
            new_argmax_s = draft_logits_chain[step - 1].argmax(-1)
            if target_argmax_dirty is not None:
                ref = target_argmax_dirty[step - 1]
            else:
                t_idx = torch.clamp(anchor_pos + step, max=L - 2)
                ref = target_argmax[0, t_idx]
            hard_accept_list.append((new_argmax_s == ref).float().detach())

        step_idx = torch.arange(max_step, device=device).unsqueeze(0)
        valid_steps = step_idx < rollout_lens.unsqueeze(1)        # [K, max_step]

        # ── Chain rollout per-step losses (steps 1..max_step-1) ──
        # Always computed when there is a chain (max_step>=2). Used by both:
        #   hybrid mode: as additive chain_anchor on top of sliding-window prefix anchor
        #   chain mode:  as the tail (after step 0) of a unified γ-step anchor loss
        chain_anchor_loss = torch.tensor(0.0, device=device)
        chain_stack = None
        need_chain = (max_step >= 2) and (
            rollout_loss_mode == 'chain' or chain_anchor_coef > 0)
        if need_chain:
            chunk_K = 256
            per_step_chain = []
            for step in range(1, max_step):
                draft_logit_step = draft_logits_chain[step - 1]   # [K, V]
                if target_argmax_dirty is not None:
                    tgt_ids_step = target_argmax_dirty[step - 1]
                    tgt_p_step = (target_p_dirty[step - 1]
                                  if target_p_dirty is not None else None)
                else:
                    t_idx = torch.clamp(anchor_pos + step, max=L - 2)
                    tgt_ids_step = target_argmax[0, t_idx]
                    tgt_p_step = (target_p[0, t_idx]
                                  if anchor == 'kl' else None)

                loss_parts = []
                for i in range(0, K, chunk_K):
                    l_chunk = draft_logit_step[i:i + chunk_K]     # [≤256, V]
                    lp = F.log_softmax(l_chunk, dim=-1)
                    if anchor == 'kl':
                        assert tgt_p_step is not None
                        tp_c = tgt_p_step[i:i + chunk_K]
                        loss_parts.append(-(tp_c * lp).sum(-1))   # [≤256]
                    else:  # 'ce'
                        ti_c = tgt_ids_step[i:i + chunk_K]
                        loss_parts.append(-lp.gather(
                            -1, ti_c.unsqueeze(-1).long()).squeeze(-1))
                per_step_chain.append(torch.cat(loss_parts, dim=0))   # [K]

            chain_stack = torch.stack(per_step_chain, dim=-1)    # [K, max_step-1]

        if rollout_loss_mode == 'chain':
            # Unified γ-step anchor loss on the rollout chain (step 0 = clean
            # prefix prediction at anchor; step 1..γ-1 = dirty rollout). No
            # sliding window on the prefix outside anchor positions.
            # Reuse per_pos (already computed for the whole sequence) — avoid
            # allocating fresh [K, V] activations for step-0 KL/CE recompute.
            step_0_anchor_loss = per_pos[0, anchor_pos]            # [K]

            if max_step >= 2 and chain_stack is not None:
                all_step_anchor = torch.cat(
                    [step_0_anchor_loss.unsqueeze(-1), chain_stack],
                    dim=-1)                                       # [K, max_step]
            else:
                all_step_anchor = step_0_anchor_loss.unsqueeze(-1)  # [K, 1]

            n_steps_used = all_step_anchor.shape[-1]
            sched = anchor_weight if anchor_weight != 'none' else 'pow08'
            w_full = _get_step_weights(
                sched, gamma, device, all_step_anchor.dtype)
            w_used = w_full[:n_steps_used]                        # [n_steps_used]
            valid_all = valid_steps[:, :n_steps_used].to(all_step_anchor.dtype)

            if rollout_truncate_at_reject and n_steps_used >= 2:
                with torch.no_grad():
                    hard_stack_tmp = torch.stack(hard_accept_list, dim=-1)
                    reject_full = ((1.0 - hard_stack_tmp)
                                   * valid_steps.float()).long()
                    cumrej_full = reject_full.cumsum(-1)
                    zeros = torch.zeros_like(cumrej_full[..., :1])
                    prev_cumrej = torch.cat(
                        [zeros, cumrej_full[..., :-1]], dim=-1)
                    trunc_mask = (prev_cumrej[:, :n_steps_used] == 0
                                  ).to(all_step_anchor.dtype)
                valid_all = valid_all * trunc_mask

            anchor_loss = (all_step_anchor * w_used * valid_all
                           ).sum(-1).mean()
            # chain_anchor_coef ignored in chain mode (signal is unified)
        else:
            # hybrid mode: chain_anchor_loss adds to sliding-window prefix anchor
            if max_step >= 2 and chain_anchor_coef > 0 and chain_stack is not None:
                n_chain = max_step - 1
                sched = anchor_weight if anchor_weight != 'none' else 'pow08'
                w_full = _get_step_weights(
                    sched, gamma, device, chain_stack.dtype)
                w_chain = w_full[1:1 + n_chain]                   # skip step-0 weight
                valid_c = valid_steps[:, 1:1 + n_chain].to(chain_stack.dtype)
                if rollout_truncate_at_reject:
                    with torch.no_grad():
                        hard_stack_tmp = torch.stack(hard_accept_list, dim=-1)
                        reject_full = ((1.0 - hard_stack_tmp)
                                       * valid_steps.float()).long()
                        cumrej_full = reject_full.cumsum(-1)
                        zeros = torch.zeros_like(cumrej_full[..., :1])
                        prev_cumrej = torch.cat(
                            [zeros, cumrej_full[..., :-1]], dim=-1)
                        trunc_mask = (prev_cumrej[:, 1:1 + n_chain] == 0
                                      ).to(chain_stack.dtype)
                    valid_c = valid_c * trunc_mask
                chain_anchor_loss = (chain_stack * w_chain * valid_c
                                     ).sum(-1).mean()
            anchor_loss = anchor_loss + chain_anchor_coef * chain_anchor_loss

        # ── Aux loss (only if aux_loss != 'none') ──
        aux = zero
        if aux_loss != 'none':
            beta_variant = 'v4' if aux_loss == 'v5' else aux_loss
            # Step 0 signal (clean target at anchor position).
            if aux_loss in ('v2', 'v4', 'v5'):
                s0 = self._compute_beta(
                    dlg0, target_argmax, temperature,
                    variant=beta_variant)[0, anchor_pos]
            else:  # 'tv' or 'al_tv' — both use Σ min(p, q) as per-step β
                dq0 = F.softmax(dlg0[0, anchor_pos], dim=-1)
                tq0 = target_p[0, anchor_pos]
                s0 = torch.sum(torch.min(tq0, dq0), dim=-1)
            signals = [s0]
            for step in range(1, max_step):
                new_logits = draft_logits_chain[step - 1]
                if target_argmax_dirty is not None:
                    tgt_ids = target_argmax_dirty[step - 1]
                else:
                    t_idx = torch.clamp(anchor_pos + step, max=L - 2)
                    tgt_ids = target_argmax[0, t_idx]

                if aux_loss in ('v2', 'v4', 'v5'):
                    signals.append(self._compute_beta(
                        new_logits, tgt_ids, temperature, variant=beta_variant))
                else:  # 'tv' or 'al_tv'
                    dqk = F.softmax(new_logits, dim=-1)
                    if target_p_dirty is not None:
                        tqk = target_p_dirty[step - 1]
                    else:
                        t_idx = torch.clamp(anchor_pos + step, max=L - 2)
                        tqk = target_p[0, t_idx]
                    signals.append(torch.sum(torch.min(tqk, dqk), dim=-1))

            weights = _get_step_weights(
                aux_weight, max_step, device, torch.float32)
            sig_stack = torch.stack(signals, dim=-1)              # [K, max_step]

            if aux_loss in ('v2', 'v4', 'v5', 'al_tv'):
                beta_safe = torch.where(
                    valid_steps, sig_stack, torch.ones_like(sig_stack))
                cum_prod = torch.cumprod(beta_safe, dim=-1)
                step_gate = valid_steps.float()
                if aux_loss == 'v5' or rollout_truncate_at_reject:
                    with torch.no_grad():
                        hard_stack_tmp = torch.stack(hard_accept_list, dim=-1)
                        reject = ((1.0 - hard_stack_tmp) * step_gate).long()
                        cumrej = reject.cumsum(-1)
                        if rollout_truncate_at_reject:
                            # Strict: keep step k iff steps [0..k-1] no reject
                            zeros = torch.zeros_like(cumrej[..., :1])
                            prev_cumrej = torch.cat(
                                [zeros, cumrej[..., :-1]], dim=-1)
                            trunc_mask = (prev_cumrej == 0
                                          ).to(cum_prod.dtype)
                        else:
                            # V5 legacy: keep step k iff cumrej[k] <= 1
                            trunc_mask = (cumrej <= 1).to(cum_prod.dtype)
                    step_gate = step_gate * trunc_mask
                acc = (cum_prod * weights * step_gate).sum(dim=-1)
                aux = -acc.mean()
            else:  # 'tv'
                tv_per_step = 1.0 - sig_stack
                aux = (tv_per_step * weights
                       * valid_steps.float()).sum(dim=-1).mean()

        total_loss = anchor_loss + sigmoid_coef * aux

        # ── Metrics ──
        with torch.no_grad():
            hard_stack = torch.stack(hard_accept_list, dim=-1)    # [K, max_step]
            hard_cp = torch.cumprod(hard_stack, dim=-1) * valid_steps.float()
            tau_per_anchor = hard_cp.sum(-1)
            mean_tau = tau_per_anchor.mean().item()

            step_acc = []
            for step in range(gamma):
                if step < max_step:
                    vk = valid_steps[:, step]
                    if vk.any():
                        step_acc.append(hard_stack[:, step][vk].mean().item())
                    else:
                        step_acc.append(0.0)
                else:
                    step_acc.append(0.0)

            tau_int = tau_per_anchor.long().clamp(max=gamma)
            tau_hist = [(tau_int == v).sum().item() for v in range(gamma + 1)]

        return total_loss, aux.detach(), {
            "total_loss": total_loss.item(),
            "kl_loss": anchor_loss.item(),
            "v2_loss": aux.item() if aux_loss in ('v2', 'v4', 'v5') else 0.0,
            "aux_loss": aux.item(),
            "eagle_loss": anchor_loss.item(),
            "mean_tau": mean_tau,
            "step_acc": step_acc,
            "tau_hist": tau_hist,
            "num_valid": nv.item(),
            "num_anchors": K,
        }
