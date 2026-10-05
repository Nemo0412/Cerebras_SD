"""
Layer-Skip Draft Model with PER-EXIT ON-POLICY ROLLOUT Training.

Each exit layer generates rollout tokens using ITS OWN hidden-state logits
(not the deepest exit's). All exits share the same backbone and run through
the same forward pass, but after each forward, each exit picks its own argmax
from its own layer's hidden state. This creates N_exits independent rollout
chains from the same set of K anchor positions.

Architecture: shared prefix KV cache + 4D attention mask. At each rollout
step, N_exits × K new tokens are appended to the cache. The 4D mask isolates
each (exit, anchor) pair — it can only attend to its own prefix slice and its
own prior rollout tokens.

Total forward calls per sample: γ (1 prefix + γ-1 rollout steps). FIXED
regardless of anchor count or exit count → ZeRO-3 compatible.

Memory: KV cache grows by N_exits × K per step. For 4 exits × 500 anchors
× 7 steps ≈ 1.6 GB cache growth. Manageable.

Requires attn_implementation="sdpa" on the draft model.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(__file__))
from layer_skip_model import LayerSkipBackbone

_DEBUG = os.environ.get("DEBUG_ROLLOUT", "0") == "1"


def _dbg(msg):
    if _DEBUG and int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0))) == 0:
        print(f"[rollout-dbg] {msg}", flush=True)


class LayerSkipRolloutModel(nn.Module):

    def __init__(self, target_path, draft_path, exit_layers,
                 gamma=7, dtype=torch.float16):
        super().__init__()
        self.gamma = gamma
        self._debug_tokenizer = None
        self._debug_count = 0

        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_path, torch_dtype=dtype)
        self.target_model.eval()
        for p in self.target_model.parameters():
            p.requires_grad = False

        self.draft = LayerSkipBackbone(
            draft_path, exit_layers, dtype=dtype,
            attn_implementation="sdpa")
        self.exit_layers = self.draft.exit_layers
        self.num_layers = self.draft.num_layers
        self.n_exits = len(self.exit_layers)

    def train(self, mode=True):
        super().train(mode)
        self.target_model.eval()
        self.draft.train(mode)
        return self

    def set_debug_tokenizer(self, tokenizer):
        self._debug_tokenizer = tokenizer

    def _logits_from_hidden(self, hidden_states, exit_layer):
        h = hidden_states
        if exit_layer < self.num_layers:
            h = self.draft.base.model.norm(h)
        return self.draft.base.lm_head(h)

    def _compute_beta(self, logits, target_ids, temperature):
        z_t = logits.gather(-1, target_ids.unsqueeze(-1).long()).squeeze(-1)
        z_m = logits.max(-1).values
        return (torch.sigmoid((z_t - z_m) / temperature) + 0.5).clamp(max=1.0)

    # ------------------------------------------------------------------
    def _build_rollout_mask(self, anchor_pos, step_k, prefix_len, K, device):
        """4D mask for per-exit rollout. Each (exit, anchor) pair is isolated.

        Token layout per step: [exit_0_anchor_0..K-1, exit_1_anchor_0..K-1, ...]
        Total new tokens per step: NK = N_exits × K.
        Total KV length: prefix_len + step_k × NK.

        Query (e_idx, k_idx) → q_index = e_idx * K + k_idx
        Attends to:
          prefix [0..t_k]
          own rollout at [prefix_len + j*NK + e_idx*K + k_idx  for j in 0..step_k-1]
        """
        NE = self.n_exits
        NK = NE * K
        total_kv = prefix_len + step_k * NK
        dtype = self.draft.base.model.dtype
        min_val = torch.finfo(dtype).min
        mask = torch.full((1, 1, NK, total_kv), min_val, device=device, dtype=dtype)

        # Prefix: anchor k attends to [0..t_k], same for all exits
        col = torch.arange(prefix_len, device=device)
        attend_prefix = col.unsqueeze(0) <= anchor_pos.unsqueeze(1)  # [K, L]
        for e_idx in range(NE):
            off = e_idx * K
            mask[0, 0, off:off + K, :prefix_len][attend_prefix] = 0.0

        # Rollout: each (exit, anchor) attends to its own chain
        anchor_idx = torch.arange(K, device=device)
        for e_idx in range(NE):
            q_idx = e_idx * K + anchor_idx  # [K]
            for j in range(step_k):
                kv_pos = prefix_len + j * NK + e_idx * K + anchor_idx
                mask[0, 0, q_idx, kv_pos] = 0.0

        return mask

    # ------------------------------------------------------------------
    def forward(self, input_ids, attention_mask, loss_mask,
                sigmoid_coef=0.1, temperature=0.1,
                baseline=None, aux_loss='acceptance_length_v2'):
        B, L = input_ids.shape
        device = input_ids.device
        gamma = self.gamma
        NE = self.n_exits

        # ── 1. Target forward ──
        with torch.no_grad():
            target_lg = self.target_model(
                input_ids=input_ids,
                attention_mask=attention_mask).logits[:, :-1, :].float()
            target_argmax = target_lg.argmax(-1)
            target_p = F.softmax(target_lg, dim=-1)

        mask = loss_mask[:, 1:].bool()
        nv = mask.float().sum()
        if nv.item() == 0:
            z = torch.tensor(0.0, device=device, requires_grad=True)
            return z, z.detach(), {"total_loss": 0, "kl_loss": 0,
                                   "v2_loss": 0, "num_valid": 0, "num_anchors": 0}

        # ── 2. Prefix draft forward ──
        prefix_out = self.draft.base.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        prefix_hs = prefix_out.hidden_states
        rollout_cache = prefix_out.past_key_values

        # ── 3. Step-0 logits per exit ──
        step_0_logits = {}
        for e in self.exit_layers:
            step_0_logits[e] = self._logits_from_hidden(prefix_hs[e], e)

        # ── 4. KL loss (teacher-forced, all positions) ──
        kl_per_exit = {}
        for e in self.exit_layers:
            dlg = step_0_logits[e][:, :-1, :].float()
            dlogp = F.log_softmax(dlg, dim=-1)
            kl_pp = -torch.sum(target_p * dlogp, dim=-1)
            kl_per_exit[e] = (kl_pp * mask.float()).sum() / (nv + 1e-8)
        kl_loss = sum(kl_per_exit.values()) / NE

        # ── 5. Step-0 β and argmax PER EXIT ──
        step_0_beta = {}
        step_0_argmax_per_exit = {}
        for e in self.exit_layers:
            dlg = step_0_logits[e][:, :-1, :].float()
            step_0_beta[e] = self._compute_beta(dlg, target_argmax, temperature)
            step_0_argmax_per_exit[e] = dlg.argmax(-1).detach()  # [1, L-1]

        # ── 6. Find anchors ──
        anchor_pos = mask[0].nonzero(as_tuple=True)[0]
        anchor_pos = anchor_pos[anchor_pos <= L - 2]
        K = anchor_pos.shape[0]

        _dbg(f"L={L} K={K} NE={NE} NK={NE*K} gamma={gamma}")

        if K == 0:
            return kl_loss, torch.tensor(0.0, device=device), {
                "total_loss": kl_loss.item(), "kl_loss": kl_loss.item(),
                "v2_loss": 0.0, "num_valid": nv.item(), "num_anchors": 0}

        rollout_lens = torch.clamp(L - 1 - anchor_pos, max=gamma)
        max_step = int(rollout_lens.max().item())

        # Collect step-0 β per exit (each exit from its own logits)
        all_betas = {e: [step_0_beta[e][0, anchor_pos]] for e in self.exit_layers}

        # ── 7. Build seed tokens: NK = NE × K ──
        # Order: [exit_0's K anchors, exit_1's K anchors, ..., exit_{NE-1}'s K]
        seed_list = []
        for e in self.exit_layers:
            seed_list.append(step_0_argmax_per_exit[e][0, anchor_pos])  # [K]
        current_tokens = torch.cat(seed_list).unsqueeze(0)  # [1, NK]

        NK = NE * K

        # Store rollout tokens per exit for debug
        _rollout_tokens = {e: [step_0_argmax_per_exit[e][0, anchor_pos].detach().clone()]
                           for e in self.exit_layers}

        # ── 8. Rollout steps 1..max_step-1 ──
        for step in range(1, max_step):
            # Position IDs: each exit's K anchors have same positions
            step_pos = anchor_pos + step  # [K]
            pos_ids = step_pos.repeat(NE).unsqueeze(0).long()  # [1, NK]

            attn_mask_4d = self._build_rollout_mask(
                anchor_pos, step, L, K, device)

            _dbg(f"step={step} tokens={current_tokens.shape} "
                 f"mask={attn_mask_4d.shape} cache={rollout_cache.get_seq_length()}")

            out = self.draft.base.model(
                input_ids=current_tokens,
                attention_mask=attn_mask_4d,
                position_ids=pos_ids,
                past_key_values=rollout_cache,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            rollout_cache = out.past_key_values
            new_hs = out.hidden_states  # tuple of [1, NK, H]

            # Target for β: same for all exits at anchor k
            max_idx = L - 2
            t_idx = torch.clamp(anchor_pos + step, max=max_idx)
            tgt_ids = target_argmax[0, t_idx]  # [K]

            # Per-exit: extract own hidden state, compute β, get next token
            next_tokens_list = []
            for e_idx, e in enumerate(self.exit_layers):
                start = e_idx * K
                end = start + K
                h_e = new_hs[e][:, start:end, :]  # [1, K, H]
                logits_e = self._logits_from_hidden(h_e, e).squeeze(0).float()  # [K, V]

                beta_e = self._compute_beta(logits_e, tgt_ids, temperature)
                all_betas[e].append(beta_e)

                next_tok = logits_e.argmax(-1).detach()  # [K]
                next_tokens_list.append(next_tok)
                _rollout_tokens[e].append(next_tok.clone())

            current_tokens = torch.cat(next_tokens_list).unsqueeze(0)  # [1, NK]

        # ── 8b. Debug ──
        if _DEBUG and self._debug_tokenizer is not None and self._debug_count < 5:
            self._debug_count += 1
            tok = self._debug_tokenizer
            n_show = min(2, K)
            print(f"\n{'='*80}")
            print(f"[rollout-tokens] {n_show}/{K} anchors, {NE} exits, call #{self._debug_count}")
            for ai in range(n_show):
                t = anchor_pos[ai].item()
                rlen = int(rollout_lens[ai].item())
                pstart = max(0, t - 9)
                prefix_text = tok.decode(input_ids[0, pstart:t+1].tolist(),
                                         skip_special_tokens=False)
                print(f"\n  anchor[{ai}] t={t} rollout_len={rlen}")
                print(f"    prefix: ...{repr(prefix_text[-50:])}")
                for e_idx, e in enumerate(self.exit_layers):
                    print(f"    --- exit={e} ---")
                    print(f"    {'step':>5}  {'actual':>12}  {'tgt_argmax':>12}  {'draft':>12}  match")
                    for s in range(min(rlen, len(_rollout_tokens[e]))):
                        pos = t + 1 + s
                        gt = tok.decode([input_ids[0, pos].item()]) if pos < L else "OOB"
                        tgt = tok.decode([target_argmax[0, min(t+s, L-2)].item()])
                        dft = tok.decode([_rollout_tokens[e][s][ai].item()])
                        m = "✓" if target_argmax[0, min(t+s, L-2)].item() == _rollout_tokens[e][s][ai].item() else "✗"
                        if pos < L and input_ids[0, pos].item() == target_argmax[0, min(t+s, L-2)].item():
                            gt_m = "(=gt)"
                        else:
                            gt_m = "(≠gt)"
                        print(f"    {s:>5}  {repr(gt):>12}  {repr(tgt):>12} {gt_m}  {repr(dft):>12}  {m}")
            print(f"{'='*80}\n")

        # ── 9. V2 loss per exit ──
        weights = torch.tensor([0.8 ** j for j in range(max_step)],
                               device=device, dtype=torch.float32)
        step_idx = torch.arange(max_step, device=device).unsqueeze(0)
        valid_steps = step_idx < rollout_lens.unsqueeze(1)  # [K, max_step]

        v2_per_exit = {}
        for e in self.exit_layers:
            beta_stack = torch.stack(all_betas[e], dim=-1)  # [K, max_step]
            beta_safe = torch.where(valid_steps, beta_stack,
                                    torch.ones_like(beta_stack))
            cum_prod = torch.cumprod(beta_safe, dim=-1)
            acc = (cum_prod * weights * valid_steps.float()).sum(dim=-1)
            v2_per_exit[e] = -acc.mean()

        v2_loss = sum(v2_per_exit.values()) / NE

        total_loss = kl_loss + sigmoid_coef * v2_loss

        _dbg(f"kl={kl_loss.item():.4f} v2={v2_loss.item():.4f} "
             f"total={total_loss.item():.4f}")

        return total_loss, v2_loss.detach(), {
            "total_loss": total_loss.item(),
            "kl_loss": kl_loss.item(),
            "v2_loss": v2_loss.item(),
            "num_valid": nv.item(),
            "num_anchors": K,
            "per_exit": {e: {"kl": kl_per_exit[e].item(),
                             "v2": v2_per_exit[e].item()}
                         for e in self.exit_layers},
        }
