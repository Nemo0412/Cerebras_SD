"""
Layer-Skip Draft Model for Speculative Decoding.

Wraps a small LM (e.g., Qwen3-0.6B) as a draft. Taps hidden states at one or
more intermediate transformer layers and feeds each through the model's own
final RMSNorm + LM head to obtain per-exit logits. Standard layer-skip
recipe (LayerSkip, CALM).

Exit layer indexing is 1-based: `exit_layer = k` means "logits computed from
the hidden state AFTER the first k transformer blocks". `exit_layer = N`
(N = num_layers) is the normal full-model output.

Loss options (matching sdpo/small_lm_model.py, applied per-exit and averaged):

  baseline='eagle_only':
    L_e = KL(target || draft_e)                                # pure distillation

  baseline='ce_only':
    L_e = -log q_e(target_argmax)                              # hard cross entropy

  aux_loss='acceptance_length_v2' (default):
    β_e(t) = σ((z_e[target_argmax] - max z_e) / T) + 0.5
    acc_e  = Σ_k 0.8^k · ∏_{j=0..k} β_e(t+j)                   # over γ window
    L_e    = KL(target || draft_e) + coef · (-acc_e)

  aux_loss='tv':
    tv_e(t) = 1 - Σ_v min(target_p(v), draft_e_p(v))
    L_e     = KL(target || draft_e) + coef · Σ_k 0.8^k · tv_e(t+k)

Final loss = mean over exit_layers of L_e. Per-exit metrics are reported in
addition to the mean, so training logs can show which exits are catching up.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


class LayerSkipBackbone(nn.Module):
    """Single small-LM backbone exposing multiple intermediate exits."""

    def __init__(self, model_path, exit_layers, dtype=torch.float16,
                 attn_implementation=None):
        super().__init__()
        kwargs = {"torch_dtype": dtype}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self.base = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        self.num_layers = len(self.base.model.layers)
        self.exit_layers = sorted(set(int(e) for e in exit_layers))
        for e in self.exit_layers:
            if e < 1 or e > self.num_layers:
                raise ValueError(
                    f"exit_layer {e} out of range [1, {self.num_layers}]")

    def forward_with_exits(self, input_ids, attention_mask=None):
        """Return {exit_layer: logits[B, L, V]} for each requested exit.

        IMPORTANT: HF Qwen3 (and similar models) stores intermediate
        `hidden_states[1..N-1]` as raw layer outputs BEFORE `self.norm`, but
        `hidden_states[N]` (the last entry) is replaced at return time with
        `last_hidden_state`, which is AFTER `self.norm`. So we must apply norm
        ONLY for intermediate exits, and skip it when exit_layer == num_layers
        to avoid double-norming (which produces garbage logits).
        """
        outputs = self.base.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hs_all = outputs.hidden_states  # len == num_layers + 1
        results = {}
        for e in self.exit_layers:
            h = hs_all[e]
            if e < self.num_layers:
                h = self.base.model.norm(h)  # intermediate: not yet normed
            # else: final layer already normed by the model
            results[e] = self.base.lm_head(h)
        return results


def _per_exit_loss(draft_lg, target_p, target_argmax, mask, gamma,
                   sigmoid_coef, temperature, baseline, aux_loss):
    """Compute loss + metrics for a single draft exit.

    Ported from small_lm_model.py:forward with one draft replaced by
    `draft_lg` (already :-1 sliced and float-cast).

    Returns:
        total_loss (scalar), aux_val (scalar), metrics_dict
    """
    device = draft_lg.device
    nv = mask.float().sum()

    draft_logp = F.log_softmax(draft_lg, dim=-1)
    kl_per_pos = -torch.sum(target_p * draft_logp, dim=-1)
    kl_loss = (kl_per_pos * mask.float()).sum() / (nv + 1e-8)

    zero_metrics_stub = {
        "step_acc": [0.0] * gamma,
        "tau_hist": [0] * (gamma + 1),
    }

    if baseline == 'eagle_only':
        return kl_loss, torch.tensor(0.0, device=device), {
            "aux_loss": 0.0, "eagle_loss": kl_loss.item(),
            "mean_tau": 0.0, "num_valid": nv.item(),
            **zero_metrics_stub,
        }

    if baseline == 'ce_only':
        ce_per_pos = -draft_logp.gather(
            -1, target_argmax.unsqueeze(-1)).squeeze(-1)
        ce_loss = (ce_per_pos * mask.float()).sum() / (nv + 1e-8)
        return ce_loss, torch.tensor(0.0, device=device), {
            "aux_loss": 0.0, "eagle_loss": ce_loss.item(),
            "mean_tau": 0.0, "num_valid": nv.item(),
            **zero_metrics_stub,
        }

    # --- TV loss branch ---
    if aux_loss == 'tv':
        draft_q = F.softmax(draft_lg, dim=-1)
        alpha = torch.sum(torch.min(target_p, draft_q), dim=-1)
        tv_per_pos = 1.0 - alpha

        with torch.no_grad():
            hard_accept = (draft_lg.argmax(dim=-1) == target_argmax)

        Lp_tv = mask.shape[1]
        usable_tv = Lp_tv - gamma + 1
        if usable_tv <= 0:
            return kl_loss, torch.tensor(0.0, device=device), {
                "aux_loss": 0.0, "eagle_loss": kl_loss.item(),
                "mean_tau": 0.0, "num_valid": 0.0, **zero_metrics_stub,
            }

        tv_stack = torch.stack(
            [tv_per_pos[:, k:k + usable_tv] for k in range(gamma)], dim=-1)
        accept_stack_tv = torch.stack(
            [hard_accept[:, k:k + usable_tv] for k in range(gamma)], dim=-1)
        win_mask_tv = mask[:, :usable_tv]

        tv_weights = torch.tensor([0.8 ** k for k in range(gamma)],
                                  device=device, dtype=tv_stack.dtype)
        weighted_tv = (tv_stack * tv_weights).sum(dim=-1)

        nv_acc_tv = win_mask_tv.float().sum()
        tv_loss = (weighted_tv * win_mask_tv.float()).sum() / (nv_acc_tv + 1e-8)
        total_loss = kl_loss + sigmoid_coef * tv_loss

        with torch.no_grad():
            hard_cp = torch.cumprod(accept_stack_tv.float(), dim=-1)
            tau = hard_cp.sum(dim=-1)
            mean_tau_val = (tau * win_mask_tv.float()).sum() / (nv_acc_tv + 1e-8)
            step_acc = [accept_stack_tv[:, :, k][win_mask_tv].float().mean().item()
                        if win_mask_tv.any() else 0.0 for k in range(gamma)]
            tau_int = tau[win_mask_tv].long().clamp(max=gamma)
            tau_hist = [(tau_int == v).sum().item() for v in range(gamma + 1)]

        return total_loss, tv_loss, {
            "aux_loss": tv_loss.item(),
            "eagle_loss": kl_loss.item(),
            "mean_tau": mean_tau_val.item(),
            "num_valid": nv_acc_tv.item(),
            "step_acc": step_acc,
            "tau_hist": tau_hist,
        }

    # --- Default: KL + sigmoid_coef * V2 acceptance length ---
    z_target = draft_lg.gather(-1, target_argmax.unsqueeze(-1)).squeeze(-1)
    z_max = draft_lg.max(dim=-1).values
    gap = (z_target - z_max) / temperature
    beta = (torch.sigmoid(gap) + 0.5).clamp(max=1.0)

    with torch.no_grad():
        hard_accept = (draft_lg.argmax(dim=-1) == target_argmax)

    Lp = mask.shape[1]
    usable = Lp - gamma + 1
    if usable <= 0:
        return kl_loss, torch.tensor(0.0, device=device), {
            "aux_loss": 0.0, "eagle_loss": kl_loss.item(),
            "mean_tau": 0.0, "num_valid": 0.0, **zero_metrics_stub,
        }

    beta_stack = torch.stack(
        [beta[:, k:k + usable] for k in range(gamma)], dim=-1)
    accept_stack = torch.stack(
        [hard_accept[:, k:k + usable] for k in range(gamma)], dim=-1)
    win_mask = mask[:, :usable]

    cum_prod = torch.cumprod(beta_stack, dim=-1)
    weights = torch.tensor([0.8 ** k for k in range(gamma)],
                           device=device, dtype=cum_prod.dtype)
    weighted = cum_prod * weights
    acc_length = weighted.sum(dim=-1)

    nv_acc = win_mask.float().sum()
    acc_loss = -(acc_length * win_mask.float()).sum() / (nv_acc + 1e-8)
    total_loss = kl_loss + sigmoid_coef * acc_loss

    with torch.no_grad():
        hard_cp = torch.cumprod(accept_stack.float(), dim=-1)
        tau = hard_cp.sum(dim=-1)
        mean_tau = (tau * win_mask.float()).sum() / (nv_acc + 1e-8)
        step_acc = [accept_stack[:, :, k][win_mask].float().mean().item()
                    if win_mask.any() else 0.0 for k in range(gamma)]
        tau_int = tau[win_mask].long().clamp(max=gamma)
        tau_hist = [(tau_int == v).sum().item() for v in range(gamma + 1)]

    return total_loss, acc_loss, {
        "aux_loss": acc_loss.item(),
        "eagle_loss": kl_loss.item(),
        "mean_tau": mean_tau.item(),
        "num_valid": nv_acc.item(),
        "step_acc": step_acc,
        "tau_hist": tau_hist,
    }


class LayerSkipDraftModel(nn.Module):
    """Training wrapper: frozen target + trainable layer-skip draft.

    forward(...) computes the same SDPO-style losses as SmallLMDraftModel, but
    applied to each exit layer independently, then averaged.

    Returns (total_loss, aux_val, metrics) where metrics contains both a
    mean-over-exits summary and a `per_exit` dict keyed by exit layer.
    """

    def __init__(self, target_path, draft_path, exit_layers,
                 gamma=7, dtype=torch.float16):
        super().__init__()
        self.gamma = gamma

        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_path, torch_dtype=dtype)
        self.target_model.eval()
        for p in self.target_model.parameters():
            p.requires_grad = False

        self.draft = LayerSkipBackbone(draft_path, exit_layers, dtype=dtype)
        self.exit_layers = self.draft.exit_layers

    def train(self, mode=True):
        super().train(mode)
        self.target_model.eval()
        self.draft.train(mode)
        return self

    def forward(self, input_ids, attention_mask, loss_mask,
                sigmoid_coef=0.1, temperature=1.0, baseline=None,
                aux_loss='acceptance_length_v2'):
        B, L = input_ids.shape
        device = input_ids.device
        mask = loss_mask[:, 1:].bool()
        gamma = self.gamma

        with torch.no_grad():
            target_lg = self.target_model(
                input_ids=input_ids,
                attention_mask=attention_mask).logits[:, :-1, :].float()
            target_argmax = target_lg.argmax(dim=-1)
            target_p = F.softmax(target_lg, dim=-1)

        logits_per_exit = self.draft.forward_with_exits(input_ids, attention_mask)

        # Per-exit loss and metrics
        per_exit_total = {}
        per_exit_aux = {}
        per_exit_metrics = {}
        total_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
        aux_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
        n_exits = len(self.exit_layers)

        for e in self.exit_layers:
            draft_lg_e = logits_per_exit[e][:, :-1, :].float()
            total_e, aux_e, m_e = _per_exit_loss(
                draft_lg_e, target_p, target_argmax, mask, gamma,
                sigmoid_coef, temperature, baseline, aux_loss)
            per_exit_total[e] = total_e
            per_exit_aux[e] = aux_e
            per_exit_metrics[e] = m_e
            total_sum = total_sum + total_e
            aux_sum = aux_sum + aux_e

        total_loss = total_sum / n_exits
        aux_val = aux_sum / n_exits

        # Aggregate metrics: mean over exits for scalars, element-wise mean
        # for step_acc, sum for tau_hist (counts).
        def _mean(key):
            return sum(per_exit_metrics[e][key] for e in self.exit_layers) / n_exits

        step_acc_mean = [
            sum(per_exit_metrics[e]["step_acc"][k] for e in self.exit_layers) / n_exits
            for k in range(gamma)
        ]
        tau_hist_sum = [
            sum(per_exit_metrics[e]["tau_hist"][v] for e in self.exit_layers)
            for v in range(gamma + 1)
        ]
        num_valid_max = max(per_exit_metrics[e]["num_valid"] for e in self.exit_layers)

        return total_loss, aux_val, {
            "aux_loss": _mean("aux_loss"),
            "eagle_loss": _mean("eagle_loss"),
            "mean_tau": _mean("mean_tau"),
            "num_valid": num_valid_max,
            "step_acc": step_acc_mean,
            "tau_hist": tau_hist_sum,
            "per_exit": per_exit_metrics,
        }
