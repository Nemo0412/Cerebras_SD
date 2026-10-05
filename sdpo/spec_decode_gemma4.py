"""Speculative decoding framework for Gemma 4.

Target: google/gemma-4-31B-it (60 layers, sliding+full hybrid attention, sliding_window=1024)
Draft:  google/gemma-4-E2B-it  (35 layers, sliding+full hybrid, sliding_window=512,
                                kv_shared_layers=20 — last 20 layers share KV with earlier layers)

Key facts (verified):
- Both share tokenizer (vocab_size=262144) → no draft↔target vocab mapping needed.
- Both use DynamicCache (not HybridCache); cache extension follows standard HF API.
- Per-layer attention mask dispatch: model internally maps layer_type → mask, but if a 4D
  attention_mask is supplied, HF skips its internal builder → we control exactly which
  tokens attend to which. For γ=7 tree depth, sliding window (≥512) is far beyond reach,
  so we don't need separate sliding/full masks at our level — the 4D tree-causal mask
  works for both.
- chat_template supports enable_thinking=True (injects <|think|> in system prompt).

This module implements:
  chain_decode():  γ-step draft rollout + parallel verify. Baseline, no tree.
  tree_decode():   top-k branching tree (γ=7, top_k=[4,3,2,1,1,1,1], budget=128).
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    Gemma4ForCausalLM,
    Gemma4ForConditionalGeneration,
    DynamicCache,
)
from transformers.modeling_outputs import CausalLMOutputWithPast


class Gemma4TextCausalWrapper(nn.Module):
    """Thin wrapper exposing text-only causal-LM forward from a multimodal checkpoint.

    Gemma4ForConditionalGeneration.__init__ builds a Gemma4Model (which contains
    `language_model: Gemma4TextModel` + vision + audio towers) plus a top-level
    `lm_head`. There is NO `language_model` attribute on the conditional class
    itself, and the standalone `Gemma4ForCausalLM` checkpoint does not exist for
    these IDs. Calling Gemma4Model.forward directly invokes multimodal logic
    we don't need.

    This wrapper holds direct references (NOT copies) to:
      - `self.model = cond.model.language_model`  (Gemma4TextModel body)
      - `self.lm_head = cond.lm_head`             (vocab projection)
    Parameters are shared with the underlying conditional model.

    `forward(input_ids, past_key_values, use_cache)` returns a SimpleNamespace
    with `.logits` and `.past_key_values`, matching the contract chain_decode
    uses (and matching Gemma4ForCausalLM's output shape).
    """

    def __init__(self, conditional_gen: Gemma4ForConditionalGeneration):
        super().__init__()
        self.model = conditional_gen.model.language_model
        self.config = conditional_gen.config.text_config
        # Gemma 4 ties lm_head ↔ embed_tokens (tie_word_embeddings=True). Holding
        # `self.lm_head = cond.lm_head` as a SEPARATE child module confuses
        # DeepSpeed ZeRO-3 — even though the underlying nn.Parameter is the
        # same object, ZeRO-3's per-module gather/release hooks treat the lm_head
        # module independently and end up using a stale/wrong gather of the
        # shared embedding matrix → garbage logits and mean_tau=0 across all
        # positions. Workaround: don't register lm_head as a child; perform the
        # output projection inline against embed_tokens.weight (same storage).
        # If a future Gemma checkpoint has UNTIED lm_head we'll need a guard.
        assert getattr(conditional_gen.config.text_config,
                        "tie_word_embeddings", False), \
            "Gemma4TextCausalWrapper assumes tied lm_head ↔ embed_tokens."

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None, **kw):
        out = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache, **kw)
        logits = F.linear(out.last_hidden_state,
                          self.model.embed_tokens.weight)
        # Use HF dataclass so DeepSpeed ZeRO-3 hooks can introspect output tensors
        # (SimpleNamespace triggers "unknown inputs or outputs type" warning and
        # prevents the post-backward hook from releasing gathered params).
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=getattr(out, "past_key_values", past_key_values))

GAMMA_DEFAULT = 7
TOP_K_DEFAULT = (4, 3, 2, 1, 1, 1, 1)
BUDGET_DEFAULT = 128


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _maybe_remap_gemma4_ckpt(path: str) -> None:
    """In-place fix for DeepSpeed-saved Gemma 4 draft ckpts.

    Training saves the wrapper's params with two prefix issues:
      1. DeepSpeed adds the top-level module name → `draft_model.`
      2. Inside the wrapper `self.model = cond.model.language_model`, so saved
         keys look like `draft_model.model.layers.0.*` — but ConditionalGen
         expects `model.language_model.layers.0.*`.

    Convert in-place. Idempotent: re-running on an already-remapped ckpt is
    a no-op (no `draft_model.` prefix, no `model.*` keys that aren't already
    `model.language_model.*`).
    """
    import os
    import glob
    if not os.path.isdir(path):
        return
    for f in glob.glob(os.path.join(path, "*.bin")):
        state = torch.load(f, map_location="cpu")
        new = {}
        changed = False
        for k, v in state.items():
            nk = k
            if nk.startswith("draft_model."):
                nk = nk.removeprefix("draft_model.")
                changed = True
            if nk.startswith("model.") and not nk.startswith("model.language_model."):
                nk = "model.language_model." + nk.removeprefix("model.")
                changed = True
            new[nk] = v
        if changed:
            print(f"[GEMMA4-LOAD] Remapped ckpt {f} (draft_model+language_model prefix fix)")
            torch.save(new, f)


def load_text_model(name: str, device: str | int = "cuda:0",
                    dtype=torch.bfloat16):
    """Load a Gemma 4 model in text-only mode on a SINGLE device.

    Gemma 4 ships as Gemma4ForConditionalGeneration. The checkpoint state_dict
    uses keys like `model.language_model.layers.*`, so loading Gemma4ForCausalLM
    directly produces random-weight init (all keys MISSING) — `from_pretrained`
    doesn't raise; it warns only. We MUST load via ConditionalGeneration and
    unwrap to `.language_model`, which IS a Gemma4ForCausalLM with weights
    correctly resolved against the checkpoint's `model.language_model.*` prefix.

    Single-device placement avoids the device_map='auto' split issue where
    embed_tokens (used inside our manual forward) ends up on a different GPU
    from input_ids.

    If `name` is a directory (trained draft ckpt), the loader applies an
    idempotent prefix remap before from_pretrained — see
    `_maybe_remap_gemma4_ckpt`.
    """
    _maybe_remap_gemma4_ckpt(name)
    cond = Gemma4ForConditionalGeneration.from_pretrained(
        name, dtype=dtype, device_map={"": device},
        attn_implementation="sdpa")
    return Gemma4TextCausalWrapper(cond)


# ---------------------------------------------------------------------------
# Chain spec decode (Phase A)
# ---------------------------------------------------------------------------
@dataclass
class ChainResult:
    output_ids: torch.LongTensor          # full sequence including prompt
    n_generated: int
    n_rounds: int
    total_accepted: int                    # sum of γ-step accepts across rounds
    mean_alpha: float                      # accepted / rounds
    elapsed: float

@torch.inference_mode()
def chain_decode(
    target,
    draft,
    input_ids: torch.LongTensor,          # [1, L]
    max_new_tokens: int = 256,
    gamma: int = GAMMA_DEFAULT,
    temperature: float = 0.0,
    draft_mode: str = "argmax",
    verify_mode: str = "auto",
    eos_token_id: Optional[int] = None,
) -> ChainResult:
    """γ-step draft rollout + parallel target verify.

    Each round:
      1. Draft generates γ tokens autoregressively (greedy or multinomial).
      2. Target runs ONE forward pass over [last_accepted_token, drafted γ tokens]
         producing logits at γ+1 positions.
      3. Verify: argmax (T=0) or r ≤ p_target / r ≤ p_target/p_draft.
      4. Accept up to first reject. Append target's correction at reject position
         (or one bonus token if all γ accepted).

    Sliding window note: even at max_new_tokens=256 and γ=7, sequence length grows
    by 7 per round → at most ~250 tokens added before EOS / cap. Far below
    sliding_window (≥512 for E2B, ≥1024 for 31B), so attention coverage is full.
    """
    if verify_mode == "auto":
        verify_mode = "greedy" if temperature == 0.0 else "simple"

    # Each model lives on its own device. Maintain device-specific copies of
    # tensors and .to() before every forward.
    t_dev = target.model.embed_tokens.weight.device
    d_dev = draft.model.embed_tokens.weight.device
    B, prefix_len = input_ids.shape
    assert B == 1, "chain_decode handles batch=1 (extend for B>1 later)."

    # Prefill both models on prefix[:-1]; keep last prompt token as pending.
    # This avoids the position-shift bug: forwarding pending later in verify
    # places it at position (prefix_len - 1), so logits[0] predicts position
    # prefix_len = the first generated token.
    t_cache = DynamicCache()
    d_cache = DynamicCache()
    t0 = time.time()
    if prefix_len > 1:
        target(input_ids=input_ids[:, :-1].to(t_dev),
               past_key_values=t_cache, use_cache=True)
        draft(input_ids=input_ids[:, :-1].to(d_dev),
              past_key_values=d_cache, use_cache=True)
        cache_len = prefix_len - 1
    else:
        cache_len = 0

    # Track output on CPU; move to the right device for each model call.
    output_ids = input_ids.cpu()
    pending = input_ids[:, -1:].cpu()       # last prompt token, NOT yet in cache

    n_rounds = 0
    total_accepted = 0
    while output_ids.shape[1] - prefix_len < max_new_tokens:
        n_rounds += 1
        # ── 1. Draft γ tokens autoregressively ──
        drafted = torch.empty((1, gamma), dtype=torch.long, device=d_dev)
        draft_probs = None
        if verify_mode == "ratio":
            draft_probs = torch.empty((1, gamma), dtype=torch.float32, device=d_dev)
        cur = pending.to(d_dev)
        for k in range(gamma):
            d_out = draft(input_ids=cur, past_key_values=d_cache, use_cache=True)
            logits = d_out.logits[:, -1, :].float()
            if draft_mode == "argmax" or temperature == 0.0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                p = F.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(p, num_samples=1)
            drafted[:, k:k + 1] = nxt
            if draft_probs is not None:
                p = F.softmax(logits / max(temperature, 1.0), dim=-1)
                draft_probs[:, k] = p.gather(-1, nxt).squeeze(-1)
            cur = nxt
        # Extra forward of drafted[γ-1] so draft cache matches target cache
        # length (both = prefix_len - 1 + γ + 1) — required for symmetric crop()
        # when all γ get accepted.
        draft(input_ids=cur, past_key_values=d_cache, use_cache=True)

        # ── 2. Target verify in one forward (input = [pending, drafted], γ+1 tokens) ──
        # logits at γ+1 positions:
        #   logits[0]   = pred after pending  → compare to drafted[0]
        #   logits[k]   = pred after drafted[k-1] → compare to drafted[k]   (k=1..γ-1)
        #   logits[γ]   = pred after drafted[γ-1] → bonus when all γ accepted
        drafted_cpu = drafted.cpu()  # cache for output_ids construction
        verify_in = torch.cat([pending.to(t_dev), drafted.to(t_dev)], dim=-1)
        t_out = target(input_ids=verify_in, past_key_values=t_cache, use_cache=True)
        t_logits = t_out.logits.float()                              # [1, γ+1, V] on t_dev

        # ── 3. Accept/reject per position (all comparisons on t_dev) ──
        n_accept = 0
        bonus_id = None
        drafted_t = drafted.to(t_dev)
        for k in range(gamma):
            tlogits_k = t_logits[:, k, :]
            d_tok = drafted_t[:, k]
            if verify_mode == "greedy":
                t_arg = tlogits_k.argmax(dim=-1)
                if (t_arg == d_tok).item():
                    n_accept += 1
                else:
                    bonus_id = t_arg
                    break
            else:
                # simple / ratio: use target's prob on drafted token
                t_prob = F.softmax(tlogits_k / max(temperature, 1.0), dim=-1)
                p_target = t_prob.gather(-1, d_tok.unsqueeze(-1)).squeeze(-1)
                if verify_mode == "ratio":
                    p_draft = draft_probs[:, k].to(t_dev).clamp(min=1e-30)
                    accept_p = (p_target / p_draft).clamp(max=1.0)
                else:
                    accept_p = p_target.clamp(max=1.0)
                r = torch.rand_like(accept_p)
                if (r <= accept_p).item():
                    n_accept += 1
                else:
                    bonus_id = torch.multinomial(t_prob, num_samples=1).squeeze(-1)
                    break

        total_accepted += n_accept
        # ── 4. Commit accepted prefix + bonus, truncate caches ──
        accepted = drafted_cpu[:, :n_accept]
        if bonus_id is None:
            # All γ accepted: bonus = pred at the γ-th (last) verify position.
            tlogits_b = t_logits[:, gamma, :]
            if verify_mode == "greedy" or temperature == 0.0:
                bonus_id = tlogits_b.argmax(dim=-1)
            else:
                p_b = F.softmax(tlogits_b / temperature, dim=-1)
                bonus_id = torch.multinomial(p_b, num_samples=1).squeeze(-1)
        bonus_cpu = bonus_id.view(1, 1).cpu()
        new_seg = torch.cat([accepted, bonus_cpu], dim=-1)
        output_ids = torch.cat([output_ids, new_seg], dim=-1)
        pending = bonus_cpu

        # Truncate caches to keep [prefix..., last_committed_token excluding bonus].
        # Cache before this round: prefix_len - 1 + (prior accepted committed).
        # Round added γ+1 positions to target (verify_in length) and γ to draft.
        # Want keep_len = (positions for the committed tokens, excluding bonus)
        #              = prefix_len - 1 + (total committed before bonus)
        #              = output_ids.shape[1] - 1.
        keep_len = output_ids.shape[1] - 1
        t_cache.crop(keep_len)
        d_cache.crop(keep_len)

        if eos_token_id is not None and (bonus_id.item() == eos_token_id):
            break

    elapsed = time.time() - t0
    return ChainResult(
        output_ids=output_ids,
        n_generated=output_ids.shape[1] - prefix_len,
        n_rounds=n_rounds,
        total_accepted=total_accepted,
        mean_alpha=total_accepted / max(n_rounds, 1),
        elapsed=elapsed,
    )


# ---------------------------------------------------------------------------
# Tree spec decode (Phase B)
#
# Algorithm matches sdpo/tree_spec_decode.py (Qwen3) with three Gemma-4-specific
# adaptations:
#
#  1. Cross-device routing: target and draft live on different GPUs. The tree is
#     built on the draft's device; before verify we move tree.tokens to the
#     target's device. Per-node tensors (depths, parents, probs) stay on draft
#     device — they're only used for path bookkeeping and don't enter forward
#     passes.
#
#  2. No early-exit context. Gemma 4 E2B has 20 KV-shared layers at the tail:
#     layers ≥ first_kv_shared_layer_idx reuse K/V from a shared earlier layer.
#     Truncating layers[:exit_layer] would break this sharing and silently emit
#     wrong outputs. We always run the full draft model.
#
#  3. We rely on HF's auto-broadcast: when a 4D attention_mask is supplied,
#     transformers skips its internal sliding/full mask construction and applies
#     our mask uniformly across all layers. For γ=7 (max sequence growth per
#     verify = 8 tokens), this is safely below sliding_window (≥512), so giving
#     all layers the same tree-causal mask is correct — sliding wouldn't have
#     masked anything within the tree anyway.
# ---------------------------------------------------------------------------

@dataclass
class DraftTree:
    """Flat tree representation. parents[i] = -1 for roots."""
    tokens: torch.Tensor                # [n_nodes] long, on draft device
    parents: torch.Tensor               # [n_nodes] long, on draft device
    depths: torch.Tensor                # [n_nodes] long, on draft device
    scores: torch.Tensor                # [n_nodes] float, cumulative log-prob
    probs: Optional[torch.Tensor] = None  # [n_nodes] float, p_draft (for ratio verify)
    depth_counts: List[int] = field(default_factory=list)
    device: torch.device = torch.device("cpu")

    @property
    def n_nodes(self) -> int:
        return self.tokens.shape[0]


@dataclass
class VerifyState:
    """Cache + pending state for ONE model (target or draft)."""
    cache: object                       # DynamicCache
    cache_len: int                      # positions already KV-cached
    pending_token: torch.Tensor         # [1, 1] — next token, not yet in cache


def _cache_reorder_select(cache, sel_idx: torch.Tensor) -> None:
    """In-place: keep only positions in sel_idx along the seq dim of every layer.

    Handles both old DynamicCache API (.key_cache/.value_cache lists, pre-5.x)
    and the newer .layers[...].keys/values attribute (transformers 5.x).
    """
    if (hasattr(cache, "layers") and cache.layers is not None
            and len(cache.layers) > 0
            and hasattr(cache.layers[0], "keys")):
        for layer in cache.layers:
            layer.keys = layer.keys.index_select(-2, sel_idx)
            layer.values = layer.values.index_select(-2, sel_idx)
    elif hasattr(cache, "key_cache") and len(cache.key_cache) > 0:
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = cache.key_cache[i].index_select(-2, sel_idx)
            cache.value_cache[i] = cache.value_cache[i].index_select(-2, sel_idx)
    else:
        raise RuntimeError(f"Cannot reorder cache of type {type(cache).__name__}")
    new_len = int(sel_idx.numel())
    if hasattr(cache, "_seen_tokens"):
        cache._seen_tokens = new_len


@torch.inference_mode()
def _prefill_minus_one(model, input_ids: torch.LongTensor) -> VerifyState:
    """Forward prefix[:-1] into a fresh cache; pending = prefix[-1]."""
    dev = model.model.embed_tokens.weight.device
    input_ids = input_ids.to(dev)
    if input_ids.shape[1] > 1:
        out = model(input_ids=input_ids[:, :-1],
                    past_key_values=DynamicCache(), use_cache=True)
        cache = out.past_key_values
        cache_len = input_ids.shape[1] - 1
    else:
        cache = DynamicCache()
        cache_len = 0
    return VerifyState(cache=cache, cache_len=cache_len,
                       pending_token=input_ids[:, -1:].contiguous())


@torch.inference_mode()
def _build_draft_tree(
    draft,
    state: VerifyState,
    gamma: int,
    top_k: Sequence[int],
    budget: int,
    draft_mode: str = "argmax",
    temperature: float = 1.0,
    need_probs: bool = False,
) -> DraftTree:
    """Incremental tree build. Mutates state (cache, cache_len) in place.

    Post-state: cache holds [prior cache..][pending][tree node 0..n-1].
    state.cache_len = pre_call_cache_len + 1 + n_nodes.
    state.pending_token is now stale; caller replaces it with target's correction.
    """
    top_k_list = list(top_k)
    if len(top_k_list) < gamma:
        top_k_list = top_k_list + [top_k_list[-1]] * (gamma - len(top_k_list))

    device = state.pending_token.device
    mdtype = next(draft.parameters()).dtype
    min_val = torch.finfo(mdtype).min

    L0 = state.cache_len            # before tree build
    prefix_len = L0 + 1             # cache length AFTER forwarding pending

    node_tokens: List[int] = []
    node_parents: List[int] = []
    node_depths: List[int] = []
    node_scores: List[float] = []
    node_probs: List[float] = []
    depth_counts: List[int] = []
    T_draft = max(temperature, 1e-6)

    # Step 1: forward pending → depth-0 root candidates.
    pos_p = torch.tensor([[L0]], device=device, dtype=torch.long)
    out = draft(input_ids=state.pending_token, position_ids=pos_p,
                past_key_values=state.cache, use_cache=True)
    state.cache = out.past_key_values
    state.cache_len = prefix_len

    last_logits = out.logits[:, -1, :].float()
    n_roots = min(top_k_list[0], budget)
    if draft_mode == "sample":
        probs_root = F.softmax(last_logits / T_draft, dim=-1)
        sampled = torch.multinomial(probs_root[0], n_roots, replacement=False)
        roots_tok = sampled.tolist()
        p_at = probs_root[0, sampled].tolist()
        log_probs = F.log_softmax(last_logits / T_draft, dim=-1)
        roots_sc = log_probs[0, sampled].tolist()
    else:
        log_probs = F.log_softmax(last_logits, dim=-1)
        topk_s, topk_i = log_probs.topk(n_roots, dim=-1)
        roots_tok = topk_i[0].tolist()
        roots_sc = topk_s[0].tolist()
        if need_probs:
            probs_root_T = F.softmax(last_logits / T_draft, dim=-1)
            p_at = probs_root_T[0, topk_i[0]].tolist()
        else:
            p_at = [1.0] * n_roots
    for r in range(n_roots):
        node_tokens.append(roots_tok[r])
        node_parents.append(-1)
        node_depths.append(0)
        node_scores.append(roots_sc[r])
        node_probs.append(p_at[r])
    depth_counts.append(n_roots)

    # Depths 1..γ-1: forward previous-depth leaves with tree-causal mask.
    # Depth γ: only place final leaves into cache; no new candidates.
    for depth in range(1, gamma + 1):
        n_leaves = depth_counts[-1]
        if n_leaves == 0:
            break

        leaf_start = len(node_tokens) - n_leaves
        leaf_indices = list(range(leaf_start, leaf_start + n_leaves))
        leaf_toks = torch.tensor(
            [node_tokens[i] for i in leaf_indices],
            device=device, dtype=torch.long).unsqueeze(0)
        pos_ids = torch.full((1, n_leaves), prefix_len + depth - 1,
                              device=device, dtype=torch.long)

        cache_len_now = state.cache_len
        total_kv = cache_len_now + n_leaves

        # 4D mask: each leaf attends to prefix [0..prefix_len-1] + its ancestors + self.
        mask_bool = torch.zeros(n_leaves, total_kv, dtype=torch.bool)
        mask_bool[:, :prefix_len] = True
        for li, leaf_idx in enumerate(leaf_indices):
            j = leaf_idx
            while j >= 0:
                mask_bool[li, prefix_len + j] = True
                j = node_parents[j]
            mask_bool[li, cache_len_now + li] = True   # attend to self in current forward
        mask = torch.where(
            mask_bool.to(device, non_blocking=True),
            torch.zeros((), dtype=mdtype, device=device),
            torch.full((), min_val, dtype=mdtype, device=device),
        ).unsqueeze(0).unsqueeze(0)                     # [1, 1, n_leaves, total_kv]

        fout = draft(input_ids=leaf_toks, attention_mask=mask,
                     position_ids=pos_ids, past_key_values=state.cache,
                     use_cache=True)
        state.cache = fout.past_key_values
        state.cache_len = cache_len_now + n_leaves

        if depth >= gamma:
            break

        new_logits = fout.logits[0].float()              # [n_leaves, V]
        new_lp = F.log_softmax(new_logits, dim=-1)
        k_this = top_k_list[depth]

        if draft_mode == "sample":
            probs_leaves = F.softmax(new_logits / T_draft, dim=-1)
            sampled = torch.multinomial(probs_leaves, k_this, replacement=False)
            topk_i = sampled
            topk_s = new_lp.gather(1, sampled)
            cand_probs = probs_leaves.gather(1, sampled)
        else:
            topk_s, topk_i = new_lp.topk(k_this, dim=-1)
            if need_probs:
                probs_T = F.softmax(new_logits / T_draft, dim=-1)
                cand_probs = probs_T.gather(1, topk_i)
            else:
                cand_probs = torch.ones_like(topk_s)

        parent_scores_t = torch.tensor(
            [node_scores[i] for i in leaf_indices],
            device=device, dtype=torch.float32)
        cum_scores = parent_scores_t.unsqueeze(1) + topk_s
        flat_scores = cum_scores.reshape(-1)
        flat_tokens = topk_i.reshape(-1)
        flat_probs = cand_probs.reshape(-1)
        leaf_idx_t = torch.tensor(leaf_indices, device=device, dtype=torch.long)
        flat_parents = leaf_idx_t.unsqueeze(1).expand(-1, k_this).reshape(-1)

        remaining = budget - len(node_tokens)
        keep = min(remaining, flat_scores.numel())
        if keep <= 0:
            depth_counts.append(0)
            break
        top_vals, top_idx = flat_scores.topk(keep)
        sel_t = flat_tokens[top_idx].tolist()
        sel_p = flat_parents[top_idx].tolist()
        sel_s = top_vals.tolist()
        sel_pr = flat_probs[top_idx].tolist()
        for i in range(keep):
            node_tokens.append(sel_t[i])
            node_parents.append(sel_p[i])
            node_depths.append(depth)
            node_scores.append(sel_s[i])
            node_probs.append(sel_pr[i])
        depth_counts.append(keep)

    probs_t = None
    if need_probs or draft_mode == "sample":
        probs_t = torch.tensor(node_probs, device=device, dtype=torch.float32)
    return DraftTree(
        tokens=torch.tensor(node_tokens, device=device, dtype=torch.long),
        parents=torch.tensor(node_parents, device=device, dtype=torch.long),
        depths=torch.tensor(node_depths, device=device, dtype=torch.long),
        scores=torch.tensor(node_scores, device=device, dtype=torch.float32),
        probs=probs_t,
        depth_counts=depth_counts,
        device=device,
    )


@torch.inference_mode()
def _verify_tree_step(
    target,
    state: VerifyState,
    tree: DraftTree,
    temperature: float = 0.0,
    verify_mode: str = "auto",
) -> Tuple[torch.Tensor, int, VerifyState, List[int]]:
    """One tree verification round on target. Returns (new_tokens, n_accept, new_state, best_path).

    Input  = [pending, tree.tokens]  → length n+1 on target's device.
    Output = [accepted_path tokens..] ++ [correction]    → length n_accept+1.
    Cache truncated to [prefix][pending][accepted_path]; correction is the new pending
    (not yet in cache).
    """
    target_dev = target.model.embed_tokens.weight.device
    n = tree.n_nodes
    cache_len = state.cache_len
    tdtype = next(target.parameters()).dtype
    min_val = torch.finfo(tdtype).min

    if n == 0:
        # No tree: just predict next token from pending.
        pos = torch.tensor([[cache_len]], device=target_dev, dtype=torch.long)
        out = target(input_ids=state.pending_token, position_ids=pos,
                     past_key_values=state.cache, use_cache=True)
        eff_v0 = verify_mode if verify_mode != "auto" else (
            "greedy" if temperature == 0.0 else "simple")
        if eff_v0 == "greedy":
            correction = out.logits[:, -1, :].argmax(-1, keepdim=True)
        else:
            T_v = max(temperature, 1e-6)
            probs = torch.softmax(out.logits[:, -1, :].float() / T_v, dim=-1)
            correction = torch.multinomial(probs[0], 1).view(1, 1)
        new_state = VerifyState(cache=out.past_key_values,
                                cache_len=cache_len + 1,
                                pending_token=correction)
        return correction, 0, new_state, []

    # Move tree tokens to target's device (tree was built on draft's device).
    tree_tokens_t = tree.tokens.to(target_dev)
    input_toks = torch.cat([state.pending_token, tree_tokens_t.unsqueeze(0)], dim=1)
    pos = torch.empty((1, n + 1), dtype=torch.long, device=target_dev)
    pos[0, 0] = cache_len
    pos[0, 1:] = cache_len + 1 + tree.depths.to(target_dev).long()

    # 4D tree-causal mask. Built on CPU then moved to target_dev.
    parents_cpu = tree.parents.cpu().tolist()
    total = cache_len + n + 1
    mask_bool = torch.zeros(n + 1, total, dtype=torch.bool)
    mask_bool[0, :cache_len + 1] = True                  # pending sees prefix + itself
    mask_bool[1:, :cache_len + 1] = True                 # all tree nodes see prefix + pending
    for i in range(n):
        j = i
        while j >= 0:
            mask_bool[i + 1, cache_len + 1 + j] = True   # node i sees ancestors + self
            j = parents_cpu[j]
    attn_mask = torch.where(
        mask_bool.to(target_dev, non_blocking=True),
        torch.zeros((), dtype=tdtype, device=target_dev),
        torch.full((), min_val, dtype=tdtype, device=target_dev),
    ).unsqueeze(0).unsqueeze(0)

    out = target(input_ids=input_toks, attention_mask=attn_mask,
                 position_ids=pos, past_key_values=state.cache, use_cache=True)
    cache = out.past_key_values
    logits = out.logits[0].float()                       # [n+1, V] on target_dev

    # Enumerate root-to-leaf paths (CPU is fine, tree is small).
    children = [[] for _ in range(n)]
    for i in range(n):
        p = parents_cpu[i]
        if p >= 0:
            children[p].append(i)
    leaves = [i for i in range(n) if len(children[i]) == 0]
    paths: List[List[int]] = []
    for leaf in leaves:
        path = []
        j = leaf
        while j >= 0:
            path.append(j)
            j = parents_cpu[j]
        path.reverse()
        paths.append(path)

    eff_verify = verify_mode if verify_mode != "auto" else (
        "greedy" if temperature == 0.0 else "simple")

    tree_tokens_cpu = tree.tokens.cpu().tolist()
    if eff_verify in ("simple", "ratio"):
        import random
        T_v = max(temperature, 1e-6)
        probs = torch.softmax(logits / T_v, dim=-1)
        probs_cpu = probs.cpu()
        tree_probs_cpu = (tree.probs.cpu().tolist()
                          if eff_verify == "ratio" and tree.probs is not None else None)
        accept_node = [False] * n
        for i in range(n):
            p = parents_cpu[i]
            pos_for_check = 0 if p < 0 else (1 + p)
            token = tree_tokens_cpu[i]
            p_t = probs_cpu[pos_for_check, token].item()
            if eff_verify == "simple":
                accept_node[i] = (random.random() <= p_t)
            else:
                p_d = tree_probs_cpu[i] if tree_probs_cpu else 1.0
                ratio = min(1.0, p_t / max(p_d, 1e-20))
                accept_node[i] = (random.random() <= ratio)
        best_accepted = 0
        best_path: List[int] = []
        for path in paths:
            acc = 0
            for ni in path:
                if accept_node[ni]:
                    acc += 1
                else:
                    break
            if acc > best_accepted:
                best_accepted = acc
                best_path = path[:acc]
        if best_accepted > 0:
            last_idx = best_path[-1]
            correction = torch.multinomial(probs[1 + last_idx], 1).view(1, 1)
            accepted_toks = tree_tokens_t[best_path].unsqueeze(0)
        else:
            correction = torch.multinomial(probs[0], 1).view(1, 1)
            accepted_toks = torch.empty(1, 0, dtype=torch.long, device=target_dev)
    else:
        argmax = logits.argmax(dim=-1)                   # [n+1]
        argmax_cpu = argmax.cpu().tolist()
        best_accepted = 0
        best_path = []
        for path in paths:
            root_idx = path[0]
            if argmax_cpu[0] != tree_tokens_cpu[root_idx]:
                continue
            acc = 1
            for pi in range(1, len(path)):
                if argmax_cpu[1 + path[pi - 1]] == tree_tokens_cpu[path[pi]]:
                    acc += 1
                else:
                    break
            if acc > best_accepted:
                best_accepted = acc
                best_path = path[:acc]
        if best_accepted > 0:
            last_idx = best_path[-1]
            correction = argmax[1 + last_idx].view(1, 1)
            accepted_toks = tree_tokens_t[best_path].unsqueeze(0)
        else:
            correction = argmax[0].view(1, 1)
            accepted_toks = torch.empty(1, 0, dtype=torch.long, device=target_dev)

    new_tokens = torch.cat([accepted_toks, correction], dim=1)

    # Truncate cache: keep [prefix..pending] + accepted path.
    sel = list(range(cache_len + 1))
    sel.extend(cache_len + 1 + idx for idx in best_path)
    sel_idx = torch.tensor(sel, device=target_dev, dtype=torch.long)
    _cache_reorder_select(cache, sel_idx)

    new_state = VerifyState(cache=cache,
                            cache_len=cache_len + 1 + best_accepted,
                            pending_token=correction)
    return new_tokens, best_accepted, new_state, best_path


@dataclass
class TreeResult:
    output_ids: torch.LongTensor          # CPU
    n_generated: int
    n_rounds: int
    total_accepted: int
    mean_alpha: float
    elapsed: float


@torch.inference_mode()
def tree_decode(
    target,
    draft,
    input_ids: torch.LongTensor,          # [1, L] on CPU OK
    max_new_tokens: int = 256,
    gamma: int = GAMMA_DEFAULT,
    top_k: Sequence[int] = TOP_K_DEFAULT,
    budget: int = BUDGET_DEFAULT,
    temperature: float = 0.0,
    draft_mode: str = "argmax",
    verify_mode: str = "auto",
    eos_token_id: Optional[int] = None,
) -> TreeResult:
    """Tree spec decode with persistent KV caches on both target and draft.

    Each round:
      1. Build a γ-depth tree on draft (incremental KV).
      2. Verify the tree on target with a tree-causal 4D mask.
      3. Truncate draft cache to [prefix][pending][accepted_path] using the
         same indexing as target, so the next round starts cleanly.
    """
    import time
    B, prefix_len = input_ids.shape
    assert B == 1, "tree_decode handles batch=1."

    t0 = time.time()
    t_state = _prefill_minus_one(target, input_ids)
    d_state = _prefill_minus_one(draft, input_ids)

    output_ids = input_ids.clone().cpu()
    target_dev = target.model.embed_tokens.weight.device
    draft_dev = draft.model.embed_tokens.weight.device

    n_rounds = 0
    total_accepted = 0
    while output_ids.shape[1] - prefix_len < max_new_tokens:
        n_rounds += 1
        # Draft pending lives on draft_dev; ensure it matches our last commit.
        # After verify, the new pending is computed on target_dev; mirror to draft.
        # (First iteration: both prefills produced pending = input_ids[:, -1:] on
        #  their own devices, already consistent.)

        # ── 1. Build draft tree (extends draft cache; pending becomes stale).
        tree = _build_draft_tree(
            draft, d_state, gamma=gamma, top_k=top_k, budget=budget,
            draft_mode=draft_mode, temperature=temperature,
            need_probs=(verify_mode == "ratio"))

        # ── 2. Verify on target.
        new_tokens, n_accept, t_state, best_path = _verify_tree_step(
            target, t_state, tree,
            temperature=temperature, verify_mode=verify_mode)
        total_accepted += n_accept

        # ── 3. Truncate draft cache to match target's accepted layout.
        # Draft cache layout post-build: [prefix..pending][tree node 0..n-1].
        # Same indexing scheme as target → use the same sel.
        sel = list(range(d_state.cache_len - tree.n_nodes))   # keep [prefix..pending]
        sel.extend((d_state.cache_len - tree.n_nodes) + idx for idx in best_path)
        sel_idx = torch.tensor(sel, device=draft_dev, dtype=torch.long)
        _cache_reorder_select(d_state.cache, sel_idx)
        d_state.cache_len = len(sel)
        # Draft pending must be set to target's correction.
        d_state.pending_token = t_state.pending_token.to(draft_dev)

        # ── 4. Append accepted tokens + correction to output_ids.
        new_tokens_cpu = new_tokens.cpu()
        output_ids = torch.cat([output_ids, new_tokens_cpu], dim=1)

        if eos_token_id is not None and (new_tokens_cpu[0, -1].item() == eos_token_id):
            break

    elapsed = time.time() - t0
    return TreeResult(
        output_ids=output_ids,
        n_generated=output_ids.shape[1] - prefix_len,
        n_rounds=n_rounds,
        total_accepted=total_accepted,
        mean_alpha=total_accepted / max(n_rounds, 1),
        elapsed=elapsed,
    )
