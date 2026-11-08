"""FlashInfer tree expansion for SmallLMTreeV2Model._exact_window_starts (--exact_attn_backend flashinfer).

The default expansion (_expand_chunks) gives every start of a chunk its own copy of its prefix
KV, padded to the longest prefix of the chunk. Here the microbatch's draft prefix KV is laid out
once per _exact_window_starts call as a token-level (page_size=1) paged cache,
[B*L + chunk*n_tree, 1, Hkv, D] per layer, and shared by every round of the call. Start i's page
list is its sequence's tokens [0, col_i] followed by its own n_tree tree slots, and the tree
ancestor mask is the custom_mask of flashinfer's BatchPrefillWithPagedKVCacheWrapper, so nothing
is copied or padded per start and peak memory does not grow with the chunk size.

Needs flashinfer; without nvcc its kernels come from the matching flashinfer-jit-cache wheel.
"""
import contextlib

import torch
import torch.nn.functional as F
import flashinfer
from transformers.cache_utils import DynamicCache, DynamicLayer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

_WRAPPERS = {}


def _wrapper(dev):
    """One BatchPrefillWithPagedKVCacheWrapper (and its 256 MiB workspace) per device."""
    if dev not in _WRAPPERS:
        ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
        _WRAPPERS[dev] = (ws, flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, kv_layout="NHD"))
    return _WRAPPERS[dev][1]


class _TreeLayer(DynamicLayer):
    """Writes the new tree tokens of every start of the chunk into its private slots of the
    paged cache. Attention reads the paged cache, so the K/V returned here are placeholders."""

    def __init__(self, ex, li, length):
        super().__init__()
        self.ex, self.li, self.cur, self.length = ex, li, 0, length
        self.keys = self.values = None
        self.is_initialized = True

    def update(self, key_states, value_states, *a, **k):
        C, H, q, D = key_states.shape
        slots = (self.ex.base.view(C, 1) + self.cur
                 + torch.arange(q, device=key_states.device).view(1, q)).reshape(-1)
        self.ex.kc[self.li][slots, 0] = key_states.permute(0, 2, 1, 3).reshape(C * q, H, D)
        self.ex.vc[self.li][slots, 0] = value_states.permute(0, 2, 1, 3).reshape(C * q, H, D)
        self.cur += q
        return key_states, value_states

    def get_seq_length(self):
        return self.length + self.cur


class TreeExpander:
    """The paged draft prefix of one _exact_window_starts call and the expansion on it."""

    def __init__(self):
        self.src = self.kc = self.vc = self.base = None
        self.cap_tree = 0

    @torch.no_grad()
    def _layout(self, keys, values, cap_tree):
        """Per layer: the prefix of all B sequences, then cap_tree slots for tree tokens."""
        B, Hkv, L, D = keys[0].shape
        self.kc, self.vc = [], []
        for k, v in zip(keys, values):
            bk = k.new_empty((B * L + cap_tree, 1, Hkv, D))
            bv = v.new_empty(bk.shape)
            bk[:B * L, 0] = k.detach().permute(0, 2, 1, 3).reshape(B * L, Hkv, D)
            bv[:B * L, 0] = v.detach().permute(0, 2, 1, 3).reshape(B * L, Hkv, D)
            self.kc.append(bk)
            self.vc.append(bv)
        self.src, self.cap_tree = keys, cap_tree
        self.B, self.L, self.Hkv, self.D = B, L, Hkv, D

    def _attention(self, module, query, key, value, attention_mask, dropout=0.0, scaling=None,
                   is_causal=None, **kw):
        # flashinfer's default sm_scale is head_dim ** -0.5, the draft's scaling
        C, H, q, D = query.shape
        out = self.wrapper.run(query.transpose(1, 2).reshape(C * q, H, D),
                               (self.kc[module.layer_idx], self.vc[module.layer_idx]))
        return out.view(C, q, H, D), None

    def _plan(self, model, d_step, b_t, a_t, max_kv, n_prev, n_in, Hq):
        """Start i attends its sequence's tokens [0, col_i] and its own tree slots
        [0, n_prev + n_in) under the tree ancestor mask of depth d_step."""
        dev, L = a_t.device, self.L
        C = a_t.numel()
        kv_pref = a_t + 1
        n_tree = n_prev + n_in
        kv_len = kv_pref + n_tree
        qo_indptr = torch.arange(0, (C + 1) * n_in, n_in, device=dev, dtype=torch.int32)
        kv_indptr = torch.zeros(C + 1, device=dev, dtype=torch.int32)
        kv_indptr[1:] = kv_len.cumsum(0)
        maxlen = max_kv + n_tree
        ar = torch.arange(maxlen, device=dev).view(1, -1)
        page = torch.where(ar < kv_pref.view(-1, 1), b_t.view(-1, 1) * L + ar,
                           self.base.view(-1, 1) + (ar - kv_pref.view(-1, 1)))
        valid = ar < kv_len.view(-1, 1)
        kv_indices = page[valid].to(torch.int32)
        tree = model._draft_step_tree_allowed(d_step, dev)               # [n_in, n_tree]
        assert tree.shape == (n_in, n_tree), (tree.shape, n_in, n_tree)
        valid3 = valid.view(C, 1, maxlen).expand(C, n_in, maxlen)
        col = ar.view(1, 1, -1) - kv_pref.view(C, 1, 1)
        in_tree = ((col >= 0) & (col < n_tree)).expand(C, n_in, maxlen)
        tv = tree.unsqueeze(0).expand(C, n_in, n_tree).gather(
            2, col.clamp(0, n_tree - 1).expand(C, n_in, maxlen))
        custom_mask = torch.where(in_tree, tv, valid3)[valid3]
        self.wrapper.plan(qo_indptr, kv_indptr, kv_indices,
                          torch.ones(C, dtype=torch.int32, device=dev), Hq, self.Hkv, self.D, 1,
                          custom_mask=custom_mask, causal=False,
                          q_data_type=model.draft_model.dtype, kv_data_type=self.kc[0].dtype)

    @torch.no_grad()
    def expand_chunks(self, model, rows, cols, rows_t, cols_t, drf_hidden, drf_keys, drf_values,
                      u, mask, path_topk, chunk, n, n_path, path_margin, n_tree, offs, Lm):
        """SmallLMTreeV2Model._expand_chunks on the paged prefix."""
        gamma, top_k = model.gamma, model.top_k_per_depth
        dev = drf_hidden.device
        self.wrapper = _wrapper(dev)
        if self.src is not drf_keys or self.cap_tree < chunk * n_tree:
            self._layout(drf_keys, drf_values, chunk * n_tree)
        B, L = self.B, self.L
        assert drf_hidden.size(0) == B, (drf_hidden.shape, B)
        Hq = model.draft_model.config.num_attention_heads
        mapping = ALL_ATTENTION_FUNCTIONS._global_mapping
        prev = mapping["sdpa"]
        mapping["sdpa"] = self._attention
        try:
            for c in range(0, len(rows), chunk):
                b_t, a_t = rows_t[c:c + chunk], cols_t[c:c + chunk]
                C = a_t.numel()
                kv_lens = [a + 1 for a in cols[c:c + chunk]]
                max_kv = max(kv_lens)
                self.base = B * L + torch.arange(C, device=dev) * n_tree
                cache = DynamicCache()
                cache.layers = [_TreeLayer(self, li, max_kv) for li in range(len(drf_keys))]
                lp = F.log_softmax(model.draft_model.lm_head(drf_hidden[b_t, a_t]).float(), -1)
                sc, tok = lp.topk(top_k[0], dim=-1)
                toks = [tok]
                n_prev = 0
                for d in range(1, gamma):
                    n_in = toks[-1].size(1)
                    self._plan(model, d, b_t, a_t, max_kv, n_prev, n_in, Hq)
                    # the tree mask is in the plan; this one only feeds HF's mask handling
                    out = model.draft_model(
                        input_ids=toks[-1],
                        attention_mask=model._draft_step_mask_batch(
                            d, kv_lens, max_kv, dev, model.draft_model.dtype),
                        position_ids=(a_t + d).view(C, 1).expand(C, n_in),
                        past_key_values=cache, use_cache=(d < gamma - 1), return_dict=True)
                    csc, ctok = F.log_softmax(out.logits.float(), -1).topk(top_k[d], dim=-1)
                    sc = (sc.unsqueeze(-1) + csc).reshape(C, -1)
                    toks.append(ctok.reshape(C, -1))
                    n_prev += n_in
                    del out
                model._score_chunk_paths(torch.cat(toks, dim=1), sc, b_t, a_t, u, mask, offs, Lm,
                                         path_topk, n, n_path, path_margin)
        finally:
            mapping["sdpa"] = prev


@contextlib.contextmanager
def scope(model):
    """Inside the block, model's tree expansions run on a fresh TreeExpander; its paged prefix
    is freed on exit."""
    prev = getattr(model, "_tree_expander", None)
    model._tree_expander = TreeExpander()
    try:
        yield
    finally:
        model._tree_expander = prev
