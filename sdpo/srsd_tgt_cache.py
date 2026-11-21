"""Online LRU cache of target parent-slot top-m logits along draft paths.

Used by eal_mode=sample_rsd: after draft keeps its top-k paths, each path's
token sequence is looked up. On a full hit the target forward is skipped and
dense parent-slot logits are reconstructed from the cached top-m. On a miss
the target verifies the pruned union as usual and the new paths are inserted.

Storing top-m (not the full vocab) keeps the bank small while covering most of
the target mass on peaked Qwen distributions; the path token is always forced
into the top-m so a(v)=u[tok]/q[tok] stays exact for drafted tokens.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Hashable, Optional, Tuple

import torch


def path_key(ctx: Hashable, path_tokens) -> Tuple:
    """Cache key: context id + path token tuple."""
    if torch.is_tensor(path_tokens):
        path_tokens = tuple(int(x) for x in path_tokens.tolist())
    else:
        path_tokens = tuple(int(x) for x in path_tokens)
    return (ctx, path_tokens)


def ctx_key_from_prefix(input_ids_row: torch.Tensor, anchor: int) -> bytes:
    """Hashable context: token prefix through the anchor position."""
    return input_ids_row[: int(anchor) + 1].detach().to(torch.int32).cpu().numpy().tobytes()


class OnlineTargetPathCache:
    """LRU of (ctx, path_tokens) -> (ids [gamma,m] int32, logits [gamma,m] bf16)."""

    def __init__(self, max_paths: int = 4096, top_m: int = 256):
        self.max_paths = int(max_paths)
        self.top_m = int(top_m)
        self._store: OrderedDict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.lookups = 0

    def __len__(self):
        return len(self._store)

    def reset_stats(self):
        self.hits = self.misses = self.lookups = 0

    @property
    def hit_rate(self) -> float:
        if self.lookups <= 0:
            return 0.0
        return float(self.hits) / float(self.lookups)

    def get(self, key: Tuple) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        self.lookups += 1
        ent = self._store.get(key)
        if ent is None:
            self.misses += 1
            return None
        self.hits += 1
        self._store.move_to_end(key)
        return ent

    def put(self, key: Tuple, ids: torch.Tensor, logits: torch.Tensor):
        """ids/logits: [gamma, m] on CPU (int32 / bf16)."""
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = (ids, logits)
        else:
            self._store[key] = (ids, logits)
            while len(self._store) > self.max_paths:
                self._store.popitem(last=False)

    @torch.no_grad()
    def pack_from_full(self, full_logits: torch.Tensor, path_tok: torch.Tensor):
        """Compress one path's parent-slot logits.

        full_logits: [gamma, V] float
        path_tok: [gamma] long — drafted tokens on the path (forced into top-m)
        Returns CPU (ids [gamma,m] int32, logits [gamma,m] bf16).
        """
        gamma, V = full_logits.shape
        m = min(self.top_m, V)
        # Reserve one slot for the path token; fill the rest with top-(m-1).
        k = max(m - 1, 1)
        vals, idxs = full_logits.topk(k, dim=-1)
        ids = torch.empty(gamma, m, device=full_logits.device, dtype=torch.long)
        out = torch.empty(gamma, m, device=full_logits.device, dtype=full_logits.dtype)
        ids[:, :k] = idxs
        out[:, :k] = vals
        if m > k:
            # Last column: path token (overwrite if already in top-k).
            pt = path_tok.long()
            ids[:, k] = pt
            out[:, k] = full_logits.gather(1, pt.view(gamma, 1)).squeeze(1)
            # If path token already appeared in top-k, duplicate is fine for scatter.
        return ids.cpu().to(torch.int32), out.cpu().to(torch.bfloat16)

    def reconstruct(self, ids: torch.Tensor, logits: torch.Tensor,
                    V: int, device, dtype=torch.float32) -> torch.Tensor:
        """ids/logits [gamma, m] -> dense [gamma, V] float logits."""
        gamma, m = ids.shape
        # Floor so softmax mass concentrates on the cached top-m.
        floor = logits.float().min().item() - 20.0
        dense = torch.full((gamma, V), floor, device=device, dtype=dtype)
        dense.scatter_(1, ids.to(device=device, dtype=torch.long),
                       logits.to(device=device, dtype=dtype))
        return dense
