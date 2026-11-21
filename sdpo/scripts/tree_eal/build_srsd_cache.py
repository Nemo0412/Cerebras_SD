#!/usr/bin/env python
"""Precompute Sampled-tree EAL offline cache (match / discard).

For each training sequence + anchor:
  1. Draft samples a full tree (same shape as online).
  2. Keep the top-`cache_paths` draft-score paths (default 8).
  3. Target verifies ONLY that pruned union → parent-slot logits.
  4. Store the union + each path's token sequence (for train-time matching).

Training (--eal_mode sample_rsd_offline):
  current draft resamples a tree, keeps --path_topk paths (e.g. 4), and for
  each path: if its token sequence equals a cached path, use that path's
  cached target logits in residual EAL; otherwise discard the path.
  Training never runs the target.

Example:
  python sdpo/scripts/tree_eal/build_srsd_cache.py \\
    --basepath ... --draftpath ... --trainpath ... --outdir ... \\
    --cache_paths 8 --max_samples 12000 --n_anchors 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from sdpo.small_lm_tree_v2_model import (  # noqa: E402
    SmallLMTreeV2Model, induce_path_topk_union, _pad_cache_batch, _extract_kv_lists,
)
from sdpo.data_pipeline import build_dataset, DataCollator  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--basepath", required=True)
    p.add_argument("--draftpath", required=True)
    p.add_argument("--trainpath", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--top_k_per_depth", default="4,3,2,1,1,1,1")
    p.add_argument("--cache_paths", type=int, default=8,
                   help="How many target-verified paths to store per anchor.")
    p.add_argument("--path_topk", type=int, default=None,
                   help="Deprecated alias for --cache_paths.")
    p.add_argument("--tree_budget", type=int, default=136)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max_len", type=int, default=2048)
    p.add_argument("--max_samples", type=int, default=12000)
    p.add_argument("--n_anchors", type=int, default=1)
    p.add_argument("--shard_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.inference_mode()
def build_one(model, batch, cache_paths, temperature, n_anchors):
    """Return a list of cache dicts for one collated microbatch (B=1 preferred)."""
    device = next(model.draft_model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    loss_mask = batch["loss_mask"].to(device)
    B, L = input_ids.shape
    assert B == 1, "build cache with batch size 1"
    top_k = list(model.top_k_per_depth)
    gamma = model.gamma
    T = max(float(temperature), 1e-6)

    out = model.draft_model.model(
        input_ids=input_ids, attention_mask=attention_mask,
        use_cache=True, return_dict=True)
    drf_hidden = out.last_hidden_state
    drf_keys, drf_values = _extract_kv_lists(out.past_key_values)
    del out
    with torch.no_grad():
        tgt_out = model.target_model.model(
            input_ids=input_ids, attention_mask=attention_mask,
            use_cache=True, return_dict=True)
        tgt_hidden = tgt_out.last_hidden_state
        tgt_keys, tgt_values = _extract_kv_lists(tgt_out.past_key_values)
        del tgt_out

    valid = loss_mask[0, 1:].nonzero(as_tuple=True)[0]
    if valid.numel() == 0:
        return []
    n = min(int(n_anchors), int(valid.numel()))
    sel = valid[torch.randperm(int(valid.numel()), device=device)[:n]]
    records = []
    for anchor in sel.tolist():
        batch_idx = [0]
        anchor_pos = [int(anchor)]
        kv_lens = [anchor + 1]
        a_t = torch.tensor(anchor_pos, device=device, dtype=torch.long)
        drf_anchor = model.draft_model.lm_head(drf_hidden[0, anchor]).float().unsqueeze(0)
        tgt_anchor = model.target_model.lm_head(tgt_hidden[0, anchor]).float().unsqueeze(0)

        rec = _sample_prune_verify(
            model, batch_idx, anchor_pos, drf_anchor, tgt_anchor,
            drf_keys, drf_values, tgt_keys, tgt_values,
            kv_lens, a_t, cache_paths, T, top_k, gamma)
        if rec is None:
            continue
        rec["input_ids"] = input_ids[0].cpu().to(torch.int32)
        rec["attention_mask"] = attention_mask[0].cpu().to(torch.uint8)
        rec["loss_mask"] = loss_mask[0].cpu().to(torch.uint8)
        rec["anchor_pos"] = int(anchor)
        records.append(rec)
    return records


def _sample_prune_verify(model, batch_idx, anchor_pos, drf_anchor, tgt_anchor,
                         drf_keys, drf_values, tgt_keys, tgt_values,
                         kv_lens, a_t, cache_paths, T, top_k, gamma):
    """Expand→Score→Prune(cache_paths)→Verify; return CPU tensors + path keys."""
    device = drf_anchor.device
    B = 1
    max_kv = max(kv_lens)
    drf_cache, _ = _pad_cache_batch(drf_keys, drf_values, batch_idx, kv_lens)
    root_lp = F.log_softmax(drf_anchor, dim=-1)
    with torch.no_grad():
        root_tok = torch.multinomial(
            torch.softmax(drf_anchor / T, dim=-1), top_k[0], replacement=False)
    root_scores = root_lp.gather(-1, root_tok)
    tokens_per_depth = [root_tok]
    scores_per_depth = [root_scores]
    full_leaf_sc = root_scores if gamma == 1 else None

    for d_step in range(1, gamma):
        n_in = tokens_per_depth[-1].size(1)
        mask = model._draft_step_mask_batch(
            d_step, kv_lens, max_kv, device, model.draft_model.dtype)
        pos = (a_t + d_step).view(B, 1).expand(B, n_in)
        step_out = model.draft_model(
            input_ids=tokens_per_depth[-1],
            attention_mask=mask, position_ids=pos,
            past_key_values=drf_cache,
            use_cache=(d_step < gamma - 1), return_dict=True)
        depth_pred = step_out.logits.float()
        child_lp = F.log_softmax(depth_pred, dim=-1)
        k = top_k[d_step]
        flat_p = torch.softmax(depth_pred / T, dim=-1).reshape(B * n_in, -1)
        child_tok = torch.multinomial(flat_p, k, replacement=False).view(B, n_in, k)
        child_sc = child_lp.gather(-1, child_tok)
        cum = scores_per_depth[-1].unsqueeze(-1) + child_sc
        flat_tok = child_tok.reshape(B, -1)
        flat_sc = cum.reshape(B, -1)
        if d_step == gamma - 1:
            full_leaf_sc = flat_sc
        tokens_per_depth.append(flat_tok)
        scores_per_depth.append(flat_sc)
        if d_step < gamma - 1:
            drf_cache = step_out.past_key_values

    tokens = torch.cat(tokens_per_depth, dim=1)
    depths = torch.tensor(model.tree_depths, device=device, dtype=torch.long)
    union_parents, gather_index, sel = induce_path_topk_union(
        model.tree_parents, model.paths, full_leaf_sc.detach(), cache_paths)
    valid_u = gather_index >= 0
    M = gather_index.size(1)
    gi = gather_index.clamp(min=0)
    tok_u = torch.where(valid_u, tokens.gather(1, gi), torch.zeros_like(gi))
    dep_full = depths.view(1, -1).expand(B, -1)
    depths_u = torch.where(valid_u, dep_full.gather(1, gi), torch.zeros_like(gi))

    tgt_cache, _ = _pad_cache_batch(tgt_keys, tgt_values, batch_idx, kv_lens)
    pos_u = a_t.view(B, 1) + 1 + depths_u
    tree_mask_u = model._target_mask_batch(
        kv_lens, max_kv, union_parents, device, model.target_model.dtype)
    tgt_out = model.target_model(
        input_ids=tok_u, attention_mask=tree_mask_u, position_ids=pos_u,
        past_key_values=tgt_cache, use_cache=False, return_dict=True)
    tgt_tree_u = tgt_out.logits.float()
    has_par = union_parents >= 0
    gathered = tgt_tree_u.gather(
        1, union_parents.clamp(min=0).unsqueeze(-1).expand(B, M, tgt_tree_u.size(-1)))
    tgt_node = torch.where(
        has_par.unsqueeze(-1), gathered,
        tgt_anchor.unsqueeze(1).expand(B, M, -1))

    m = int(valid_u[0].sum().item())
    if m == 0:
        return None

    # Path keys for train-time match: token sequence + union-node index per depth.
    P = int(sel.size(1))
    path_node_ids = model.paths.to(device)[sel[0]]  # [P, gamma] full-tree node ids
    path_tokens = tokens[0, path_node_ids]          # [P, gamma]
    # Map full-tree node id → compact union column via gather_index[0, :m].
    full_to_u = torch.full((int(tokens.size(1)),), -1, device=device, dtype=torch.long)
    full_to_u[gather_index[0, :m]] = torch.arange(m, device=device)
    path_nodes = full_to_u[path_node_ids]           # [P, gamma]

    return {
        "tokens": tok_u[0, :m].cpu().to(torch.int32),
        "parents": union_parents[0, :m].cpu().to(torch.int16),
        "depths": depths_u[0, :m].cpu().to(torch.int8),
        "tgt_node": tgt_node[0, :m].cpu().to(torch.bfloat16),
        "path_tokens": path_tokens.cpu().to(torch.int32),
        "path_nodes": path_nodes.cpu().to(torch.int16),
        "valid_m": m,
        "n_paths": P,
    }


def main():
    args = parse_args()
    cache_paths = int(args.path_topk) if args.path_topk is not None else int(args.cache_paths)
    torch.manual_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    top_k = [int(x) for x in args.top_k_per_depth.split(",") if x.strip()]
    tok = AutoTokenizer.from_pretrained(args.draftpath, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = build_dataset(tok, args.trainpath, args.max_len,
                       max_samples=args.max_samples)
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=DataCollator(), num_workers=0)
    model = SmallLMTreeV2Model(
        target_path=args.basepath, draft_path=args.draftpath,
        top_k_per_depth=top_k, dtype=torch.bfloat16,
        tree_budget=args.tree_budget).cuda().eval()

    meta = {
        "cache_paths": cache_paths, "top_k_per_depth": top_k,
        "temperature": args.temperature, "max_samples": args.max_samples,
        "n_anchors": args.n_anchors, "seed": args.seed,
        "match_discard": True,
    }
    with open(os.path.join(args.outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    shard, shard_i, n_rec = [], 0, 0
    for bi, batch in enumerate(loader):
        recs = build_one(model, batch, cache_paths, args.temperature, args.n_anchors)
        for r in recs:
            shard.append(r)
            n_rec += 1
            if len(shard) >= args.shard_size:
                path = os.path.join(args.outdir, f"shard_{shard_i:05d}.pt")
                torch.save(shard, path)
                print(f"[cache] wrote {path} ({len(shard)} recs, total={n_rec})",
                      flush=True)
                shard, shard_i = [], shard_i + 1
        if (bi + 1) % 50 == 0:
            print(f"[cache] scanned {bi+1}/{len(loader)} seqs, recs={n_rec}", flush=True)
    if shard:
        path = os.path.join(args.outdir, f"shard_{shard_i:05d}.pt")
        torch.save(shard, path)
        print(f"[cache] wrote {path} ({len(shard)} recs, total={n_rec})", flush=True)
    print(f"[cache] done n_rec={n_rec} outdir={args.outdir} cache_paths={cache_paths}")


if __name__ == "__main__":
    main()
