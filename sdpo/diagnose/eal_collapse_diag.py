"""Diagnose the EAL-only loss drop versus the analytic E[L] collapse.

Reads saved draft checkpoints only. Does not touch the training process.
Fixed anchors: the middle loss-mask position of each example.
"""
import json
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_pipeline import DataCollator, build_dataset
from small_lm_tree_v2_model import (
    SmallLMTreeV2Model,
    _extract_kv_lists,
    _pad_cache_batch,
    induce_path_topk_union,
    node_marginal_eal_loss,
    tree_expected_accepted_length,
)

TARGET = "/scratch/ll5914/models/qwen3-8b"
DRAFT_INIT = "/scratch/ll5914/models/Qwen3-0.6B"
CKPT_ROOT = "/scratch/ll5914/loss_train_q8_q06/tree_eal_only_k16"
TRAIN_PATH = "/scratch/yf3005/gto_data/sharegpt_qwen3_8b_regen.jsonl"
VAL_PATH = (
    "/scratch/tx856/spec_reason/accept_length/"
    "SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_val_80.jsonl"
)
OUT_PATH = "/scratch/ll5914/logs/tree_eal/eal_collapse_diag.json"
TOP_K = (4, 3, 2, 1, 1, 1, 1)
PATH_TOPK = 16
TEMPERATURE = 1.0
MAX_LEN = 2048
N_EXAMPLES = 16
MICRO_BS = 2


def clone_kv(keys, values):
    return [k.detach().clone() for k in keys], [v.detach().clone() for v in values]


def middle_anchors(loss_mask):
    batch_idx, anchor_pos = [], []
    for b in range(loss_mask.size(0)):
        valid = (loss_mask[b] == 1).nonzero(as_tuple=True)[0]
        if len(valid) == 0:
            continue
        batch_idx.append(b)
        anchor_pos.append(int(valid[len(valid) // 2].item()))
    return batch_idx, anchor_pos


def spearman(x, y):
    x = x.float()
    y = y.float()
    rx = x.argsort().argsort().float()
    ry = y.argsort().argsort().float()
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = rx.norm() * ry.norm()
    if float(denom) == 0.0:
        return 0.0
    return float((rx * ry).sum() / denom)


def own_tree(model, drf_root, tgt_root, drf_keys, drf_values, tgt_keys, tgt_values,
             batch_idx, anchor_pos):
    """Same node-marginal tree as training, plus the tensors the loss hides."""
    device = drf_root.device
    B = len(batch_idx)
    gamma = model.gamma
    top_k = model.top_k_per_depth
    depth_counts = model.tree_depth_counts
    depth_starts = model.tree_depth_starts
    kv_lens = [ap + 1 for ap in anchor_pos]
    T = TEMPERATURE
    a_t = torch.tensor(anchor_pos, device=device, dtype=torch.long)

    drf_cache, max_kv = _pad_cache_batch(drf_keys, drf_values, batch_idx, kv_lens)
    root_lp = F.log_softmax(drf_root, dim=-1)
    root_scores, root_tok = root_lp.topk(top_k[0], dim=-1)
    tokens_per_depth = [root_tok]
    scores_per_depth = [root_scores]
    prefix_logp = [root_scores]
    full_leaf_tok = root_tok
    full_leaf_sc = root_scores
    full_leaf_logp = root_scores

    for d_step in range(1, gamma):
        n_in = tokens_per_depth[-1].size(1)
        mask = model._draft_step_mask_batch(
            d_step, kv_lens, max_kv, device, model.draft_model.dtype)
        pos = (a_t + d_step).view(B, 1).expand(B, n_in)
        step_out = model.draft_model(
            input_ids=tokens_per_depth[-1],
            attention_mask=mask,
            position_ids=pos,
            past_key_values=drf_cache,
            use_cache=(d_step < gamma - 1),
            return_dict=True,
        )
        child_lp = F.log_softmax(step_out.logits.float(), dim=-1)
        k = top_k[d_step]
        child_sc, child_tok = child_lp.topk(k, dim=-1)
        flat_tok = child_tok.reshape(B, -1)
        flat_sc = (scores_per_depth[-1].unsqueeze(-1) + child_sc).reshape(B, -1)
        flat_logp = child_sc.reshape(B, -1)
        if d_step == gamma - 1:
            full_leaf_tok, full_leaf_sc, full_leaf_logp = flat_tok, flat_sc, flat_logp
        else:
            prefix_logp.append(flat_logp)
        tokens_per_depth.append(flat_tok)
        scores_per_depth.append(flat_sc)
        if d_step < gamma - 1:
            drf_cache = step_out.past_key_values

    tokens = torch.cat(tokens_per_depth, dim=1)
    parents = torch.tensor(model.tree_parents, device=device, dtype=torch.long)
    parents = parents.view(1, -1).expand(B, -1).contiguous()
    depths = torch.tensor(model.tree_depths, device=device, dtype=torch.long)
    full_tokens = torch.cat(tokens_per_depth[:-1] + [full_leaf_tok], dim=1)
    full_logp = torch.cat(prefix_logp + [full_leaf_logp], dim=1)

    tgt_cache, _ = _pad_cache_batch(tgt_keys, tgt_values, batch_idx, kv_lens)
    pos = a_t.view(B, 1) + 1 + depths.view(1, -1)
    tree_mask = model._target_mask_batch(
        kv_lens, max_kv, parents, device, model.target_model.dtype)
    with torch.no_grad():
        tgt_out = model.target_model(
            input_ids=tokens,
            attention_mask=tree_mask,
            position_ids=pos,
            past_key_values=tgt_cache,
            use_cache=False,
            return_dict=True,
        )
    tgt_tree = tgt_out.logits.float()
    del tgt_out
    has_parent = parents >= 0
    gathered = tgt_tree.gather(
        1, parents.clamp(min=0).unsqueeze(-1).expand(B, tokens.size(1), tgt_tree.size(-1)))
    tgt_node = torch.where(
        has_parent.unsqueeze(-1), gathered,
        tgt_root.unsqueeze(1).expand(B, tokens.size(1), -1))

    n_prefix = depth_starts[-1]
    n_paths = depth_counts[-1]
    prefix_tokens = torch.cat(tokens_per_depth[:-1], dim=1)
    alpha_prefix = torch.softmax(tgt_node[:, :n_prefix] / T, dim=-1).gather(
        -1, prefix_tokens.long().unsqueeze(-1)).squeeze(-1)
    leaf_parents = torch.tensor(
        [model.tree_parents[n_prefix + j] for j in range(n_paths)],
        device=device, dtype=torch.long)
    leaf_logits = tgt_tree.index_select(1, leaf_parents)
    alpha_leaves = torch.softmax(leaf_logits / T, dim=-1).gather(
        -1, full_leaf_tok.long().unsqueeze(-1)).squeeze(-1)
    alpha_full = torch.cat([alpha_prefix, alpha_leaves], dim=1)

    union_parents, gather_index, sel = induce_path_topk_union(
        model.tree_parents, model.paths, full_leaf_sc.detach(), PATH_TOPK)
    valid = gather_index >= 0
    alpha_u = torch.where(valid, alpha_full.gather(1, gather_index.clamp(min=0)), 0)
    logp_u = full_logp.gather(1, gather_index.clamp(min=0))
    loss, eal = node_marginal_eal_loss(alpha_u, union_parents, logp_u, valid)

    alpha_live = alpha_u.detach().requires_grad_(True)
    eal_vec = tree_expected_accepted_length(alpha_live, union_parents)
    weights = torch.autograd.grad(eal_vec.sum(), alpha_live)[0]
    weights = torch.where(valid, weights, torch.zeros_like(weights)).detach()
    depth_u = depths[gather_index.clamp(min=0)]
    p_u = logp_u.exp()

    path_alpha = alpha_full[:, model.paths.to(device)].clamp(min=1e-8).prod(dim=-1)
    return {
        "loss": loss.detach(),
        "eal_vec": eal_vec.detach(),
        "eal_mean_check": eal.detach(),
        "weights": weights.detach(),
        "alpha_u": alpha_u.detach(),
        "logp_u": logp_u.detach(),
        "p_u": p_u.detach(),
        "valid": valid.detach(),
        "depth_u": depth_u.detach(),
        "gather_index": gather_index.detach(),
        "path_sel": sel.detach(),
        "path_scores": full_leaf_sc.detach(),
        "path_alpha": path_alpha.detach(),
        "full_tokens": full_tokens.detach(),
        "full_logp": full_logp.detach(),
        "alpha_full": alpha_full.detach(),
    }


def teacher_force_logp(model, base_tokens, drf_root, drf_keys, drf_values,
                       batch_idx, anchor_pos):
    """log p_draft of a frozen token tree, one entry per topology node."""
    device = base_tokens.device
    B = base_tokens.size(0)
    gamma = model.gamma
    counts = model.tree_depth_counts
    starts = model.tree_depth_starts
    top_k = model.top_k_per_depth
    kv_lens = [ap + 1 for ap in anchor_pos]
    a_t = torch.tensor(anchor_pos, device=device, dtype=torch.long)
    drf_cache, max_kv = _pad_cache_batch(drf_keys, drf_values, batch_idx, kv_lens)
    parts = [F.log_softmax(drf_root, dim=-1).gather(
        1, base_tokens[:, starts[0]:starts[0] + counts[0]].long())]
    prev = base_tokens[:, starts[0]:starts[0] + counts[0]]
    for d_step in range(1, gamma):
        n_in = prev.size(1)
        mask = model._draft_step_mask_batch(
            d_step, kv_lens, max_kv, device, model.draft_model.dtype)
        pos = (a_t + d_step).view(B, 1).expand(B, n_in)
        step_out = model.draft_model(
            input_ids=prev,
            attention_mask=mask,
            position_ids=pos,
            past_key_values=drf_cache,
            use_cache=(d_step < gamma - 1),
            return_dict=True,
        )
        k = top_k[d_step]
        child = base_tokens[:, starts[d_step]:starts[d_step] + counts[d_step]].view(B, n_in, k)
        lp = F.log_softmax(step_out.logits.float(), dim=-1)
        parts.append(lp.gather(-1, child.long()).reshape(B, -1))
        prev = base_tokens[:, starts[d_step]:starts[d_step] + counts[d_step]]
        if d_step < gamma - 1:
            drf_cache = step_out.past_key_values
    return torch.cat(parts, dim=1)


def example_record(pack):
    rows = []
    B = pack["valid"].size(0)
    gamma = 7
    for b in range(B):
        valid = pack["valid"][b]
        w = pack["weights"][b][valid]
        alpha = pack["alpha_u"][b][valid]
        p = pack["p_u"][b][valid]
        logp = pack["logp_u"][b][valid]
        depth = pack["depth_u"][b][valid]
        scores = pack["path_scores"][b]
        order = scores.argsort(descending=True)
        top = order[:PATH_TOPK]
        rest = order[PATH_TOPK:]
        path_alpha = pack["path_alpha"][b]
        rec = {
            "eal": float(pack["eal_vec"][b]),
            "loss": float((-(pack["weights"][b] * pack["logp_u"][b]) * valid).sum()),
            "w_mean": float(w.mean()),
            "w_sum": float(w.sum()),
            "w_max": float(w.max()),
            "n_unique": int(valid.sum()),
            "alpha_mean": float(alpha.mean()),
            "p_mean": float(p.mean()),
            "top16_scores": [float(x) for x in scores[top].tolist()],
            "top16_score_mean": float(scores[top].mean()),
            "rest_score_mean": float(scores[rest].mean()) if rest.numel() else None,
            "top16_path_alpha": float(path_alpha[top].mean()),
            "rest_path_alpha": float(path_alpha[rest].mean()) if rest.numel() else None,
            "score_alpha_spearman": spearman(scores, path_alpha),
            "path_sel": [int(x) for x in pack["path_sel"][b].tolist()],
            "tokens": [int(x) for x in pack["full_tokens"][b].tolist()],
        }
        for d in range(gamma):
            m = depth == d
            rec[f"alpha_d{d}"] = float(alpha[m].mean()) if bool(m.any()) else None
            rec[f"p_d{d}"] = float(p[m].mean()) if bool(m.any()) else None
            rec[f"w_d{d}"] = float(w[m].mean()) if bool(m.any()) else None
            rec[f"n_d{d}"] = int(m.sum())
        rows.append(rec)
    return rows


def frozen_shift(pack, base_pack):
    """Draft probability of the initial tree's T_K nodes, split by initial w."""
    rows = []
    B = pack["valid"].size(0)
    for b in range(B):
        valid = base_pack["valid"][b]
        idx = base_pack["gather_index"][b][valid]
        w = base_pack["weights"][b][valid]
        p0 = base_pack["full_logp"][b][idx].exp()
        p1 = pack["forced_logp"][b][idx].exp()
        med = w.median()
        high = w >= med
        low = ~high
        rows.append({
            "p_init_high": float(p0[high].mean()) if bool(high.any()) else None,
            "p_init_low": float(p0[low].mean()) if bool(low.any()) else None,
            "p_now_high": float(p1[high].mean()) if bool(high.any()) else None,
            "p_now_low": float(p1[low].mean()) if bool(low.any()) else None,
            "delta_high": float((p1 - p0)[high].mean()) if bool(high.any()) else None,
            "delta_low": float((p1 - p0)[low].mean()) if bool(low.any()) else None,
        })
    return rows


def token_overlap(rows, base_rows):
    out = []
    for rec, base in zip(rows, base_rows):
        tok = torch.tensor(rec["tokens"])
        base_tok = torch.tensor(base["tokens"])
        same = tok == base_tok
        counts = [4, 12, 24, 24, 24, 24, 24]
        start = 0
        by_depth = []
        for c in counts:
            by_depth.append(float(same[start:start + c].float().mean()))
            start += c
        sa, sb = set(rec["path_sel"]), set(base["path_sel"])
        out.append({
            "token_match": float(same.float().mean()),
            "token_match_by_depth": by_depth,
            "path_jaccard": len(sa & sb) / len(sa | sb),
        })
    return out


def mean_key(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None
    if isinstance(vals[0], list):
        n = len(vals[0])
        return [sum(v[i] for v in vals) / len(vals) for i in range(n)]
    return sum(vals) / len(vals)


def summarize(rows, frozen_rows, overlap_rows):
    summary = {
        "n": len(rows),
        "eal": mean_key(rows, "eal"),
        "loss": mean_key(rows, "loss"),
        "w_mean": sum(r["w_sum"] for r in rows) / max(sum(r["n_unique"] for r in rows), 1),
        "w_sum": mean_key(rows, "w_sum"),
        "w_max": max(r["w_max"] for r in rows),
        "n_unique": mean_key(rows, "n_unique"),
        "alpha_mean": mean_key(rows, "alpha_mean"),
        "p_mean": mean_key(rows, "p_mean"),
        "top16_scores": mean_key(rows, "top16_scores"),
        "top16_score_mean": mean_key(rows, "top16_score_mean"),
        "rest_score_mean": mean_key(rows, "rest_score_mean"),
        "top16_path_alpha": mean_key(rows, "top16_path_alpha"),
        "rest_path_alpha": mean_key(rows, "rest_path_alpha"),
        "score_alpha_spearman": mean_key(rows, "score_alpha_spearman"),
        "alpha_by_depth": [mean_key(rows, f"alpha_d{d}") for d in range(7)],
        "p_by_depth": [mean_key(rows, f"p_d{d}") for d in range(7)],
        "w_by_depth": [mean_key(rows, f"w_d{d}") for d in range(7)],
        "n_by_depth": [mean_key(rows, f"n_d{d}") for d in range(7)],
    }
    if frozen_rows:
        for key in ("p_init_high", "p_init_low", "p_now_high", "p_now_low",
                    "delta_high", "delta_low"):
            summary[key] = mean_key(frozen_rows, key)
    if overlap_rows:
        summary["token_match"] = mean_key(overlap_rows, "token_match")
        summary["token_match_by_depth"] = mean_key(overlap_rows, "token_match_by_depth")
        summary["path_jaccard"] = mean_key(overlap_rows, "path_jaccard")
    return summary


def load_split(tokenizer, path, n):
    ds = build_dataset(tokenizer, path, MAX_LEN, max_samples=n * 3, gamma=7)
    features = [ds[i] for i in range(min(n, len(ds)))]
    return DataCollator()(features)


def pack_cpu(pack):
    return {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in pack.items()}


def target_prefix(model, batch):
    ids = batch["input_ids"].cuda()
    mask = batch["attention_mask"].cuda()
    with torch.no_grad():
        tgt = model.target_model(
            input_ids=ids, attention_mask=mask, use_cache=True, return_dict=True)
    logits = tgt.logits.float().cpu()
    kv = clone_kv(*_extract_kv_lists(tgt.past_key_values))
    kv = ([k.cpu() for k in kv[0]], [v.cpu() for v in kv[1]])
    del tgt
    torch.cuda.empty_cache()
    return logits, kv


def draft_prefix(model, batch):
    ids = batch["input_ids"].cuda()
    mask = batch["attention_mask"].cuda()
    with torch.no_grad():
        drf = model.draft_model(
            input_ids=ids, attention_mask=mask, use_cache=True, return_dict=True)
    logits = drf.logits.float()
    kv = clone_kv(*_extract_kv_lists(drf.past_key_values))
    del drf
    return logits, kv


def run_chunk(model, batch, tgt_logits_cpu, tgt_kv_cpu, base_pack, check_ref=False):
    loss_mask = batch["loss_mask"].cuda()
    batch_idx, anchor_pos = middle_anchors(loss_mask)
    drf_logits, drf_kv = draft_prefix(model, batch)
    device = drf_logits.device
    b_t = torch.tensor(batch_idx, device=device)
    a_t = torch.tensor(anchor_pos, device=device)
    drf_root = drf_logits[b_t, a_t].float()
    tgt_root = tgt_logits_cpu[b_t.cpu(), a_t.cpu()].to(device).float()
    tgt_keys = [k.to(device) for k in tgt_kv_cpu[0]]
    tgt_vals = [v.to(device) for v in tgt_kv_cpu[1]]
    if check_ref:
        keys_r, vals_r = clone_kv(*drf_kv)
        ref = model._rollout_batch(
            batch_idx, anchor_pos, drf_logits, tgt_logits_cpu.to(device),
            keys_r, vals_r, tgt_keys, tgt_vals, TEMPERATURE,
            eal_mode="node_marginal", path_topk=PATH_TOPK)
        print(f"ref_eal={float(ref['eal']):.6f}", flush=True)
    keys_a, vals_a = clone_kv(*drf_kv)
    pack = own_tree(
        model, drf_root, tgt_root, keys_a, vals_a, tgt_keys, tgt_vals,
        batch_idx, anchor_pos)
    rows = example_record(pack)
    frozen_rows = []
    if base_pack is not None:
        base_gpu = {k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in base_pack.items()}
        keys_b, vals_b = clone_kv(*drf_kv)
        forced = teacher_force_logp(
            model, base_gpu["full_tokens"], drf_root, keys_b, vals_b,
            batch_idx, anchor_pos)
        pack["forced_logp"] = forced
        frozen_rows = frozen_shift(pack, base_gpu)
    if check_ref:
        diag_eal = float(pack["eal_vec"].mean())
        diff = abs(diag_eal - float(ref["eal"]))
        print(f"diag_eal={diag_eal:.6f} absdiff={diff:.6e}", flush=True)
        if diff > 2e-2:
            raise RuntimeError(f"diagnostic E[L] does not match training rollout ({diff})")
    del drf_logits, tgt_keys, tgt_vals
    torch.cuda.empty_cache()
    return rows, frozen_rows, pack_cpu(pack), anchor_pos


def load_draft(model, path):
    tmp = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.draft_model.load_state_dict(tmp.state_dict())
    del tmp
    model.draft_model.cuda().eval()
    torch.cuda.empty_cache()


def main():
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(TARGET, trust_remote_code=True)
    print("loading fixed splits", flush=True)
    splits = {
        "val": load_split(tokenizer, VAL_PATH, N_EXAMPLES),
        "train": load_split(tokenizer, TRAIN_PATH, N_EXAMPLES),
    }
    print("loading models", flush=True)
    model = SmallLMTreeV2Model(
        TARGET, DRAFT_INIT, TOP_K, dtype=torch.bfloat16, tree_budget=128)
    model.cuda().eval()
    model.target_model.eval()
    model.draft_model.eval()
    print(f"attention_dropout={model.draft_model.config.attention_dropout}", flush=True)

    ckpts = [("init", DRAFT_INIT)]
    for epoch in range(3):
        ckpts.append((f"epoch_{epoch}", os.path.join(CKPT_ROOT, f"hf_epoch_{epoch}")))

    report = {
        "n_examples": N_EXAMPLES,
        "anchor": "middle loss-mask position",
        "note": "Fixed batches. init is the untouched Qwen3-0.6B draft.",
        "anchors": {},
        "splits": {},
    }
    for split_name, batch in splits.items():
        print(f"=== split {split_name} n={batch['input_ids'].size(0)} ===", flush=True)
        chunks = []
        tgt_cached = []
        for start in range(0, batch["input_ids"].size(0), MICRO_BS):
            chunk = {k: v[start:start + MICRO_BS] for k, v in batch.items()}
            chunks.append(chunk)
            print(f"target prefix {split_name} [{start}:{start + chunk['input_ids'].size(0)}]",
                  flush=True)
            tgt_cached.append(target_prefix(model, chunk))
        init_packs = []
        init_rows = []
        init_counts = []
        split_report = []
        for label, path in ckpts:
            print(f"loading {label}", flush=True)
            load_draft(model, path)
            all_rows, all_frozen, all_overlap = [], [], []
            anchor_dump = []
            for ci, chunk in enumerate(chunks):
                base = None if label == "init" else init_packs[ci]
                rows, frozen_rows, pack, anchors = run_chunk(
                    model, chunk, tgt_cached[ci][0], tgt_cached[ci][1], base,
                    check_ref=(label == "init" and split_name == "val" and ci == 0))
                all_rows.extend(rows)
                all_frozen.extend(frozen_rows)
                anchor_dump.append(anchors)
                if label == "init":
                    init_packs.append(pack)
                    init_rows.extend(rows)
                    init_counts.append(len(rows))
                else:
                    start = sum(init_counts[:ci])
                    all_overlap.extend(token_overlap(
                        rows, init_rows[start:start + len(rows)]))
            if label == "init":
                report["anchors"][split_name] = anchor_dump
            summary = summarize(all_rows, all_frozen, all_overlap)
            summary["label"] = label
            split_report.append(summary)
            print(json.dumps({
                "split": split_name, "label": label,
                "eal": summary["eal"], "loss": summary["loss"],
                "w_mean": summary["w_mean"], "w_sum": summary["w_sum"],
                "w_max": summary["w_max"], "n_unique": summary["n_unique"],
                "alpha_by_depth": summary["alpha_by_depth"],
                "p_by_depth": summary["p_by_depth"],
                "top16_scores": summary["top16_scores"],
                "top16_path_alpha": summary["top16_path_alpha"],
                "rest_path_alpha": summary["rest_path_alpha"],
                "spearman": summary["score_alpha_spearman"],
                "delta_high": summary.get("delta_high"),
                "delta_low": summary.get("delta_low"),
                "token_match": summary.get("token_match"),
                "path_jaccard": summary.get("path_jaccard"),
            }), flush=True)
        report["splits"][split_name] = split_report

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
