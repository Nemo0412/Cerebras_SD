"""Per-tree-round EAL surrogate vs actual tree accept length, 1-2-4 tree shape.

Matches the tree topology used in sdpo/small_lm_tree_train_model.py:
       node 0 (root, depth 0)
       /             \\
    node 1         node 2     (depth 1, top-2 children of root)
    /    \\         /    \\
   n3    n4       n5    n6    (depth 2, top-2 children of each)

Per tree round:
  1. Draft builds the 1-2-4 tree (root top-1 → 2 children top-2 → 4 gc top-2).
  2. Target verifies with one forward + 4D tree attention mask.
  3. Greedy accept walk: at depth d, target's prediction at parent slot
     (= target's prediction of "what comes at depth d") must equal one of the
     candidate children; if matches node k, continue into k's subtree.
     accept length ∈ {0, 1, 2, 3}.
  4. Per-path EAL = Σ_k cumprod(β_k) where β_k = P_target(draft_token_k)
     at parent slot (same formula as training).  Tree EAL = mean over 4 paths.

Final: Pearson / Spearman of tree EAL vs hard_tau across all rounds.

Usage:
  python sdpo/diagnose/al_correlation_smalllm_tree_inference.py \\
    --base-model-path Qwen/Qwen3-8B \\
    --draft-model-path Qwen/Qwen3-0.6B \\
    --num-samples 5 --max-new-tokens 256
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None


TREE_PARENTS = (-1, 0, 0, 1, 1, 2, 2)
TREE_DEPTHS = (0, 1, 1, 2, 2, 2, 2)
TREE_N_NODES = 7
TREE_PATHS = ((0, 1, 3), (0, 1, 4), (0, 2, 5), (0, 2, 6))


def _tree_attn_mask(prefix_len, device, dtype):
    """4D mask for target tree verify: nodes 0..6 see prefix + ancestors."""
    n = TREE_N_NODES
    min_val = torch.finfo(dtype).min
    mask = torch.full((1, 1, n, prefix_len + n), min_val,
                      device=device, dtype=dtype)
    mask[0, 0, :, :prefix_len] = 0.0
    for i in range(n):
        j = i
        while j >= 0:
            mask[0, 0, i, prefix_len + j] = 0.0
            j = TREE_PARENTS[j]
    return mask


def _sibling_mask(prefix_plus_root_len, device, dtype):
    """4D mask for draft step-2 fwd: both children see prefix+root + own slot."""
    min_val = torch.finfo(dtype).min
    total_kv = prefix_plus_root_len + 2
    mask = torch.full((1, 1, 2, total_kv), min_val,
                      device=device, dtype=dtype)
    mask[0, 0, :, :prefix_plus_root_len] = 0.0
    mask[0, 0, 0, prefix_plus_root_len + 0] = 0.0
    mask[0, 0, 1, prefix_plus_root_len + 1] = 0.0
    return mask


@torch.inference_mode()
def tree_decode_round(target_model, draft_model, input_ids):
    """Build 1-2-4 tree at end of input_ids, verify, return:
        accepted_tokens  [n_accepted_tokens]  (does NOT include correction)
        correction_token  scalar
        n_accepted  (0..3)
        tree_EAL    (mean over 4 paths of Σ cumprod(β))
    """
    device = input_ids.device
    prefix_len = input_ids.shape[1]

    # ── Draft prefix forward → root prediction ──
    drf_out = draft_model(input_ids, use_cache=True)
    drf_prefix_cache = drf_out.past_key_values
    root_pred_lg = drf_out.logits[0, -1, :].float()
    root_token = root_pred_lg.argmax(-1)

    # ── Step 1: forward root → children predictions ──
    step1_out = draft_model(
        input_ids=root_token.view(1, 1),
        past_key_values=drf_prefix_cache,
        use_cache=True,
    )
    child_pred_lg = step1_out.logits[0, -1, :].float()
    child_tokens = torch.topk(child_pred_lg, 2).indices    # [2]

    # ── Step 2: forward [c0, c1] with sibling mask → grandchildren preds ──
    sib_mask = _sibling_mask(prefix_len + 1, device, draft_model.dtype)
    pos_ids = torch.tensor([[prefix_len + 1, prefix_len + 1]],
                           device=device, dtype=torch.long)
    step2_out = draft_model(
        input_ids=child_tokens.view(1, 2),
        attention_mask=sib_mask,
        position_ids=pos_ids,
        past_key_values=step1_out.past_key_values,
        use_cache=False,
    )
    gc_pred_lg = step2_out.logits[0].float()    # [2, V]
    gc_tokens = torch.topk(gc_pred_lg, 2, dim=-1).indices    # [2, 2]

    tree_tokens = torch.stack([
        root_token,
        child_tokens[0], child_tokens[1],
        gc_tokens[0, 0], gc_tokens[0, 1],
        gc_tokens[1, 0], gc_tokens[1, 1],
    ], dim=0)    # [7]

    # ── Target tree verify (single forward + tree attn) ──
    node_positions = torch.tensor(
        [[prefix_len + TREE_DEPTHS[i] for i in range(TREE_N_NODES)]],
        device=device, dtype=torch.long)
    tree_mask = _tree_attn_mask(prefix_len, device, target_model.dtype)
    # Need target's prefix cache + prefix logits (for root prediction)
    tgt_pref_out = target_model(input_ids, use_cache=True)
    tgt_pref_cache = tgt_pref_out.past_key_values
    tgt_root_pred_lg = tgt_pref_out.logits[0, -1, :].float()
    tgt_tree_out = target_model(
        input_ids=tree_tokens.view(1, TREE_N_NODES),
        attention_mask=tree_mask,
        position_ids=node_positions,
        past_key_values=tgt_pref_cache,
        use_cache=False,
    )
    tgt_tree_lg = tgt_tree_out.logits[0].float()    # [7, V]

    # Target's prediction of each node i = parent slot logits
    #   node 0 ← prefix slot (tgt_root_pred_lg)
    #   nodes 1,2 ← root slot (tgt_tree_lg[0])
    #   nodes 3,4 ← node1 slot (tgt_tree_lg[1])
    #   nodes 5,6 ← node2 slot (tgt_tree_lg[2])
    tgt_node_pred_lg = torch.stack([
        tgt_root_pred_lg,
        tgt_tree_lg[0], tgt_tree_lg[0],
        tgt_tree_lg[1], tgt_tree_lg[1],
        tgt_tree_lg[2], tgt_tree_lg[2],
    ], dim=0)    # [7, V]
    tgt_node_p = F.softmax(tgt_node_pred_lg, dim=-1)

    # ── Per-path EAL = Σ cumprod(P_target(draft_token)) along 3-step path ──
    beta_per_node = tgt_node_p.gather(
        -1, tree_tokens.long().unsqueeze(-1)).squeeze(-1)    # [7]
    paths = torch.tensor(TREE_PATHS, device=device, dtype=torch.long)    # [4, 3]
    path_beta = beta_per_node[paths]            # [4, 3]
    cum = path_beta.cumprod(dim=-1)              # [4, 3]
    eal_per_path = cum.sum(-1)                   # [4]
    tree_EAL_mean = eal_per_path.mean().item()
    tree_EAL_max = eal_per_path.max().item()

    # ── Greedy accept walk (longest matched path) ──
    # depth 0: tgt_root_pred_lg.argmax vs root_token
    tgt_argmax_at_node = tgt_node_pred_lg.argmax(-1)    # [7]
    n_accepted = 0
    accepted_idx_path = []    # list of node indices accepted (length 0..3)

    if tgt_argmax_at_node[0].item() == tree_tokens[0].item():
        accepted_idx_path.append(0)
        n_accepted = 1
        # depth 1: target's pred at root slot vs child_tokens[0] or [1]
        tgt_pred_d1 = tgt_argmax_at_node[1].item()   # same for node 1 and 2 (shared)
        if tgt_pred_d1 == tree_tokens[1].item():
            accepted_idx_path.append(1)
            n_accepted = 2
            # depth 2: tgt pred at node1 slot vs gc[0,0] or gc[0,1]
            tgt_pred_d2 = tgt_argmax_at_node[3].item()    # shared for nodes 3, 4
            if tgt_pred_d2 == tree_tokens[3].item():
                accepted_idx_path.append(3)
                n_accepted = 3
            elif tgt_pred_d2 == tree_tokens[4].item():
                accepted_idx_path.append(4)
                n_accepted = 3
        elif tgt_pred_d1 == tree_tokens[2].item():
            accepted_idx_path.append(2)
            n_accepted = 2
            tgt_pred_d2 = tgt_argmax_at_node[5].item()
            if tgt_pred_d2 == tree_tokens[5].item():
                accepted_idx_path.append(5)
                n_accepted = 3
            elif tgt_pred_d2 == tree_tokens[6].item():
                accepted_idx_path.append(6)
                n_accepted = 3

    # Tokens to append: accepted path's tokens + correction
    accepted_tokens = tree_tokens[accepted_idx_path] if accepted_idx_path \
        else tree_tokens.new_zeros(0)
    # Correction: target's prediction at the FIRST rejected slot
    # If n_accepted == 0: reject at depth 0 → use tgt's prefix prediction = tgt_root_pred_lg.argmax
    # If n_accepted == 1: reject at depth 1 → use target's pred at root slot = tgt_tree_lg[0]
    # If n_accepted == 2: reject at depth 2 → use target's pred at accepted-depth-1 slot
    # If n_accepted == 3: nothing was rejected within tree → use target's pred at the leaf slot for next-token correction
    if n_accepted == 0:
        correction = tgt_argmax_at_node[0]
    elif n_accepted == 1:
        correction = tgt_argmax_at_node[1]    # = tgt_tree_lg[0].argmax
    elif n_accepted == 2:
        # whichever child accepted is parent of depth 2 candidates
        parent_at_d1 = accepted_idx_path[1]    # 1 or 2
        correction = tgt_tree_lg[parent_at_d1].argmax(-1)
    else:   # n_accepted == 3
        leaf_idx = accepted_idx_path[2]
        correction = tgt_tree_lg[leaf_idx].argmax(-1)

    # EAL of the path the verifier actually walked (greedy-accepted prefix's path)
    # If n_accepted == 0: no path accepted → use path (0,1,3) as "would-have-been-tried" reference
    # Else: find which path matches accepted_idx_path's prefix
    if n_accepted == 0:
        accepted_path_idx = 0   # path (0,1,3) by convention
    else:
        # Find unique path containing accepted_idx_path as prefix
        accepted_path_idx = 0
        for pi, path in enumerate(TREE_PATHS):
            if list(path[:len(accepted_idx_path)]) == accepted_idx_path:
                accepted_path_idx = pi
                break
    tree_EAL_accepted = eal_per_path[accepted_path_idx].item()

    return (accepted_tokens, correction, n_accepted,
            tree_EAL_mean, tree_EAL_max, tree_EAL_accepted)


@torch.inference_mode()
def generate_tree_with_hook(target_model, draft_model, input_ids, max_new_tokens):
    rounds = []
    cur_ids = input_ids
    total_new = 0
    eos = target_model.config.eos_token_id
    if not isinstance(eos, list):
        eos = [eos] if eos is not None else []

    while total_new < max_new_tokens:
        accepted, correction, n_acc, eal_mean, eal_max, eal_acc = tree_decode_round(
            target_model, draft_model, cur_ids)
        new = torch.cat([accepted, correction.view(1)], dim=0)
        cur_ids = torch.cat([cur_ids, new.unsqueeze(0)], dim=1)
        total_new += new.shape[0]
        rounds.append({"hard_tau": n_acc,
                       "EAL_mean": eal_mean,
                       "EAL_max": eal_max,
                       "EAL_accepted": eal_acc})
        if any(t in new.tolist() for t in eos):
            break
    return cur_ids, rounds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model-path", default="Qwen/Qwen3-8B")
    p.add_argument("--draft-model-path", default="Qwen/Qwen3-0.6B")
    p.add_argument("--bench-name", default="mt_bench")
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--output", default="al_correlation_smalllm_tree.json")
    args = p.parse_args()

    print(f"[TREE-AL] target={args.base_model_path}  draft={args.draft_model_path}")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, torch_dtype=torch.float16, device_map="auto",
        attn_implementation="sdpa")
    target_model.eval()
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model_path, torch_dtype=torch.float16, device_map="auto",
        attn_implementation="sdpa")
    draft_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path,
                                              trust_remote_code=True)

    qpath = f"data/{args.bench_name}/question.jsonl"
    with open(qpath) as f:
        questions = [json.loads(l) for l in f][:args.num_samples]
    print(f"[TREE-AL] {len(questions)} questions from {args.bench_name}")

    all_rounds = []
    for qi, q in enumerate(tqdm(questions)):
        prompt = q["turns"][0]
        msgs = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(text, return_tensors="pt",
                              add_special_tokens=False).input_ids
        input_ids = input_ids.to(target_model.device)

        t0 = time.time()
        _, rounds = generate_tree_with_hook(
            target_model, draft_model, input_ids, args.max_new_tokens)
        elapsed = time.time() - t0
        a = np.mean([r["hard_tau"] for r in rounds]) if rounds else 0.0
        em = np.mean([r["EAL_mean"] for r in rounds]) if rounds else 0.0
        eM = np.mean([r["EAL_max"] for r in rounds]) if rounds else 0.0
        ea = np.mean([r["EAL_accepted"] for r in rounds]) if rounds else 0.0
        print(f"  q{qi}: {len(rounds)} rds  hard_τ={a:.2f}  "
              f"EAL_mean={em:.2f}  EAL_max={eM:.2f}  EAL_acc={ea:.2f}  {elapsed:.1f}s")
        all_rounds.extend(rounds)

    if not all_rounds:
        print("[TREE-AL] No rounds collected.")
        return

    cols = ["hard_tau", "EAL_mean", "EAL_max", "EAL_accepted"]
    M = np.array([[r[c] for c in cols] for r in all_rounds])
    print(f"\n[TREE-AL] Total {M.shape[0]} rounds (1-2-4 tree, max τ = 3)")
    for i, c in enumerate(cols):
        print(f"  {c:<14}  mean={M[:,i].mean():.3f}  std={M[:,i].std():.3f}  "
              f"min={M[:,i].min():.2f}  max={M[:,i].max():.2f}")

    correlations = {}
    if M.shape[0] >= 2:
        print(f"\n  Correlations vs hard_tau:")
        print(f"  {'metric':<14}{'pearson':>10}{'spearman':>11}")
        for i, c in enumerate(cols[1:], start=1):
            pe = pearsonr(M[:, 0], M[:, i])
            sp = spearmanr(M[:, 0], M[:, i])
            print(f"  {c:<14}{pe.statistic:>10.4f}{sp.statistic:>11.4f}")
            correlations[c] = {"pearson": float(pe.statistic),
                               "spearman": float(sp.statistic)}

    result = {
        "meta": {"target": args.base_model_path,
                 "draft": args.draft_model_path,
                 "n_rounds": int(M.shape[0])},
        "stats": {c: {"mean": float(M[:, i].mean()),
                      "std": float(M[:, i].std())} for i, c in enumerate(cols)},
        "correlations_vs_hard_tau": correlations,
        "rounds": all_rounds,
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[TREE-AL] Wrote {args.output}")


if __name__ == "__main__":
    main()
