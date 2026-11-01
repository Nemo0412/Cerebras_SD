"""A few real tree-greedy rounds vs the two chain approximations.

True accept length is the spec-decode path length from eval: the draft
expands a greedy top-k tree, and a node is kept only when it equals the
target argmax. mean_alpha is the average of that length over rounds.

At the same context, before the round, both approximations follow the
target argmax for gamma steps:

  Approach 1
      a_j = q_j(t_j)
      E[L] = sum_d prod_{j<=d} a_j

  Approach 2
      m_j = q_j(t_j) - q_j^{(-t_j),(K_j)}
      a_j = sigmoid(m_j / tau)
      E[L] = sum_d prod_{j<=d} a_j

K_j is 4,3,2,1,1,1,1. tau is 1. The K_j-th term is the K_j-th largest
draft probability among tokens other than t_j.
"""

import argparse
import glob
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tree_spec_decode import (
    build_draft_tree_incremental,
    prefill_minus_one,
    verify_tree_step,
    _cache_reorder,
)


def resolve_draft_path(path):
    """Train checkpoints store weights under draft_model.*; eval strips that prefix."""
    if path.rstrip("/").endswith("_stripped"):
        return path
    fixed_dir = path.rstrip("/") + "_stripped"
    if glob.glob(os.path.join(fixed_dir, "*.bin")):
        return fixed_dir
    ckpt_files = glob.glob(os.path.join(path, "*.bin"))
    for f in ckpt_files:
        state = torch.load(f, map_location="cpu")
        prefix_hit = None
        for prefix in ("draft.base.", "draft_model."):
            if any(k.startswith(prefix) for k in state):
                prefix_hit = prefix
                break
        if prefix_hit is None:
            del state
            return path
        fixed_dir = path.rstrip("/") + "_stripped"
        os.makedirs(fixed_dir, exist_ok=True)
        new_f = os.path.join(fixed_dir, os.path.basename(f))
        if not os.path.exists(new_f):
            print(f"stripping {prefix_hit} from {f}", flush=True)
            stripped = {k.removeprefix(prefix_hit): v for k, v in state.items()}
            torch.save(stripped, new_f)
        del state
        for sib in os.listdir(path):
            if sib.endswith(".bin"):
                continue
            dst = os.path.join(fixed_dir, sib)
            if not os.path.exists(dst):
                os.symlink(os.path.join(path, sib), dst)
        return fixed_dir
    return path


def _el(factors):
    total = 0.0
    prod = 1.0
    for a in factors:
        prod *= a
        total += prod
    return total


def probe_approximations(draft_model, target_model, draft_state, target_state, topk, tau):
    """E[L] of approach 1 and 2. Rolls gamma target-argmax steps, then crops the cache back."""
    d_cache = draft_state.cache
    t_cache = target_state.cache
    d0 = d_cache.get_seq_length()
    t0 = t_cache.get_seq_length()
    cur = draft_state.pending_token
    a_vals = []
    b_vals = []
    for j in range(len(topk)):
        d_out = draft_model(input_ids=cur, past_key_values=d_cache, use_cache=True)
        t_out = target_model(input_ids=cur, past_key_values=t_cache, use_cache=True)
        d_cache = d_out.past_key_values
        t_cache = t_out.past_key_values
        q = torch.softmax(d_out.logits[:, -1, :].float(), dim=-1)[0]
        t_tok = int(t_out.logits[:, -1, :].argmax(dim=-1))
        q_t = q[t_tok]
        k = min(int(topk[j]), int(q.numel()) - 1)
        others = q.clone()
        others[t_tok] = -1.0
        kth = torch.topk(others, k).values[-1]
        a_vals.append(float(q_t))
        b_vals.append(float(torch.sigmoid((q_t - kth) / tau)))
        cur = torch.tensor([[t_tok]], device=q.device)
        del d_out, t_out
    d_cache.crop(d0)
    t_cache.crop(t0)
    if d_cache.get_seq_length() != d0 or t_cache.get_seq_length() != t0:
        raise RuntimeError(
            f"cache crop failed: draft {d_cache.get_seq_length()}!={d0} "
            f"target {t_cache.get_seq_length()}!={t0}")
    draft_state.cache = d_cache
    target_state.cache = t_cache
    return _el(a_vals), _el(b_vals)


def commit_tokens(model, state, token_ids):
    """Append token_ids to the probe cache. token_ids is [1, n], n >= 1."""
    out = model(input_ids=token_ids, past_key_values=state.cache, use_cache=True)
    state.cache = out.past_key_values
    state.pending_token = None


@torch.inference_mode()
def run_one(target_model, draft_model, input_ids, max_new_tokens,
            exit_layer, gamma, top_k, budget, eos_token_id, tau):
    device = input_ids.device
    cur_ids = input_ids.clone()
    total_tokens = 0
    rows = []

    draft_state = prefill_minus_one(draft_model, cur_ids)
    target_state = prefill_minus_one(target_model, cur_ids)
    # Separate caches for the approximation, so the tree build can mutate
    # the decode caches without changing the probe context.
    probe_draft = prefill_minus_one(draft_model, cur_ids)
    probe_target = prefill_minus_one(target_model, cur_ids)

    while total_tokens < max_new_tokens:
        ea, eb = probe_approximations(
            draft_model, target_model, probe_draft, probe_target, top_k, tau)
        pending_old = probe_draft.pending_token

        pre_len = draft_state.cache_len
        tree, _dt = build_draft_tree_incremental(
            draft_model, draft_state, exit_layer, gamma, top_k, budget,
            draft_mode="argmax", temperature=1.0, need_probs=False)
        new_tokens, n_acc, _tt, target_state, best_path = verify_tree_step(
            target_model, target_state, tree, temperature=0.0,
            verify_mode="greedy")
        sel = list(range(pre_len + 1))
        sel.extend(pre_len + 1 + idx for idx in best_path)
        sel_idx = torch.tensor(sel, device=device, dtype=torch.long)
        _cache_reorder(draft_state.cache, sel_idx)
        draft_state.cache_len = pre_len + 1 + n_acc
        draft_state.pending_token = new_tokens[:, -1:].contiguous()

        # Cache holds cur[:-1]. This round writes the old pending plus the
        # accepted path; the correction becomes the new pending.
        accepted = new_tokens[:, :-1]
        to_commit = pending_old if accepted.shape[1] == 0 else torch.cat(
            [pending_old, accepted], dim=1)
        commit_tokens(draft_model, probe_draft, to_commit)
        commit_tokens(target_model, probe_target, to_commit)
        probe_draft.pending_token = new_tokens[:, -1:].contiguous()
        probe_target.pending_token = new_tokens[:, -1:].contiguous()

        cur_ids = torch.cat([cur_ids, new_tokens], dim=1)
        total_tokens += new_tokens.shape[1]
        rows.append({"true": int(n_acc), "approach1": ea, "approach2": eb})
        if eos_token_id is not None:
            eos_list = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
            if any(t in new_tokens[0].tolist() for t in eos_list):
                break
        if tree.n_nodes == 0 and n_acc == 0 and new_tokens.shape[1] == 0:
            break
    return rows


def summarize(rows):
    n = len(rows)
    true = sum(r["true"] for r in rows) / n
    a = sum(r["approach1"] for r in rows) / n
    b = sum(r["approach2"] for r in rows) / n
    mae_a = sum(abs(r["approach1"] - r["true"]) for r in rows) / n
    mae_b = sum(abs(r["approach2"] - r["true"]) for r in rows) / n
    return {
        "rounds": n,
        "true_accept_length": true,
        "approach1": a,
        "approach2": b,
        "abs_gap_approach1": abs(a - true),
        "abs_gap_approach2": abs(b - true),
        "mae_approach1": mae_a,
        "mae_approach2": mae_b,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--draft-model-path", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--questions", default="data/mt_bench/question.jsonl")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--gamma", type=int, default=7)
    parser.add_argument("--top-k", default="4,3,2,1,1,1,1")
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    top_k = [int(x) for x in args.top_k.split(",") if x.strip()]
    assert len(top_k) == args.gamma

    print(f"GPU {torch.cuda.get_device_name(0)}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, torch_dtype=torch.float16,
        attn_implementation="sdpa").cuda().eval()
    draft_path = resolve_draft_path(args.draft_model_path)
    print(f"draft_path {draft_path}", flush=True)
    draft = AutoModelForCausalLM.from_pretrained(
        draft_path, torch_dtype=torch.float16,
        attn_implementation="sdpa").cuda().eval()
    n_layers = len(draft.model.layers)
    eos = getattr(target.config, "eos_token_id", None)
    alloc = torch.cuda.memory_allocated() / 1024 ** 3
    print(f"weights_resident_gib {alloc:.2f}", flush=True)

    questions = []
    with open(args.questions) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    questions = questions[:args.num_samples]

    all_rows = []
    per_q = []
    for qi, q in enumerate(questions):
        prompt = q.get("turns", [q.get("prompt", "")])[0]
        msgs = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=True)
        except TypeError:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        rows = run_one(
            target, draft, ids, args.max_new_tokens, n_layers,
            args.gamma, top_k, args.budget, eos, args.tau)
        stats = summarize(rows)
        stats["question"] = qi
        stats["rounds_detail"] = rows
        per_q.append(stats)
        all_rows.extend(rows)
        print(
            f"q{qi}: true={stats['true_accept_length']:.3f} "
            f"A={stats['approach1']:.3f} B={stats['approach2']:.3f} "
            f"rounds={stats['rounds']} "
            f"peak_gib={torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}",
            flush=True)
        torch.cuda.empty_cache()

    overall = summarize(all_rows)
    overall["closer_by_mean_gap"] = (
        "approach1" if overall["abs_gap_approach1"] <= overall["abs_gap_approach2"]
        else "approach2")
    overall["closer_by_per_round_mae"] = (
        "approach1" if overall["mae_approach1"] <= overall["mae_approach2"]
        else "approach2")
    overall["tag"] = args.tag
    overall["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 1024 ** 3
    overall["per_question"] = [
        {k: v for k, v in s.items() if k != "rounds_detail"} for s in per_q]
    overall["rounds"] = all_rows
    # keep the aggregate round count under a stable key; "rounds" above is the list
    overall["n_rounds"] = len(all_rows)
    print(
        "OVERALL "
        + json.dumps({k: overall[k] for k in overall if k not in ("per_question", "rounds")}),
        flush=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(overall, f, indent=2)
    print("wrote", args.output, flush=True)


if __name__ == "__main__":
    main()
