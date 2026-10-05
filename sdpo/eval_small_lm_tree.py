"""
Tree Speculative Decoding Eval for Small LM Draft.

Small LM (e.g., Qwen3-0.6B, Qwen3-4B) as draft, large LM (e.g., Qwen3-32B) as
target. Draft builds a tree via top-k expansion; target verifies with 4D tree
attention mask and picks the longest accepted path.

Also runs baseline chain spec decoding for comparison (with 4D causal mask so
SDPA math backend is used on both, making the comparison fair).

Usage:
  python sdpo/eval_small_lm_tree.py \\
    --base-model-path Qwen/Qwen3-32B \\
    --draft-model-path Qwen/Qwen3-0.6B \\
    --bench-name mt_bench \\
    --gamma 5 --top-k 5 --budget 30 \\
    --max-new-tokens 128 \\
    --num-samples 40 \\
    --tag smalllm_tree \\
    --run-baseline
"""

import argparse
import random

import numpy as np


def _seed_all(s=0):
    """Seed every RNG source for reproducible sample-verify."""
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    random.seed(s)
    np.random.seed(s)


def _al_stats(al):
    """Per-window accept-length distribution stats over a flat list."""
    if not al:
        return {"n_windows": 0, "median_al": 0.0, "p5_al": 0.0, "p25_al": 0.0,
                "p75_al": 0.0, "p95_al": 0.0, "std_al": 0.0, "max_al": 0}
    a = np.asarray(al, dtype=np.float64)
    return {
        "n_windows": int(a.size),
        "median_al": float(np.median(a)),
        "p5_al": float(np.percentile(a, 5)),
        "p25_al": float(np.percentile(a, 25)),
        "p75_al": float(np.percentile(a, 75)),
        "p95_al": float(np.percentile(a, 95)),
        "std_al": float(a.std(ddof=1)) if a.size >= 2 else 0.0,
        "max_al": int(a.max()),
    }
import glob
import json
import os
import sys
import time

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from tree_spec_decode import (
    build_draft_tree,
    build_draft_tree_incremental,
    verify_tree,
    prefill_minus_one,
    verify_tree_step,
    _cache_reorder,
    VerifyState,
)


def load_questions(bench_name, data_dir="data"):
    path = os.path.join(data_dir, bench_name, "question.jsonl")
    if not os.path.exists(path):
        print(f"  Bench {bench_name} not found at {path}, skipping")
        return []
    with open(path) as f:
        return [json.loads(l) for l in f]


@torch.inference_mode()
def spec_decode_tree_smalllm(
    target_model, draft_model, input_ids, max_new_tokens,
    exit_layer, gamma, top_k, budget, eos_token_id=None,
    temperature=0.0,
    draft_mode='argmax', verify_mode='auto',
    draft_top_k_vocab=0, target_top_k=0,
):
    """Tree spec decode with INCREMENTAL KV on BOTH draft and target.

    - Both models prefill prefix[:-1] once; prefix[-1] is the initial pending.
    - Each round: draft runs build_draft_tree_incremental (forwards pending,
      then γ rounds of leaf-batch forwards, placing all tree nodes into cache).
      Target runs verify_tree_step on [pending, tree.tokens]. Both caches are
      truncated with the same sel_idx (prefix + pending + accepted path).
    - Correction becomes the new pending for both.
    """
    device = input_ids.device
    cur_ids = input_ids.clone()
    total_tokens = 0
    total_rounds = 0
    total_accepted = 0
    accept_lengths = []
    total_draft_time = 0.0
    total_target_time = 0.0

    t0 = time.time()
    draft_state = prefill_minus_one(draft_model, cur_ids)
    total_draft_time += time.time() - t0
    t0 = time.time()
    target_state = prefill_minus_one(target_model, cur_ids)
    total_target_time += time.time() - t0

    while total_tokens < max_new_tokens:
        pre_len = draft_state.cache_len        # before this round's tree build
        need_probs = verify_mode in ('ratio', 'ratio_fix')
        tree, dt = build_draft_tree_incremental(
            draft_model, draft_state, exit_layer, gamma, top_k, budget,
            draft_mode=draft_mode, temperature=max(temperature, 1.0) if temperature > 0 else 1.0,
            need_probs=need_probs, draft_top_k_vocab=draft_top_k_vocab)
        total_draft_time += dt

        new_tokens, n_acc, tt, target_state, best_path = verify_tree_step(
            target_model, target_state, tree, temperature=temperature,
            verify_mode=verify_mode, target_top_k=target_top_k)
        total_target_time += tt

        # Truncate draft cache to [prefix][pending][accepted_path], same layout
        # as target. sel_idx is identical to target's because both caches share
        # the layout [0..pre_len-1][pending @ pre_len][tree nodes @ pre_len+1..].
        sel = list(range(pre_len + 1))
        sel.extend(pre_len + 1 + idx for idx in best_path)
        sel_idx = torch.tensor(sel, device=device, dtype=torch.long)
        _cache_reorder(draft_state.cache, sel_idx)
        draft_state.cache_len = pre_len + 1 + n_acc
        draft_state.pending_token = new_tokens[:, -1:].contiguous()  # correction

        cur_ids = torch.cat([cur_ids, new_tokens], dim=1)
        total_tokens += new_tokens.shape[1]
        total_rounds += 1
        total_accepted += n_acc
        accept_lengths.append(n_acc)

        if eos_token_id is not None:
            eos_list = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
            if any(t in new_tokens[0].tolist() for t in eos_list):
                break

    total_time = total_draft_time + total_target_time
    prompt_len = input_ids.shape[1]
    return {
        "total_tokens": total_tokens,
        "total_rounds": total_rounds,
        "total_accepted": total_accepted,
        "mean_alpha": total_accepted / max(total_rounds, 1),
        "tokens_per_sec": total_tokens / max(total_time, 1e-6),
        "total_time": total_time,
        "draft_time": total_draft_time,
        "target_time": total_target_time,
        "accept_lengths": accept_lengths,
        "output_ids": cur_ids[0, prompt_len:].tolist(),
    }


@torch.inference_mode()
def baseline_chain_decode(target_model, draft_model, input_ids,
                          max_new_tokens, gamma=7, force_sdpa_math=True,
                          temperature=0.0,
                          draft_mode='argmax', verify_mode='auto',
                          draft_top_k=0, target_top_k=0):
    """Chain spec decode with INCREMENTAL KV on both draft and target.

    draft_mode: 'argmax' (top-1) or 'sample' (multinomial @ T)
    verify_mode:
      'auto'   — greedy if T=0, simple otherwise (legacy behavior)
      'greedy' — argmax(target) == draft_token
      'simple' — r ≤ p_target(draft_token)
      'ratio'  — r ≤ p_target(token) / p_draft(token)   (exact rejection sampling)
    """
    device = input_ids.device
    tdtype = next(target_model.parameters()).dtype
    min_val = torch.finfo(tdtype).min

    cur_ids = input_ids.clone()
    total_tokens = 0
    total_rounds = 0
    total_accepted = 0
    accept_lengths = []
    t0 = time.time()

    d_state = prefill_minus_one(draft_model, input_ids)
    t_state = prefill_minus_one(target_model, input_ids)

    while total_tokens < max_new_tokens:
        # Draft: γ+1 one-token forwards. For ratio verify we also store p_draft.
        draft_tokens_list = []
        draft_p_list = []   # p_draft(chosen_token), only used in ratio mode
        draft_probs_list = []  # full [V] p_draft distribution — only ratio_fix needs it
        need_full_probs = (verify_mode == 'ratio_fix')
        cur_tok = d_state.pending_token
        d_cache = d_state.cache
        T_d = max(temperature, 1e-6)   # sample/softmax temperature for draft
        for step in range(gamma + 1):
            pos_d = torch.tensor(
                [[d_state.cache_len + step]], device=device, dtype=torch.long)
            out = draft_model(
                input_ids=cur_tok,
                position_ids=pos_d,
                past_key_values=d_cache,
                use_cache=True,
            )
            d_cache = out.past_key_values
            if step < gamma:
                logits = out.logits[:, -1, :].float()
                if draft_mode == 'sample':
                    d_probs = torch.softmax(logits / T_d, dim=-1)  # [1, V]
                    if draft_top_k > 0:
                        # Restrict to top-K, renormalize. p_draft used in verify
                        # is the renormalized top-K probability (0 outside).
                        topk_vals, topk_idx = d_probs.topk(draft_top_k, dim=-1)
                        topk_norm = topk_vals / topk_vals.sum(-1, keepdim=True).clamp(min=1e-20)
                        idx_in_k = torch.multinomial(topk_norm[0], 1)   # [1]
                        next_tok = topk_idx[0, idx_in_k].view(1, 1)
                        p_draft_at = topk_norm[0, idx_in_k[0]].item()
                        if need_full_probs:
                            # Build renormalized top-K distribution over full vocab
                            d_probs_tk = torch.zeros_like(d_probs)
                            d_probs_tk.scatter_(-1, topk_idx, topk_norm)
                            draft_probs_list.append(d_probs_tk[0].detach())
                    else:
                        next_tok = torch.multinomial(d_probs[0], 1).view(1, 1)
                        p_draft_at = d_probs[0, next_tok[0, 0]].item()
                        if need_full_probs:
                            draft_probs_list.append(d_probs[0].detach())
                else:  # argmax
                    next_tok = logits.argmax(-1, keepdim=True)
                    if verify_mode in ('ratio', 'ratio_fix'):
                        d_probs = torch.softmax(logits / T_d, dim=-1)
                        p_draft_at = d_probs[0, next_tok[0, 0]].item()
                        if need_full_probs:
                            draft_probs_list.append(d_probs[0].detach())
                    else:
                        p_draft_at = 1.0
                draft_tokens_list.append(next_tok)
                draft_p_list.append(p_draft_at)
                cur_tok = next_tok
        draft_tokens = torch.cat(draft_tokens_list, dim=1)       # [1, γ]

        # Target: [pending, draft_tokens] (length γ+1) in one shot with cache.
        input_toks = torch.cat([t_state.pending_token, draft_tokens], dim=1)
        seq_len = gamma + 1
        t_cache_len = t_state.cache_len
        pos_t = torch.arange(
            t_cache_len, t_cache_len + seq_len, device=device).unsqueeze(0)

        if force_sdpa_math:
            mask_bool = torch.zeros(seq_len, t_cache_len + seq_len, dtype=torch.bool)
            mask_bool[:, :t_cache_len] = True
            mask_bool[:, t_cache_len:] = torch.tril(
                torch.ones(seq_len, seq_len, dtype=torch.bool))
            attn_mask = torch.where(
                mask_bool.to(device, non_blocking=True),
                torch.zeros((), dtype=tdtype, device=device),
                torch.full((), min_val, dtype=tdtype, device=device),
            ).unsqueeze(0).unsqueeze(0)
            t_out = target_model(
                input_ids=input_toks,
                attention_mask=attn_mask,
                position_ids=pos_t,
                past_key_values=t_state.cache,
                use_cache=True,
            )
        else:
            t_out = target_model(
                input_ids=input_toks,
                position_ids=pos_t,
                past_key_values=t_state.cache,
                use_cache=True,
            )
        t_cache = t_out.past_key_values

        eff_verify = verify_mode
        if eff_verify == 'auto':
            eff_verify = 'greedy' if temperature == 0.0 else 'simple'
        draft_cpu = draft_tokens[0].cpu().tolist()
        if eff_verify == 'greedy':
            t_argmax = t_out.logits[0].argmax(dim=-1)
            t_argmax_cpu = t_argmax.cpu().tolist()
            n_accepted = 0
            for i in range(gamma):
                if t_argmax_cpu[i] == draft_cpu[i]:
                    n_accepted += 1
                else:
                    break
            correction = t_argmax[n_accepted].view(1, 1)
        elif eff_verify == 'simple':
            T_v = max(temperature, 1e-6)
            t_probs = torch.softmax(t_out.logits[0].float() / T_v, dim=-1)
            if target_top_k > 0:
                tv, ti = t_probs.topk(target_top_k, dim=-1)
                tn = tv / tv.sum(-1, keepdim=True).clamp(min=1e-20)
                t_probs = torch.zeros_like(t_probs).scatter_(-1, ti, tn)
            n_accepted = 0
            for i in range(gamma):
                p_t = t_probs[i, draft_cpu[i]].item()
                if random.random() <= p_t:
                    n_accepted += 1
                else:
                    break
            correction = torch.multinomial(t_probs[n_accepted], 1).view(1, 1)
        elif eff_verify in ('ratio', 'ratio_fix'):
            # ratio: r ≤ p_t/p_d, correction ~ p_target (biased, legacy)
            # ratio_fix: same accept rule, correction ~ normalize(max(0, p_t - p_d))
            #            (exact SpecDec correction, Leviathan et al.)
            T_v = max(temperature, 1e-6)
            t_probs = torch.softmax(t_out.logits[0].float() / T_v, dim=-1)
            if target_top_k > 0:
                tv, ti = t_probs.topk(target_top_k, dim=-1)
                tn = tv / tv.sum(-1, keepdim=True).clamp(min=1e-20)
                t_probs = torch.zeros_like(t_probs).scatter_(-1, ti, tn)
            n_accepted = 0
            for i in range(gamma):
                p_t = t_probs[i, draft_cpu[i]].item()
                p_d = draft_p_list[i]
                ratio = min(1.0, p_t / max(p_d, 1e-20))
                if random.random() <= ratio:
                    n_accepted += 1
                else:
                    break
            if eff_verify == 'ratio_fix':
                if n_accepted < gamma and n_accepted < len(draft_probs_list):
                    # correction at rejected slot: subtract full draft distribution
                    p_diff = torch.clamp(
                        t_probs[n_accepted] - draft_probs_list[n_accepted].to(t_probs.device),
                        min=0.0)
                    denom = p_diff.sum()
                    if denom > 1e-20:
                        correction = torch.multinomial(
                            p_diff / denom, 1).view(1, 1)
                    else:
                        correction = torch.multinomial(t_probs[n_accepted], 1).view(1, 1)
                else:
                    # all γ accepted → correction from clean target dist at last slot
                    correction = torch.multinomial(t_probs[n_accepted], 1).view(1, 1)
            else:  # 'ratio' (legacy — biased correction)
                correction = torch.multinomial(t_probs[n_accepted], 1).view(1, 1)
        new_tokens = torch.cat(
            [draft_tokens[:, :n_accepted], correction], dim=1)   # [1, n_acc+1]

        # Truncate both caches to [prefix-1][pending][accepted].
        keep_len = t_cache_len + 1 + n_accepted
        sel_idx = torch.arange(keep_len, device=device, dtype=torch.long)
        _cache_reorder(t_cache, sel_idx)
        _cache_reorder(d_cache, sel_idx)

        cur_ids = torch.cat([cur_ids, new_tokens], dim=1)
        total_tokens += new_tokens.shape[1]
        total_rounds += 1
        total_accepted += n_accepted
        accept_lengths.append(n_accepted)

        t_state = VerifyState(cache=t_cache, cache_len=keep_len,
                              pending_token=correction)
        d_state = VerifyState(cache=d_cache, cache_len=keep_len,
                              pending_token=correction)

        if hasattr(target_model.config, 'eos_token_id'):
            eos = target_model.config.eos_token_id
            eos_list = eos if isinstance(eos, list) else [eos]
            if any(t in new_tokens[0].tolist() for t in eos_list):
                break

    elapsed = time.time() - t0
    prompt_len = input_ids.shape[1]
    return {
        "total_tokens": total_tokens,
        "total_rounds": total_rounds,
        "total_accepted": total_accepted,
        "mean_alpha": total_accepted / max(total_rounds, 1),
        "tokens_per_sec": total_tokens / max(elapsed, 1e-6),
        "total_time": elapsed,
        "accept_lengths": accept_lengths,
        "output_ids": cur_ids[0, prompt_len:].tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model-path', required=True)
    parser.add_argument('--draft-model-path', required=True)
    parser.add_argument('--bench-name', default='mt_bench')
    parser.add_argument('--gamma', type=int, default=5, help='Tree depth')
    parser.add_argument('--top-k', default='5',
                        help='Branching factor per tree node. Either int '
                             '(same for all depths) or comma-separated list '
                             '(EAGLE-style depth-aware, e.g., "4,3,2,2,1")')
    parser.add_argument('--budget', type=int, default=30, help='Max tree nodes')
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--num-samples', type=int, default=40)
    parser.add_argument('--print-samples', type=int, default=1,
                        help='Print prompt + generated text for the first N questions per bench/mode.')
    parser.add_argument('--tag', default='smalllm_tree')
    parser.add_argument('--output-dir', default='smalllm_tree_eval_results')
    parser.add_argument('--run-baseline', action='store_true',
                        help='Also run chain baseline alongside tree decode')
    parser.add_argument('--baseline-only', action='store_true',
                        help='Run ONLY the chain baseline, skip tree decode')
    parser.add_argument('--baseline-gamma', type=int, default=7,
                        help='Chain spec decode gamma for baseline')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='temperature=0 → greedy verify (argmax-match); '
                             '>0 → EAGLE3-style sample verify (r ≤ p_target). '
                             'RNG seeded per question via _seed_all(args.seed * 1000 + qi).')
    parser.add_argument('--seed', type=int, default=0,
                        help='Base RNG seed. Per-question RNG uses (seed*1000 + question_id).')
    parser.add_argument('--draft-top-k', type=int, default=0,
                        help='If >0 and draft-mode=sample, restrict draft to top-K then '
                             'renormalize. p_draft used in ratio verify is the renormalized '
                             'top-K probability (0 outside top-K). Standard Leviathan verify '
                             'stays unbiased since it works for any draft distribution.')
    parser.add_argument('--target-top-k', type=int, default=0,
                        help='If >0, restrict target dist to top-K then renormalize BEFORE '
                             'verify. p_target lookup returns renormalized value (0 outside '
                             'top-K → drafted token outside → auto-reject). Output dist is '
                             'target top-K distribution (biased away from tail).')
    parser.add_argument('--draft-mode', choices=['argmax', 'sample'], default='argmax',
                        help='Draft generation mode: argmax (top-k tree / top-1 chain) or '
                             'sample (multinomial @ T).')
    parser.add_argument('--enable-thinking', dest='enable_thinking',
                        action='store_true', default=True,
                        help='Qwen3 chat template: inject <think> tags. Default True.')
    parser.add_argument('--no-thinking', dest='enable_thinking',
                        action='store_false',
                        help='Disable Qwen3 thinking mode in chat template.')
    parser.add_argument('--verify-mode',
                        choices=['auto', 'greedy', 'simple', 'ratio', 'ratio_fix'],
                        default='auto',
                        help='Target verify rule: auto=greedy if T=0 else simple; '
                             'greedy=argmax match; simple=r ≤ p_t; ratio=r ≤ p_t/p_d; '
                             'ratio_fix=ratio + correction from normalize(max(0, p_t - p_d)) '
                             '(exact SpecDec correction).')
    args = parser.parse_args()

    # Parse top_k (int or comma-separated list)
    if ',' in args.top_k:
        args.top_k = [int(k.strip()) for k in args.top_k.split(',') if k.strip()]
    else:
        args.top_k = int(args.top_k)

    benches = ['mt_bench', 'gsm8k', 'humaneval', 'qa', 'sum', 'alpaca', 'aime'] \
        if args.bench_name == 'all' else \
        [b.strip() for b in args.bench_name.split(',') if b.strip()]

    print(f"[TREE-EVAL] Target: {args.base_model_path}")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, torch_dtype=torch.float16,
        attn_implementation="sdpa").cuda()
    target_model.eval()

    print(f"[TREE-EVAL] Draft: {args.draft_model_path}")
    # If the checkpoint has draft.base./draft_model. prefixes, strip them into a
    # SIBLING directory instead of rewriting the source .bin in place (rewriting
    # in place races with concurrent trainers reading the same ckpt).
    ckpt_files = glob.glob(os.path.join(args.draft_model_path, "*.bin"))
    draft_load_path = args.draft_model_path
    for f in ckpt_files:
        state = torch.load(f, map_location="cpu")
        prefix_hit = None
        for prefix in ("draft.base.", "draft_model."):
            if any(k.startswith(prefix) for k in state):
                prefix_hit = prefix; break
        if prefix_hit is not None:
            fixed_dir = args.draft_model_path.rstrip('/') + '_stripped'
            os.makedirs(fixed_dir, exist_ok=True)
            new_f = os.path.join(fixed_dir, os.path.basename(f))
            if not os.path.exists(new_f):
                print(f"[TREE-EVAL] Stripping '{prefix_hit}' from {f} → {new_f}")
                stripped = {k.removeprefix(prefix_hit): v for k, v in state.items()}
                torch.save(stripped, new_f)
            # Copy config.json/tokenizer files (symlink to keep sibling small)
            for sib in os.listdir(args.draft_model_path):
                if sib.endswith('.bin'): continue
                dst = os.path.join(fixed_dir, sib)
                if not os.path.exists(dst):
                    os.symlink(os.path.join(args.draft_model_path, sib), dst)
            draft_load_path = fixed_dir
        break   # one .bin per ckpt dir

    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_load_path, torch_dtype=torch.float16,
        attn_implementation="sdpa").cuda()
    draft_model.eval()

    num_layers = len(draft_model.model.layers)
    print(f"[TREE-EVAL] Draft has {num_layers} layers")
    print(f"[TREE-EVAL] Tree: gamma={args.gamma}, top_k={args.top_k}, "
          f"budget={args.budget}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path, trust_remote_code=True)

    eos_id = getattr(target_model.config, 'eos_token_id', None)

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    for bench in benches:
        questions = load_questions(bench)
        if not questions:
            continue
        if args.num_samples is not None:
            questions = questions[:args.num_samples]
            print(f"[TREE-EVAL] Limited to first {len(questions)} samples")

        all_results[bench] = {}

        def tokenize_prompt(q):
            prompt = q.get("turns", [q.get("prompt", "")])[0] \
                if "turns" in q else q.get("prompt", "")
            msgs = [{"role": "user", "content": prompt}]
            try:
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=args.enable_thinking)
            except TypeError:
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)
            return tokenizer(text, return_tensors="pt",
                             add_special_tokens=False).input_ids.to(target_model.device)

        # Baseline chain (also implied by --baseline-only)
        if args.run_baseline or args.baseline_only:
            print(f"\n[TREE-EVAL] {bench} baseline chain (gamma={args.baseline_gamma})")
            agg = {"total_tokens": 0, "total_rounds": 0, "total_accepted": 0, "total_time": 0.0}
            all_al = []
            for qi, q in enumerate(tqdm(questions, desc=f"baseline/{bench}")):
                _seed_all(args.seed * 1000 + qi)
                ids = tokenize_prompt(q)
                r = baseline_chain_decode(
                    target_model, draft_model, ids,
                    args.max_new_tokens, args.baseline_gamma,
                    temperature=args.temperature,
                    draft_mode=args.draft_mode,
                    verify_mode=args.verify_mode,
                    draft_top_k=args.draft_top_k,
                    target_top_k=args.target_top_k)
                for k in agg:
                    agg[k] += r[k]
                all_al.extend(r.get("accept_lengths", []))
                if qi < args.print_samples:
                    prompt_txt = q.get("turns", [q.get("prompt", "")])[0]
                    full_prompt_txt = tokenizer.decode(
                        ids[0].tolist(), skip_special_tokens=False)
                    gen_txt = tokenizer.decode(
                        r["output_ids"], skip_special_tokens=False)
                    print(f"\n  --- baseline sample {qi} ---")
                    print(f"  QUESTION: {prompt_txt}")
                    print(f"  FULL PROMPT (tokenized): {full_prompt_txt}")
                    print(f"  GEN ({len(r['output_ids'])} tok, "
                          f"α={r['mean_alpha']:.2f}):\n{gen_txt}\n")
            alpha = agg["total_accepted"] / max(agg["total_rounds"], 1)
            tps = agg["total_tokens"] / max(agg["total_time"], 1e-6)
            al_stats = _al_stats(all_al)
            all_results[bench]["baseline_chain"] = {
                "mean_alpha": alpha, "tokens_per_sec": tps,
                "total_time": agg["total_time"],
                **al_stats,
                "accept_lengths": all_al,
            }
            print(f"  baseline: α={alpha:.3f}  median={al_stats['median_al']:.1f}  "
                  f"p25={al_stats['p25_al']:.1f}  p75={al_stats['p75_al']:.1f}  "
                  f"std={al_stats['std_al']:.3f}  n={al_stats['n_windows']}  tok/s={tps:.1f}")

        # Tree decode (skip if baseline-only)
        if args.baseline_only:
            out_path = os.path.join(args.output_dir, f"{args.tag}.json")
            with open(out_path, 'w') as f:
                json.dump({"tag": args.tag, "results": all_results}, f, indent=2)
            continue

        print(f"\n[TREE-EVAL] {bench} tree decode "
              f"(γ={args.gamma}, top_k={args.top_k}, budget={args.budget})")
        agg = {"total_tokens": 0, "total_rounds": 0, "total_accepted": 0,
               "total_time": 0.0, "draft_time": 0.0, "target_time": 0.0}
        all_al = []
        for qi, q in enumerate(tqdm(questions, desc=f"tree/{bench}")):
            _seed_all(args.seed * 1000 + qi)
            ids = tokenize_prompt(q)
            r = spec_decode_tree_smalllm(
                target_model, draft_model, ids,
                args.max_new_tokens, num_layers,
                args.gamma, args.top_k, args.budget,
                eos_token_id=eos_id,
                temperature=args.temperature,
                draft_mode=args.draft_mode,
                verify_mode=args.verify_mode,
                draft_top_k_vocab=args.draft_top_k,
                target_top_k=args.target_top_k)
            for k in agg:
                agg[k] += r[k]
            all_al.extend(r.get("accept_lengths", []))
            if qi < args.print_samples:
                prompt_txt = q.get("turns", [q.get("prompt", "")])[0]
                full_prompt_txt = tokenizer.decode(
                    ids[0].tolist(), skip_special_tokens=False)
                gen_txt = tokenizer.decode(
                    r["output_ids"], skip_special_tokens=False)
                print(f"\n  --- tree sample {qi} ---")
                print(f"  QUESTION: {prompt_txt}")
                print(f"  FULL PROMPT (tokenized): {full_prompt_txt}")
                print(f"  GEN ({len(r['output_ids'])} tok, "
                      f"α={r['mean_alpha']:.2f}):\n{gen_txt}\n")
        alpha = agg["total_accepted"] / max(agg["total_rounds"], 1)
        tps = agg["total_tokens"] / max(agg["total_time"], 1e-6)
        df = agg["draft_time"] / max(agg["total_time"], 1e-6)
        al_stats = _al_stats(all_al)
        all_results[bench]["tree"] = {
            "mean_alpha": alpha, "tokens_per_sec": tps,
            "total_time": agg["total_time"],
            "draft_time": agg["draft_time"],
            "target_time": agg["target_time"],
            "draft_fraction": df,
            "gamma": args.gamma, "top_k": args.top_k, "budget": args.budget,
            **al_stats,
            "accept_lengths": all_al,
        }
        print(f"  tree: α={alpha:.3f}  median={al_stats['median_al']:.1f}  "
              f"p25={al_stats['p25_al']:.1f}  p75={al_stats['p75_al']:.1f}  "
              f"std={al_stats['std_al']:.3f}  n={al_stats['n_windows']}  "
              f"tok/s={tps:.1f}  draft={df:.1%}")

        # Save incrementally
        out_path = os.path.join(args.output_dir, f"{args.tag}.json")
        with open(out_path, 'w') as f:
            json.dump({"tag": args.tag, "results": all_results}, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print(f"  SMALL LM TREE SPEC DECODE SUMMARY: {args.tag}")
    for bench, modes in all_results.items():
        print(f"\n  {bench}:")
        for mode_key, stats in modes.items():
            extra = ""
            if 'draft_fraction' in stats:
                extra = f"  draft={stats['draft_fraction']:.1%}"
            print(f"    {mode_key:<15} α={stats['mean_alpha']:.3f}  "
                  f"tok/s={stats['tokens_per_sec']:.1f}{extra}")
            print(f"    accepted_length={stats['mean_alpha']:.6f}")
    print(f"{'='*70}")
    print(f"[TREE-EVAL] Saved to {out_path}")


if __name__ == "__main__":
    main()
