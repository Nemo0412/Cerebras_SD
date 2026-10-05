"""
Diagnostic script: compare baseline vs trained draft checkpoint at token level.

Single-GPU, no DeepSpeed. Loads both checkpoints sequentially, runs
teacher-forced forward on the same data, compares per-position logits.

Uses evaluation datasets (question.jsonl) with target-model-generated
responses as reference text, matching the actual speculative decoding
evaluation setup.

Outputs:
  1. Four-quadrant matrix (A=keep, B=degrade, C=fix, D=still_wrong) per step
  2. Logit gap distribution at rejected positions (before/after training)
  3. Per-baseline-τ bucket analysis
  4. Concrete examples with token-level detail

Usage:
  python sdpo/diagnose.py \
    --basepath  meta-llama/Llama-3.1-8B-Instruct \
    --draftpath yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
    --trainedpath sdpo_checkpoints/state_0 \
    --datasets mt_bench \
    --num_samples 10 \
    --gamma 7
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as sf_load
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from traineagle3.configs import EConfig
from rejection_model import RejectionModel

torch.backends.cuda.matmul.allow_tf32 = True

# ── CLI ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--basepath', required=True, help='target LLM path')
parser.add_argument('--draftpath', required=True, help='baseline draft checkpoint')
parser.add_argument('--trainedpath', required=True, help='trained draft checkpoint')
parser.add_argument('--datasets', type=str, default='mt_bench',
                    help='comma-separated dataset names under data_dir (e.g. mt_bench,alpaca)')
parser.add_argument('--data_dir', type=str,
                    default=os.path.join(os.path.dirname(__file__), '..', 'data'),
                    help='root data directory containing dataset subdirs')
parser.add_argument('--cache_path', type=str, default='cache.pt',
                    help='path to d2t/t2d vocab mapping cache')
parser.add_argument('--num_samples', type=int, default=10)
parser.add_argument('--gamma', type=int, default=7)
parser.add_argument('--max_len', type=int, default=2048)
parser.add_argument('--max_new_tokens', type=int, default=256,
                    help='max tokens to generate per question')
parser.add_argument('--config_path', type=str,
                    default=os.path.join(os.path.dirname(__file__), 'config.json'))
parser.add_argument('--output', type=str, default=None,
                    help='output file (default: diagnose_report_{datasets}.txt)')
args = parser.parse_args()
if args.output is None:
    args.output = f"diagnose_report_{args.datasets.replace(',', '_')}.txt"

device = torch.device('cuda:0')
report_file = open(args.output, 'w')


def log(msg=''):
    """Print to both stdout and report file."""
    print(msg)
    report_file.write(msg + '\n')


# ── System prompt (must match evaluation/gen_ea_answer_llama3chat.py) ─
SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as "
    "helpfully as possible, while being safe.  Your answers should not "
    "include any harmful, unethical, racist, sexist, toxic, dangerous, or "
    "illegal content. Please ensure that your responses are socially "
    "unbiased and positive in nature.\n\nIf a question does not make any "
    "sense, or is not factually coherent, explain why instead of answering "
    "something not correct. If you don't know the answer to a question, "
    "please don't share false information."
)


# ── Data loading (question.jsonl, matching eval) ─────────────────────
def load_questions(data_dir, datasets, tokenizer, num_samples):
    """Load question.jsonl files and tokenize prompts matching eval."""
    questions = []
    for ds_name in datasets.split(','):
        ds_name = ds_name.strip()
        path = os.path.join(data_dir, ds_name, 'question.jsonl')
        if not os.path.exists(path):
            print(f"  [WARN] {path} not found, skipping")
            continue
        with open(path) as f:
            for line in f:
                q = json.loads(line)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q["turns"][0]},
                ]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                prompt_ids = tokenizer(
                    prompt, add_special_tokens=False, return_tensors="pt"
                ).input_ids  # [1, P]
                questions.append({
                    "question_id": q["question_id"],
                    "dataset": ds_name,
                    "prompt_ids": prompt_ids,
                })
                if num_samples and len(questions) >= num_samples:
                    break
        if num_samples and len(questions) >= num_samples:
            break
    return questions[:num_samples] if num_samples else questions


# ── Greedy generation with target model ──────────────────────────────
@torch.no_grad()
def generate_greedy(target_model, prompt_ids, tokenizer, max_new_tokens, max_len):
    """Generate greedy response using target LLM (no KV cache).

    The target model uses a custom KV cache (modeling_llama_kv.py) that
    doesn't support standard HuggingFace past_key_values, so we run
    full-sequence forward at each step.

    Returns (full_ids [1, P+G], prompt_len int).
    """
    stop_ids = {tokenizer.eos_token_id}
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot is not None and eot != tokenizer.unk_token_id:
        stop_ids.add(eot)

    prompt_len = prompt_ids.shape[1]
    cap = min(max_new_tokens, max_len - prompt_len)
    if cap <= 0:
        return prompt_ids, prompt_len

    orig_hs = target_model.config.output_hidden_states
    target_model.config.output_hidden_states = False
    try:
        ids = prompt_ids
        for _ in range(cap):
            out = target_model(input_ids=ids)
            next_tok = out.logits[:, -1:].argmax(dim=-1)
            ids = torch.cat([ids, next_tok], dim=1)
            if next_tok.item() in stop_ids:
                break
    finally:
        target_model.config.output_hidden_states = orig_hs

    return ids, prompt_len


def prepare_sample(full_ids, prompt_len):
    """Build input tensors for collect_logits from generated sequence."""
    L = full_ids.shape[1]
    attention_mask = torch.ones(1, L, dtype=torch.long)
    loss_mask = torch.zeros(1, L, dtype=torch.long)
    loss_mask[0, prompt_len:] = 1
    # Last position is garbage after dataprepare's left-shift padding
    loss_mask[0, -1] = 0
    return full_ids, attention_mask, loss_mask


# ── Path resolution ───────────────────────────────────────────────────
def resolve_path(path, label):
    """Resolve HF model ID to local path via snapshot_download if needed."""
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
        return path
    print(f"  {label}: {path} is not a local directory, downloading from HuggingFace...")
    local = snapshot_download(path)
    print(f"  {label}: resolved to {local}")
    return local


# ── Model creation ────────────────────────────────────────────────────
def create_model(basepath, draftpath, config_path, gamma):
    """Create a RejectionModel (with target LLM loaded)."""
    draft_cfg_path = os.path.join(draftpath, "config.json")
    config = EConfig.from_pretrained(
        draft_cfg_path if os.path.exists(draft_cfg_path) else config_path)
    config.gradient_checkpointing = False

    ds_config_dummy = {"zero_optimization": {"stage": 0}}
    train_config = SimpleNamespace(
        bs=1, num_epochs=1, num_workers=0, max_len=args.max_len,
        config_path=config_path, gradient_checkpointing=False,
        eagle_coef=0, kl_coef=0, gamma=gamma, baseline=None,
    )
    model = RejectionModel(config, ds_config_dummy, train_config,
                           path=basepath, load_emb=True, load_head=True)
    model.length = gamma
    return model


def load_vocab_mapping(model, cache_path):
    """Load d2t/t2d vocab mapping from cache.pt."""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Vocab cache not found at {cache_path}. "
            "Run training first to generate cache.pt, or specify --cache_path.")
    cache = torch.load(cache_path, map_location="cpu")
    d2t = cache["d2t"]
    t2d = cache["t2d"]
    model.register_buffer("d2t", d2t)
    model.register_buffer("t2d", t2d)
    print(f"  Loaded vocab mapping from {cache_path} "
          f"(draft_vocab={len(d2t)}, target_vocab={len(t2d)})")


def load_weights(model, ckpt_path):
    """Load draft weights from a checkpoint directory."""
    sf_path = os.path.join(ckpt_path, "model.safetensors")
    bin_path = os.path.join(ckpt_path, "pytorch_model.bin")
    if os.path.exists(sf_path):
        state = sf_load(sf_path, device="cpu")
    elif os.path.exists(bin_path):
        state = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No weights in {ckpt_path}")

    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    # Remove vocab buffers (they're registered separately)
    for vk in ("d2t", "t2d"):
        state.pop(vk, None)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"  [WARN] Unexpected keys: {unexpected}")
    print(f"  Loaded weights from {ckpt_path} (missing: {len(missing)})")


# ── Teacher-forced forward: collect per-step logits ───────────────────
@torch.no_grad()
def collect_logits(model, input_ids, attention_mask, loss_mask):
    """
    Run teacher-forced draft forward, return per-step full logits.

    Returns:
        all_logits: list of [B, L, V_draft] tensors, length gamma
        target_greedy: [B, L] target LLM greedy tokens (full vocab)
        loss_mask_2d: [B, L] bool
    """
    hidden_states, target, loss_mask_3d, input_ids_shifted = model.dataprepare(
        input_ids, attention_mask, loss_mask)
    loss_mask_2d = loss_mask_3d.squeeze(-1).bool()
    B, L, _ = hidden_states.shape
    dev = hidden_states.device
    gamma = model.length

    hidden_states = hidden_states.to(model.fc.weight.dtype)
    hs_projected = model.fc(hidden_states)

    attn_mask = model._prepare_decoder_attention_mask(
        attention_mask, (B, L), hs_projected, 0)
    position_ids = torch.arange(L, dtype=torch.long, device=dev).unsqueeze(0)

    target_greedy = target.argmax(dim=-1)  # [B, L], full vocab IDs

    all_logits = []
    cache_hidden = [[], []]
    cur_ids = input_ids_shifted
    cur_tgt = target_greedy.clone()

    for idx in range(gamma):
        embeds = model.embed_tokens(cur_ids).to(hs_projected.dtype)
        layer_out, cache_hidden = model.midlayer(
            input_emb=embeds, hidden_states=hs_projected if idx == 0 else cur_hs,
            cache_hidden=cache_hidden, attention_mask=attn_mask,
            position_ids=position_ids, past_key_value=None,
            output_attentions=False, use_cache=True)
        cur_hs = layer_out[0]
        logits = model.lm_head(model.norm(cur_hs)).float()  # [B, L, V_draft]
        all_logits.append(logits.cpu())

        if idx < gamma - 1:
            cur_ids = cur_tgt.detach()
            cur_tgt = torch.cat(
                [cur_tgt[:, 1:], torch.zeros_like(cur_tgt[:, :1])], dim=1)

    return all_logits, target_greedy.cpu(), loss_mask_2d.cpu()


# ── Acceptance check using vocab mapping ──────────────────────────────
def compute_accepts(all_logits, target_greedy, d2t, t2d):
    """
    Check argmax acceptance at each step.
    Returns accepts [B, L, gamma] bool, draft_argmax_full [B, L, gamma] long.
    """
    gamma = len(all_logits)
    B, L = target_greedy.shape
    accepts = []
    draft_argmaxes = []
    cur_tgt = target_greedy.clone()

    for idx in range(gamma):
        logits = all_logits[idx]  # [B, L, V_draft]
        draft_d = logits.argmax(dim=-1)  # [B, L]
        draft_full = draft_d + d2t[draft_d]  # map to full vocab
        tgt_in = t2d[cur_tgt]  # is target in draft vocab?
        accept = (draft_full == cur_tgt) & tgt_in
        accepts.append(accept)
        draft_argmaxes.append(draft_full)

        if idx < gamma - 1:
            cur_tgt = torch.cat(
                [cur_tgt[:, 1:], torch.zeros_like(cur_tgt[:, :1])], dim=1)

    accepts = torch.stack(accepts, dim=2)  # [B, L, gamma]
    draft_argmaxes = torch.stack(draft_argmaxes, dim=2)  # [B, L, gamma]
    return accepts, draft_argmaxes


def compute_tau(accepts):
    """τ = length of accepted prefix."""
    return torch.cumprod(accepts.float(), dim=2).sum(dim=2).long()  # [B, L]


# ── Logit gap at rejected positions ───────────────────────────────────
def compute_logit_gaps(all_logits, target_greedy, d2t, t2d, accepts):
    """
    At each rejected position (first rejection, step τ+1), compute:
      gap = max_logit(y != y*) - logit(y*)
    Returns list of gap values.
    """
    gamma = len(all_logits)
    B, L = target_greedy.shape
    tau = compute_tau(accepts)  # [B, L]

    # Build full2draft mapping
    full2draft = torch.full((t2d.shape[0],), -1, dtype=torch.long)
    draft_ids = torch.arange(len(d2t))
    full2draft[draft_ids + d2t] = draft_ids

    gaps = []
    cur_tgt = target_greedy.clone()

    for idx in range(gamma):
        # Positions where this step is the first rejection (tau == idx)
        is_first_rej = (tau == idx)  # [B, L]
        if not is_first_rej.any():
            if idx < gamma - 1:
                cur_tgt = torch.cat(
                    [cur_tgt[:, 1:], torch.zeros_like(cur_tgt[:, :1])], dim=1)
            continue

        logits = all_logits[idx]  # [B, L, V_draft]
        tgt_d = full2draft[cur_tgt].clamp(min=0)  # [B, L]
        tgt_in = t2d[cur_tgt]  # [B, L]

        for b in range(B):
            for l in range(L):
                if not is_first_rej[b, l] or not tgt_in[b, l]:
                    continue
                logit_vec = logits[b, l]  # [V_draft]
                tgt_idx = tgt_d[b, l].item()
                tgt_logit = logit_vec[tgt_idx].item()
                # Max logit excluding target
                logit_vec_masked = logit_vec.clone()
                logit_vec_masked[tgt_idx] = float('-inf')
                max_other = logit_vec_masked.max().item()
                gaps.append({
                    'gap': max_other - tgt_logit,
                    'tgt_logit': tgt_logit,
                    'max_other': max_other,
                    'step': idx,
                    'b': b, 'l': l,
                    'tgt_rank': (logit_vec > tgt_logit).sum().item() + 1,
                })

        if idx < gamma - 1:
            cur_tgt = torch.cat(
                [cur_tgt[:, 1:], torch.zeros_like(cur_tgt[:, :1])], dim=1)

    return gaps


# ── Main ──────────────────────────────────────────────────────────────
print("=" * 70)
print("SDPO Diagnostic: Baseline vs Trained Checkpoint")
print("=" * 70)

basepath = resolve_path(args.basepath, "basepath")
draftpath = resolve_path(args.draftpath, "draftpath")

# Load tokenizer and questions
tokenizer = AutoTokenizer.from_pretrained(basepath)
questions = load_questions(args.data_dir, args.datasets, tokenizer, args.num_samples)
print(f"Loaded {len(questions)} questions from: {args.datasets}")

# Create model (target LLM loaded once, draft weights swapped)
print("\nCreating model (loading target LLM)...")
model = create_model(basepath, draftpath, args.config_path, args.gamma)
load_vocab_mapping(model, args.cache_path)
model.eval()
model.to(device)

d2t = model.d2t.cpu()
t2d = model.t2d.cpu()

# ── Generate reference responses with target LLM ─────────────────────
print("\nGenerating reference responses with target LLM (greedy)...")
samples = []
for q in questions:
    prompt_ids = q["prompt_ids"].to(device)
    full_ids, prompt_len = generate_greedy(
        model.target_model, prompt_ids, tokenizer,
        args.max_new_tokens, args.max_len)
    input_ids, attn_mask, loss_mask = prepare_sample(full_ids, prompt_len)
    n_gen = full_ids.shape[1] - prompt_len
    samples.append({
        "input_ids": input_ids, "attention_mask": attn_mask,
        "loss_mask": loss_mask, "question_id": q["question_id"],
        "dataset": q["dataset"], "prompt_len": prompt_len,
    })
    print(f"  Q{q['question_id']} ({q['dataset']}): "
          f"prompt={prompt_len}, generated={n_gen}, total={full_ids.shape[1]}")

# ── Run baseline ──────────────────────────────────────────────────────
print("\n--- Loading BASELINE weights ---")
load_weights(model, draftpath)
model.to(device)

baseline_results = []
for i, sample in enumerate(samples):
    input_ids = sample['input_ids'].to(device)
    attn_mask = sample['attention_mask'].to(device)
    loss_mask = sample['loss_mask'].to(device)
    logits_list, tgt_greedy, lm2d = collect_logits(model, input_ids, attn_mask, loss_mask)
    baseline_results.append((logits_list, tgt_greedy, lm2d))
    print(f"  Baseline Q{sample['question_id']}: {lm2d.sum().item():.0f} valid positions")

# ── Run trained ───────────────────────────────────────────────────────
print("\n--- Loading TRAINED weights ---")
load_weights(model, args.trainedpath)
model.to(device)

trained_results = []
for i, sample in enumerate(samples):
    input_ids = sample['input_ids'].to(device)
    attn_mask = sample['attention_mask'].to(device)
    loss_mask = sample['loss_mask'].to(device)
    logits_list, tgt_greedy, lm2d = collect_logits(model, input_ids, attn_mask, loss_mask)
    trained_results.append((logits_list, tgt_greedy, lm2d))

# ── Analysis ──────────────────────────────────────────────────────────
log("\n" + "=" * 70)
log("ANALYSIS")
log("=" * 70)

# Accumulators
quad_per_step = defaultdict(lambda: {'A': 0, 'B': 0, 'C': 0, 'D': 0})
quad_total = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
baseline_gaps_all = []
trained_gaps_all = []
tau_bucket_changes = defaultdict(list)  # baseline_tau -> list of trained_tau
per_sample_details = []

for i in range(len(samples)):
    bl_logits, tgt_greedy, lm2d = baseline_results[i]
    tr_logits, _, _ = trained_results[i]

    bl_accepts, bl_argmax = compute_accepts(bl_logits, tgt_greedy, d2t, t2d)
    tr_accepts, tr_argmax = compute_accepts(tr_logits, tgt_greedy, d2t, t2d)
    bl_tau = compute_tau(bl_accepts)  # [1, L]
    tr_tau = compute_tau(tr_accepts)  # [1, L]

    mask = lm2d[0]  # [L]

    # ── Diagnostic 1: Four-quadrant matrix per step ───────────────
    for step in range(args.gamma):
        bl_ok = bl_accepts[0, :, step] & mask
        tr_ok = tr_accepts[0, :, step] & mask
        bl_no = (~bl_accepts[0, :, step]) & mask
        tr_no = (~tr_accepts[0, :, step]) & mask

        a = (bl_ok & tr_ok).sum().item()
        b = (bl_ok & tr_no).sum().item()  # degradation
        c = (bl_no & tr_ok).sum().item()  # fix
        d = (bl_no & tr_no).sum().item()

        quad_per_step[step]['A'] += a
        quad_per_step[step]['B'] += b
        quad_per_step[step]['C'] += c
        quad_per_step[step]['D'] += d
        quad_total['A'] += a
        quad_total['B'] += b
        quad_total['C'] += c
        quad_total['D'] += d

    # ── Diagnostic 2: Logit gap at first rejection ────────────────
    bl_gap_list = compute_logit_gaps(bl_logits, tgt_greedy, d2t, t2d, bl_accepts)
    tr_gap_list = compute_logit_gaps(tr_logits, tgt_greedy, d2t, t2d, bl_accepts)
    # NOTE: both use bl_accepts for tau — we want to compare the SAME positions
    # Tag with sample index to avoid cross-sample key collision
    for g in bl_gap_list:
        g['sample'] = i
    for g in tr_gap_list:
        g['sample'] = i
    baseline_gaps_all.extend(bl_gap_list)
    trained_gaps_all.extend(tr_gap_list)

    # ── Diagnostic 3: Per-baseline-τ bucket ───────────────────────
    for l in range(mask.shape[0]):
        if not mask[l]:
            continue
        bt = bl_tau[0, l].item()
        tt = tr_tau[0, l].item()
        tau_bucket_changes[bt].append(tt)

    # ── Collect per-sample detail for examples ────────────────────
    bl_mean = (bl_tau[0][mask].float().mean().item() if mask.any() else 0)
    tr_mean = (tr_tau[0][mask].float().mean().item() if mask.any() else 0)
    per_sample_details.append({
        'idx': i,
        'question_id': samples[i]['question_id'],
        'dataset': samples[i]['dataset'],
        'n_valid': mask.sum().item(),
        'bl_mean_tau': bl_mean,
        'tr_mean_tau': tr_mean,
        'delta_tau': tr_mean - bl_mean,
        'bl_tau': bl_tau[0].clone(),
        'tr_tau': tr_tau[0].clone(),
        'bl_accepts': bl_accepts[0].clone(),
        'tr_accepts': tr_accepts[0].clone(),
        'bl_argmax': bl_argmax[0].clone(),
        'tr_argmax': tr_argmax[0].clone(),
        'tgt_greedy': tgt_greedy[0].clone(),
        'mask': mask.clone(),
        'bl_logits': bl_logits,
        'tr_logits': tr_logits,
    })


# ══════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════

# ── 1. Four-quadrant matrix ───────────────────────────────────────────
log("\n┌─────────────────────────────────────────────────┐")
log("│  DIAGNOSTIC 1: Four-Quadrant Matrix (per step)  │")
log("└─────────────────────────────────────────────────┘")
log(f"{'step':>4}  {'A(keep)':>10}  {'B(degrade)':>10}  {'C(fix)':>10}  {'D(still_wrong)':>14}  {'net(C-B)':>8}")
log("-" * 70)
for step in range(args.gamma):
    q = quad_per_step[step]
    total = q['A'] + q['B'] + q['C'] + q['D']
    net = q['C'] - q['B']
    log(f"{step:>4}  {q['A']:>10}  {q['B']:>10}  {q['C']:>10}  {q['D']:>14}  {net:>+8}")
log("-" * 70)
total_all = sum(quad_total.values())
log(f"{'ALL':>4}  {quad_total['A']:>10}  {quad_total['B']:>10}  {quad_total['C']:>10}  {quad_total['D']:>14}  {quad_total['C']-quad_total['B']:>+8}")
if total_all > 0:
    log(f"\n  A(keep):       {quad_total['A']/total_all*100:.1f}%")
    log(f"  B(degrade):    {quad_total['B']/total_all*100:.1f}%")
    log(f"  C(fix):        {quad_total['C']/total_all*100:.1f}%")
    log(f"  D(still_wrong):{quad_total['D']/total_all*100:.1f}%")
    log(f"  Net effect:    C-B = {quad_total['C']-quad_total['B']:+d} positions")

# ── 2. Logit gap distribution ─────────────────────────────────────────
log("\n┌─────────────────────────────────────────────────┐")
log("│  DIAGNOSTIC 2: Logit Gap at First Rejection     │")
log("└─────────────────────────────────────────────────┘")

if baseline_gaps_all:
    bl_lookup = {}
    for g in baseline_gaps_all:
        bl_lookup[(g['sample'], g['b'], g['l'], g['step'])] = g
    tr_lookup = {}
    for g in trained_gaps_all:
        tr_lookup[(g['sample'], g['b'], g['l'], g['step'])] = g

    paired_keys = set(bl_lookup.keys()) & set(tr_lookup.keys())
    log(f"  Paired positions: {len(paired_keys)}")

    bl_gaps = np.array([bl_lookup[k]['gap'] for k in paired_keys])
    tr_gaps = np.array([tr_lookup[k]['gap'] for k in paired_keys])
    bl_ranks = np.array([bl_lookup[k]['tgt_rank'] for k in paired_keys])
    tr_ranks = np.array([tr_lookup[k]['tgt_rank'] for k in paired_keys])
    gap_delta = tr_gaps - bl_gaps

    log(f"\n  Baseline gap:  mean={bl_gaps.mean():.3f}, median={np.median(bl_gaps):.3f}")
    log(f"  Trained gap:   mean={tr_gaps.mean():.3f}, median={np.median(tr_gaps):.3f}")
    log(f"  Gap delta:     mean={gap_delta.mean():.3f}, median={np.median(gap_delta):.3f}")
    log(f"  Positions where gap shrank:   {(gap_delta < 0).sum()}/{len(gap_delta)} ({(gap_delta < 0).mean()*100:.1f}%)")
    log(f"  Positions where gap grew:     {(gap_delta > 0).sum()}/{len(gap_delta)} ({(gap_delta > 0).mean()*100:.1f}%)")
    log(f"  Positions where argmax flipped (gap<0 after): {(tr_gaps < 0).sum()}/{len(tr_gaps)} ({(tr_gaps < 0).mean()*100:.1f}%)")

    log(f"\n  Baseline target rank: mean={bl_ranks.mean():.1f}, median={np.median(bl_ranks):.0f}")
    log(f"  Trained target rank:  mean={tr_ranks.mean():.1f}, median={np.median(tr_ranks):.0f}")

    log(f"\n  Baseline gap distribution:")
    for lo, hi, label in [(float('-inf'), 0, '  gap<0 (already correct)'),
                          (0, 1, '  0<gap<1 (close)'),
                          (1, 3, '  1<gap<3 (medium)'),
                          (3, 5, '  3<gap<5 (hard)'),
                          (5, float('inf'), '  gap>5 (very hard)')]:
        n = ((bl_gaps >= lo) & (bl_gaps < hi)).sum()
        log(f"    {label}: {n} ({n/len(bl_gaps)*100:.1f}%)")

    close_mask = bl_gaps < 1
    if close_mask.sum() > 0:
        flipped_in_close = (tr_gaps[close_mask] < 0).sum()
        log(f"\n  Among 'close' positions (baseline gap<1): {flipped_in_close}/{close_mask.sum()} flipped to correct")

    bl_tgt_logits = np.array([bl_lookup[k]['tgt_logit'] for k in paired_keys])
    tr_tgt_logits = np.array([tr_lookup[k]['tgt_logit'] for k in paired_keys])
    logit_delta = tr_tgt_logits - bl_tgt_logits
    log(f"\n  Target token logit change:")
    log(f"    mean delta: {logit_delta.mean():+.4f}")
    log(f"    positions where target logit increased: {(logit_delta > 0).sum()}/{len(logit_delta)} ({(logit_delta > 0).mean()*100:.1f}%)")
    log(f"    positions where target logit decreased: {(logit_delta < 0).sum()}/{len(logit_delta)} ({(logit_delta < 0).mean()*100:.1f}%)")

else:
    log("  No rejected positions found (all τ=γ?)")

# ── 3. Per-baseline-τ bucket analysis ─────────────────────────────────
log("\n┌─────────────────────────────────────────────────┐")
log("│  DIAGNOSTIC 3: τ Change by Baseline τ Bucket    │")
log("└─────────────────────────────────────────────────┘")
log(f"{'bl_τ':>4}  {'count':>6}  {'bl_mean':>8}  {'tr_mean':>8}  {'delta':>8}  {'improved':>8}  {'degraded':>8}  {'same':>8}")
log("-" * 80)
all_bl_taus = []
all_tr_taus = []
for bt in sorted(tau_bucket_changes.keys()):
    vals = tau_bucket_changes[bt]
    tr_vals = np.array(vals)
    n = len(vals)
    tr_mean = tr_vals.mean()
    improved = (tr_vals > bt).sum()
    degraded = (tr_vals < bt).sum()
    same = (tr_vals == bt).sum()
    log(f"{bt:>4}  {n:>6}  {bt:>8.1f}  {tr_mean:>8.2f}  {tr_mean-bt:>+8.2f}  {improved:>8}  {degraded:>8}  {same:>8}")
    all_bl_taus.extend([bt] * n)
    all_tr_taus.extend(vals)

all_bl_taus = np.array(all_bl_taus)
all_tr_taus = np.array(all_tr_taus)
log("-" * 80)
log(f"{'ALL':>4}  {len(all_bl_taus):>6}  {all_bl_taus.mean():>8.2f}  {all_tr_taus.mean():>8.2f}  {all_tr_taus.mean()-all_bl_taus.mean():>+8.3f}")

# ── 4. Concrete examples ─────────────────────────────────────────────
log("\n┌─────────────────────────────────────────────────┐")
log("│  DIAGNOSTIC 4: Concrete Examples                │")
log("└─────────────────────────────────────────────────┘")

# Sort by delta_tau to show worst degradation, best improvement, and median
sorted_details = sorted(per_sample_details, key=lambda x: x['delta_tau'])

examples_to_show = []
if len(sorted_details) >= 3:
    examples_to_show.append(('WORST DEGRADATION', sorted_details[0]))
    examples_to_show.append(('BEST IMPROVEMENT', sorted_details[-1]))
    examples_to_show.append(('MEDIAN', sorted_details[len(sorted_details)//2]))
else:
    for d in sorted_details:
        examples_to_show.append(('SAMPLE', d))

for label, detail in examples_to_show:
    log(f"\n  === {label} (Q{detail['question_id']}, {detail['dataset']}) ===")
    log(f"  Valid positions: {detail['n_valid']}")
    log(f"  Baseline mean τ: {detail['bl_mean_tau']:.3f}")
    log(f"  Trained mean τ:  {detail['tr_mean_tau']:.3f}")
    log(f"  Delta:           {detail['delta_tau']:+.3f}")

    mask = detail['mask']
    bl_tau = detail['bl_tau']
    tr_tau = detail['tr_tau']

    # Show τ distribution for this sample
    for t in range(args.gamma + 1):
        bl_n = ((bl_tau[mask] == t).sum().item())
        tr_n = ((tr_tau[mask] == t).sum().item())
        bar_bl = '█' * max(1, int(bl_n / detail['n_valid'] * 40))
        bar_tr = '▒' * max(1, int(tr_n / detail['n_valid'] * 40))
        log(f"    τ={t}: baseline {bl_n:>4} {bar_bl}")
        log(f"         trained  {tr_n:>4} {bar_tr}")

    # Pick a few positions to show token-level detail
    # Find positions with B (degradation) and C (fix)
    degrade_positions = []
    fix_positions = []
    for l in range(mask.shape[0]):
        if not mask[l]:
            continue
        if bl_tau[l] > tr_tau[l]:
            degrade_positions.append(l)
        elif bl_tau[l] < tr_tau[l]:
            fix_positions.append(l)

    full2draft = torch.full((t2d.shape[0],), -1, dtype=torch.long)
    draft_ids = torch.arange(len(d2t))
    full2draft[draft_ids + d2t] = draft_ids

    def show_position(l, label_pos):
        log(f"\n    Position {l} ({label_pos}):")
        log(f"      baseline τ={bl_tau[l].item()}, trained τ={tr_tau[l].item()}")
        tgt = detail['tgt_greedy']
        log(f"      {'step':>4}  {'bl_argmax':>12}  {'tr_argmax':>12}  {'target':>12}  {'bl_ok':>5}  {'tr_ok':>5}  {'bl_gap':>8}  {'tr_gap':>8}")
        log(f"      " + "-" * 80)
        cur_tgt_per_step = tgt.clone()
        for step in range(args.gamma):
            tgt_id = cur_tgt_per_step[l].item()
            tgt_token = tokenizer.decode([tgt_id]).strip()[:10]
            bl_arg_id = detail['bl_argmax'][l, step].item()
            tr_arg_id = detail['tr_argmax'][l, step].item()
            bl_tok = tokenizer.decode([bl_arg_id]).strip()[:10]
            tr_tok = tokenizer.decode([tr_arg_id]).strip()[:10]
            bl_ok = '✓' if detail['bl_accepts'][l, step] else '✗'
            tr_ok = '✓' if detail['tr_accepts'][l, step] else '✗'

            # Logit gaps
            bl_logits_step = detail['bl_logits'][step][0, l]
            tr_logits_step = detail['tr_logits'][step][0, l]
            tgt_d = full2draft[tgt_id].item()
            if tgt_d >= 0:
                bl_tgt_l = bl_logits_step[tgt_d].item()
                tr_tgt_l = tr_logits_step[tgt_d].item()
                bl_max_other = bl_logits_step.clone()
                bl_max_other[tgt_d] = float('-inf')
                tr_max_other = tr_logits_step.clone()
                tr_max_other[tgt_d] = float('-inf')
                bl_g = bl_max_other.max().item() - bl_tgt_l
                tr_g = tr_max_other.max().item() - tr_tgt_l
                bl_gap_str = f"{bl_g:+.2f}"
                tr_gap_str = f"{tr_g:+.2f}"
            else:
                bl_gap_str = "OOV"
                tr_gap_str = "OOV"

            log(f"      {step:>4}  {bl_tok:>12}  {tr_tok:>12}  {tgt_token:>12}  {bl_ok:>5}  {tr_ok:>5}  {bl_gap_str:>8}  {tr_gap_str:>8}")

            if step < args.gamma - 1:
                cur_tgt_per_step = torch.cat(
                    [cur_tgt_per_step[1:], torch.zeros(1, dtype=cur_tgt_per_step.dtype)])

    # Show up to 2 degradation and 2 fix positions
    for l in degrade_positions[:2]:
        show_position(l, "DEGRADED")
    for l in fix_positions[:2]:
        show_position(l, "FIXED")
    # If neither, show a random valid position
    if not degrade_positions and not fix_positions:
        valid_positions = mask.nonzero().squeeze(-1).tolist()
        if valid_positions:
            show_position(valid_positions[0], "UNCHANGED")

log("\n" + "=" * 70)
log("DIAGNOSTIC COMPLETE")
log("=" * 70)

report_file.close()
print(f"\nReport saved to: {args.output}")
