"""
Small LM Draft Training Script.

Train a small LLM (e.g., Qwen3-1.7B) as draft for speculative decoding
with a large LLM (e.g., Qwen3-8B) as target. Same tokenizer/vocab required.

L_total = L_KL + sigmoid_coef * L_acceptance_length_v2

Usage:
  deepspeed sdpo/main_small_lm.py \
    --basepath Qwen/Qwen3-8B \
    --draftpath Qwen/Qwen3-1.7B \
    --trainpath sdpo/data/mixed_train_10K.jsonl \
    --testpath  sdpo/data/mixed_val_80.jsonl \
    --deepspeed_config sdpo/sdpo_config_qwen3.json \
    --savedir /scratch/tx856/spec_reason/scratch/loss_train_smalllm/test \
    --sigmoid_coef 0.1 --T_max 0.1 --T_min 0.1 --gamma 7 --num_epochs 3
"""

import argparse
import json
import math
import os
import re
import shutil
import sys

import deepspeed
import random

import numpy as np
import torch
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from small_lm_model import SmallLMDraftModel

torch.backends.cuda.matmul.allow_tf32 = True

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--basepath', required=True, help='Target LLM (e.g., Qwen/Qwen3-8B)')
parser.add_argument('--draftpath', required=True, help='Draft LLM (e.g., Qwen/Qwen3-1.7B)')
parser.add_argument('--trainpath', required=True)
parser.add_argument('--testpath', required=True)
parser.add_argument('--savedir', default='/scratch/tx856/spec_reason/scratch/loss_train_smalllm')
parser.add_argument('--save_epochs', type=str, default=None,
                    help='Comma-separated 0-indexed epoch indices to save (e.g. "2,5"). '
                         'Default: save every epoch.')
parser.add_argument('--save_steps', type=int, default=0,
                    help='If >0, also save mid-epoch every N global steps '
                         '(as state_step{N}). Useful for slow training that may '
                         'timeout before finishing an epoch.')
parser.add_argument('--sigmoid_coef', type=float, default=0.1)
parser.add_argument('--T_max', type=float, default=0.1)
parser.add_argument('--T_min', type=float, default=0.1)
parser.add_argument('--prob_T_max', type=float, default=1.0,
                    help='Initial T applied to softmax(z/T) for AL_TV / AL_KL / WKL aux losses. '
                         'Linearly anneals to --prob_T_min over training. Default 1.0 (no schedule).')
parser.add_argument('--prob_T_min', type=float, default=1.0,
                    help='Final T for prob-space losses (end of training). '
                         'Recommended schedule: T_max=1.0 → T_min=0.01 (sharpen toward argmax).')
parser.add_argument('--draft_T_max', type=float, default=1.0,
                    help='Initial T applied to draft log_softmax in KL anchor loss '
                         '(draft_logp = log_softmax(z_d / T)). Default 1.0 (no scaling). '
                         'Higher T softens draft prob → KL forces wider logit gap.')
parser.add_argument('--draft_T_min', type=float, default=1.0,
                    help='Final T for draft softmax (linearly anneals from --draft_T_max).')
parser.add_argument('--eal_weight', type=float, default=0.5,
                    help='λ_eal: weight of EAL term in aux_loss=v4eal '
                         '(L = cumprod(β_v4) + λ_eal · cumprod(β_eal)).')
parser.add_argument('--gamma', type=int, default=7)
parser.add_argument('--max_len', type=int, default=2048)
parser.add_argument('--max_train_samples', type=int, default=None)
parser.add_argument('--num_epochs', type=int, default=3)
parser.add_argument('--baseline', type=str, default=None, choices=['eagle_only', 'ce_only'],
                    help='Legacy: eagle_only → --anchor kl --aux_loss none; ce_only → --anchor ce --aux_loss none')
parser.add_argument('--anchor', type=str, default='kl',
                    choices=['kl', 'ce', 'eal', 'none', 'kl_rkl'],
                    help='Main loss: kl (soft-target KL) / ce (hard-label CE) / '
                         'eal (Expected Accept Length over γ-window) / '
                         'none (skip anchor entirely; e.g., GRPO-only training) / '
                         'kl_rkl (forward KL on accepted positions, reverse KL on '
                         'rejected positions; accept mask is detached).')
parser.add_argument('--aux_loss', type=str, default='v2',
                    choices=['none', 'v2', 'v4', 'v4_w1', 'v4eal', 'v5', 'v6', 'v7', 'v8',
                             'tv', 'eal', 'eal_w1', 'peal',
                             'al_tv', 'al_kl', 'wkl',
                             'al_tv_intersect_topk', 'al_tv_target_topk',
                             'al_tv_target_topp',
                             'al_tv_target_topk_topp',
                             'al_tv_renewal',
                             'al_tv_target_topk_renewal',
                             'tree_v4', 'tree_altv', 'tree_ce', 'tree_altv_wk',
                             'tree_ce_hard', 'tree_ce_soft', 'altv_p_tree_altv',
                             'topk_softce',
                             'acceptance_length_v2', 'acceptance_length_v4'],
                    help='Auxiliary loss: none / v2 / v4 / v5 / v6 / v7 / v8 / tv / eal / '
                         'peal / al_tv / al_kl / wkl / tree_v4 / tree_altv / tree_ce / '
                         'tree_altv_wk / tree_ce_hard / tree_ce_soft / altv_p_tree_altv / '
                         'topk_softce. tree_* variants require --top_k_per_depth.')
parser.add_argument('--anchor_weight', type=str, default='none',
                    choices=['none', 'uniform', 'pow08', 'dec', 'inc', 'v6'],
                    help='Per-step weight scheme for anchor (KL/CE). '
                         'none = per-position uniform average (default). '
                         'uniform/pow08/dec/inc = sliding γ-window with fixed weights. '
                         'v6 = dynamic per-window weight from hard_accept: '
                         'before first reject = 1.0, first reject = 1.5, after = 0.')
parser.add_argument('--aux_weight', type=str, default='pow08',
                    choices=['uniform', 'pow08', 'dec', 'inc', 'v6'],
                    help='Per-step weight scheme for aux (V2/V4/V5/TV/EAL). '
                         'Default pow08 matches original 0.8^k. '
                         'v6 = dynamic per-window: before first reject = 1.0, '
                         'first reject = 1.5, after = 0.')
parser.add_argument('--top_k_per_depth', type=str, default='',
                    help='Comma-separated top-K branching per depth for tree losses '
                         '(aux_loss ∈ {tree_v4, tree_altv, tree_ce}). '
                         'Length must equal gamma. Example: "4,3,2,1,1,1,1" '
                         '(matches production eval tree at γ=7).')
parser.add_argument('--coverage_lambda', type=float, default=0.5,
                    help='Coverage penalty weight for aux_loss=tree_ce. '
                         'λ · Σ_d relu(q(K-th) − q(y*)) pushes y* into top-K '
                         'when it falls out. Only used by tree_ce.')
parser.add_argument('--wk_bonus', type=float, default=1.0,
                    help='Top-K bonus weight for aux_loss=tree_altv_wk. '
                         'β_d = Σ_v min(q,p) · (1 + wk_bonus · 1[v ∈ draft_top_K]). '
                         'wk_bonus=0 → identical to full ALTV; '
                         'wk_bonus=∞ (in limit) → identical to tree_altv (top-K restricted). '
                         'Only used by tree_altv_wk.')
parser.add_argument('--altv_topk', type=int, default=20,
                    help='K for al_tv_intersect_topk / al_tv_target_topk. Default 20.')
parser.add_argument('--altv_topp', type=float, default=0.99,
                    help='P (cumulative mass threshold) for al_tv_target_topp. Default 0.99.')
parser.add_argument('--altv_focal_alpha', type=float, default=0.0,
                    help='Focal weighting alpha for al_tv anchors. 0 = uniform (default).')
parser.add_argument('--altv_focal_mode', type=str, default='etau',
                    choices=['etau', 'beta'],
                    help='Focal difficulty basis: etau=(γ-E[τ])/γ; beta=1-mean(β).')
parser.add_argument('--tree_altv_mix', type=float, default=0.1,
                    help='Tree-ALTV mixing coefficient for aux_loss=altv_p_tree_altv. '
                         'aux = L_altv + tree_altv_mix · L_tree_altv. Default 0.1.')
parser.add_argument('--grpo_coef', type=float, default=0.0,
                    help='GRPO loss weight (ω). 0 = disabled (default). '
                         'When > 0, loads a frozen ref draft copy and adds '
                         'group-standardized PPO-clipped policy gradient.')
parser.add_argument('--grpo_k_groups', type=int, default=8,
                    help='Number of non-overlapping groups per sequence for GRPO.')
parser.add_argument('--grpo_m', type=int, default=4,
                    help='Consecutive sliding γ-windows per group (group size).')
parser.add_argument('--grpo_eps', type=float, default=0.2,
                    help='PPO clip ε for GRPO.')
parser.add_argument('--grpo_mode', type=str, default='window',
                    choices=['window', 'sample'],
                    help='GRPO group composition: window=m_win consecutive sliding '
                         'γ-windows (greedy traces); sample=m_win multinomial samples '
                         'sharing one anchor prefix (stochastic traces).')
parser.add_argument('--grpo_sample_temp', type=float, default=1.0,
                    help='Temperature for multinomial sampling (only used when '
                         '--grpo_mode=sample).')
parser.add_argument('--grpo_reward', type=str, default='hard',
                    choices=['hard', 'eal', 'hard_dist'],
                    help='GRPO reward: hard=Σ cumprod(hit) (real τ on greedy/sample); '
                         'eal=Σ cumprod(P_target) (Expected Accept Length, soft); '
                         'hard_dist=hard + R_dist auxiliary when k=0 '
                         '(Distribution-Based Proximity Reward, paper §4.3.2).')
parser.add_argument('--grpo_reward_eta', type=float, default=1.0,
                    help='η: R_dist magnitude when triggered (hard_dist reward only).')
parser.add_argument('--grpo_reward_eps', type=float, default=2.0,
                    help='ε: Δ tolerance threshold; R_dist activates iff Δ < ε. '
                         'Δ = Σ[logp_target(y_t) − logp_target(ŷ_t)]. Lower ε = '
                         'stricter "close to target distribution" requirement.')
parser.add_argument('--lr', type=float, default=None,
                    help='Override learning rate (default: use config value)')
parser.add_argument('--seed', type=int, default=42,
                    help='RNG seed for ds.shuffle + sampler + torch RNG.')
parser.add_argument('--local_rank', type=int, default=-1)
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

with open(args.deepspeed_config) as f:
    ds_config = json.load(f)

# ---------------------------------------------------------------------------
# Data (same as sdpo/main.py, with model-family auto-detection)
# ---------------------------------------------------------------------------
SYSTEM_MSG = (
    "You are a helpful, respectful and honest assistant. Always answer as "
    "helpfully as possible, while being safe.  Your answers should not "
    "include any harmful, unethical, racist, sexist, toxic, dangerous, or "
    "illegal content. Please ensure that your responses are socially "
    "unbiased and positive in nature.\n\nIf a question does not make any "
    "sense, or is not factually coherent, explain why instead of answering "
    "something not correct. If you don't know the answer to a question, "
    "please don't share false information."
)

SEP_TOKENS = {
    "llama": {
        "sep_asst": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "sep_user": "<|eot_id|><|start_header_id|>user<|end_header_id|>",
        "use_system": True,
    },
    "qwen": {
        "sep_asst": "<|im_start|>assistant\n",
        "sep_user": "<|im_start|>user\n",
        "use_system": False,
    },
    "gemma": {
        # Gemma 4 chat template: <|turn>user\n...<turn|>\n<|turn>model\n...<turn|>\n
        "sep_asst": "<|turn>model\n",
        "sep_user": "<|turn>user\n",
        "use_system": False,
    },
}


def detect_model_family(tokenizer):
    name = getattr(tokenizer, 'name_or_path', '').lower()
    if 'qwen' in name:
        return 'qwen'
    if 'gemma' in name:
        return 'gemma'
    return 'llama'


def build_dataset(tokenizer, datapath, max_len, max_samples=None, gamma=7,
                  seed=42):
    ds = load_dataset('json', data_files=datapath)['train']
    ds = ds.shuffle(seed=seed)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    family = detect_model_family(tokenizer)
    sep_cfg = SEP_TOKENS[family]
    SEP_ASST = sep_cfg["sep_asst"]
    SEP_USER = sep_cfg["sep_user"]
    use_system = sep_cfg["use_system"]

    def preprocess(examples):
        out = {"attention_mask": [], "input_ids": [], "loss_mask": []}
        roles = {"human": "user", "gpt": "assistant",
                 "user": "user", "assistant": "assistant"}

        for i in range(len(examples['id'])):
            src = examples['conversations'][i]
            if not src:
                continue
            if roles.get(src[0]["from"]) != "user":
                src = src[1:]

            msgs = []
            if use_system:
                msgs.append({"role": "system", "content": SYSTEM_MSG})
            for j, sent in enumerate(src):
                msgs.append({"role": roles[sent["from"]], "content": sent["value"]})

            # enable_thinking=True: Qwen3 defaults True, Qwen3.5 needs explicit.
            # Non-Qwen models silently ignore this kwarg.
            try:
                conversation = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False,
                    enable_thinking=True)
            except TypeError:
                conversation = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False)
            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id = tokenizer.unk_token_id
            ids = tokenizer(conversation, return_tensors="pt",
                            add_special_tokens=False).input_ids[0]
            if len(ids) > max_len:
                if max_samples is not None:
                    # Truncate: preserve intended sample count when the user
                    # explicitly asked for --max_train_samples N.
                    ids = ids[:max_len]
                else:
                    # Drop: legacy behavior, keeps old runs reproducible.
                    continue

            loss_mask = torch.ones_like(ids)
            turns_raw = conversation.split(SEP_USER)
            turns_raw[1] = turns_raw[0] + SEP_USER + turns_raw[1]
            turns = turns_raw[1:]
            cur_len = 1
            loss_mask[:1] = 0
            for ti, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)
                parts = turn.split(SEP_ASST)
                if len(parts) != 2:
                    break
                parts[0] += SEP_ASST
                instr_len = len(tokenizer(parts[0]).input_ids) - 1
                if ti == 0:
                    loss_mask[cur_len: cur_len + instr_len - 2] = 0
                else:
                    loss_mask[cur_len - 3: cur_len + instr_len + 1] = 0
                cur_len += turn_len
                if ti != 0:
                    cur_len += 3
            loss_mask[cur_len:] = 0

            # Skip samples that won't produce rank-consistent training
            # signal in forward(). Three conditions must hold:
            #   1. loss_mask has at least one 1 (nv > 0 in forward)
            #   2. len(ids) > gamma (usable > 0, at least one γ-window)
            #   3. loss_mask[1:L-gamma+1] has at least one 1 (nv_win > 0)
            # Missing #3 is the subtle case: assistant tokens only in the
            # last γ-1 positions → window mask all zero → rank-divergent
            # skip in training loop → NCCL hang.
            L = len(ids)
            if L <= gamma:
                continue
            if loss_mask[1:L - gamma + 1].sum().item() == 0:
                continue

            out["input_ids"].append(ids.tolist())
            out["loss_mask"].append(loss_mask.tolist())
            out["attention_mask"].append([1] * len(ids))
        return out

    ds = ds.map(preprocess, batched=True, num_proc=1,
                remove_columns=ds.column_names)
    return ds


class DataCollator:
    def __call__(self, features):
        max_len = max(len(f['input_ids']) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "loss_mask": []}
        for f in features:
            for key in batch:
                vals = f[key]
                if not isinstance(vals, list):
                    vals = vals.tolist()
                pad = max_len - len(vals)
                batch[key].append(torch.tensor(vals + [0] * pad, dtype=torch.long))
        return {k: torch.stack(v) for k, v in batch.items()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_latest_checkpoint(directory):
    if not os.path.isdir(directory):
        return None, 0
    max_epoch = -1
    for sub in os.listdir(directory):
        m = re.match(r"state_(\d+)", sub)
        if m and os.path.isdir(os.path.join(directory, sub)):
            max_epoch = max(max_epoch, int(m.group(1)))
    if max_epoch == -1:
        return None, 0
    return f"{directory}/state_{max_epoch}", max_epoch + 1


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

tokenizer = AutoTokenizer.from_pretrained(args.basepath, trust_remote_code=True)
traindataset = build_dataset(tokenizer, args.trainpath, args.max_len,
                             max_samples=args.max_train_samples,
                             gamma=args.gamma, seed=args.seed)
testdataset = build_dataset(tokenizer, args.testpath, args.max_len,
                            gamma=args.gamma)

# LR schedule
n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
micro_bs = ds_config["train_micro_batch_size_per_gpu"]
grad_accum = ds_config["gradient_accumulation_steps"]
effective_bs = micro_bs * grad_accum * n_gpus
steps_per_epoch = math.ceil(len(traindataset) / effective_bs)
total_steps = steps_per_epoch * args.num_epochs
warmup_steps = total_steps // 10
ds_config["scheduler"]["params"]["total_num_steps"] = total_steps
ds_config["scheduler"]["params"]["warmup_num_steps"] = warmup_steps
if args.lr is not None:
    ds_config["scheduler"]["params"]["warmup_max_lr"] = args.lr

samples_per_gpu = math.ceil(len(traindataset) / n_gpus)
iters_per_epoch = math.ceil(samples_per_gpu / micro_bs)
total_iters = iters_per_epoch * args.num_epochs

print(f"[SmallLM] {len(traindataset)} samples, {n_gpus} GPUs, "
      f"effective_bs={effective_bs}, {steps_per_epoch} steps/epoch, "
      f"total={total_steps}, warmup={warmup_steps}")

model_dtype = torch.bfloat16 if ds_config.get("bf16", {}).get("enabled") else torch.float16
zero_stage = ds_config.get("zero_optimization", {}).get("stage", 0)
# HfDeepSpeedConfig sentinel: when alive, transformers.from_pretrained uses
# deepspeed ZeRO-3 sharded construction. Must be created BEFORE from_pretrained
# and kept alive for the lifetime of the model. This is the officially
# supported path (vs. the raw zero.Init context which breaks the strict
# shape-check in newer transformers).
_hf_ds_config = None
# For Gemma 4 ZeRO-3, disable HfDeepSpeedConfig sentinel: partitioning Gemma 4
# 31B frozen target via ZeRO-3 produces garbage output (target argmax becomes
# unused tokens '<unused81>' etc — verified 2026-06-29). Without sentinel,
# from_pretrained loads target FULL on each rank; deepspeed.initialize then
# only partitions trainable params (draft). Memory: target 62G + draft 9G +
# grads 5G + optim 10G + acts 15G = 101G / 143G on H200.
_is_gemma4 = ("gemma-4" in args.basepath.lower() or
              "gemma_4" in args.basepath.lower())
if zero_stage == 3 and not _is_gemma4:
    from transformers.integrations import HfDeepSpeedConfig
    _hf_ds_config = HfDeepSpeedConfig(ds_config)

model = SmallLMDraftModel(
    target_path=args.basepath,
    draft_path=args.draftpath,
    gamma=args.gamma,
    dtype=model_dtype,
    enable_grpo=(args.grpo_coef > 0.0),
)
model.eal_weight = args.eal_weight   # used by aux_loss='v4eal'
model.grpo_reward_eta = args.grpo_reward_eta   # used by grpo_reward='hard_dist'
model.grpo_reward_eps = args.grpo_reward_eps
# Tree loss config (aux_loss ∈ {tree_v4, tree_altv, tree_ce})
if args.top_k_per_depth:
    tk_list = [int(x) for x in args.top_k_per_depth.split(',') if x.strip()]
    assert len(tk_list) == args.gamma, \
        f"--top_k_per_depth length ({len(tk_list)}) must equal --gamma ({args.gamma})"
    model.top_k_per_depth = tk_list
model.coverage_lambda = args.coverage_lambda   # used by aux_loss='tree_ce'
model.wk_bonus = args.wk_bonus                 # used by aux_loss='tree_altv_wk'
model.tree_altv_mix = args.tree_altv_mix       # used by aux_loss='altv_p_tree_altv'
model.altv_topk = args.altv_topk               # used by aux_loss='al_tv_*_topk'
model.altv_topp = args.altv_topp               # used by aux_loss='al_tv_target_topp'
model.altv_focal_alpha = args.altv_focal_alpha # used by aux_loss='al_tv' focal weighting
model.altv_focal_mode = args.altv_focal_mode

# Pass config dict directly to deepspeed.initialize to avoid file-write race
# (both ranks were writing the same JSON simultaneously, occasionally causing
# rank 1 to read a truncated file and fail with
# "Either train_batch_size or train_micro_batch_size_per_gpu needs to be provided").
args.deepspeed_config = None
trainable_params = [p for p in model.parameters() if p.requires_grad]
model_engine, optimizer, _, _ = deepspeed.initialize(
    args=args, model=model, model_parameters=trainable_params,
    config=ds_config)

# Materialize ref_draft_model OUTSIDE ZeRO-3 partition (release sentinel first,
# then load a fresh full copy from disk). Frozen ref must not be tracked by
# the ZeRO-3 partition coordinator, otherwise its params end up NOT_AVAILABLE.
if args.grpo_coef > 0.0:
    if zero_stage == 3 and _hf_ds_config is not None:
        del _hf_ds_config
        _hf_ds_config = None
    model_engine.module._init_ref_draft_model_from_path(
        args.draftpath, model_dtype)

global_rank = deepspeed.comm.get_rank()
rank = deepspeed.comm.get_local_rank()
world_size = deepspeed.comm.get_world_size()

if global_rank == 0:
    run_name = f"smalllm_a{args.sigmoid_coef}_T{args.T_max}"
    if args.baseline:
        run_name = f"smalllm_{args.baseline}"
    wandb.init(project="smalllm_draft", name=run_name, config={
        "sigmoid_coef": args.sigmoid_coef,
        "T_max": args.T_max,
        "T_min": args.T_min,
        "gamma": args.gamma,
        "num_epochs": args.num_epochs,
        "basepath": args.basepath,
        "draftpath": args.draftpath,
        "world_size": world_size,
        "baseline": args.baseline,
    })

os.makedirs(args.savedir, exist_ok=True)
train_sampler = DistributedSampler(traindataset, num_replicas=world_size,
                                   rank=global_rank, shuffle=True,
                                   seed=args.seed)
test_sampler = DistributedSampler(testdataset, num_replicas=world_size,
                                  rank=global_rank, shuffle=False,
                                  seed=args.seed)
train_loader = DataLoader(traindataset, batch_size=micro_bs,
                          sampler=train_sampler, num_workers=0,
                          pin_memory=True, collate_fn=DataCollator())
test_loader = DataLoader(testdataset, batch_size=micro_bs,
                         sampler=test_sampler, num_workers=0,
                         pin_memory=True, collate_fn=DataCollator())

ckpt_path, start_epoch = find_latest_checkpoint(args.savedir)
if ckpt_path:
    print(f"[SmallLM] Resuming from {ckpt_path}")
    model_engine.load_checkpoint(ckpt_path)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
global_step = 0
for epoch in range(start_epoch, args.num_epochs):
    train_sampler.set_epoch(epoch + 1)
    model.train()
    print(f"[SmallLM] Epoch {epoch}")

    epoch_metrics = {"aux_loss": [], "eagle_loss": [], "grpo_loss": [], "mean_tau": []}
    epoch_step_acc = [[] for _ in range(args.gamma)]
    epoch_tau_hist = [0] * (args.gamma + 1)

    for data in tqdm(train_loader, desc=f"Epoch {epoch} train"):
        t_frac = global_step / max(total_iters - 1, 1)
        temperature = args.T_min + (args.T_max - args.T_min) * t_frac
        prob_temperature = args.prob_T_max + (args.prob_T_min - args.prob_T_max) * t_frac
        draft_temperature = args.draft_T_max + (args.draft_T_min - args.draft_T_max) * t_frac

        model.zero_grad()
        total_loss, aux_val, metrics = model_engine(
            input_ids=data["input_ids"].to(rank),
            attention_mask=data["attention_mask"].to(rank),
            loss_mask=data["loss_mask"].to(rank),
            sigmoid_coef=args.sigmoid_coef,
            temperature=temperature,
            prob_temperature=prob_temperature,
            draft_temperature=draft_temperature,
            anchor=args.anchor,
            aux_loss=args.aux_loss,
            anchor_weight=args.anchor_weight,
            aux_weight=args.aux_weight,
            grpo_coef=args.grpo_coef,
            grpo_k_groups=args.grpo_k_groups,
            grpo_m=args.grpo_m,
            grpo_eps=args.grpo_eps,
            grpo_mode=args.grpo_mode,
            grpo_sample_temp=args.grpo_sample_temp,
            grpo_reward=args.grpo_reward,
            baseline=args.baseline,
        )
        if metrics["num_valid"] == 0:
            continue

        model_engine.backward(total_loss)
        grad_norm = model_engine.get_global_grad_norm()
        model_engine.step()
        global_step += 1

        if args.save_steps > 0 and global_step % args.save_steps == 0:
            ckpt_dir = f"{args.savedir}/state_step{global_step}"
            model_engine.save_16bit_model(ckpt_dir, exclude_frozen_parameters=True)
            if global_rank == 0:
                from transformers import AutoConfig
                AutoConfig.from_pretrained(args.draftpath).save_pretrained(ckpt_dir)
                print(f"  [mid-epoch save] {ckpt_dir}  epoch={epoch} step={global_step}")
            deepspeed.comm.barrier()

        epoch_metrics["aux_loss"].append(metrics["aux_loss"])
        epoch_metrics["eagle_loss"].append(metrics["eagle_loss"])
        epoch_metrics["grpo_loss"].append(metrics.get("grpo_loss", 0.0))
        epoch_metrics["mean_tau"].append(metrics["mean_tau"])
        for k in range(args.gamma):
            if k < len(metrics["step_acc"]):
                epoch_step_acc[k].append(metrics["step_acc"][k])
        for v in range(args.gamma + 1):
            if v < len(metrics["tau_hist"]):
                epoch_tau_hist[v] += metrics["tau_hist"][v]

        if global_rank == 0:
            log = {
                "train/total_loss": total_loss.item(),
                "train/aux_loss": metrics["aux_loss"],
                "train/eagle_loss": metrics["eagle_loss"],
                "train/grpo_loss": metrics.get("grpo_loss", 0.0),
                "train/mean_tau": metrics["mean_tau"],
                "train/temperature": temperature,
                "train/grad_norm": grad_norm if grad_norm is not None else float("nan"),
                "train/lr": model_engine.get_lr()[0],
            }
            for k in range(args.gamma):
                if k < len(metrics["step_acc"]):
                    log[f"train/step{k}_acc"] = metrics["step_acc"][k]
            wandb.log(log, step=global_step)

    # Epoch summary
    if global_rank == 0:
        for name in epoch_metrics:
            vals = epoch_metrics[name]
            if vals:
                v = sum(vals) / len(vals)
                print(f"  epoch {epoch} | {name}: {v:.4f}")
        total_tau = sum(epoch_tau_hist)
        if total_tau > 0:
            print(f"  τ distribution: {[f'{h/total_tau:.3f}' for h in epoch_tau_hist]}")

    # Eval
    model.eval()
    eval_metrics = {"aux_loss": [], "mean_tau": []}
    eval_step_acc = [[] for _ in range(args.gamma)]

    for data in tqdm(test_loader, desc=f"Epoch {epoch} eval"):
        with torch.no_grad():
            _, _, metrics = model_engine(
                input_ids=data["input_ids"].to(rank),
                attention_mask=data["attention_mask"].to(rank),
                loss_mask=data["loss_mask"].to(rank),
                sigmoid_coef=args.sigmoid_coef,
                temperature=temperature,
                prob_temperature=prob_temperature,
                draft_temperature=draft_temperature,
                baseline=args.baseline,
                aux_loss=args.aux_loss,
            )
        if metrics["num_valid"] > 0:
            eval_metrics["aux_loss"].append(metrics["aux_loss"])
            eval_metrics["mean_tau"].append(metrics["mean_tau"])
            for k in range(args.gamma):
                if k < len(metrics["step_acc"]):
                    eval_step_acc[k].append(metrics["step_acc"][k])

    if global_rank == 0:
        eval_log = {"epoch": epoch}
        for name in eval_metrics:
            vals = eval_metrics[name]
            if vals:
                v = sum(vals) / len(vals)
                eval_log[f"eval/{name}"] = v
                print(f"  epoch {epoch} | eval {name}: {v:.4f}")
        for k in range(args.gamma):
            if eval_step_acc[k]:
                eval_log[f"eval/step{k}_acc"] = sum(eval_step_acc[k]) / len(eval_step_acc[k])
        wandb.log(eval_log, step=global_step)

    deepspeed.comm.barrier()
    torch.cuda.empty_cache()

    # Save checkpoint (only draft model weights)
    save_epochs = (set(int(x) for x in args.save_epochs.split(',') if x.strip())
                   if args.save_epochs else None)
    if save_epochs is None or epoch in save_epochs:
        ckpt_dir = f"{args.savedir}/state_{epoch}"
        model_engine.save_16bit_model(ckpt_dir, exclude_frozen_parameters=True)
        if global_rank == 0:
            from transformers import AutoConfig
            draft_config = AutoConfig.from_pretrained(args.draftpath)
            draft_config.save_pretrained(ckpt_dir)
            print(f"  Saved checkpoint to {ckpt_dir}")
