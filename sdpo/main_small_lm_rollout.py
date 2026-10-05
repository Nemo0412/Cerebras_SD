"""
Small LM Draft Training Script with ON-POLICY ROLLOUT.

Uses sdpo/small_lm_rollout_model.py. Each training step runs γ-step
autoregressive rollout from every assistant anchor position; V2 loss applied
as cumulative product over the rollout chain. KL anchor loss is teacher-forced
at all positions (same as main_small_lm.py).

Loss:
    L = KL(target || draft_step0) + sigmoid_coef · V2_rollout

Usage:
  deepspeed sdpo/main_small_lm_rollout.py \\
    --basepath Qwen/Qwen3-32B \\
    --draftpath Qwen/Qwen3-0.6B \\
    --trainpath sdpo/data/mixed_train_10K.jsonl \\
    --testpath  sdpo/data/mixed_val_80.jsonl \\
    --deepspeed_config sdpo/sdpo_config_qwen3_bf16_zero3_2gpu.json \\
    --savedir /scratch/tx856/spec_reason/scratch/loss_train_smalllm_rollout/test \\
    --sigmoid_coef 0.1 --T_max 0.1 --T_min 0.1 \\
    --gamma 7 --num_epochs 3 --lr 2e-6
"""

import argparse
import json
import math
import os
import re
import sys

import deepspeed
import torch
import wandb
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline import build_dataset, DataCollator
from small_lm_rollout_model import SmallLMRolloutModel

torch.backends.cuda.matmul.allow_tf32 = True

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--basepath', required=True, help='Target LLM (frozen)')
parser.add_argument('--draftpath', required=True, help='Draft LLM (small, trainable)')
parser.add_argument('--trainpath', required=True)
parser.add_argument('--testpath', required=True)
parser.add_argument('--savedir',
                    default='/scratch/tx856/spec_reason/scratch/loss_train_smalllm_rollout')
parser.add_argument('--sigmoid_coef', type=float, default=0.1)
parser.add_argument('--T_max', type=float, default=0.1)
parser.add_argument('--T_min', type=float, default=0.1)
parser.add_argument('--gamma', type=int, default=7)
parser.add_argument('--max_len', type=int, default=1024)
parser.add_argument('--max_train_samples', type=int, default=None)
parser.add_argument('--num_epochs', type=int, default=3)
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--baseline', type=str, default=None, choices=['eagle_only', 'ce_only'],
                    help='Legacy: eagle_only → --anchor kl --aux_loss none; ce_only → --anchor ce --aux_loss none')
parser.add_argument('--anchor', type=str, default='kl', choices=['kl', 'ce'],
                    help='Main loss: kl (soft-target KL) or ce (hard-label CE)')
parser.add_argument('--aux_loss', type=str, default='v2',
                    choices=['none', 'v2', 'v4', 'v5', 'tv', 'al_tv',
                             'acceptance_length_v2', 'acceptance_length_v4'],
                    help='Auxiliary loss: none / v2 / v4 / v5 / tv / al_tv')
parser.add_argument('--anchor_weight', type=str, default='none',
                    choices=['none', 'uniform', 'pow08', 'dec', 'inc'])
parser.add_argument('--aux_weight', type=str, default='pow08',
                    choices=['uniform', 'pow08', 'dec', 'inc'])
parser.add_argument('--on_policy_target', action='store_true',
                    help='After draft rollout, re-forward target on draft dirty '
                         'chain to get dirty-context target_argmax for β/α '
                         '(fixes clean/dirty context mismatch in aux loss).')
parser.add_argument('--soft_rollout', action='store_true',
                    help='Feed expected embedding (softmax(logits) @ embed.weight) '
                         'between rollout steps instead of argmax token. '
                         'Gradient flows through the rollout chain. '
                         'Incompatible with --on_policy_target.')
parser.add_argument('--chain_anchor_coef', type=float, default=1.0,
                    help='Weight on the chain-anchor loss (per-step KL/CE on '
                         'rollout chain steps 1..γ-1). Additive to prefix '
                         'anchor. Only effective with rollout (soro/ropo).')
parser.add_argument('--rollout_truncate_at_reject', action='store_true',
                    help='Truncate chain_anchor and V4/V2 aux at first reject '
                         'step in dirty rollout (first reject still counted; '
                         'later steps masked). Mitigates clean-target / dirty-'
                         'input context mismatch beyond first divergence.')
parser.add_argument('--rollout_loss_mode', default='chain',
                    choices=['chain', 'hybrid'],
                    help="'chain' (new default): anchor loss only on γ-step "
                         "rollout chain at K anchor positions (step 0=clean "
                         "prefix at anchor, step 1..γ-1=dirty rollout). γ "
                         "controls chain length only; no sliding window. "
                         "γ=1 ⇒ single per-anchor loss. "
                         "'hybrid' (legacy): sliding γ-window on prefix per_pos "
                         "+ chain_anchor_coef * chain on dirty steps 1..γ-1.")
parser.add_argument('--local_rank', type=int, default=-1)
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

with open(args.deepspeed_config) as f:
    ds_config = json.load(f)

if ds_config["train_micro_batch_size_per_gpu"] != 1:
    print(f"[WARN] rollout training needs micro_bs=1; forcing to 1 "
          f"(was {ds_config['train_micro_batch_size_per_gpu']})")
    ds_config["train_micro_batch_size_per_gpu"] = 1


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
# Dataset
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(args.basepath, trust_remote_code=True)
traindataset = build_dataset(tokenizer, args.trainpath, args.max_len,
                             max_samples=args.max_train_samples,
                             gamma=args.gamma)
testdataset = build_dataset(tokenizer, args.testpath, args.max_len,
                            gamma=args.gamma)

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

print(f"[SmallLMRollout] {len(traindataset)} samples, {n_gpus} GPUs, "
      f"micro_bs={micro_bs}, effective_bs={effective_bs}, "
      f"{steps_per_epoch} steps/epoch, total={total_steps}")
print(f"[SmallLMRollout] gamma={args.gamma} coef={args.sigmoid_coef} "
      f"T={args.T_min}-{args.T_max}")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model_dtype = torch.bfloat16 if ds_config.get("bf16", {}).get("enabled") else torch.float16
zero_stage = ds_config.get("zero_optimization", {}).get("stage", 0)
_hf_ds_config = None
if zero_stage == 3:
    from transformers.integrations import HfDeepSpeedConfig
    _hf_ds_config = HfDeepSpeedConfig(ds_config)

model = SmallLMRolloutModel(
    target_path=args.basepath,
    draft_path=args.draftpath,
    gamma=args.gamma,
    dtype=model_dtype,
)

args.deepspeed_config = None  # pass config dict directly to avoid rank race
trainable_params = [p for p in model.parameters() if p.requires_grad]
model_engine, optimizer, _, _ = deepspeed.initialize(
    args=args, model=model, model_parameters=trainable_params,
    config=ds_config)

global_rank = deepspeed.comm.get_rank()
rank = deepspeed.comm.get_local_rank()
world_size = deepspeed.comm.get_world_size()

if global_rank == 0:
    run_name = f"smalllm_rollout_c{args.sigmoid_coef}_T{args.T_max}"
    wandb.init(project="smalllm_rollout", name=run_name, config={
        "sigmoid_coef": args.sigmoid_coef,
        "T_max": args.T_max, "T_min": args.T_min,
        "gamma": args.gamma, "num_epochs": args.num_epochs,
        "basepath": args.basepath, "draftpath": args.draftpath,
        "world_size": world_size, "lr": args.lr,
    })

os.makedirs(args.savedir, exist_ok=True)
train_sampler = DistributedSampler(traindataset, num_replicas=world_size,
                                   rank=global_rank, shuffle=True)
test_sampler = DistributedSampler(testdataset, num_replicas=world_size,
                                  rank=global_rank, shuffle=False)
train_loader = DataLoader(traindataset, batch_size=micro_bs,
                          sampler=train_sampler, num_workers=0,
                          pin_memory=True, collate_fn=DataCollator())
test_loader = DataLoader(testdataset, batch_size=micro_bs,
                         sampler=test_sampler, num_workers=0,
                         pin_memory=True, collate_fn=DataCollator())

ckpt_path, start_epoch = find_latest_checkpoint(args.savedir)
if ckpt_path:
    print(f"[SmallLMRollout] Resuming from {ckpt_path}")
    model_engine.load_checkpoint(ckpt_path)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
global_step = 0
for epoch in range(start_epoch, args.num_epochs):
    train_sampler.set_epoch(epoch + 1)
    model.train()
    print(f"[SmallLMRollout] Epoch {epoch}")

    sum_total = 0.0
    sum_kl = 0.0
    sum_v2 = 0.0
    sum_tau = 0.0
    sum_anchors = 0
    n_batches = 0

    for data in tqdm(train_loader, desc=f"Epoch {epoch} train"):
        t_frac = global_step / max(total_iters - 1, 1)
        temperature = args.T_min + (args.T_max - args.T_min) * t_frac

        model.zero_grad()
        total_loss, _, metrics = model_engine(
            input_ids=data["input_ids"].to(rank),
            attention_mask=data["attention_mask"].to(rank),
            loss_mask=data["loss_mask"].to(rank),
            sigmoid_coef=args.sigmoid_coef,
            temperature=temperature,
            anchor=args.anchor,
            aux_loss=args.aux_loss,
            anchor_weight=args.anchor_weight,
            aux_weight=args.aux_weight,
            on_policy_target=args.on_policy_target,
            soft_rollout=args.soft_rollout,
            chain_anchor_coef=args.chain_anchor_coef,
            rollout_truncate_at_reject=args.rollout_truncate_at_reject,
            rollout_loss_mode=args.rollout_loss_mode,
            baseline=args.baseline,
        )
        if metrics["num_valid"] == 0 or metrics.get("num_anchors", 0) == 0:
            continue

        model_engine.backward(total_loss)
        grad_norm = model_engine.get_global_grad_norm()
        model_engine.step()
        global_step += 1
        n_batches += 1

        sum_total += metrics["total_loss"]
        sum_kl += metrics["kl_loss"]
        sum_v2 += metrics["v2_loss"]
        sum_tau += metrics["mean_tau"]
        sum_anchors += metrics["num_anchors"]

        if global_rank == 0:
            log = {
                "train/total_loss": metrics["total_loss"],
                "train/kl_loss": metrics["kl_loss"],
                "train/v2_loss": metrics["v2_loss"],
                "train/mean_tau": metrics["mean_tau"],
                "train/num_anchors": metrics["num_anchors"],
                "train/temperature": temperature,
                "train/grad_norm": grad_norm if grad_norm is not None else float("nan"),
                "train/lr": model_engine.get_lr()[0],
            }
            for k, v in enumerate(metrics.get("step_acc", [])):
                log[f"train/step{k}_acc"] = v
            wandb.log(log, step=global_step)

    if global_rank == 0 and n_batches > 0:
        print(f"  epoch {epoch} | total={sum_total/n_batches:.4f}  "
              f"kl={sum_kl/n_batches:.4f}  v2={sum_v2/n_batches:.4f}  "
              f"tau={sum_tau/n_batches:.4f}  "
              f"avg_anchors={sum_anchors/n_batches:.1f}")

    # Eval
    model.eval()
    e_total, e_kl, e_v2, e_tau, e_n = 0.0, 0.0, 0.0, 0.0, 0
    for data in tqdm(test_loader, desc=f"Epoch {epoch} eval"):
        with torch.no_grad():
            _, _, m = model_engine(
                input_ids=data["input_ids"].to(rank),
                attention_mask=data["attention_mask"].to(rank),
                loss_mask=data["loss_mask"].to(rank),
                sigmoid_coef=args.sigmoid_coef,
                temperature=temperature,
                anchor=args.anchor,
                aux_loss=args.aux_loss,
                anchor_weight=args.anchor_weight,
                aux_weight=args.aux_weight,
                on_policy_target=args.on_policy_target,
                soft_rollout=args.soft_rollout,
                chain_anchor_coef=args.chain_anchor_coef,
                rollout_truncate_at_reject=args.rollout_truncate_at_reject,
                rollout_loss_mode=args.rollout_loss_mode,
                baseline=args.baseline,
            )
        if m["num_valid"] == 0 or m.get("num_anchors", 0) == 0:
            continue
        e_total += m["total_loss"]
        e_kl += m["kl_loss"]
        e_v2 += m["v2_loss"]
        e_tau += m["mean_tau"]
        e_n += 1

    if global_rank == 0 and e_n > 0:
        eval_log = {
            "epoch": epoch,
            "eval/total_loss": e_total / e_n,
            "eval/kl_loss": e_kl / e_n,
            "eval/v2_loss": e_v2 / e_n,
            "eval/mean_tau": e_tau / e_n,
        }
        print(f"  epoch {epoch} | eval total={e_total/e_n:.4f}  "
              f"kl={e_kl/e_n:.4f}  v2={e_v2/e_n:.4f}  tau={e_tau/e_n:.4f}")
        wandb.log(eval_log, step=global_step)

    deepspeed.comm.barrier()
    torch.cuda.empty_cache()

    ckpt_dir = f"{args.savedir}/state_{epoch}"
    model_engine.save_16bit_model(ckpt_dir, exclude_frozen_parameters=True)
    if global_rank == 0:
        from transformers import AutoConfig
        draft_config = AutoConfig.from_pretrained(args.draftpath)
        draft_config.save_pretrained(ckpt_dir)
        print(f"  Saved checkpoint to {ckpt_dir}")
