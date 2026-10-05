"""
Layer-Skip Draft Training Script with ON-POLICY ROLLOUT.

Uses sdpo/layer_skip_rollout_model.py instead of sdpo/layer_skip_model.py.
Each training step runs a γ-step autoregressive rollout from every assistant
anchor position; the V2 loss is applied as a cumulative product over the
rollout sequence per exit. KL anchor loss is still teacher-forced at all
positions.

Loss:
    L = KL(target || draft_step0) + sigmoid_coef · V2_rollout

Usage:
  deepspeed sdpo/main_layer_skip_rollout.py \\
    --basepath Qwen/Qwen3-32B \\
    --draftpath Qwen/Qwen3-0.6B \\
    --exit-layers 20,24,26,28 \\
    --trainpath sdpo/data/mixed_train_10K.jsonl \\
    --testpath  sdpo/data/mixed_val_80.jsonl \\
    --deepspeed_config sdpo/sdpo_config_qwen3_bf16_zero3_2gpu.json \\
    --savedir /scratch/tx856/spec_reason/scratch/loss_train_layerskip_rollout/test \\
    --sigmoid_coef 0.1 --T_max 0.1 --T_min 0.1 \\
    --gamma 7 --num_epochs 3 --lr 5e-6
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
from layer_skip_rollout_model import LayerSkipRolloutModel

torch.backends.cuda.matmul.allow_tf32 = True

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--basepath', required=True)
parser.add_argument('--draftpath', required=True)
parser.add_argument('--exit-layers', required=True, dest='exit_layers',
                    help='comma-separated 1-indexed exit layers')
parser.add_argument('--trainpath', required=True)
parser.add_argument('--testpath', required=True)
parser.add_argument('--savedir',
                    default='/scratch/tx856/spec_reason/scratch/loss_train_layerskip_rollout')
parser.add_argument('--sigmoid_coef', type=float, default=0.1)
parser.add_argument('--T_max', type=float, default=0.1)
parser.add_argument('--T_min', type=float, default=0.1)
parser.add_argument('--gamma', type=int, default=7)
parser.add_argument('--max_len', type=int, default=1024)
parser.add_argument('--max_train_samples', type=int, default=None)
parser.add_argument('--num_epochs', type=int, default=3)
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--local_rank', type=int, default=-1)
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

exit_layers = [int(e) for e in args.exit_layers.split(',') if e.strip()]

with open(args.deepspeed_config) as f:
    ds_config = json.load(f)

# ---------------------------------------------------------------------------
# Sanity: rollout requires micro_bs = 1
# ---------------------------------------------------------------------------
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
                             max_samples=args.max_train_samples)
testdataset = build_dataset(tokenizer, args.testpath, args.max_len)

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

print(f"[LayerSkipRollout] {len(traindataset)} samples, {n_gpus} GPUs, "
      f"micro_bs={micro_bs}, effective_bs={effective_bs}, "
      f"{steps_per_epoch} steps/epoch, total={total_steps}")
print(f"[LayerSkipRollout] exit_layers={exit_layers}  gamma={args.gamma}  "
      f"coef={args.sigmoid_coef}  T={args.T_min}-{args.T_max}")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model_dtype = torch.bfloat16 if ds_config.get("bf16", {}).get("enabled") else torch.float16
zero_stage = ds_config.get("zero_optimization", {}).get("stage", 0)
_hf_ds_config = None
if zero_stage == 3:
    from transformers.integrations import HfDeepSpeedConfig
    _hf_ds_config = HfDeepSpeedConfig(ds_config)

model = LayerSkipRolloutModel(
    target_path=args.basepath,
    draft_path=args.draftpath,
    exit_layers=exit_layers,
    gamma=args.gamma,
    dtype=model_dtype,
)
model.set_debug_tokenizer(tokenizer)

# Pass the modified ds_config dict directly to deepspeed.initialize via the
# `config` kwarg instead of writing it back to disk. This avoids the rank-0 /
# rank-1 race condition where both processes `json.dump` to the same file and
# one reads a truncated intermediate state, producing the misleading
# "Either train_batch_size or train_micro_batch_size_per_gpu needs to be
# provided" assertion.
# Clear the path on args so deepspeed prefers our in-memory config dict.
args.deepspeed_config = None
trainable_params = [p for p in model.parameters() if p.requires_grad]
model_engine, optimizer, _, _ = deepspeed.initialize(
    args=args, model=model, model_parameters=trainable_params,
    config=ds_config)

global_rank = deepspeed.comm.get_rank()
rank = deepspeed.comm.get_local_rank()
world_size = deepspeed.comm.get_world_size()

if global_rank == 0:
    run_name = f"lsr_{'-'.join(map(str, exit_layers))}_c{args.sigmoid_coef}_T{args.T_max}"
    wandb.init(project="layerskip_rollout", name=run_name, config={
        "exit_layers": exit_layers,
        "sigmoid_coef": args.sigmoid_coef,
        "T_max": args.T_max, "T_min": args.T_min,
        "gamma": args.gamma,
        "num_epochs": args.num_epochs,
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
    print(f"[LayerSkipRollout] Resuming from {ckpt_path}")
    model_engine.load_checkpoint(ckpt_path)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
total_iters = steps_per_epoch * args.num_epochs
global_step = 0
for epoch in range(start_epoch, args.num_epochs):
    train_sampler.set_epoch(epoch + 1)
    model.train()
    print(f"[LayerSkipRollout] Epoch {epoch}")

    sum_total = 0.0
    sum_kl = 0.0
    sum_v2 = 0.0
    sum_anchors = 0
    n_batches = 0
    sum_per_exit = {e: {"kl": 0.0, "v2": 0.0} for e in exit_layers}

    for data in tqdm(train_loader, desc=f"Epoch {epoch} train"):
        t_frac = global_step / max(total_iters - 1, 1)
        temperature = args.T_min + (args.T_max - args.T_min) * t_frac

        model.zero_grad()
        total_loss, v2_val, metrics = model_engine(
            input_ids=data["input_ids"].to(rank),
            attention_mask=data["attention_mask"].to(rank),
            loss_mask=data["loss_mask"].to(rank),
            sigmoid_coef=args.sigmoid_coef,
            temperature=temperature,
        )
        if metrics["num_valid"] == 0 or metrics["num_anchors"] == 0:
            continue

        model_engine.backward(total_loss)
        grad_norm = model_engine.get_global_grad_norm()
        model_engine.step()
        global_step += 1
        n_batches += 1

        sum_total += metrics["total_loss"]
        sum_kl += metrics["kl_loss"]
        sum_v2 += metrics["v2_loss"]
        sum_anchors += metrics["num_anchors"]
        for e in exit_layers:
            sum_per_exit[e]["kl"] += metrics["per_exit"][e]["kl"]
            sum_per_exit[e]["v2"] += metrics["per_exit"][e]["v2"]

        if global_rank == 0:
            log = {
                "train/total_loss": metrics["total_loss"],
                "train/kl_loss": metrics["kl_loss"],
                "train/v2_loss": metrics["v2_loss"],
                "train/num_anchors": metrics["num_anchors"],
                "train/temperature": temperature,
                "train/grad_norm": grad_norm if grad_norm is not None else float("nan"),
                "train/lr": model_engine.get_lr()[0],
            }
            for e in exit_layers:
                log[f"train/exit{e}/kl"] = metrics["per_exit"][e]["kl"]
                log[f"train/exit{e}/v2"] = metrics["per_exit"][e]["v2"]
            wandb.log(log, step=global_step)

    if global_rank == 0 and n_batches > 0:
        print(f"  epoch {epoch} | total={sum_total/n_batches:.4f}  "
              f"kl={sum_kl/n_batches:.4f}  v2={sum_v2/n_batches:.4f}  "
              f"avg_anchors={sum_anchors/n_batches:.1f}")
        for e in exit_layers:
            print(f"    exit={e:>3}  kl={sum_per_exit[e]['kl']/n_batches:.4f}  "
                  f"v2={sum_per_exit[e]['v2']/n_batches:.4f}")

    # Eval
    model.eval()
    e_sum_total = 0.0
    e_sum_kl = 0.0
    e_sum_v2 = 0.0
    e_n = 0
    for data in tqdm(test_loader, desc=f"Epoch {epoch} eval"):
        with torch.no_grad():
            _, _, m = model_engine(
                input_ids=data["input_ids"].to(rank),
                attention_mask=data["attention_mask"].to(rank),
                loss_mask=data["loss_mask"].to(rank),
                sigmoid_coef=args.sigmoid_coef,
                temperature=temperature,
            )
        if m["num_valid"] == 0 or m["num_anchors"] == 0:
            continue
        e_sum_total += m["total_loss"]
        e_sum_kl += m["kl_loss"]
        e_sum_v2 += m["v2_loss"]
        e_n += 1

    if global_rank == 0 and e_n > 0:
        eval_log = {
            "epoch": epoch,
            "eval/total_loss": e_sum_total / e_n,
            "eval/kl_loss": e_sum_kl / e_n,
            "eval/v2_loss": e_sum_v2 / e_n,
        }
        print(f"  epoch {epoch} | eval total={e_sum_total/e_n:.4f}  "
              f"kl={e_sum_kl/e_n:.4f}  v2={e_sum_v2/e_n:.4f}")
        wandb.log(eval_log, step=global_step)

    deepspeed.comm.barrier()
    torch.cuda.empty_cache()

    ckpt_dir = f"{args.savedir}/state_{epoch}"
    model_engine.save_16bit_model(ckpt_dir, exclude_frozen_parameters=True)
    if global_rank == 0:
        from transformers import AutoConfig
        draft_config = AutoConfig.from_pretrained(args.draftpath)
        draft_config.save_pretrained(ckpt_dir)
        with open(os.path.join(ckpt_dir, "exit_layers.json"), 'w') as f:
            json.dump({"exit_layers": exit_layers}, f)
        print(f"  Saved checkpoint to {ckpt_dir}")
