"""
Layer-Skip Draft Training Script.

Fine-tunes a small LM (e.g., Qwen3-0.6B) whose intermediate layer exits act
as independent draft predictors for a large target (e.g., Qwen3-32B).

Loss configurations (mirror sdpo/main_small_lm.py, applied per-exit and
averaged):

  --baseline eagle_only     : mean over exits of KL(target || draft_e)
  --baseline ce_only        : mean over exits of CE(target_argmax, draft_e)
  --aux_loss acceptance_length_v2 (default, used unless --baseline set):
                              KL + sigmoid_coef · L_acc_v2, per exit
  --aux_loss tv              : KL + sigmoid_coef · L_tv, per exit

Usage:
  deepspeed sdpo/main_layer_skip.py \\
    --basepath Qwen/Qwen3-32B \\
    --draftpath Qwen/Qwen3-0.6B \\
    --exit-layers 4,8,12,16,20,24,28 \\
    --trainpath sdpo/data/mixed_train_10K.jsonl \\
    --testpath  sdpo/data/mixed_val_80.jsonl \\
    --deepspeed_config sdpo/sdpo_config_qwen3_bf16_zero3_2gpu.json \\
    --savedir /scratch/tx856/spec_reason/scratch/loss_train_layerskip/test \\
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
from layer_skip_model import LayerSkipDraftModel

torch.backends.cuda.matmul.allow_tf32 = True

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--basepath', required=True, help='Target LLM (frozen)')
parser.add_argument('--draftpath', required=True, help='Draft small LM to fine-tune')
parser.add_argument('--exit-layers', required=True, dest='exit_layers',
                    help='comma-separated 1-indexed exit layers, e.g. 4,8,12,16')
parser.add_argument('--trainpath', required=True)
parser.add_argument('--testpath', required=True)
parser.add_argument('--savedir', default='/scratch/tx856/spec_reason/scratch/loss_train_layerskip')
parser.add_argument('--sigmoid_coef', type=float, default=0.1)
parser.add_argument('--T_max', type=float, default=0.1)
parser.add_argument('--T_min', type=float, default=0.1)
parser.add_argument('--gamma', type=int, default=7)
parser.add_argument('--max_len', type=int, default=2048)
parser.add_argument('--max_train_samples', type=int, default=None)
parser.add_argument('--num_epochs', type=int, default=3)
parser.add_argument('--baseline', type=str, default=None,
                    choices=['eagle_only', 'ce_only'])
parser.add_argument('--aux_loss', type=str, default='acceptance_length_v2',
                    choices=['acceptance_length_v2', 'tv'],
                    help='Auxiliary loss added to KL when --baseline not set')
parser.add_argument('--lr', type=float, default=None,
                    help='Override learning rate (default: use config value)')
parser.add_argument('--local_rank', type=int, default=-1)
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

exit_layers = [int(e) for e in args.exit_layers.split(',') if e.strip()]

with open(args.deepspeed_config) as f:
    ds_config = json.load(f)


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

samples_per_gpu = math.ceil(len(traindataset) / n_gpus)
iters_per_epoch = math.ceil(samples_per_gpu / micro_bs)
total_iters = iters_per_epoch * args.num_epochs

print(f"[LayerSkip] {len(traindataset)} samples, {n_gpus} GPUs, "
      f"effective_bs={effective_bs}, {steps_per_epoch} steps/epoch, "
      f"total={total_steps}, warmup={warmup_steps}")
print(f"[LayerSkip] exit_layers={exit_layers}  "
      f"loss={'baseline='+args.baseline if args.baseline else 'KL+'+args.aux_loss}")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model_dtype = torch.bfloat16 if ds_config.get("bf16", {}).get("enabled") else torch.float16
zero_stage = ds_config.get("zero_optimization", {}).get("stage", 0)
_hf_ds_config = None
if zero_stage == 3:
    from transformers.integrations import HfDeepSpeedConfig
    _hf_ds_config = HfDeepSpeedConfig(ds_config)

model = LayerSkipDraftModel(
    target_path=args.basepath,
    draft_path=args.draftpath,
    exit_layers=exit_layers,
    gamma=args.gamma,
    dtype=model_dtype,
)

with open(args.deepspeed_config, 'w') as f:
    json.dump(ds_config, f, indent=2)

trainable_params = [p for p in model.parameters() if p.requires_grad]
model_engine, optimizer, _, _ = deepspeed.initialize(
    args=args, model=model, model_parameters=trainable_params)

global_rank = deepspeed.comm.get_rank()
rank = deepspeed.comm.get_local_rank()
world_size = deepspeed.comm.get_world_size()

if global_rank == 0:
    run_name = f"ls_{'-'.join(map(str, exit_layers))}_a{args.sigmoid_coef}_T{args.T_max}"
    if args.baseline:
        run_name = f"ls_{'-'.join(map(str, exit_layers))}_{args.baseline}"
    wandb.init(project="layerskip_draft", name=run_name, config={
        "exit_layers": exit_layers,
        "sigmoid_coef": args.sigmoid_coef,
        "T_max": args.T_max,
        "T_min": args.T_min,
        "gamma": args.gamma,
        "num_epochs": args.num_epochs,
        "basepath": args.basepath,
        "draftpath": args.draftpath,
        "world_size": world_size,
        "baseline": args.baseline,
        "aux_loss": args.aux_loss,
        "lr": args.lr,
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
    print(f"[LayerSkip] Resuming from {ckpt_path}")
    model_engine.load_checkpoint(ckpt_path)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
global_step = 0
for epoch in range(start_epoch, args.num_epochs):
    train_sampler.set_epoch(epoch + 1)
    model.train()
    print(f"[LayerSkip] Epoch {epoch}")

    epoch_metrics = {"aux_loss": [], "eagle_loss": [], "mean_tau": []}
    epoch_step_acc = [[] for _ in range(args.gamma)]
    epoch_tau_hist = [0] * (args.gamma + 1)
    epoch_per_exit_tau = {e: [] for e in exit_layers}
    epoch_per_exit_loss = {e: [] for e in exit_layers}

    for data in tqdm(train_loader, desc=f"Epoch {epoch} train"):
        t_frac = global_step / max(total_iters - 1, 1)
        temperature = args.T_min + (args.T_max - args.T_min) * t_frac

        model.zero_grad()
        total_loss, aux_val, metrics = model_engine(
            input_ids=data["input_ids"].to(rank),
            attention_mask=data["attention_mask"].to(rank),
            loss_mask=data["loss_mask"].to(rank),
            sigmoid_coef=args.sigmoid_coef,
            temperature=temperature,
            baseline=args.baseline,
            aux_loss=args.aux_loss,
        )
        if metrics["num_valid"] == 0:
            continue

        model_engine.backward(total_loss)
        grad_norm = model_engine.get_global_grad_norm()
        model_engine.step()
        global_step += 1

        epoch_metrics["aux_loss"].append(metrics["aux_loss"])
        epoch_metrics["eagle_loss"].append(metrics["eagle_loss"])
        epoch_metrics["mean_tau"].append(metrics["mean_tau"])
        for k in range(args.gamma):
            if k < len(metrics["step_acc"]):
                epoch_step_acc[k].append(metrics["step_acc"][k])
        for v in range(args.gamma + 1):
            if v < len(metrics["tau_hist"]):
                epoch_tau_hist[v] += metrics["tau_hist"][v]
        for e in exit_layers:
            epoch_per_exit_tau[e].append(metrics["per_exit"][e]["mean_tau"])
            epoch_per_exit_loss[e].append(metrics["per_exit"][e]["eagle_loss"])

        if global_rank == 0:
            log = {
                "train/total_loss": total_loss.item(),
                "train/aux_loss": metrics["aux_loss"],
                "train/eagle_loss": metrics["eagle_loss"],
                "train/mean_tau": metrics["mean_tau"],
                "train/temperature": temperature,
                "train/grad_norm": grad_norm if grad_norm is not None else float("nan"),
                "train/lr": model_engine.get_lr()[0],
            }
            for k in range(args.gamma):
                if k < len(metrics["step_acc"]):
                    log[f"train/step{k}_acc"] = metrics["step_acc"][k]
            for e in exit_layers:
                log[f"train/exit{e}/mean_tau"] = metrics["per_exit"][e]["mean_tau"]
                log[f"train/exit{e}/eagle_loss"] = metrics["per_exit"][e]["eagle_loss"]
                log[f"train/exit{e}/aux_loss"] = metrics["per_exit"][e]["aux_loss"]
            wandb.log(log, step=global_step)

    if global_rank == 0:
        for name in epoch_metrics:
            vals = epoch_metrics[name]
            if vals:
                print(f"  epoch {epoch} | {name}: {sum(vals)/len(vals):.4f}")
        total_tau = sum(epoch_tau_hist)
        if total_tau > 0:
            print(f"  τ distribution: {[f'{h/total_tau:.3f}' for h in epoch_tau_hist]}")
        for e in exit_layers:
            if epoch_per_exit_tau[e]:
                tau_e = sum(epoch_per_exit_tau[e]) / len(epoch_per_exit_tau[e])
                loss_e = sum(epoch_per_exit_loss[e]) / len(epoch_per_exit_loss[e])
                print(f"  epoch {epoch} | exit={e:>3}  "
                      f"mean_tau={tau_e:.4f}  eagle_loss={loss_e:.4f}")

    # Eval
    model.eval()
    eval_metrics = {"aux_loss": [], "mean_tau": []}
    eval_step_acc = [[] for _ in range(args.gamma)]
    eval_per_exit_tau = {e: [] for e in exit_layers}

    for data in tqdm(test_loader, desc=f"Epoch {epoch} eval"):
        with torch.no_grad():
            _, _, metrics = model_engine(
                input_ids=data["input_ids"].to(rank),
                attention_mask=data["attention_mask"].to(rank),
                loss_mask=data["loss_mask"].to(rank),
                sigmoid_coef=args.sigmoid_coef,
                temperature=temperature,
                baseline=args.baseline,
                aux_loss=args.aux_loss,
            )
        if metrics["num_valid"] > 0:
            eval_metrics["aux_loss"].append(metrics["aux_loss"])
            eval_metrics["mean_tau"].append(metrics["mean_tau"])
            for k in range(args.gamma):
                if k < len(metrics["step_acc"]):
                    eval_step_acc[k].append(metrics["step_acc"][k])
            for e in exit_layers:
                eval_per_exit_tau[e].append(metrics["per_exit"][e]["mean_tau"])

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
        for e in exit_layers:
            if eval_per_exit_tau[e]:
                v = sum(eval_per_exit_tau[e]) / len(eval_per_exit_tau[e])
                eval_log[f"eval/exit{e}/mean_tau"] = v
                print(f"  epoch {epoch} | eval exit={e:>3}  mean_tau={v:.4f}")
        wandb.log(eval_log, step=global_step)

    deepspeed.comm.barrier()
    torch.cuda.empty_cache()

    # Save (draft only)
    ckpt_dir = f"{args.savedir}/state_{epoch}"
    model_engine.save_16bit_model(ckpt_dir, exclude_frozen_parameters=True)
    if global_rank == 0:
        from transformers import AutoConfig
        draft_config = AutoConfig.from_pretrained(args.draftpath)
        draft_config.save_pretrained(ckpt_dir)
        with open(os.path.join(ckpt_dir, "exit_layers.json"), 'w') as f:
            json.dump({"exit_layers": exit_layers}, f)
        print(f"  Saved checkpoint to {ckpt_dir}")
