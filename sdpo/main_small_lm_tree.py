"""Tree on-policy rollout training entry point for Small-LM draft.

Each training step:
  - Per sample, pick one valid anchor in the assistant turn (loss_mask=1).
  - Draft self-rolls out a 1-2-4 tree (root + 2 children + 4 grandchildren).
  - Target online tree-verifies (single 4D-masked forward).
  - Loss = anchor_kl_coef · mean_KL_per_node  -  eal_aux_coef · mean_EAL_per_path.

Usage:
  deepspeed sdpo/main_small_lm_tree.py \\
    --basepath Qwen/Qwen3-32B \\
    --draftpath Qwen/Qwen3-0.6B \\
    --trainpath sdpo/data/mixed_train_10K.jsonl \\
    --testpath  sdpo/data/mixed_val_80.jsonl \\
    --deepspeed_config sdpo/sdpo_config_qwen3_bf16_zero3_2gpu.json \\
    --savedir /scratch/.../tree_train/test \\
    --anchor_kl_coef 1.0 --eal_aux_coef 0.1 \\
    --max_len 2048 --num_epochs 3 --lr 2e-6
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
from small_lm_tree_train_model import SmallLMTreeTrainModel

torch.backends.cuda.matmul.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument('--basepath', required=True, help='Target LLM (frozen)')
parser.add_argument('--draftpath', required=True, help='Draft LLM (trainable)')
parser.add_argument('--trainpath', required=True)
parser.add_argument('--testpath', required=True)
parser.add_argument('--savedir',
                    default='/scratch/tx856/spec_reason/scratch/loss_train_smalllm_tree')
parser.add_argument('--anchor_kl_coef', type=float, default=1.0)
parser.add_argument('--eal_aux_coef', type=float, default=0.1)
parser.add_argument('--max_len', type=int, default=2048,
                    help='L2K = 2048 max sequence length.')
parser.add_argument('--max_train_samples', type=int, default=None)
parser.add_argument('--num_epochs', type=int, default=3)
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--gamma', type=int, default=7,
                    help='Only for dataset chunking compat — not used by tree model.')
parser.add_argument('--tree_depth', type=int, default=2,
                    help='Tree max depth (root=0). depth=2 → 1-2-4 (7 nodes, 4 paths). '
                         'depth=5 → 63 nodes, 32 paths.')
parser.add_argument('--local_rank', type=int, default=-1)
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

with open(args.deepspeed_config) as f:
    ds_config = json.load(f)

if ds_config["train_micro_batch_size_per_gpu"] != 1:
    print(f"[WARN] tree training needs micro_bs=1; forcing to 1 "
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

print(f"[SmallLMTree] {len(traindataset)} samples, {n_gpus} GPUs, "
      f"micro_bs={micro_bs}, effective_bs={effective_bs}, "
      f"{steps_per_epoch} steps/epoch, total={total_steps}")
print(f"[SmallLMTree] anchor_kl_coef={args.anchor_kl_coef} "
      f"eal_aux_coef={args.eal_aux_coef} max_len={args.max_len} "
      f"tree_depth={args.tree_depth}")

model_dtype = torch.bfloat16 if ds_config.get("bf16", {}).get("enabled") else torch.float16
zero_stage = ds_config.get("zero_optimization", {}).get("stage", 0)
_hf_ds_config = None
if zero_stage == 3:
    from transformers.integrations import HfDeepSpeedConfig
    _hf_ds_config = HfDeepSpeedConfig(ds_config)

model = SmallLMTreeTrainModel(
    target_path=args.basepath,
    draft_path=args.draftpath,
    tree_depth=args.tree_depth,
    dtype=model_dtype,
)

args.deepspeed_config = None
trainable_params = [p for p in model.parameters() if p.requires_grad]
model_engine, optimizer, _, _ = deepspeed.initialize(
    args=args, model=model, model_parameters=trainable_params,
    config=ds_config)

global_rank = deepspeed.comm.get_rank()
rank = deepspeed.comm.get_local_rank()
world_size = deepspeed.comm.get_world_size()

if global_rank == 0:
    run_name = f"smalllm_tree_kl{args.anchor_kl_coef}_eal{args.eal_aux_coef}"
    wandb.init(project="smalllm_tree", name=run_name, config={
        "anchor_kl_coef": args.anchor_kl_coef,
        "eal_aux_coef": args.eal_aux_coef,
        "max_len": args.max_len, "num_epochs": args.num_epochs,
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
    print(f"[SmallLMTree] Resuming from {ckpt_path}")
    model_engine.load_checkpoint(ckpt_path)

global_step = 0
for epoch in range(start_epoch, args.num_epochs):
    train_sampler.set_epoch(epoch + 1)
    model.train()
    print(f"[SmallLMTree] Epoch {epoch}")

    sum_total = 0.0
    sum_kl = 0.0
    sum_eal = 0.0
    sum_anchors = 0
    n_batches = 0

    for data in tqdm(train_loader, desc=f"Epoch {epoch} train"):
        model.zero_grad()
        total_loss, _, metrics = model_engine(
            input_ids=data["input_ids"].to(rank),
            attention_mask=data["attention_mask"].to(rank),
            loss_mask=data["loss_mask"].to(rank),
            anchor_kl_coef=args.anchor_kl_coef,
            eal_aux_coef=args.eal_aux_coef,
        )
        # Always call backward+step (even if num_anchors==0, loss==0 with valid
        # graph) — `continue` here causes per-rank divergence → NCCL timeout.
        # See .claude/mem/memory.md 2026-04-21.
        model_engine.backward(total_loss)
        grad_norm = model_engine.get_global_grad_norm()
        model_engine.step()
        global_step += 1
        if metrics.get("num_anchors", 0) == 0:
            continue
        n_batches += 1

        sum_total += metrics["total_loss"]
        sum_kl += metrics["kl_loss"]
        sum_eal += metrics["eal_mean"]
        sum_anchors += metrics["num_anchors"]

        if global_rank == 0:
            wandb.log({
                "train/total_loss": metrics["total_loss"],
                "train/kl_loss": metrics["kl_loss"],
                "train/aux_loss": metrics["aux_loss"],
                "train/eal_mean": metrics["eal_mean"],
                "train/num_anchors": metrics["num_anchors"],
                "train/grad_norm": grad_norm if grad_norm is not None else float("nan"),
                "train/lr": model_engine.get_lr()[0],
            }, step=global_step)

    if global_rank == 0 and n_batches > 0:
        print(f"  epoch {epoch} | total={sum_total/n_batches:.4f}  "
              f"kl={sum_kl/n_batches:.4f}  eal={sum_eal/n_batches:.4f}  "
              f"avg_anchors={sum_anchors/n_batches:.1f}")

    # Eval
    model.eval()
    e_total, e_kl, e_eal, e_n = 0.0, 0.0, 0.0, 0
    for data in tqdm(test_loader, desc=f"Epoch {epoch} eval"):
        with torch.no_grad():
            _, _, m = model_engine(
                input_ids=data["input_ids"].to(rank),
                attention_mask=data["attention_mask"].to(rank),
                loss_mask=data["loss_mask"].to(rank),
                anchor_kl_coef=args.anchor_kl_coef,
                eal_aux_coef=args.eal_aux_coef,
            )
        if m["num_valid"] == 0 or m.get("num_anchors", 0) == 0:
            continue
        e_total += m["total_loss"]
        e_kl += m["kl_loss"]
        e_eal += m["eal_mean"]
        e_n += 1

    if global_rank == 0 and e_n > 0:
        print(f"  epoch {epoch} | eval total={e_total/e_n:.4f}  "
              f"kl={e_kl/e_n:.4f}  eal={e_eal/e_n:.4f}")
        wandb.log({
            "epoch": epoch,
            "eval/total_loss": e_total / e_n,
            "eval/kl_loss": e_kl / e_n,
            "eval/eal_mean": e_eal / e_n,
        }, step=global_step)

    deepspeed.comm.barrier()
    torch.cuda.empty_cache()

    ckpt_dir = f"{args.savedir}/state_{epoch}"
    model_engine.save_16bit_model(ckpt_dir, exclude_frozen_parameters=True)
    if global_rank == 0:
        from transformers import AutoConfig
        draft_config = AutoConfig.from_pretrained(args.draftpath)
        draft_config.save_pretrained(ckpt_dir)
        print(f"  Saved checkpoint to {ckpt_dir}")
