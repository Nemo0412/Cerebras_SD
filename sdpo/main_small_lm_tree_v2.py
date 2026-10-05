"""Training entry for SmallLMTreeV2Model (arbitrary top_k_per_depth).

Matches production eval tree shape:
  --top_k_per_depth 4,3,2,1,1,1,1   # default = eval tree (gamma=7)

Usage:
  deepspeed sdpo/main_small_lm_tree_v2.py \\
    --basepath Qwen/Qwen3-8B \\
    --draftpath Qwen/Qwen3-0.6B \\
    --trainpath sdpo/data/sharegpt_qwen3_8b_regen.jsonl \\
    --testpath  sdpo/data/mixed_val_80.jsonl \\
    --deepspeed_config sdpo/sdpo_config_qwen3_bf16_zero3_2gpu.json \\
    --savedir ... \\
    --top_k_per_depth 4,3,2,1,1,1,1 \\
    --max_len 2048 --num_epochs 6
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys

import deepspeed
import torch
import wandb
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline import build_dataset, DataCollator
from small_lm_tree_v2_model import SmallLMTreeV2Model

torch.backends.cuda.matmul.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument('--basepath', required=True)
parser.add_argument('--draftpath', required=True)
parser.add_argument('--trainpath', required=True)
parser.add_argument('--testpath', required=True)
parser.add_argument('--savedir',
                    default='/scratch/tx856/spec_reason/scratch/loss_train_smalllm_tree_v2')
parser.add_argument('--anchor_kl_coef', type=float, default=1.0)
parser.add_argument('--eal_aux_coef', type=float, default=0.1)
parser.add_argument('--max_len', type=int, default=2048)
parser.add_argument('--max_train_samples', type=int, default=None)
parser.add_argument('--num_epochs', type=int, default=6)
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--gamma', type=int, default=7,
                    help='Only for dataset chunking compat — tree shape uses top_k_per_depth.')
parser.add_argument('--top_k_per_depth', type=str, default='4,3,2,1,1,1,1',
                    help='Comma-separated branching factors per tree depth. '
                         'Default = production eval tree (gamma=7).')
parser.add_argument('--tree_budget', type=int, default=0,
                    help='Max tree nodes. 0 keeps the full expansion. '
                         '128 with top_k 4,3,2,1,1,1,1 keeps 16 of 24 depth-6 nodes.')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='Target softmax temperature for Tree-EAL acceptance '
                         'probabilities. Draft tree stays greedy top-k.')
parser.add_argument('--eal_mode', type=str, default='budget',
                    choices=['budget', 'path_topk', 'node_marginal', 'soft_topk'],
                    help='budget: KL minus E[L] on the node-budget tree. '
                         'path_topk: KL minus exact E[L] on the Top-K path union. '
                         'node_marginal: old node-weighted NLL, not used. '
                         'soft_topk: -TreeEAL(beta * alpha) with soft Top-K.')
parser.add_argument('--path_topk', type=int, default=16,
                    help='K for path_topk, node_marginal, and soft_topk.')
parser.add_argument('--local_rank', type=int, default=-1)
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

top_k_list = [int(x) for x in args.top_k_per_depth.split(',') if x.strip()]
assert len(top_k_list) >= 1, "top_k_per_depth must be non-empty"

with open(args.deepspeed_config) as f:
    ds_config = json.load(f)

micro_bs_cfg = ds_config["train_micro_batch_size_per_gpu"]
print(f"[SmallLMTreeV2] micro_bs={micro_bs_cfg} "
      f"(tree rollout is batched across the microbatch)")


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

# Print tree shape
depth_counts = [top_k_list[0]]
for d in range(1, len(top_k_list)):
    depth_counts.append(depth_counts[-1] * top_k_list[d])
N_nodes = sum(depth_counts)
n_paths = depth_counts[-1]

print(f"[SmallLMTreeV2] {len(traindataset)} samples, {n_gpus} GPUs, "
      f"micro_bs={micro_bs}, effective_bs={effective_bs}, "
      f"{steps_per_epoch} steps/epoch, total={total_steps}")
print(f"[SmallLMTreeV2] top_k_per_depth={top_k_list}")
print(f"[SmallLMTreeV2] depth_counts={depth_counts}  N_nodes={N_nodes}  n_paths={n_paths}")
print(f"[SmallLMTreeV2] anchor_kl_coef={args.anchor_kl_coef} eal_aux_coef={args.eal_aux_coef} "
      f"temperature={args.temperature} tree_budget={args.tree_budget or N_nodes} "
      f"max_len={args.max_len}")
print("[SmallLMTreeV2] loss = anchor_kl_coef * KL - eal_aux_coef * E[L_tree] "
      f"eal_mode={args.eal_mode} path_topk={args.path_topk}")

model_dtype = torch.bfloat16 if ds_config.get("bf16", {}).get("enabled") else torch.float16
zero_stage = ds_config.get("zero_optimization", {}).get("stage", 0)
_hf_ds_config = None
if zero_stage == 3:
    from transformers.integrations import HfDeepSpeedConfig
    _hf_ds_config = HfDeepSpeedConfig(ds_config)

model = SmallLMTreeV2Model(
    target_path=args.basepath,
    draft_path=args.draftpath,
    top_k_per_depth=top_k_list,
    dtype=model_dtype,
    tree_budget=(args.tree_budget if args.tree_budget > 0 else None),
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
    tag = "_".join(str(x) for x in top_k_list)
    run_name = (f"tree_eal_{args.eal_mode}_k{args.path_topk}_topk{tag}"
                f"_b{args.tree_budget or N_nodes}"
                f"_kl{args.anchor_kl_coef}_eal{args.eal_aux_coef}_T{args.temperature}")
    wandb.init(project="smalllm_tree_v2", name=run_name, config={
        "anchor_kl_coef": args.anchor_kl_coef,
        "eal_aux_coef": args.eal_aux_coef,
        "temperature": args.temperature,
        "tree_budget": args.tree_budget or N_nodes,
        "eal_mode": args.eal_mode,
        "path_topk": args.path_topk,
        "loss": "KL - E[L_tree]",
        "max_len": args.max_len, "num_epochs": args.num_epochs,
        "basepath": args.basepath, "draftpath": args.draftpath,
        "world_size": world_size, "lr": args.lr,
        "top_k_per_depth": top_k_list,
        "tree_N_nodes": N_nodes, "tree_n_paths": n_paths,
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
    print(f"[SmallLMTreeV2] Resuming from {ckpt_path}")
    model_engine.load_checkpoint(ckpt_path)

step_log_path = os.path.join(args.savedir, "train_steps.jsonl")
epoch_log_path = os.path.join(args.savedir, "epoch_metrics.jsonl")
eval_sbatch = "/scratch/ll5914/logs/tree_eal/eval_soft_topk_epoch.sbatch"


def _append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


def _submit_epoch_eval(epoch):
    """Queue the real verify_tree_step eval for this epoch's draft."""
    if not os.path.isfile(eval_sbatch):
        print(f"  epoch {epoch} | eval sbatch missing: {eval_sbatch}")
        return None
    partition = os.environ.get("SLURM_JOB_PARTITION", "")
    account = os.environ.get("SLURM_JOB_ACCOUNT", "")
    cmd = ["sbatch", "--parsable",
           f"--job-name=eal_ev_e{epoch}",
           f"--export=ALL,EPOCH={epoch}"]
    if partition:
        cmd.append(f"--partition={partition}")
    if account:
        cmd.append(f"--account={account}")
    cmd.append(eval_sbatch)
    try:
        out = subprocess.check_output(cmd, text=True).strip()
    except subprocess.CalledProcessError as exc:
        print(f"  epoch {epoch} | eval submit failed: {exc}")
        return None
    print(f"  epoch {epoch} | queued accept-length eval job {out}")
    return out


global_step = start_epoch * iters_per_epoch
for epoch in range(start_epoch, args.num_epochs):
    train_sampler.set_epoch(epoch + 1)
    model.train()
    print(f"[SmallLMTreeV2] Epoch {epoch}")

    sum_total = 0.0
    sum_kl = 0.0
    sum_eal = 0.0
    sum_eal_loss = 0.0
    sum_gn = 0.0
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
            temperature=args.temperature,
            eal_mode=args.eal_mode,
            path_topk=args.path_topk,
        )
        # Always backward+step (rank-safe). See memory.md 2026-04-21.
        model_engine.backward(total_loss)
        grad_norm = model_engine.get_global_grad_norm()
        model_engine.step()
        global_step += 1
        if metrics.get("num_anchors", 0) == 0:
            continue
        n_batches += 1

        gn = float("nan") if grad_norm is None else float(grad_norm)
        sum_total += metrics["total_loss"]
        sum_kl += metrics["kl_loss"]
        sum_eal += metrics["eal_mean"]
        sum_eal_loss += metrics["eal_loss"]
        sum_gn += 0.0 if gn != gn else gn
        sum_anchors += metrics["num_anchors"]

        if global_rank == 0:
            lr = model_engine.get_lr()[0]
            row = {
                "step": global_step,
                "epoch": epoch,
                "loss": metrics["total_loss"],
                "eal_loss": metrics["eal_loss"],
                "analytic_accepted_length": metrics["eal_mean"],
                "kl": metrics["kl_loss"],
                "grad_norm": gn,
                "lr": lr,
                "num_anchors": metrics["num_anchors"],
                "tree_nodes": metrics["tree_nodes"],
            }
            _append_jsonl(step_log_path, row)
            print(f"  step {global_step} epoch {epoch} | "
                  f"loss={row['loss']:.4f}  eal_loss={row['eal_loss']:.4f}  "
                  f"analytic_accepted_length={row['analytic_accepted_length']:.4f}  "
                  f"kl={row['kl']:.4f}  grad_norm={row['grad_norm']:.4f}",
                  flush=True)
            wandb.log({
                "train/total_loss": metrics["total_loss"],
                "train/kl_loss": metrics["kl_loss"],
                "train/aux_loss": metrics["aux_loss"],
                "train/eal_loss": metrics["eal_loss"],
                "train/eal_mean": metrics["eal_mean"],
                "train/num_anchors": metrics["num_anchors"],
                "train/grad_norm": gn,
                "train/lr": lr,
            }, step=global_step)

    if global_rank == 0 and n_batches > 0:
        print(f"  epoch {epoch} | total={sum_total/n_batches:.4f}  "
              f"eal_loss={sum_eal_loss/n_batches:.4f}  "
              f"analytic_E[L]={sum_eal/n_batches:.4f}  "
              f"grad_norm={sum_gn/n_batches:.4f}  "
              f"kl={sum_kl/n_batches:.4f}  "
              f"avg_anchors={sum_anchors/n_batches:.1f}")

    model.eval()
    e_total, e_kl, e_eal, e_eal_loss, e_n = 0.0, 0.0, 0.0, 0.0, 0
    for data in tqdm(test_loader, desc=f"Epoch {epoch} eval"):
        with torch.no_grad():
            _, _, m = model_engine(
                input_ids=data["input_ids"].to(rank),
                attention_mask=data["attention_mask"].to(rank),
                loss_mask=data["loss_mask"].to(rank),
                anchor_kl_coef=args.anchor_kl_coef,
                eal_aux_coef=args.eal_aux_coef,
                temperature=args.temperature,
                eal_mode=args.eal_mode,
                path_topk=args.path_topk,
            )
        if m["num_valid"] == 0 or m.get("num_anchors", 0) == 0:
            continue
        e_total += m["total_loss"]
        e_kl += m["kl_loss"]
        e_eal += m["eal_mean"]
        e_eal_loss += m["eal_loss"]
        e_n += 1

    if global_rank == 0 and e_n > 0:
        eval_row = {
            "epoch": epoch,
            "train_loss": sum_total / max(n_batches, 1),
            "train_eal_loss": sum_eal_loss / max(n_batches, 1),
            "train_analytic_accepted_length": sum_eal / max(n_batches, 1),
            "train_kl": sum_kl / max(n_batches, 1),
            "train_grad_norm": sum_gn / max(n_batches, 1),
            "test_loss": e_total / e_n,
            "test_eal_loss": e_eal_loss / e_n,
            "test_analytic_accepted_length": e_eal / e_n,
            "test_kl": e_kl / e_n,
            "test_batches": e_n,
        }
        print(f"  epoch {epoch} | test loss={eval_row['test_loss']:.4f}  "
              f"test_eal_loss={eval_row['test_eal_loss']:.4f}  "
              f"test_analytic_accepted_length="
              f"{eval_row['test_analytic_accepted_length']:.4f}  "
              f"test_kl={eval_row['test_kl']:.4f}",
              flush=True)
        _append_jsonl(epoch_log_path, eval_row)
        wandb.log({
            "epoch": epoch,
            "eval/total_loss": eval_row["test_loss"],
            "eval/eal_loss": eval_row["test_eal_loss"],
            "eval/kl_loss": eval_row["test_kl"],
            "eval/eal_mean": eval_row["test_analytic_accepted_length"],
        }, step=global_step)

    deepspeed.comm.barrier()
    torch.cuda.empty_cache()

    ckpt_dir = f"{args.savedir}/state_{epoch}"
    model_engine.save_checkpoint(ckpt_dir)
    if global_rank == 0:
        hf_dir = f"{args.savedir}/hf_epoch_{epoch}"
        model_engine.module.draft_model.save_pretrained(hf_dir)
        tokenizer.save_pretrained(hf_dir)
        print(f"  Saved DeepSpeed checkpoint to {ckpt_dir}")
        print(f"  Saved HF draft to {hf_dir}")
        eval_job = _submit_epoch_eval(epoch)
        if eval_job is not None:
            _append_jsonl(epoch_log_path, {
                "epoch": epoch,
                "accept_length_eval_job": eval_job,
                "draft": hf_dir,
            })
    deepspeed.comm.barrier()
