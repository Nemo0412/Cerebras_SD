"""
Rejection-Focused SDPO Training Script.

L_total = L_SDPO (on rejected suffix) + eagle_coef * L_EAGLE (anchor on all positions)

Usage:
  deepspeed sdpo/main.py \
    --basepath  /path/to/llama3-8b-instruct \
    --draftpath /path/to/EAGLE3-LLaMA3.1-Instruct-8B \
    --trainpath sdpo/data/train.jsonl \
    --testpath  sdpo/data/test.jsonl \
    --savedir   sdpo_checkpoints \
    --deepspeed_config sdpo/sdpo_config.json \
    --eagle_coef 0.003 --gamma 7 --num_epochs 3
"""

import argparse
import json
import math
import os
import re
import shutil
import sys
from types import SimpleNamespace

import deepspeed
import torch
import wandb
from datasets import load_dataset
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as sf_load, save_file as sf_save
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from traineagle3.configs import EConfig
from rejection_model import RejectionModel

torch.backends.cuda.matmul.allow_tf32 = True

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--basepath', required=True)
parser.add_argument('--draftpath', required=True)
parser.add_argument('--trainpath', required=True)
parser.add_argument('--testpath', required=True)
parser.add_argument('--savedir', default='sdpo_checkpoints')
parser.add_argument('--eagle_coef', type=float, default=0.05)
parser.add_argument('--gamma', type=int, default=7)
parser.add_argument('--max_len', type=int, default=2048)
parser.add_argument('--max_train_samples', type=int, default=None)
parser.add_argument('--num_epochs', type=int, default=3)
parser.add_argument('--baseline', type=str, default=None, choices=['eagle_only'])
parser.add_argument('--config_path', type=str,
                    default=os.path.join(os.path.dirname(__file__), 'config.json'))
parser.add_argument('--local_rank', type=int, default=-1)
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

with open(args.deepspeed_config) as f:
    ds_config = json.load(f)

train_config = SimpleNamespace(
    bs=ds_config["train_micro_batch_size_per_gpu"],
    num_epochs=args.num_epochs,
    num_workers=0,
    max_len=args.max_len,
    config_path=args.config_path,
    gradient_checkpointing=True,
    eagle_coef=args.eagle_coef,
    kl_coef=0,
    gamma=args.gamma,
    baseline=args.baseline,
)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
SEP_ASST = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
SEP_USER = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
SYSTEM_MSG = ("You are a helpful, respectful and honest assistant. "
              "Always answer as helpfully as possible, while being safe.")


def build_dataset(tokenizer, datapath, max_len, max_samples=None):
    ds = load_dataset('json', data_files=datapath)['train']
    ds = ds.shuffle(seed=42)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    def preprocess(examples):
        out = {"attention_mask": [], "input_ids": [], "loss_mask": []}
        roles = {"human": "user", "gpt": "assistant"}

        for i in range(len(examples['id'])):
            src = examples['conversations'][i]
            if not src:
                continue
            if roles.get(src[0]["from"]) != "user":
                src = src[1:]

            msgs = [{"role": "system", "content": SYSTEM_MSG}]
            for j, sent in enumerate(src):
                msgs.append({"role": roles[sent["from"]], "content": sent["value"]})

            conversation = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False)
            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id = tokenizer.unk_token_id
            ids = tokenizer(conversation, return_tensors="pt",
                            add_special_tokens=False).input_ids[0]
            if len(ids) > max_len:
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

            out["input_ids"].append(ids[None, :])
            out["loss_mask"].append(loss_mask[None, :])
            out["attention_mask"].append(torch.ones_like(loss_mask)[None, :])
        return out

    ds = ds.map(preprocess, batched=True, num_proc=8,
                remove_columns=ds.column_names)
    ds.set_format(type="torch")
    return ds


class DataCollator:
    @staticmethod
    def pad2d(tensors, N):
        B, _ = tensors[0].shape
        return torch.cat([
            torch.cat([t, torch.zeros(B, N - t.shape[1], dtype=t.dtype)], dim=1)
            for t in tensors
        ])

    def __call__(self, features):
        max_len = max(f['input_ids'].shape[1] for f in features)
        return {
            "input_ids": self.pad2d([f['input_ids'] for f in features], max_len),
            "attention_mask": self.pad2d([f['attention_mask'] for f in features], max_len),
            "loss_mask": self.pad2d([f['loss_mask'] for f in features], max_len),
        }


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------
def resolve_hf_path(path, label):
    if os.path.isdir(path):
        return path
    print(f"[SDPO] Downloading {label}: {path}")
    return snapshot_download(path)


def load_draft_weights(model, draftpath):
    sf_path = os.path.join(draftpath, "model.safetensors")
    bin_path = os.path.join(draftpath, "pytorch_model.bin")
    if os.path.exists(sf_path):
        state = sf_load(sf_path, device="cpu")
    elif os.path.exists(bin_path):
        state = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No draft weights in {draftpath}")

    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    for vk in ("d2t", "t2d"):
        state.pop(vk, None)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys: {unexpected}")
    print(f"[SDPO] Loaded draft weights from {draftpath}")


def find_latest_checkpoint(directory):
    max_epoch = -1
    for sub in os.listdir(directory):
        m = re.match(r"state_(\d+)", sub)
        if m and os.path.exists(os.path.join(directory, sub, "zero_to_fp32.py")):
            max_epoch = max(max_epoch, int(m.group(1)))
    if max_epoch == -1:
        return None, 0
    return f"{directory}/state_{max_epoch}", max_epoch + 1


def patch_vocab_buffers(ckpt_dir, model):
    """Patch d2t/t2d into saved checkpoint (not saved by save_16bit_model)."""
    sf_path = os.path.join(ckpt_dir, "model.safetensors")
    bin_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    if os.path.exists(sf_path) and os.path.isfile(sf_path):
        sd = sf_load(sf_path)
        sd["d2t"] = model.d2t.cpu()
        sd["t2d"] = model.t2d.cpu()
        sf_save(sd, sf_path)
    elif os.path.isfile(bin_path):
        sd = torch.load(bin_path, map_location="cpu")
        sd["d2t"] = model.d2t.cpu()
        sd["t2d"] = model.t2d.cpu()
        torch.save(sd, bin_path)
    elif os.path.isdir(bin_path):
        shards = sorted(f for f in os.listdir(bin_path) if f.endswith(".bin"))
        last = os.path.join(bin_path, shards[-1])
        sd = torch.load(last, map_location="cpu")
        sd["d2t"] = model.d2t.cpu()
        sd["t2d"] = model.t2d.cpu()
        torch.save(sd, last)
        idx_path = os.path.join(bin_path, "pytorch_model.bin.index.json")
        if os.path.exists(idx_path):
            with open(idx_path) as f:
                idx = json.load(f)
            idx["weight_map"]["d2t"] = shards[-1]
            idx["weight_map"]["t2d"] = shards[-1]
            with open(idx_path, "w") as f:
                json.dump(idx, f, indent=2)


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------
basepath = resolve_hf_path(args.basepath, "base model")
draftpath = resolve_hf_path(args.draftpath, "draft model")

tokenizer = AutoTokenizer.from_pretrained(basepath)
traindataset = build_dataset(tokenizer, args.trainpath, args.max_len,
                             max_samples=args.max_train_samples)
testdataset = build_dataset(tokenizer, args.testpath, args.max_len)

# Auto-compute LR schedule
n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
micro_bs = ds_config["train_micro_batch_size_per_gpu"]
grad_accum = ds_config["gradient_accumulation_steps"]
effective_bs = micro_bs * grad_accum * n_gpus
steps_per_epoch = math.ceil(len(traindataset) / effective_bs)
total_steps = steps_per_epoch * args.num_epochs
warmup_steps = total_steps // 10
ds_config["scheduler"]["params"]["total_num_steps"] = total_steps
ds_config["scheduler"]["params"]["warmup_num_steps"] = warmup_steps
print(f"[SDPO] {len(traindataset)} samples, {n_gpus} GPUs, "
      f"effective_bs={effective_bs}, {steps_per_epoch} steps/epoch, "
      f"total={total_steps}, warmup={warmup_steps}")

draft_cfg_path = os.path.join(draftpath, "config.json")
config = EConfig.from_pretrained(
    draft_cfg_path if os.path.exists(draft_cfg_path) else args.config_path)
config.gradient_checkpointing = True

model = RejectionModel(config, ds_config, train_config, path=basepath,
                       load_emb=True, load_head=True)
load_draft_weights(model, draftpath)

draft_cache = os.path.join(draftpath, "cache.pt")
if os.path.exists(draft_cache):
    shutil.copy(draft_cache, "cache.pt")
model.scandata(args.trainpath, basepath)

model_engine, optimizer, _, _ = deepspeed.initialize(
    args=args, model=model, model_parameters=model.parameters())

global_rank = deepspeed.comm.get_rank()
rank = deepspeed.comm.get_local_rank()
world_size = deepspeed.comm.get_world_size()
model_engine.module.length = args.gamma

if global_rank == 0:
    wandb.init(project="sdpo_rejection", config={
        "eagle_coef": args.eagle_coef,
        "gamma": args.gamma,
        "num_epochs": args.num_epochs,
        "basepath": args.basepath,
        "draftpath": args.draftpath,
        "world_size": world_size,
        "baseline": args.baseline,
    })

os.makedirs(args.savedir, exist_ok=True)
train_sampler = DistributedSampler(traindataset, num_replicas=world_size,
                                   rank=global_rank, shuffle=True)
test_sampler = DistributedSampler(testdataset, num_replicas=world_size,
                                  rank=global_rank, shuffle=False)
train_loader = DataLoader(traindataset, batch_size=train_config.bs,
                          sampler=train_sampler, num_workers=0,
                          pin_memory=True, collate_fn=DataCollator())
test_loader = DataLoader(testdataset, batch_size=train_config.bs,
                         sampler=test_sampler, num_workers=0,
                         pin_memory=True, collate_fn=DataCollator())

ckpt_path, start_epoch = find_latest_checkpoint(args.savedir)
if ckpt_path:
    print(f"[SDPO] Resuming from {ckpt_path}")
    model_engine.load_checkpoint(ckpt_path)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
global_step = 0
for epoch in range(start_epoch, args.num_epochs):
    train_sampler.set_epoch(epoch + 1)
    model.train()
    print(f"[SDPO] Epoch {epoch}")

    epoch_metrics = {"sdpo_loss": [], "eagle_loss": [], "mean_tau": []}
    epoch_step_acc = [[] for _ in range(args.gamma)]
    epoch_tau_hist = [0] * (args.gamma + 1)

    for data in tqdm(train_loader, desc=f"Epoch {epoch} train"):
        model.zero_grad()
        total_loss, sdpo_loss, metrics = model_engine(
            input_ids=data["input_ids"].to(rank),
            attention_mask=data["attention_mask"].to(rank),
            loss_mask=data["loss_mask"].to(rank),
            eagle_coef=args.eagle_coef,
            baseline=args.baseline,
        )
        if metrics["num_valid"] == 0:
            continue

        model_engine.backward(total_loss)
        grad_norm = model_engine.get_global_grad_norm()
        model_engine.step()
        global_step += 1

        epoch_metrics["sdpo_loss"].append(metrics["sdpo_loss"])
        epoch_metrics["eagle_loss"].append(metrics["eagle_loss"])
        epoch_metrics["mean_tau"].append(metrics["mean_tau"])
        for k in range(args.gamma):
            epoch_step_acc[k].append(metrics["step_acc"][k])
        for v in range(args.gamma + 1):
            epoch_tau_hist[v] += metrics["tau_hist"][v]

        if global_rank == 0:
            log = {
                "train/total_loss": total_loss.item(),
                "train/sdpo_loss": metrics["sdpo_loss"],
                "train/eagle_loss": metrics["eagle_loss"],
                "train/mean_tau": metrics["mean_tau"],
                "train/grad_norm": grad_norm if grad_norm is not None else float("nan"),
                "train/lr": model_engine.get_lr()[0],
            }
            for k in range(args.gamma):
                log[f"train/step{k}_acc"] = metrics["step_acc"][k]
            wandb.log(log, step=global_step)

    # Epoch summary
    if global_rank == 0:
        for name in epoch_metrics:
            v = sum(epoch_metrics[name]) / len(epoch_metrics[name])
            print(f"  epoch {epoch} | {name}: {v:.4f}")
        total = sum(epoch_tau_hist)
        print(f"  τ distribution: {[f'{h/total:.3f}' for h in epoch_tau_hist]}")

    # Eval
    model.eval()
    eval_metrics = {"sdpo_loss": [], "mean_tau": []}
    eval_step_acc = [[] for _ in range(args.gamma)]

    for data in tqdm(test_loader, desc=f"Epoch {epoch} eval"):
        with torch.no_grad():
            _, _, metrics = model_engine(
                input_ids=data["input_ids"].to(rank),
                attention_mask=data["attention_mask"].to(rank),
                loss_mask=data["loss_mask"].to(rank),
                eagle_coef=args.eagle_coef,
                baseline=args.baseline,
            )
        if metrics["num_valid"] > 0:
            eval_metrics["sdpo_loss"].append(metrics["sdpo_loss"])
            eval_metrics["mean_tau"].append(metrics["mean_tau"])
            for k in range(args.gamma):
                eval_step_acc[k].append(metrics["step_acc"][k])

    if global_rank == 0:
        eval_log = {"epoch": epoch}
        for name in eval_metrics:
            v = sum(eval_metrics[name]) / len(eval_metrics[name])
            eval_log[f"eval/{name}"] = v
            print(f"  epoch {epoch} | eval {name}: {v:.4f}")
        for k in range(args.gamma):
            eval_log[f"eval/step{k}_acc"] = sum(eval_step_acc[k]) / len(eval_step_acc[k])
        wandb.log(eval_log, step=global_step)

    deepspeed.comm.barrier()
    torch.cuda.empty_cache()

    # Save checkpoint
    ckpt_dir = f"{args.savedir}/state_{epoch}"
    deepspeed.DeepSpeedEngine.save_checkpoint(model_engine, save_dir=ckpt_dir)
    model_engine.save_16bit_model(ckpt_dir, exclude_frozen_parameters=True)
    if global_rank == 0:
        patch_vocab_buffers(ckpt_dir, model)
