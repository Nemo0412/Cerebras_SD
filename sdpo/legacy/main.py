"""
SDPO Training Script for EAGLE3 Draft Model

Usage:
  deepspeed sdpo/main.py \
    --basepath  /path/to/llama3-8b \
    --draftpath yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
    --trainpath /path/to/train.jsonl \
    --testpath  /path/to/test.jsonl \
    --savedir   sdpo_checkpoints \
    --deepspeed_config sdpo/sdpo_config.json
"""

import argparse
import json
import math
import os
import re
import shutil
import sys
from types import SimpleNamespace
from typing import Any, Dict, List

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
from sdpo_model import SDPOModel

torch.backends.cuda.matmul.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument('--basepath', required=True)
parser.add_argument('--draftpath', required=True,
                    help='Pre-trained EAGLE3 checkpoint dir or HF repo')
parser.add_argument('--trainpath', required=True)
parser.add_argument('--testpath', required=True)
parser.add_argument('--savedir', default='sdpo_checkpoints')
parser.add_argument('--eagle_coef', type=float, default=0.003,
                    help='Weight on L_EAGLE anchor loss. 0=pure SDPO.')
parser.add_argument('--kl_coef', type=float, default=0.1)
parser.add_argument('--num_rollouts', type=int, default=4)
parser.add_argument('--gamma', type=int, default=7)
parser.add_argument('--max_len', type=int, default=2048)
parser.add_argument('--max_train_samples', type=int, default=None)
parser.add_argument('--num_epochs', type=int, default=3)
parser.add_argument('--mode', type=str, default='eager', choices=['eager', 'sampling'],
                    help='Advantage mode: eager uses S-advantage (unbiased, greedy chain fixed); '
                         'sampling uses tau-advantage to correct confidence-vs-length bias.')
parser.add_argument('--baseline', type=str, default=None, choices=['eagle_only'],
                    help='Baseline mode. eagle_only: pure L_EAGLE, no rollouts.')
parser.add_argument('--config_path', type=str,
                    default=os.path.join(os.path.dirname(__file__), 'config.json'))
parser.add_argument('--local_rank', type=int, default=-1)
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

with open(args.deepspeed_config) as f:
    ds_config = json.load(f)

train_config = {
    "bs": ds_config["train_micro_batch_size_per_gpu"],
    "num_epochs": args.num_epochs,
    "num_workers": 4,
    "max_len": args.max_len,
    "config_path": args.config_path,
    "gradient_checkpointing": True,
    "eagle_coef": args.eagle_coef,
    "kl_coef": args.kl_coef,
    "num_rollouts": args.num_rollouts,
    "gamma": args.gamma,
    "mode": args.mode,
    "baseline": args.baseline,
}
# cnets.py accesses train_config via dot notation, not dict
model_ns = SimpleNamespace(**train_config)

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
        convroles = ["user", "assistant"]

        for i in range(len(examples['id'])):
            src = examples['conversations'][i]
            if not src:
                continue
            if roles.get(src[0]["from"]) != "user":
                src = src[1:]

            msgs = [{"role": "system", "content": SYSTEM_MSG}]
            for j, sent in enumerate(src):
                role = roles[sent["from"]]
                assert role == convroles[j % 2], f"role mismatch at {i}"
                msgs.append({"role": role, "content": sent["value"]})

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

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(f['input_ids'].shape[1] for f in features)
        return {
            "input_ids": self.pad2d([f['input_ids'] for f in features], max_len),
            "attention_mask": self.pad2d([f['attention_mask'] for f in features], max_len),
            "loss_mask": self.pad2d([f['loss_mask'] for f in features], max_len),
        }


def resolve_hf_path(path, label):
    if os.path.isdir(path):
        return path
    print(f"[SDPO] Downloading {label} from HuggingFace: {path}")
    return snapshot_download(path)


def load_draft_weights(model, draftpath):
    sf_path = os.path.join(draftpath, "model.safetensors")
    bin_path = os.path.join(draftpath, "pytorch_model.bin")
    if os.path.exists(sf_path):
        state = sf_load(sf_path, device="cpu")
    elif os.path.exists(bin_path):
        state = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No draft weights found in {draftpath}")

    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}

    # d2t/t2d are buffers registered by scandata(), not parameters
    for vk in ("d2t", "t2d"):
        state.pop(vk, None)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in draft checkpoint:\n{unexpected}")
    draft_missing = [k for k in missing
                     if not k.startswith("target_model.") and not k.startswith("embed_tokens.")]
    if draft_missing:
        print(f"[SDPO] WARNING: draft keys missing from checkpoint: {draft_missing}")
    print(f"[SDPO] Loaded draft weights from {draftpath}")


def find_latest_checkpoint(directory):
    max_epoch = -1
    for sub in os.listdir(directory):
        m = re.match(r"state_(\d+)", sub)
        if m:
            if os.path.exists(os.path.join(directory, sub, "zero_to_fp32.py")):
                max_epoch = max(max_epoch, int(m.group(1)))
    if max_epoch == -1:
        return None, 0
    return f"{directory}/state_{max_epoch}", max_epoch + 1


basepath_resolved = resolve_hf_path(args.basepath, "base model")
draftpath_resolved = resolve_hf_path(args.draftpath, "draft model")

tokenizer = AutoTokenizer.from_pretrained(basepath_resolved)
traindataset = build_dataset(tokenizer, args.trainpath, train_config["max_len"],
                             max_samples=args.max_train_samples)
testdataset = build_dataset(tokenizer, args.testpath, train_config["max_len"])

# Auto-compute LR scheduler steps from actual dataset size.
# This removes the need to manually update sdpo_config.json when changing
# epochs, dataset, batch size, or GPU count.
_n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
_micro_bs = ds_config["train_micro_batch_size_per_gpu"]
_grad_accum = ds_config["gradient_accumulation_steps"]
_effective_bs = _micro_bs * _grad_accum * _n_gpus
_steps_per_epoch = math.ceil(len(traindataset) / _effective_bs)
_total_steps = _steps_per_epoch * args.num_epochs
_warmup_steps = _total_steps // 10
ds_config["scheduler"]["params"]["total_num_steps"] = _total_steps
ds_config["scheduler"]["params"]["warmup_num_steps"] = _warmup_steps
print(f"[SDPO] Auto LR schedule: {len(traindataset)} samples, {_n_gpus} GPUs, "
      f"effective_bs={_effective_bs}, {_steps_per_epoch} steps/epoch, "
      f"total={_total_steps}, warmup={_warmup_steps}")

draft_config_path = os.path.join(draftpath_resolved, "config.json")
config = EConfig.from_pretrained(
    draft_config_path if os.path.exists(draft_config_path) else train_config["config_path"]
)
config.gradient_checkpointing = train_config["gradient_checkpointing"]

model = SDPOModel(config, ds_config, model_ns, path=basepath_resolved,
                  load_emb=True, load_head=True)
load_draft_weights(model, draftpath_resolved)

# cache.pt holds d2t/t2d for this checkpoint; run extract_cache.py to generate it.
draft_cache = os.path.join(draftpath_resolved, "cache.pt")
if os.path.exists(draft_cache):
    shutil.copy(draft_cache, "cache.pt")
    print(f"[SDPO] Using draft vocab cache from {draft_cache}")
model.scandata(args.trainpath, basepath_resolved)

model_engine, optimizer, _, _ = deepspeed.initialize(
    args=args,
    model=model,
    model_parameters=model.parameters(),
)

global_rank = deepspeed.comm.get_rank()
rank = deepspeed.comm.get_local_rank()
world_size = deepspeed.comm.get_world_size()

model_engine.module.length = train_config["gamma"]

# Ref model created after deepspeed.initialize so ZeRO doesn't wrap it.
ref_model = SDPOModel(config, ds_config, model_ns, path=basepath_resolved,
                      load_emb=True, load_head=True)
load_draft_weights(ref_model, draftpath_resolved)
ref_model = ref_model.cuda(rank)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad_(False)
# object.__setattr__ bypasses nn.Module so DeepSpeed doesn't register ref_model params.
object.__setattr__(model_engine.module, 'ref_model', ref_model)

if global_rank == 0:
    wandb.init(project="sdpo_eagle3", config={
        **train_config,
        "basepath": args.basepath,
        "draftpath": args.draftpath,
        "world_size": world_size,
    })

os.makedirs(args.savedir, exist_ok=True)

train_sampler = DistributedSampler(traindataset, num_replicas=world_size,
                                   rank=global_rank, shuffle=True)
test_sampler = DistributedSampler(testdataset, num_replicas=world_size,
                                  rank=global_rank, shuffle=False)
train_loader = DataLoader(traindataset, batch_size=train_config["bs"],
                          sampler=train_sampler, num_workers=train_config["num_workers"],
                          pin_memory=True, collate_fn=DataCollator())
test_loader = DataLoader(testdataset, batch_size=train_config["bs"],
                         sampler=test_sampler, num_workers=train_config["num_workers"],
                         pin_memory=True, collate_fn=DataCollator())

ckpt_path, start_epoch = find_latest_checkpoint(args.savedir)
if ckpt_path:
    print(f"[SDPO] Resuming from {ckpt_path}")
    model_engine.load_checkpoint(ckpt_path)

global_step = 0
for epoch in range(start_epoch, train_config["num_epochs"]):
    train_sampler.set_epoch(epoch + 1)
    print(f"[SDPO] Epoch {epoch}")
    model.train()

    epoch_sdpo_loss = []
    epoch_kl_loss = []
    epoch_S = []
    epoch_tau = []
    epoch_step_acc = [[] for _ in range(train_config["gamma"])]

    for data in tqdm(train_loader, desc=f"Epoch {epoch} train"):
        model.zero_grad()
        total_loss, sdpo_loss, kl_loss, metrics = model_engine(
            input_ids=data["input_ids"].to(rank),
            attention_mask=data["attention_mask"].to(rank),
            loss_mask=data["loss_mask"].to(rank),
            G=train_config["num_rollouts"],
            eagle_coef=train_config["eagle_coef"],
            kl_coef=train_config["kl_coef"],
            mode=train_config["mode"],
            baseline=train_config["baseline"],
        )
        if metrics["num_valid"] == 0:
            continue

        model_engine.backward(total_loss)
        grad_norm = model_engine.get_global_grad_norm()
        model_engine.step()
        global_step += 1

        epoch_sdpo_loss.append(metrics["sdpo_loss"])
        epoch_kl_loss.append(metrics["kl_loss"])
        epoch_S.append(metrics["mean_S"])
        epoch_tau.append(metrics["mean_tau"])
        for k in range(train_config["gamma"]):
            epoch_step_acc[k].append(metrics["step_acc"][k])

        if global_rank == 0:
            log = {
                "train/total_loss": total_loss.item(),
                "train/sdpo_loss": metrics["sdpo_loss"],
                "train/eagle_loss": metrics["eagle_loss"],
                "train/kl_loss": metrics["kl_loss"],
                "train/mean_S": metrics["mean_S"],
                "train/mean_tau": metrics["mean_tau"],
                "train/num_valid": metrics["num_valid"],
                "train/adv_degenerate_frac": metrics["adv_degenerate_frac"],
                "train/grad_norm": grad_norm if grad_norm is not None else float("nan"),
                "train/lr": model_engine.get_lr()[0],
                "epoch": epoch,
            }
            for k in range(train_config["gamma"]):
                log[f"train/step{k}_acc"] = metrics["step_acc"][k]
            wandb.log(log, step=global_step)

    epoch_log = {"epoch": epoch}
    for name, vals in [("sdpo_loss", epoch_sdpo_loss), ("kl_loss", epoch_kl_loss),
                       ("mean_S", epoch_S), ("mean_tau", epoch_tau)]:
        v = torch.tensor(vals, dtype=torch.float32).cuda().mean()
        deepspeed.comm.all_reduce(v, op=deepspeed.comm.ReduceOp.AVG)
        if global_rank == 0:
            epoch_log[f"train/epoch_{name}"] = v.item()
            print(f"  epoch {epoch} | {name}: {v.item():.4f}")
    for k in range(train_config["gamma"]):
        v = torch.tensor(epoch_step_acc[k], dtype=torch.float32).cuda().mean()
        deepspeed.comm.all_reduce(v, op=deepspeed.comm.ReduceOp.AVG)
        if global_rank == 0:
            epoch_log[f"train/epoch_step{k}_acc"] = v.item()
    if global_rank == 0:
        wandb.log(epoch_log, step=global_step)

    model.eval()
    eval_S, eval_tau, eval_sdpo = [], [], []
    eval_step_acc = [[] for _ in range(train_config["gamma"])]

    for data in tqdm(test_loader, desc=f"Epoch {epoch} eval"):
        with torch.no_grad():
            _, _, _, metrics = model_engine(
                input_ids=data["input_ids"].to(rank),
                attention_mask=data["attention_mask"].to(rank),
                loss_mask=data["loss_mask"].to(rank),
                G=train_config["num_rollouts"],
                eagle_coef=train_config["eagle_coef"],
                kl_coef=train_config["kl_coef"],
                mode=train_config["mode"],
                baseline=train_config["baseline"],
            )
        if metrics["num_valid"] > 0:
            eval_S.append(metrics["mean_S"])
            eval_tau.append(metrics["mean_tau"])
            eval_sdpo.append(metrics["sdpo_loss"])
            for k in range(train_config["gamma"]):
                eval_step_acc[k].append(metrics["step_acc"][k])

    eval_log = {"epoch": epoch}
    for name, vals in [("sdpo_loss", eval_sdpo), ("mean_S", eval_S), ("mean_tau", eval_tau)]:
        v = torch.tensor(vals, dtype=torch.float32).cuda().mean()
        deepspeed.comm.all_reduce(v, op=deepspeed.comm.ReduceOp.AVG)
        if global_rank == 0:
            eval_log[f"eval/epoch_{name}"] = v.item()
            print(f"  epoch {epoch} | eval {name}: {v.item():.4f}")
    for k in range(train_config["gamma"]):
        v = torch.tensor(eval_step_acc[k], dtype=torch.float32).cuda().mean()
        deepspeed.comm.all_reduce(v, op=deepspeed.comm.ReduceOp.AVG)
        if global_rank == 0:
            eval_log[f"eval/epoch_step{k}_acc"] = v.item()
    if global_rank == 0:
        wandb.log(eval_log, step=global_step)

    deepspeed.comm.barrier()

    torch.cuda.empty_cache()

    ckpt_dir = f"{args.savedir}/state_{epoch}"
    deepspeed.DeepSpeedEngine.save_checkpoint(model_engine, save_dir=ckpt_dir)
    model_engine.save_16bit_model(ckpt_dir, exclude_frozen_parameters=True)
    # Patch d2t/t2d back in: save_16bit_model only saves named_parameters(), not buffers.
    # Only rank 0 writes the file; other ranks must not touch it.
    if global_rank == 0:
        _sf_path = os.path.join(ckpt_dir, "model.safetensors")
        _bin_path = os.path.join(ckpt_dir, "pytorch_model.bin")
        if os.path.exists(_sf_path) and os.path.isfile(_sf_path):
            _sd = sf_load(_sf_path)
            _sd["d2t"] = model.d2t.cpu()
            _sd["t2d"] = model.t2d.cpu()
            sf_save(_sd, _sf_path)
        elif os.path.isfile(_bin_path):
            _sd = torch.load(_bin_path, map_location="cpu")
            _sd["d2t"] = model.d2t.cpu()
            _sd["t2d"] = model.t2d.cpu()
            torch.save(_sd, _bin_path)
        elif os.path.isdir(_bin_path):
            # Sharded format: patch d2t/t2d into last shard + update index
            _shards = sorted(f for f in os.listdir(_bin_path) if f.endswith(".bin"))
            _last = os.path.join(_bin_path, _shards[-1])
            _sd = torch.load(_last, map_location="cpu")
            _sd["d2t"] = model.d2t.cpu()
            _sd["t2d"] = model.t2d.cpu()
            torch.save(_sd, _last)
            _idx_path = os.path.join(_bin_path, "pytorch_model.bin.index.json")
            if os.path.exists(_idx_path):
                with open(_idx_path) as _f:
                    _idx = json.load(_f)
                _idx["weight_map"]["d2t"] = _shards[-1]
                _idx["weight_map"]["t2d"] = _shards[-1]
                with open(_idx_path, "w") as _f:
                    json.dump(_idx, _f, indent=2)
        else:
            raise FileNotFoundError(
                f"save_16bit_model produced neither model.safetensors nor pytorch_model.bin in {ckpt_dir}"
            )
