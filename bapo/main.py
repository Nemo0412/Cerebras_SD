"""
BAPO Training Script — Boundary-Aware Accepted Prefix Optimization.

Supports two modes:
  --mode scratch   : random-init draft → Phase 1 distill → Phase 2 BAPO (→ Phase 3 PG)
  --mode finetune  : load pre-trained draft → Phase 2 BAPO (→ Phase 3 PG)

Phases are controlled by epoch counts:
  --distill_epochs N1   Phase 1 soft-target KL warm-start
  --bapo_epochs    N2   Phase 2 boundary-weighted rollout supervision
  --pg_epochs      N3   Phase 3 optional REINFORCE correction

Usage (finetune):
  deepspeed bapo/main.py \\
    --basepath  meta-llama/Llama-3.1-8B-Instruct \\
    --draftpath yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \\
    --trainpath sdpo/data/mixed_train_10K.jsonl \\
    --testpath  sdpo/data/mixed_val.jsonl \\
    --deepspeed_config bapo/bapo_config.json \\
    --mode finetune --bapo_epochs 3 \\
    --w_far 1.0 --w_near 2.0 --w_fail 6.0 --kl_coef 0.01 --gamma 7

Usage (from scratch):
  deepspeed bapo/main.py \\
    --basepath  meta-llama/Llama-3.1-8B-Instruct \\
    --trainpath sdpo/data/mixed_train_10K.jsonl \\
    --testpath  sdpo/data/mixed_val.jsonl \\
    --deepspeed_config bapo/bapo_config.json \\
    --mode scratch --distill_epochs 3 --bapo_epochs 2 --gamma 7
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
from bapo_model import BAPOModel

torch.backends.cuda.matmul.allow_tf32 = True

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--basepath', required=True,
                    help='Path or HF repo for the frozen target LLM')
parser.add_argument('--draftpath', default=None,
                    help='Path or HF repo for pre-trained EAGLE3 draft '
                         '(required for --mode finetune)')
parser.add_argument('--trainpath', required=True)
parser.add_argument('--testpath', required=True)
parser.add_argument('--savedir', default='bapo_checkpoints')

# Mode & phases
parser.add_argument('--mode', choices=['scratch', 'finetune'], default='finetune',
                    help='scratch = random-init + distill; finetune = load pre-trained')
parser.add_argument('--distill_epochs', type=int, default=0,
                    help='Epochs for Phase 1 (KL distillation warm start)')
parser.add_argument('--bapo_epochs', type=int, default=3,
                    help='Epochs for Phase 2 (boundary-weighted rollout)')
parser.add_argument('--pg_epochs', type=int, default=0,
                    help='Epochs for Phase 3 (REINFORCE correction)')

# BAPO hyper-parameters
parser.add_argument('--w_far', type=float, default=1.0,
                    help='CE weight for early accepted tokens')
parser.add_argument('--w_near', type=float, default=2.0,
                    help='CE weight for last accepted token')
parser.add_argument('--w_fail', type=float, default=6.0,
                    help='CE weight for boundary (first rejected) token')
parser.add_argument('--w_post', type=float, default=0.5,
                    help='CE weight for post-boundary tokens (mild distillation)')
parser.add_argument('--kl_coef', type=float, default=0.01,
                    help='KL regularization coefficient')
parser.add_argument('--pg_coef', type=float, default=0.1,
                    help='PG loss coefficient for Phase 3')
parser.add_argument('--sampling', choices=['greedy', 'sample'], default='greedy',
                    help='Rollout sampling mode for Phase 2')
parser.add_argument('--prod_coef', type=float, default=0.0,
                    help='Coefficient for product acceptance loss (0 = disabled)')
parser.add_argument('--prod_tau', type=float, default=1.0,
                    help='Temperature for soft acceptance indicator in product loss')

# Training
parser.add_argument('--gamma', type=int, default=7)
parser.add_argument('--max_len', type=int, default=2048)
parser.add_argument('--max_train_samples', type=int, default=None)
parser.add_argument('--config_path', type=str,
                    default=os.path.join(os.path.dirname(__file__),
                                         '..', 'sdpo', 'config.json'))
parser.add_argument('--local_rank', type=int, default=-1)
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

if args.mode == 'finetune' and args.draftpath is None:
    parser.error("--draftpath is required for --mode finetune")

total_epochs = args.distill_epochs + args.bapo_epochs + args.pg_epochs
if total_epochs == 0:
    parser.error("Total epochs must be > 0 (set at least one of "
                 "--distill_epochs, --bapo_epochs, --pg_epochs)")

# Adjust phases for scratch mode: require at least some distillation
if args.mode == 'scratch' and args.distill_epochs == 0:
    print("[BAPO] WARNING: scratch mode with 0 distill_epochs; "
          "the draft model will be randomly initialized with no warm-up.")

with open(args.deepspeed_config) as f:
    ds_config = json.load(f)

train_config = SimpleNamespace(
    bs=ds_config["train_micro_batch_size_per_gpu"],
    num_epochs=total_epochs,
    num_workers=0,
    max_len=args.max_len,
    config_path=args.config_path,
    gradient_checkpointing=True,
    eagle_coef=0,
    kl_coef=0,
    gamma=args.gamma,
    baseline=None,
)

# ---------------------------------------------------------------------------
# Data  (same preprocessing as sdpo/main.py)
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
SEP_ASST = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
SEP_USER = "<|eot_id|><|start_header_id|>user<|end_header_id|>"


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
            for sent in src:
                msgs.append({"role": roles[sent["from"]],
                              "content": sent["value"]})
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
                batch[key].append(torch.tensor(vals + [0] * pad,
                                               dtype=torch.long))
        return {k: torch.stack(v) for k, v in batch.items()}


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def resolve_hf_path(path, label):
    if os.path.isdir(path):
        return path
    print(f"[BAPO] Downloading {label}: {path}")
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

    d2t = state.pop("d2t", None)
    t2d = state.pop("t2d", None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys: {unexpected}")

    if d2t is not None and t2d is not None:
        model.register_buffer("d2t", d2t)
        model.register_buffer("t2d", t2d)
        print(f"[BAPO] Loaded draft + vocab mapping "
              f"(draft_vocab={len(d2t)}, target_vocab={len(t2d)})")
    else:
        print("[BAPO] WARNING: no d2t/t2d in checkpoint, will need scandata()")
    print(f"[BAPO] Loaded draft weights from {draftpath}")


def find_latest_checkpoint(directory):
    if not os.path.isdir(directory):
        return None, 0
    max_epoch = -1
    for sub in os.listdir(directory):
        m = re.match(r"state_(\d+)", sub)
        if m and os.path.exists(os.path.join(directory, sub, "zero_to_fp32.py")):
            max_epoch = max(max_epoch, int(m.group(1)))
    if max_epoch == -1:
        return None, 0
    return f"{directory}/state_{max_epoch}", max_epoch + 1


def patch_vocab_buffers(ckpt_dir, model):
    """Write d2t/t2d into the 16-bit saved checkpoint."""
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


def epoch_to_phase(epoch):
    """Map a global epoch index to (phase_name, epoch_within_phase)."""
    if epoch < args.distill_epochs:
        return 'distill', epoch
    elif epoch < args.distill_epochs + args.bapo_epochs:
        return 'bapo', epoch - args.distill_epochs
    else:
        return 'pg', epoch - args.distill_epochs - args.bapo_epochs


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------
basepath = resolve_hf_path(args.basepath, "base model")
draftpath = None
if args.draftpath:
    draftpath = resolve_hf_path(args.draftpath, "draft model")

tokenizer = AutoTokenizer.from_pretrained(basepath)
traindataset = build_dataset(tokenizer, args.trainpath, args.max_len,
                             max_samples=args.max_train_samples)
testdataset = build_dataset(tokenizer, args.testpath, args.max_len)

# LR schedule
n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
micro_bs = ds_config["train_micro_batch_size_per_gpu"]
grad_accum = ds_config["gradient_accumulation_steps"]
effective_bs = micro_bs * grad_accum * n_gpus
steps_per_epoch = math.ceil(len(traindataset) / effective_bs)
total_steps = steps_per_epoch * total_epochs
warmup_steps = total_steps // 10
ds_config["scheduler"]["params"]["total_num_steps"] = total_steps
ds_config["scheduler"]["params"]["warmup_num_steps"] = warmup_steps

# For moving-average baseline (Phase 3)
samples_per_gpu = math.ceil(len(traindataset) / n_gpus)
iters_per_epoch = math.ceil(samples_per_gpu / micro_bs)
total_iters = iters_per_epoch * total_epochs

print(f"[BAPO] mode={args.mode}, phases: "
      f"distill={args.distill_epochs}, bapo={args.bapo_epochs}, "
      f"pg={args.pg_epochs} epochs")
print(f"[BAPO] {len(traindataset)} samples, {n_gpus} GPUs, "
      f"eff_bs={effective_bs}, {steps_per_epoch} steps/epoch, "
      f"total={total_steps}, warmup={warmup_steps}")

# Model config — from draft checkpoint or default
cfg_path = args.config_path
if draftpath and os.path.exists(os.path.join(draftpath, "config.json")):
    cfg_path = os.path.join(draftpath, "config.json")
config = EConfig.from_pretrained(cfg_path)
config.gradient_checkpointing = True

model = BAPOModel(config, ds_config, train_config, path=basepath,
                  load_emb=True, load_head=True)

if args.mode == 'finetune':
    load_draft_weights(model, draftpath)
else:
    print("[BAPO] From-scratch mode: draft layers randomly initialized")

# Vocab mapping
if not hasattr(model, 'd2t') or model.d2t is None:
    if draftpath:
        draft_cache = os.path.join(draftpath, "cache.pt")
        if os.path.exists(draft_cache):
            shutil.copy(draft_cache, "cache.pt")
    model.scandata(args.trainpath, basepath)

# Write updated DS config & initialise engine
with open(args.deepspeed_config, 'w') as f:
    json.dump(ds_config, f, indent=2)
model_engine, optimizer, _, _ = deepspeed.initialize(
    args=args, model=model, model_parameters=model.parameters())

global_rank = deepspeed.comm.get_rank()
rank = deepspeed.comm.get_local_rank()
world_size = deepspeed.comm.get_world_size()
model_engine.module.length = args.gamma

if global_rank == 0:
    run_name = (f"bapo_{args.mode}_w{args.w_far}-{args.w_near}-{args.w_fail}"
                f"_kl{args.kl_coef}")
    wandb.init(entity="tonyteng66", project="bapo", name=run_name, config={
        "mode": args.mode,
        "distill_epochs": args.distill_epochs,
        "bapo_epochs": args.bapo_epochs,
        "pg_epochs": args.pg_epochs,
        "w_far": args.w_far,
        "w_near": args.w_near,
        "w_fail": args.w_fail,
        "w_post": args.w_post,
        "kl_coef": args.kl_coef,
        "pg_coef": args.pg_coef,
        "sampling": args.sampling,
        "prod_coef": args.prod_coef,
        "prod_tau": args.prod_tau,
        "gamma": args.gamma,
        "basepath": args.basepath,
        "draftpath": args.draftpath,
        "world_size": world_size,
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
    print(f"[BAPO] Resuming from {ckpt_path}")
    model_engine.load_checkpoint(ckpt_path)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
global_step = 0
pg_baseline = 0.0                               # moving-average for PG phase
pg_baseline_decay = 0.99

for epoch in range(start_epoch, total_epochs):
    phase, phase_epoch = epoch_to_phase(epoch)
    train_sampler.set_epoch(epoch + 1)
    model.train()
    print(f"[BAPO] Epoch {epoch}  phase={phase}  phase_epoch={phase_epoch}")

    epoch_metrics = {}
    epoch_step_acc = [[] for _ in range(args.gamma)]
    epoch_tau_hist = [0] * (args.gamma + 1)

    for data in tqdm(train_loader, desc=f"Epoch {epoch} [{phase}]"):
        model.zero_grad()

        # -- build phase-specific kwargs --
        fwd_kwargs = {"phase": phase}
        if phase == 'bapo':
            fwd_kwargs.update(
                w_far=args.w_far, w_near=args.w_near, w_fail=args.w_fail,
                w_post=args.w_post, kl_coef=args.kl_coef, sampling=args.sampling,
                prod_coef=args.prod_coef, prod_tau=args.prod_tau)
        elif phase == 'pg':
            fwd_kwargs.update(
                kl_coef=args.kl_coef, pg_coef=args.pg_coef,
                baseline=pg_baseline)

        total_loss, metrics = model_engine(
            input_ids=data["input_ids"].to(rank),
            attention_mask=data["attention_mask"].to(rank),
            loss_mask=data["loss_mask"].to(rank),
            **fwd_kwargs,
        )
        if metrics["num_valid"] == 0:
            continue

        model_engine.backward(total_loss)
        grad_norm = model_engine.get_global_grad_norm()
        model_engine.step()
        global_step += 1

        # Update PG baseline
        if phase == 'pg':
            pg_baseline = (pg_baseline_decay * pg_baseline
                           + (1 - pg_baseline_decay) * metrics["mean_tau"])

        # Accumulate epoch metrics
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and k not in ("num_valid", "baseline"):
                epoch_metrics.setdefault(k, []).append(v)
        for k in range(args.gamma):
            if k < len(metrics.get("step_acc", [])):
                epoch_step_acc[k].append(metrics["step_acc"][k])
        for v in range(args.gamma + 1):
            if v < len(metrics.get("tau_hist", [])):
                epoch_tau_hist[v] += metrics["tau_hist"][v]

        # Wandb train logging
        if global_rank == 0:
            log = {
                "train/total_loss": total_loss.item(),
                "train/phase": {"distill": 0, "bapo": 1, "pg": 2}[phase],
                "train/grad_norm": (grad_norm if grad_norm is not None
                                    else float("nan")),
                "train/lr": model_engine.get_lr()[0],
            }
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and k != "num_valid":
                    log[f"train/{k}"] = v
            for k in range(args.gamma):
                if k < len(metrics.get("step_acc", [])):
                    log[f"train/step{k}_acc"] = metrics["step_acc"][k]
            wandb.log(log, step=global_step)

    # -- Epoch summary -------------------------------------------------
    if global_rank == 0:
        for name, vals in epoch_metrics.items():
            avg = sum(vals) / len(vals) if vals else 0
            print(f"  epoch {epoch} | {name}: {avg:.4f}")
        total_tau = sum(epoch_tau_hist)
        if total_tau > 0:
            print(f"  tau dist: "
                  f"{[f'{h / total_tau:.3f}' for h in epoch_tau_hist]}")

    # -- Eval ----------------------------------------------------------
    model.eval()
    eval_metrics = {}
    eval_step_acc = [[] for _ in range(args.gamma)]

    for data in tqdm(test_loader, desc=f"Epoch {epoch} eval"):
        with torch.no_grad():
            eval_kwargs = {"phase": phase}
            if phase == 'bapo':
                eval_kwargs.update(
                    w_far=args.w_far, w_near=args.w_near, w_fail=args.w_fail,
                    w_post=args.w_post, kl_coef=args.kl_coef, sampling='greedy',
                    prod_coef=args.prod_coef, prod_tau=args.prod_tau)
            elif phase == 'pg':
                eval_kwargs.update(
                    kl_coef=args.kl_coef, pg_coef=args.pg_coef,
                    baseline=pg_baseline)

            _, metrics = model_engine(
                input_ids=data["input_ids"].to(rank),
                attention_mask=data["attention_mask"].to(rank),
                loss_mask=data["loss_mask"].to(rank),
                **eval_kwargs,
            )
        if metrics["num_valid"] > 0:
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and k not in ("num_valid", "baseline"):
                    eval_metrics.setdefault(k, []).append(v)
            for k in range(args.gamma):
                if k < len(metrics.get("step_acc", [])):
                    eval_step_acc[k].append(metrics["step_acc"][k])

    if global_rank == 0:
        eval_log = {"epoch": epoch, "eval/phase": phase}
        for name, vals in eval_metrics.items():
            avg = sum(vals) / len(vals) if vals else 0
            eval_log[f"eval/{name}"] = avg
            print(f"  epoch {epoch} | eval {name}: {avg:.4f}")
        for k in range(args.gamma):
            if eval_step_acc[k]:
                eval_log[f"eval/step{k}_acc"] = (
                    sum(eval_step_acc[k]) / len(eval_step_acc[k]))
        wandb.log(eval_log, step=global_step)

    deepspeed.comm.barrier()
    torch.cuda.empty_cache()

    # -- Checkpoint ----------------------------------------------------
    ckpt_dir = f"{args.savedir}/state_{epoch}"
    deepspeed.DeepSpeedEngine.save_checkpoint(model_engine, save_dir=ckpt_dir)
    model_engine.save_16bit_model(ckpt_dir, exclude_frozen_parameters=True)
    if global_rank == 0:
        patch_vocab_buffers(ckpt_dir, model)
        # Copy draft config so EaModel.from_pretrained can load this checkpoint
        import shutil
        src_cfg = args.config_path
        dst_cfg = os.path.join(ckpt_dir, "config.json")
        if not os.path.exists(dst_cfg):
            shutil.copy(src_cfg, dst_cfg)

    # -- Phase transition log ------------------------------------------
    next_phase, _ = epoch_to_phase(epoch + 1) if epoch + 1 < total_epochs else (None, 0)
    if next_phase and next_phase != phase and global_rank == 0:
        print(f"[BAPO] === Phase transition: {phase} → {next_phase} ===")
