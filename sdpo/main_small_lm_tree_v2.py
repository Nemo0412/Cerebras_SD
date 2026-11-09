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
import shutil
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
parser.add_argument('--stop_after_epochs', type=int, default=0,
                    help='Stop after this many epochs; the LR schedule still '
                         'spans --num_epochs.')
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
                    choices=['budget', 'path_topk', 'node_marginal', 'soft_topk',
                             'sample_minpq', 'sample_anyk', 'sample_cover', 'wsoft',
                             'greedy_topk', 'dense_tree', 'top4_kl_eal'],
                    help='budget: KL minus E[L] on the node-budget tree. '
                         'path_topk: KL minus exact E[L] on the Top-K path union. '
                         'node_marginal: old node-weighted NLL, not used. '
                         'soft_topk: loss = -E[L] on the Top-K path union. '
                         'Forward α = p_target(drafted token). '
                         'Backward moves the draft toward the target. '
                         'sample_minpq: draft samples the tree, α = min(p, q), '
                         'loss = -sum over nodes of prod α along the path. '
                         'Logged accept length is one realized eval-rule draw. '
                         'Eval with --draft-mode sample --verify-mode ratio. '
                         'sample_anyk: same sampled tree, but scored with the '
                         'rule eval runs: each node accepted with min(1, p/q) '
                         'and a slot advances if ANY sibling is accepted. Adds '
                         'the score-function term for the tree sampling. '
                         'sample_cover: sampled tree, one target draw per slot. '
                         'Each slot is scored with 1 - sum_v p(v)(1-q(v))^k '
                         'using only that slot\'s child count, so a wide root '
                         'spreads and a deep node spikes. Eval with '
                         '--draft-mode sample --verify-mode exclusive. '
                         'wsoft: greedy top-k tree. Each candidate scores '
                         'p(v) * sigmoid((z(v) - kth other logit) / tau). '
                         'Candidates are the drafted children plus the target '
                         'top-m left out of the tree. E[L] is the simple-verify '
                         'survival recursion. Loss adds anchor_kl_coef * KL(p||q). '
                         'Eval with --draft-mode argmax --verify-mode simple. '
                         'greedy_topk: greedy top-k tree. Keep the --path_topk '
                         'paths with the largest product of draft node '
                         'probabilities. Each path is a chain EAL, '
                         'P_j = softmax(z_j / temperature)[argmax p_j], '
                         'E[L] = sum_d prod_{j<=d} P_j. Path ranking stays at '
                         'temperature 1. Loss is minus the weighted sum of those '
                         'chain EALs. --path_eal_weights is in descending '
                         'draft-probability order; empty weights are all 1. '
                         'The logged accept length is the unweighted max. '
                         'Eval with --draft-mode argmax --verify-mode greedy. '
                         'dense_tree: dense KL(p||q) over the sequence + '
                         'anchor_kl_coef * KL on the nodes of the --path_topk '
                         'rollout paths - eal_aux_coef * (teacher-forced tree '
                         'E[L] - path_eal_var_coef * window variance). Depth j '
                         'accepts iff the target mode is in the draft top-k_j, '
                         'scored by sigmoid((z(u) - k_j-th other logit) / '
                         '--greedy_tau). '
                         'top4_kl_eal: only the --path_topk rollout paths. '
                         'anchor_kl_coef * mean KL over the distributions that '
                         'drafted their nodes - eal_aux_coef * (mean chain EAL '
                         'of the paths - path_eal_var_coef * their variance). '
                         'With --dense_eal_start the anchors are drawn only '
                         'from positions where greedy tree decoding opens a '
                         'window.')
parser.add_argument('--greedy_tau', type=float, default=1.0,
                    help='Temperature of the soft top-k score in eal_mode=wsoft.')
parser.add_argument('--soft_target_m', type=int, default=4,
                    help='Target tokens added beside the drafted children in '
                         'eal_mode=wsoft. Tokens already in the draft top-k '
                         'are not added twice.')
parser.add_argument('--n_draws', type=int, default=1,
                    help='Independent tree draws per anchor. sample_anyk needs '
                         '>=2 for its score-function baseline.')
parser.add_argument('--eal_score_coef', type=float, default=0.1,
                    help='Weight of the score-function (sampling) term in '
                         'sample_anyk. 0 keeps only the value term.')
parser.add_argument('--n_anchors', type=int, default=1,
                    help='Trees per sequence per step. >1 raises the supervision '
                         'per sample and gives sample_anyk its baseline.')
parser.add_argument('--path_topk', type=int, default=16,
                    help='How many paths to keep. Used by path_topk, '
                         'node_marginal, soft_topk, and greedy_topk.')
parser.add_argument('--path_eal_weights', type=str, default='',
                    help='Comma-separated greedy_topk weights, best path first. '
                         'Example: 1,2,3,4 gives the highest draft-probability '
                         'path weight 1 and the fourth path weight 4. '
                         'Empty keeps an unweighted sum.')
parser.add_argument('--path_eal_mean', action='store_true',
                    help='greedy_topk without weights: average the kept paths\' '
                         'chain EALs instead of summing them.')
parser.add_argument('--path_eal_var_coef', type=float, default=0.0,
                    help='greedy_topk: add this times the variance of the kept '
                         'paths\' chain EALs to the loss, so all paths improve '
                         'together. 0 disables it.')
parser.add_argument('--dense_eal_start', action='store_true',
                    help='dense_tree: average E[L] only over the windows greedy '
                         'tree decoding opens with the current draft, instead '
                         'of a window at every position. top4_kl_eal: draw '
                         'anchors only from those window starts.')
parser.add_argument('--eal_start_split', action='store_true',
                    help='top4_kl_eal with --dense_eal_start: keep n_anchors '
                         'uniformly drawn trees for the KL and add n_anchors '
                         'trees at decoding window starts that carry only the EAL.')
parser.add_argument('--top4path_starts', action='store_true',
                    help='--dense_eal_start: find window starts with the '
                         'path_topk-path verification rule instead of the '
                         'full top_k_per_depth tree.')
parser.add_argument('--exact_starts', action='store_true',
                    help='--dense_eal_start: find window starts by running greedy '
                         'path_topk-path verification with the draft tree at every '
                         'supervised position.')
parser.add_argument('--exact_starts_chunk', type=int, default=64,
                    help='Positions expanded together by --exact_starts.')
parser.add_argument('--exact_attn_backend', choices=['naive', 'flashinfer'], default='flashinfer',
                    help='--exact_starts tree expansion. flashinfer: paged attention on one '
                         'shared prefix layout per step (sdpo/tree_expand_fi.py), peak memory '
                         'flat in --exact_starts_chunk. naive: per-start padded prefix copies '
                         'and SDPA.')
parser.add_argument('--exact_batching', choices=['per_round', 'continuous'], default='continuous',
                    help='--exact_starts walk. continuous: fill every expansion round up to a '
                         'multiple of --exact_starts_chunk with the positions most likely to be '
                         'on the walk, so far fewer rounds are needed. per_round: expand only '
                         'what each round needs. Same starts.')
parser.add_argument('--exact_vocab_topk', choices=['fused', 'torch'], default='fused',
                    help='--exact_attn_backend flashinfer: top-k of the draft log-probs at each '
                         'tree depth. fused: one Triton pass over the bf16 logits '
                         '(sdpo/vocab_topk.py). torch: fp32 log_softmax, then topk. Same scores '
                         'up to fp32 rounding; fused breaks ties towards the lower token id.')
parser.add_argument('--top4_start_eal', action='store_true',
                    help='dense_tree + --exact_starts: E[L] at every window start is '
                         'the mean chain EAL of the path_topk paths the verification '
                         'keeps there.')
parser.add_argument('--exact_segment_frac', type=float, default=0.0,
                    help='--exact_starts: expand only a random contiguous segment of '
                         'this fraction of each sequence (plus --exact_burn positions '
                         'before it); E[L] is trained inside the segment.')
parser.add_argument('--exact_random_frac', type=float, default=0.0,
                    help='--exact_starts --top4_start_eal: train the unweighted top-4 '
                         'mean-path E[L] on this fraction of supervised positions, '
                         'drawn uniformly, instead of the window starts.')
parser.add_argument('--exact_burn', type=int, default=32,
                    help='Burn-in positions walked before the --exact_segment_frac '
                         'segment.')
parser.add_argument('--prob_starts', action='store_true',
                    help='--top4_start_eal: weight the E[L] of every position by the '
                         'probability a window starts there (renewal recursion over '
                         'the greedy tree acceptance surrogate) instead of the exact '
                         'starts.')
parser.add_argument('--dense_kl', action='store_true',
                    help='top4_kl_eal: add the teacher-forced KL(p||q) over every '
                         'supervised position of the sequence (the chain KL).')
parser.add_argument('--eal_gradnorm_ratio', type=float, default=0.0,
                    help='top4_kl_eal: replace eal_aux_coef by a coefficient that '
                         'makes the EAL gradient this many times as long as the KL '
                         'gradient (norms measured every --eal_gradnorm_every '
                         'micro-batches, EMA). 0 disables it.')
parser.add_argument('--eal_gradnorm_every', type=int, default=10)
parser.add_argument('--save_every_steps', type=int, default=1000,
                    help='Save a resumable mid-epoch checkpoint every N micro-batches '
                         'into savedir/mid. Only the newest one is kept, without the '
                         'frozen target weights. Rounded up to a multiple of '
                         'gradient_accumulation_steps. 0 saves only at epoch end.')
parser.add_argument('--local_rank', type=int, default=-1)
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

top_k_list = [int(x) for x in args.top_k_per_depth.split(',') if x.strip()]
assert len(top_k_list) >= 1, "top_k_per_depth must be non-empty"
path_eal_weights = [float(x) for x in args.path_eal_weights.split(',') if x.strip()]

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
        m = re.match(r"state_(\d+)$", sub)
        # DeepSpeed writes `latest` last, so a state dir without it is a partial save.
        if m and os.path.isfile(os.path.join(directory, sub, "latest")):
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
save_every = args.save_every_steps
if save_every > 0:
    save_every = math.ceil(save_every / grad_accum) * grad_accum

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
model.greedy_tau = args.greedy_tau
model.soft_target_m = args.soft_target_m
model.dense_eal_start = args.dense_eal_start
model.dense_kl = args.dense_kl
model.eal_start_split = args.eal_start_split
model.top4path_starts = args.top4path_starts
model.exact_starts = args.exact_starts
model.exact_starts_chunk = args.exact_starts_chunk
model.exact_attn_backend = args.exact_attn_backend
model.exact_batching = args.exact_batching
model.exact_vocab_topk = args.exact_vocab_topk
model.top4_start_eal = args.top4_start_eal
model.prob_starts = args.prob_starts
model.exact_segment_frac = args.exact_segment_frac
model.exact_burn = args.exact_burn
model.exact_random_frac = args.exact_random_frac
model.eal_gn_ratio = args.eal_gradnorm_ratio
model.eal_gn_every = args.eal_gradnorm_every

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

class _SkipSampler(torch.utils.data.Sampler):
    """DistributedSampler that drops the first `skip_batches` micro-batches."""

    def __init__(self, base, batch_size):
        self.base = base
        self.batch_size = batch_size
        self.skip_batches = 0

    def set_epoch(self, epoch):
        self.base.set_epoch(epoch)

    def __iter__(self):
        return iter(list(self.base)[self.skip_batches * self.batch_size:])

    def __len__(self):
        return max(len(self.base) - self.skip_batches * self.batch_size, 0)


os.makedirs(args.savedir, exist_ok=True)
train_sampler = _SkipSampler(
    DistributedSampler(traindataset, num_replicas=world_size,
                       rank=global_rank, shuffle=True),
    micro_bs)
test_sampler = DistributedSampler(testdataset, num_replicas=world_size,
                                  rank=global_rank, shuffle=False)
train_loader = DataLoader(traindataset, batch_size=micro_bs,
                          sampler=train_sampler, num_workers=0,
                          pin_memory=True, collate_fn=DataCollator())
test_loader = DataLoader(testdataset, batch_size=micro_bs,
                         sampler=test_sampler, num_workers=0,
                         pin_memory=True, collate_fn=DataCollator())

mid_dir = os.path.join(args.savedir, "mid")
mid_meta = os.path.join(mid_dir, "progress.json")
_SUM_KEYS = ("sum_total", "sum_kl", "sum_eal", "sum_eal_loss", "sum_gn",
             "sum_anchors", "n_batches")

ckpt_path, start_epoch = find_latest_checkpoint(args.savedir)
global_step = start_epoch * iters_per_epoch
resume_skip = 0
resume_sums = None
mid_progress = None
if os.path.isfile(mid_meta):
    with open(mid_meta) as f:
        mid_progress = json.load(f)
    if mid_progress["epoch"] < start_epoch:
        mid_progress = None
if mid_progress is not None:
    print(f"[SmallLMTreeV2] Resuming mid-epoch from {mid_dir}/{mid_progress['tag']}")
    model_engine.load_checkpoint(mid_dir, tag=mid_progress["tag"],
                                 load_module_strict=False)
    start_epoch = mid_progress["epoch"]
    resume_skip = mid_progress["batches_done"]
    global_step = mid_progress["global_step"]
    resume_sums = mid_progress["sums"]
elif ckpt_path:
    print(f"[SmallLMTreeV2] Resuming from {ckpt_path}")
    model_engine.load_checkpoint(ckpt_path, load_module_strict=False)

step_log_path = os.path.join(args.savedir, "train_steps.jsonl")
epoch_log_path = os.path.join(args.savedir, "epoch_metrics.jsonl")
eval_sbatch = os.environ.get("EVAL_SBATCH", "")

if global_rank == 0 and os.path.isfile(step_log_path):
    # Steps logged after the checkpoint we resume from are replayed, so drop them.
    with open(step_log_path) as f:
        kept = [ln for ln in f
                if ln.strip() and json.loads(ln).get("step", 0) <= global_step]
    with open(step_log_path, "w") as f:
        f.writelines(kept)


def _save_mid(epoch, batches_done):
    tag = f"e{epoch}_b{batches_done}"
    model_engine.save_checkpoint(mid_dir, tag=tag, exclude_frozen_parameters=True)
    deepspeed.comm.barrier()
    if global_rank == 0:
        prev_tag = None
        if os.path.isfile(mid_meta):
            with open(mid_meta) as f:
                prev_tag = json.load(f).get("tag")
        g = globals()
        progress = {
            "tag": tag,
            "epoch": epoch,
            "batches_done": batches_done,
            "global_step": global_step,
            "sums": {k: g[k] for k in _SUM_KEYS},
        }
        tmp = mid_meta + ".tmp"
        with open(tmp, "w") as f:
            json.dump(progress, f)
        os.replace(tmp, mid_meta)
        if prev_tag and prev_tag != tag:
            shutil.rmtree(os.path.join(mid_dir, prev_tag), ignore_errors=True)
        print(f"  [mid-epoch save] {mid_dir}/{tag}  step={global_step}", flush=True)
    deepspeed.comm.barrier()


def _append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


def _submit_epoch_eval(epoch):
    """Queue verify_tree_step eval only when EVAL_SBATCH is set."""
    if not eval_sbatch:
        return None
    if not os.path.isfile(eval_sbatch):
        print(f"  epoch {epoch} | eval sbatch missing: {eval_sbatch}")
        return None
    partition = os.environ.get("EVAL_PARTITION") or os.environ.get("SLURM_JOB_PARTITION", "")
    account = os.environ.get("EVAL_ACCOUNT") or os.environ.get("SLURM_JOB_ACCOUNT", "")
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


for epoch in range(start_epoch, args.num_epochs):
    train_sampler.set_epoch(epoch + 1)
    train_sampler.skip_batches = resume_skip
    model.train()
    print(f"[SmallLMTreeV2] Epoch {epoch}"
          + (f" (resuming after batch {resume_skip})" if resume_skip else ""))

    sum_total = 0.0
    sum_kl = 0.0
    sum_eal = 0.0
    sum_eal_loss = 0.0
    sum_gn = 0.0
    sum_anchors = 0
    n_batches = 0
    if resume_sums is not None:
        globals().update(resume_sums)
        resume_sums = None

    for batch_idx, data in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch} train",
                 initial=resume_skip, total=iters_per_epoch),
            start=resume_skip):
        if (save_every > 0 and batch_idx > resume_skip
                and batch_idx % save_every == 0):
            _save_mid(epoch, batch_idx)
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
            path_eal_weights=path_eal_weights,
            path_eal_mean=args.path_eal_mean,
            path_eal_var_coef=args.path_eal_var_coef,
            eal_score_coef=args.eal_score_coef,
            n_anchors=args.n_anchors,
            n_draws=args.n_draws,
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
                "dense_loss": metrics["dense_loss"],
                "analytic_accepted_length": metrics["eal_mean"],
                "kl": metrics["kl_loss"],
                "grad_norm": gn,
                "lr": lr,
                "num_anchors": metrics["num_anchors"],
                "tree_nodes": metrics["tree_nodes"],
            }
            if "path_eal_std" in metrics:
                row["path_eal_std"] = metrics["path_eal_std"]
            for key in ("dense_eal", "dense_var", "dense_hard_al",
                        "eal_gn_coef", "gnorm_kl", "gnorm_eal", "windows_per_seq",
                        "exact_al_at_starts", "exact_al_all_pos",
                        "prob_windows_per_seq", "exact_expanded_frac",
                        "exact_rounds"):
                if key in metrics:
                    row[key] = metrics[key]
            for key in metrics:
                if key.startswith("path") and key.endswith(("_eal", "_al", "_var")):
                    row[key] = metrics[key]
            _append_jsonl(step_log_path, row)
            print(f"  step {global_step} epoch {epoch} | "
                  f"loss={row['loss']:.4f}  eal_loss={row['eal_loss']:.4f}  "
                  f"dense_loss={metrics['dense_loss']:.4f}  "
                  f"analytic_accepted_length={row['analytic_accepted_length']:.4f}  "
                  f"kl={row['kl']:.4f}  grad_norm={row['grad_norm']:.4f}",
                  flush=True)
            wandb.log({
                "train/total_loss": metrics["total_loss"],
                "train/kl_loss": metrics["kl_loss"],
                "train/aux_loss": metrics["aux_loss"],
                "train/eal_loss": metrics["eal_loss"],
                "train/dense_loss": metrics["dense_loss"],
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
    resume_skip = 0
    train_sampler.skip_batches = 0

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
                path_eal_weights=path_eal_weights,
                path_eal_mean=args.path_eal_mean,
                path_eal_var_coef=args.path_eal_var_coef,
                eal_score_coef=args.eal_score_coef,
                n_anchors=args.n_anchors,
                n_draws=args.n_draws,
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

    # The frozen 8B target is ~30GB per save; only the draft and optimizer
    # are needed to resume, and only the newest epoch state is kept.
    ckpt_dir = f"{args.savedir}/state_{epoch}"
    model_engine.save_checkpoint(ckpt_dir, exclude_frozen_parameters=True)
    if global_rank == 0:
        hf_dir = f"{args.savedir}/hf_epoch_{epoch}"
        model_engine.module.draft_model.save_pretrained(hf_dir)
        tokenizer.save_pretrained(hf_dir)
        print(f"  Saved DeepSpeed checkpoint to {ckpt_dir}")
        print(f"  Saved HF draft to {hf_dir}")
        shutil.rmtree(mid_dir, ignore_errors=True)
        for old in range(epoch):
            shutil.rmtree(f"{args.savedir}/state_{old}", ignore_errors=True)
        eval_job = _submit_epoch_eval(epoch)
        if eval_job is not None:
            _append_jsonl(epoch_log_path, {
                "epoch": epoch,
                "accept_length_eval_job": eval_job,
                "draft": hf_dir,
            })
    deepspeed.comm.barrier()
    if args.stop_after_epochs and epoch + 1 >= args.stop_after_epochs:
        print(f"[SmallLMTreeV2] Stopping after epoch {epoch} (--stop_after_epochs)")
        break
