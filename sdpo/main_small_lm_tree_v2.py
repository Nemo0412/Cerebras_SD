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
import collections
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
                             'sample_minpq', 'sample_rsd', 'sample_rsd_offline',
                             'sample_anyk',
                             'sample_cover', 'wsoft',
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
                         'sample_rsd: online Sampled tree EAL. Draft samples '
                         'the full tree, keeps --path_topk draft-score paths, '
                         'target verifies ONLY that pruned union. '
                         'Train loss = dense residual TV '
                         '(β=Σ_v min(u_v,q_v) under RSD residual; Chain '
                         'al_tv analogue; no RL). Logged metric = analytic '
                         'RSD E[L]. loss = anchor_kl_coef*KL + '
                         'eal_aux_coef*(-dense_tv). '
                         'Eval with --draft-mode sample --verify-mode rsd. '
                         'sample_rsd_offline: draft resamples a tree and keeps '
                         '--path_topk paths; a path is used only if its token '
                         'sequence matches a cached target-verified path from '
                         '--srsd_cache, else discarded. Training never runs '
                         'the target. '
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
parser.add_argument('--n_draws', type=int, default=2,
                    help='Independent tree draws per anchor. sample_anyk '
                         'auto-bumps to >=2 when eal_score_coef!=0 so the '
                         'within-anchor score-function baseline is live. '
                         'sample_rsd ignores this for the loss (dense_tv).')
parser.add_argument('--eal_score_coef', type=float, default=0.1,
                    help='Weight of the score-function (sampling) term in '
                         'sample_anyk only. Needs --n_draws>=2 for a non-zero '
                         'within-anchor baseline (enforced in forward). '
                         '0 keeps only the value term. Unused by sample_rsd.')
parser.add_argument('--n_anchors', type=int, default=1,
                    help='Trees per sequence per step. >1 raises the supervision '
                         'per sample and gives sample_anyk its baseline.')
parser.add_argument('--anchor_chunk', type=int, default=0,
                    help='sample_rsd only: roll out at most this many anchors at '
                         'once (activation-checkpointed). 0 disables chunking. '
                         'Use e.g. --n_anchors 16 --anchor_chunk 8 to raise '
                         'coverage without the peak memory of 16 simultaneous trees.')
parser.add_argument('--path_topk', type=int, default=16,
                    help='How many paths to keep. Used by path_topk, '
                         'node_marginal, soft_topk, greedy_topk, sample_rsd, '
                         'and sample_rsd_offline (use 4 for Sampled tree EAL).')
parser.add_argument('--srsd_cache', type=str, default='',
                    help='Directory of precomputed target-verified path bank '
                         'for sample_rsd_offline (from build_srsd_cache.py). '
                         'Train matches draft paths against this bank.')
parser.add_argument('--srsd_tgt_cache', action='store_true', default=True,
                    help='sample_rsd: LRU-cache target top-m logits along draft '
                         'paths; skip target tree decode on a full path hit.')
parser.add_argument('--no_srsd_tgt_cache', action='store_false', dest='srsd_tgt_cache',
                    help='Disable the online target path cache for sample_rsd.')
parser.add_argument('--srsd_tgt_cache_m', type=int, default=256,
                    help='Top-m target logits stored per path node in the online cache.')
parser.add_argument('--srsd_tgt_cache_max_paths', type=int, default=4096,
                    help='LRU capacity (number of paths) for the online target cache.')
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
parser.add_argument('--seed', type=int, default=0,
                    help='Data-order seed; > 0 also seeds torch/numpy/random.')
parser.add_argument('--dpace_alpha', type=float, default=-1.0,
                    help='dense_tree: replace the dense KL by the D-PACE weighted CE '
                         '(blocks of gamma positions at every supervised position) with '
                         'this smoothing alpha; < 0 disables.')
parser.add_argument('--gto_coef', type=float, default=0.0,
                    help='dense_tree: add gto_coef * the GTO group-tree PPO loss against '
                         'a frozen copy of the initial draft.')
parser.add_argument('--eal_path_agg', choices=['mean', 'max'], default='mean',
                    help='--top4_start_eal: score each window by the mean of its kept '
                         'paths\' E[L] or by the longest one.')
parser.add_argument('--eal_alpha', type=float, default=0.0,
                    help='--top4_start_eal: smooth each path factor q_j to '
                         '(1 - alpha) q_j + alpha, as in D-PACE.')
parser.add_argument('--boundary_tau', type=float, default=0.0,
                    help='--top4_start_eal: replace the path E[L] by softplus(-margin/tau) '
                         'on the positions each window reaches, margin = z(u) minus the '
                         'k_j-th largest other draft logit.')
parser.add_argument('--eal_all_starts', action='store_true',
                    help='--top4_start_eal: train the E[L] of a window at every '
                         'supervised position with equal weight and no truncation at '
                         'the first mismatch (uniform baseline for the weighted runs).')
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
parser.add_argument('--snapshot_frac', type=float, default=0.0,
                    help='Also save the draft as savedir/hf_e{epoch}_p{percent} every '
                         'this fraction of an epoch (kept, for training-curve evals).')
parser.add_argument('--early_stop_al_window', type=int, default=0,
                    help='If >0, track a rolling mean of analytic accept length over '
                         'this many recent micro-batches (checked at mid-saves).')
parser.add_argument('--early_stop_al_delta', type=float, default=0.02,
                    help='Stop when rolling AL falls this far below the best seen '
                         '(requires --early_stop_al_window and mid-saves).')
parser.add_argument('--early_stop_al_patience', type=int, default=2,
                    help='Number of consecutive mid-save checks with AL decline '
                         'before stopping. 0 disables early stop.')
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


class SRSDCacheDataset(torch.utils.data.Dataset):
    """Offline Sampled-tree EAL cache (lazy per-shard load).

    The on-disk bank is ~180GB of full-vocab target logits. Loading every
    shard into RAM OOMs a 200GB node; we only keep an LRU of a few shards
    resident and index (shard_id, local_id) up front.
    """

    def __init__(self, cache_dir, keep_shards=2):
        import glob
        from collections import OrderedDict
        self.shard_paths = sorted(glob.glob(os.path.join(cache_dir, "shard_*.pt")))
        if not self.shard_paths:
            raise RuntimeError(f"no shard_*.pt under {cache_dir}")
        self.keep_shards = max(int(keep_shards), 1)
        self._lru = OrderedDict()  # path -> list[record]
        self.index = []  # (shard_i, local_i)
        # Peek shard lengths one at a time (peak ≈ one shard, ~2GB).
        for si, path in enumerate(self.shard_paths):
            recs = torch.load(path, map_location="cpu")
            if si == 0:
                if "path_tokens" not in recs[0] or "path_nodes" not in recs[0]:
                    raise RuntimeError(
                        f"{cache_dir} is an old teacher-force cache; rebuild "
                        "with build_srsd_cache.py (needs path_tokens/path_nodes)")
            for li in range(len(recs)):
                self.index.append((si, li))
            del recs
        print(f"[SmallLMTreeV2] offline cache index {len(self.index)} recs "
              f"in {len(self.shard_paths)} shards "
              f"(lazy, keep_shards={self.keep_shards})")

    def __len__(self):
        return len(self.index)

    def _shard(self, si):
        path = self.shard_paths[si]
        if path in self._lru:
            self._lru.move_to_end(path)
            return self._lru[path]
        recs = torch.load(path, map_location="cpu")
        self._lru[path] = recs
        while len(self._lru) > self.keep_shards:
            self._lru.popitem(last=False)
        return recs

    def __getitem__(self, idx):
        si, li = self.index[idx]
        return self._shard(si)[li]


class SRSDCacheCollator:
    def __call__(self, features):
        max_len = max(int(f["input_ids"].numel()) for f in features)
        max_m = max(int(f["tokens"].numel()) for f in features)
        max_p = max(int(f["path_tokens"].size(0)) for f in features)
        gamma = int(features[0]["path_tokens"].size(1))
        B = len(features)
        V = int(features[0]["tgt_node"].size(-1))
        batch = {
            "input_ids": torch.zeros(B, max_len, dtype=torch.long),
            "attention_mask": torch.zeros(B, max_len, dtype=torch.long),
            "loss_mask": torch.zeros(B, max_len, dtype=torch.long),
            "srsd_tokens": torch.zeros(B, max_m, dtype=torch.long),
            "srsd_parents": torch.full((B, max_m), -1, dtype=torch.long),
            "srsd_depths": torch.zeros(B, max_m, dtype=torch.long),
            "srsd_tgt_node": torch.zeros(B, max_m, V, dtype=torch.bfloat16),
            "srsd_valid": torch.zeros(B, max_m, dtype=torch.bool),
            "srsd_path_tokens": torch.full((B, max_p, gamma), -1, dtype=torch.long),
            "srsd_path_nodes": torch.full((B, max_p, gamma), -1, dtype=torch.long),
            "srsd_anchor_pos": torch.zeros(B, dtype=torch.long),
        }
        for i, f in enumerate(features):
            L = int(f["input_ids"].numel())
            M = int(f["tokens"].numel())
            P = int(f["path_tokens"].size(0))
            batch["input_ids"][i, :L] = f["input_ids"].long()
            batch["attention_mask"][i, :L] = f["attention_mask"].long()
            batch["loss_mask"][i, :L] = f["loss_mask"].long()
            batch["srsd_tokens"][i, :M] = f["tokens"].long()
            batch["srsd_parents"][i, :M] = f["parents"].long()
            batch["srsd_depths"][i, :M] = f["depths"].long()
            batch["srsd_tgt_node"][i, :M] = f["tgt_node"]
            batch["srsd_valid"][i, :M] = True
            batch["srsd_path_tokens"][i, :P] = f["path_tokens"].long()
            batch["srsd_path_nodes"][i, :P] = f["path_nodes"].long()
            batch["srsd_anchor_pos"][i] = int(f["anchor_pos"])
        return batch


tokenizer = AutoTokenizer.from_pretrained(args.basepath, trust_remote_code=True)
if args.eal_mode == "sample_rsd_offline":
    if not args.srsd_cache:
        raise SystemExit("sample_rsd_offline requires --srsd_cache DIR")
    traindataset = SRSDCacheDataset(args.srsd_cache)
    testdataset = build_dataset(tokenizer, args.testpath, args.max_len,
                                gamma=args.gamma)
    print(f"[SmallLMTreeV2] offline cache {args.srsd_cache}: "
          f"{len(traindataset)} records")
else:
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
warmup_steps = max(1, total_steps // 10)
ds_config["scheduler"]["params"]["total_num_steps"] = max(1, total_steps)
ds_config["scheduler"]["params"]["warmup_num_steps"] = warmup_steps
if args.lr is not None:
    ds_config["scheduler"]["params"]["warmup_max_lr"] = args.lr

samples_per_gpu = math.ceil(len(traindataset) / n_gpus)
iters_per_epoch = math.ceil(samples_per_gpu / micro_bs)
total_iters = iters_per_epoch * args.num_epochs
save_every = args.save_every_steps
if save_every > 0:
    save_every = math.ceil(save_every / grad_accum) * grad_accum
snapshots = {}
if args.snapshot_frac > 0:
    for k in range(1, int(math.ceil(1.0 / args.snapshot_frac))):
        frac = k * args.snapshot_frac
        b = int(round(frac * iters_per_epoch / grad_accum)) * grad_accum
        if 0 < b < iters_per_epoch:
            snapshots[b] = int(round(100 * frac))

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

if args.seed > 0:
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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
model.srsd_tgt_cache = bool(args.srsd_tgt_cache) and args.eal_mode == "sample_rsd"
model.anchor_chunk = int(args.anchor_chunk)
if args.eal_mode == "sample_rsd":
    print("[SmallLMTreeV2] sample_rsd loss = dense residual TV (Chain al_tv analogue)")
    if model.anchor_chunk > 0:
        print(f"[SmallLMTreeV2] anchor_chunk={model.anchor_chunk} "
              f"(n_anchors={args.n_anchors})")
if model.srsd_tgt_cache:
    try:
        from sdpo.srsd_tgt_cache import OnlineTargetPathCache
    except ImportError:
        from srsd_tgt_cache import OnlineTargetPathCache
    model._srsd_tgt_bank = OnlineTargetPathCache(
        max_paths=args.srsd_tgt_cache_max_paths, top_m=args.srsd_tgt_cache_m)
    print(f"[SmallLMTreeV2] online target path cache "
          f"m={args.srsd_tgt_cache_m} max_paths={args.srsd_tgt_cache_max_paths}")
else:
    model._srsd_tgt_bank = None
model.top4_start_eal = args.top4_start_eal
model.prob_starts = args.prob_starts
model.eal_all_starts = args.eal_all_starts
model.boundary_tau = args.boundary_tau
model.eal_path_agg = args.eal_path_agg
model.eal_alpha = args.eal_alpha
model.exact_segment_frac = args.exact_segment_frac
model.exact_burn = args.exact_burn
model.exact_random_frac = args.exact_random_frac
model.eal_gn_ratio = args.eal_gradnorm_ratio
model.eal_gn_every = args.eal_gradnorm_every
model.dpace_alpha = args.dpace_alpha
model.gto_coef = args.gto_coef
if args.gto_coef > 0:
    model.init_gto_ref(args.draftpath, model_dtype)

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
                       rank=global_rank, shuffle=True, seed=args.seed),
    micro_bs)
test_sampler = DistributedSampler(testdataset, num_replicas=world_size,
                                  rank=global_rank, shuffle=False)
train_collate = (SRSDCacheCollator() if args.eal_mode == "sample_rsd_offline"
                 else DataCollator())
train_loader = DataLoader(traindataset, batch_size=micro_bs,
                          sampler=train_sampler, num_workers=0,
                          pin_memory=True, collate_fn=train_collate)
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


def _save_hf_draft(hf_dir):
    if global_rank != 0:
        return
    try:
        os.makedirs(hf_dir, exist_ok=True)
        model_engine.module.draft_model.save_pretrained(hf_dir)
        tokenizer.save_pretrained(hf_dir)
        print(f"  Saved HF draft to {hf_dir}", flush=True)
    except Exception as e:
        print(f"  WARNING: HF draft save failed ({hf_dir}): {e}", flush=True)


def _rolling_al(hist):
    if not hist:
        return None
    return sum(hist) / len(hist)


# Early-stop on train analytic AL decline (checked at mid-saves).
al_hist = collections.deque(
    maxlen=max(int(args.early_stop_al_window), 1)
) if args.early_stop_al_window > 0 else None
best_roll_al = None
al_decline_streak = 0
early_stop = False
hf_best_dir = os.path.join(args.savedir, "hf_best_al")


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
            # Keep a rolling HF draft at the latest mid-save for quick eval.
            deepspeed.comm.barrier()
            if global_rank == 0:
                _save_hf_draft(os.path.join(args.savedir, "hf_mid"))
            deepspeed.comm.barrier()
            # AL early-stop check at mid-save points (needs a filled window).
            stop_dev = (torch.device("cuda", rank)
                        if torch.cuda.is_available() else torch.device("cpu"))
            stop_flag = torch.zeros(1, dtype=torch.int32, device=stop_dev)
            if (global_rank == 0 and al_hist is not None
                    and args.early_stop_al_patience > 0
                    and len(al_hist) >= min(args.early_stop_al_window, 50)):
                roll = _rolling_al(al_hist)
                improved = (best_roll_al is None
                            or roll > best_roll_al + 1e-6)
                if improved:
                    best_roll_al = roll
                    al_decline_streak = 0
                    _save_hf_draft(hf_best_dir)
                    meta = {
                        "best_roll_al": best_roll_al,
                        "step": global_step,
                        "epoch": epoch,
                        "window": len(al_hist),
                    }
                    with open(os.path.join(hf_best_dir, "best_al.json"), "w") as f:
                        json.dump(meta, f, indent=2)
                    print(f"  [early-stop] new best roll_AL={roll:.4f} "
                          f"step={global_step} -> {hf_best_dir}", flush=True)
                elif (best_roll_al is not None
                      and roll < best_roll_al - args.early_stop_al_delta):
                    al_decline_streak += 1
                    print(f"  [early-stop] AL decline "
                          f"roll={roll:.4f} best={best_roll_al:.4f} "
                          f"streak={al_decline_streak}/"
                          f"{args.early_stop_al_patience} step={global_step}",
                          flush=True)
                    if al_decline_streak >= args.early_stop_al_patience:
                        stop_flag.fill_(1)
                        print(f"  [early-stop] stopping: AL fell "
                              f">={args.early_stop_al_delta} below best for "
                              f"{al_decline_streak} mid-checks", flush=True)
                else:
                    al_decline_streak = 0
                    print(f"  [early-stop] roll_AL={roll:.4f} "
                          f"best={best_roll_al:.4f} (flat) step={global_step}",
                          flush=True)
            deepspeed.comm.broadcast(stop_flag, 0)
            if int(stop_flag.item()) == 1:
                early_stop = True
                break
        if batch_idx in snapshots and batch_idx > resume_skip and global_rank == 0:
            snap_dir = f"{args.savedir}/hf_e{epoch}_p{snapshots[batch_idx]:03d}"
            if not os.path.exists(os.path.join(snap_dir, "config.json")):
                model_engine.module.draft_model.save_pretrained(snap_dir)
        model.zero_grad()
        fwd_kwargs = dict(
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
        # anchor_chunk is read from model.anchor_chunk inside forward
        if args.eal_mode == "sample_rsd_offline":
            fwd_kwargs.update(
                srsd_tokens=data["srsd_tokens"].to(rank),
                srsd_parents=data["srsd_parents"].to(rank),
                srsd_depths=data["srsd_depths"].to(rank),
                srsd_tgt_node=data["srsd_tgt_node"].to(rank),
                srsd_valid=data["srsd_valid"].to(rank),
                srsd_path_tokens=data["srsd_path_tokens"].to(rank),
                srsd_path_nodes=data["srsd_path_nodes"].to(rank),
                srsd_anchor_pos=data["srsd_anchor_pos"].to(rank),
            )
        total_loss, _, metrics = model_engine(**fwd_kwargs)
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
            if al_hist is not None:
                al_hist.append(float(metrics["eal_mean"]))
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
                        "exact_rounds", "match_rate", "n_matched_paths",
                        "num_valid", "tgt_cache_hit_rate", "tgt_cache_size",
                        "tgt_cache_skipped"):
                if key in metrics:
                    row[key] = metrics[key]
            for key in metrics:
                if key.startswith("path") and key.endswith(("_eal", "_al", "_var")):
                    row[key] = metrics[key]
            _append_jsonl(step_log_path, row)
            extra = ""
            if "tgt_cache_hit_rate" in row:
                extra += (f"  tgt_hit={row['tgt_cache_hit_rate']:.2f}"
                          f"  cache_n={int(row.get('tgt_cache_size', 0))}")
            if "match_rate" in row:
                extra += f"  match={row['match_rate']:.2f}"
            print(f"  step {global_step} epoch {epoch} | "
                  f"loss={row['loss']:.4f}  eal_loss={row['eal_loss']:.4f}  "
                  f"dense_loss={metrics['dense_loss']:.4f}  "
                  f"analytic_accepted_length={row['analytic_accepted_length']:.4f}  "
                  f"kl={row['kl']:.4f}  grad_norm={row['grad_norm']:.4f}{extra}",
                  flush=True)
            wb = {
                "train/total_loss": metrics["total_loss"],
                "train/kl_loss": metrics["kl_loss"],
                "train/aux_loss": metrics["aux_loss"],
                "train/eal_loss": metrics["eal_loss"],
                "train/dense_loss": metrics["dense_loss"],
                "train/eal_mean": metrics["eal_mean"],
                "train/num_anchors": metrics["num_anchors"],
                "train/grad_norm": gn,
                "train/lr": lr,
            }
            for k in ("tgt_cache_hit_rate", "tgt_cache_size", "tgt_cache_skipped",
                      "match_rate", "n_matched_paths", "tree_nodes"):
                if k in metrics:
                    wb[f"train/{k}"] = metrics[k]
            wandb.log(wb, step=global_step)

    if early_stop:
        deepspeed.comm.barrier()
        if global_rank == 0:
            print(f"[SmallLMTreeV2] Early stop at epoch={epoch} step={global_step}; "
                  f"best roll_AL={best_roll_al} kept at {hf_best_dir}", flush=True)
            # Point epoch-0 HF path at best so the existing eval sbatch works.
            hf_link = f"{args.savedir}/hf_epoch_{epoch}"
            if os.path.isdir(hf_best_dir):
                if os.path.islink(hf_link) or os.path.exists(hf_link):
                    if os.path.islink(hf_link):
                        os.unlink(hf_link)
                    else:
                        shutil.rmtree(hf_link, ignore_errors=True)
                try:
                    os.symlink(hf_best_dir, hf_link)
                except OSError:
                    shutil.copytree(hf_best_dir, hf_link)
            eval_job = _submit_epoch_eval(epoch)
            if eval_job:
                _append_jsonl(epoch_log_path, {
                    "epoch": epoch,
                    "early_stop": True,
                    "best_roll_al": best_roll_al,
                    "step": global_step,
                    "accept_length_eval_job": eval_job,
                    "hf_dir": hf_best_dir,
                })
        break

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

    # Save HF draft FIRST: on 1x40GB, ZeRO save_checkpoint can OOM/NCCL-fail
    # after epoch eval and would otherwise skip the usable eval weights.
    ckpt_dir = f"{args.savedir}/state_{epoch}"
    hf_dir = f"{args.savedir}/hf_epoch_{epoch}"
    if global_rank == 0:
        try:
            # Early-stop may have left hf_epoch_* as a symlink to hf_best_al;
            # replace it with a real directory so we do not overwrite best.
            if os.path.islink(hf_dir):
                os.unlink(hf_dir)
            elif os.path.isdir(hf_dir):
                shutil.rmtree(hf_dir, ignore_errors=True)
            model_engine.module.draft_model.save_pretrained(hf_dir)
            tokenizer.save_pretrained(hf_dir)
            print(f"  Saved HF draft to {hf_dir}", flush=True)
        except Exception as e:
            print(f"  WARNING: HF draft save failed: {e}", flush=True)
    deepspeed.comm.barrier()
    try:
        model_engine.save_checkpoint(ckpt_dir, exclude_frozen_parameters=True)
        if global_rank == 0:
            print(f"  Saved DeepSpeed checkpoint to {ckpt_dir}", flush=True)
    except Exception as e:
        if global_rank == 0:
            print(f"  WARNING: DeepSpeed checkpoint save failed (HF draft kept): {e}",
                  flush=True)
    if global_rank == 0:
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
