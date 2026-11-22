#!/bin/bash
# Aligned single-seed run matching README ALTV scale:
#   max_len=2048, max_samples=60000, epochs=3, seed=0
# Order: Chain EAL (al_tv) -> Tree EAL (sample_rsd + dense_tv, EAL-as-loss) -> KL
# Then eval: chain@T=0.7 sample/ratio + tree sample/rsd path_topk=4
set -uo pipefail
cd /home/ubuntu/workspace/SD/Cerebras_SD
LOG=logs/run_aligned_60k_3ep.log
mkdir -p logs checkpoints

export WANDB_MODE=offline WANDB_DIR=logs/wandb
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-$PWD/hf}
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
unset PYTORCH_CUDA_ALLOC_CONF

BASE=models/Qwen3-8B
DRAFT=models/Qwen3-0.6B
TRAIN=data/sharegpt_qwen3_8b_regen_60k.jsonl
VAL=sdpo/data/mixed_val_80.jsonl
DS=sdpo/scripts/tree_eal/ds_zero2_a100_4gpu_l2k.json
LR=1e-6
EPOCHS=3
MAX_LEN=2048
MAX_SAMPLES=60000
SEED=0
GPUS=0,1,2,3
PORT=29900

echo "[pipeline] start $(date -Is)" | tee -a "$LOG"
echo "train=$TRAIN max_len=$MAX_LEN max_samples=$MAX_SAMPLES epochs=$EPOCHS seed=$SEED" | tee -a "$LOG"

if [ ! -f "$TRAIN" ]; then
  echo "MISSING $TRAIN — regen first" | tee -a "$LOG"
  exit 1
fi
echo "regen lines=$(wc -l < "$TRAIN")" | tee -a "$LOG"

train_chain () {
  local TAG=$1; shift
  local SAVE=checkpoints/$TAG
  rm -rf "$SAVE"
  echo "========== TRAIN $TAG ==========" | tee -a "$LOG"
  deepspeed --master_port "$PORT" --include "localhost:$GPUS" \
    sdpo/main_small_lm.py \
    --basepath "$BASE" --draftpath "$DRAFT" \
    --trainpath "$TRAIN" --testpath "$VAL" \
    --deepspeed_config "$DS" --savedir "$SAVE" \
    --lr "$LR" --max_len "$MAX_LEN" --max_train_samples "$MAX_SAMPLES" \
    --num_epochs "$EPOCHS" --gamma 7 --seed "$SEED" \
    --T_max 0.1 --T_min 0.1 \
    "$@" \
    > "logs/${TAG}_train.log" 2>&1
  echo "[train] $TAG exit=$?" | tee -a "$LOG"
  PORT=$((PORT+1))
}

train_tree () {
  local TAG=$1; shift
  local SAVE=checkpoints/$TAG
  rm -rf "$SAVE"
  echo "========== TRAIN $TAG ==========" | tee -a "$LOG"
  deepspeed --master_port "$PORT" --include "localhost:$GPUS" \
    sdpo/main_small_lm_tree_v2.py \
    --basepath "$BASE" --draftpath "$DRAFT" \
    --trainpath "$TRAIN" --testpath "$VAL" \
    --deepspeed_config "$DS" --savedir "$SAVE" \
    --top_k_per_depth 4,3,2,1,1,1,1 --tree_budget 0 \
    --temperature 1.0 --path_topk 4 --lr "$LR" \
    --max_len "$MAX_LEN" --max_train_samples "$MAX_SAMPLES" \
    --num_epochs "$EPOCHS" --seed "$SEED" \
    "$@" \
    > "logs/${TAG}_train.log" 2>&1
  echo "[train] $TAG exit=$?" | tee -a "$LOG"
  PORT=$((PORT+1))
}

pick_ckpt () {
  local SAVE=$1 d
  for d in "$SAVE"/hf_epoch_*; do [ -d "$d" ] && echo "$d" && return 0; done
  for d in "$SAVE"/state_[0-9]*; do
    [ -d "$d" ] || continue
    case "$d" in *_stripped) continue ;; esac
    echo "$d"; return 0
  done
  # latest numeric state
  local best=""
  for d in "$SAVE"/state_*; do
    case "$d" in *_stripped) continue ;; esac
    [ -d "$d" ] && best=$d
  done
  [ -n "$best" ] && echo "$best" && return 0
  return 1
}

# 1) Chain EAL = ALTV (aligned with README)
train_chain sample_chain_eal_l2k_3ep \
  --anchor none --aux_loss al_tv --sigmoid_coef 1.0 \
  --anchor_weight none --aux_weight pow08 \
  --prob_T_max 1.0 --prob_T_min 0.1

# 2) Tree EAL = sample_rsd + dense_tv (RSD E[L] as differentiable loss)
train_tree sample_tree_eal_l2k_3ep \
  --eal_mode sample_rsd \
  --n_anchors 4 --anchor_kl_coef 0.1 --eal_aux_coef 1.0 \
  --eal_score_coef 0.0 --n_draws 2

# 3) KL baseline
train_chain kl_only_l2k_3ep \
  --anchor kl --aux_loss none --sigmoid_coef 0.0 \
  --anchor_weight none --aux_weight pow08 \
  --prob_T_max 1.0 --prob_T_min 0.1

echo "========== EVAL ==========" | tee -a "$LOG"

eval_chain07 () {
  local TAG=$1 GPU=$2
  local CKPT; CKPT=$(pick_ckpt "checkpoints/$TAG") || { echo "[eval] $TAG no ckpt"; return 1; }
  if [ -d "${CKPT}_stripped" ] && [ -e "${CKPT}_stripped/config.json" ]; then
    CKPT=${CKPT}_stripped
  fi
  mkdir -p "checkpoints/$TAG/eval_chain07"
  echo "[eval chain07] $TAG gpu=$GPU ckpt=$CKPT" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU python3 sdpo/eval_small_lm_tree.py \
    --base-model-path "$BASE" --draft-model-path "$CKPT" \
    --bench-name mt_bench,gsm8k,humaneval --baseline-only --baseline-gamma 7 \
    --max-new-tokens 256 --num-samples 80 --temperature 0.7 --seed 0 \
    --draft-mode sample --verify-mode ratio \
    --tag "${TAG}_chain_t07" --output-dir "checkpoints/$TAG/eval_chain07" \
    > "logs/${TAG}_chain07_eval.log" 2>&1
  echo "[eval chain07] $TAG exit=$?" | tee -a "$LOG"
}

eval_tree_rsd () {
  local TAG=$1 GPU=$2
  local CKPT; CKPT=$(pick_ckpt "checkpoints/$TAG") || { echo "[eval] $TAG no ckpt"; return 1; }
  if [ -d "${CKPT}_stripped" ] && [ -e "${CKPT}_stripped/config.json" ]; then
    CKPT=${CKPT}_stripped
  fi
  mkdir -p "checkpoints/$TAG/eval_tree"
  echo "[eval tree-rsd] $TAG gpu=$GPU ckpt=$CKPT" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU python3 sdpo/eval_small_lm_tree.py \
    --base-model-path "$BASE" --draft-model-path "$CKPT" \
    --bench-name mt_bench,gsm8k,humaneval \
    --gamma 7 --top-k 4,3,2,1,1,1,1 --budget 128 \
    --max-new-tokens 256 --num-samples 80 --temperature 1.0 --seed 0 \
    --draft-mode sample --verify-mode rsd --path-topk 4 \
    --tag "${TAG}_tree_rsd" --output-dir "checkpoints/$TAG/eval_tree" \
    > "logs/${TAG}_tree_eval.log" 2>&1
  echo "[eval tree-rsd] $TAG exit=$?" | tee -a "$LOG"
}

# Parallel evals across GPUs: 3 methods x chain07 on 0/1/2, then tree on 0/1/2
eval_chain07 sample_chain_eal_l2k_3ep 0 &
eval_chain07 sample_tree_eal_l2k_3ep 1 &
eval_chain07 kl_only_l2k_3ep 2 &
wait

eval_tree_rsd sample_chain_eal_l2k_3ep 0 &
eval_tree_rsd sample_tree_eal_l2k_3ep 1 &
eval_tree_rsd kl_only_l2k_3ep 2 &
wait

python3 - <<'PY' | tee -a logs/run_aligned_60k_3ep.log
import json, glob, statistics, os
root = "/home/ubuntu/workspace/SD/Cerebras_SD"
benches = ["mt_bench", "gsm8k", "humaneval"]
tags = [
    ("sample_chain_eal_l2k_3ep", "Chain EAL"),
    ("sample_tree_eal_l2k_3ep", "Tree EAL dense_tv"),
    ("kl_only_l2k_3ep", "KL"),
]
print("=== chain @ T=0.7 sample/ratio ===")
print(f"{'method':<22}" + "".join(f"{b:>10}" for b in benches) + f"{'avg':>10}")
for tag, name in tags:
    paths = sorted(glob.glob(f"{root}/checkpoints/{tag}/eval_chain07/{tag}_chain_t07*.json"))
    if not paths:
        print(f"{name:<22} NA"); continue
    res = json.load(open(paths[-1]))["results"]
    alphas, cells = [], []
    for b in benches:
        a = res.get(b, {}).get("baseline_chain", {}).get("mean_alpha")
        cells.append(f"{a:.4f}" if isinstance(a, (int, float)) else "NA")
        if isinstance(a, (int, float)):
            alphas.append(a)
    avg = statistics.mean(alphas) if len(alphas) == 3 else None
    print(f"{name:<22}" + "".join(f"{c:>10}" for c in cells)
          + f"{(f'{avg:.4f}' if avg else 'NA'):>10}")

print("\n=== tree RSD @ T=1 path_topk=4 ===")
print(f"{'method':<22}" + "".join(f"{b:>10}" for b in benches) + f"{'avg':>10}")
for tag, name in tags:
    paths = sorted(glob.glob(f"{root}/checkpoints/{tag}/eval_tree/{tag}_tree_rsd*.json"))
    if not paths:
        print(f"{name:<22} NA"); continue
    res = json.load(open(paths[-1]))["results"]
    alphas, cells = [], []
    for b in benches:
        a = res.get(b, {}).get("tree", {}).get("mean_alpha")
        cells.append(f"{a:.4f}" if isinstance(a, (int, float)) else "NA")
        if isinstance(a, (int, float)):
            alphas.append(a)
    avg = statistics.mean(alphas) if len(alphas) == 3 else None
    print(f"{name:<22}" + "".join(f"{c:>10}" for c in cells)
          + f"{(f'{avg:.4f}' if avg else 'NA'):>10}")
PY

echo "[pipeline] done $(date -Is)" | tee -a "$LOG"
