#!/bin/bash
# Fast A/B: does Tree EAL (dense_tv) beat Chain EAL (al_tv) on tree-rsd?
# Uses existing 2k regen, max_len=1024 (~341 samples), 3 epochs, seed=0, 4 GPUs.
# Skips 60k regen and KL. Eval: tree RSD (+ chain@0.7 for reference).
set -uo pipefail
cd /home/ubuntu/workspace/SD/Cerebras_SD
LOG=logs/run_fast_tree_vs_chain.log
mkdir -p logs checkpoints

export WANDB_MODE=offline WANDB_DIR=logs/wandb
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-$PWD/hf}
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
unset PYTORCH_CUDA_ALLOC_CONF

BASE=models/Qwen3-8B
DRAFT=models/Qwen3-0.6B
TRAIN=data/sharegpt_qwen3_8b_regen.jsonl
VAL=sdpo/data/mixed_val_80.jsonl
DS=sdpo/scripts/tree_eal/ds_zero2_a100_4gpu_l2k.json
LR=1e-6
EPOCHS=2
MAX_LEN=1024
SEED=0
GPUS=0,1,2,3
PORT=29910

echo "[fast] start $(date -Is) max_len=$MAX_LEN epochs=$EPOCHS" | tee "$LOG"

pick_ckpt () {
  local SAVE=$1 d best=""
  # Prefer latest HF epoch (lexicographic last: hf_epoch_1 > hf_epoch_0)
  for d in "$SAVE"/hf_epoch_*; do [ -d "$d" ] && best=$d; done
  if [ -n "$best" ]; then echo "$best"; return 0; fi
  for d in "$SAVE"/state_[0-9]*; do
    case "$d" in *_stripped) continue ;; esac
    [ -d "$d" ] && best=$d
  done
  [ -n "$best" ] && echo "$best" && return 0
  return 1
}

# 1) Chain EAL
TAG=fast_chain_eal
rm -rf checkpoints/$TAG
echo "========== TRAIN $TAG ==========" | tee -a "$LOG"
deepspeed --master_port $PORT --include "localhost:$GPUS" \
  sdpo/main_small_lm.py \
  --basepath "$BASE" --draftpath "$DRAFT" \
  --trainpath "$TRAIN" --testpath "$VAL" \
  --deepspeed_config "$DS" --savedir checkpoints/$TAG \
  --lr "$LR" --max_len "$MAX_LEN" --num_epochs "$EPOCHS" --seed "$SEED" \
  --gamma 7 --T_max 0.1 --T_min 0.1 \
  --anchor none --aux_loss al_tv --sigmoid_coef 1.0 \
  --anchor_weight none --aux_weight pow08 \
  --prob_T_max 1.0 --prob_T_min 0.1 \
  > logs/${TAG}_train.log 2>&1
echo "[train] $TAG exit=$?" | tee -a "$LOG"
PORT=$((PORT+1))

# 2) Tree EAL = RSD E[L] as loss (dense_tv); n_anchors=2 to fit 40GB@1024
TAG=fast_tree_eal
rm -rf checkpoints/$TAG
echo "========== TRAIN $TAG ==========" | tee -a "$LOG"
deepspeed --master_port $PORT --include "localhost:$GPUS" \
  sdpo/main_small_lm_tree_v2.py \
  --basepath "$BASE" --draftpath "$DRAFT" \
  --trainpath "$TRAIN" --testpath "$VAL" \
  --deepspeed_config "$DS" --savedir checkpoints/$TAG \
  --top_k_per_depth 4,3,2,1,1,1,1 --tree_budget 0 \
  --temperature 1.0 --path_topk 4 --lr "$LR" \
  --max_len "$MAX_LEN" --num_epochs "$EPOCHS" --seed "$SEED" \
  --eal_mode sample_rsd \
  --n_anchors 2 --anchor_kl_coef 0.1 --eal_aux_coef 1.0 \
  --eal_score_coef 0.0 --n_draws 2 \
  > logs/${TAG}_train.log 2>&1
echo "[train] $TAG exit=$?" | tee -a "$LOG"

echo "========== EVAL ==========" | tee -a "$LOG"

eval_tree () {
  local TAG=$1 GPU=$2
  local CKPT; CKPT=$(pick_ckpt checkpoints/$TAG) || { echo "no ckpt $TAG"; return 1; }
  case "$CKPT" in
    *state_*) [ -d "${CKPT}_stripped" ] && [ -e "${CKPT}_stripped/config.json" ] && CKPT=${CKPT}_stripped ;;
  esac
  mkdir -p checkpoints/$TAG/eval_tree
  CUDA_VISIBLE_DEVICES=$GPU python3 sdpo/eval_small_lm_tree.py \
    --base-model-path "$BASE" --draft-model-path "$CKPT" \
    --bench-name mt_bench,gsm8k,humaneval \
    --gamma 7 --top-k 4,3,2,1,1,1,1 --budget 128 \
    --max-new-tokens 256 --num-samples 80 --temperature 1.0 --seed 0 \
    --draft-mode sample --verify-mode rsd --path-topk 4 \
    --tag "${TAG}_tree_rsd" --output-dir checkpoints/$TAG/eval_tree \
    > logs/${TAG}_tree_eval.log 2>&1
  echo "[eval tree] $TAG exit=$?" | tee -a "$LOG"
}

eval_chain07 () {
  local TAG=$1 GPU=$2
  local CKPT; CKPT=$(pick_ckpt checkpoints/$TAG) || return 1
  case "$CKPT" in
    *state_*) [ -d "${CKPT}_stripped" ] && [ -e "${CKPT}_stripped/config.json" ] && CKPT=${CKPT}_stripped ;;
  esac
  mkdir -p checkpoints/$TAG/eval_chain07
  CUDA_VISIBLE_DEVICES=$GPU python3 sdpo/eval_small_lm_tree.py \
    --base-model-path "$BASE" --draft-model-path "$CKPT" \
    --bench-name mt_bench,gsm8k,humaneval --baseline-only --baseline-gamma 7 \
    --max-new-tokens 256 --num-samples 80 --temperature 0.7 --seed 0 \
    --draft-mode sample --verify-mode ratio \
    --tag "${TAG}_chain_t07" --output-dir checkpoints/$TAG/eval_chain07 \
    > logs/${TAG}_chain07_eval.log 2>&1
  echo "[eval chain07] $TAG exit=$?" | tee -a "$LOG"
}

eval_tree fast_chain_eal 0 &
eval_tree fast_tree_eal 1 &
eval_chain07 fast_chain_eal 2 &
eval_chain07 fast_tree_eal 3 &
wait

python3 - <<'PY' | tee -a logs/run_fast_tree_vs_chain.log
import json, glob, statistics
root = "."
benches = ["mt_bench", "gsm8k", "humaneval"]

def load(tag, kind):
    if kind == "tree":
        paths = sorted(glob.glob(f"checkpoints/{tag}/eval_tree/{tag}_tree_rsd*.json"))
        key = ("tree",)
    else:
        paths = sorted(glob.glob(f"checkpoints/{tag}/eval_chain07/{tag}_chain_t07*.json"))
        key = ("baseline_chain",)
    if not paths:
        return None
    res = json.load(open(paths[-1]))["results"]
    out = {}
    for b in benches:
        cur = res.get(b, {})
        for k in key:
            cur = cur.get(k, {}) if isinstance(cur, dict) else {}
        a = cur.get("mean_alpha") if isinstance(cur, dict) else None
        out[b] = a
    return out

print("=== PRIMARY: tree RSD (should favor Tree EAL) ===")
print(f"{'method':<16}" + "".join(f"{b:>10}" for b in benches) + f"{'avg':>10}")
rows = []
for tag, name in [("fast_tree_eal", "Tree EAL"), ("fast_chain_eal", "Chain EAL")]:
    d = load(tag, "tree")
    if not d:
        print(f"{name:<16} NA"); continue
    alphas = [d[b] for b in benches if isinstance(d[b], (int, float))]
    cells = [f"{d[b]:.4f}" if isinstance(d[b], (int, float)) else "NA" for b in benches]
    avg = statistics.mean(alphas) if len(alphas) == 3 else None
    print(f"{name:<16}" + "".join(f"{c:>10}" for c in cells) + f"{(f'{avg:.4f}' if avg else 'NA'):>10}")
    if avg is not None:
        rows.append((name, avg))
if len(rows) == 2:
    print(f"delta (Tree - Chain) = {rows[0][1] - rows[1][1]:+.4f}")

print("\n=== REF: chain @ T=0.7 ===")
for tag, name in [("fast_tree_eal", "Tree EAL"), ("fast_chain_eal", "Chain EAL")]:
    d = load(tag, "chain")
    if not d:
        print(f"{name:<16} NA"); continue
    alphas = [d[b] for b in benches if isinstance(d[b], (int, float))]
    cells = [f"{d[b]:.4f}" if isinstance(d[b], (int, float)) else "NA" for b in benches]
    avg = statistics.mean(alphas) if len(alphas) == 3 else None
    print(f"{name:<16}" + "".join(f"{c:>10}" for c in cells) + f"{(f'{avg:.4f}' if avg else 'NA'):>10}")
PY
echo "[fast] done $(date -Is)" | tee -a "$LOG"
