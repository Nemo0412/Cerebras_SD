#!/bin/bash
# Train Sample Tree EAL / Sample Chain EAL / KL on README regen data, THEN eval on tree.
# Data (README Tree-EAL section):
#   train: data/sharegpt_qwen3_8b_regen.jsonl  (Qwen3-8B regenerated ShareGPT)
#   val:   sdpo/data/mixed_val_80.jsonl
#   eval:  MT-Bench via eval_small_lm_tree.py (sample + rsd, path_topk=4)
set -uo pipefail

REPO=${REPO:-/home/ubuntu/workspace/SD/Cerebras_SD}
ROOT=${ROOT:-$REPO}
PY=${PY:-python3}
DS_BIN=${DS_BIN:-deepspeed}
GPUS=${GPUS:-0,1}
PORT_BASE=${PORT_BASE:-29700}
LR=${LR:-1e-6}
EPOCHS=${EPOCHS:-1}
MAX_SAMPLES=${MAX_SAMPLES:-0}   # 0 = use full regen file
MAX_LEN=${MAX_LEN:-512}
NUM_EVAL=${NUM_EVAL:-80}

BASE=$ROOT/models/Qwen3-8B
DRAFT=$ROOT/models/Qwen3-0.6B
TRAIN=$ROOT/data/sharegpt_qwen3_8b_regen.jsonl
VAL=$ROOT/sdpo/data/mixed_val_80.jsonl
DS_TREE=$REPO/sdpo/scripts/tree_eal/ds_zero2_a100_2gpu.json
DS_CHAIN=$REPO/sdpo/scripts/tree_eal/ds_zero2_a100_chain.json
SAVE_ROOT=$ROOT/checkpoints
LOG=$ROOT/logs

export HF_HOME=${HF_HOME:-$ROOT/hf}
export WANDB_MODE=offline
export WANDB_DIR=$LOG/wandb
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$LOG" "$SAVE_ROOT"
cd "$REPO"

if [ ! -f "$TRAIN" ]; then
  echo "MISSING train data: $TRAIN"
  exit 1
fi
if [ ! -f "$VAL" ]; then
  echo "MISSING val data: $VAL"
  exit 1
fi

SAMPLE_FLAG=""
if [ -n "$MAX_SAMPLES" ] && [ "$MAX_SAMPLES" != "0" ]; then
  SAMPLE_FLAG="--max_train_samples $MAX_SAMPLES"
fi

echo "Train data: $TRAIN ($(wc -l < "$TRAIN") lines)"
echo "Val data:   $VAL ($(wc -l < "$VAL") lines)"
echo "max_samples=${MAX_SAMPLES:-full} max_len=$MAX_LEN epochs=$EPOCHS gpus=$GPUS"

train_tree () {
  local TAG=$1; shift
  local PORT=$1; shift
  local SAVE=$SAVE_ROOT/$TAG
  rm -rf "$SAVE"
  echo "========== TRAIN $TAG =========="
  "$DS_BIN" --master_port "$PORT" --include "localhost:$GPUS" \
    sdpo/main_small_lm_tree_v2.py \
    --basepath "$BASE" --draftpath "$DRAFT" \
    --trainpath "$TRAIN" --testpath "$VAL" \
    --deepspeed_config "$DS_TREE" --savedir "$SAVE" \
    --top_k_per_depth 4,3,2,1,1,1,1 --tree_budget 0 \
    --temperature 1.0 --path_topk 4 --lr "$LR" \
    --max_len "$MAX_LEN" --num_epochs "$EPOCHS" \
    $SAMPLE_FLAG "$@" \
    > "$LOG/${TAG}_train.log" 2>&1
  echo "[train] $TAG exit=$?"
}

train_chain_style () {
  local TAG=$1; shift
  local PORT=$1; shift
  local SAVE=$SAVE_ROOT/$TAG
  rm -rf "$SAVE"
  echo "========== TRAIN $TAG =========="
  "$DS_BIN" --master_port "$PORT" --include "localhost:$GPUS" \
    sdpo/main_small_lm.py \
    --basepath "$BASE" --draftpath "$DRAFT" \
    --trainpath "$TRAIN" --testpath "$VAL" \
    --deepspeed_config "$DS_CHAIN" --savedir "$SAVE" \
    --lr "$LR" --max_len "$MAX_LEN" --num_epochs "$EPOCHS" --gamma 7 \
    --T_max 0.1 --T_min 0.1 \
    $SAMPLE_FLAG "$@" \
    > "$LOG/${TAG}_train.log" 2>&1
  echo "[train] $TAG exit=$?"
}

# 1) Sample Tree EAL
train_tree sample_tree_eal $((PORT_BASE+0)) \
  --eal_mode sample_rsd \
  --n_anchors 4 --anchor_kl_coef 0.1 --eal_aux_coef 1.0

# 2) Sample Chain EAL (position chain, α=min(p,q), no siblings)
train_chain_style sample_chain_eal $((PORT_BASE+1)) \
  --anchor none --aux_loss al_tv --sigmoid_coef 1.0 \
  --anchor_weight none --aux_weight uniform \
  --prob_T_max 1.0 --prob_T_min 1.0

# 3) KL per position
train_chain_style kl_only $((PORT_BASE+2)) \
  --anchor kl --aux_loss none --sigmoid_coef 0.0 \
  --anchor_weight none --aux_weight pow08

echo "========== ALL TRAINS DONE — START TREE EVAL =========="
# Parallel eval: one method per GPU across mt_bench,gsm8k,humaneval.
BENCHES=${BENCHES:-mt_bench,gsm8k,humaneval}
EVAL_GPUS=${EVAL_GPUS:-0,1,2,3}
IFS=',' read -r -a GPU_ARR <<< "$EVAL_GPUS"

pick_ckpt () {
  local SAVE=$1
  local d
  for d in "$SAVE"/hf_epoch_*; do
    [ -d "$d" ] && echo "$d" && return 0
  done
  # Prefer plain state_N over *_stripped
  for d in "$SAVE"/state_[0-9]*; do
    [ -d "$d" ] || continue
    case "$d" in *_stripped) continue ;; esac
    echo "$d"; return 0
  done
  return 1
}

eval_tree () {
  local TAG=$1
  local GPU=$2
  local SAVE=$SAVE_ROOT/$TAG
  local CKPT
  CKPT=$(pick_ckpt "$SAVE") || { echo "[eval] $TAG: no checkpoint"; return 1; }
  echo "========== EVAL TREE $TAG on GPU $GPU (benches=$BENCHES) =========="
  mkdir -p "$SAVE/eval"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" sdpo/eval_small_lm_tree.py \
    --base-model-path "$BASE" --draft-model-path "$CKPT" \
    --bench-name "$BENCHES" --gamma 7 --top-k 4,3,2,1,1,1,1 --budget 128 \
    --max-new-tokens 256 --num-samples "$NUM_EVAL" --temperature 1.0 --seed 0 \
    --draft-mode sample --verify-mode rsd --path-topk 4 \
    --tag "${TAG}_tree_rsd" --output-dir "$SAVE/eval" \
    > "$LOG/${TAG}_eval.log" 2>&1
  echo "[eval] $TAG exit=$?"
}

TAGS=(sample_tree_eal sample_chain_eal kl_only)
PIDS=()
for i in "${!TAGS[@]}"; do
  TAG=${TAGS[$i]}
  GPU=${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}
  eval_tree "$TAG" "$GPU" &
  PIDS+=($!)
done
for pid in "${PIDS[@]}"; do
  wait "$pid" || true
done

echo "========== SUMMARY =========="
"$PY" - <<'PY'
import json, glob, statistics
root="/home/ubuntu/workspace/SD/Cerebras_SD"
benches=["mt_bench","gsm8k","humaneval"]
hdr=f"{'method':<20}" + "".join(f"{b:>10}" for b in benches) + f"{'avg':>10} {'tok/s':>10}"
print(hdr)
rows=[]
for tag in ["sample_tree_eal","sample_chain_eal","kl_only"]:
    paths=sorted(glob.glob(f"{root}/checkpoints/{tag}/eval/{tag}_tree_rsd*.json"))
    if not paths:
        print(f"{tag:<20}" + "       NA"*4 + "       NA"); continue
    res=json.load(open(paths[-1])).get("results",{})
    alphas=[]; tps=[]
    cells=[]
    for b in benches:
        t=res.get(b,{}).get("tree",{})
        a=t.get("mean_alpha"); tp=t.get("tokens_per_sec")
        cells.append(f"{a:.4f}" if isinstance(a,(int,float)) else "NA")
        if isinstance(a,(int,float)): alphas.append(a)
        if isinstance(tp,(int,float)): tps.append(tp)
    avg=statistics.mean(alphas) if alphas else None
    tp=statistics.mean(tps) if tps else None
    aa=f"{avg:.4f}" if avg is not None else "NA"
    tt=f"{tp:.2f}" if tp is not None else "NA"
    print(f"{tag:<20}" + "".join(f"{c:>10}" for c in cells) + f"{aa:>10} {tt:>10}")
    if avg is not None: rows.append((tag, avg))
best=[r for r in rows if isinstance(r[1],(int,float))]
if best:
    w=max(best, key=lambda x:x[1])
    print(f"\nBest avg(mt+gsm+he): {w[0]} mean_alpha={w[1]:.4f}")
PY
