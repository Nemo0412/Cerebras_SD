#!/bin/bash
# Corrected 1-epoch comparison (Tree-EAL branch semantics):
#   Sample Tree EAL : sample_rsd + dense_tv + n_anchors=4 + KL=0.1  (sibling residual)
#   Sample Chain EAL: position-wise chain EAL with α=min(p,q) = al_tv (NO siblings)
#   KL              : per-position KL(target||draft)
# All three evaluated on TREE (sample + rsd, path_topk=4).
set -uo pipefail

REPO=${REPO:-/home/ubuntu/workspace/SD/Cerebras_SD}
ROOT=${ROOT:-$REPO}
PY=${PY:-python3}
DS_BIN=${DS_BIN:-deepspeed}
GPUS=${GPUS:-0,1}
PORT_BASE=${PORT_BASE:-29600}
LR=${LR:-1e-6}
EPOCHS=${EPOCHS:-1}
MAX_SAMPLES=${MAX_SAMPLES:-256}
MAX_LEN=${MAX_LEN:-512}
NUM_EVAL=${NUM_EVAL:-20}

BASE=$ROOT/models/Qwen3-8B
DRAFT=$ROOT/models/Qwen3-0.6B
TRAIN=$ROOT/sdpo/data/mixed_train.jsonl
VAL=$ROOT/sdpo/data/mixed_val.jsonl
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

eval_tree () {
  local TAG=$1
  local SAVE=$SAVE_ROOT/$TAG
  local CKPT=""
  for d in "$SAVE"/hf_epoch_*; do [ -d "$d" ] && CKPT=$d; done
  if [ -z "$CKPT" ]; then
    for d in "$SAVE"/state_*; do [ -d "$d" ] && CKPT=$d; done
  fi
  if [ -z "$CKPT" ] || [ ! -d "$CKPT" ]; then
    echo "[eval] $TAG: no checkpoint"
    return 1
  fi
  echo "========== EVAL TREE $TAG ckpt=$CKPT =========="
  mkdir -p "$SAVE/eval"
  CUDA_VISIBLE_DEVICES=${EVAL_GPU:-2} "$PY" sdpo/eval_small_lm_tree.py \
    --base-model-path "$BASE" --draft-model-path "$CKPT" \
    --bench-name mt_bench --gamma 7 --top-k 4,3,2,1,1,1,1 --budget 128 \
    --max-new-tokens 256 --num-samples "$NUM_EVAL" --temperature 1.0 --seed 0 \
    --draft-mode sample --verify-mode rsd --path-topk 4 \
    --tag "${TAG}_tree_rsd" --output-dir "$SAVE/eval" \
    > "$LOG/${TAG}_eval.log" 2>&1
  echo "[eval] $TAG exit=$?"
  "$PY" - <<PY
import json, glob
paths = sorted(glob.glob("$SAVE/eval/${TAG}_tree_rsd*.json"))
print("files:", paths)
for p in paths:
    j=json.load(open(p)); r=j.get("results",{}).get("mt_bench",{}).get("tree",{})
    print(f"[$TAG] mean_alpha={r.get('mean_alpha')} tok/s={r.get('tokens_per_sec')}")
PY
}

# ── 1) Sample Tree EAL (reuse if already trained) ─────────────────────────
TAG=sample_tree_eal
SAVE=$SAVE_ROOT/$TAG
if [ -d "$SAVE/hf_epoch_0" ] || [ -d "$SAVE/state_0" ]; then
  echo "[skip] $TAG already trained"
else
  echo "========== TRAIN $TAG (sample_rsd + dense_tv + n_anchors=4 + KL=0.1) =========="
  "$DS_BIN" --master_port $((PORT_BASE+0)) --include "localhost:$GPUS" \
    sdpo/main_small_lm_tree_v2.py \
    --basepath "$BASE" --draftpath "$DRAFT" \
    --trainpath "$TRAIN" --testpath "$VAL" \
    --deepspeed_config "$DS_TREE" --savedir "$SAVE" \
    --top_k_per_depth 4,3,2,1,1,1,1 --tree_budget 0 \
    --eal_mode sample_rsd \
    --n_anchors 4 --path_topk 4 --temperature 1.0 \
    --anchor_kl_coef 0.1 --eal_aux_coef 1.0 \
    --lr "$LR" --max_len "$MAX_LEN" --max_train_samples "$MAX_SAMPLES" \
    --num_epochs "$EPOCHS" \
    > "$LOG/${TAG}_train.log" 2>&1
  echo "[train] $TAG exit=$?"
fi

# ── 2) Sample Chain EAL: position-wise chain EAL, α=min(p,q), no siblings ─
TAG=sample_chain_eal
SAVE=$SAVE_ROOT/$TAG
rm -rf "$SAVE"
echo "========== TRAIN $TAG (al_tv chain EAL from positions) =========="
"$DS_BIN" --master_port $((PORT_BASE+1)) --include "localhost:$GPUS" \
  sdpo/main_small_lm.py \
  --basepath "$BASE" --draftpath "$DRAFT" \
  --trainpath "$TRAIN" --testpath "$VAL" \
  --deepspeed_config "$DS_CHAIN" --savedir "$SAVE" \
  --anchor none --aux_loss al_tv --sigmoid_coef 1.0 \
  --anchor_weight none --aux_weight uniform \
  --T_max 0.1 --T_min 0.1 \
  --prob_T_max 1.0 --prob_T_min 1.0 \
  --gamma 7 --lr "$LR" --max_len "$MAX_LEN" \
  --max_train_samples "$MAX_SAMPLES" --num_epochs "$EPOCHS" \
  > "$LOG/${TAG}_train.log" 2>&1
echo "[train] $TAG exit=$?"

# ── 3) KL: per-position KL(target||draft) ────────────────────────────────
TAG=kl_only
SAVE=$SAVE_ROOT/$TAG
rm -rf "$SAVE"
echo "========== TRAIN $TAG (per-position KL) =========="
"$DS_BIN" --master_port $((PORT_BASE+2)) --include "localhost:$GPUS" \
  sdpo/main_small_lm.py \
  --basepath "$BASE" --draftpath "$DRAFT" \
  --trainpath "$TRAIN" --testpath "$VAL" \
  --deepspeed_config "$DS_CHAIN" --savedir "$SAVE" \
  --anchor kl --aux_loss none --sigmoid_coef 0.0 \
  --anchor_weight none --aux_weight pow08 \
  --T_max 0.1 --T_min 0.1 \
  --gamma 7 --lr "$LR" --max_len "$MAX_LEN" \
  --max_train_samples "$MAX_SAMPLES" --num_epochs "$EPOCHS" \
  > "$LOG/${TAG}_train.log" 2>&1
echo "[train] $TAG exit=$?"

# ── Eval all on tree ─────────────────────────────────────────────────────
for TAG in sample_tree_eal sample_chain_eal kl_only; do
  eval_tree "$TAG" || true
done

echo "========== SUMMARY =========="
"$PY" - <<'PY'
import json, glob, os
root = "/home/ubuntu/workspace/SD/Cerebras_SD"
print(f"{'method':<20} {'mean_alpha':>12} {'tok/s':>10}")
rows=[]
for tag in ["sample_tree_eal","sample_chain_eal","kl_only"]:
    paths=sorted(glob.glob(f"{root}/checkpoints/{tag}/eval/{tag}_tree_rsd*.json"))
    if not paths:
        print(f"{tag:<20} {'NA':>12} {'NA':>10}"); continue
    r=json.load(open(paths[-1])).get("results",{}).get("mt_bench",{}).get("tree",{})
    a,t=r.get("mean_alpha"), r.get("tokens_per_sec")
    rows.append((tag,a,t))
    aa=f"{a:.4f}" if isinstance(a,(int,float)) else "NA"
    tt=f"{t:.2f}" if isinstance(t,(int,float)) else "NA"
    print(f"{tag:<20} {aa:>12} {tt:>10}")
best=[r for r in rows if isinstance(r[1],(int,float))]
if best:
    w=max(best, key=lambda x:x[1])
    print(f"\nBest on tree (mean accept length): {w[0]} = {w[1]:.4f}")
PY
