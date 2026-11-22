#!/bin/bash
# 1-epoch comparison: Sample Tree EAL / Sample Chain EAL / KL
# Eval all on tree with sample + rsd (sibling residual verify).
set -uo pipefail

REPO=${REPO:-/home/ubuntu/workspace/SD/Cerebras_SD}
ROOT=${ROOT:-$REPO}
PY=${PY:-python3}
DS_BIN=${DS_BIN:-deepspeed}
GPUS=${GPUS:-0,1}
PORT_BASE=${PORT_BASE:-29551}
LR=${LR:-1e-6}
EPOCHS=${EPOCHS:-1}
MAX_SAMPLES=${MAX_SAMPLES:-512}
MAX_LEN=${MAX_LEN:-1024}
NUM_EVAL=${NUM_EVAL:-40}

BASE=$ROOT/models/Qwen3-8B
DRAFT=$ROOT/models/Qwen3-0.6B
TRAIN=$ROOT/sdpo/data/mixed_train.jsonl
VAL=$ROOT/sdpo/data/mixed_val.jsonl
DS=$REPO/sdpo/scripts/tree_eal/ds_zero2_a100_2gpu.json
SAVE_ROOT=$ROOT/checkpoints
LOG=$ROOT/logs

export HF_HOME=${HF_HOME:-$ROOT/hf}
export WANDB_MODE=offline
export WANDB_DIR=$LOG/wandb
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$LOG" "$SAVE_ROOT"
cd "$REPO"

run_one () {
  local TAG=$1; shift
  local SAVE=$SAVE_ROOT/$TAG
  local PORT=$1; shift
  mkdir -p "$SAVE/eval"
  echo "========== TRAIN $TAG =========="
  "$DS_BIN" --master_port "$PORT" --include "localhost:$GPUS" \
    sdpo/main_small_lm_tree_v2.py \
    --basepath "$BASE" \
    --draftpath "$DRAFT" \
    --trainpath "$TRAIN" \
    --testpath "$VAL" \
    --deepspeed_config "$DS" \
    --savedir "$SAVE" \
    --top_k_per_depth 4,3,2,1,1,1,1 \
    --tree_budget 0 \
    --temperature 1.0 \
    --path_topk 4 \
    --lr "$LR" \
    --max_len "$MAX_LEN" \
    --max_train_samples "$MAX_SAMPLES" \
    --num_epochs "$EPOCHS" \
    "$@" \
    > "$LOG/${TAG}_train.log" 2>&1
  local rc=$?
  echo "[train] $TAG exit=$rc"
  return $rc
}

eval_one () {
  local TAG=$1
  local SAVE=$SAVE_ROOT/$TAG
  # Prefer hf_epoch_* then state_*
  local CKPT=""
  for d in "$SAVE"/hf_epoch_*; do
    [ -d "$d" ] && CKPT=$d
  done
  if [ -z "$CKPT" ]; then
    for d in "$SAVE"/state_*; do
      [ -d "$d" ] && CKPT=$d
    done
  fi
  if [ -z "$CKPT" ] || [ ! -d "$CKPT" ]; then
    echo "[eval] $TAG: no checkpoint under $SAVE"
    return 1
  fi
  echo "========== EVAL $TAG ckpt=$CKPT =========="
  CUDA_VISIBLE_DEVICES=${EVAL_GPU:-2} "$PY" sdpo/eval_small_lm_tree.py \
    --base-model-path "$BASE" --draft-model-path "$CKPT" \
    --bench-name mt_bench --gamma 7 --top-k 4,3,2,1,1,1,1 --budget 128 \
    --max-new-tokens 256 --num-samples "$NUM_EVAL" --temperature 1.0 --seed 0 \
    --draft-mode sample --verify-mode rsd --path-topk 4 \
    --tag "${TAG}_tree_rsd" --output-dir "$SAVE/eval" \
    > "$LOG/${TAG}_eval.log" 2>&1
  local rc=$?
  echo "[eval] $TAG exit=$rc"
  "$PY" - <<PY
import json, glob, os
paths = glob.glob("$SAVE/eval/${TAG}_tree_rsd*.json")
print("eval files:", paths)
for p in paths:
    j = json.load(open(p))
    r = j.get("results", {}).get("mt_bench", {}).get("tree", {})
    print(f"[$TAG] mean_alpha={r.get('mean_alpha')} tokens_per_sec={r.get('tokens_per_sec')}")
PY
  return $rc
}

# 1) Sample Tree EAL: sample_rsd + dense_tv + n_anchors=4 + KL=0.1 (siblings / RSD)
run_one sample_tree_eal $((PORT_BASE+0)) \
  --eal_mode sample_rsd \
  --n_anchors 4 \
  --anchor_kl_coef 0.1 \
  --eal_aux_coef 1.0

# 2) Sample Chain EAL: sample_minpq (path-sum, no residual sibling)
run_one sample_chain_eal $((PORT_BASE+1)) \
  --eal_mode sample_minpq \
  --n_anchors 4 \
  --anchor_kl_coef 0.1 \
  --eal_aux_coef 1.0

# 3) KL only on the same sampled-tree setup
run_one kl_only $((PORT_BASE+2)) \
  --eal_mode sample_rsd \
  --n_anchors 4 \
  --anchor_kl_coef 1.0 \
  --eal_aux_coef 0.0

for TAG in sample_tree_eal sample_chain_eal kl_only; do
  eval_one "$TAG" || true
done

echo "========== SUMMARY =========="
"$PY" - <<'PY'
import json, glob, os
root = os.environ.get("ROOT", "/home/ubuntu/workspace/SD/Cerebras_SD")
rows = []
for tag in ["sample_tree_eal", "sample_chain_eal", "kl_only"]:
    paths = sorted(glob.glob(f"{root}/checkpoints/{tag}/eval/{tag}_tree_rsd*.json"))
    if not paths:
        rows.append((tag, None, None))
        continue
    j = json.load(open(paths[-1]))
    r = j.get("results", {}).get("mt_bench", {}).get("tree", {})
    rows.append((tag, r.get("mean_alpha"), r.get("tokens_per_sec")))
print(f"{'method':<20} {'mean_alpha':>12} {'tok/s':>10}")
for tag, a, t in rows:
    aa = f"{a:.4f}" if isinstance(a, (int, float)) else "NA"
    tt = f"{t:.2f}" if isinstance(t, (int, float)) else "NA"
    print(f"{tag:<20} {aa:>12} {tt:>10}")
best = [r for r in rows if isinstance(r[1], (int, float))]
if best:
    winner = max(best, key=lambda x: x[1])
    print(f"\nBest on tree (mean accept length): {winner[0]} = {winner[1]:.4f}")
PY
