#!/bin/bash
# Tree EAL (sampled tree, alpha = min(p, q)) on Qwen3-8B-generated replies.
#
# Loss  : -E[L],  E[L] = sum over tree nodes of prod_{v on path} min(p_v, q_v)
#         p = frozen target prob of the drafted token, q = draft prob (gets grad).
# Tree  : draft SAMPLES k distinct children per node (top_k_per_depth), T=1.
# Eval  : --draft-mode sample --verify-mode ratio (must match the training rule).
#
# Usage:
#   ROOT=/path/to/workdir bash sdpo/scripts/tree_eal/train_tree_eal_minpq.sh
# Expects under $ROOT:
#   models/qwen3-8b, models/Qwen3-0.6B
#   data/alpaca_qwen3_8b_regen.jsonl   (from gen_alpaca_regen.py)
#   data/mixed_val_80.jsonl            (any small ShareGPT-format val file)
set -uo pipefail

ROOT=${ROOT:-/mnt/ssd_ext/lls/cerebras_sd}
REPO=${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
PY=${PY:-python}
DS_BIN=${DS_BIN:-deepspeed}
GPUS=${GPUS:-0,1,2}            # training GPUs
EVAL_GPU=${EVAL_GPU:-3}        # per-epoch mt_bench eval
LR=${LR:-1e-6}
EPOCHS=${EPOCHS:-3}
PORT=${PORT:-29551}
TAG=${TAG:-tree_eal_minpq}

TRAIN=$ROOT/data/alpaca_qwen3_8b_regen.jsonl
VAL=$ROOT/data/mixed_val_80.jsonl
BASE=$ROOT/models/qwen3-8b
DRAFT=$ROOT/models/Qwen3-0.6B
DS=$REPO/sdpo/scripts/tree_eal/ds_zero2_a6000_3gpu.json
SAVE=$ROOT/checkpoints/$TAG
LOG=$ROOT/logs

export HF_HOME=${HF_HOME:-$ROOT/hf}
export WANDB_MODE=offline
export WANDB_DIR=$LOG/wandb
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$LOG" "$SAVE/eval"
cd "$REPO"

echo "[train] $TAG lr=$LR epochs=$EPOCHS gpus=$GPUS"
"$DS_BIN" --master_port "$PORT" --include "localhost:$GPUS" sdpo/main_small_lm_tree_v2.py \
  --basepath "$BASE" \
  --draftpath "$DRAFT" \
  --trainpath "$TRAIN" \
  --testpath "$VAL" \
  --deepspeed_config "$DS" \
  --savedir "$SAVE" \
  --top_k_per_depth 4,3,2,1,1,1,1 \
  --tree_budget 128 \
  --eal_mode sample_minpq \
  --temperature 1.0 \
  --anchor_kl_coef 0 \
  --eal_aux_coef 0 \
  --lr "$LR" \
  --max_len 2048 \
  --num_epochs "$EPOCHS" \
  > "$LOG/${TAG}_train.log" 2>&1
echo "[train] exit=$?"

# mt_bench tree eval of every saved epoch, same protocol as the baselines.
for d in "$SAVE"/hf_epoch_*; do
  [ -d "$d" ] || continue
  n=$(basename "$d")
  CUDA_VISIBLE_DEVICES=$EVAL_GPU "$PY" sdpo/eval_small_lm_tree.py \
    --base-model-path "$BASE" --draft-model-path "$d" \
    --bench-name mt_bench --gamma 7 --top-k 4,3,2,1,1,1,1 --budget 128 \
    --max-new-tokens 256 --num-samples 80 --temperature 1.0 --seed 0 \
    --draft-mode sample --verify-mode ratio \
    --tag "${TAG}_${n}_sr" --output-dir "$SAVE/eval" > "$LOG/${TAG}_${n}_eval.log" 2>&1
  "$PY" -c "import json;print('[eval] $n mt_bench accept length', json.load(open('$SAVE/eval/${TAG}_${n}_sr.json'))['results']['mt_bench']['tree']['mean_alpha'])"
done
