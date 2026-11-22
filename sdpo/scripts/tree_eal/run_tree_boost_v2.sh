#!/bin/bash
# Tree boost v2 — FAIR vs Chain, sample_rsd dense residual TV (no RL):
#   same init Qwen3-0.6B, same data, max_len=1024, 2 ep, lr=1e-6, seed=0
#   loss = Σ_v min(u,q) under RSD residual (Chain al_tv analogue on the tree)
set -uo pipefail
cd /home/ubuntu/workspace/SD/Cerebras_SD
LOG=logs/run_tree_boost_v2.log
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
PORT=29930
TAG=fast_tree_eal_v2

echo "[boost] start $(date -Is) FAIR dense_tv (no RL)" | tee "$LOG"
echo "[boost] draft=$DRAFT grad=dense_tv n_anchors=3 lr=$LR ep=$EPOCHS" | tee -a "$LOG"

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
  --n_anchors 3 --anchor_kl_coef 0.1 --eal_aux_coef 1.0 \
  --eal_score_coef 0.0 --n_draws 1 \
  > logs/${TAG}_train.log 2>&1
EC=$?
echo "[train] $TAG exit=$EC" | tee -a "$LOG"
if [ $EC -ne 0 ]; then
  echo "[boost] n_anchors=3 failed; retry n_anchors=2" | tee -a "$LOG"
  rm -rf checkpoints/$TAG
  deepspeed --master_port $((PORT+1)) --include "localhost:$GPUS" \
    sdpo/main_small_lm_tree_v2.py \
    --basepath "$BASE" --draftpath "$DRAFT" \
    --trainpath "$TRAIN" --testpath "$VAL" \
    --deepspeed_config "$DS" --savedir checkpoints/$TAG \
    --top_k_per_depth 4,3,2,1,1,1,1 --tree_budget 0 \
    --temperature 1.0 --path_topk 4 --lr "$LR" \
    --max_len "$MAX_LEN" --num_epochs "$EPOCHS" --seed "$SEED" \
    --eal_mode sample_rsd \
    --n_anchors 2 --anchor_kl_coef 0.1 --eal_aux_coef 1.0 \
    --eal_score_coef 0.0 --n_draws 1 \
    > logs/${TAG}_train.log 2>&1
  echo "[train] $TAG retry exit=$?" | tee -a "$LOG"
fi

CKPT=checkpoints/$TAG/hf_epoch_$((EPOCHS-1))
[ -d "$CKPT" ] || CKPT=$(ls -d checkpoints/$TAG/hf_epoch_* 2>/dev/null | sort | tail -1)
echo "[boost] eval ckpt=$CKPT" | tee -a "$LOG"
[ -d "$CKPT" ] || { echo "no ckpt"; exit 1; }

mkdir -p checkpoints/$TAG/eval_tree
CUDA_VISIBLE_DEVICES=0 python3 sdpo/eval_small_lm_tree.py \
  --base-model-path "$BASE" --draft-model-path "$CKPT" \
  --bench-name mt_bench,gsm8k,humaneval \
  --gamma 7 --top-k 4,3,2,1,1,1,1 --budget 128 \
  --max-new-tokens 256 --num-samples 80 --temperature 1.0 --seed 0 \
  --draft-mode sample --verify-mode rsd --path-topk 4 \
  --tag ${TAG}_tree_rsd --output-dir checkpoints/$TAG/eval_tree \
  > logs/${TAG}_tree_eval.log 2>&1
echo "[eval] $TAG exit=$?" | tee -a "$LOG"

python3 - <<'PY' | tee -a logs/run_tree_boost_v2.log
import json, statistics
benches = ["mt_bench", "gsm8k", "humaneval"]
def load(tag):
    p = f"checkpoints/{tag}/eval_tree/{tag}_tree_rsd.json"
    try: res = json.load(open(p))["results"]
    except Exception as e:
        print(f"{tag}: missing ({e})"); return None
    return {b: res.get(b, {}).get("tree", {}).get("mean_alpha") for b in benches}
print("=== FAIR tree RSD (dense_tv loss, no RL) ===")
print(f"{'method':<18}" + "".join(f"{b:>10}" for b in benches) + f"{'avg':>10}")
rows = []
for tag, name in [("fast_tree_eal_v2","Tree dense_tv"),
                  ("fast_tree_eal","Tree v1"),
                  ("fast_chain_eal","Chain al_tv")]:
    d = load(tag)
    if not d: print(f"{name:<18} NA"); continue
    alphas = [d[b] for b in benches if isinstance(d[b], (int, float))]
    cells = [f"{d[b]:.4f}" if isinstance(d[b], (int, float)) else "NA" for b in benches]
    avg = statistics.mean(alphas) if len(alphas)==3 else None
    print(f"{name:<18}" + "".join(f"{c:>10}" for c in cells) + f"{(f'{avg:.4f}' if avg else 'NA'):>10}")
    if avg is not None: rows.append((name, avg))
by = dict(rows)
if "Tree dense_tv" in by and "Chain al_tv" in by:
    print(f"delta (Tree dense_tv - Chain) = {by['Tree dense_tv']-by['Chain al_tv']:+.4f}")
PY
echo "[boost] done $(date -Is)" | tee -a "$LOG"