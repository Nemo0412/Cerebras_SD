#!/bin/bash
# ALTV + GRPO continuation training from existing ALTV ckpt.
# Pure ALTV+GRPO (NO KL anchor): anchor=none, aux=al_tv, sigmoid_coef=1.0,
# grpo_coef=0.1 (sample mode, hard reward) — matches KLV4 GRPO settings except
# anchor.
#
# After training: 3-mode × 5-bench eval (γ=7 b=128) in same slurm.
#
# Required env:
#   TAG          — output tag (e.g. q8_q06_altv_grpo_v1)
#   BASEPATH     — target HF id
#   DRAFTPATH    — source ALTV state_5 ckpt path
#   SAVEROOT     — train save root (e.g. .../loss_train_smalllm_06b)
#   TRAINPATH    — training jsonl
#   DEEPSPEED_CONFIG (optional, default Qwen3 ZeRO-2)

#SBATCH --job-name=altv_grpo
#SBATCH --time=1-23:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=192G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --constraint="h100|h200"
#SBATCH --requeue
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tx856@nyu.edu
#SBATCH --output=logs/altv_grpo/%x_%j.out
#SBATCH --error=logs/altv_grpo/%x_%j.err

REPO=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
cd "$REPO"
mkdir -p logs/altv_grpo smalllm_tree_eval_results

source ~/.bashrc
conda activate ${CONDA_ENV:-longreason}

export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/scratch/tx856/.cache/torch_extensions}
export TRITON_CACHE_DIR=/scratch/tx856/.cache/triton
export HF_HOME=${HF_HOME:-/scratch/tx856/.huggingface}

: ${TAG:?must set TAG}
: ${BASEPATH:?must set BASEPATH}
: ${DRAFTPATH:?must set DRAFTPATH (source ALTV ckpt path)}
: ${SAVEROOT:?must set SAVEROOT}
: ${TRAINPATH:?must set TRAINPATH}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-sdpo/sdpo_config_qwen3.json}

echo "=============================="
echo "  ALTV + GRPO train + eval: $TAG"
echo "  Target:    $BASEPATH"
echo "  Source ckpt: $DRAFTPATH"
echo "  Save root: $SAVEROOT/$TAG"
echo "  DS config: $DEEPSPEED_CONFIG"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

# ────── TRAIN ──────
MASTER_PORT=$((29500 + RANDOM % 1000))

deepspeed --master_port $MASTER_PORT sdpo/main_small_lm.py \
    --basepath "$BASEPATH" \
    --draftpath "$DRAFTPATH" \
    --trainpath "$TRAINPATH" \
    --testpath sdpo/data/mixed_val_80.jsonl \
    --deepspeed_config "$DEEPSPEED_CONFIG" \
    --savedir "$SAVEROOT/$TAG" \
    --anchor none \
    --aux_loss al_tv \
    --anchor_weight none \
    --aux_weight pow08 \
    --sigmoid_coef 1.0 \
    --T_max 0.1 --T_min 0.1 \
    --prob_T_max 1.0 --prob_T_min 0.1 \
    --draft_T_max 1.0 --draft_T_min 1.0 \
    --eal_weight 0.5 \
    --grpo_coef 0.1 \
    --grpo_k_groups 8 --grpo_m 4 --grpo_eps 0.2 \
    --grpo_mode sample \
    --grpo_sample_temp 1.0 \
    --grpo_reward hard \
    --gamma 7 \
    --max_len 2048 \
    --num_epochs 3 \
    --max_train_samples 60000 \
    --seed 42 \
    --lr 1e-6
TRAIN_RC=$?

echo ""
echo "=============================="
echo "  Train done rc=$TRAIN_RC"
echo "=============================="

if [ $TRAIN_RC -ne 0 ]; then
    echo "[EVAL] skipped: train failed (rc=$TRAIN_RC)"
    exit $TRAIN_RC
fi

# Find final ckpt: prefer state_2 (3 epochs → last epoch). Fallback to newest state_*
CKPT="$SAVEROOT/$TAG/state_2"
if [ ! -f "$CKPT/pytorch_model.bin" ]; then
    CKPT=$(ls -d "$SAVEROOT/$TAG"/state_* 2>/dev/null | sort -V | tail -1)
    echo "[EVAL] state_2 not found, fallback to newest: $CKPT"
fi

# ────── EVAL ──────
OUT=smalllm_tree_eval_results
BENCHES="mt_bench,gsm8k,humaneval,longwriter_top50,osr2_top50"

declare -a MODES=(
    "greedy         --temperature 0.0 --draft-mode argmax --verify-mode auto"
    "samp_t05_ratio --temperature 0.5 --draft-mode sample --verify-mode ratio"
    "samp_t1_ratio  --temperature 1.0 --draft-mode sample --verify-mode ratio"
)

echo ""
echo "=============================="
echo "  EVAL: $TAG  ckpt=$CKPT"
echo "  Benches: $BENCHES"
echo "  γ=7  top_k=4,3,2,1,1,1,1  budget=128"
echo "=============================="

for MODE_LINE in "${MODES[@]}"; do
    read -r MODE_SUF MODE_ARGS <<< "$MODE_LINE"
    FULL_TAG="${TAG}_${MODE_SUF}"
    OUT_FILE="$OUT/${FULL_TAG}.json"
    if [ -f "$OUT_FILE" ]; then
        echo "[EVAL] SKIP $FULL_TAG (done)"
        continue
    fi
    echo ""
    echo "==== EVAL $MODE_SUF → $FULL_TAG ===="
    set +e
    python sdpo/eval_small_lm_tree.py \
        --base-model-path "$BASEPATH" \
        --draft-model-path "$CKPT" \
        --bench-name "$BENCHES" \
        --gamma 7 --top-k "4,3,2,1,1,1,1" --budget 128 \
        --max-new-tokens 256 \
        --num-samples 40 \
        --output-dir "$OUT" \
        --seed 0 \
        --tag "$FULL_TAG" \
        $MODE_ARGS
    RC=$?
    set -e
    echo "  rc=$RC"
done

echo ""
echo "=============================="
echo "  ALL DONE  $(date)"
echo "=============================="
