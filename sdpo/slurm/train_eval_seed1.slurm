#!/bin/bash
# Parameterized train+eval slurm for seed-1 CE/ALTV retraining.
# Required env: TAG, BASEPATH, DRAFTPATH, TRAINPATH, SAVEROOT, ANCHOR, AUX_LOSS,
# SIGMOID_COEF, NUM_EPOCHS, DS_CONFIG.

#SBATCH --job-name=trainevalseed1
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
#SBATCH --output=logs/train_eval_seed1/%x_%j.out
#SBATCH --error=logs/train_eval_seed1/%x_%j.err

REPO=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
cd "$REPO"
mkdir -p logs/train_eval_seed1 smalllm_tree_eval_results

source ~/.bashrc
conda activate ${CONDA_ENV:-longreason}
set -u

export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/scratch/tx856/.cache/torch_extensions}
export TRITON_CACHE_DIR=/scratch/tx856/.cache/triton
export HF_HOME=${HF_HOME:-/scratch/tx856/.huggingface}

: ${TAG:?must set TAG}
: ${BASEPATH:?must set BASEPATH}
: ${DRAFTPATH:?must set DRAFTPATH}
: ${TRAINPATH:?must set TRAINPATH}
: ${SAVEROOT:?must set SAVEROOT}
: ${ANCHOR:?must set ANCHOR}
: ${AUX_LOSS:?must set AUX_LOSS}
SIGMOID_COEF=${SIGMOID_COEF:-0.1}
NUM_EPOCHS=${NUM_EPOCHS:-3}
DS_CONFIG=${DS_CONFIG:-sdpo/sdpo_config_qwen3.json}
SAVE_EPOCHS_FLAG=""
if [ -n "${SAVE_EPOCHS:-}" ]; then
    SAVE_EPOCHS_FLAG="--save_epochs $SAVE_EPOCHS"
fi

echo "============================================"
echo "  Train+Eval seed=1: $TAG"
echo "  Target: $BASEPATH   Draft: $DRAFTPATH"
echo "  Anchor: $ANCHOR   Aux: $AUX_LOSS   coef=$SIGMOID_COEF"
echo "  Epochs=$NUM_EPOCHS  Data=$TRAINPATH"
echo "  Save: $SAVEROOT/$TAG"
echo "  DS config: $DS_CONFIG   save_epochs=$SAVE_EPOCHS_FLAG"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "============================================"

MASTER_PORT=$((29500 + RANDOM % 1000))

deepspeed --master_port $MASTER_PORT sdpo/main_small_lm.py \
    --basepath "$BASEPATH" \
    --draftpath "$DRAFTPATH" \
    --trainpath "$TRAINPATH" \
    --testpath sdpo/data/mixed_val_80.jsonl \
    --deepspeed_config "$DS_CONFIG" \
    --savedir "$SAVEROOT/$TAG" \
    --anchor "$ANCHOR" \
    --aux_loss "$AUX_LOSS" \
    --anchor_weight none \
    --aux_weight pow08 \
    --sigmoid_coef "$SIGMOID_COEF" \
    --T_max 0.1 --T_min 0.1 \
    --prob_T_max 1.0 --prob_T_min 0.1 \
    --draft_T_max 1.0 --draft_T_min 1.0 \
    --eal_weight 0.5 \
    --grpo_coef 0.0 \
    --grpo_k_groups 8 --grpo_m 4 --grpo_eps 0.2 \
    --grpo_mode window --grpo_sample_temp 1.0 --grpo_reward hard \
    --gamma 7 --max_len 2048 \
    --num_epochs "$NUM_EPOCHS" \
    --seed 1 --lr 1e-6 \
    --max_train_samples 60000 \
    $SAVE_EPOCHS_FLAG
TRAIN_RC=$?
echo "Train done rc=$TRAIN_RC"
if [ $TRAIN_RC -ne 0 ]; then
    echo "[EVAL] skipped: train failed"; exit $TRAIN_RC
fi

# Find final ckpt
CKPT="$SAVEROOT/$TAG/state_$((NUM_EPOCHS - 1))"
if [ ! -f "$CKPT/pytorch_model.bin" ]; then
    CKPT=$(ls -d "$SAVEROOT/$TAG"/state_* 2>/dev/null | sort -V | tail -1)
    echo "[EVAL] state_$((NUM_EPOCHS-1)) not found, fallback to $CKPT"
fi

OUT=smalllm_tree_eval_results
BENCHES_3="mt_bench,gsm8k,humaneval"
BENCHES_LW="longwriter_top50,osr2_top50"

# ────── Chain γ=7 eval (3-bench × 3 modes × 3 eval seeds) ──────
declare -a CHAIN_MODES=(
    "greedy         --temperature 0.0 --draft-mode argmax --verify-mode auto"
    "samp_t05_ratio --temperature 0.5 --draft-mode sample --verify-mode ratio"
    "samp_t1_ratio  --temperature 1.0 --draft-mode sample --verify-mode ratio"
)
for MODE_LINE in "${CHAIN_MODES[@]}"; do
    read -r MODE_SUF MODE_ARGS <<< "$MODE_LINE"
    for ESEED in 0 1 2; do
        FT="${TAG}_eseed${ESEED}_chain_g7_${MODE_SUF}"
        [ -f "$OUT/$FT.json" ] && { echo "[SKIP] $FT"; continue; }
        echo ""; echo "=== chain $FT ==="
        set +e
        python sdpo/eval_small_lm_tree.py \
            --base-model-path "$BASEPATH" --draft-model-path "$CKPT" \
            --bench-name "$BENCHES_3" --max-new-tokens 256 --num-samples 40 \
            --output-dir "$OUT" --seed "$ESEED" --tag "$FT" \
            --baseline-only --baseline-gamma 7 \
            $MODE_ARGS
        set -e
    done
done

# ────── Tree γ=7 eval (3-bench + LW/OSR2, think mode default) ──────
declare -a TREE_MODES=(
    "greedy               --temperature 0.0 --draft-mode argmax --verify-mode auto"
    "samp_t05_smp_ratio   --temperature 0.5 --draft-mode sample --verify-mode ratio"
    "samp_t1_smp_ratio    --temperature 1.0 --draft-mode sample --verify-mode ratio"
)
for MODE_LINE in "${TREE_MODES[@]}"; do
    read -r MODE_SUF MODE_ARGS <<< "$MODE_LINE"
    # 3-bench
    FT="${TAG}_${MODE_SUF}"
    if [ ! -f "$OUT/$FT.json" ]; then
        echo ""; echo "=== tree $FT ==="
        set +e
        python sdpo/eval_small_lm_tree.py \
            --base-model-path "$BASEPATH" --draft-model-path "$CKPT" \
            --bench-name "$BENCHES_3" --gamma 7 --top-k "4,3,2,1,1,1,1" --budget 128 \
            --max-new-tokens 256 --num-samples 40 --output-dir "$OUT" \
            --seed 0 --tag "$FT" $MODE_ARGS
        set -e
    fi
    # LW/OSR2
    FT_LW="${TAG}_lwosr2_${MODE_SUF}"
    if [ ! -f "$OUT/$FT_LW.json" ]; then
        echo ""; echo "=== tree $FT_LW ==="
        set +e
        python sdpo/eval_small_lm_tree.py \
            --base-model-path "$BASEPATH" --draft-model-path "$CKPT" \
            --bench-name "$BENCHES_LW" --gamma 7 --top-k "4,3,2,1,1,1,1" --budget 128 \
            --max-new-tokens 256 --num-samples 40 --output-dir "$OUT" \
            --seed 0 --tag "$FT_LW" $MODE_ARGS
        set -e
    fi
done

echo ""
echo "=========== ALL DONE $(date) ==========="
