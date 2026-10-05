#!/bin/bash
# Generic parameterized train+eval slurm.
# Env: TAG, BASEPATH, DRAFTPATH, TRAINPATH, SAVEROOT, ANCHOR, AUX_LOSS,
#      SIGMOID_COEF, NUM_EPOCHS, SEED, DS_CONFIG,
#      GRPO_COEF (default 0), GRPO_MODE (default window), GRPO_REWARD (default hard)

#SBATCH --job-name=trainevalgen
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
#SBATCH --output=logs/train_eval_generic/%x_%j.out
#SBATCH --error=logs/train_eval_generic/%x_%j.err

REPO=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
cd "$REPO"
mkdir -p logs/train_eval_generic smalllm_tree_eval_results

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
: ${SEED:?must set SEED}
SIGMOID_COEF=${SIGMOID_COEF:-0.1}
NUM_EPOCHS=${NUM_EPOCHS:-3}
DS_CONFIG=${DS_CONFIG:-sdpo/sdpo_config_qwen3.json}
GRPO_COEF=${GRPO_COEF:-0.0}
GRPO_MODE=${GRPO_MODE:-window}
GRPO_REWARD=${GRPO_REWARD:-hard}
TOP_K_PER_DEPTH=${TOP_K_PER_DEPTH:-""}
# sbatch --export comma-split bug: TOP_K_PER_DEPTH="4,3,2,1,1,1,1" gets
# truncated to "4". Accept underscore-encoded TOP_K_PER_DEPTH_ENC="4_3_2_1_1_1_1"
# as fallback and convert to comma inside script.
TOP_K_PER_DEPTH_ENC=${TOP_K_PER_DEPTH_ENC:-""}
if [ -z "$TOP_K_PER_DEPTH" ] && [ -n "$TOP_K_PER_DEPTH_ENC" ]; then
    TOP_K_PER_DEPTH="${TOP_K_PER_DEPTH_ENC//_/,}"
fi
COVERAGE_LAMBDA=${COVERAGE_LAMBDA:-0.5}
WK_BONUS=${WK_BONUS:-1.0}
TREE_ALTV_MIX=${TREE_ALTV_MIX:-0.1}
TREE_ARGS=""
if [ -n "$TOP_K_PER_DEPTH" ]; then
    TREE_ARGS="--top_k_per_depth $TOP_K_PER_DEPTH --coverage_lambda $COVERAGE_LAMBDA --wk_bonus $WK_BONUS --tree_altv_mix $TREE_ALTV_MIX"
fi

echo "============================================"
echo "  Train+Eval: $TAG (seed=$SEED)"
echo "  Target: $BASEPATH   Draft init: $DRAFTPATH"
echo "  Anchor: $ANCHOR   Aux: $AUX_LOSS   coef=$SIGMOID_COEF"
echo "  GRPO: coef=$GRPO_COEF mode=$GRPO_MODE reward=$GRPO_REWARD"
echo "  Epochs=$NUM_EPOCHS   Data=$TRAINPATH"
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
    --grpo_coef "$GRPO_COEF" \
    --grpo_k_groups 8 --grpo_m 4 --grpo_eps 0.2 \
    --grpo_mode "$GRPO_MODE" --grpo_sample_temp 1.0 --grpo_reward "$GRPO_REWARD" \
    --gamma 7 --max_len 2048 \
    --num_epochs "$NUM_EPOCHS" \
    --seed "$SEED" --lr 1e-6 \
    --max_train_samples 60000 \
    $TREE_ARGS
TRAIN_RC=$?
echo "Train done rc=$TRAIN_RC"
if [ $TRAIN_RC -ne 0 ]; then echo "[EVAL] skipped"; exit $TRAIN_RC; fi

CKPT="$SAVEROOT/$TAG/state_$((NUM_EPOCHS - 1))"
if [ ! -f "$CKPT/pytorch_model.bin" ]; then
    CKPT=$(ls -d "$SAVEROOT/$TAG"/state_* 2>/dev/null | sort -V | tail -1)
fi

OUT=smalllm_tree_eval_results
BENCHES_3="mt_bench,gsm8k,humaneval"
BENCHES_LW="longwriter_top50,osr2_top50"

# Chain γ=7: 3-bench × 3 modes × 3 eval seeds
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

# Tree γ=7: 3-bench + LW/OSR2 × 3 modes
declare -a TREE_MODES=(
    "greedy               --temperature 0.0 --draft-mode argmax --verify-mode auto"
    "samp_t05_smp_ratio   --temperature 0.5 --draft-mode sample --verify-mode ratio"
    "samp_t1_smp_ratio    --temperature 1.0 --draft-mode sample --verify-mode ratio"
)
for MODE_LINE in "${TREE_MODES[@]}"; do
    read -r MODE_SUF MODE_ARGS <<< "$MODE_LINE"
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
