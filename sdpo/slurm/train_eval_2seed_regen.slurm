#!/bin/bash
# Train CE with regen data on 2 seeds (0 and 1) in one slurm job + full eval.
# Env: TAG_BASE, BASEPATH, DRAFTPATH, TRAINPATH, SAVEROOT, DS_CONFIG, NUM_EPOCHS

#SBATCH --job-name=ce_regen_2seed
#SBATCH --time=2-23:59:00
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
#SBATCH --output=logs/train_eval_2seed/%x_%j.out
#SBATCH --error=logs/train_eval_2seed/%x_%j.err

REPO=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
cd "$REPO"
mkdir -p logs/train_eval_2seed smalllm_tree_eval_results

source ~/.bashrc
conda activate ${CONDA_ENV:-longreason}
set -u

export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/scratch/tx856/.cache/torch_extensions}
export TRITON_CACHE_DIR=/scratch/tx856/.cache/triton
export HF_HOME=${HF_HOME:-/scratch/tx856/.huggingface}

: ${TAG_BASE:?must set TAG_BASE (e.g. q17_q06_ce_regen_l2k_6ep)}
: ${BASEPATH:?must set BASEPATH}
: ${DRAFTPATH:?must set DRAFTPATH}
: ${TRAINPATH:?must set TRAINPATH}
: ${SAVEROOT:?must set SAVEROOT}
NUM_EPOCHS=${NUM_EPOCHS:-6}
DS_CONFIG=${DS_CONFIG:-sdpo/sdpo_config_qwen3.json}

echo "============================================"
echo "  CE 2-seed regen train+eval: $TAG_BASE"
echo "  Target: $BASEPATH   Draft: $DRAFTPATH"
echo "  Data: $TRAINPATH"
echo "  Epochs: $NUM_EPOCHS   DS: $DS_CONFIG"
echo "  Save root: $SAVEROOT"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "============================================"

do_train_eval() {
    local SEED=$1
    local TAG="${TAG_BASE}_s${SEED}"
    echo ""
    echo "=========== SEED $SEED TRAIN: $TAG ==========="

    # Skip training if state_$((NUM_EPOCHS-1)) already exists
    if [ -f "$SAVEROOT/$TAG/state_$((NUM_EPOCHS-1))/pytorch_model.bin" ]; then
        echo "[SKIP TRAIN] $TAG state_$((NUM_EPOCHS-1)) already exists"
    else
        MASTER_PORT=$((29500 + RANDOM % 1000))
        deepspeed --master_port $MASTER_PORT sdpo/main_small_lm.py \
            --basepath "$BASEPATH" \
            --draftpath "$DRAFTPATH" \
            --trainpath "$TRAINPATH" \
            --testpath sdpo/data/mixed_val_80.jsonl \
            --deepspeed_config "$DS_CONFIG" \
            --savedir "$SAVEROOT/$TAG" \
            --anchor ce \
            --aux_loss none \
            --anchor_weight none \
            --aux_weight pow08 \
            --sigmoid_coef 0.1 \
            --T_max 0.1 --T_min 0.1 \
            --prob_T_max 1.0 --prob_T_min 0.1 \
            --draft_T_max 1.0 --draft_T_min 1.0 \
            --eal_weight 0.5 \
            --grpo_coef 0.0 \
            --grpo_k_groups 8 --grpo_m 4 --grpo_eps 0.2 \
            --grpo_mode window --grpo_sample_temp 1.0 --grpo_reward hard \
            --gamma 7 --max_len 2048 \
            --num_epochs "$NUM_EPOCHS" \
            --seed "$SEED" --lr 1e-6 \
            --max_train_samples 60000
        TRAIN_RC=$?
        echo "seed $SEED train rc=$TRAIN_RC"
        if [ $TRAIN_RC -ne 0 ]; then
            echo "[EVAL] seed $SEED skipped: train failed"
            return $TRAIN_RC
        fi
    fi

    CKPT="$SAVEROOT/$TAG/state_$((NUM_EPOCHS - 1))"
    if [ ! -f "$CKPT/pytorch_model.bin" ]; then
        CKPT=$(ls -d "$SAVEROOT/$TAG"/state_* 2>/dev/null | sort -V | tail -1)
        echo "[EVAL] state_$((NUM_EPOCHS-1)) not found, fallback to $CKPT"
    fi

    echo ""
    echo "=========== SEED $SEED EVAL: $TAG  ckpt=$CKPT ==========="

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

    # Tree γ=7: 3-bench + LW/OSR2 companion × 3 modes
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
    echo "=========== SEED $SEED DONE $(date) ==========="
}

do_train_eval 0
do_train_eval 1

echo ""
echo "=========== ALL SEEDS DONE $(date) ==========="
