#!/bin/bash
# Q0.6B → Q32B eval with draft top-K + target top-K + T=1 sample_ratio.
# Env: CKPT_TAG, DRAFT_TOP_K (default 20), TARGET_TOP_K (default 20)

#SBATCH --job-name=altv_topk_eval_q32_q06
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --constraint="a100|h100|h200"
#SBATCH --requeue
#SBATCH --account=torch_pr_74_tandon_priority
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tx856@nyu.edu
#SBATCH --output=logs/sdpo_smalllm_q32_q06/%x_%j.out
#SBATCH --error=logs/sdpo_smalllm_q32_q06/%x_%j.err

set -o pipefail
REPO=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
cd "$REPO"
mkdir -p logs/sdpo_smalllm_q32_q06 smalllm_tree_eval_results
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD"

source ~/.bashrc
conda activate ${CONDA_ENV:-longreason}
set -u
export HF_HOME=${HF_HOME:-/scratch/tx856/.huggingface}
export TORCH_EXTENSIONS_DIR=/scratch/tx856/.cache/torch_extensions

: ${CKPT_TAG:?must set CKPT_TAG}
DRAFT_TOP_K=${DRAFT_TOP_K:-20}
TARGET_TOP_K=${TARGET_TOP_K:-20}

CKPT_STATE=${CKPT_STATE:-state_1}
CKPT=/scratch/tx856/spec_reason/scratch/loss_train_smalllm_q32_q06/${CKPT_TAG}/${CKPT_STATE}
[ ! -f "$CKPT/pytorch_model.bin" ] && { echo "Missing $CKPT"; exit 1; }

BASEPATH=Qwen/Qwen3-32B
OUT=smalllm_tree_eval_results
BENCHES=mt_bench,gsm8k,humaneval
SEEDS=(0 1 2)

D_SFX=$([ "$DRAFT_TOP_K" -gt 0 ] && echo "dtopk${DRAFT_TOP_K}" || echo "dfull")
T_SFX=$([ "$TARGET_TOP_K" -gt 0 ] && echo "_ttopk${TARGET_TOP_K}" || echo "")
SUFFIX="${D_SFX}${T_SFX}"

TOTAL=$((${#SEEDS[@]} * 2))
START=$(date '+%s')
echo "===== $CKPT_TAG · T=1 samp_ratio eval (d=$DRAFT_TOP_K t=$TARGET_TOP_K, $SUFFIX) — $TOTAL evals ====="

I=0
for SEED in "${SEEDS[@]}"; do
    # Chain
    I=$((I+1))
    FT="${CKPT_TAG}_eseed${SEED}_chain_g7_samp_t1_ratio_${SUFFIX}_al"
    OUT_FILE="$OUT/${FT}.json"
    if [ -f "$OUT_FILE" ]; then
        echo "[$I/$TOTAL] SKIP $FT"
    else
        echo ""; echo "[$I/$TOTAL] === $FT ===  start=$(date '+%H:%M:%S')"
        set +e
        python sdpo/eval_small_lm_tree.py \
            --base-model-path "$BASEPATH" --draft-model-path "$CKPT" \
            --bench-name "$BENCHES" --max-new-tokens 256 --num-samples 40 \
            --output-dir "$OUT" --seed "$SEED" --tag "$FT" \
            --gamma 7 --top-k "4,3,2,1,1,1,1" --budget 128 \
            --baseline-only --baseline-gamma 7 \
            --temperature 1.0 --draft-mode sample --verify-mode ratio \
            --draft-top-k "$DRAFT_TOP_K" --target-top-k "$TARGET_TOP_K"
        RC=$?; set -e
        echo "  end=$(date '+%H:%M:%S') rc=$RC"
    fi
    # Tree
    I=$((I+1))
    FT="${CKPT_TAG}_tree_eseed${SEED}_samp_t1_ratio_${SUFFIX}_al"
    OUT_FILE="$OUT/${FT}.json"
    if [ -f "$OUT_FILE" ]; then
        echo "[$I/$TOTAL] SKIP $FT"
    else
        echo ""; echo "[$I/$TOTAL] === $FT ===  start=$(date '+%H:%M:%S')"
        set +e
        python sdpo/eval_small_lm_tree.py \
            --base-model-path "$BASEPATH" --draft-model-path "$CKPT" \
            --bench-name "$BENCHES" --max-new-tokens 256 --num-samples 40 \
            --output-dir "$OUT" --seed "$SEED" --tag "$FT" \
            --gamma 7 --top-k "4,3,2,1,1,1,1" --budget 128 \
            --temperature 1.0 --draft-mode sample --verify-mode ratio \
            --draft-top-k "$DRAFT_TOP_K" --target-top-k "$TARGET_TOP_K"
        RC=$?; set -e
        echo "  end=$(date '+%H:%M:%S') rc=$RC"
    fi
done
echo ""
echo "======= ALL DONE $(date) elapsed=$(( $(date '+%s') - START ))s ======="
