#!/bin/bash
# Eval-only slurm: chain g7 3 modes × 3 eseed + tree g7 3 modes.
# Env: TAG, BASEPATH, CKPT

#SBATCH --job-name=eval_only
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
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

export HF_HOME=${HF_HOME:-/scratch/tx856/.huggingface}

: ${TAG:?must set TAG}
: ${BASEPATH:?must set BASEPATH}
: ${CKPT:?must set CKPT}

OUT=smalllm_tree_eval_results
BENCHES_3="mt_bench,gsm8k,humaneval"
BENCHES_LW="longwriter_top50,osr2_top50"

echo "============================================"
echo "  Eval-only: $TAG"
echo "  Ckpt: $CKPT"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "============================================"

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
