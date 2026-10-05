#!/bin/bash
# Q32B: chain + tree × 3 bench inside one slurm job (6 sub-evals sequentially).
# Each sub-eval writes to its own file → preemption-safe SKIP granularity.
# Env: DRAFT_CKPT, TAG_PREFIX, (opt) TOPK_ARGS, FILE_SUFFIX

#SBATCH --job-name=altv_max4096_all_q32
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --constraint="h100|h200"
#SBATCH --requeue
#SBATCH --account=torch_pr_74_tandon_priority
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tx856@nyu.edu
#SBATCH --output=logs/altv_max4096/%x_%j.out
#SBATCH --error=logs/altv_max4096/%x_%j.err

set -o pipefail
REPO=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
cd "$REPO"
mkdir -p logs/altv_max4096 smalllm_tree_eval_results
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD"

source ~/.bashrc
conda activate ${CONDA_ENV:-longreason}
set -u
export HF_HOME=${HF_HOME:-/scratch/tx856/.huggingface}
export TORCH_EXTENSIONS_DIR=/scratch/tx856/.cache/torch_extensions

: ${DRAFT_CKPT:?must set DRAFT_CKPT}
: ${TAG_PREFIX:?must set TAG_PREFIX}
BASEPATH=Qwen/Qwen3-32B
OUT=smalllm_tree_eval_results
SEED=0
MAX_NEW=4096
NSAMPLES=20
TOPK_ARGS=${TOPK_ARGS:-"--draft-top-k 20 --target-top-k 20"}
FILE_SUFFIX=${FILE_SUFFIX:-"dtopk20_ttopk20_"}

BENCHES=(mt_bench gsm8k humaneval)
DECODES=(chain tree)

START=$(date '+%s')
echo "===== $TAG_PREFIX · chain+tree × 3 bench · max=$MAX_NEW n=$NSAMPLES ====="
echo "  draft: $DRAFT_CKPT  base: $BASEPATH"

I=0; TOTAL=6
for DECODE in "${DECODES[@]}"; do
    for BENCH in "${BENCHES[@]}"; do
        I=$((I+1))
        if [ "$DECODE" = "chain" ]; then
            FT="${TAG_PREFIX}_eseed${SEED}_chain_g7_samp_t1_ratio_${FILE_SUFFIX}max4096_${BENCH}_al"
            EXTRA_ARGS="--baseline-only --baseline-gamma 7"
        else
            FT="${TAG_PREFIX}_tree_eseed${SEED}_samp_t1_ratio_${FILE_SUFFIX}max4096_${BENCH}_al"
            EXTRA_ARGS=""
        fi
        OUT_FILE="$OUT/${FT}.json"
        if [ -f "$OUT_FILE" ]; then
            echo "[$I/$TOTAL] SKIP $FT"
            continue
        fi
        echo ""; echo "[$I/$TOTAL] === $FT === start=$(date '+%H:%M:%S')"
        set +e
        python sdpo/eval_small_lm_tree.py \
            --base-model-path "$BASEPATH" --draft-model-path "$DRAFT_CKPT" \
            --bench-name "$BENCH" --max-new-tokens "$MAX_NEW" --num-samples "$NSAMPLES" \
            --output-dir "$OUT" --seed "$SEED" --tag "$FT" \
            --gamma 7 --top-k "4,3,2,1,1,1,1" --budget 128 \
            --temperature 1.0 --draft-mode sample --verify-mode ratio \
            $EXTRA_ARGS $TOPK_ARGS
        RC=$?; set -e
        echo "  end=$(date '+%H:%M:%S') rc=$RC"
    done
done

echo ""
echo "======= DONE $(date) elapsed=$(( $(date '+%s') - START ))s ======="
