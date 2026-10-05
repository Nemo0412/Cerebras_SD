#!/bin/bash
# Q32B variant: single (bench, decode) eval at max_new=4096 with 2 GPU.
# Env: DRAFT_CKPT, TAG_PREFIX, BENCH, DECODE, TOPK_ARGS, FILE_SUFFIX

#SBATCH --job-name=altv_max4096_q32
#SBATCH --time=12:00:00
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
: ${BENCH:?must set BENCH}
: ${DECODE:?must set DECODE (chain|tree)}
BASEPATH=Qwen/Qwen3-32B
OUT=smalllm_tree_eval_results
SEED=0
MAX_NEW=4096
NSAMPLES=20
TOPK_ARGS=${TOPK_ARGS:-"--draft-top-k 20 --target-top-k 20"}
FILE_SUFFIX=${FILE_SUFFIX:-"dtopk20_ttopk20_"}

if [ "$DECODE" = "chain" ]; then
    FT="${TAG_PREFIX}_eseed${SEED}_chain_g7_samp_t1_ratio_${FILE_SUFFIX}max4096_${BENCH}_al"
    EXTRA_ARGS="--baseline-only --baseline-gamma 7"
else
    FT="${TAG_PREFIX}_tree_eseed${SEED}_samp_t1_ratio_${FILE_SUFFIX}max4096_${BENCH}_al"
    EXTRA_ARGS=""
fi

OUT_FILE="$OUT/${FT}.json"
if [ -f "$OUT_FILE" ]; then
    echo "SKIP $FT (already exists)"
    exit 0
fi

echo "===== $FT (Q32B, gpu:2) ====="
echo "  draft: $DRAFT_CKPT"

python sdpo/eval_small_lm_tree.py \
    --base-model-path "$BASEPATH" --draft-model-path "$DRAFT_CKPT" \
    --bench-name "$BENCH" --max-new-tokens "$MAX_NEW" --num-samples "$NSAMPLES" \
    --output-dir "$OUT" --seed "$SEED" --tag "$FT" \
    --gamma 7 --top-k "4,3,2,1,1,1,1" --budget 128 \
    --temperature 1.0 --draft-mode sample --verify-mode ratio \
    $EXTRA_ARGS $TOPK_ARGS
echo "======= DONE $(date) ======="
