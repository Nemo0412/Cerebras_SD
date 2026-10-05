#!/bin/bash
# Submit eval jobs for sweep checkpoints + EAGLE3 baseline.
#
# Usage:
#   bash sdpo/slurm/submit_eval.sh              # eval ALL checkpoints
#   bash sdpo/slurm/submit_eval.sh v2            # eval only tags matching "v2"
#   bash sdpo/slurm/submit_eval.sh v2_coef       # eval only tags matching "v2_coef"
#   bash sdpo/slurm/submit_eval.sh --epoch 2     # evaluate epoch 2 checkpoint
#   bash sdpo/slurm/submit_eval.sh --no-baseline # skip eagle3_original baseline

set -e
cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization

SLURM_EVAL=sdpo/slurm/acclength_eval.slurm
EPOCH=""
FILTER=""
SKIP_BASELINE=0

# Parse args
while [ $# -gt 0 ]; do
    case "$1" in
        --epoch) EPOCH="$2"; shift 2 ;;
        --no-baseline) SKIP_BASELINE=1; shift ;;
        *) FILTER="$1"; shift ;;
    esac
done

echo "Submitting eval jobs (filter=${FILTER:-ALL}, epoch=${EPOCH:-latest})"
echo "================================================================"

# 1. Original EAGLE3 baseline
if [ $SKIP_BASELINE -eq 0 ]; then
    JOB=$(sbatch --job-name="eval_eagle3_orig" \
        --output="logs/sdpo_acclength/eval_eagle3_original_%j.out" \
        --error="logs/sdpo_acclength/eval_eagle3_original_%j.err" \
        --export=ALL,TAG="eagle3_original",EA_MODEL_PATH="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B" \
        "$SLURM_EVAL" | awk '{print $4}')
    echo "  Submitted eagle3_original -> $JOB"
fi

# 2. Sweep checkpoints (filtered by pattern)
count=0
for dir in /scratch/tx856/spec_reason/scratch/loss_train/*/; do
    tag=$(basename "$dir")

    # Apply filter if specified
    if [ -n "$FILTER" ] && [[ "$tag" != *"$FILTER"* ]]; then
        continue
    fi

    if [ -n "$EPOCH" ]; then
        ckpt="$dir/state_$((EPOCH - 1))"
    else
        ckpt=$(ls -d "$dir"/state_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
    fi

    if [ -z "$ckpt" ] || [ ! -d "$ckpt" ]; then
        echo "  SKIP $tag (no checkpoint found)"
        continue
    fi

    JOB=$(sbatch --job-name="eval_${tag}" \
        --output="logs/sdpo_acclength/eval_${tag}_%j.out" \
        --error="logs/sdpo_acclength/eval_${tag}_%j.err" \
        --export=ALL,TAG="$tag",EA_MODEL_PATH="$ckpt" \
        "$SLURM_EVAL" | awk '{print $4}')
    echo "  Submitted $tag ($ckpt) -> $JOB"
    count=$((count + 1))
done

echo ""
echo "$count eval jobs submitted. Monitor with: squeue -u \$USER"
echo "Results: bash sdpo/slurm/summarize_eval.sh"
