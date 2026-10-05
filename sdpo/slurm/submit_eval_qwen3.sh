#!/bin/bash
# Submit eval jobs for Qwen3 sweep checkpoints.
#
# Usage:
#   bash sdpo/slurm/submit_eval_qwen3.sh              # eval ALL
#   bash sdpo/slurm/submit_eval_qwen3.sh q3_coef       # eval matching filter
#   bash sdpo/slurm/submit_eval_qwen3.sh --no-baseline  # skip original baseline
#   bash sdpo/slurm/submit_eval_qwen3.sh --epoch 2      # eval epoch 2

set -e
cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization

mkdir -p logs/sdpo_qwen3

SLURM_EVAL=sdpo/slurm/qwen3_eval.slurm
EPOCH=""
FILTER=""
SKIP_BASELINE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --epoch) EPOCH="$2"; shift 2 ;;
        --no-baseline) SKIP_BASELINE=1; shift ;;
        *) FILTER="$1"; shift ;;
    esac
done

echo "Submitting Qwen3 eval jobs (filter=${FILTER:-ALL}, epoch=${EPOCH:-latest})"
echo "================================================================"

# 1. Original EAGLE3 Qwen3 baseline
if [ $SKIP_BASELINE -eq 0 ]; then
    JOB=$(sbatch --job-name="eval_q3_orig" \
        --output="logs/sdpo_qwen3/eval_q3_original_%j.out" \
        --error="logs/sdpo_qwen3/eval_q3_original_%j.err" \
        --export=ALL,TAG="q3_original",EA_MODEL_PATH="AngelSlim/Qwen3-8B_eagle3" \
        "$SLURM_EVAL" | awk '{print $4}')
    echo "  Submitted q3_original -> $JOB"
fi

# 2. Sweep checkpoints
count=0
for dir in /scratch/tx856/spec_reason/scratch/loss_train_qwen3/*/; do
    tag=$(basename "$dir")

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
        --output="logs/sdpo_qwen3/eval_${tag}_%j.out" \
        --error="logs/sdpo_qwen3/eval_${tag}_%j.err" \
        --export=ALL,TAG="$tag",EA_MODEL_PATH="$ckpt" \
        "$SLURM_EVAL" | awk '{print $4}')
    echo "  Submitted $tag ($ckpt) -> $JOB"
    count=$((count + 1))
done

echo ""
echo "$count eval jobs submitted. Monitor with: squeue -u \$USER"
echo "Results: bash sdpo/slurm/summarize_eval.sh logs/sdpo_qwen3"
