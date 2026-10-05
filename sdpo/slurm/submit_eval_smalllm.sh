#!/bin/bash
# Submit eval jobs for SmallLM sweep checkpoints.
#
# Usage:
#   bash sdpo/slurm/submit_eval_smalllm.sh              # eval ALL
#   bash sdpo/slurm/submit_eval_smalllm.sh sl_coef      # eval matching filter
#   bash sdpo/slurm/submit_eval_smalllm.sh --no-baseline # skip original baseline

set -e
cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization

mkdir -p logs/sdpo_smalllm

SLURM_EVAL=sdpo/slurm/smalllm_eval.slurm
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

echo "Submitting SmallLM eval jobs (filter=${FILTER:-ALL}, epoch=${EPOCH:-latest})"
echo "================================================================"

# 1. Original Qwen3-1.7B baseline (no training)
if [ $SKIP_BASELINE -eq 0 ]; then
    JOB=$(sbatch --job-name="eval_sl_orig" \
        --output="logs/sdpo_smalllm/eval_sl_original_%j.out" \
        --error="logs/sdpo_smalllm/eval_sl_original_%j.err" \
        --export=ALL,TAG="sl_original",DRAFT_CKPT="Qwen/Qwen3-1.7B" \
        "$SLURM_EVAL" | awk '{print $4}')
    echo "  Submitted sl_original (Qwen3-1.7B untrained) -> $JOB"
fi

# 2. Sweep checkpoints
count=0
for dir in /scratch/tx856/spec_reason/scratch/loss_train_smalllm/*/; do
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
        --output="logs/sdpo_smalllm/eval_${tag}_%j.out" \
        --error="logs/sdpo_smalllm/eval_${tag}_%j.err" \
        --export=ALL,TAG="$tag",DRAFT_CKPT="$ckpt" \
        "$SLURM_EVAL" | awk '{print $4}')
    echo "  Submitted $tag ($ckpt) -> $JOB"
    count=$((count + 1))
done

echo ""
echo "$count eval jobs submitted. Monitor with: squeue -u \$USER"
echo "Results in: smalllm_eval_results/"
