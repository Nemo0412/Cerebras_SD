#!/bin/bash
# Submit eval jobs for Qwen3.5-9B target + Qwen3.5-0.8B draft sweep.
#
# Usage:
#   bash sdpo/slurm/submit_eval_smalllm_q35.sh              # eval ALL
#   bash sdpo/slurm/submit_eval_smalllm_q35.sh q35_best     # filter
#   bash sdpo/slurm/submit_eval_smalllm_q35.sh --no-baseline

set -e
cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization

mkdir -p logs/sdpo_smalllm_q35

SLURM_EVAL=sdpo/slurm/smalllm_eval.slurm
BASEPATH=Qwen/Qwen3.5-9B
SAVEROOT=/scratch/tx856/spec_reason/scratch/loss_train_smalllm_q35
OUTPUT_DIR=smalllm_q35_eval_results

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

echo "Submitting Q35 eval jobs (filter=${FILTER:-ALL}, epoch=${EPOCH:-latest})"
echo "  target: $BASEPATH"
echo "================================================================"

# 1. Original Qwen3.5-0.8B baseline (no training)
if [ $SKIP_BASELINE -eq 0 ]; then
    JOB=$(sbatch --job-name="eval_q35_orig" \
        --output="logs/sdpo_smalllm_q35/eval_q35_original_%j.out" \
        --error="logs/sdpo_smalllm_q35/eval_q35_original_%j.err" \
        --export=ALL,TAG="q35_original",DRAFT_CKPT="Qwen/Qwen3.5-0.8B",BASEPATH="$BASEPATH",OUTPUT_DIR="$OUTPUT_DIR",CONDA_ENV=longreason \
        "$SLURM_EVAL" | awk '{print $4}')
    echo "  Submitted q35_original -> $JOB"
fi

# 2. Sweep checkpoints
count=0
for dir in "$SAVEROOT"/*/; do
    [ -d "$dir" ] || continue
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
        --output="logs/sdpo_smalllm_q35/eval_${tag}_%j.out" \
        --error="logs/sdpo_smalllm_q35/eval_${tag}_%j.err" \
        --export=ALL,TAG="$tag",DRAFT_CKPT="$ckpt",BASEPATH="$BASEPATH",OUTPUT_DIR="$OUTPUT_DIR",CONDA_ENV=longreason \
        "$SLURM_EVAL" | awk '{print $4}')
    echo "  Submitted $tag ($ckpt) -> $JOB"
    count=$((count + 1))
done

echo ""
echo "$count eval jobs submitted. Monitor with: squeue -u \$USER"
echo "Results in: $OUTPUT_DIR/"
