#!/bin/bash
# Submit eval jobs for Qwen3-4B-Thinking EAGLE3 sweep checkpoints.
#
# Usage:
#   bash sdpo/slurm/submit_eval_q4t.sh                    # eval ALL
#   bash sdpo/slurm/submit_eval_q4t.sh q4t_best           # filter by substring
#   bash sdpo/slurm/submit_eval_q4t.sh --no-baseline      # skip original baseline
#   bash sdpo/slurm/submit_eval_q4t.sh --epoch 2          # eval epoch 2

set -e
cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization

mkdir -p logs/sdpo_q4t

SLURM_EVAL=sdpo/slurm/qwen3_eval.slurm
BASEPATH=Qwen/Qwen3-4B-Thinking-2507
SAVEROOT=/scratch/tx856/spec_reason/scratch/loss_train_q4t
OUTPUT_DIR=q4t_eval_results

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

echo "Submitting Q4T eval jobs (filter=${FILTER:-ALL}, epoch=${EPOCH:-latest})"
echo "  target: $BASEPATH"
echo "================================================================"

# 1. Original EAGLE draft baseline (no training)
if [ $SKIP_BASELINE -eq 0 ]; then
    JOB=$(sbatch --job-name="eval_q4t_orig" \
        --output="logs/sdpo_q4t/eval_q4t_original_%j.out" \
        --error="logs/sdpo_q4t/eval_q4t_original_%j.err" \
        --export=ALL,TAG="q4t_original",EA_MODEL_PATH="taobao-mnn/Qwen3-4B-Thinking-2507-Eagle",BASEPATH="$BASEPATH",OUTPUT_DIR="$OUTPUT_DIR" \
        "$SLURM_EVAL" | awk '{print $4}')
    echo "  Submitted q4t_original -> $JOB"
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
        --output="logs/sdpo_q4t/eval_${tag}_%j.out" \
        --error="logs/sdpo_q4t/eval_${tag}_%j.err" \
        --export=ALL,TAG="$tag",EA_MODEL_PATH="$ckpt",BASEPATH="$BASEPATH",OUTPUT_DIR="$OUTPUT_DIR" \
        "$SLURM_EVAL" | awk '{print $4}')
    echo "  Submitted $tag ($ckpt) -> $JOB"
    count=$((count + 1))
done

echo ""
echo "$count eval jobs submitted. Monitor with: squeue -u \$USER"
echo "Results in: $OUTPUT_DIR/"
