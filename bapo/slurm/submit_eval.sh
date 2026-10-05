#!/bin/bash
# Submit eval jobs for all sweep checkpoints + original EAGLE3 baseline.
#
# Usage:
#   bash bapo/slurm/submit_eval.sh
#   bash bapo/slurm/submit_eval.sh 2    # evaluate epoch 2 checkpoint (default: last)

set -e
cd /scratch/xt2251/SDPO-Speculative-Decoding-Policy-Optimization

SLURM_EVAL=bapo/slurm/bapo_eval.slurm
EPOCH=${1:-}  # empty = find latest

echo "Submitting eval jobs"
echo "================================================================"

# 1. Original EAGLE3 (no training at all) — the true baseline
JOB=$(sbatch --job-name="eval_eagle3_orig" \
    --export=ALL,TAG="eagle3_original",EA_MODEL_PATH="/scratch/xt2251/models/EAGLE3-LLaMA3.1-Instruct-8B" \
    "$SLURM_EVAL" | awk '{print $4}')
echo "  Submitted eagle3_original -> $JOB"

# 2. All sweep checkpoints
for dir in bapo_sweep/*/; do
    tag=$(basename "$dir")

    # Find checkpoint: use specified epoch or latest
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
        --export=ALL,TAG="$tag",EA_MODEL_PATH="$ckpt" \
        "$SLURM_EVAL" | awk '{print $4}')
    echo "  Submitted $tag ($ckpt) -> $JOB"
done

echo ""
echo "Monitor with: squeue -u \$USER"
echo "Results in: bapo_eval_results/"
