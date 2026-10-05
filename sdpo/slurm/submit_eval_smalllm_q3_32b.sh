#!/bin/bash
# Submit eval jobs for Qwen3-32B target + Qwen3-4B draft sweep.
#
# Usage:
#   bash sdpo/slurm/submit_eval_smalllm_q3_32b.sh              # eval ALL
#   bash sdpo/slurm/submit_eval_smalllm_q3_32b.sh q32_best     # filter
#   bash sdpo/slurm/submit_eval_smalllm_q3_32b.sh --no-baseline

set -e
cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization

mkdir -p logs/sdpo_smalllm_q3_32b

SLURM_EVAL=sdpo/slurm/smalllm_eval.slurm
BASEPATH=Qwen/Qwen3-32B
SAVEROOT=/scratch/tx856/spec_reason/scratch/loss_train_smalllm_q3_32b
OUTPUT_DIR=smalllm_q3_32b_eval_results

EPOCH=""
FILTERS=()
SKIP_BASELINE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --epoch) EPOCH="$2"; shift 2 ;;
        --no-baseline) SKIP_BASELINE=1; shift ;;
        *) FILTERS+=("$1"); shift ;;
    esac
done

# BENCH is inherited via --export=ALL (sbatch --export parses commas as var
# separators, so we CAN'T put "mt_bench,gsm8k,humaneval" inside --export).
export BENCH="${BENCH:-all}"

FILTER_DESC="${FILTERS[*]:-ALL}"
echo "Submitting Q3-32B eval jobs (filter=${FILTER_DESC}, epoch=${EPOCH:-latest}, bench=$BENCH)"
echo "  target: $BASEPATH"
echo "================================================================"

# 1. Original Qwen3-4B baseline (no training)
if [ $SKIP_BASELINE -eq 0 ]; then
    JOB=$(sbatch --job-name="eval_q32_orig" \
        --output="logs/sdpo_smalllm_q3_32b/eval_q32_original_%j.out" \
        --error="logs/sdpo_smalllm_q3_32b/eval_q32_original_%j.err" \
        --gres=gpu:2 \
        --constraint="a100|h100|h200" \
        --mem=192G \
        --export=ALL,TAG="q32_original",DRAFT_CKPT="Qwen/Qwen3-4B",BASEPATH="$BASEPATH",OUTPUT_DIR="$OUTPUT_DIR",CONDA_ENV=longreason \
        "$SLURM_EVAL" | awk '{print $4}')
    echo "  Submitted q32_original -> $JOB"
fi

# 2. Sweep checkpoints
count=0
for dir in "$SAVEROOT"/*/; do
    [ -d "$dir" ] || continue
    tag=$(basename "$dir")

    if [ ${#FILTERS[@]} -gt 0 ]; then
        match=0
        for f in "${FILTERS[@]}"; do
            if [[ "$tag" == "$f" || "$tag" == *"$f"* ]]; then
                match=1; break
            fi
        done
        [ $match -eq 0 ] && continue
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
        --output="logs/sdpo_smalllm_q3_32b/eval_${tag}_%j.out" \
        --error="logs/sdpo_smalllm_q3_32b/eval_${tag}_%j.err" \
        --gres=gpu:2 \
        --constraint="a100|h100|h200" \
        --mem=192G \
        --export=ALL,TAG="$tag",DRAFT_CKPT="$ckpt",BASEPATH="$BASEPATH",OUTPUT_DIR="$OUTPUT_DIR",CONDA_ENV=longreason \
        "$SLURM_EVAL" | awk '{print $4}')
    echo "  Submitted $tag ($ckpt) -> $JOB"
    count=$((count + 1))
done

echo ""
echo "$count eval jobs submitted. Monitor with: squeue -u \$USER"
echo "Results in: $OUTPUT_DIR/"
