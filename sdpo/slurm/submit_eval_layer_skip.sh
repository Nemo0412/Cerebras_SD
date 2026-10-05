#!/bin/bash
# Submit layer-skip draft eval jobs. Covers:
#   (1) Untrained baseline: Qwen3-0.6B with various exit layers (layer sweep)
#   (2) Trained checkpoints: each directory under $SAVEROOT
#
# Usage:
#   bash sdpo/slurm/submit_eval_layer_skip.sh                                     # eval baseline + all trained
#   bash sdpo/slurm/submit_eval_layer_skip.sh ls_best                             # filter tag
#   bash sdpo/slurm/submit_eval_layer_skip.sh ls_best ls_e4_8_12 --no-baseline    # multi + skip baseline
#   EXIT_LAYERS=8,12,16 BENCH=mt_bench,gsm8k \
#     bash sdpo/slurm/submit_eval_layer_skip.sh

set -e
cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
mkdir -p logs/layerskip

SLURM_EVAL=sdpo/slurm/layer_skip_eval.slurm
BASEPATH=${BASEPATH:-Qwen/Qwen3-32B}
DRAFTPATH=${DRAFTPATH:-Qwen/Qwen3-0.6B}
SAVEROOT=${SAVEROOT:-/scratch/tx856/spec_reason/scratch/loss_train_layerskip}
OUTPUT_DIR=${OUTPUT_DIR:-layerskip_eval_results}
EXIT_LAYERS=${EXIT_LAYERS:-4,8,12,16,20,24,28}

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

# BENCH is inherited via --export=ALL; don't put commas inside --export.
export BENCH="${BENCH:-all}"
export EXIT_LAYERS

FILTER_DESC="${FILTERS[*]:-ALL}"
echo "Submitting LayerSkip eval jobs"
echo "  target:      $BASEPATH"
echo "  draft path:  $DRAFTPATH (for untrained baseline)"
echo "  exit layers: $EXIT_LAYERS"
echo "  filter:      $FILTER_DESC"
echo "  epoch:       ${EPOCH:-latest}"
echo "  bench:       $BENCH"
echo "================================================================"

# 1. Untrained small-LM baseline (no training, just exit sweep)
# BASELINE_TAG can be set to distinguish parallel baseline jobs (e.g., one per
# exit layer). Defaults to ls_original. Writing to the same tag in parallel
# would race and overwrite; use different tags when splitting by exit.
BASELINE_TAG=${BASELINE_TAG:-ls_original}
if [ $SKIP_BASELINE -eq 0 ]; then
    JOB=$(sbatch --job-name="eval_${BASELINE_TAG}" \
        --output="logs/layerskip/eval_${BASELINE_TAG}_%j.out" \
        --error="logs/layerskip/eval_${BASELINE_TAG}_%j.err" \
        --gres=gpu:${GPUS:-1} \
        --constraint="a100|h100|h200" \
        --mem=${MEM:-96G} \
        --export=ALL,TAG="$BASELINE_TAG",DRAFT_CKPT="$DRAFTPATH",BASEPATH="$BASEPATH",OUTPUT_DIR="$OUTPUT_DIR",CONDA_ENV=longreason \
        "$SLURM_EVAL" | awk '{print $4}')
    echo "  Submitted $BASELINE_TAG -> $JOB"
fi

# 2. Trained checkpoints under $SAVEROOT
count=0
if [ -d "$SAVEROOT" ]; then
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
            echo "  SKIP $tag (no checkpoint)"
            continue
        fi

        JOB=$(sbatch --job-name="eval_${tag}" \
            --output="logs/layerskip/eval_${tag}_%j.out" \
            --error="logs/layerskip/eval_${tag}_%j.err" \
            --gres=gpu:2 \
            --constraint="a100|h100|h200" \
            --mem=192G \
            --export=ALL,TAG="$tag",DRAFT_CKPT="$ckpt",BASEPATH="$BASEPATH",OUTPUT_DIR="$OUTPUT_DIR",CONDA_ENV=longreason \
            "$SLURM_EVAL" | awk '{print $4}')
        echo "  Submitted $tag ($ckpt) -> $JOB"
        count=$((count + 1))
    done
fi

echo ""
if [ $SKIP_BASELINE -eq 0 ]; then
    echo "Baseline submitted (1 job) + $count trained-checkpoint eval jobs. Monitor: squeue -u \$USER"
else
    echo "$count trained-checkpoint eval jobs submitted (baseline skipped). Monitor: squeue -u \$USER"
fi
echo "Results in: $OUTPUT_DIR/"
