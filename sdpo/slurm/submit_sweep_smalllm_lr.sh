#!/bin/bash
# Submit Small LM learning rate sweep.
#
# Usage:
#   bash sdpo/slurm/submit_sweep_smalllm_lr.sh                    # submit ALL
#   bash sdpo/slurm/submit_sweep_smalllm_lr.sh sl_lr_1e-6         # submit single
#   bash sdpo/slurm/submit_sweep_smalllm_lr.sh --list             # list tasks
#
# Environment variables:
#   EPOCHS=1 bash sdpo/slurm/submit_sweep_smalllm_lr.sh sl_lr_1e-6

set -e
cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization

mkdir -p logs/sdpo_smalllm

SLURM_SCRIPT=sdpo/slurm/smalllm_train.slurm
NUM_EPOCHS=${EPOCHS:-3}
MAX_SAMPLES=${MAX_SAMPLES:-""}

# Format: "tag|coef|tmax|tmin|lr|baseline"
TASKS=(
    # LR sweep with acceptance_length_v2 (coef=0.1, T=0.1)
    "sl_lr_1e-7|0.1|0.1|0.1|1e-7|"
    "sl_lr_5e-7|0.1|0.1|0.1|5e-7|"
    "sl_lr_1e-6|0.1|0.1|0.1|1e-6|"
    "sl_lr_5e-6|0.1|0.1|0.1|5e-6|"
    "sl_lr_1e-5|0.1|0.1|0.1|1e-5|"
    "sl_lr_3e-5|0.1|0.1|0.1|3e-5|"
    # Eagle-only with different LRs
    "sl_eo_lr_1e-7|0.0|0.1|0.1|1e-7|eagle_only"
    "sl_eo_lr_1e-6|0.0|0.1|0.1|1e-6|eagle_only"
    "sl_eo_lr_5e-6|0.0|0.1|0.1|5e-6|eagle_only"
    "sl_eo_lr_1e-5|0.0|0.1|0.1|1e-5|eagle_only"
)

if [ "$1" = "--list" ]; then
    echo "Available SmallLM LR sweep tasks:"
    for task in "${TASKS[@]}"; do
        IFS='|' read -r tag coef tmax tmin lr bl <<< "$task"
        printf "  %-20s  coef=%-5s lr=%-8s %s\n" "$tag" "$coef" "$lr" "${bl:+baseline=$bl}"
    done
    exit 0
fi

submit() {
    local tag=$1 coef=$2 tmax=$3 tmin=$4 lr=$5 bl=$6

    EXTRA_EXPORT=""
    [ -n "$bl" ] && EXTRA_EXPORT=",BASELINE=$bl"

    JOB_ID=$(sbatch \
        --job-name="sl_${tag}" \
        --output="logs/sdpo_smalllm/train_${tag}_%j.out" \
        --error="logs/sdpo_smalllm/train_${tag}_%j.err" \
        --export=ALL,TAG="$tag",SIGMOID_COEF="$coef",T_MAX="$tmax",T_MIN="$tmin",LR="$lr",NUM_EPOCHS="$NUM_EPOCHS",MAX_SAMPLES="$MAX_SAMPLES"${EXTRA_EXPORT} \
        "$SLURM_SCRIPT" | awk '{print $4}')

    echo "  Submitted $tag -> $JOB_ID"
}

SELECTED=("$@")

echo "SmallLM LR sweep (epochs=$NUM_EPOCHS, samples=${MAX_SAMPLES:-full})"
echo "================================================================"

count=0
for task in "${TASKS[@]}"; do
    IFS='|' read -r tag coef tmax tmin lr bl <<< "$task"

    if [ ${#SELECTED[@]} -gt 0 ]; then
        match=0
        for sel in "${SELECTED[@]}"; do
            [ "$sel" = "$tag" ] && match=1 && break
        done
        [ $match -eq 0 ] && continue
    fi

    submit "$tag" "$coef" "$tmax" "$tmin" "$lr" "$bl"
    count=$((count + 1))
done

echo ""
echo "$count jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs in: logs/sdpo_smalllm/"
