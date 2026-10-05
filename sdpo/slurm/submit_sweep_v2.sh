#!/bin/bash
# Submit acceptance_length_v2 (sum + 0.8^t decay) sweep.
#
# Usage:
#   bash sdpo/slurm/submit_sweep_v2.sh                          # submit ALL tasks, 3 epochs
#   bash sdpo/slurm/submit_sweep_v2.sh v2_coef_0.1              # submit single task
#   bash sdpo/slurm/submit_sweep_v2.sh v2_coef_0.1 v2_coef_0.5  # submit multiple tasks
#   bash sdpo/slurm/submit_sweep_v2.sh --list                   # list all available task names
#
# Environment variables (optional):
#   EPOCHS=1 MAX_SAMPLES=6800 bash sdpo/slurm/submit_sweep_v2.sh v2_coef_0.1

set -e
cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization

SLURM_SCRIPT=sdpo/slurm/acclength_train.slurm
NUM_EPOCHS=${EPOCHS:-3}
MAX_SAMPLES=${MAX_SAMPLES:-""}

# ─── Task definitions ────────────────────────────────────────
# Format: "tag|aux_loss|coef|tmax|tmin"
TASKS=(
    # Baselines
    "v2_eagle_only|sigmoid|0.0|1.0|1.0"
    "v2_sigmoid|sigmoid|0.1|5.0|1.0"
    # V2 coef sweep
    "v2_coef_0.01|acceptance_length_v2|0.01|5.0|1.0"
    "v2_coef_0.05|acceptance_length_v2|0.05|5.0|1.0"
    "v2_coef_0.1|acceptance_length_v2|0.1|5.0|1.0"
    "v2_coef_0.2|acceptance_length_v2|0.2|5.0|1.0"
    "v2_coef_0.5|acceptance_length_v2|0.5|5.0|1.0"
    "v2_coef_1.0|acceptance_length_v2|1.0|5.0|1.0"
    # V2 temperature sweep (coef=0.1)
    "v2_T_0.1|acceptance_length_v2|0.1|0.1|0.1"
    "v2_T_0.5|acceptance_length_v2|0.1|0.5|0.5"
    "v2_T_1.0|acceptance_length_v2|0.1|1.0|1.0"
    "v2_T_2.0|acceptance_length_v2|0.1|2.0|2.0"
    "v2_T_10.0|acceptance_length_v2|0.1|10.0|10.0"
    # V2 temperature schedule
    "v2_sched_01_5|acceptance_length_v2|0.1|5.0|0.1"
    "v2_sched_1_10|acceptance_length_v2|0.1|10.0|1.0"
    # V2 high coef + low T
    "v2_c0.5_T1|acceptance_length_v2|0.5|1.0|1.0"
    "v2_c1.0_T2|acceptance_length_v2|1.0|2.0|2.0"
)

# ─── List mode ────────────────────────────────────────────────
if [ "$1" = "--list" ]; then
    echo "Available tasks:"
    for task in "${TASKS[@]}"; do
        IFS='|' read -r tag aux coef tmax tmin <<< "$task"
        printf "  %-20s  aux=%-25s coef=%-5s T=(%s->%s)\n" "$tag" "$aux" "$coef" "$tmin" "$tmax"
    done
    exit 0
fi

# ─── Submit function ──────────────────────────────────────────
submit() {
    local tag=$1 aux=$2 coef=$3 tmax=$4 tmin=$5

    JOB_ID=$(sbatch \
        --job-name="v2_${tag}" \
        --output="logs/sdpo_acclength/train_${tag}_%j.out" \
        --error="logs/sdpo_acclength/train_${tag}_%j.err" \
        --export=ALL,TAG="$tag",AUX_LOSS="$aux",SIGMOID_COEF="$coef",T_MAX="$tmax",T_MIN="$tmin",NUM_EPOCHS="$NUM_EPOCHS",MAX_SAMPLES="$MAX_SAMPLES" \
        "$SLURM_SCRIPT" | awk '{print $4}')

    echo "  Submitted $tag -> $JOB_ID"
}

# ─── Determine which tasks to submit ─────────────────────────
SELECTED=("$@")

echo "AccLength V2 sweep (sum + 0.8^t, epochs=$NUM_EPOCHS, samples=${MAX_SAMPLES:-full})"
echo "================================================================"

count=0
for task in "${TASKS[@]}"; do
    IFS='|' read -r tag aux coef tmax tmin <<< "$task"

    # If specific tasks requested, skip non-matching ones
    if [ ${#SELECTED[@]} -gt 0 ]; then
        match=0
        for sel in "${SELECTED[@]}"; do
            if [ "$sel" = "$tag" ]; then match=1; break; fi
        done
        [ $match -eq 0 ] && continue
    fi

    submit "$tag" "$aux" "$coef" "$tmax" "$tmin"
    count=$((count + 1))
done

echo ""
echo "$count jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs in: logs/sdpo_acclength/"
