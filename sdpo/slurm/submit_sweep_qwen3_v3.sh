#!/bin/bash
# Submit Qwen3 acceptance_length_v3 (loss ≈ -tau) sweep.
#
# Usage:
#   bash sdpo/slurm/submit_sweep_qwen3_v3.sh                          # submit ALL
#   bash sdpo/slurm/submit_sweep_qwen3_v3.sh q3v3_coef_0.1            # submit single task
#   bash sdpo/slurm/submit_sweep_qwen3_v3.sh --list                   # list all tasks
#
# Environment variables:
#   EPOCHS=1 MAX_SAMPLES=500 bash sdpo/slurm/submit_sweep_qwen3_v3.sh q3v3_coef_0.1

set -e
cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization

mkdir -p logs/sdpo_qwen3

SLURM_SCRIPT=sdpo/slurm/qwen3_train.slurm
NUM_EPOCHS=${EPOCHS:-3}
MAX_SAMPLES=${MAX_SAMPLES:-""}

# ─── Task definitions ────────────────────────────────────────
# Format: "tag|aux_loss|coef|tmax|tmin"
TASKS=(
    # Baselines
    "q3v3_eagle_only|sigmoid|0.0|1.0|1.0"
    "q3v3_sigmoid|sigmoid|0.1|5.0|1.0"
    # V3 coef sweep (T=1.0 fixed, since margin handles the shift)
    "q3v3_coef_0.01|acceptance_length_v3|0.01|1.0|1.0"
    "q3v3_coef_0.05|acceptance_length_v3|0.05|1.0|1.0"
    "q3v3_coef_0.1|acceptance_length_v3|0.1|1.0|1.0"
    "q3v3_coef_0.2|acceptance_length_v3|0.2|1.0|1.0"
    "q3v3_coef_0.5|acceptance_length_v3|0.5|1.0|1.0"
    "q3v3_coef_1.0|acceptance_length_v3|1.0|1.0|1.0"
    # V3 temperature sweep (coef=0.1)
    "q3v3_T_0.1|acceptance_length_v3|0.1|0.1|0.1"
    "q3v3_T_0.5|acceptance_length_v3|0.1|0.5|0.5"
    "q3v3_T_2.0|acceptance_length_v3|0.1|2.0|2.0"
    "q3v3_T_5.0|acceptance_length_v3|0.1|5.0|5.0"
    # V3 high coef + low T
    "q3v3_c0.5_T0.5|acceptance_length_v3|0.5|0.5|0.5"
    "q3v3_c1.0_T1|acceptance_length_v3|1.0|1.0|1.0"

    "q3v3_coef_0.1_T0.1|acceptance_length_v3|0.2|1.0|0.1"
)

# ─── List mode ────────────────────────────────────────────────
if [ "$1" = "--list" ]; then
    echo "Available Qwen3 V3 tasks:"
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
        --job-name="q3v3_${tag}" \
        --output="logs/sdpo_qwen3/train_${tag}_%j.out" \
        --error="logs/sdpo_qwen3/train_${tag}_%j.err" \
        --export=ALL,TAG="$tag",AUX_LOSS="$aux",SIGMOID_COEF="$coef",T_MAX="$tmax",T_MIN="$tmin",NUM_EPOCHS="$NUM_EPOCHS",MAX_SAMPLES="$MAX_SAMPLES" \
        "$SLURM_SCRIPT" | awk '{print $4}')

    echo "  Submitted $tag -> $JOB_ID"
}

# ─── Determine which tasks to submit ─────────────────────────
SELECTED=("$@")

echo "Qwen3 V3 sweep (loss≈-tau, epochs=$NUM_EPOCHS, samples=${MAX_SAMPLES:-full})"
echo "================================================================"

count=0
for task in "${TASKS[@]}"; do
    IFS='|' read -r tag aux coef tmax tmin <<< "$task"

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
echo "Logs in: logs/sdpo_qwen3/"
