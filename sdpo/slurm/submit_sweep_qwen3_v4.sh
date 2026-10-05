#!/bin/bash
# Submit Qwen3-8B EAGLE3 acceptance_length_v4 sweep.
# V4: β = 2σ(gap/T), range [0, 1]. accepted→1, rejected→0.
# Sum + 0.8^t decay.
#
# Usage:
#   bash sdpo/slurm/submit_sweep_qwen3_v4.sh                    # submit ALL
#   bash sdpo/slurm/submit_sweep_qwen3_v4.sh q3v4_coef_0.1      # submit single
#   bash sdpo/slurm/submit_sweep_qwen3_v4.sh --list              # list tasks

set -e
cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization

mkdir -p logs/sdpo_qwen3

SLURM_SCRIPT=sdpo/slurm/qwen3_train.slurm
NUM_EPOCHS=${EPOCHS:-3}
MAX_SAMPLES=${MAX_SAMPLES:-""}

# Format: "tag|aux_loss|coef|tmax|tmin|dataset|baseline"
# baseline empty = normal; can be 'eagle_only' or 'ce_only'
TRAIN_10K=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_train_10K.jsonl
VAL_10K=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_val_80.jsonl
TRAIN_FULL=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_train.jsonl
VAL_FULL=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_val.jsonl

TASKS=(
    # ─── 10K dataset ─────────────────────────────────
    "q3v4_eagle_only|sigmoid|0.0|1.0|1.0|10k"
    # V4 coef sweep (T=1.0)
    "q3v4_coef_0.01|acceptance_length_v4|0.01|1.0|1.0|10k"
    "q3v4_coef_0.05|acceptance_length_v4|0.05|1.0|1.0|10k"
    "q3v4_coef_0.1|acceptance_length_v4|0.1|1.0|1.0|10k"
    "q3v4_coef_0.2|acceptance_length_v4|0.2|1.0|1.0|10k"
    "q3v4_coef_0.5|acceptance_length_v4|0.5|1.0|1.0|10k"
    "q3v4_coef_1.0|acceptance_length_v4|1.0|1.0|1.0|10k"
    # V4 temperature sweep (coef=0.1)
    "q3v4_T_0.1|acceptance_length_v4|0.1|0.1|0.1|10k"
    "q3v4_T_0.5|acceptance_length_v4|0.1|0.5|0.5|10k"
    "q3v4_T_2.0|acceptance_length_v4|0.1|2.0|2.0|10k"
    "q3v4_T_5.0|acceptance_length_v4|0.1|5.0|5.0|10k"

    "q3v4_coef_0.2_T_0.1|acceptance_length_v4|0.2|0.1|0.1|10k"
    "q3v4_coef_0.2_T_0.5|acceptance_length_v4|0.2|0.5|0.5|10k"

    # V4 high coef + low T
    "q3v4_c0.5_T0.5|acceptance_length_v4|0.5|0.5|0.5|10k"
    "q3v4_c1.0_T1|acceptance_length_v4|1.0|1.0|1.0|10k"
    # V2 comparison
    "q3v4_v2_coef_0.1|acceptance_length_v2|0.1|1.0|1.0|10k"
    "q3v4_v2_coef_0.5|acceptance_length_v2|0.5|1.0|1.0|10k"

    # ─── Full 68K dataset ────────────────────────────
    "q3v4_full_eagle_only|sigmoid|0.0|1.0|1.0|full"
    "q3v4_full_coef_0.1|acceptance_length_v4|0.1|1.0|1.0|full"
    "q3v4_full_coef_0.5|acceptance_length_v4|0.5|1.0|1.0|full"
    "q3v4_full_coef_1.0|acceptance_length_v4|1.0|1.0|1.0|full"
    "q3v4_full_T_0.1|acceptance_length_v4|0.1|0.1|0.1|full"
    "q3v4_full_v2_0.1|acceptance_length_v2|0.1|1.0|1.0|full"
    "q3v4_full_v2_0.5|acceptance_length_v2|0.5|1.0|1.0|full"

    # ─── CE-only baseline (hard cross-entropy with target argmax) ──
    "q3v4_ce_only|sigmoid|0.0|0.1|0.1|10k|ce_only"
    "q3v4_full_ce_only|sigmoid|0.0|0.1|0.1|full|ce_only"
)

if [ "$1" = "--list" ]; then
    echo "Available Qwen3 V4 tasks:"
    for task in "${TASKS[@]}"; do
        IFS='|' read -r tag aux coef tmax tmin ds bl <<< "$task"
        printf "  %-30s  aux=%-25s coef=%-5s data=%-4s %s\n" "$tag" "$aux" "$coef" "$ds" "${bl:+baseline=$bl}"
    done
    exit 0
fi

submit() {
    local tag=$1 aux=$2 coef=$3 tmax=$4 tmin=$5 ds=$6 bl=$7

    if [ "$ds" = "full" ]; then
        local trainpath="$TRAIN_FULL"
        local testpath="$VAL_FULL"
    else
        local trainpath="$TRAIN_10K"
        local testpath="$VAL_10K"
    fi

    EXTRA_EXPORT=""
    [ -n "$bl" ] && EXTRA_EXPORT=",BASELINE=$bl"

    JOB_ID=$(sbatch \
        --job-name="q3v4_${tag}" \
        --output="logs/sdpo_qwen3/train_${tag}_%j.out" \
        --error="logs/sdpo_qwen3/train_${tag}_%j.err" \
        --export=ALL,TAG="$tag",AUX_LOSS="$aux",SIGMOID_COEF="$coef",T_MAX="$tmax",T_MIN="$tmin",NUM_EPOCHS="$NUM_EPOCHS",MAX_SAMPLES="$MAX_SAMPLES",TRAINPATH="$trainpath",TESTPATH="$testpath"${EXTRA_EXPORT} \
        "$SLURM_SCRIPT" | awk '{print $4}')

    echo "  Submitted $tag ($ds) -> $JOB_ID"
}

SELECTED=("$@")

echo "Qwen3 V4 sweep (β=2σ, sum+0.8^t, epochs=$NUM_EPOCHS, samples=${MAX_SAMPLES:-full})"
echo "================================================================"

count=0
for task in "${TASKS[@]}"; do
    IFS='|' read -r tag aux coef tmax tmin ds bl <<< "$task"

    if [ ${#SELECTED[@]} -gt 0 ]; then
        match=0
        for sel in "${SELECTED[@]}"; do
            if [ "$sel" = "$tag" ]; then match=1; break; fi
        done
        [ $match -eq 0 ] && continue
    fi

    submit "$tag" "$aux" "$coef" "$tmax" "$tmin" "$ds" "$bl"
    count=$((count + 1))
done

echo ""
echo "$count jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs in: logs/sdpo_qwen3/"
