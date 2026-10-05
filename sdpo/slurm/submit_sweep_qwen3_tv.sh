#!/bin/bash
# Submit Qwen3-8B EAGLE3 TV loss sweep.
# TV loss = 1 - alpha = Total Variation distance, directly targets acceptance rate.
# Used as auxiliary: L_total = L_EAGLE + coef * TV(p,q)
#
# Usage:
#   bash sdpo/slurm/submit_sweep_qwen3_tv.sh                    # submit ALL
#   bash sdpo/slurm/submit_sweep_qwen3_tv.sh q3tv_coef_0.1      # submit single
#   bash sdpo/slurm/submit_sweep_qwen3_tv.sh --list              # list tasks

set -e
cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization

mkdir -p logs/sdpo_qwen3

SLURM_SCRIPT=sdpo/slurm/qwen3_train.slurm
NUM_EPOCHS=${EPOCHS:-3}
MAX_SAMPLES=${MAX_SAMPLES:-""}

# Format: "tag|aux_loss|coef|tmax|tmin|dataset|baseline"
# dataset: 10k or full
# baseline empty = normal; can be 'eagle_only' or 'ce_only'
TRAIN_10K=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_train_10K.jsonl
VAL_10K=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_val_80.jsonl
TRAIN_FULL=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_train.jsonl
VAL_FULL=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_val.jsonl

TASKS=(
    # ─── 10K dataset ─────────────────────────────────
    # Baselines
    "q3tv_eagle_only|sigmoid|0.0|0.1|0.1|10k"
    "q3tv_sigmoid|sigmoid|0.1|0.1|0.1|10k"
    # TV loss coef sweep
    "q3tv_coef_0.01|tv|0.01|0.1|0.1|10k"
    "q3tv_coef_0.05|tv|0.05|0.1|0.1|10k"
    "q3tv_coef_0.1|tv|0.1|0.1|0.1|10k"
    "q3tv_coef_0.2|tv|0.2|0.1|0.1|10k"
    "q3tv_coef_0.5|tv|0.5|0.1|0.1|10k"
    "q3tv_coef_1.0|tv|1.0|0.1|0.1|10k"
    "q3tv_coef_2.0|tv|2.0|0.1|0.1|10k"
    "q3tv_coef_5.0|tv|5.0|0.1|0.1|10k"
    # TV loss vs V2
    "q3tv_v2_coef_0.1|acceptance_length_v2|0.1|0.1|0.1|10k"
    "q3tv_v2_coef_0.5|acceptance_length_v2|0.5|0.1|0.1|10k"

    # ─── Full 68K dataset ────────────────────────────
    "q3tv_full_eagle_only|sigmoid|0.0|0.1|0.1|full"
    "q3tv_full_sigmoid|sigmoid|0.1|0.1|0.1|full"
    "q3tv_full_coef_0.1|tv|0.1|0.1|0.1|full"
    "q3tv_full_coef_0.5|tv|0.5|0.1|0.1|full"
    "q3tv_full_coef_1.0|tv|1.0|0.1|0.1|full"
    "q3tv_full_coef_2.0|tv|2.0|0.1|0.1|full"
    "q3tv_full_v2_0.1|acceptance_length_v2|0.1|0.1|0.1|full"
    "q3tv_full_v2_0.5|acceptance_length_v2|0.5|0.1|0.1|full"

    # ─── CE-only baseline (hard cross-entropy with target argmax) ──
    "q3tv_ce_only|sigmoid|0.0|0.1|0.1|10k|ce_only"
    "q3tv_full_ce_only|sigmoid|0.0|0.1|0.1|full|ce_only"
)

if [ "$1" = "--list" ]; then
    echo "Available Qwen3 TV loss tasks:"
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
        --job-name="q3tv_${tag}" \
        --output="logs/sdpo_qwen3/train_${tag}_%j.out" \
        --error="logs/sdpo_qwen3/train_${tag}_%j.err" \
        --export=ALL,TAG="$tag",AUX_LOSS="$aux",SIGMOID_COEF="$coef",T_MAX="$tmax",T_MIN="$tmin",NUM_EPOCHS="$NUM_EPOCHS",MAX_SAMPLES="$MAX_SAMPLES",TRAINPATH="$trainpath",TESTPATH="$testpath"${EXTRA_EXPORT} \
        "$SLURM_SCRIPT" | awk '{print $4}')

    echo "  Submitted $tag ($ds) -> $JOB_ID"
}

SELECTED=("$@")

echo "Qwen3 TV loss sweep (L_eagle + coef*TV, epochs=$NUM_EPOCHS, samples=${MAX_SAMPLES:-full})"
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
