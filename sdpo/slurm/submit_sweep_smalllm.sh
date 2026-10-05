#!/bin/bash
# Submit Small LM draft training sweep (Qwen3-8B target + Qwen3-1.7B draft).
#
# Usage:
#   bash sdpo/slurm/submit_sweep_smalllm.sh                    # submit ALL
#   bash sdpo/slurm/submit_sweep_smalllm.sh sl_coef_0.1        # submit single
#   bash sdpo/slurm/submit_sweep_smalllm.sh --list             # list tasks
#
# Environment variables:
#   EPOCHS=1 MAX_SAMPLES=500 bash sdpo/slurm/submit_sweep_smalllm.sh sl_coef_0.1

set -e
cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization

mkdir -p logs/sdpo_smalllm

SLURM_SCRIPT=sdpo/slurm/smalllm_train.slurm
NUM_EPOCHS=${EPOCHS:-3}
MAX_SAMPLES=${MAX_SAMPLES:-""}

# Format: "tag|coef|tmax|tmin|baseline|dataset|lr|aux"
# aux empty = acceptance_length_v2 (default), can be 'tv'
TRAIN_10K=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_train_10K.jsonl
VAL_10K=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_val_80.jsonl
TRAIN_FULL=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_train.jsonl
VAL_FULL=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_val.jsonl

TASKS=(
    # ─── 10K dataset ─────────────────────────────────
    "sl_eagle_only|0.0|1.0|1.0|eagle_only|10k|"
    "sl_no_train|0.0|1.0|1.0||10k|"
    # Coef sweep
    "sl_coef_0.01|0.01|1.0|1.0||10k|"
    "sl_coef_0.05|0.05|1.0|1.0||10k|"
    "sl_coef_0.1|0.1|1.0|1.0||10k|"
    "sl_coef_0.2|0.2|1.0|1.0||10k|"
    "sl_coef_0.5|0.5|1.0|1.0||10k|"
    "sl_coef_1.0|1.0|1.0|1.0||10k|"
    # Temperature sweep
    "sl_T_0.1|0.1|0.1|0.1||10k|"
    "sl_T_0.5|0.1|0.5|0.5||10k|"
    "sl_T_2.0|0.1|2.0|2.0||10k|"
    "sl_T_5.0|0.1|5.0|5.0||10k|"
    "sl_c0.5_T0.5|0.5|0.5|0.5||10k|"
    "sl_c1.0_T1|1.0|1.0|1.0||10k|"
    # LR sweep (coef=0.1, T=0.1)
    "sl_lr_1e-7|0.1|0.1|0.1||10k|1e-7"
    "sl_lr_5e-7|0.1|0.1|0.1||10k|5e-7"
    "sl_lr_1e-6|0.1|0.1|0.1||10k|1e-6"
    "sl_lr_5e-6|0.1|0.1|0.1||10k|5e-6"
    "sl_lr_1e-5|0.1|0.1|0.1||10k|1e-5"
    "sl_lr_3e-5|0.1|0.1|0.1||10k|3e-5"
    # Eagle-only LR sweep
    "sl_eo_lr_1e-7|0.0|0.1|0.1|eagle_only|10k|1e-7"
    "sl_eo_lr_1e-6|0.0|0.1|0.1|eagle_only|10k|1e-6"
    "sl_eo_lr_5e-6|0.0|0.1|0.1|eagle_only|10k|5e-6"
    "sl_eo_lr_1e-5|0.0|0.1|0.1|eagle_only|10k|1e-5"

    # ─── Full 68K dataset ────────────────────────────
    "sl_full_eagle_only|0.0|1.0|1.0|eagle_only|full|1e-6"
    "sl_full_coef_0.1|0.1|1.0|1.0||full|1e-6"
    "sl_full_coef_0.5|0.5|1.0|1.0||full|1e-6"
    "sl_full_coef_1.0|1.0|1.0|1.0||full|1e-6"
    "sl_full_T_0.1|0.1|0.1|0.1||full|1e-6"
    "sl_full_c0.5_T0.5|0.5|0.5|0.5||full|1e-6"
    # Full dataset LR sweep
    "sl_full_lr_1e-6|0.1|0.1|0.1||full|1e-6"
    "sl_full_lr_5e-6|0.1|0.1|0.1||full|5e-6"
    "sl_full_lr_1e-5|0.1|0.1|0.1||full|1e-5"
    "sl_full_eo_lr_1e-6|0.0|0.1|0.1|eagle_only|full|1e-6"
    "sl_full_eo_lr_1e-5|0.0|0.1|0.1|eagle_only|full|1e-5"

    # ─── TV loss (KL + coef * TV, 10K) ───────────────
    "sl_tv_coef_0.1|0.1|0.1|0.1||10k|1e-6|tv"
    "sl_tv_coef_0.5|0.5|0.1|0.1||10k|1e-6|tv"
    "sl_tv_coef_1.0|1.0|0.1|0.1||10k|1e-6|tv"
    "sl_tv_coef_2.0|2.0|0.1|0.1||10k|1e-6|tv"

    # ─── TV loss (full 68K) ──────────────────────────
    "sl_full_tv_coef_0.1|0.1|0.1|0.1||full|1e-6|tv"
    "sl_full_tv_coef_0.5|0.5|0.1|0.1||full|1e-6|tv"
    "sl_full_tv_coef_1.0|1.0|0.1|0.1||full|1e-6|tv"

    # ─── CE-only baseline (hard cross-entropy with target argmax) ──
    "sl_ce_only|0.0|0.1|0.1|ce_only|10k|1e-6|"
    "sl_ce_only_lr_1e-7|0.0|0.1|0.1|ce_only|10k|1e-7|"
    "sl_ce_only_lr_5e-6|0.0|0.1|0.1|ce_only|10k|5e-6|"
    "sl_full_ce_only|0.0|0.1|0.1|ce_only|full|1e-6|"
)

if [ "$1" = "--list" ]; then
    echo "Available SmallLM tasks:"
    for task in "${TASKS[@]}"; do
        IFS='|' read -r tag coef tmax tmin bl ds lr aux <<< "$task"
        printf "  %-25s  coef=%-5s T=(%s->%s) data=%-4s lr=%-8s aux=%-22s %s\n" "$tag" "$coef" "$tmin" "$tmax" "$ds" "${lr:-default}" "${aux:-acceptance_length_v2}" "${bl:+baseline=$bl}"
    done
    exit 0
fi

submit() {
    local tag=$1 coef=$2 tmax=$3 tmin=$4 bl=$5 ds=$6 lr=$7 aux=$8

    EXTRA_EXPORT=""
    [ -n "$bl" ] && EXTRA_EXPORT=",BASELINE=$bl"
    [ -n "$lr" ] && EXTRA_EXPORT="${EXTRA_EXPORT},LR=$lr"
    [ -n "$aux" ] && EXTRA_EXPORT="${EXTRA_EXPORT},AUX_LOSS=$aux"

    if [ "$ds" = "full" ]; then
        EXTRA_EXPORT="${EXTRA_EXPORT},TRAINPATH=$TRAIN_FULL,TESTPATH=$VAL_FULL"
    fi

    JOB_ID=$(sbatch \
        --job-name="sl_${tag}" \
        --output="logs/sdpo_smalllm/train_${tag}_%j.out" \
        --error="logs/sdpo_smalllm/train_${tag}_%j.err" \
        --export=ALL,TAG="$tag",SIGMOID_COEF="$coef",T_MAX="$tmax",T_MIN="$tmin",NUM_EPOCHS="$NUM_EPOCHS",MAX_SAMPLES="$MAX_SAMPLES"${EXTRA_EXPORT} \
        "$SLURM_SCRIPT" | awk '{print $4}')

    echo "  Submitted $tag ($ds, lr=${lr:-default}) -> $JOB_ID"
}

SELECTED=("$@")

echo "SmallLM sweep (Qwen3-8B→1.7B, epochs=$NUM_EPOCHS, samples=${MAX_SAMPLES:-full})"
echo "================================================================"

count=0
for task in "${TASKS[@]}"; do
    IFS='|' read -r tag coef tmax tmin bl ds lr aux <<< "$task"

    if [ ${#SELECTED[@]} -gt 0 ]; then
        match=0
        for sel in "${SELECTED[@]}"; do
            [ "$sel" = "$tag" ] && match=1 && break
        done
        [ $match -eq 0 ] && continue
    fi

    submit "$tag" "$coef" "$tmax" "$tmin" "$bl" "$ds" "$lr" "$aux"
    count=$((count + 1))
done

echo ""
echo "$count jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs in: logs/sdpo_smalllm/"
