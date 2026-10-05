#!/bin/bash
# Submit Small LM sweep: Qwen3-8B target + Qwen3-0.6B draft.
#
# Usage:
#   bash sdpo/slurm/submit_sweep_smalllm_06b.sh                    # submit ALL
#   bash sdpo/slurm/submit_sweep_smalllm_06b.sh sl06_coef_0.1      # submit single
#   bash sdpo/slurm/submit_sweep_smalllm_06b.sh --list             # list tasks
#
# Environment variables:
#   EPOCHS=1 MAX_SAMPLES=500 bash sdpo/slurm/submit_sweep_smalllm_06b.sh sl06_coef_0.1

set -e
cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization

mkdir -p logs/sdpo_smalllm_06b

SLURM_SCRIPT=sdpo/slurm/smalllm_06b_train.slurm
NUM_EPOCHS=${EPOCHS:-3}
MAX_SAMPLES=${MAX_SAMPLES:-""}

# Format: "tag|coef|tmax|tmin|lr|baseline|dataset|aux|anchor"
# aux empty = acceptance_length_v2 (default), can be 'v2' | 'v4' | 'tv'
# anchor empty = kl (default), can be 'ce'
TRAIN_10K=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_train_10K.jsonl
VAL_10K=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_val_80.jsonl
TRAIN_FULL=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_train.jsonl
VAL_FULL=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_val.jsonl

TASKS=(
    # ─── 10K dataset ─────────────────────────────────
    "sl06_eagle_only|0.0|0.1|0.1|1e-5|eagle_only|10k"
    # Coef sweep
    "sl06_coef_0.01|0.01|0.1|0.1|1e-5||10k"
    "sl06_coef_0.05|0.05|0.1|0.1|1e-5||10k"
    "sl06_coef_0.1|0.1|0.1|0.1|1e-5||10k"
    "sl06_coef_0.5|0.5|0.1|0.1|1e-5||10k"
    "sl06_coef_1.0|1.0|0.1|0.1|1e-5||10k"
    # LR sweep
    "sl06_lr_1e-7|0.1|0.1|0.1|1e-7||10k"
    "sl06_lr_1e-6|0.1|0.1|0.1|1e-6||10k"
    "sl06_lr_5e-6|0.1|0.1|0.1|5e-6||10k"
    "sl06_lr_1e-5|0.1|0.1|0.1|1e-5||10k"
    "sl06_lr_3e-5|0.1|0.1|0.1|3e-5||10k"
    # Temperature sweep
    "sl06_T_0.1|0.1|0.1|0.1|1e-5||10k"
    "sl06_T_0.2|0.1|0.2|0.2|1e-5||10k"
    "sl06_T_0.5|0.1|0.5|0.5|1e-5||10k"
    # Eagle-only LR sweep
    "sl06_eo_lr_1e-6|0.0|0.1|0.1|1e-6|eagle_only|10k"
    "sl06_eo_lr_1e-5|0.0|0.1|0.1|1e-5|eagle_only|10k"

    # ─── Full 68K dataset ────────────────────────────
    "sl06_full_eagle_only|0.0|0.1|0.1|1e-5|eagle_only|full"
    "sl06_full_coef_0.1|0.1|0.1|0.1|1e-5||full"
    "sl06_full_coef_0.5|0.5|0.1|0.1|1e-5||full"
    "sl06_full_coef_1.0|1.0|0.1|0.1|1e-5||full"
    "sl06_full_lr_1e-6|0.1|0.1|0.1|1e-6||full"
    "sl06_full_lr_1e-5|0.1|0.1|0.1|1e-5||full"
    "sl06_full_eo_lr_1e-5|0.0|0.1|0.1|1e-5|eagle_only|full"

    # ─── TV loss (KL + coef * TV, 10K) ───────────────
    "sl06_tv_coef_0.1|0.1|0.1|0.1|1e-6||10k|tv"
    "sl06_tv_coef_0.5|0.5|0.1|0.1|1e-6||10k|tv"
    "sl06_tv_coef_1.0|1.0|0.1|0.1|1e-6||10k|tv"
    "sl06_tv_coef_2.0|2.0|0.1|0.1|1e-6||10k|tv"

    # ─── TV loss (full 68K) ──────────────────────────
    "sl06_full_tv_coef_0.1|0.1|0.1|0.1|1e-6||full|tv"
    "sl06_full_tv_coef_0.5|0.5|0.1|0.1|1e-6||full|tv"
    "sl06_full_tv_coef_1.0|1.0|0.1|0.1|1e-6||full|tv"

    # ─── CE-only baseline (hard cross-entropy) ──────
    "sl06_ce_only|0.0|0.1|0.1|1e-6|ce_only|10k|"
    "sl06_ce_only_lr_1e-7|0.0|0.1|0.1|1e-7|ce_only|10k|"
    "sl06_ce_only_lr_5e-6|0.0|0.1|0.1|5e-6|ce_only|10k|"
    "sl06_full_ce_only|0.0|0.1|0.1|1e-6|ce_only|full|"

    # ─── CE + aux combos (anchor=ce, same best-known coef/T/lr) ─────
    # CE + V2 (coef 0.1 from sl06_lr_1e-6)
    "sl06_cev2_coef_0.1|0.1|0.1|0.1|1e-6||10k|v2|ce"
    "sl06_cev2_coef_0.5|0.5|0.1|0.1|1e-6||10k|v2|ce"
    # CE + V4 (β = 2σ(gap/T), no 0.5 floor)
    "sl06_cev4_coef_0.1|0.1|0.1|0.1|1e-6||10k|v4|ce"
    "sl06_cev4_coef_0.2|0.2|0.1|0.1|1e-6||10k|v4|ce"
    # CE + TV (coef 0.5 from sl06_tv_coef_0.5)
    "sl06_cetv_coef_0.1|0.1|0.1|0.1|1e-6||10k|tv|ce"
    "sl06_cetv_coef_0.5|0.5|0.1|0.1|1e-6||10k|tv|ce"
)

if [ "$1" = "--list" ]; then
    echo "Available Qwen3-0.6B tasks:"
    for task in "${TASKS[@]}"; do
        IFS='|' read -r tag coef tmax tmin lr bl ds aux anchor <<< "$task"
        printf "  %-25s  coef=%-5s lr=%-8s data=%-4s aux=%-22s anchor=%-2s %s\n" \
            "$tag" "$coef" "$lr" "$ds" "${aux:-v2}" "${anchor:-kl}" "${bl:+baseline=$bl}"
    done
    exit 0
fi

submit() {
    local tag=$1 coef=$2 tmax=$3 tmin=$4 lr=$5 bl=$6 ds=$7 aux=$8 anchor=$9

    EXTRA_EXPORT=""
    [ -n "$bl" ] && EXTRA_EXPORT=",BASELINE=$bl"
    [ -n "$aux" ] && EXTRA_EXPORT="${EXTRA_EXPORT},AUX_LOSS=$aux"
    [ -n "$anchor" ] && EXTRA_EXPORT="${EXTRA_EXPORT},ANCHOR=$anchor"

    if [ "$ds" = "full" ]; then
        EXTRA_EXPORT="${EXTRA_EXPORT},TRAINPATH=$TRAIN_FULL,TESTPATH=$VAL_FULL"
    fi

    JOB_ID=$(sbatch \
        --job-name="sl06_${tag}" \
        --output="logs/sdpo_smalllm_06b/train_${tag}_%j.out" \
        --error="logs/sdpo_smalllm_06b/train_${tag}_%j.err" \
        --export=ALL,TAG="$tag",SIGMOID_COEF="$coef",T_MAX="$tmax",T_MIN="$tmin",LR="$lr",NUM_EPOCHS="$NUM_EPOCHS",MAX_SAMPLES="$MAX_SAMPLES"${EXTRA_EXPORT} \
        "$SLURM_SCRIPT" | awk '{print $4}')

    echo "  Submitted $tag ($ds) -> $JOB_ID"
}

SELECTED=("$@")

echo "SmallLM 0.6B sweep (Qwen3-8B→0.6B, epochs=$NUM_EPOCHS, samples=${MAX_SAMPLES:-full})"
echo "================================================================"

count=0
for task in "${TASKS[@]}"; do
    IFS='|' read -r tag coef tmax tmin lr bl ds aux anchor <<< "$task"

    if [ ${#SELECTED[@]} -gt 0 ]; then
        match=0
        for sel in "${SELECTED[@]}"; do
            [ "$sel" = "$tag" ] && match=1 && break
        done
        [ $match -eq 0 ] && continue
    fi

    submit "$tag" "$coef" "$tmax" "$tmin" "$lr" "$bl" "$ds" "$aux" "$anchor"
    count=$((count + 1))
done

echo ""
echo "$count jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs in: logs/sdpo_smalllm_06b/"
