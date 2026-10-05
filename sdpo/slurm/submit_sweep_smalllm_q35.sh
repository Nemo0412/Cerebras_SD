#!/bin/bash
# Submit Qwen3.5-9B target + Qwen3.5-0.8B draft sweep.
# Both models use thinking mode (enable_thinking=True).
# Uses best acceptance length config from Qwen3-1.7B experiments.
#
# Usage:
#   bash sdpo/slurm/submit_sweep_smalllm_q35.sh                     # submit ALL
#   bash sdpo/slurm/submit_sweep_smalllm_q35.sh q35_best            # submit single
#   bash sdpo/slurm/submit_sweep_smalllm_q35.sh --list

set -e
cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization

mkdir -p logs/sdpo_smalllm_q35

SLURM_SCRIPT=sdpo/slurm/smalllm_train.slurm
NUM_EPOCHS=${EPOCHS:-3}
MAX_SAMPLES=${MAX_SAMPLES:-""}

BASEPATH=Qwen/Qwen3.5-9B
DRAFTPATH=Qwen/Qwen3.5-0.8B
SAVEROOT=/scratch/tx856/spec_reason/scratch/loss_train_smalllm_q35

TRAIN_10K=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_train_10K_aime.jsonl
VAL_10K=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_val_80.jsonl
TRAIN_FULL=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_train_aime.jsonl
VAL_FULL=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_val.jsonl

# Format: "tag|coef|tmax|tmin|baseline|dataset|lr|aux"
# Best config from 1.7B sweep: KL + acceptance_length_v2, lr=1e-6, coef=1.0 (full) or 0.1 (10K)
TASKS=(
    # Baselines
    "q35_eagle_only|0.0|1.0|1.0|eagle_only|10k|1e-6|"
    "q35_ce_only|0.0|0.1|0.1|ce_only|10k|1e-6|"

    # Best config (V2, lr=1e-6)
    "q35_best|0.1|1.0|1.0||10k|1e-6|"
    "q35_best_full|1.0|1.0|1.0||full|1e-6|"

    # Coef sweep around best
    "q35_coef_0.5|0.5|1.0|1.0||10k|1e-6|"
    "q35_coef_1.0|1.0|1.0|1.0||10k|1e-6|"

    # TV loss variant (next best)
    "q35_tv_coef_0.1|0.1|0.1|0.1||10k|1e-6|tv"
    "q35_tv_coef_1.0|1.0|0.1|0.1||10k|1e-6|tv"

    # Eagle-only + full dataset
    "q35_full_eagle_only|0.0|1.0|1.0|eagle_only|full|1e-6|"
)

if [ "$1" = "--list" ]; then
    echo "Available Qwen3.5 tasks:"
    echo "  target: $BASEPATH  (thinking mode)"
    echo "  draft:  $DRAFTPATH  (thinking mode)"
    echo ""
    for task in "${TASKS[@]}"; do
        IFS='|' read -r tag coef tmax tmin bl ds lr aux <<< "$task"
        printf "  %-25s  coef=%-5s T=(%s->%s) data=%-4s lr=%-8s aux=%-22s %s\n" \
            "$tag" "$coef" "$tmin" "$tmax" "$ds" "${lr:-default}" "${aux:-acceptance_length_v2}" "${bl:+baseline=$bl}"
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
        --job-name="q35_${tag}" \
        --output="logs/sdpo_smalllm_q35/train_${tag}_%j.out" \
        --error="logs/sdpo_smalllm_q35/train_${tag}_%j.err" \
        --export=ALL,TAG="$tag",SIGMOID_COEF="$coef",T_MAX="$tmax",T_MIN="$tmin",NUM_EPOCHS="$NUM_EPOCHS",MAX_SAMPLES="$MAX_SAMPLES",BASEPATH="$BASEPATH",DRAFTPATH="$DRAFTPATH",SAVEROOT="$SAVEROOT",CONDA_ENV=longreason,DEEPSPEED_CONFIG=sdpo/sdpo_config_qwen3_bf16.json${EXTRA_EXPORT} \
        "$SLURM_SCRIPT" | awk '{print $4}')

    echo "  Submitted $tag ($ds, lr=${lr:-default}) -> $JOB_ID"
}

SELECTED=("$@")

echo "Qwen3.5 sweep (target: $BASEPATH → draft: $DRAFTPATH, thinking mode)"
echo "epochs=$NUM_EPOCHS, samples=${MAX_SAMPLES:-full}"
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
echo "Logs in: logs/sdpo_smalllm_q35/"
