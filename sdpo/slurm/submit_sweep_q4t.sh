#!/bin/bash
# Submit Qwen3-4B-Thinking EAGLE3 sweep.
# Target: Qwen/Qwen3-4B-Thinking-2507
# Draft: taobao-mnn/Qwen3-4B-Thinking-2507-Eagle
# Uses best acceptance length config from Qwen3-8B experiments.
#
# Usage:
#   bash sdpo/slurm/submit_sweep_q4t.sh                    # submit ALL
#   bash sdpo/slurm/submit_sweep_q4t.sh q4t_best           # submit single
#   bash sdpo/slurm/submit_sweep_q4t.sh --list

set -e
cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization

mkdir -p logs/sdpo_q4t

SLURM_SCRIPT=sdpo/slurm/qwen3_train.slurm
NUM_EPOCHS=${EPOCHS:-3}
MAX_SAMPLES=${MAX_SAMPLES:-""}

BASEPATH=Qwen/Qwen3-4B-Thinking-2507
DRAFTPATH=taobao-mnn/Qwen3-4B-Thinking-2507-Eagle
SAVEROOT=/scratch/tx856/spec_reason/scratch/loss_train_q4t

TRAIN_10K=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_train_10K_aime.jsonl
VAL_10K=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_val_80.jsonl
TRAIN_FULL=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_train_aime.jsonl
VAL_FULL=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_val.jsonl

# Format: "tag|aux_loss|coef|tmax|tmin|dataset|baseline"
# Using best acceptance length config from Qwen3-8B sweep:
# acceptance_length_v2 + coef=0.1 + T=0.1 + lr=1e-6
TASKS=(
    # Baselines
    "q4t_original|sigmoid|0.0|0.1|0.1|10k|eagle_only"
    "q4t_ce_only|sigmoid|0.0|0.1|0.1|10k|ce_only"

    # Best config (V2, T=0.1, coef=0.1)
    "q4t_best|acceptance_length_v2|0.1|0.1|0.1|10k|"
    "q4t_best_full|acceptance_length_v2|0.1|0.1|0.1|full|"

    # Coef sweep around best
    "q4t_coef_0.05|acceptance_length_v2|0.05|0.1|0.1|10k|"
    "q4t_coef_0.2|acceptance_length_v2|0.2|0.1|0.1|10k|"
    "q4t_coef_0.5|acceptance_length_v2|0.5|0.1|0.1|10k|"

    # TV loss variants (next best)
    "q4t_tv_coef_0.1|tv|0.1|0.1|0.1|10k|"
    "q4t_tv_coef_0.5|tv|0.5|0.1|0.1|10k|"
)

if [ "$1" = "--list" ]; then
    echo "Available Qwen3-4B-Thinking tasks:"
    echo "  target: $BASEPATH"
    echo "  draft:  $DRAFTPATH"
    echo ""
    for task in "${TASKS[@]}"; do
        IFS='|' read -r tag aux coef tmax tmin ds bl <<< "$task"
        printf "  %-25s  aux=%-25s coef=%-5s T=%s data=%-4s %s\n" \
            "$tag" "$aux" "$coef" "$tmin" "$ds" "${bl:+baseline=$bl}"
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
        --job-name="q4t_${tag}" \
        --output="logs/sdpo_q4t/train_${tag}_%j.out" \
        --error="logs/sdpo_q4t/train_${tag}_%j.err" \
        --export=ALL,TAG="$tag",AUX_LOSS="$aux",SIGMOID_COEF="$coef",T_MAX="$tmax",T_MIN="$tmin",NUM_EPOCHS="$NUM_EPOCHS",MAX_SAMPLES="$MAX_SAMPLES",TRAINPATH="$trainpath",TESTPATH="$testpath",BASEPATH="$BASEPATH",DRAFTPATH="$DRAFTPATH",SAVEROOT="$SAVEROOT"${EXTRA_EXPORT} \
        "$SLURM_SCRIPT" | awk '{print $4}')

    echo "  Submitted $tag ($ds) -> $JOB_ID"
}

SELECTED=("$@")

echo "Qwen3-4B-Thinking sweep (epochs=$NUM_EPOCHS)"
echo "  target: $BASEPATH"
echo "  draft:  $DRAFTPATH"
echo "================================================================"

count=0
for task in "${TASKS[@]}"; do
    IFS='|' read -r tag aux coef tmax tmin ds bl <<< "$task"

    if [ ${#SELECTED[@]} -gt 0 ]; then
        match=0
        for sel in "${SELECTED[@]}"; do
            [ "$sel" = "$tag" ] && match=1 && break
        done
        [ $match -eq 0 ] && continue
    fi

    submit "$tag" "$aux" "$coef" "$tmax" "$tmin" "$ds" "$bl"
    count=$((count + 1))
done

echo ""
echo "$count jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs in: logs/sdpo_q4t/"
