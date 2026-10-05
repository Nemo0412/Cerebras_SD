#!/bin/bash
# Submit Qwen3-32B target + Qwen3-4B draft sweep.
#
# MEMORY (4x 80GB with ZeRO-3):
#   Qwen3-32B (BF16, frozen): 64GB total, sharded to ~16GB per rank via ZeRO-3
#   Qwen3-4B draft + grads + AdamW optimizer: ~48GB total, sharded to ~12GB per rank
#   Activations ~15GB per rank
#   → ~43GB per GPU. Fits on 4x 80GB A100/H100/H200 comfortably.
#   Uses sdpo_config_qwen3_bf16_zero3.json (ZeRO-3).
#
# Uses best acceptance length config from Qwen3-1.7B experiments.
#
# Usage:
#   bash sdpo/slurm/submit_sweep_smalllm_q3_32b.sh                    # submit ALL
#   bash sdpo/slurm/submit_sweep_smalllm_q3_32b.sh q32_best           # submit single
#   bash sdpo/slurm/submit_sweep_smalllm_q3_32b.sh --list

set -e
cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization

mkdir -p logs/sdpo_smalllm_q3_32b

SLURM_SCRIPT=sdpo/slurm/smalllm_train.slurm
NUM_EPOCHS=${EPOCHS:-3}
MAX_SAMPLES=${MAX_SAMPLES:-""}
GPUS=${GPUS:-4}

if [ "$GPUS" = "2" ]; then
    DS_CFG=sdpo/sdpo_config_qwen3_bf16_zero3_2gpu.json
    MEM=256G
    CONSTRAINT="h100|h200"
else
    DS_CFG=sdpo/sdpo_config_qwen3_bf16_zero3.json
    MEM=384G
    CONSTRAINT="a100|h100|h200"
fi

BASEPATH=Qwen/Qwen3-32B
DRAFTPATH=Qwen/Qwen3-4B
SAVEROOT=/scratch/tx856/spec_reason/scratch/loss_train_smalllm_q3_32b

TRAIN_10K=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_train_10K.jsonl
VAL_10K=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_val_80.jsonl
TRAIN_FULL=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_train.jsonl
VAL_FULL=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_val.jsonl

# Format: "tag|coef|tmax|tmin|baseline|dataset|lr|aux"
# Best config from 1.7B sweep: KL + V2, lr=1e-6
TASKS=(
    # Baselines
    "q32_eagle_only|0.0|1.0|1.0|eagle_only|10k|1e-6|"
    "q32_ce_only|0.0|0.1|0.1|ce_only|10k|1e-6|"

    # Best config (V2)
    "q32_best|0.1|1.0|1.0||10k|1e-6|"
    "q32_best_full|1.0|1.0|1.0||full|1e-6|"

    "q32_best_0.1|0.1|0.1|0.1||10k|1e-6|"
    "q32_best_0.1_lr5e-6|0.1|0.1|0.1||10k|5e-6|"
    "q32_best_0.1_full|0.1|0.1|0.1||full|1e-6|"
    "q32_best_0.1_lr5e-6_full|0.1|0.1|0.1||full|5e-6|"

    # Coef sweep
    "q32_coef_0.5|0.5|1.0|1.0||10k|1e-6|"
    "q32_coef_1.0|1.0|1.0|1.0||10k|1e-6|"

    # TV loss
    "q32_tv_coef_0.1|0.1|0.1|0.1||10k|1e-6|tv"
    "q32_tv_coef_1.0|1.0|0.1|0.1||10k|1e-6|tv"
)

if [ "$1" = "--list" ]; then
    echo "Available Qwen3-32B → 4B tasks:"
    echo "  target: $BASEPATH"
    echo "  draft:  $DRAFTPATH"
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
        --job-name="q32_${tag}" \
        --output="logs/sdpo_smalllm_q3_32b/train_${tag}_%j.out" \
        --error="logs/sdpo_smalllm_q3_32b/train_${tag}_%j.err" \
        --gres=gpu:$GPUS \
        --constraint="$CONSTRAINT" \
        --mem=$MEM \
        --export=ALL,TAG="$tag",SIGMOID_COEF="$coef",T_MAX="$tmax",T_MIN="$tmin",NUM_EPOCHS="$NUM_EPOCHS",MAX_SAMPLES="$MAX_SAMPLES",BASEPATH="$BASEPATH",DRAFTPATH="$DRAFTPATH",SAVEROOT="$SAVEROOT",CONDA_ENV=longreason,DEEPSPEED_CONFIG=$DS_CFG${EXTRA_EXPORT} \
        "$SLURM_SCRIPT" | awk '{print $4}')

    echo "  Submitted $tag ($ds, lr=${lr:-default}) -> $JOB_ID"
}

SELECTED=("$@")

echo "Qwen3-32B → 4B sweep (target: $BASEPATH → draft: $DRAFTPATH)"
echo "epochs=$NUM_EPOCHS, samples=${MAX_SAMPLES:-full}, GPUs=$GPUS, ZeRO-3 ($DS_CFG)"
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
echo "Logs in: logs/sdpo_smalllm_q3_32b/"
