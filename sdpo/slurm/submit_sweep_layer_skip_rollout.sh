#!/bin/bash
# Submit LayerSkip ROLLOUT training sweep.
# Target: Qwen3-32B, Draft: Qwen3-0.6B (28 transformer layers).
# Uses sdpo/main_layer_skip_rollout.py — on-policy γ-step rollout training
# with V2 cumulative-product loss.
#
# Usage:
#   bash sdpo/slurm/submit_sweep_layer_skip_rollout.sh --list
#   bash sdpo/slurm/submit_sweep_layer_skip_rollout.sh lsr_deep_v2_lr2e6
#   bash sdpo/slurm/submit_sweep_layer_skip_rollout.sh lsr_deep_v2_lr2e6 lsr_deep_v2_lr5e6

set -e
cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
mkdir -p logs/layerskip_rollout

SLURM_SCRIPT=sdpo/slurm/layer_skip_rollout_train.slurm
NUM_EPOCHS=${EPOCHS:-3}
MAX_SAMPLES=${MAX_SAMPLES:-""}
GPUS=${GPUS:-2}

BASEPATH=Qwen/Qwen3-32B
DRAFTPATH=Qwen/Qwen3-0.6B
SAVEROOT=/scratch/tx856/spec_reason/scratch/loss_train_layerskip_rollout

if [ "$GPUS" = "2" ]; then
    DS_CFG=sdpo/sdpo_config_qwen3_bf16_zero3_2gpu.json
    MEM=256G
    CONSTRAINT="a100|h100|h200"
else
    DS_CFG=sdpo/sdpo_config_qwen3_bf16_zero3.json
    MEM=384G
    CONSTRAINT="a100|h100|h200"
fi

# Format: "tag|exit_layers|dataset|lr|coef|tmin|tmax|chunk_size"
# All tasks use KL + V2 rollout loss (no eagle_only / ce_only variants here —
# those are from the teacher-forced recipe in submit_sweep_layer_skip.sh).
TASKS=(
    # ==== Deep-exit rollout on 10K (first check if training signal exists) ====
    "lsr_deep_v2_lr2e6|20,24,26,28|10k|2e-6|0.1|0.1|0.1|16"
    "lsr_deep_v2_lr5e6|20,24,26,28|10k|5e-6|0.1|0.1|0.1|16"

    # ==== Same config on full 68K ====
    "lsr_deep_v2_full_lr2e6|20,24,26,28|full|2e-6|0.1|0.1|0.1|16"
    "lsr_deep_v2_full_lr5e6|20,24,26,28|full|5e-6|0.1|0.1|0.1|16"

    # ==== Single-exit rollout (isolate from multi-exit gradient conflict) ====
    "lsr_e24_v2_lr2e6|24|10k|2e-6|0.1|0.1|0.1|16"
    "lsr_e26_v2_lr2e6|26|10k|2e-6|0.1|0.1|0.1|16"
    "lsr_e28_v2_lr2e6|28|10k|2e-6|0.1|0.1|0.1|16"

    "lsr_e28_v2_full_lr2e6|28|full|2e-6|0.1|0.1|0.1|16"

    # ==== Per-exit rollout (each exit uses own logits for token generation) ====
    "lsr_pe_deep_v2_full_lr2e6|20,24,26,28|full|2e-6|0.1|0.1|0.1|16"
    "lsr_pe_deep_v2_lr2e6|20,24,26,28|10k|2e-6|0.1|0.1|0.1|16"

    # ==== Coef sweep on deep-exit 10K ====
    "lsr_deep_v2_c0.5_lr2e6|20,24,26,28|10k|2e-6|0.5|0.1|0.1|16"
    "lsr_deep_v2_c1.0_lr2e6|20,24,26,28|10k|2e-6|1.0|0.1|0.1|16"
)

if [ "$1" = "--list" ]; then
    echo "Available LayerSkipRollout tasks (target=$BASEPATH draft=$DRAFTPATH):"
    for task in "${TASKS[@]}"; do
        IFS='|' read -r tag exits ds lr coef tmin tmax chunk <<< "$task"
        printf "  %-30s  exits=%-22s  coef=%-4s T=(%s->%s)  lr=%-4s  data=%-4s  chunk=%s\n" \
            "$tag" "$exits" "$coef" "$tmin" "$tmax" "$lr" "$ds" "$chunk"
    done
    exit 0
fi

submit() {
    local tag=$1 exits=$2 ds=$3 lr=$4 coef=$5 tmin=$6 tmax=$7 chunk=$8

    EXTRA_EXPORT=""
    [ -n "$lr" ] && EXTRA_EXPORT=",LR=$lr"
    EXTRA_EXPORT="${EXTRA_EXPORT},SIGMOID_COEF=$coef,T_MIN=$tmin,T_MAX=$tmax,CHUNK_SIZE=$chunk"

    # EXIT_LAYERS must be exported OUTSIDE --export (sbatch --export parses
    # commas as variable separators). Rely on --export=ALL inheritance.
    export EXIT_LAYERS="$exits"

    if [ "$ds" = "full" ]; then
        EXTRA_EXPORT="${EXTRA_EXPORT},TRAINPATH=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_train.jsonl,TESTPATH=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_val.jsonl"
    fi

    JOB_ID=$(sbatch \
        --job-name="${tag}" \
        --output="logs/layerskip_rollout/train_${tag}_%j.out" \
        --error="logs/layerskip_rollout/train_${tag}_%j.err" \
        --gres=gpu:$GPUS \
        --constraint="$CONSTRAINT" \
        --mem=$MEM \
        --export=ALL,TAG="$tag",NUM_EPOCHS="$NUM_EPOCHS",MAX_SAMPLES="$MAX_SAMPLES",BASEPATH="$BASEPATH",DRAFTPATH="$DRAFTPATH",SAVEROOT="$SAVEROOT",CONDA_ENV=longreason,DEEPSPEED_CONFIG=$DS_CFG${EXTRA_EXPORT} \
        "$SLURM_SCRIPT" | awk '{print $4}')

    echo "  Submitted $tag (exits=$exits, $ds, lr=$lr, chunk=$chunk) -> $JOB_ID"
}

SELECTED=("$@")

echo "LayerSkipRollout sweep (target: $BASEPATH → draft: $DRAFTPATH)"
echo "epochs=$NUM_EPOCHS, samples=${MAX_SAMPLES:-full}, GPUs=$GPUS, $DS_CFG"
echo "================================================================"

count=0
for task in "${TASKS[@]}"; do
    IFS='|' read -r tag exits ds lr coef tmin tmax chunk <<< "$task"

    if [ ${#SELECTED[@]} -gt 0 ]; then
        match=0
        for sel in "${SELECTED[@]}"; do
            [ "$sel" = "$tag" ] && match=1 && break
        done
        [ $match -eq 0 ] && continue
    fi

    submit "$tag" "$exits" "$ds" "$lr" "$coef" "$tmin" "$tmax" "$chunk"
    count=$((count + 1))
done

echo ""
echo "$count jobs submitted. Monitor: squeue -u \$USER"
echo "Logs in: logs/layerskip_rollout/"
