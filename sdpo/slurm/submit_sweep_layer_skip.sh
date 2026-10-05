#!/bin/bash
# Submit layer-skip draft training sweep.
# Target: Qwen3-32B   Draft: Qwen3-0.6B (28 transformer layers)
#
# Each task trains a different subset of exit layers. Baseline is "full" exit
# (all layers). Subsets let us see which intermediate layers benefit from
# targeted distillation.
#
# Usage:
#   bash sdpo/slurm/submit_sweep_layer_skip.sh                    # submit all
#   bash sdpo/slurm/submit_sweep_layer_skip.sh ls_best             # single
#   bash sdpo/slurm/submit_sweep_layer_skip.sh --list

set -e
cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
mkdir -p logs/layerskip

SLURM_SCRIPT=sdpo/slurm/layer_skip_train.slurm
NUM_EPOCHS=${EPOCHS:-3}
MAX_SAMPLES=${MAX_SAMPLES:-""}
GPUS=${GPUS:-2}

BASEPATH=Qwen/Qwen3-32B
DRAFTPATH=Qwen/Qwen3-0.6B
SAVEROOT=/scratch/tx856/spec_reason/scratch/loss_train_layerskip

if [ "$GPUS" = "2" ]; then
    DS_CFG=sdpo/sdpo_config_qwen3_bf16_zero3_2gpu.json
    MEM=256G
    CONSTRAINT="a100|h100|h200"
else
    DS_CFG=sdpo/sdpo_config_qwen3_bf16_zero3.json
    MEM=384G
    CONSTRAINT="a100|h100|h200"
fi

# Format: "tag|exit_layers|dataset|lr|loss|coef|tmin|tmax"
# loss: eagle_only / ce_only / v2 / tv   (v2/tv auto-imply KL+aux)
# For eagle_only/ce_only the coef/tmin/tmax fields are ignored.
# Qwen3-0.6B has 28 transformer layers.
TASKS=(
    # ==== All-exits KL distillation baselines ====
    "ls_all_eagle|4,8,12,16,20,24,28|10k|5e-6|eagle_only|0.0|1.0|1.0"
    "ls_all_ce|4,8,12,16,20,24,28|10k|5e-6|ce_only|0.0|0.1|0.1"
    "ls_all_v2|4,8,12,16,20,24,28|10k|5e-6|v2|0.1|0.1|0.1"
    "ls_all_tv|4,8,12,16,20,24,28|10k|5e-6|tv|0.1|0.1|0.1"

    # ==== Single-exit studies with V2 (find the sweet spot exit) ====
    "ls_e12_v2|12|10k|5e-6|v2|0.1|0.1|0.1"
    "ls_e16_v2|16|10k|5e-6|v2|0.1|0.1|0.1"
    "ls_e20_v2|20|10k|5e-6|v2|0.1|0.1|0.1"
    "ls_e24_v2|24|10k|5e-6|v2|0.1|0.1|0.1"
    "ls_full_v2|28|10k|5e-6|v2|0.1|0.1|0.1"

    # ==== Subset: mid+deep, V2 ====
    "ls_e16_20_24_28_v2|16,20,24,28|10k|5e-6|v2|0.1|0.1|0.1"

    # ==== V2 coef sweep on all exits ====
    "ls_all_v2_c0.5|4,8,12,16,20,24,28|10k|5e-6|v2|0.5|0.1|0.1"
    "ls_all_v2_c1.0|4,8,12,16,20,24,28|10k|5e-6|v2|1.0|0.1|0.1"

    # ==== Full 68K dataset variants (deep exits only — the promising ones) ====
    # Based on baseline: exits 4/8/16 α<0.1 hopeless, exits 20-28 worth training.
    "ls_deep_v2_full|20,24,26,28|full|5e-6|v2|0.1|0.1|0.1"
    "ls_deep_eagle_full|20,24,26,28|full|5e-6|eagle_only|0.0|0.1|0.1"

    # Full 68K on ALL exits (expensive; include if curious about shallow exits).
    "ls_all_v2_full|4,8,12,16,20,24,28|full|5e-6|v2|0.1|0.1|0.1"

    # ==== LR sweep on ls_deep_v2_full (exits=20,24,26,28, full 68K, V2) ====
    "ls_deep_v2_full_lr1e6|20,24,26,28|full|1e-6|v2|0.1|0.1|0.1"
    "ls_deep_v2_full_lr2e6|20,24,26,28|full|2e-6|v2|0.1|0.1|0.1"
    "ls_deep_v2_full_lr3e6|20,24,26,28|full|3e-6|v2|0.1|0.1|0.1"

    # ==== Exit subset sweep: 2 exits × 2 configs × 2 LRs = 4 tasks (10K) ====
    "ls_e24_28_lr2e6|24,28|10k|2e-6|v2|0.1|0.1|0.1"
    "ls_e24_28_lr5e6|24,28|10k|5e-6|v2|0.1|0.1|0.1"
    "ls_e26_28_lr2e6|26,28|10k|2e-6|v2|0.1|0.1|0.1"
    "ls_e26_28_lr5e6|26,28|10k|5e-6|v2|0.1|0.1|0.1"

    # ==== Exit subset sweep: 3 exits × 2 configs × 2 LRs = 4 tasks (10K) ====
    "ls_e24_26_28_lr2e6|24,26,28|10k|2e-6|v2|0.1|0.1|0.1"
    "ls_e24_26_28_lr5e6|24,26,28|10k|5e-6|v2|0.1|0.1|0.1"
    "ls_e20_24_28_lr2e6|20,24,28|10k|2e-6|v2|0.1|0.1|0.1"
    "ls_e20_24_28_lr5e6|20,24,28|10k|5e-6|v2|0.1|0.1|0.1"
)

if [ "$1" = "--list" ]; then
    echo "Available LayerSkip tasks (target=$BASEPATH draft=$DRAFTPATH):"
    for task in "${TASKS[@]}"; do
        IFS='|' read -r tag exits ds lr loss coef tmin tmax <<< "$task"
        printf "  %-24s  exits=%-22s  loss=%-11s  coef=%-4s T=(%s->%s)  lr=%s\n" \
            "$tag" "$exits" "$loss" "$coef" "$tmin" "$tmax" "$lr"
    done
    exit 0
fi

submit() {
    local tag=$1 exits=$2 ds=$3 lr=$4 loss=$5 coef=$6 tmin=$7 tmax=$8

    EXTRA_EXPORT=""
    [ -n "$lr" ] && EXTRA_EXPORT=",LR=$lr"

    # EXIT_LAYERS contains commas which sbatch --export would parse as variable
    # separators (same bug we hit with BENCH). Export it in THIS shell so
    # --export=ALL inherits it into the job environment.
    export EXIT_LAYERS="$exits"

    # Loss selection:
    #   eagle_only / ce_only → pass BASELINE; ignore coef/T
    #   v2                   → AUX_LOSS=acceptance_length_v2, pass coef + T
    #   tv                   → AUX_LOSS=tv, pass coef + T
    case "$loss" in
        eagle_only|ce_only)
            EXTRA_EXPORT="${EXTRA_EXPORT},BASELINE=$loss"
            ;;
        v2)
            EXTRA_EXPORT="${EXTRA_EXPORT},AUX_LOSS=acceptance_length_v2,SIGMOID_COEF=$coef,T_MIN=$tmin,T_MAX=$tmax"
            ;;
        tv)
            EXTRA_EXPORT="${EXTRA_EXPORT},AUX_LOSS=tv,SIGMOID_COEF=$coef,T_MIN=$tmin,T_MAX=$tmax"
            ;;
    esac

    # Full-68K training data path (optional).
    if [ "$ds" = "full" ]; then
        EXTRA_EXPORT="${EXTRA_EXPORT},TRAINPATH=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_train.jsonl,TESTPATH=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_val.jsonl"
    fi

    JOB_ID=$(sbatch \
        --job-name="ls_${tag}" \
        --output="logs/layerskip/train_${tag}_%j.out" \
        --error="logs/layerskip/train_${tag}_%j.err" \
        --gres=gpu:$GPUS \
        --constraint="$CONSTRAINT" \
        --mem=$MEM \
        --export=ALL,TAG="$tag",NUM_EPOCHS="$NUM_EPOCHS",MAX_SAMPLES="$MAX_SAMPLES",BASEPATH="$BASEPATH",DRAFTPATH="$DRAFTPATH",SAVEROOT="$SAVEROOT",CONDA_ENV=longreason,DEEPSPEED_CONFIG=$DS_CFG${EXTRA_EXPORT} \
        "$SLURM_SCRIPT" | awk '{print $4}')

    echo "  Submitted $tag (exits=$exits, loss=$loss, $ds, lr=$lr) -> $JOB_ID"
}

SELECTED=("$@")

echo "LayerSkip sweep (target: $BASEPATH → draft: $DRAFTPATH)"
echo "epochs=$NUM_EPOCHS, samples=${MAX_SAMPLES:-full}, GPUs=$GPUS, ZeRO-3 ($DS_CFG)"
echo "================================================================"

count=0
for task in "${TASKS[@]}"; do
    IFS='|' read -r tag exits ds lr loss coef tmin tmax <<< "$task"

    if [ ${#SELECTED[@]} -gt 0 ]; then
        match=0
        for sel in "${SELECTED[@]}"; do
            [ "$sel" = "$tag" ] && match=1 && break
        done
        [ $match -eq 0 ] && continue
    fi

    submit "$tag" "$exits" "$ds" "$lr" "$loss" "$coef" "$tmin" "$tmax"
    count=$((count + 1))
done

echo ""
echo "$count jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs in: logs/layerskip/"
