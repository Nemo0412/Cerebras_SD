#!/bin/bash

#SBATCH --job-name=slr_train
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --constraint="a100|h100|h200"
#SBATCH --requeue
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=tx856@nyu.edu
#SBATCH --output=logs/smalllm_rollout/train_%j.out
#SBATCH --error=logs/smalllm_rollout/train_%j.err

cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
mkdir -p logs/smalllm_rollout

source ~/.bashrc
conda activate ${CONDA_ENV:-longreason}

BASEPATH=${BASEPATH:-Qwen/Qwen3-32B}
DRAFTPATH=${DRAFTPATH:-Qwen/Qwen3-0.6B}
NUM_EPOCHS=${NUM_EPOCHS:-3}
SAVEROOT=${SAVEROOT:-/scratch/tx856/spec_reason/scratch/loss_train_smalllm_rollout}
SIGMOID_COEF=${SIGMOID_COEF:-0.1}
T_MAX=${T_MAX:-0.1}
T_MIN=${T_MIN:-0.1}
ANCHOR=${ANCHOR:-kl}
AUX_LOSS=${AUX_LOSS:-v2}
ANCHOR_WEIGHT=${ANCHOR_WEIGHT:-none}
AUX_WEIGHT=${AUX_WEIGHT:-pow08}
ON_POLICY_TARGET=${ON_POLICY_TARGET:-0}
SOFT_ROLLOUT=${SOFT_ROLLOUT:-0}
CHAIN_ANCHOR_COEF=${CHAIN_ANCHOR_COEF:-1.0}
ROLLOUT_TRUNCATE_AT_REJECT=${ROLLOUT_TRUNCATE_AT_REJECT:-0}
ROLLOUT_LOSS_MODE=${ROLLOUT_LOSS_MODE:-chain}
MAX_LEN=${MAX_LEN:-1024}

ON_POLICY_FLAG=""
if [ "$ON_POLICY_TARGET" = "1" ]; then
    ON_POLICY_FLAG="--on_policy_target"
fi
SOFT_FLAG=""
if [ "$SOFT_ROLLOUT" = "1" ]; then
    SOFT_FLAG="--soft_rollout"
fi
TRUNC_FLAG=""
if [ "$ROLLOUT_TRUNCATE_AT_REJECT" = "1" ]; then
    TRUNC_FLAG="--rollout_truncate_at_reject"
fi

SAMPLE_FLAG=""
if [ -n "$MAX_SAMPLES" ]; then
    SAMPLE_FLAG="--max_train_samples $MAX_SAMPLES"
fi

LR_FLAG=""
if [ -n "$LR" ]; then
    LR_FLAG="--lr $LR"
fi

BASELINE_FLAG=""
if [ -n "$BASELINE" ]; then
    BASELINE_FLAG="--baseline $BASELINE"
fi

echo "=============================="
echo "  SmallLMRollout Train: $TAG"
echo "  target=$BASEPATH draft=$DRAFTPATH"
echo "  anchor=$ANCHOR aux_loss=$AUX_LOSS"
echo "  anchor_weight=$ANCHOR_WEIGHT aux_weight=$AUX_WEIGHT"
echo "  on_policy_target=$ON_POLICY_TARGET soft_rollout=$SOFT_ROLLOUT"
echo "  coef=$SIGMOID_COEF T=($T_MIN->$T_MAX) lr=${LR:-default}"
echo "  gamma=${GAMMA:-7} max_len=$MAX_LEN"
echo "  epochs=$NUM_EPOCHS samples=${MAX_SAMPLES:-full}"
echo "  trainpath=${TRAINPATH:-default}"
echo "  saveroot=$SAVEROOT"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

export MASTER_PORT=$((29500 + RANDOM % 1000))

deepspeed --master_port $MASTER_PORT sdpo/main_small_lm_rollout.py \
    --basepath "$BASEPATH" \
    --draftpath "$DRAFTPATH" \
    --trainpath "${TRAINPATH:-/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_train_10K.jsonl}" \
    --testpath "${TESTPATH:-/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_val_80.jsonl}" \
    --deepspeed_config "${DEEPSPEED_CONFIG:-sdpo/sdpo_config_qwen3_bf16_zero3_2gpu.json}" \
    --savedir "${SAVEROOT}/${TAG}" \
    --sigmoid_coef "$SIGMOID_COEF" \
    --T_max "$T_MAX" --T_min "$T_MIN" \
    --anchor "$ANCHOR" \
    --aux_loss "$AUX_LOSS" \
    --anchor_weight "$ANCHOR_WEIGHT" \
    --aux_weight "$AUX_WEIGHT" \
    $ON_POLICY_FLAG \
    $SOFT_FLAG \
    $TRUNC_FLAG \
    --chain_anchor_coef "$CHAIN_ANCHOR_COEF" \
    --rollout_loss_mode "$ROLLOUT_LOSS_MODE" \
    --gamma "${GAMMA:-7}" \
    --max_len "$MAX_LEN" \
    --num_epochs "$NUM_EPOCHS" \
    $BASELINE_FLAG \
    $LR_FLAG \
    $SAMPLE_FLAG
