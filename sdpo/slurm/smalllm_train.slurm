#!/bin/bash

#SBATCH --job-name=smalllm
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --constraint="a100|h100|h200"
#SBATCH --requeue
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=tx856@nyu.edu
#SBATCH --output=logs/sdpo_smalllm/train_%j.out
#SBATCH --error=logs/sdpo_smalllm/train_%j.err

cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization

source ~/.bashrc
conda activate ${CONDA_ENV:-longreason}

SIGMOID_COEF=${SIGMOID_COEF:-0.1}
T_MAX=${T_MAX:-0.1}
T_MIN=${T_MIN:-0.1}
NUM_EPOCHS=${NUM_EPOCHS:-3}
BASEPATH=${BASEPATH:-Qwen/Qwen3-8B}
DRAFTPATH=${DRAFTPATH:-Qwen/Qwen3-1.7B}
AUX_LOSS=${AUX_LOSS:-acceptance_length_v2}
SAVEROOT=${SAVEROOT:-/scratch/tx856/spec_reason/scratch/loss_train_smalllm}

SAMPLE_FLAG=""
if [ -n "$MAX_SAMPLES" ]; then
    SAMPLE_FLAG="--max_train_samples $MAX_SAMPLES"
fi

BASELINE_FLAG=""
if [ -n "$BASELINE" ]; then
    BASELINE_FLAG="--baseline $BASELINE"
fi

LR_FLAG=""
if [ -n "$LR" ]; then
    LR_FLAG="--lr $LR"
fi

echo "=============================="
echo "  SmallLM Train: $TAG"
echo "  target=$BASEPATH draft=$DRAFTPATH"
echo "  coef=$SIGMOID_COEF T=($T_MIN→$T_MAX) lr=${LR:-default}"
echo "  epochs=$NUM_EPOCHS samples=${MAX_SAMPLES:-full}"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

export MASTER_PORT=$((29500 + RANDOM % 1000))

deepspeed --master_port $MASTER_PORT sdpo/main_small_lm.py \
    --basepath "$BASEPATH" \
    --draftpath "$DRAFTPATH" \
    --trainpath "${TRAINPATH:-/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_train_10K.jsonl}" \
    --testpath "${TESTPATH:-/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_val_80.jsonl}" \
    --deepspeed_config "${DEEPSPEED_CONFIG:-sdpo/sdpo_config_qwen3.json}" \
    --savedir "${SAVEROOT}/${TAG}" \
    --sigmoid_coef "$SIGMOID_COEF" \
    --T_max "$T_MAX" --T_min "$T_MIN" \
    --aux_loss "$AUX_LOSS" \
    --gamma 7 \
    --max_len 1024 \
    --num_epochs "$NUM_EPOCHS" \
    $BASELINE_FLAG \
    $LR_FLAG \
    $SAMPLE_FLAG
