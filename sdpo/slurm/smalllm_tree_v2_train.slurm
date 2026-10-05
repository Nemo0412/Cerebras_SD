#!/bin/bash

#SBATCH --job-name=slm_tree_v2_train
#SBATCH --time=7-00:00:00
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
#SBATCH --output=logs/smalllm_tree/v2_train_%j.out
#SBATCH --error=logs/smalllm_tree/v2_train_%j.err

cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
mkdir -p logs/smalllm_tree

source ~/.bashrc
conda activate ${CONDA_ENV:-longreason}

BASEPATH=${BASEPATH:-Qwen/Qwen3-8B}
DRAFTPATH=${DRAFTPATH:-Qwen/Qwen3-0.6B}
NUM_EPOCHS=${NUM_EPOCHS:-6}
SAVEROOT=${SAVEROOT:-/scratch/tx856/spec_reason/scratch/loss_train_smalllm_tree_v2}
ANCHOR_KL_COEF=${ANCHOR_KL_COEF:-1.0}
EAL_AUX_COEF=${EAL_AUX_COEF:-0.1}
MAX_LEN=${MAX_LEN:-2048}
TOP_K_PER_DEPTH=${TOP_K_PER_DEPTH:-4,3,2,1,1,1,1}

SAMPLE_FLAG=""
if [ -n "$MAX_SAMPLES" ]; then
    SAMPLE_FLAG="--max_train_samples $MAX_SAMPLES"
fi

LR_FLAG=""
if [ -n "$LR" ]; then
    LR_FLAG="--lr $LR"
fi

echo "=============================="
echo "  SmallLM Tree V2 Train: $TAG"
echo "  target=$BASEPATH draft=$DRAFTPATH"
echo "  top_k_per_depth=$TOP_K_PER_DEPTH"
echo "  anchor_kl_coef=$ANCHOR_KL_COEF eal_aux_coef=$EAL_AUX_COEF"
echo "  max_len=$MAX_LEN epochs=$NUM_EPOCHS samples=${MAX_SAMPLES:-full}"
echo "  trainpath=${TRAINPATH:-default}"
echo "  saveroot=$SAVEROOT"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

export MASTER_PORT=$((29500 + RANDOM % 1000))

deepspeed --master_port $MASTER_PORT sdpo/main_small_lm_tree_v2.py \
    --basepath "$BASEPATH" \
    --draftpath "$DRAFTPATH" \
    --trainpath "${TRAINPATH:-/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/sharegpt_qwen3_8b_regen.jsonl}" \
    --testpath "${TESTPATH:-/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_val_80.jsonl}" \
    --deepspeed_config "${DEEPSPEED_CONFIG:-sdpo/sdpo_config_qwen3_bf16_zero3_2gpu.json}" \
    --savedir "${SAVEROOT}/${TAG}" \
    --anchor_kl_coef "$ANCHOR_KL_COEF" \
    --eal_aux_coef "$EAL_AUX_COEF" \
    --max_len "$MAX_LEN" \
    --top_k_per_depth "$TOP_K_PER_DEPTH" \
    --num_epochs "$NUM_EPOCHS" \
    $LR_FLAG \
    $SAMPLE_FLAG
