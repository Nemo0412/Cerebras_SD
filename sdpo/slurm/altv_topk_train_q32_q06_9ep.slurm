#!/bin/bash
# Q0.6B → Q32B, 9 epochs from raw Q0.6B.
# Env: AUX_LOSS (default al_tv), TAG_SUFFIX (default baseline_9ep), ALTV_TOPK (default 20)

#SBATCH --job-name=altv_train_q32_q06_9ep
#SBATCH --time=5-23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --constraint="h100|h200"
#SBATCH --requeue
#SBATCH --account=torch_pr_74_tandon_priority
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tx856@nyu.edu
#SBATCH --output=logs/sdpo_smalllm_q32_q06/%x_%j.out
#SBATCH --error=logs/sdpo_smalllm_q32_q06/%x_%j.err

set -o pipefail
REPO=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
cd "$REPO"
mkdir -p logs/sdpo_smalllm_q32_q06
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD"

source ~/.bashrc
conda activate ${CONDA_ENV:-longreason}
set -u
export HF_HOME=${HF_HOME:-/scratch/tx856/.huggingface}
export TORCH_EXTENSIONS_DIR=/scratch/tx856/.cache/torch_extensions

AUX_LOSS=${AUX_LOSS:-al_tv}
TAG_SUFFIX=${TAG_SUFFIX:-baseline_9ep}
ALTV_TOPK=${ALTV_TOPK:-20}

TAG=q32_q06_altv_${TAG_SUFFIX}
SAVEROOT=/scratch/tx856/spec_reason/scratch/loss_train_smalllm_q32_q06
MASTER_PORT=$((29500 + RANDOM % 1000))

echo "===== $TAG train (Q0.6B→Q32B, aux=$AUX_LOSS, 9ep) ====="

deepspeed --master_port $MASTER_PORT sdpo/main_small_lm.py \
    --basepath Qwen/Qwen3-32B \
    --draftpath Qwen/Qwen3-0.6B \
    --trainpath "${TRAINPATH:-data/sharegpt_qwen3_32b_regen.jsonl}" \
    --testpath sdpo/data/mixed_val_80.jsonl \
    --deepspeed_config sdpo/sdpo_config_qwen3_bf16_zero3_2gpu.json \
    --savedir "$SAVEROOT/$TAG" \
    --anchor none --aux_loss "$AUX_LOSS" \
    --altv_topk "$ALTV_TOPK" \
    --anchor_weight none --aux_weight "${AUX_WEIGHT:-uniform}" \
    --sigmoid_coef 1.0 \
    --T_max 0.1 --T_min 0.1 \
    --prob_T_max 1.0 --prob_T_min 1.0 \
    --draft_T_max 1.0 --draft_T_min 1.0 \
    --eal_weight 0.5 \
    --grpo_coef 0.0 \
    --grpo_k_groups 8 --grpo_m 4 --grpo_eps 0.2 \
    --grpo_mode window --grpo_sample_temp 1.0 --grpo_reward hard \
    --gamma 7 --max_len 2048 --num_epochs 9 --seed 42 --lr 1e-6 \
    --save_epochs 1,3,5,7,8 \
    --max_train_samples 60000
