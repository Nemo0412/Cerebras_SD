#!/bin/bash
# Continue finetune Q1.7B → Q32B from existing state ckpt.
# Env: PRETRAIN, AUX_LOSS, TAG_SUFFIX, ALTV_TOPK (default 20), AUX_WEIGHT (default uniform)

#SBATCH --job-name=altv_ft_q32_q17
#SBATCH --time=23:59:00
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
#SBATCH --output=logs/sdpo_smalllm_q32_q17/%x_%j.out
#SBATCH --error=logs/sdpo_smalllm_q32_q17/%x_%j.err

set -o pipefail
REPO=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
cd "$REPO"
mkdir -p logs/sdpo_smalllm_q32_q17
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD"

source ~/.bashrc
conda activate ${CONDA_ENV:-longreason}
set -u
export HF_HOME=${HF_HOME:-/scratch/tx856/.huggingface}
export TORCH_EXTENSIONS_DIR=/scratch/tx856/.cache/torch_extensions

: ${PRETRAIN:?must set PRETRAIN}
: ${AUX_LOSS:?must set AUX_LOSS}
: ${TAG_SUFFIX:?must set TAG_SUFFIX}
AUX_WEIGHT=${AUX_WEIGHT:-uniform}
[ ! -f "$PRETRAIN/pytorch_model.bin" ] && { echo "MISSING pretrain $PRETRAIN"; exit 1; }

TAG=q32_q17_altv_${TAG_SUFFIX}
SAVEROOT=${SAVEROOT:-/scratch/tx856/spec_reason/scratch/loss_train_smalllm_q32_q17}
MASTER_PORT=$((29500 + RANDOM % 1000))

echo "===== $TAG finetune 2ep (Q1.7B→Q32B, aux=$AUX_LOSS, aux_weight=$AUX_WEIGHT) ====="
echo "  init from: $PRETRAIN"

deepspeed --master_port $MASTER_PORT sdpo/main_small_lm.py \
    --basepath Qwen/Qwen3-32B \
    --draftpath "$PRETRAIN" \
    --trainpath "${TRAINPATH:-data/sharegpt_qwen3_32b_regen.jsonl}" \
    --testpath sdpo/data/mixed_val_80.jsonl \
    --deepspeed_config sdpo/sdpo_config_qwen3_bf16_zero3_2gpu.json \
    --savedir "$SAVEROOT/$TAG" \
    --anchor none --aux_loss "$AUX_LOSS" \
    --altv_topk "${ALTV_TOPK:-20}" \
    --anchor_weight none --aux_weight "$AUX_WEIGHT" \
    --sigmoid_coef 1.0 \
    --T_max 0.1 --T_min 0.1 \
    --prob_T_max 1.0 --prob_T_min 1.0 \
    --draft_T_max 1.0 --draft_T_min 1.0 \
    --eal_weight 0.5 \
    --grpo_coef 0.0 \
    --grpo_k_groups 8 --grpo_m 4 --grpo_eps 0.2 \
    --grpo_mode window --grpo_sample_temp 1.0 --grpo_reward hard \
    --gamma 7 --max_len 2048 --num_epochs 2 --seed 42 --lr 1e-6 \
    --save_epochs 0,1 \
    --max_train_samples 60000
