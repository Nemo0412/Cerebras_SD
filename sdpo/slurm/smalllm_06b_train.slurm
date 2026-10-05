#!/bin/bash

#SBATCH --job-name=sl06
#SBATCH --time=1-23:30:00
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
#SBATCH --output=logs/sdpo_smalllm_06b/train_%j.out
#SBATCH --error=logs/sdpo_smalllm_06b/train_%j.err

cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization

source ~/.bashrc
conda activate longreason

SIGMOID_COEF=${SIGMOID_COEF:-0.1}
T_MAX=${T_MAX:-1.0}
T_MIN=${T_MIN:-1.0}
PROB_T_MAX=${PROB_T_MAX:-1.0}
PROB_T_MIN=${PROB_T_MIN:-1.0}
DRAFT_T_MAX=${DRAFT_T_MAX:-1.0}
DRAFT_T_MIN=${DRAFT_T_MIN:-1.0}
EAL_WEIGHT=${EAL_WEIGHT:-0.5}
NUM_EPOCHS=${NUM_EPOCHS:-3}
BASEPATH=${BASEPATH:-Qwen/Qwen3-8B}
DRAFTPATH=${DRAFTPATH:-Qwen/Qwen3-0.6B}
ANCHOR=${ANCHOR:-kl}
AUX_LOSS=${AUX_LOSS:-v2}
ANCHOR_WEIGHT=${ANCHOR_WEIGHT:-none}
AUX_WEIGHT=${AUX_WEIGHT:-pow08}
GRPO_COEF=${GRPO_COEF:-0.0}
GRPO_K_GROUPS=${GRPO_K_GROUPS:-8}
GRPO_M=${GRPO_M:-4}
GRPO_EPS=${GRPO_EPS:-0.2}
GRPO_MODE=${GRPO_MODE:-window}
GRPO_SAMPLE_TEMP=${GRPO_SAMPLE_TEMP:-1.0}
GRPO_REWARD=${GRPO_REWARD:-hard}
GRPO_REWARD_ETA=${GRPO_REWARD_ETA:-1.0}
GRPO_REWARD_EPS=${GRPO_REWARD_EPS:-2.0}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-sdpo/sdpo_config_qwen3.json}
SAVEROOT=${SAVEROOT:-/scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b}
MAX_LEN=${MAX_LEN:-1024}
SEED=${SEED:-42}
SAVE_EPOCHS=${SAVE_EPOCHS:-}

SAVE_EPOCHS_FLAG=""
if [ -n "$SAVE_EPOCHS" ]; then
    SAVE_EPOCHS_FLAG="--save_epochs $SAVE_EPOCHS"
fi

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
echo "  anchor=$ANCHOR aux_loss=$AUX_LOSS"
echo "  anchor_weight=$ANCHOR_WEIGHT aux_weight=$AUX_WEIGHT"
echo "  coef=$SIGMOID_COEF T=($T_MIN→$T_MAX) prob_T=($PROB_T_MAX→$PROB_T_MIN) lr=${LR:-default}"
echo "  gamma=7 max_len=$MAX_LEN"
echo "  epochs=$NUM_EPOCHS samples=${MAX_SAMPLES:-full}"
echo "  trainpath=${TRAINPATH:-default}"
echo "  ds_config=$DEEPSPEED_CONFIG"
echo "  saveroot=$SAVEROOT"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

export MASTER_PORT=$((29500 + RANDOM % 1000))

deepspeed --master_port $MASTER_PORT sdpo/main_small_lm.py \
    --basepath "$BASEPATH" \
    --draftpath "$DRAFTPATH" \
    --trainpath "${TRAINPATH:-/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_train_10K.jsonl}" \
    --testpath "${TESTPATH:-/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_val_80.jsonl}" \
    --deepspeed_config "$DEEPSPEED_CONFIG" \
    --savedir "${SAVEROOT}/${TAG}" \
    --sigmoid_coef "$SIGMOID_COEF" \
    --T_max "$T_MAX" --T_min "$T_MIN" \
    --prob_T_max "$PROB_T_MAX" --prob_T_min "$PROB_T_MIN" \
    --draft_T_max "$DRAFT_T_MAX" --draft_T_min "$DRAFT_T_MIN" \
    --eal_weight "$EAL_WEIGHT" \
    --anchor "$ANCHOR" \
    --aux_loss "$AUX_LOSS" \
    --anchor_weight "$ANCHOR_WEIGHT" \
    --aux_weight "$AUX_WEIGHT" \
    --grpo_coef "$GRPO_COEF" \
    --grpo_k_groups "$GRPO_K_GROUPS" \
    --grpo_m "$GRPO_M" \
    --grpo_eps "$GRPO_EPS" \
    --grpo_mode "$GRPO_MODE" \
    --grpo_sample_temp "$GRPO_SAMPLE_TEMP" \
    --grpo_reward "$GRPO_REWARD" \
    --grpo_reward_eta "$GRPO_REWARD_ETA" \
    --grpo_reward_eps "$GRPO_REWARD_EPS" \
    --gamma 7 \
    --max_len "$MAX_LEN" \
    --num_epochs "$NUM_EPOCHS" \
    --seed "$SEED" \
    $SAVE_EPOCHS_FLAG \
    $BASELINE_FLAG \
    $LR_FLAG \
    $SAMPLE_FLAG
