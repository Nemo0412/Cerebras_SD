#!/bin/bash

#SBATCH --job-name=acclength
#SBATCH --time=0-12:00:00
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
#SBATCH --output=logs/sdpo_acclength/train_%j.out
#SBATCH --error=logs/sdpo_acclength/train_%j.err

cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization

source ~/.bashrc
conda activate longreason

# Variables set by launcher: TAG, SIGMOID_COEF, T_MAX, T_MIN, NUM_EPOCHS, MAX_SAMPLES, AUX_LOSS
SIGMOID_COEF=${SIGMOID_COEF:-0.1}
T_MAX=${T_MAX:-5.0}
T_MIN=${T_MIN:-1.0}
NUM_EPOCHS=${NUM_EPOCHS:-3}
AUX_LOSS=${AUX_LOSS:-acceptance_length}

SAMPLE_FLAG=""
if [ -n "$MAX_SAMPLES" ]; then
    SAMPLE_FLAG="--max_train_samples $MAX_SAMPLES"
fi

BASELINE_FLAG=""
if [ -n "$BASELINE" ]; then
    BASELINE_FLAG="--baseline $BASELINE"
fi

echo "=============================="
echo "  AccLength Train: $TAG"
echo "  aux_loss=$AUX_LOSS coef=$SIGMOID_COEF T=($T_MIN→$T_MAX)"
echo "  epochs=$NUM_EPOCHS samples=${MAX_SAMPLES:-full}"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

export MASTER_PORT=$((29500 + RANDOM % 1000))

deepspeed --master_port $MASTER_PORT sdpo/main.py \
    --basepath meta-llama/Llama-3.1-8B-Instruct \
    --draftpath yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
    --trainpath /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_train_10K.jsonl \
    --testpath /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_val_80.jsonl \
    --deepspeed_config sdpo/sdpo_config.json \
    --config_path sdpo/config.json \
    --savedir "/scratch/tx856/spec_reason/scratch/loss_train/${TAG}" \
    --aux_loss "$AUX_LOSS" \
    --sigmoid_coef "$SIGMOID_COEF" \
    --T_max "$T_MAX" --T_min "$T_MIN" \
    --gamma 7 \
    --num_epochs "$NUM_EPOCHS" \
    $BASELINE_FLAG \
    $SAMPLE_FLAG
