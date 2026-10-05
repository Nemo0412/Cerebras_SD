#!/bin/bash

#SBATCH --job-name=qwen3_al
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
#SBATCH --output=logs/sdpo_qwen3/train_%j.out
#SBATCH --error=logs/sdpo_qwen3/train_%j.err

cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization

source ~/.bashrc
conda activate ${CONDA_ENV:-longreason_eagle3}

# Move temp caches to local-scratch to avoid NFS busy/SIGBUS during cleanup.
# Each rank gets its own temp dir under /tmp (local to node).
export TMPDIR=/tmp/sdpo_${SLURM_JOB_ID:-$$}
mkdir -p "$TMPDIR"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export HF_HOME="${HF_HOME:-/scratch/yf3005/.huggingface}"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
export TORCH_HOME="$TMPDIR/torch"
# multiprocess.shared_memory uses /dev/shm; verify there's space
ls /dev/shm > /dev/null 2>&1 || mkdir -p /dev/shm

# Variables set by launcher: TAG, SIGMOID_COEF, T_MAX, T_MIN, NUM_EPOCHS, MAX_SAMPLES, AUX_LOSS
SIGMOID_COEF=${SIGMOID_COEF:-0.1}
T_MAX=${T_MAX:-5.0}
T_MIN=${T_MIN:-1.0}
PROB_T_MAX=${PROB_T_MAX:-1.0}
PROB_T_MIN=${PROB_T_MIN:-1.0}
NUM_EPOCHS=${NUM_EPOCHS:-3}
AUX_LOSS=${AUX_LOSS:-acceptance_length_v2}
TRAINPATH=${TRAINPATH:-/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_train_10K.jsonl}
TESTPATH=${TESTPATH:-/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_val_80.jsonl}
BASEPATH=${BASEPATH:-Qwen/Qwen3-8B}
DRAFTPATH=${DRAFTPATH:-AngelSlim/Qwen3-8B_eagle3}
SAVEROOT=${SAVEROOT:-/scratch/tx856/spec_reason/scratch/loss_train_qwen3}

# GRPO env vars (default off / SFT only)
GRPO_COEF=${GRPO_COEF:-0.0}
GRPO_MODE=${GRPO_MODE:-window}
GRPO_REWARD=${GRPO_REWARD:-hard}
GRPO_K_GROUPS=${GRPO_K_GROUPS:-8}
GRPO_M=${GRPO_M:-4}
GRPO_EPS=${GRPO_EPS:-0.2}
GRPO_SAMPLE_TEMP=${GRPO_SAMPLE_TEMP:-1.0}

SAMPLE_FLAG=""
if [ -n "$MAX_SAMPLES" ]; then
    SAMPLE_FLAG="--max_train_samples $MAX_SAMPLES"
fi

BASELINE_FLAG=""
if [ -n "$BASELINE" ]; then
    BASELINE_FLAG="--baseline $BASELINE"
fi

echo "=============================="
echo "  Qwen3 Train: $TAG"
echo "  aux_loss=$AUX_LOSS coef=$SIGMOID_COEF T=($T_MIN→$T_MAX)"
echo "  grpo_coef=$GRPO_COEF mode=$GRPO_MODE reward=$GRPO_REWARD K=$GRPO_K_GROUPS m=$GRPO_M"
echo "  epochs=$NUM_EPOCHS samples=${MAX_SAMPLES:-full}"
echo "  draftpath=$DRAFTPATH"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

export MASTER_PORT=$((29500 + RANDOM % 1000))

deepspeed --master_port $MASTER_PORT sdpo/main.py \
    --basepath "$BASEPATH" \
    --draftpath "$DRAFTPATH" \
    --trainpath "$TRAINPATH" \
    --testpath "$TESTPATH" \
    --deepspeed_config sdpo/sdpo_config_qwen3.json \
    --config_path sdpo/config_qwen3.json \
    --savedir "${SAVEROOT}/${TAG}" \
    --aux_loss "$AUX_LOSS" \
    --sigmoid_coef "$SIGMOID_COEF" \
    --T_max "$T_MAX" --T_min "$T_MIN" \
    --prob_T_max "$PROB_T_MAX" --prob_T_min "$PROB_T_MIN" \
    --gamma 7 \
    --num_epochs "$NUM_EPOCHS" \
    --grpo_coef "$GRPO_COEF" \
    --grpo_mode "$GRPO_MODE" \
    --grpo_reward "$GRPO_REWARD" \
    --grpo_k_groups "$GRPO_K_GROUPS" \
    --grpo_m "$GRPO_M" \
    --grpo_eps "$GRPO_EPS" \
    --grpo_sample_temp "$GRPO_SAMPLE_TEMP" \
    $BASELINE_FLAG \
    $SAMPLE_FLAG

# Cleanup local temp on exit
rm -rf "$TMPDIR" 2>/dev/null || true
