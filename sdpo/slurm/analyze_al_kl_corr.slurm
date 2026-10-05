#!/bin/bash

#SBATCH --job-name=al_kl_corr
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100|h200"
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --output=logs/diagnose/al_kl_corr_%j.out
#SBATCH --error=logs/diagnose/al_kl_corr_%j.err

cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
mkdir -p logs/diagnose

source ~/.bashrc
conda activate ${CONDA_ENV:-longreason_eagle3}

BASEPATH=${BASEPATH:-Qwen/Qwen3-8B}
DRAFT_CKPT=${DRAFT_CKPT}
TAG=${TAG:-al_kl_corr}
MAX_SAMPLES=${MAX_SAMPLES:-50}
GAMMA=${GAMMA:-7}
TESTPATH=${TESTPATH:-sdpo/data/mixed_val_80.jsonl}

echo "=============================="
echo "  AL_KL correlation analysis"
echo "  Tag: $TAG"
echo "  Draft: $DRAFT_CKPT"
echo "  Max samples: $MAX_SAMPLES"
echo "=============================="

python sdpo/diagnose/analyze_al_kl_correlation.py \
    --base-model-path "$BASEPATH" \
    --draft-model-path "$DRAFT_CKPT" \
    --testpath "$TESTPATH" \
    --gamma "$GAMMA" \
    --max-samples "$MAX_SAMPLES" \
    --output "diagnose_al_kl_corr_${TAG}.json"
