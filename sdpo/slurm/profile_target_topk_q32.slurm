#!/bin/bash
# Profile Q32B target top-K distribution (5 samples × 3 bench).

#SBATCH --job-name=profile_target_topk_q32
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --constraint="h100|h200"
#SBATCH --requeue
#SBATCH --account=torch_pr_74_tandon_priority
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tx856@nyu.edu
#SBATCH --output=logs/profile/%x_%j.out
#SBATCH --error=logs/profile/%x_%j.err

set -o pipefail
REPO=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
cd "$REPO"
mkdir -p logs/profile
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD"

source ~/.bashrc
conda activate ${CONDA_ENV:-longreason}
set -u
export HF_HOME=${HF_HOME:-/scratch/tx856/.huggingface}

python sdpo/profile_target_topk.py \
    --target-model Qwen/Qwen3-32B \
    --bench-name mt_bench,gsm8k,humaneval \
    --num-samples 15 --max-new-tokens 256 --temperature 1.0 \
    --output smalllm_tree_eval_results/target_topk_profile_q32_v3.json
