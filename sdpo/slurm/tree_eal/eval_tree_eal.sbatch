#!/bin/bash
#SBATCH --job-name=tree_eal_ev
#SBATCH --partition=h200_tandon
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ll5914/logs/tree_eal/eval_%j.out
#SBATCH --error=/scratch/ll5914/logs/tree_eal/eval_%j.err

set -euo pipefail
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD

cd /home/ll5914/Cerebras_SD/SDPO-Speculative-Decoding-Policy-Optimization
mkdir -p /scratch/ll5914/loss_train_q8_q06/tree_eal_pathk16/eval

export HF_HOME=/scratch/ll5914/huggingface
export TOKENIZERS_PARALLELISM=false

DRAFT=/scratch/ll5914/loss_train_q8_q06/tree_eal_pathk16/hf_epoch_5
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} DRAFT=${DRAFT}"
nvidia-smi -L
ls -la "${DRAFT}"

python sdpo/eval_small_lm_tree.py \
  --base-model-path /scratch/ll5914/models/qwen3-8b \
  --draft-model-path "${DRAFT}" \
  --bench-name mt_bench \
  --gamma 7 \
  --top-k 4,3,2,1,1,1,1 \
  --budget 128 \
  --max-new-tokens 256 \
  --num-samples 80 \
  --temperature 1.0 \
  --seed 0 \
  --draft-mode argmax \
  --verify-mode simple \
  --tag tree_eal_q8_q06 \
  --output-dir /scratch/ll5914/loss_train_q8_q06/tree_eal_pathk16/eval
