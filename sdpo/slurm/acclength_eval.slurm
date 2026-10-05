#!/bin/bash

#SBATCH --job-name=acclength_eval
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100|h200"
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --mail-type=END
#SBATCH --mail-user=tx856@nyu.edu
#SBATCH --output=logs/sdpo_acclength/eval_%j.out
#SBATCH --error=logs/sdpo_acclength/eval_%j.err

cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization

source ~/.bashrc
conda activate longreason

echo "=============================="
echo "  AccLength Eval: $TAG"
echo "  Checkpoint: $EA_MODEL_PATH"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

python bapo/eval.py \
    --base-model-path meta-llama/Llama-3.1-8B-Instruct \
    --ea-model-path "$EA_MODEL_PATH" \
    --tag "$TAG" \
    --bench-name all \
    --temperature 0.0 \
    --output-dir acclength_eval_results
