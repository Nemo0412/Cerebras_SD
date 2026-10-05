#!/bin/bash

#SBATCH --job-name=bapo_eval
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100|h200"
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --mail-type=END
#SBATCH --mail-user=xt2251@nyu.edu
#SBATCH --output=logs/bapo/eval_%j.out
#SBATCH --error=logs/bapo/eval_%j.err

cd /scratch/xt2251/SDPO-Speculative-Decoding-Policy-Optimization

source /scratch/xt2251/miniconda3/bin/activate
conda activate spec

echo "=============================="
echo "  BAPO Eval: $TAG"
echo "  Checkpoint: $EA_MODEL_PATH"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

python bapo/eval.py \
    --base-model-path /scratch/xt2251/models/Llama-3.1-8B-Instruct \
    --ea-model-path "$EA_MODEL_PATH" \
    --tag "$TAG" \
    --bench-name all \
    --temperature 0.0 \
    --output-dir bapo_eval_results
