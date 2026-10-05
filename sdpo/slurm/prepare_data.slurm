#!/bin/bash

#SBATCH --job-name=prep_data
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --mail-type=END
#SBATCH --mail-user=tx856@nyu.edu
#SBATCH --output=logs/prep_data_%j.out
#SBATCH --error=logs/prep_data_%j.err

cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization

source ~/.bashrc
conda activate longreason

echo "=============================="
echo "  Preparing 68K mixed dataset"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

python sdpo/data/prepare_data.py \
    --output-dir sdpo/data \
    --humaneval-test-jsonl data/humaneval/question.jsonl \
    --sharegpt-train 50000 \
    --sharegpt-val 500 \
    --gsm8k-train 7473 \
    --gsm8k-val 300 \
    --alpaca-train 10445 \
    --alpaca-val 200 \
    --seed 42

echo ""
echo "Done. Files:"
ls -lh sdpo/data/mixed_*.jsonl
wc -l sdpo/data/mixed_*.jsonl
