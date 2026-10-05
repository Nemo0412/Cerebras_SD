#!/bin/bash

#SBATCH --job-name=kl_vs_klv4
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100|h200"
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --mail-type=END
#SBATCH --mail-user=tx856@nyu.edu
#SBATCH --output=logs/smalllm_rollout/kl_vs_klv4_%j.out
#SBATCH --error=logs/smalllm_rollout/kl_vs_klv4_%j.err

cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
mkdir -p logs/smalllm_rollout smalllm_tree_eval_results

source ~/.bashrc
conda activate ${CONDA_ENV:-longreason}

NUM_SAMPLES=${NUM_SAMPLES:-80}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
BENCHES=${BENCHES:-mt_bench,gsm8k,humaneval}
OUTPUT=${OUTPUT:-smalllm_tree_eval_results/kl_vs_klv4_per_sample.json}

python sdpo/analyze_kl_vs_klv4.py \
    --benches "$BENCHES" \
    --num-samples "$NUM_SAMPLES" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --gamma 7 \
    --top-k 4,3,2,1,1,1,1 \
    --budget 128 \
    --output "$OUTPUT"
