#!/bin/bash

#SBATCH --job-name=smalllm_eval
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100|h200"
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --mail-type=END
#SBATCH --mail-user=tx856@nyu.edu
#SBATCH --output=logs/sdpo_smalllm/eval_%j.out
#SBATCH --error=logs/sdpo_smalllm/eval_%j.err

cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization

source ~/.bashrc
conda activate ${CONDA_ENV:-longreason}

BASEPATH=${BASEPATH:-Qwen/Qwen3-8B}
OUTPUT_DIR=${OUTPUT_DIR:-smalllm_eval_results}
BENCH=${BENCH:-all}

echo "=============================="
echo "  SmallLM Eval: $TAG"
echo "  Target: $BASEPATH"
echo "  Draft checkpoint: $DRAFT_CKPT"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

python sdpo/eval_small_lm.py \
    --base-model-path "$BASEPATH" \
    --draft-model-path "$DRAFT_CKPT" \
    --tag "$TAG" \
    --bench-name "$BENCH" \
    --temperature 0.0 \
    --gamma 7 \
    --output-dir "$OUTPUT_DIR"
