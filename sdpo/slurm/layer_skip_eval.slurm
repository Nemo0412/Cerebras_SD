#!/bin/bash

#SBATCH --job-name=layerskip_eval
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100|h200"
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --mail-type=END
#SBATCH --mail-user=tx856@nyu.edu
#SBATCH --output=logs/layerskip/eval_%j.out
#SBATCH --error=logs/layerskip/eval_%j.err

cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
mkdir -p logs/layerskip

source ~/.bashrc
conda activate ${CONDA_ENV:-longreason}

BASEPATH=${BASEPATH:-Qwen/Qwen3-32B}
DRAFT_CKPT=${DRAFT_CKPT:-Qwen/Qwen3-0.6B}
EXIT_LAYERS=${EXIT_LAYERS:-4,8,12,16,20,24,28}
OUTPUT_DIR=${OUTPUT_DIR:-layerskip_eval_results}
BENCH=${BENCH:-all}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}

echo "=============================="
echo "  LayerSkip Eval: $TAG"
echo "  Target: $BASEPATH"
echo "  Draft:  $DRAFT_CKPT"
echo "  Exits:  $EXIT_LAYERS"
echo "  Bench:  $BENCH"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

python sdpo/eval_layer_skip.py \
    --base-model-path "$BASEPATH" \
    --draft-model-path "$DRAFT_CKPT" \
    --exit-layers "$EXIT_LAYERS" \
    --tag "$TAG" \
    --bench-name "$BENCH" \
    --gamma 7 \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --output-dir "$OUTPUT_DIR"
