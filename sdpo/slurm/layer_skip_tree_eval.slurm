#!/bin/bash

#SBATCH --job-name=tree_eval
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
#SBATCH --output=logs/layerskip/tree_eval_%j.out
#SBATCH --error=logs/layerskip/tree_eval_%j.err

cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
mkdir -p logs/layerskip

source ~/.bashrc
conda activate ${CONDA_ENV:-longreason}

BASEPATH=${BASEPATH:-Qwen/Qwen3-32B}
DRAFT_CKPT=${DRAFT_CKPT:-Qwen/Qwen3-0.6B}
EXIT_LAYERS=${EXIT_LAYERS:-24,28}
MODE=${MODE:-both}
BENCH=${BENCH:-mt_bench}
GAMMA=${GAMMA:-7}
TOP_K=${TOP_K:-10}
TOP_K_PER_EXIT=${TOP_K_PER_EXIT:-5}
BUDGET=${BUDGET:-63}
BUDGET_PER_EXIT=${BUDGET_PER_EXIT:-15}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
OUTPUT_DIR=${OUTPUT_DIR:-layerskip_tree_eval_results}
NUM_SAMPLES=${NUM_SAMPLES:-}

BASELINE_FLAG=""
if [ "${RUN_BASELINE:-0}" = "1" ]; then
    BASELINE_FLAG="--run-baseline"
fi

NUM_SAMPLES_FLAG=""
if [ -n "$NUM_SAMPLES" ]; then
    NUM_SAMPLES_FLAG="--num-samples $NUM_SAMPLES"
fi

echo "=============================="
echo "  Tree Eval: $TAG"
echo "  Target: $BASEPATH"
echo "  Draft:  $DRAFT_CKPT"
echo "  Exits:  $EXIT_LAYERS"
echo "  Mode:   $MODE"
echo "  Bench:  $BENCH"
echo "  Tree:   gamma=$GAMMA top_k=$TOP_K budget=$BUDGET"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

python sdpo/eval_layer_skip_tree.py \
    --base-model-path "$BASEPATH" \
    --draft-model-path "$DRAFT_CKPT" \
    --mode "$MODE" \
    --exit-layers "$EXIT_LAYERS" \
    --bench-name "$BENCH" \
    --gamma "$GAMMA" \
    --top-k "$TOP_K" \
    --top-k-per-exit "$TOP_K_PER_EXIT" \
    --budget "$BUDGET" \
    --budget-per-exit "$BUDGET_PER_EXIT" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --tag "$TAG" \
    --output-dir "$OUTPUT_DIR" \
    $BASELINE_FLAG $NUM_SAMPLES_FLAG
