#!/bin/bash
# Gemma 4 eval — T=1.0 sample_ratio ONLY. Backfill for evals that already
# ran greedy + T=0.5 sample_ratio.
#   sample_ratio_t1 : --temperature 1.0 --draft-mode sample --verify-mode ratio

#SBATCH --job-name=g4_eval_t1
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --constraint="h200"
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --mail-type=END
#SBATCH --mail-user=tx856@nyu.edu
#SBATCH --output=logs/sdpo_gemma4/eval_t1_%j.out
#SBATCH --error=logs/sdpo_gemma4/eval_t1_%j.err

cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
mkdir -p logs/sdpo_gemma4

source ~/.bashrc
conda activate ${CONDA_ENV:-gemma4}

BASEPATH=${BASEPATH:-google/gemma-4-31B-it}
DRAFT_CKPT=${DRAFT_CKPT:?must set DRAFT_CKPT to a state_* dir or model id}
TAG=${TAG:?must set TAG}
BENCH=${BENCH:-mt_bench,gsm8k,humaneval,longwriter_top50,osr2_top50}
GAMMA=${GAMMA:-7}
TOP_K=${TOP_K:-4,3,2,1,1,1,1}
BUDGET=${BUDGET:-128}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
NUM_SAMPLES=${NUM_SAMPLES:-}
OUTPUT_DIR=${OUTPUT_DIR:-smalllm_tree_eval_results}

NUM_SAMPLES_FLAG=""
[ -n "$NUM_SAMPLES" ] && NUM_SAMPLES_FLAG="--num-samples $NUM_SAMPLES"

echo "=============================="
echo "  Gemma 4 T=1 eval only: $TAG"
echo "  Target: $BASEPATH"
echo "  Draft:  $DRAFT_CKPT"
echo "  Bench:  $BENCH"
echo "  Tree:   gamma=$GAMMA top_k=$TOP_K budget=$BUDGET"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

echo ""
echo "==== SAMPLE_RATIO (T=1.0, sample/ratio) ===="
python sdpo/eval_gemma4_chain.py \
    --target-model "$BASEPATH" \
    --draft-model "$DRAFT_CKPT" \
    --bench-name "$BENCH" \
    --mode tree \
    --gamma "$GAMMA" --top-k "$TOP_K" --budget "$BUDGET" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature 1.0 \
    --draft-mode sample \
    --verify-mode ratio \
    --tag "${TAG}_smp_t1_ratio" \
    --output-dir "$OUTPUT_DIR" \
    $NUM_SAMPLES_FLAG

echo ""
echo "==== done ===="
