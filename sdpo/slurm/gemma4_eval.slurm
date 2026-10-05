#!/bin/bash

# Gemma 4 bundled eval: greedy + sample_ratio T=0.5 + sample_ratio T=1.0 in ONE slurm job per ckpt.
#   greedy           : --temperature 0.0 --draft-mode argmax  --verify-mode auto
#   sample_ratio_t05 : --temperature 0.5 --draft-mode sample  --verify-mode ratio
#   sample_ratio_t1  : --temperature 1.0 --draft-mode sample  --verify-mode ratio

#SBATCH --job-name=g4_eval
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --constraint="h200"
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --mail-type=END
#SBATCH --mail-user=tx856@nyu.edu
#SBATCH --output=logs/sdpo_gemma4/eval_%j.out
#SBATCH --error=logs/sdpo_gemma4/eval_%j.err

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
echo "  Gemma 4 bundled eval (greedy + samp_ratio T=0.5 + samp_ratio T=1): $TAG"
echo "  Target: $BASEPATH"
echo "  Draft:  $DRAFT_CKPT"
echo "  Bench:  $BENCH"
echo "  Tree:   gamma=$GAMMA top_k=$TOP_K budget=$BUDGET"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

# ── Greedy ──
echo ""
echo "==== GREEDY (T=0, argmax/auto) ===="
python sdpo/eval_gemma4_chain.py \
    --target-model "$BASEPATH" \
    --draft-model "$DRAFT_CKPT" \
    --bench-name "$BENCH" \
    --mode tree \
    --gamma "$GAMMA" --top-k "$TOP_K" --budget "$BUDGET" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature 0.0 \
    --draft-mode argmax \
    --verify-mode auto \
    --tag "${TAG}_greedy" \
    --output-dir "$OUTPUT_DIR" \
    $NUM_SAMPLES_FLAG

# ── Sample-ratio T=0.5 ──
echo ""
echo "==== SAMPLE_RATIO (T=0.5, sample/ratio) ===="
python sdpo/eval_gemma4_chain.py \
    --target-model "$BASEPATH" \
    --draft-model "$DRAFT_CKPT" \
    --bench-name "$BENCH" \
    --mode tree \
    --gamma "$GAMMA" --top-k "$TOP_K" --budget "$BUDGET" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature 0.5 \
    --draft-mode sample \
    --verify-mode ratio \
    --tag "${TAG}_smp_ratio" \
    --output-dir "$OUTPUT_DIR" \
    $NUM_SAMPLES_FLAG

# ── Sample-ratio T=1.0 ──
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
