#!/bin/bash

# SmallLM 0.6B bundled eval: greedy + sample_ratio (T=0.5) per ckpt in ONE job.
#   greedy:        --draft-mode argmax --verify-mode auto --temperature 0.0
#   sample_ratio:  --draft-mode sample --verify-mode ratio --temperature 0.5

#SBATCH --job-name=sl06_bndl_eval
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100|h200"
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --mail-type=END
#SBATCH --mail-user=tx856@nyu.edu
#SBATCH --output=logs/sdpo_smalllm_06b/eval_bundled_%j.out
#SBATCH --error=logs/sdpo_smalllm_06b/eval_bundled_%j.err

cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
mkdir -p logs/sdpo_smalllm_06b

source ~/.bashrc
conda activate ${CONDA_ENV:-longreason}

BASEPATH=${BASEPATH:-Qwen/Qwen3-8B}
DRAFT_CKPT=${DRAFT_CKPT:?must set DRAFT_CKPT}
TAG=${TAG:?must set TAG}
BENCH=${BENCH:-mt_bench,gsm8k,humaneval,longwriter_top50,osr2_top50}
GAMMA=${GAMMA:-7}
TOP_K=${TOP_K:-4,3,2,1,1,1,1}
BUDGET=${BUDGET:-128}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
NUM_SAMPLES=${NUM_SAMPLES:-40}
EVAL_SEED=${EVAL_SEED:-0}
OUTPUT_DIR=${OUTPUT_DIR:-smalllm_tree_eval_results}

echo "=============================="
echo "  SmallLM 0.6B bundled eval: $TAG"
echo "  Target:  $BASEPATH"
echo "  Draft:   $DRAFT_CKPT"
echo "  Bench:   $BENCH"
echo "  Tree:    gamma=$GAMMA top_k=$TOP_K budget=$BUDGET"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

# ── Greedy ──
echo ""
echo "==== GREEDY (T=0, argmax/auto) → TAG=${TAG}_greedy ===="
python sdpo/eval_small_lm_tree.py \
    --base-model-path "$BASEPATH" \
    --draft-model-path "$DRAFT_CKPT" \
    --bench-name "$BENCH" \
    --gamma "$GAMMA" --top-k "$TOP_K" --budget "$BUDGET" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature 0.0 \
    --seed "$EVAL_SEED" \
    --draft-mode argmax --verify-mode auto \
    --tag "${TAG}_greedy" \
    --output-dir "$OUTPUT_DIR" \
    --num-samples "$NUM_SAMPLES"

# ── Sample_ratio T=0.5 ──
echo ""
echo "==== SAMPLE_RATIO (T=0.5, sample/ratio) → TAG=${TAG}_smp_ratio ===="
python sdpo/eval_small_lm_tree.py \
    --base-model-path "$BASEPATH" \
    --draft-model-path "$DRAFT_CKPT" \
    --bench-name "$BENCH" \
    --gamma "$GAMMA" --top-k "$TOP_K" --budget "$BUDGET" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature 0.5 \
    --seed "$EVAL_SEED" \
    --draft-mode sample --verify-mode ratio \
    --tag "${TAG}_smp_ratio" \
    --output-dir "$OUTPUT_DIR" \
    --num-samples "$NUM_SAMPLES"

echo ""
echo "==== done ===="
