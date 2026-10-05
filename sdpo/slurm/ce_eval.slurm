#!/bin/bash
# CE-only ckpt eval: greedy + sample_ratio T=1 across 5 benches.
# No-thinking chat template, γ=7 baseline tree config.
#
# Required env vars:
#   TAG          — output tag prefix
#   DRAFT_CKPT   — absolute path to draft ckpt dir (with pytorch_model.bin)
#   BASEPATH     — target HF id (Qwen3-8B / Qwen3-32B)
# Optional:
#   CONDA_ENV    — default longreason

#SBATCH --job-name=ce_eval
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100|h200"
#SBATCH --requeue
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tx856@nyu.edu
#SBATCH --output=logs/ce_eval/%x_%j.out
#SBATCH --error=logs/ce_eval/%x_%j.err

REPO=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
cd "$REPO"
mkdir -p logs/ce_eval smalllm_tree_eval_results

source ~/.bashrc
conda activate ${CONDA_ENV:-longreason}

export HF_HOME=/scratch/tx856/.huggingface
export TORCH_EXTENSIONS_DIR=/scratch/tx856/.cache/torch_extensions
export TRITON_CACHE_DIR=/scratch/tx856/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/scratch/tx856/.cache/torchinductor

: ${TAG:?must set TAG}
: ${DRAFT_CKPT:?must set DRAFT_CKPT}
: ${BASEPATH:?must set BASEPATH}

OUT=smalllm_tree_eval_results
BENCHES="mt_bench,gsm8k,humaneval,longwriter_top50,osr2_top50"

COMMON_ARGS="
    --base-model-path $BASEPATH
    --draft-model-path $DRAFT_CKPT
    --bench-name $BENCHES
    --gamma 7
    --top-k 4,3,2,1,1,1,1
    --budget 128
    --max-new-tokens 256
    --num-samples 40
    --output-dir $OUT
    --no-thinking
    --seed 0
"

echo "=============================="
echo "  CE-only eval: $TAG"
echo "  Target:  $BASEPATH"
echo "  CKPT:    $DRAFT_CKPT"
echo "  Benches: $BENCHES"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

# ── Greedy ──
echo ""
echo "==== GREEDY ===="
set +e
python sdpo/eval_small_lm_tree.py \
    $COMMON_ARGS \
    --temperature 0.0 --draft-mode argmax --verify-mode auto \
    --tag "${TAG}_nothink_greedy"
RC1=$?
set -e
echo "  rc=$RC1"

# ── Sample_ratio T=1 ──
echo ""
echo "==== SAMPLE_RATIO T=1 ===="
set +e
python sdpo/eval_small_lm_tree.py \
    $COMMON_ARGS \
    --temperature 1.0 --draft-mode sample --verify-mode ratio \
    --tag "${TAG}_nothink_samp_t1_ratio"
RC2=$?
set -e
echo "  rc=$RC2"

echo ""
echo "==== DONE  rc=$RC1/$RC2 ===="
