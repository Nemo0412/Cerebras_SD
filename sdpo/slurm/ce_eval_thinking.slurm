#!/bin/bash
# CE-only ckpt eval with thinking mode ON (matches legacy KL/ALTV eval setup).
# 5 benches × 2 modes (greedy + samp_t1_ratio), γ=7 b=128 top_k=[4,3,2,1,1,1,1].
#
# Required env vars:
#   TAG          — output tag prefix
#   DRAFT_CKPT   — absolute path to draft ckpt dir
#   BASEPATH     — target HF id

#SBATCH --job-name=ce_eval_think
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

# Note: no --no-thinking flag → default enable_thinking=True (Qwen3 thinking mode).
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
    --seed 0
"

echo "=============================="
echo "  CE-only THINKING ON eval: $TAG"
echo "  Target:  $BASEPATH"
echo "  CKPT:    $DRAFT_CKPT"
echo "  Benches: $BENCHES"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

# ── Greedy ──
echo ""
echo "==== GREEDY (thinking ON) ===="
set +e
python sdpo/eval_small_lm_tree.py \
    $COMMON_ARGS \
    --temperature 0.0 --draft-mode argmax --verify-mode auto \
    --tag "${TAG}_think_greedy"
RC1=$?
set -e
echo "  rc=$RC1"

# ── Sample_ratio T=1 ──
echo ""
echo "==== SAMPLE_RATIO T=1 (thinking ON) ===="
set +e
python sdpo/eval_small_lm_tree.py \
    $COMMON_ARGS \
    --temperature 1.0 --draft-mode sample --verify-mode ratio \
    --tag "${TAG}_think_samp_t1_ratio"
RC2=$?
set -e
echo "  rc=$RC2"

echo ""
echo "==== DONE  rc=$RC1/$RC2 ===="
