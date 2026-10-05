#!/bin/bash
# Q4B CE tree γ=7 samp_t05_ratio: 3-bench + LW/OSR2 companion.
# Ckpt: q32_q4_ce_l2k_6ep_s0/state_2 (existing mixed_10K training).

#SBATCH --job-name=q4b_ce_tree_samp05
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100|h200"
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tx856@nyu.edu
#SBATCH --output=logs/tree_ce_q4b/%x_%j.out
#SBATCH --error=logs/tree_ce_q4b/%x_%j.err

REPO=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
cd "$REPO"
mkdir -p logs/tree_ce_q4b smalllm_tree_eval_results

source ~/.bashrc
conda activate ${CONDA_ENV:-longreason}
set -u

export HF_HOME=${HF_HOME:-/scratch/tx856/.huggingface}
export XDG_CACHE_HOME=/scratch/tx856/.cache
export VLLM_CACHE_ROOT=/scratch/tx856/.cache/vllm
export TRITON_CACHE_DIR=/scratch/tx856/.cache/triton
export TORCH_EXTENSIONS_DIR=/scratch/tx856/.cache/torch_extensions
export PYTORCH_ALLOC_CONF=expandable_segments:True

BASE=Qwen/Qwen3-32B
CKPT=/scratch/tx856/spec_reason/scratch/loss_train_smalllm_q32_q4/q32_q4_ce_l2k_6ep_s0/state_2
OUT=smalllm_tree_eval_results
TAG=q32_q4_ce_l2k_6ep_s0_think

if [ ! -f "$CKPT/pytorch_model.bin" ]; then echo "[FATAL] no $CKPT/pytorch_model.bin"; exit 1; fi
echo "=== $(date) job=$SLURM_JOB_ID ==="

COMMON="--base-model-path $BASE --draft-model-path $CKPT --gamma 7 --top-k 4,3,2,1,1,1,1 --budget 128 --max-new-tokens 256 --num-samples 40 --output-dir $OUT --seed 0"
ARGS="--temperature 0.5 --draft-mode sample --verify-mode ratio"

# 3-bench
FT="${TAG}_samp_t05_ratio"
if [ ! -f "$OUT/$FT.json" ]; then
    echo ""; echo "=== $FT ($(date '+%H:%M:%S')) ==="
    python sdpo/eval_small_lm_tree.py $COMMON --bench-name "mt_bench,gsm8k,humaneval" --tag "$FT" $ARGS
    echo "  rc=$? end=$(date '+%H:%M:%S')"
else
    echo "[SKIP] $FT (done)"
fi

# LW/OSR2 companion (reduced samples if OOM risk)
FT_LW="${TAG}_lwosr2_samp_t05_ratio"
if [ ! -f "$OUT/$FT_LW.json" ]; then
    echo ""; echo "=== $FT_LW ($(date '+%H:%M:%S')) ==="
    python sdpo/eval_small_lm_tree.py $COMMON --bench-name "longwriter_top50,osr2_top50" --tag "$FT_LW" $ARGS
    echo "  rc=$? end=$(date '+%H:%M:%S')"
else
    echo "[SKIP] $FT_LW (done)"
fi

echo "=== ALL DONE $(date) ==="
