#!/bin/bash
#SBATCH --job-name=q4b_klv4grpo_tree_samp
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100|h200"
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tx856@nyu.edu
#SBATCH --output=logs/tree_klv4grpo/%x_%j.out
#SBATCH --error=logs/tree_klv4grpo/%x_%j.err

# Fill missing Q4B tree γ=7 KLV4_GRPO samp_t05/samp_t1 (think mode).
# 3-bench + LW/OSR2 companion, matching KL/ALTV Q4B tree-think naming.

REPO=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
cd "$REPO"
mkdir -p logs/tree_klv4grpo smalllm_tree_eval_results

source ~/.bashrc
conda activate ${CONDA_ENV:-longreason}
set -u

export HF_HOME=/scratch/tx856/.huggingface
export XDG_CACHE_HOME=/scratch/tx856/.cache
export VLLM_CACHE_ROOT=/scratch/tx856/.cache/vllm
export TRITON_CACHE_DIR=/scratch/tx856/.cache/triton
export TORCH_EXTENSIONS_DIR=/scratch/tx856/.cache/torch_extensions

BASE=Qwen/Qwen3-32B
R4=/scratch/tx856/spec_reason/scratch/loss_train_smalllm_q32_q4
OUT=smalllm_tree_eval_results
CKPT="$R4/q32_q4_klv4_grpo_smp_fix/state_2"
TAG=q32_q4_klv4_grpo_smp_fix

COMMON="--base-model-path $BASE --draft-model-path $CKPT --gamma 7 --top-k 4,3,2,1,1,1,1 --budget 128 --max-new-tokens 256 --num-samples 40 --output-dir $OUT --seed 0"

if [ ! -f "$CKPT/pytorch_model.bin" ]; then echo "[FATAL] no $CKPT/pytorch_model.bin"; exit 1; fi
echo "=== job=$SLURM_JOB_ID start $(date) ==="

# 4 runs: 2 modes × 2 bench splits
declare -a RUNS=(
    "samp_t05_smp_ratio          mt_bench,gsm8k,humaneval           --temperature 0.5 --draft-mode sample --verify-mode ratio"
    "samp_t1_smp_ratio           mt_bench,gsm8k,humaneval           --temperature 1.0 --draft-mode sample --verify-mode ratio"
    "lwosr2_samp_t05_smp_ratio   longwriter_top50,osr2_top50        --temperature 0.5 --draft-mode sample --verify-mode ratio"
    "lwosr2_samp_t1_smp_ratio    longwriter_top50,osr2_top50        --temperature 1.0 --draft-mode sample --verify-mode ratio"
)

for RUN in "${RUNS[@]}"; do
    read -r SUF BENCHES ARGS <<< "$RUN"
    FT="${TAG}_${SUF}"
    OUT_FILE="$OUT/$FT.json"
    if [ -f "$OUT_FILE" ]; then echo "[SKIP] $FT (done)"; continue; fi
    echo ""; echo "=== $FT ($(date '+%H:%M:%S')) ==="
    python sdpo/eval_small_lm_tree.py $COMMON --bench-name "$BENCHES" --tag "$FT" $ARGS
    echo "=== done $FT rc=$? ($(date '+%H:%M:%S')) ==="
done

echo "=== ALL DONE $(date) ==="
