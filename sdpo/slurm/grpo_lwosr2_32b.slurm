#!/bin/bash
#SBATCH --job-name=grpo_lwosr2_32b
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="h200"
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --mail-type=END
#SBATCH --mail-user=tx856@nyu.edu
#SBATCH --output=logs/grpo_lwosr2/32b_%j.out
#SBATCH --error=logs/grpo_lwosr2/32b_%j.err

# KLV4-GRPO lwosr2 gap-fill, Qwen3-32B target (q32_q4 fix, state_2).
set -u
cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
mkdir -p logs/grpo_lwosr2 smalllm_tree_eval_results
source ~/.bashrc
conda activate ${CONDA_ENV:-longreason}

export HF_HOME=/scratch/tx856/.huggingface
export XDG_CACHE_HOME=/scratch/tx856/.cache
export VLLM_CACHE_ROOT=/scratch/tx856/.cache/vllm
export TRITON_CACHE_DIR=/scratch/tx856/.cache/triton
export TORCH_EXTENSIONS_DIR=/scratch/tx856/.cache/torch_extensions
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

BASE=Qwen/Qwen3-32B
R4=/scratch/tx856/spec_reason/scratch/loss_train_smalllm_q32_q4
OUT=smalllm_tree_eval_results

CKPT="$R4/q32_q4_klv4_grpo_smp_fix/state_2"
TAG=q32_q4_klv4_grpo_smp_fix

declare -a MODES=(
  "lwosr2_greedy             --temperature 0.0 --draft-mode argmax --verify-mode auto"
  "lwosr2_samp_t05_smp_ratio --temperature 0.5 --draft-mode sample --verify-mode ratio"
  "lwosr2_samp_t1_smp_ratio  --temperature 1.0 --draft-mode sample --verify-mode ratio"
)

COMMON="--base-model-path $BASE --bench-name longwriter_top50,osr2_top50 --gamma 7 --top-k 4,3,2,1,1,1,1 --budget 128 --max-new-tokens 256 --num-samples 40 --output-dir $OUT --no-thinking --seed 0"

echo "=== grpo_lwosr2_32b start $(date) job=$SLURM_JOB_ID ==="
if [ ! -f "$CKPT/pytorch_model.bin" ]; then echo "[FATAL] no $CKPT/pytorch_model.bin"; exit 1; fi
for ml in "${MODES[@]}"; do
  read -r SUF ARGS <<< "$ml"
  FT="${TAG}_${SUF}"
  if [ -f "$OUT/$FT.json" ]; then echo "[SKIP] $FT (done)"; continue; fi
  echo ""; echo "=== $FT ($(date '+%H:%M:%S')) ==="
  python sdpo/eval_small_lm_tree.py $COMMON --draft-model-path "$CKPT" --tag "$FT" $ARGS
  echo "=== done $FT rc=$? ($(date '+%H:%M:%S')) ==="
done
echo "=== ALL DONE $(date) ==="
