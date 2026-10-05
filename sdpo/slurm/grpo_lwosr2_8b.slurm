#!/bin/bash
#SBATCH --job-name=grpo_lwosr2_8b
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100|h200"
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --mail-type=END
#SBATCH --mail-user=tx856@nyu.edu
#SBATCH --output=logs/grpo_lwosr2/8b_%j.out
#SBATCH --error=logs/grpo_lwosr2/8b_%j.err

# KLV4-GRPO lwosr2 gap-fill, Qwen3-8B target (q8_q06 5 train-seeds + q17 fix).
set -u
cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
mkdir -p logs/grpo_lwosr2 smalllm_tree_eval_results
source ~/.bashrc
conda activate ${CONDA_ENV:-longreason}

# keep caches off /home (the inode quota that crashed g4_regen_all)
export HF_HOME=/scratch/tx856/.huggingface
export XDG_CACHE_HOME=/scratch/tx856/.cache
export VLLM_CACHE_ROOT=/scratch/tx856/.cache/vllm
export TRITON_CACHE_DIR=/scratch/tx856/.cache/triton
export TORCH_EXTENSIONS_DIR=/scratch/tx856/.cache/torch_extensions
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

BASE=Qwen/Qwen3-8B
R6=/scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b
OUT=smalllm_tree_eval_results

declare -a CKPTS=(
  "klv4_grpo_smp_s0       $R6/q8_q06_klv4_grpo_smp_s0/state_2"
  "klv4_grpo_smp_s1       $R6/q8_q06_klv4_grpo_smp_s1/state_2"
  "klv4_grpo_smp_s2       $R6/q8_q06_klv4_grpo_smp_s2/state_2"
  "klv4_grpo_smp_s3       $R6/q8_q06_klv4_grpo_smp_s3/state_2"
  "klv4_grpo_smp_s4       $R6/q8_q06_klv4_grpo_smp_s4/state_2"
  "q17_klv4_grpo_smp_fix  $R6/q17_q06_klv4_grpo_smp_fix/state_2"
)

declare -a MODES=(
  "lwosr2_greedy             --temperature 0.0 --draft-mode argmax --verify-mode auto"
  "lwosr2_samp_t05_smp_ratio --temperature 0.5 --draft-mode sample --verify-mode ratio"
  "lwosr2_samp_t1_smp_ratio  --temperature 1.0 --draft-mode sample --verify-mode ratio"
)

COMMON="--base-model-path $BASE --bench-name longwriter_top50,osr2_top50 --gamma 7 --top-k 4,3,2,1,1,1,1 --budget 128 --max-new-tokens 256 --num-samples 40 --output-dir $OUT --no-thinking --seed 0"

echo "=== grpo_lwosr2_8b start $(date) job=$SLURM_JOB_ID ==="
for line in "${CKPTS[@]}"; do
  read -r TAG CKPT <<< "$line"
  if [ ! -f "$CKPT/pytorch_model.bin" ]; then echo "[SKIP] $TAG: no $CKPT/pytorch_model.bin"; continue; fi
  for ml in "${MODES[@]}"; do
    read -r SUF ARGS <<< "$ml"
    FT="${TAG}_${SUF}"
    if [ -f "$OUT/$FT.json" ]; then echo "[SKIP] $FT (done)"; continue; fi
    echo ""; echo "=== $FT  ($(date '+%H:%M:%S')) ==="
    python sdpo/eval_small_lm_tree.py $COMMON --draft-model-path "$CKPT" --tag "$FT" $ARGS
    echo "=== done $FT rc=$? ($(date '+%H:%M:%S')) ==="
  done
done
echo "=== ALL DONE $(date) ==="
