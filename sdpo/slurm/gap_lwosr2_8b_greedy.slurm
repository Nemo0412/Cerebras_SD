#!/bin/bash
#SBATCH --job-name=gap_lwosr2_8b
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100|h200"
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --mail-type=END
#SBATCH --mail-user=tx856@nyu.edu
#SBATCH --output=logs/grpo_lwosr2/gap8b_%j.out
#SBATCH --error=logs/grpo_lwosr2/gap8b_%j.err

# lwosr2 greedy gap-fill, Qwen3-8B: q8_q06 KLALTV (5 seeds) + q17 KLTV/ALTV/KLALTV.
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

BASE=Qwen/Qwen3-8B
R6=/scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b
OUT=smalllm_tree_eval_results

declare -a CKPTS=(
  "klaltv_T10_c05_s0            $R6/q8_q06_klaltv_T10_c05_l2k_6ep_s0/state_5"
  "klaltv_T10_c05_s1            $R6/q8_q06_klaltv_T10_c05_l2k_6ep_s1/state_5"
  "klaltv_T10_c05_s2            $R6/q8_q06_klaltv_T10_c05_l2k_6ep_s2/state_5"
  "klaltv_T10_c05_s3            $R6/q8_q06_klaltv_T10_c05_l2k_6ep_s3/state_5"
  "klaltv_T10_c05_s4            $R6/q8_q06_klaltv_T10_c05_l2k_6ep_s4/state_5"
  "q17_kltv_l2k_6ep_s0          $R6/q17_q06_kltv_l2k_6ep_s0/state_5"
  "q17_altv_only_l2k_6ep        $R6/q17_q06_altv_only_l2k_6ep/state_5"
  "q17_klaltv_T10_c05_l2k_6ep   $R6/q17_q06_klaltv_T10_c05_l2k_6ep/state_5"
)
declare -a MODES=(
  "lwosr2_greedy --temperature 0.0 --draft-mode argmax --verify-mode auto"
)
COMMON="--base-model-path $BASE --bench-name longwriter_top50,osr2_top50 --gamma 7 --top-k 4,3,2,1,1,1,1 --budget 128 --max-new-tokens 256 --num-samples 40 --output-dir $OUT --no-thinking --seed 0"

echo "=== gap_lwosr2_8b start $(date) job=$SLURM_JOB_ID ==="
for line in "${CKPTS[@]}"; do
  read -r TAG CKPT <<< "$line"
  if [ ! -f "$CKPT/pytorch_model.bin" ]; then echo "[SKIP] $TAG: no ckpt"; continue; fi
  for ml in "${MODES[@]}"; do
    read -r SUF ARGS <<< "$ml"
    FT="${TAG}_${SUF}"
    if [ -f "$OUT/$FT.json" ]; then echo "[SKIP] $FT (done)"; continue; fi
    echo ""; echo "=== $FT ($(date '+%H:%M:%S')) ==="
    python sdpo/eval_small_lm_tree.py $COMMON --draft-model-path "$CKPT" --tag "$FT" $ARGS
    echo "=== done $FT rc=$? ==="
  done
done
echo "=== ALL DONE $(date) ==="
