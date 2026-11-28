#!/bin/bash
#SBATCH --job-name=prep_mix4k5
#SBATCH --partition=a100_tandon
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ll5914/logs/tree_eal/prep_mix4k5_%j.out
#SBATCH --error=/scratch/ll5914/logs/tree_eal/prep_mix4k5_%j.err
#
# Build ~4.5k train mix: 3600 ShareGPT + 600 GSM8K + 82 HumanEval + 80 MT-Bench.
# MT-Bench turn-1 answers are regenerated with Qwen3-8B.
set -euo pipefail
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-/scratch/ll5914/conda_envs/SVD}"

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
# Prefer checkout root that contains sdpo/
if [ ! -f "$REPO/sdpo/scripts/tree_eal/build_mix_sg_gsm_he_mt.py" ]; then
  REPO=/home/ll5914/Cerebras_SD/SDPO_dense_tv
fi
cd "$REPO"

export HF_HOME=${HF_HOME:-/scratch/ll5914/huggingface}
export TOKENIZERS_PARALLELISM=false

OUT_DIR=${OUT_DIR:-/scratch/ll5914/loss_train_q8_q06/data}
BASE=${BASE:-/scratch/ll5914/models/qwen3-8b}
SG=${SG:-/scratch/yf3005/gto_data/sharegpt_qwen3_8b_regen.jsonl}
GSM=${GSM:-/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/gsm8k_qwen3_8b_regen.jsonl}
HE=${HE:-/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/humaneval_qwen3_8b_regen.jsonl}
MTQ=${MTQ:-$REPO/data/mt_bench/question.jsonl}

echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} REPO=$REPO start=$(date -Is)"
nvidia-smi -L

python3 sdpo/scripts/tree_eal/build_mix_sg_gsm_he_mt.py \
  --out-dir "$OUT_DIR" \
  --sharegpt "$SG" \
  --gsm8k "$GSM" \
  --humaneval "$HE" \
  --mt-bench-questions "$MTQ" \
  --base-model "$BASE" \
  --n-sharegpt "${N_SG:-3600}" \
  --n-gsm8k "${N_GSM:-600}" \
  --seed "${SEED:-42}"

echo "done=$(date -Is)"
ls -lh "$OUT_DIR"/mix_sg_gsm_he_mt_4k5.jsonl "$OUT_DIR"/mt_bench_qwen3_8b_regen.jsonl \
       "$OUT_DIR"/mix_sg_gsm_he_mt_4k5.meta.json
