#!/bin/bash

#SBATCH --job-name=osr2_regen
#SBATCH --time=1-23:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100|h200"
#SBATCH --requeue
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --mail-type=END
#SBATCH --mail-user=tx856@nyu.edu
#SBATCH --output=logs/osr2_regen/osr2_regen_%j.out
#SBATCH --error=logs/osr2_regen/osr2_regen_%j.err

cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
mkdir -p logs/osr2_regen

source ~/.bashrc
conda activate ${CONDA_ENV:-longreason}

MODEL=${MODEL:-Qwen/Qwen3-8B}
OUTPUT=${OUTPUT:-sdpo/data/osr2_qwen3_8b_regen.jsonl}
NUM_SAMPLES=${NUM_SAMPLES:-10000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_TOKENS=${MAX_TOKENS:-32000}
TEMPERATURE=${TEMPERATURE:-0.0}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
TP_SIZE=${TP_SIZE:-1}

RESUME_FLAG=""
if [ -f "$OUTPUT" ] && [ -s "$OUTPUT" ]; then
    RESUME_FLAG="--resume"
fi

echo "=============================="
echo "  OSR-2 regen"
echo "  model=$MODEL"
echo "  output=$OUTPUT"
echo "  num_samples=$NUM_SAMPLES"
echo "  max_model_len=$MAX_MODEL_LEN max_tokens=$MAX_TOKENS"
echo "  temperature=$TEMPERATURE max_num_seqs=$MAX_NUM_SEQS tp_size=$TP_SIZE"
echo "  resume_flag=$RESUME_FLAG"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

python sdpo/data/prepare_osr2_regen.py \
    --model "$MODEL" \
    --output "$OUTPUT" \
    --num-samples "$NUM_SAMPLES" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --tensor-parallel-size "$TP_SIZE" \
    $RESUME_FLAG

echo "[DONE]"
wc -l "$OUTPUT"
