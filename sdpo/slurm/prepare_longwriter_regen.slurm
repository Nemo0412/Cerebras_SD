#!/bin/bash

#SBATCH --job-name=lw_regen
#SBATCH --time=1-12:00:00
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
#SBATCH --output=logs/longwriter_regen/lw_regen_%j.out
#SBATCH --error=logs/longwriter_regen/lw_regen_%j.err

cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
mkdir -p logs/longwriter_regen

source ~/.bashrc
conda activate ${CONDA_ENV:-longreason}

MODEL=${MODEL:-Qwen/Qwen3-8B}
OUTPUT=${OUTPUT:-sdpo/data/longwriter_qwen3_8b_regen.jsonl}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_TOKENS=${MAX_TOKENS:-32000}
TEMPERATURE=${TEMPERATURE:-0.0}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
TP_SIZE=${TP_SIZE:-1}

RESUME_FLAG=""
if [ -f "$OUTPUT" ] && [ -s "$OUTPUT" ]; then
    RESUME_FLAG="--resume"
fi

SAMPLE_FLAG=""
if [ -n "$NUM_SAMPLES" ]; then
    SAMPLE_FLAG="--num-samples $NUM_SAMPLES"
fi

echo "=============================="
echo "  LongWriter regen"
echo "  model=$MODEL"
echo "  output=$OUTPUT"
echo "  max_model_len=$MAX_MODEL_LEN max_tokens=$MAX_TOKENS"
echo "  temperature=$TEMPERATURE max_num_seqs=$MAX_NUM_SEQS tp_size=$TP_SIZE"
echo "  resume_flag=$RESUME_FLAG"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

python sdpo/data/prepare_longwriter_regen.py \
    --model "$MODEL" \
    --output "$OUTPUT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --tensor-parallel-size "$TP_SIZE" \
    $RESUME_FLAG $SAMPLE_FLAG

echo "[DONE] Final line count:"
wc -l "$OUTPUT"
