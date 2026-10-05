#!/bin/bash

#SBATCH --job-name=bench_regen
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --requeue
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --mail-type=END
#SBATCH --mail-user=tx856@nyu.edu
#SBATCH --output=logs/bench_regen/%x_%j.out
#SBATCH --error=logs/bench_regen/%x_%j.err

cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
mkdir -p logs/bench_regen

source ~/.bashrc
conda activate ${CONDA_ENV:-longreason}

BENCH=${BENCH:-gsm8k}
MODEL=${MODEL:-Qwen/Qwen3-8B}
OUTPUT=${OUTPUT:-sdpo/data/${BENCH}_qwen3_regen.jsonl}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_TOKENS=${MAX_TOKENS:-4096}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.9}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
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
echo "  Bench regen"
echo "  bench=$BENCH  model=$MODEL"
echo "  output=$OUTPUT"
echo "  max_model_len=$MAX_MODEL_LEN  max_tokens=$MAX_TOKENS"
echo "  temperature=$TEMPERATURE  top_p=$TOP_P"
echo "  max_num_seqs=$MAX_NUM_SEQS  tp_size=$TP_SIZE"
echo "  resume_flag=$RESUME_FLAG"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

python sdpo/data/prepare_bench_regen.py \
    --bench "$BENCH" \
    --model "$MODEL" \
    --output "$OUTPUT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --tensor-parallel-size "$TP_SIZE" \
    $RESUME_FLAG $SAMPLE_FLAG
