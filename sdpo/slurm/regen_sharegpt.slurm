#!/bin/bash

#SBATCH --job-name=regen_sg
#SBATCH --time=1-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --constraint="h200"
#SBATCH --requeue
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --mail-type=END
#SBATCH --mail-user=tx856@nyu.edu
#SBATCH --output=logs/regen/regen_sg_%j.out
#SBATCH --error=logs/regen/regen_sg_%j.err

REPO_ROOT=${REPO_ROOT:-/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization}
cd "$REPO_ROOT"
mkdir -p logs/regen

source ~/.bashrc
conda activate ${CONDA_ENV:-longreason}

MODEL=${MODEL:?must set MODEL}
DATA=${DATA:-/scratch/yf3005/.huggingface/hub/datasets--Aeala--ShareGPT_Vicuna_unfiltered/snapshots/8b0048ad6ae8c22f46a78c15559dec98feef5539/ShareGPT_V4.3_unfiltered_cleaned_split.json}
OUTPUT=${OUTPUT:?must set OUTPUT}

TP=${TP:-2}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_TOKENS=${MAX_TOKENS:-2048}
BATCH_SIZE=${BATCH_SIZE:-200}
MAX_SAMPLES=${MAX_SAMPLES:-0}
DTYPE=${DTYPE:-bfloat16}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.9}

echo "=============================="
echo "  Sharegpt regen"
echo "  Model:         $MODEL"
echo "  Data:          $DATA"
echo "  Output:        $OUTPUT"
echo "  TP:            $TP"
echo "  max_tokens:    $MAX_TOKENS"
echo "  max_model_len: $MAX_MODEL_LEN"
echo "  samples:       ${MAX_SAMPLES} (0 = full)"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

mkdir -p "$(dirname "$OUTPUT")"

python sdpo/data/regen_sharegpt.py \
    --model "$MODEL" \
    --data "$DATA" \
    --output "$OUTPUT" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-tokens "$MAX_TOKENS" \
    --batch-size "$BATCH_SIZE" \
    --max-samples "$MAX_SAMPLES" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --dtype "$DTYPE"
