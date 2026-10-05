#!/bin/bash

# Unified Gemma 4 31B regen: humaneval → gsm8k → osr2 → longwriter → sharegpt.
# Runs sequentially in ONE slurm job (consolidated from 5 separate jobs
# 11629686-11629690).
#
# Why merged: original 5 jobs competed on QOSGrpGRES (GPU quota) and waited
# in queue separately. One job = one quota slot, runs straight through.
# Each step does its own vLLM load (~2-5 min); total overhead ~25 min,
# acceptable vs queue wait.

#SBATCH --job-name=g4_regen_all
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --constraint="h200"
#SBATCH --requeue
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tx856@nyu.edu
#SBATCH --output=logs/regen/g4_regen_all_%j.out
#SBATCH --error=logs/regen/g4_regen_all_%j.err

REPO=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
cd "$REPO"
mkdir -p logs/regen sdpo/data

source ~/.bashrc
conda activate gemma4

# Redirect vLLM / torch / triton compile caches to /scratch (TB-scale). The
# default ~/.cache locations exhausted home quota in the prior run (11703315
# crashed on Errno 122 "Disk quota exceeded" during osr2's vllm load), wasting
# 10h GPU time with zero output.
export VLLM_CACHE_ROOT=/scratch/tx856/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/scratch/tx856/.cache/torchinductor
export TRITON_CACHE_DIR=/scratch/tx856/.cache/triton
export TORCH_EXTENSIONS_DIR=/scratch/tx856/.cache/torch_extensions
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

MODEL=google/gemma-4-31B-it
DATA_DIR="$REPO/sdpo/data"

echo "=============================="
echo "  Gemma 4 unified regen sweep"
echo "  Model:   $MODEL"
echo "  Output:  $DATA_DIR/{gsm8k,humaneval,longwriter,osr2,sharegpt}_gemma4_31b_regen.jsonl"
echo "  Order:   humaneval → gsm8k → osr2 → longwriter → sharegpt (short→long)"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

run_step() {
    local label=$1; shift
    echo ""
    echo ">>>>> [$label] $(date '+%H:%M:%S') <<<<<"
    "$@"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[$label] FAILED with rc=$rc — continuing to next step."
    else
        echo "[$label] DONE."
    fi
}

# ── 1. HumanEval (~1h) ──
run_step humaneval \
    python sdpo/data/prepare_bench_regen.py \
        --bench humaneval \
        --model "$MODEL" \
        --output "$DATA_DIR/humaneval_gemma4_31b_regen.jsonl" \
        --max-model-len 8192 \
        --max-tokens 4096 \
        --temperature 0.7 \
        --top-p 0.9 \
        --max-num-seqs 32 \
        --tensor-parallel-size 2

# ── 2. GSM8K (~8h) ──
run_step gsm8k \
    python sdpo/data/prepare_bench_regen.py \
        --bench gsm8k \
        --model "$MODEL" \
        --output "$DATA_DIR/gsm8k_gemma4_31b_regen.jsonl" \
        --max-model-len 8192 \
        --max-tokens 4096 \
        --temperature 0.7 \
        --top-p 0.9 \
        --max-num-seqs 32 \
        --tensor-parallel-size 2

# ── 3. OSR2 (~12h) ──
run_step osr2 \
    python sdpo/data/prepare_osr2_regen.py \
        --model "$MODEL" \
        --output "$DATA_DIR/osr2_gemma4_31b_regen.jsonl" \
        --num-samples 8000 \
        --max-model-len 32768 \
        --max-tokens 16000 \
        --temperature 0.0 \
        --max-num-seqs 16 \
        --tensor-parallel-size 2

# ── 4. LongWriter (~12h) ──
run_step longwriter \
    python sdpo/data/prepare_longwriter_regen.py \
        --model "$MODEL" \
        --output "$DATA_DIR/longwriter_gemma4_31b_regen.jsonl" \
        --max-model-len 32768 \
        --max-tokens 16000 \
        --temperature 0.0 \
        --max-num-seqs 16 \
        --tensor-parallel-size 2

# ── 5. ShareGPT 60K (~1d12h) ──
DATA_SHAREGPT=/scratch/yf3005/.huggingface/hub/datasets--Aeala--ShareGPT_Vicuna_unfiltered/snapshots/8b0048ad6ae8c22f46a78c15559dec98feef5539/ShareGPT_V4.3_unfiltered_cleaned_split.json
run_step sharegpt \
    python sdpo/data/regen_sharegpt.py \
        --model "$MODEL" \
        --data "$DATA_SHAREGPT" \
        --output "$DATA_DIR/sharegpt_gemma4_31b_regen.jsonl" \
        --tensor-parallel-size 2 \
        --max-model-len 8192 \
        --max-tokens 2048 \
        --batch-size 200 \
        --max-samples 60000 \
        --temperature 0.7 \
        --top-p 0.9 \
        --dtype bfloat16

echo ""
echo "=============================="
echo "  ALL STEPS COMPLETE   $(date)"
echo "=============================="
for f in humaneval gsm8k osr2 longwriter sharegpt; do
    p="$DATA_DIR/${f}_gemma4_31b_regen.jsonl"
    if [ -f "$p" ]; then
        echo "  $f: $(wc -l < "$p") lines  ($p)"
    else
        echo "  $f: MISSING"
    fi
done
