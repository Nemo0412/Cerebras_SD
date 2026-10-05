#!/bin/bash

#SBATCH --job-name=qwen3_eval
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100|h200"
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --mail-type=END
#SBATCH --mail-user=tx856@nyu.edu
#SBATCH --output=logs/sdpo_qwen3/eval_%j.out
#SBATCH --error=logs/sdpo_qwen3/eval_%j.err

cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization

source ~/.bashrc
conda activate ${CONDA_ENV:-longreason_eagle3}

BASEPATH=${BASEPATH:-Qwen/Qwen3-8B}
OUTPUT_DIR=${OUTPUT_DIR:-qwen3_eval_results}
BENCH=${BENCH:-all}
TOP_K=${TOP_K:-10}
DEPTH=${DEPTH:-5}
TOTAL_TOKEN=${TOTAL_TOKEN:-60}
# TEMP=0.0 → greedy verify (argmax match); TEMP>0 → EAGLE3 sample-based verify
TEMP=${TEMP:-0.0}
SEED=${SEED:-0}
# SEEDS: comma-separated list of seeds to run in sequence within this job.
# If unset, falls back to single SEED for backward compat.
SEEDS=${SEEDS:-$SEED}

echo "=============================="
echo "  Qwen3 Eval: $TAG"
echo "  Base: $BASEPATH"
echo "  Checkpoint: $EA_MODEL_PATH"
echo "  temperature=$TEMP (>0 ⇒ sample-based verify branch)"
echo "  seeds=$SEEDS"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

IFS=',' read -ra SEED_LIST <<< "$SEEDS"
for s in "${SEED_LIST[@]}"; do
    s_trim=$(echo "$s" | xargs)
    seed_tag="${TAG}_seed${s_trim}"
    echo ""
    echo "--- Running tag=$seed_tag seed=$s_trim ---"
    python bapo/eval.py \
        --base-model-path "$BASEPATH" \
        --ea-model-path "$EA_MODEL_PATH" \
        --tag "$seed_tag" \
        --bench-name "$BENCH" \
        --top-k "$TOP_K" \
        --depth "$DEPTH" \
        --total-token "$TOTAL_TOKEN" \
        --temperature "$TEMP" \
        --seed "$s_trim" \
        --output-dir "$OUTPUT_DIR"
done
