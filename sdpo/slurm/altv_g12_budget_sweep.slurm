#!/bin/bash
# ALTV ckpt eval at gamma=12 + budget sweep (one ckpt per job).
# Loops 5 benches × 2 modes (greedy + samp_t1_ratio) × 3 budgets = 30 runs.
# Tree top_k = [5,4,3,2,1,1,1,1,1,1,1,1] (length 12), wider root + tail at top_k=1.
#
# Required env vars:
#   TAG           — eval result tag prefix
#   DRAFT_CKPT    — absolute path to draft ckpt dir
#   BASEPATH      — target HF id (Qwen3-8B or Qwen3-32B)
# Optional env vars:
#   CONDA_ENV     — default longreason

#SBATCH --job-name=altv_g12
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100|h200"
#SBATCH --requeue
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tx856@nyu.edu
#SBATCH --output=logs/altv_g12/%x_%j.out
#SBATCH --error=logs/altv_g12/%x_%j.err

REPO=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
cd "$REPO"
mkdir -p logs/altv_g12 smalllm_tree_eval_results

source ~/.bashrc
conda activate ${CONDA_ENV:-longreason}

export HF_HOME=/scratch/tx856/.huggingface
export TORCH_EXTENSIONS_DIR=/scratch/tx856/.cache/torch_extensions
export TRITON_CACHE_DIR=/scratch/tx856/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/scratch/tx856/.cache/torchinductor

: ${TAG:?must set TAG}
: ${DRAFT_CKPT:?must set DRAFT_CKPT}
: ${BASEPATH:?must set BASEPATH}

OUT=smalllm_tree_eval_results

# 3 budgets
BUDGETS=(128 256 512)
# 2 modes
declare -a MODES=(
    "greedy        --temperature 0.0 --draft-mode argmax --verify-mode auto"
    "samp_t1_ratio --temperature 1.0 --draft-mode sample --verify-mode ratio"
)

GAMMA=12
TOP_K="5,4,3,2,1,1,1,1,1,1,1,1"
BENCHES="mt_bench,gsm8k,humaneval,longwriter_top50,osr2_top50"

echo "=============================="
echo "  ALTV gamma=12 budget sweep"
echo "  TAG:       $TAG"
echo "  CKPT:      $DRAFT_CKPT"
echo "  TARGET:    $BASEPATH"
echo "  γ=12 top_k=[$TOP_K]"
echo "  budgets:   ${BUDGETS[*]}"
echo "  benches:   $BENCHES"
echo "  modes:     greedy / samp_t1_ratio"
echo "  Total:     ${#BUDGETS[@]} budgets × ${#MODES[@]} modes = $((${#BUDGETS[@]} * ${#MODES[@]})) eval calls"
echo "             (each call covers all 5 benches)"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

START=$(date '+%s')
I=0
TOTAL=$((${#BUDGETS[@]} * ${#MODES[@]}))

for BUDGET in "${BUDGETS[@]}"; do
    for MODE_LINE in "${MODES[@]}"; do
        I=$((I + 1))
        read -r SUFFIX MODE_ARGS <<< "$MODE_LINE"
        FULL_TAG="${TAG}_g12_b${BUDGET}_nothink_${SUFFIX}"
        OUT_FILE="$OUT/${FULL_TAG}.json"
        if [ -f "$OUT_FILE" ]; then
            echo "[$I/$TOTAL] SKIP $FULL_TAG (done)"
            continue
        fi
        echo ""
        echo "[$I/$TOTAL] === $FULL_TAG ==="
        echo "  start: $(date '+%H:%M:%S')"
        set +e
        python sdpo/eval_small_lm_tree.py \
            --base-model-path "$BASEPATH" \
            --draft-model-path "$DRAFT_CKPT" \
            --bench-name "$BENCHES" \
            --gamma "$GAMMA" --top-k "$TOP_K" --budget "$BUDGET" \
            --max-new-tokens 256 --num-samples 40 \
            --output-dir "$OUT" \
            --no-thinking --seed 0 \
            --tag "$FULL_TAG" \
            $MODE_ARGS
        RC=$?
        set -e
        echo "  end:   $(date '+%H:%M:%S')  rc=$RC"
        if [ $RC -eq 0 ] && [ -f "$OUT_FILE" ]; then
            python -c "
import json
with open('$OUT_FILE') as f: d = json.load(f)
for b, modes in d.get('results', {}).items():
    for m, v in modes.items():
        a = v.get('mean_alpha') if isinstance(v, dict) else None
        if a is None and isinstance(v, dict):
            a = v.get('tree', {}).get('mean_alpha') or v.get('chain', {}).get('mean_alpha')
        if a is not None:
            print(f'    {b} ({m}): {a:.4f}')
"
        fi
    done
done

ELAPSED=$(( $(date '+%s') - START ))
echo ""
echo "=============================="
echo "  DONE  $(date)  elapsed=${ELAPSED}s"
echo "=============================="
