#!/bin/bash
# ALTV ckpt tree window-size sweep: γ ∈ {4, 7, 10, 15} × 2 top_k/budget configs.
# 8 (γ, config) pairs × 2 modes (greedy + samp_t1_ratio T=1) = 16 evals per job.
# Each config has budget ≥ tree's max nodes so the tree can fully expand to depth γ.
#
# Required env:
#   TAG         — tag prefix
#   DRAFT_CKPT  — abs path to draft ckpt dir
#   BASEPATH    — target HF id

#SBATCH --job-name=altv_tree_sweep
#SBATCH --time=12:00:00
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
#SBATCH --output=logs/altv_tree_sweep/%x_%j.out
#SBATCH --error=logs/altv_tree_sweep/%x_%j.err

REPO=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
cd "$REPO"
mkdir -p logs/altv_tree_sweep smalllm_tree_eval_results

source ~/.bashrc
conda activate ${CONDA_ENV:-longreason}

export HF_HOME=/scratch/tx856/.huggingface
export TORCH_EXTENSIONS_DIR=/scratch/tx856/.cache/torch_extensions
export TRITON_CACHE_DIR=/scratch/tx856/.cache/triton

: ${TAG:?must set TAG}
: ${DRAFT_CKPT:?must set DRAFT_CKPT}
: ${BASEPATH:?must set BASEPATH}

OUT=smalllm_tree_eval_results
BENCHES="mt_bench,gsm8k,humaneval"

# (gamma | top_k | budget | label)
declare -a CONFIGS=(
    "4   4,3,2,1                                  64    narrow"
    "4   5,4,3,2                                 256    wide"
    "7   4,3,2,1,1,1,1                           192    narrow"
    "7   5,4,3,2,1,1,1                           640    wide"
    "10  4,3,2,1,1,1,1,1,1,1                     256    narrow"
    "10  5,4,3,2,1,1,1,1,1,1                    1024    wide"
    "15  4,3,2,1,1,1,1,1,1,1,1,1,1,1,1           384    narrow"
    "15  5,4,3,2,1,1,1,1,1,1,1,1,1,1,1          1600    wide"
)

declare -a MODES=(
    "greedy        --temperature 0.0 --draft-mode argmax --verify-mode auto"
    "samp_t1_ratio --temperature 1.0 --draft-mode sample --verify-mode ratio"
)

echo "=============================="
echo "  ALTV tree window sweep: $TAG"
echo "  Target: $BASEPATH"
echo "  CKPT:   $DRAFT_CKPT"
echo "  Bench:  $BENCHES"
echo "  Total:  ${#CONFIGS[@]} configs × ${#MODES[@]} modes = $((${#CONFIGS[@]} * ${#MODES[@]})) evals"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

TOTAL=$((${#CONFIGS[@]} * ${#MODES[@]}))
START=$(date '+%s')
I=0

for CFG_LINE in "${CONFIGS[@]}"; do
    read -r GAMMA TOPK BUDGET LABEL <<< "$CFG_LINE"
    for MODE_LINE in "${MODES[@]}"; do
        I=$((I + 1))
        read -r MODE_SUF MODE_ARGS <<< "$MODE_LINE"
        FULL_TAG="${TAG}_g${GAMMA}_${LABEL}_b${BUDGET}_${MODE_SUF}"
        OUT_FILE="$OUT/${FULL_TAG}.json"
        if [ -f "$OUT_FILE" ]; then
            echo "[$I/$TOTAL] SKIP $FULL_TAG (done)"
            continue
        fi
        echo ""
        echo "[$I/$TOTAL] === $FULL_TAG ==="
        echo "  γ=$GAMMA top_k=$TOPK budget=$BUDGET mode=$MODE_SUF"
        echo "  start: $(date '+%H:%M:%S')"
        set +e
        python sdpo/eval_small_lm_tree.py \
            --base-model-path "$BASEPATH" \
            --draft-model-path "$DRAFT_CKPT" \
            --bench-name "$BENCHES" \
            --gamma "$GAMMA" \
            --top-k "$TOPK" \
            --budget "$BUDGET" \
            --max-new-tokens 256 \
            --num-samples 40 \
            --output-dir "$OUT" \
            --seed 0 \
            --tag "$FULL_TAG" \
            $MODE_ARGS
        RC=$?
        set -e
        echo "  end:   $(date '+%H:%M:%S')  rc=$RC"
        if [ $RC -eq 0 ] && [ -f "$OUT_FILE" ]; then
            python -c "
import json
d = json.load(open('$OUT_FILE'))
for b, modes in d.get('results', {}).items():
    a = (modes.get('tree') or {}).get('mean_alpha') or modes.get('mean_alpha')
    if a is not None: print(f'    {b}: α={a:.4f}')
"
        fi
    done
done

ELAPSED=$(( $(date '+%s') - START ))
echo ""
echo "=============================="
echo "  DONE  $(date)  elapsed=${ELAPSED}s"
echo "=============================="
