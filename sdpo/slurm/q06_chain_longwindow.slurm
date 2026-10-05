#!/bin/bash
# Q0.6B + Q8B target chain decode at long γ (= 256 or 1024). Single-ckpt sweep:
#   (γ=256, max_new=256)
#   (γ=256, max_new=1024)
#   (γ=1024, max_new=1024)
# All in samp_t1_ratio mode (T=1, sample, ratio verify). 3 benches (mt/gsm/he).
#
# Required env:
#   TAG          — output tag prefix (e.g. q8_q06_kl, q8_q06_altv)
#   DRAFT_CKPT   — abs path to draft ckpt dir
#   BASEPATH     — target HF id

#SBATCH --job-name=q06_chain_lw
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
#SBATCH --output=logs/q06_chain_lw/%x_%j.out
#SBATCH --error=logs/q06_chain_lw/%x_%j.err

REPO=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
cd "$REPO"
mkdir -p logs/q06_chain_lw smalllm_tree_eval_results

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

# (gamma, max_new_tokens, suffix)
declare -a CONFIGS=(
    " 256   256   g256_mn256"
    " 256  1024   g256_mn1024"
    "1024  1024   g1024_mn1024"
)

TOTAL=${#CONFIGS[@]}
START=$(date '+%s')

echo "=============================="
echo "  Q0.6B chain long γ sweep: $TAG"
echo "  Target: $BASEPATH"
echo "  CKPT:   $DRAFT_CKPT"
echo "  Bench:  $BENCHES"
echo "  Total:  $TOTAL configs (samp_t1_ratio)"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

I=0
for CFG in "${CONFIGS[@]}"; do
    I=$((I + 1))
    read -r GAMMA MAXNEW SUFFIX <<< "$CFG"
    FULL_TAG="${TAG}_chain_${SUFFIX}_samp_t1_ratio"
    OUT_FILE="$OUT/${FULL_TAG}.json"
    if [ -f "$OUT_FILE" ]; then
        echo "[$I/$TOTAL] SKIP $FULL_TAG (done)"
        continue
    fi
    echo ""
    echo "[$I/$TOTAL] === $FULL_TAG ==="
    echo "  γ=$GAMMA  max_new_tokens=$MAXNEW"
    echo "  start: $(date '+%H:%M:%S')"
    set +e
    python sdpo/eval_small_lm_tree.py \
        --base-model-path "$BASEPATH" \
        --draft-model-path "$DRAFT_CKPT" \
        --bench-name "$BENCHES" \
        --max-new-tokens "$MAXNEW" \
        --num-samples 40 \
        --output-dir "$OUT" \
        --seed 0 \
        --tag "$FULL_TAG" \
        --baseline-only \
        --baseline-gamma "$GAMMA" \
        --temperature 1.0 \
        --draft-mode sample \
        --verify-mode ratio
    RC=$?
    set -e
    echo "  end:   $(date '+%H:%M:%S')  rc=$RC"
    if [ $RC -eq 0 ] && [ -f "$OUT_FILE" ]; then
        python -c "
import json
d = json.load(open('$OUT_FILE'))
for b, modes in d.get('results', {}).items():
    a = (modes.get('baseline_chain') or modes.get('chain') or {}).get('mean_alpha') or modes.get('mean_alpha')
    if a is not None: print(f'    {b}: α={a:.4f}')
"
    fi
done

ELAPSED=$(( $(date '+%s') - START ))
echo ""
echo "=============================="
echo "  DONE  $(date)  elapsed=${ELAPSED}s"
echo "=============================="
