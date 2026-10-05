#!/bin/bash
# Chain decoding window-size sweep: γ ∈ {7, 10, 12, 15} × samp_t1_ratio mode.
# Single mode (sample, ratio verify, T=1), 4 γ values per job = 4 evals.
#
# Required env:
#   TAG         — tag prefix
#   DRAFT_CKPT  — abs path to draft ckpt dir
#   BASEPATH    — target HF id

#SBATCH --job-name=chain_sweep
#SBATCH --time=8:00:00
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
#SBATCH --output=logs/chain_sweep/%x_%j.out
#SBATCH --error=logs/chain_sweep/%x_%j.err

REPO=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
cd "$REPO"
mkdir -p logs/chain_sweep smalllm_tree_eval_results

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
GAMMAS="7 10 12 15"

echo "=============================="
echo "  Chain window sweep: $TAG"
echo "  Target: $BASEPATH  Draft: $DRAFT_CKPT"
echo "  γ values: $GAMMAS   mode: samp_t1_ratio (T=1, sample, ratio)"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

TOTAL=$(echo $GAMMAS | wc -w)
I=0
START=$(date '+%s')

for G in $GAMMAS; do
    I=$((I + 1))
    FULL_TAG="${TAG}_chain_g${G}_samp_t1_ratio"
    OUT_FILE="$OUT/${FULL_TAG}.json"
    if [ -f "$OUT_FILE" ]; then
        echo "[$I/$TOTAL] SKIP $FULL_TAG (done)"
        continue
    fi
    echo ""
    echo "[$I/$TOTAL] === $FULL_TAG (γ=$G) ==="
    echo "  start: $(date '+%H:%M:%S')"
    set +e
    python sdpo/eval_small_lm_tree.py \
        --base-model-path "$BASEPATH" \
        --draft-model-path "$DRAFT_CKPT" \
        --bench-name "$BENCHES" \
        --max-new-tokens 256 \
        --num-samples 40 \
        --output-dir "$OUT" \
        --seed 0 \
        --tag "$FULL_TAG" \
        --baseline-only \
        --baseline-gamma "$G" \
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
