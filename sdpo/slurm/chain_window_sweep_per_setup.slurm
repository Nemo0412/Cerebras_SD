#!/bin/bash
# Chain window-size sweep: 1 slurm per target/draft setup; sweeps 3 losses ×
# 4 γ values × samp_t1_ratio mode = 12 evals per job.
#
# Required env:
#   SETUP_TAG    — output tag prefix (e.g. q8_q06, q17_q06, q32_q4)
#   BASEPATH     — target HF id
#   CKPT_KL      — abs path to KL draft state_*
#   CKPT_KLV4    — abs path to KLV4 draft state_*
#   CKPT_ALTV    — abs path to ALTV draft state_*

#SBATCH --job-name=chain_setup_sweep
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

: ${SETUP_TAG:?must set SETUP_TAG}
: ${BASEPATH:?must set BASEPATH}
: ${CKPT_KL:?must set CKPT_KL}
: ${CKPT_KLV4:?must set CKPT_KLV4}
: ${CKPT_ALTV:?must set CKPT_ALTV}

OUT=smalllm_tree_eval_results
BENCHES="mt_bench,gsm8k,humaneval"
GAMMAS="7 10 12 15"

declare -a LOSSES=(
    "kl     $CKPT_KL"
    "klv4   $CKPT_KLV4"
    "altv   $CKPT_ALTV"
)

TOTAL=$((${#LOSSES[@]} * $(echo $GAMMAS | wc -w)))
I=0
START=$(date '+%s')

echo "=============================="
echo "  Chain sweep: $SETUP_TAG"
echo "  Target: $BASEPATH"
echo "  Losses: kl / klv4 / altv  ×  γ = {7,10,12,15}"
echo "  Mode:   samp_t1_ratio (T=1, sample, ratio verify)"
echo "  Total:  $TOTAL evals"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

for ENTRY in "${LOSSES[@]}"; do
    read -r LOSS CKPT <<< "$ENTRY"
    if [ ! -f "$CKPT/pytorch_model.bin" ]; then
        echo "[SKIP loss=$LOSS] missing $CKPT/pytorch_model.bin"
        I=$((I + $(echo $GAMMAS | wc -w)))
        continue
    fi
    for G in $GAMMAS; do
        I=$((I + 1))
        FULL_TAG="${SETUP_TAG}_${LOSS}_chain_g${G}_samp_t1_ratio"
        OUT_FILE="$OUT/${FULL_TAG}.json"
        if [ -f "$OUT_FILE" ]; then
            echo "[$I/$TOTAL] SKIP $FULL_TAG (done)"
            continue
        fi
        echo ""
        echo "[$I/$TOTAL] === $FULL_TAG ==="
        echo "  ckpt: $CKPT  γ=$G"
        echo "  start: $(date '+%H:%M:%S')"
        set +e
        python sdpo/eval_small_lm_tree.py \
            --base-model-path "$BASEPATH" \
            --draft-model-path "$CKPT" \
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
done

ELAPSED=$(( $(date '+%s') - START ))
echo ""
echo "=============================="
echo "  ALL DONE  $(date)  elapsed=${ELAPSED}s"
echo "=============================="
