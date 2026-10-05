#!/bin/bash
# ALTV+GRPO chain γ-sweep, 3 eval seeds, 3 modes.
# Per (setup) slurm: γ ∈ {7, 9, 12, 15, 20} × 3 modes × 3 eval seeds = 45 evals.
#
# Required env:
#   SETUP_TAG   — q8_q06 / q17_q06
#   BASEPATH    — target HF id
#   DRAFT_CKPT  — path to ALTV+GRPO state_2

#SBATCH --job-name=altv_grpo_chain
#SBATCH --time=1-23:30:00
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
#SBATCH --output=logs/chain_gsweep/%x_%j.out
#SBATCH --error=logs/chain_gsweep/%x_%j.err

REPO=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
cd "$REPO"
mkdir -p logs/chain_gsweep smalllm_tree_eval_results

source ~/.bashrc
conda activate ${CONDA_ENV:-longreason}

export HF_HOME=/scratch/tx856/.huggingface
export TORCH_EXTENSIONS_DIR=/scratch/tx856/.cache/torch_extensions
export TRITON_CACHE_DIR=/scratch/tx856/.cache/triton

: ${SETUP_TAG:?must set SETUP_TAG}
: ${BASEPATH:?must set BASEPATH}
: ${DRAFT_CKPT:?must set DRAFT_CKPT}

OUT=smalllm_tree_eval_results
BENCHES="mt_bench,gsm8k,humaneval"
GAMMAS="7 9 12 15 20"
EVAL_SEEDS="0 1 2"

declare -a MODES=(
    "greedy         --temperature 0.0 --draft-mode argmax --verify-mode auto"
    "samp_t05_ratio --temperature 0.5 --draft-mode sample --verify-mode ratio"
    "samp_t1_ratio  --temperature 1.0 --draft-mode sample --verify-mode ratio"
)

# Base tag suffix (e.g. q8_q06_altv_grpo_smp_v1)
BASE_TAG="${SETUP_TAG}_altv_grpo_smp_v1"

TOTAL=$(( $(echo $GAMMAS | wc -w) * ${#MODES[@]} * $(echo $EVAL_SEEDS | wc -w) ))
echo "=============================="
echo "  ALTV+GRPO chain γ-sweep: $SETUP_TAG"
echo "  Target: $BASEPATH"
echo "  Draft:  $DRAFT_CKPT"
echo "  γ = $GAMMAS   eval_seeds = $EVAL_SEEDS"
echo "  Total runs: $TOTAL"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

I=0
START=$(date '+%s')

for G in $GAMMAS; do
    for MODE_LINE in "${MODES[@]}"; do
        read -r MODE_SUF MODE_ARGS <<< "$MODE_LINE"
        for ESEED in $EVAL_SEEDS; do
            I=$((I + 1))
            FULL_TAG="${BASE_TAG}_eseed${ESEED}_chain_g${G}_${MODE_SUF}"
            OUT_FILE="$OUT/${FULL_TAG}.json"
            if [ -f "$OUT_FILE" ]; then
                echo "[$I/$TOTAL] SKIP $FULL_TAG (done)"
                continue
            fi
            echo ""
            echo "[$I/$TOTAL] === $FULL_TAG ==="
            echo "  γ=$G  mode=$MODE_SUF  eseed=$ESEED"
            echo "  start: $(date '+%H:%M:%S')"
            set +e
            python sdpo/eval_small_lm_tree.py \
                --base-model-path "$BASEPATH" \
                --draft-model-path "$DRAFT_CKPT" \
                --bench-name "$BENCHES" \
                --max-new-tokens 256 \
                --num-samples 40 \
                --output-dir "$OUT" \
                --seed "$ESEED" \
                --tag "$FULL_TAG" \
                --baseline-only \
                --baseline-gamma "$G" \
                $MODE_ARGS
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
done

ELAPSED=$(( $(date '+%s') - START ))
echo ""
echo "=============================="
echo "  DONE  $(date)  elapsed=${ELAPSED}s"
echo "=============================="
