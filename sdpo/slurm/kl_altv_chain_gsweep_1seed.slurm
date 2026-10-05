#!/bin/bash
# KL + ALTV chain γ-sweep, 1 seed only (fast). One slurm per setup.
# γ ∈ {7, 9, 12, 15, 20} × 3 modes × 2 losses = 30 evals per slurm.
#
# Required env:
#   SETUP_TAG  — q17_q06 / q32_q4
#   BASEPATH   — target HF id

#SBATCH --job-name=kl_altv_chain_1seed
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

OUT=smalllm_tree_eval_results
BENCHES="mt_bench,gsm8k,humaneval"
GAMMAS="7 9 12 15 20"

ROOT06=/scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b
ROOTQ4=/scratch/tx856/spec_reason/scratch/loss_train_smalllm_q32_q4

# 1 ckpt per loss (s0 for train-multi-seed, eseed0 for single-train)
declare -a ENTRIES=()

case "$SETUP_TAG" in
    q17_q06)
        ENTRIES=(
            "q17_q06_kl_l2k_6ep_s0                  $ROOT06/q17_q06_kl_l2k_6ep_s0/state_5              0"
            "q17_q06_altv_only_l2k_6ep_eseed0       $ROOT06/q17_q06_altv_only_l2k_6ep/state_5          0"
        )
        ;;
    q32_q4)
        ENTRIES=(
            "q32_q4_kl_l2k_6ep_s0                   $ROOTQ4/q32_q4_kl_l2k_6ep_s0/state_2               0"
            "q32_q4_altv_l2k_6ep_s0                 $ROOTQ4/q32_q4_altv_l2k_6ep_s0/state_2             0"
        )
        ;;
    *) echo "[ERROR] unknown SETUP_TAG"; exit 1 ;;
esac

declare -a MODES=(
    "greedy         --temperature 0.0 --draft-mode argmax --verify-mode auto"
    "samp_t05_ratio --temperature 0.5 --draft-mode sample --verify-mode ratio"
    "samp_t1_ratio  --temperature 1.0 --draft-mode sample --verify-mode ratio"
)

TOTAL=$(( ${#ENTRIES[@]} * $(echo $GAMMAS | wc -w) * ${#MODES[@]} ))
echo "=============================="
echo "  KL + ALTV chain γ-sweep (1 seed) / $SETUP_TAG"
echo "  Target: $BASEPATH"
echo "  γ = $GAMMAS"
echo "  Total: $TOTAL evals"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

I=0
START=$(date '+%s')

for ENTRY in "${ENTRIES[@]}"; do
    read -r SUFFIX CKPT ESEED <<< "$ENTRY"
    if [ ! -f "$CKPT/pytorch_model.bin" ]; then
        echo "[SKIP] $SUFFIX missing"
        I=$((I + $(echo $GAMMAS | wc -w) * ${#MODES[@]}))
        continue
    fi
    for G in $GAMMAS; do
        for MODE_LINE in "${MODES[@]}"; do
            I=$((I + 1))
            read -r MODE_SUF MODE_ARGS <<< "$MODE_LINE"
            FULL_TAG="${SUFFIX}_chain_g${G}_${MODE_SUF}"
            OUT_FILE="$OUT/${FULL_TAG}.json"
            if [ -f "$OUT_FILE" ]; then
                echo "[$I/$TOTAL] SKIP $FULL_TAG (done)"
                continue
            fi
            echo ""
            echo "[$I/$TOTAL] === $FULL_TAG ==="
            echo "  ckpt:  $CKPT"
            echo "  γ=$G  mode=$MODE_SUF  eseed=$ESEED"
            echo "  start: $(date '+%H:%M:%S')"
            set +e
            python sdpo/eval_small_lm_tree.py \
                --base-model-path "$BASEPATH" \
                --draft-model-path "$CKPT" \
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
