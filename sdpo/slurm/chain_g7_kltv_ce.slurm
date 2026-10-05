#!/bin/bash
# Chain decode γ=7 for KLTV + CE losses, all 3 modes, 3 eval seeds per ckpt.
# One slurm per setup (Q0.6B / Q1.7B / Q4B).
#
# Required env:
#   SETUP_TAG  — q8_q06 / q17_q06 / q32_q4
#   BASEPATH   — target HF id

#SBATCH --job-name=chain_kltv_ce
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
#SBATCH --output=logs/chain_g7/%x_%j.out
#SBATCH --error=logs/chain_g7/%x_%j.err

REPO=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
cd "$REPO"
mkdir -p logs/chain_g7 smalllm_tree_eval_results

source ~/.bashrc
conda activate ${CONDA_ENV:-longreason}

export HF_HOME=/scratch/tx856/.huggingface
export TORCH_EXTENSIONS_DIR=/scratch/tx856/.cache/torch_extensions
export TRITON_CACHE_DIR=/scratch/tx856/.cache/triton

: ${SETUP_TAG:?must set SETUP_TAG}
: ${BASEPATH:?must set BASEPATH}

OUT=smalllm_tree_eval_results
BENCHES="mt_bench,gsm8k,humaneval"
GAMMA=7

ROOT06=/scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b
ROOTQ4=/scratch/tx856/spec_reason/scratch/loss_train_smalllm_q32_q4

# (suffix, ckpt_path, num_eval_seeds) — if num_eval_seeds=0, treat as single train seed (--seed 0 only).
# If num_eval_seeds=3, run with eval seeds 0/1/2 (for ckpts that lack train multi-seed).
declare -a CKPTS=()

case "$SETUP_TAG" in
    q8_q06)
        CKPTS=(
            # KLTV — 1 ckpt × 3 eval seeds
            "q8_q06_kltv_regen_l2k_6ep            $ROOT06/q8_q06_kltv_regen_l2k_6ep/state_5      3"
            # CE — 1 ckpt × 3 eval seeds (state_2 because Q06 CE only 3-ep trained)
            "q8_q06_ce_regen_l2k                  $ROOT06/q8_q06_ce_regen_l2k/state_2            3"
            # KL+ALTV — 5 train seeds
            "q8_q06_klaltv_T10_c05_l2k_6ep_s0     $ROOT06/q8_q06_klaltv_T10_c05_l2k_6ep_s0/state_5  1"
            "q8_q06_klaltv_T10_c05_l2k_6ep_s1     $ROOT06/q8_q06_klaltv_T10_c05_l2k_6ep_s1/state_5  1"
            "q8_q06_klaltv_T10_c05_l2k_6ep_s2     $ROOT06/q8_q06_klaltv_T10_c05_l2k_6ep_s2/state_5  1"
            "q8_q06_klaltv_T10_c05_l2k_6ep_s3     $ROOT06/q8_q06_klaltv_T10_c05_l2k_6ep_s3/state_5  1"
            "q8_q06_klaltv_T10_c05_l2k_6ep_s4     $ROOT06/q8_q06_klaltv_T10_c05_l2k_6ep_s4/state_5  1"
        )
        ;;
    q17_q06)
        CKPTS=(
            "q17_q06_kltv_l2k_6ep_s0              $ROOT06/q17_q06_kltv_l2k_6ep_s0/state_5        3"
            "q17_q06_ce_l2k_6ep_s0                $ROOT06/q17_q06_ce_l2k_6ep_s0/state_5          3"
            # KL+ALTV — 1 ckpt × 3 eval seeds (no train multi-seed for Q1.7B)
            "q17_q06_klaltv_T10_c05_l2k_6ep       $ROOT06/q17_q06_klaltv_T10_c05_l2k_6ep/state_5 3"
        )
        ;;
    q32_q4)
        CKPTS=(
            "q32_q4_kltv_l2k_6ep_s0               $ROOTQ4/q32_q4_kltv_l2k_6ep_s0/state_2         3"
            "q32_q4_ce_l2k_6ep_s0                 $ROOTQ4/q32_q4_ce_l2k_6ep_s0/state_5           3"
        )
        ;;
    *) echo "[ERROR] unknown SETUP_TAG"; exit 1 ;;
esac

declare -a MODES=(
    "greedy        --temperature 0.0 --draft-mode argmax --verify-mode auto"
    "samp_t05_ratio --temperature 0.5 --draft-mode sample --verify-mode ratio"
    "samp_t1_ratio  --temperature 1.0 --draft-mode sample --verify-mode ratio"
)
TOTAL=0
for E in "${CKPTS[@]}"; do
    read -r _S _C N <<< "$E"
    TOTAL=$((TOTAL + ${#MODES[@]} * N))
done
echo "=============================="
echo "  Chain γ=$GAMMA / $SETUP_TAG / KLTV + CE + KL_ALTV"
echo "  Target: $BASEPATH"
echo "  Modes:  greedy / samp_t05_ratio / samp_t1_ratio"
echo "  Total runs: $TOTAL"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

I=0
START=$(date '+%s')

for CKPT_LINE in "${CKPTS[@]}"; do
    read -r SUFFIX CKPT NSEED <<< "$CKPT_LINE"
    if [ ! -f "$CKPT/pytorch_model.bin" ]; then
        echo "[SKIP] $SUFFIX: $CKPT/pytorch_model.bin missing"
        I=$((I + ${#MODES[@]} * NSEED))
        continue
    fi
    # Train-seed ckpt: NSEED=1 → just one eval with --seed 0
    # Single-train ckpt: NSEED=3 → three eval with --seed 0,1,2
    if [ "$NSEED" = "3" ]; then
        SEEDS_LIST="0 1 2"
    else
        SEEDS_LIST="0"
    fi
    for MODE_LINE in "${MODES[@]}"; do
        read -r MODE_SUF MODE_ARGS <<< "$MODE_LINE"
        for ESEED in $SEEDS_LIST; do
            I=$((I + 1))
            if [ "$NSEED" = "3" ]; then
                FULL_TAG="${SUFFIX}_eseed${ESEED}_chain_g${GAMMA}_${MODE_SUF}"
            else
                FULL_TAG="${SUFFIX}_chain_g${GAMMA}_${MODE_SUF}"
            fi
            OUT_FILE="$OUT/${FULL_TAG}.json"
            if [ -f "$OUT_FILE" ]; then
                echo "[$I/$TOTAL] SKIP $FULL_TAG (done)"
                continue
            fi
            echo ""
            echo "[$I/$TOTAL] === $FULL_TAG ==="
            echo "  ckpt:  $CKPT"
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
                --baseline-gamma "$GAMMA" \
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
