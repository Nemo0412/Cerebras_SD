#!/bin/bash
# Chain decode γ=7 evaluation, multi-seed coverage for 5 losses.
# Use train seeds where available; for single-train ckpts use 3 eval seeds.
# One slurm = one (SETUP, MODE) pair.
#
# Required env:
#   SETUP_TAG  — q8_q06 / q17_q06 / q32_q4
#   BASEPATH   — target HF id
#   MODE       — greedy / samp_t05_ratio / samp_t1_ratio

#SBATCH --job-name=chain_g7_multi
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
: ${MODE:?must set MODE}
# PHASE: 1=kl+altv, 2=klv4 (no GRPO), 3=grpo; default unset → all losses
PHASE=${PHASE:-}

OUT=smalllm_tree_eval_results
BENCHES="mt_bench,gsm8k,humaneval"
GAMMA=7

case "$MODE" in
    greedy)         MODE_ARGS="--temperature 0.0 --draft-mode argmax --verify-mode auto" ;;
    samp_t05_ratio) MODE_ARGS="--temperature 0.5 --draft-mode sample --verify-mode ratio" ;;
    samp_t1_ratio)  MODE_ARGS="--temperature 1.0 --draft-mode sample --verify-mode ratio" ;;
    *) echo "[ERROR] unknown MODE: $MODE" ; exit 1 ;;
esac

ROOT06=/scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b
ROOTQ4=/scratch/tx856/spec_reason/scratch/loss_train_smalllm_q32_q4

# Each entry: "<tag_suffix>  <ckpt_path>  <eval_seed>"
# tag_suffix becomes the output filename suffix: <suffix>_chain_g7_<mode>.json
declare -a ENTRIES=()

case "$SETUP_TAG" in
    q8_q06)
        ENTRIES=(
            # KL — 5 train seeds (sharegpt regen data, matches KLV4 seeds)
            "q8_q06_kl_l2k_6ep_s0                 $ROOT06/q8_q06_kl_l2k_6ep_s0/state_5                0"
            "q8_q06_kl_l2k_6ep_s1                 $ROOT06/q8_q06_kl_l2k_6ep_s1/state_5                0"
            "q8_q06_kl_l2k_6ep_s2                 $ROOT06/q8_q06_kl_l2k_6ep_s2/state_5                0"
            "q8_q06_kl_l2k_6ep_s3                 $ROOT06/q8_q06_kl_l2k_6ep_s3/state_5                0"
            "q8_q06_kl_l2k_6ep_s4                 $ROOT06/q8_q06_kl_l2k_6ep_s4/state_5                0"
            # KLTV — 1 train ckpt × 3 eval seeds
            "q8_q06_kltv_regen_l2k_6ep_eseed0     $ROOT06/q8_q06_kltv_regen_l2k_6ep/state_5           0"
            "q8_q06_kltv_regen_l2k_6ep_eseed1     $ROOT06/q8_q06_kltv_regen_l2k_6ep/state_5           1"
            "q8_q06_kltv_regen_l2k_6ep_eseed2     $ROOT06/q8_q06_kltv_regen_l2k_6ep/state_5           2"
            # KLV4 — 5 train seeds
            "q8_q06_klv4_l2k_6ep_s0               $ROOT06/q8_q06_klv4_l2k_6ep_s0/state_5              0"
            "q8_q06_klv4_l2k_6ep_s1               $ROOT06/q8_q06_klv4_l2k_6ep_s1/state_5              0"
            "q8_q06_klv4_l2k_6ep_s2               $ROOT06/q8_q06_klv4_l2k_6ep_s2/state_5              0"
            "q8_q06_klv4_l2k_6ep_s3               $ROOT06/q8_q06_klv4_l2k_6ep_s3/state_5              0"
            "q8_q06_klv4_l2k_6ep_s4               $ROOT06/q8_q06_klv4_l2k_6ep_s4/state_5              0"
            # KLV4 GRPO — 5 train seeds
            "q8_q06_klv4_grpo_smp_s0              $ROOT06/q8_q06_klv4_grpo_smp_s0/state_2             0"
            "q8_q06_klv4_grpo_smp_s1              $ROOT06/q8_q06_klv4_grpo_smp_s1/state_2             0"
            "q8_q06_klv4_grpo_smp_s2              $ROOT06/q8_q06_klv4_grpo_smp_s2/state_2             0"
            "q8_q06_klv4_grpo_smp_s3              $ROOT06/q8_q06_klv4_grpo_smp_s3/state_2             0"
            "q8_q06_klv4_grpo_smp_s4              $ROOT06/q8_q06_klv4_grpo_smp_s4/state_2             0"
            # ALTV — 1 train ckpt × 3 eval seeds
            "q8_q06_altv_only_regen_l2k_6ep_eseed0 $ROOT06/q8_q06_altv_only_regen_l2k_6ep/state_5     0"
            "q8_q06_altv_only_regen_l2k_6ep_eseed1 $ROOT06/q8_q06_altv_only_regen_l2k_6ep/state_5     1"
            "q8_q06_altv_only_regen_l2k_6ep_eseed2 $ROOT06/q8_q06_altv_only_regen_l2k_6ep/state_5     2"
        )
        ;;
    q17_q06)
        ENTRIES=(
            # KL — 5 train seeds
            "q17_q06_kl_l2k_6ep_s0                $ROOT06/q17_q06_kl_l2k_6ep_s0/state_5               0"
            "q17_q06_kl_l2k_6ep_s1                $ROOT06/q17_q06_kl_l2k_6ep_s1/state_5               0"
            "q17_q06_kl_l2k_6ep_s2                $ROOT06/q17_q06_kl_l2k_6ep_s2/state_5               0"
            "q17_q06_kl_l2k_6ep_s3                $ROOT06/q17_q06_kl_l2k_6ep_s3/state_5               0"
            "q17_q06_kl_l2k_6ep_s4                $ROOT06/q17_q06_kl_l2k_6ep_s4/state_5               0"
            # KLTV — 3 train seeds
            "q17_q06_kltv_l2k_6ep_s0              $ROOT06/q17_q06_kltv_l2k_6ep_s0/state_5             0"
            "q17_q06_kltv_l2k_6ep_s1              $ROOT06/q17_q06_kltv_l2k_6ep_s1/state_5             0"
            "q17_q06_kltv_l2k_6ep_s2              $ROOT06/q17_q06_kltv_l2k_6ep_s2/state_5             0"
            # KLV4 — 5 train seeds
            "q17_q06_klv4_l2k_6ep_s0              $ROOT06/q17_q06_klv4_l2k_6ep_s0/state_5             0"
            "q17_q06_klv4_l2k_6ep_s1              $ROOT06/q17_q06_klv4_l2k_6ep_s1/state_5             0"
            "q17_q06_klv4_l2k_6ep_s2              $ROOT06/q17_q06_klv4_l2k_6ep_s2/state_5             0"
            "q17_q06_klv4_l2k_6ep_s3              $ROOT06/q17_q06_klv4_l2k_6ep_s3/state_5             0"
            "q17_q06_klv4_l2k_6ep_s4              $ROOT06/q17_q06_klv4_l2k_6ep_s4/state_5             0"
            # KLV4 GRPO — 1 ckpt × 3 eval seeds
            "q17_q06_klv4_grpo_smp_fix_eseed0     $ROOT06/q17_q06_klv4_grpo_smp_fix/state_2           0"
            "q17_q06_klv4_grpo_smp_fix_eseed1     $ROOT06/q17_q06_klv4_grpo_smp_fix/state_2           1"
            "q17_q06_klv4_grpo_smp_fix_eseed2     $ROOT06/q17_q06_klv4_grpo_smp_fix/state_2           2"
            # ALTV — 1 ckpt × 3 eval seeds
            "q17_q06_altv_only_l2k_6ep_eseed0     $ROOT06/q17_q06_altv_only_l2k_6ep/state_5           0"
            "q17_q06_altv_only_l2k_6ep_eseed1     $ROOT06/q17_q06_altv_only_l2k_6ep/state_5           1"
            "q17_q06_altv_only_l2k_6ep_eseed2     $ROOT06/q17_q06_altv_only_l2k_6ep/state_5           2"
        )
        ;;
    q32_q4)
        ENTRIES=(
            # KL — 3 train seeds
            "q32_q4_kl_l2k_6ep_s0                 $ROOTQ4/q32_q4_kl_l2k_6ep_s0/state_2                0"
            "q32_q4_kl_l2k_6ep_s1                 $ROOTQ4/q32_q4_kl_l2k_6ep_s1/state_2                0"
            "q32_q4_kl_l2k_6ep_s2                 $ROOTQ4/q32_q4_kl_l2k_6ep_s2/state_2                0"
            # KLTV — 1 ckpt × 3 eval seeds
            "q32_q4_kltv_l2k_6ep_s0_eseed0        $ROOTQ4/q32_q4_kltv_l2k_6ep_s0/state_2              0"
            "q32_q4_kltv_l2k_6ep_s0_eseed1        $ROOTQ4/q32_q4_kltv_l2k_6ep_s0/state_2              1"
            "q32_q4_kltv_l2k_6ep_s0_eseed2        $ROOTQ4/q32_q4_kltv_l2k_6ep_s0/state_2              2"
            # KLV4 — 3 train seeds
            "q32_q4_klv4_l2k_6ep_s0               $ROOTQ4/q32_q4_klv4_l2k_6ep_s0/state_2              0"
            "q32_q4_klv4_l2k_6ep_s1               $ROOTQ4/q32_q4_klv4_l2k_6ep_s1/state_2              0"
            "q32_q4_klv4_l2k_6ep_s2               $ROOTQ4/q32_q4_klv4_l2k_6ep_s2/state_2              0"
            # KLV4 GRPO — 1 ckpt × 3 eval seeds
            "q32_q4_klv4_grpo_smp_fix_eseed0      $ROOTQ4/q32_q4_klv4_grpo_smp_fix/state_2            0"
            "q32_q4_klv4_grpo_smp_fix_eseed1      $ROOTQ4/q32_q4_klv4_grpo_smp_fix/state_2            1"
            "q32_q4_klv4_grpo_smp_fix_eseed2      $ROOTQ4/q32_q4_klv4_grpo_smp_fix/state_2            2"
            # ALTV — 3 train seeds
            "q32_q4_altv_l2k_6ep_s0               $ROOTQ4/q32_q4_altv_l2k_6ep_s0/state_2              0"
            "q32_q4_altv_l2k_6ep_s1               $ROOTQ4/q32_q4_altv_l2k_6ep_s1/state_2              0"
            "q32_q4_altv_l2k_6ep_s2               $ROOTQ4/q32_q4_altv_l2k_6ep_s2/state_2              0"
        )
        ;;
    *) echo "[ERROR] unknown SETUP_TAG: $SETUP_TAG"; exit 1 ;;
esac

# Phase filter: 1 = kl/altv; 2 = klv4 (NOT grpo); 3 = grpo. Empty = all.
keep_entry() {
    local suffix="$1"
    case "$PHASE" in
        1) [[ "$suffix" == *_kl_* || "$suffix" == *_altv_* ]] && return 0 ;;
        2) [[ "$suffix" == *_klv4_l2k_* ]] && return 0 ;;
        3) [[ "$suffix" == *_grpo_* ]] && return 0 ;;
        "") return 0 ;;
    esac
    return 1
}

FILTERED=()
for E in "${ENTRIES[@]}"; do
    read -r SUF _REST <<< "$E"
    if keep_entry "$SUF"; then FILTERED+=("$E"); fi
done
ENTRIES=("${FILTERED[@]}")

TOTAL=${#ENTRIES[@]}
echo "=============================="
echo "  Chain γ=$GAMMA / $SETUP_TAG / $MODE / PHASE=$PHASE"
echo "  Target:    $BASEPATH"
echo "  Total runs: $TOTAL"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

I=0
START=$(date '+%s')

for ENTRY in "${ENTRIES[@]}"; do
    I=$((I + 1))
    read -r SUFFIX CKPT ESEED <<< "$ENTRY"
    if [ ! -f "$CKPT/pytorch_model.bin" ]; then
        echo "[$I/$TOTAL] SKIP $SUFFIX: $CKPT/pytorch_model.bin missing"
        continue
    fi
    FULL_TAG="${SUFFIX}_chain_g${GAMMA}_${MODE}"
    OUT_FILE="$OUT/${FULL_TAG}.json"
    if [ -f "$OUT_FILE" ]; then
        echo "[$I/$TOTAL] SKIP $FULL_TAG (done)"
        continue
    fi
    echo ""
    echo "[$I/$TOTAL] === $FULL_TAG (eval_seed=$ESEED) ==="
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

ELAPSED=$(( $(date '+%s') - START ))
echo ""
echo "=============================="
echo "  DONE  $(date)  elapsed=${ELAPSED}s"
echo "=============================="
