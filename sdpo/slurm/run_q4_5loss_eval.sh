#!/bin/bash
# Q32B target + Q4B draft 5-loss eval driver.
#
# ⚠️ All Q4B ckpts are state_2 ONLY (SAVE_EPOCHS=2,5 was eaten by sbatch's
# comma-split bug → state_5 never saved). Best we have is end-of-3ep.
#
# Setups:
#   KL        q32_q4_kl_l2k_6ep_s{0,1,2}              3 seeds × state_2
#   KLV4      q32_q4_klv4_l2k_6ep_s{0,1,2}            3 seeds × state_2
#   KLTV      q32_q4_kltv_l2k_6ep_s0                  1 ckpt   × state_2
#   KLV4_SMP  q32_q4_klv4_grpo_smp_fix                1 ckpt   × state_2
#   ALTV      q32_q4_altv_l2k_6ep_s{0,1,2}            3 seeds × state_2

set -e
cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization

source ~/.bashrc
conda activate ${CONDA_ENV:-longreason}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCH_EXTENSIONS_DIR=/scratch/tx856/.cache/torch_extensions
export HF_HOME=${HF_HOME:-/scratch/tx856/.huggingface}

BASEPATH=Qwen/Qwen3-32B
ROOT=/scratch/tx856/spec_reason/scratch/loss_train_smalllm_q32_q4
OUT=smalllm_tree_eval_results
LOG_DIR=logs/q4_5loss_eval
mkdir -p "$LOG_DIR" "$OUT"

declare -a CKPTS=(
    "q32_q4_kl_l2k_6ep_s0               $ROOT/q32_q4_kl_l2k_6ep_s0/state_2"
    "q32_q4_kl_l2k_6ep_s1               $ROOT/q32_q4_kl_l2k_6ep_s1/state_2"
    "q32_q4_kl_l2k_6ep_s2               $ROOT/q32_q4_kl_l2k_6ep_s2/state_2"

    "q32_q4_klv4_l2k_6ep_s0             $ROOT/q32_q4_klv4_l2k_6ep_s0/state_2"
    "q32_q4_klv4_l2k_6ep_s1             $ROOT/q32_q4_klv4_l2k_6ep_s1/state_2"
    "q32_q4_klv4_l2k_6ep_s2             $ROOT/q32_q4_klv4_l2k_6ep_s2/state_2"

    "q32_q4_kltv_l2k_6ep_s0             $ROOT/q32_q4_kltv_l2k_6ep_s0/state_2"

    "q32_q4_klv4_grpo_smp_fix           $ROOT/q32_q4_klv4_grpo_smp_fix/state_2"

    "q32_q4_altv_l2k_6ep_s0             $ROOT/q32_q4_altv_l2k_6ep_s0/state_2"
    "q32_q4_altv_l2k_6ep_s1             $ROOT/q32_q4_altv_l2k_6ep_s1/state_2"
    "q32_q4_altv_l2k_6ep_s2             $ROOT/q32_q4_altv_l2k_6ep_s2/state_2"
)

declare -a MODES=(
    "nothink_greedy        --temperature 0.0 --draft-mode argmax --verify-mode auto"
    "nothink_samp_t1_ratio --temperature 1.0 --draft-mode sample --verify-mode ratio"
)

COMMON_ARGS="
    --base-model-path $BASEPATH
    --bench-name mt_bench,gsm8k,humaneval,longwriter_top50,osr2_top50
    --gamma 7
    --top-k 4,3,2,1,1,1,1
    --budget 128
    --max-new-tokens 256
    --num-samples 40
    --output-dir $OUT
    --no-thinking
    --seed 0
"

TOTAL=$((${#CKPTS[@]} * ${#MODES[@]}))
START=$(date '+%s')
echo "=============================="
echo "  Q4B (q32_q4_*) 5-loss eval driver"
echo "  Target: $BASEPATH  (32B — needs 1× H200, ~64 GB VRAM)"
echo "  Total:  ${#CKPTS[@]} ckpts × ${#MODES[@]} modes = $TOTAL"
echo "  GPU:    $CUDA_VISIBLE_DEVICES"
echo "  Logs:   $LOG_DIR"
echo "=============================="

I=0
for CKPT_LINE in "${CKPTS[@]}"; do
    read -r TAG CKPT <<< "$CKPT_LINE"
    if [ ! -f "$CKPT/pytorch_model.bin" ]; then
        echo "[SKIP] $TAG: $CKPT/pytorch_model.bin not found"
        I=$((I + ${#MODES[@]}))
        continue
    fi
    for MODE_LINE in "${MODES[@]}"; do
        I=$((I + 1))
        read -r SUFFIX MODE_ARGS <<< "$MODE_LINE"
        FULL_TAG="${TAG}_${SUFFIX}"
        OUT_FILE="$OUT/${FULL_TAG}.json"
        LOG_FILE="$LOG_DIR/${FULL_TAG}.log"
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
            $COMMON_ARGS \
            --draft-model-path "$CKPT" \
            --tag "$FULL_TAG" \
            $MODE_ARGS > "$LOG_FILE" 2>&1
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
echo "  ALL DONE  $(date)  elapsed=${ELAPSED}s"
echo "=============================="
