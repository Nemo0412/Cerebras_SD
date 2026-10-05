#!/bin/bash
# Q8B target + Q1.7B-trained draft (q17_q06_* ckpt) 5-loss eval driver.
# Same shape as run_q06_5loss_eval.sh. ckpts trained against Q1.7B target then
# eval'd with Q8B target — matches prior "q17_*" naming convention.
#
# Setups (state_5 unless noted):
#   KL        q17_q06_kl_l2k_6ep_s{0..4}              5 seeds × state_5
#   KLV4      q17_q06_klv4_l2k_6ep_s{0..4}            5 seeds × state_5
#   KLTV      q17_q06_kltv_l2k_6ep_s0                 1 ckpt   × state_5
#   KLV4_SMP  q17_q06_klv4_grpo_smp_fix  state_2      1 ckpt   × state_2 (GRPO 3ep)
#   ALTV      q17_q06_altv_only_l2k_6ep               1 ckpt   × state_5

set -e
cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization

source ~/.bashrc
conda activate ${CONDA_ENV:-longreason}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCH_EXTENSIONS_DIR=/scratch/tx856/.cache/torch_extensions
export HF_HOME=${HF_HOME:-/scratch/tx856/.huggingface}

BASEPATH=Qwen/Qwen3-8B
ROOT=/scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b
OUT=smalllm_tree_eval_results
LOG_DIR=logs/q17_5loss_eval
mkdir -p "$LOG_DIR" "$OUT"

declare -a CKPTS=(
    "q17_q06_kl_l2k_6ep_s0               $ROOT/q17_q06_kl_l2k_6ep_s0/state_5"
    "q17_q06_kl_l2k_6ep_s1               $ROOT/q17_q06_kl_l2k_6ep_s1/state_5"
    "q17_q06_kl_l2k_6ep_s2               $ROOT/q17_q06_kl_l2k_6ep_s2/state_5"
    "q17_q06_kl_l2k_6ep_s3               $ROOT/q17_q06_kl_l2k_6ep_s3/state_5"
    "q17_q06_kl_l2k_6ep_s4               $ROOT/q17_q06_kl_l2k_6ep_s4/state_5"

    "q17_q06_klv4_l2k_6ep_s0             $ROOT/q17_q06_klv4_l2k_6ep_s0/state_5"
    "q17_q06_klv4_l2k_6ep_s1             $ROOT/q17_q06_klv4_l2k_6ep_s1/state_5"
    "q17_q06_klv4_l2k_6ep_s2             $ROOT/q17_q06_klv4_l2k_6ep_s2/state_5"
    "q17_q06_klv4_l2k_6ep_s3             $ROOT/q17_q06_klv4_l2k_6ep_s3/state_5"
    "q17_q06_klv4_l2k_6ep_s4             $ROOT/q17_q06_klv4_l2k_6ep_s4/state_5"

    "q17_q06_kltv_l2k_6ep_s0             $ROOT/q17_q06_kltv_l2k_6ep_s0/state_5"

    "q17_q06_klv4_grpo_smp_fix_state2    $ROOT/q17_q06_klv4_grpo_smp_fix/state_2"

    "q17_q06_altv_only_l2k_6ep           $ROOT/q17_q06_altv_only_l2k_6ep/state_5"
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
echo "  Q1.7B (q17_q06_*) 5-loss eval driver"
echo "  Target: $BASEPATH"
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
