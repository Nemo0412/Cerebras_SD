#!/bin/bash
# Submit tree+chain eval for Qwen3-32B target + Qwen3-0.6B draft ckpts.
# One sbatch per ckpt. Uses sdpo/slurm/smalllm_tree_eval.slurm.
#
# Tree config (your spec):
#   gamma=7, budget=128, top-k=4,3,2,1,1,1,1
# Chain baseline: gamma=7
# Benches: mt_bench, gsm8k, humaneval (80 samples each)
#
# Usage:
#   bash sdpo/slurm/submit_eval_q32_q06_tree.sh              # auto-detect latest state for each completed ckpt
#   bash sdpo/slurm/submit_eval_q32_q06_tree.sh --list       # list what would be submitted
#   bash sdpo/slurm/submit_eval_q32_q06_tree.sh q32_q06_klv2_regen_ro  # single tag (uses latest state)
#   STATE=1 bash sdpo/slurm/submit_eval_q32_q06_tree.sh q32_q06_klv2_regen_ro   # force specific state

set -e
cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
mkdir -p logs/smalllm_rollout logs/sdpo_smalllm_q32_q06

SLURM_SCRIPT=sdpo/slurm/smalllm_tree_eval.slurm

BASEPATH=Qwen/Qwen3-32B
# NOTE: BENCH and TOP_K contain commas. They CANNOT be put in --export list
# (sbatch splits --export by comma regardless of escapes). Instead, set them
# as env vars of the sbatch process itself — --export=ALL then forwards them.
BENCH_VAL="mt_bench,gsm8k,humaneval"
TOP_K_VAL="4,3,2,1,1,1,1"
NUM_SAMPLES=80
MAX_NEW_TOKENS=256

# Tree config
GAMMA=7
BUDGET=128
BASELINE_GAMMA=7
RUN_BASELINE=1

OUTPUT_DIR=smalllm_tree_eval_results

STATE_OVERRIDE=${STATE:-}

# ── ckpt roots for noro / ro ──────────────────────────────────────────────
NORO_ROOT=/scratch/tx856/spec_reason/scratch/loss_train_smalllm_q32_q06
RO_ROOT=/scratch/tx856/spec_reason/scratch/loss_train_smalllm_rollout

# Find latest state_N dir (N in 0..2) for a tag+root.
latest_state() {
    local root=$1 tag=$2
    local dir="$root/$tag"
    local n
    if [ -n "$STATE_OVERRIDE" ]; then
        if [ -d "$dir/state_$STATE_OVERRIDE" ]; then
            echo "state_$STATE_OVERRIDE"
        fi
        return 0
    fi
    for n in 2 1 0; do
        if [ -d "$dir/state_$n" ]; then
            echo "state_$n"
            return 0
        fi
    done
    return 0
}

# Collect all available ckpts from both roots
declare -a CKPTS
for root in "$NORO_ROOT" "$RO_ROOT"; do
    [ -d "$root" ] || continue
    for tag_dir in "$root"/q32_q06_*/; do
        [ -d "$tag_dir" ] || continue
        tag=$(basename "$tag_dir")
        state=$(latest_state "$root" "$tag")
        [ -z "$state" ] && continue
        CKPTS+=("$tag|$root|$state")
    done
done

if [ "$1" = "--list" ]; then
    echo "Available ckpts to eval:"
    for entry in "${CKPTS[@]}"; do
        IFS='|' read -r tag root state <<< "$entry"
        echo "  $tag  [$state]  -> $root/$tag/$state"
    done
    exit 0
fi

SELECTED=("$@")

submit() {
    local tag=$1 root=$2 state=$3
    local ckpt="$root/$tag/$state"
    local eval_tag="${tag}_${state}"
    [ -f "$OUTPUT_DIR/${eval_tag}.json" ] && {
        echo "  skip $eval_tag (json exists)"; return
    }

    JOB_ID=$(BENCH="$BENCH_VAL" TOP_K="$TOP_K_VAL" sbatch \
        --job-name="ev_${tag}" \
        --output="logs/smalllm_rollout/eval_${eval_tag}_%j.out" \
        --error="logs/smalllm_rollout/eval_${eval_tag}_%j.err" \
        --constraint="a100|h100|h200" \
        --mem=128G \
        --time=0-08:00:00 \
        --export=ALL,TAG="$eval_tag",BASEPATH="$BASEPATH",DRAFT_CKPT="$ckpt",GAMMA="$GAMMA",BUDGET="$BUDGET",MAX_NEW_TOKENS="$MAX_NEW_TOKENS",NUM_SAMPLES="$NUM_SAMPLES",BASELINE_GAMMA="$BASELINE_GAMMA",RUN_BASELINE="$RUN_BASELINE",OUTPUT_DIR="$OUTPUT_DIR" \
        "$SLURM_SCRIPT" | awk '{print $4}')

    printf "  %-32s %-9s -> %s\n" "$tag" "$state" "$JOB_ID"
}

echo "Tree+Chain eval (target=Qwen3-32B, draft=Qwen3-0.6B)"
echo "  benches=$BENCH_VAL"
echo "  tree: gamma=$GAMMA budget=$BUDGET top_k=$TOP_K_VAL"
echo "  samples=$NUM_SAMPLES  max_new_tokens=$MAX_NEW_TOKENS"
echo "=========================================================="

count=0
for entry in "${CKPTS[@]}"; do
    IFS='|' read -r tag root state <<< "$entry"

    if [ ${#SELECTED[@]} -gt 0 ]; then
        match=0
        for sel in "${SELECTED[@]}"; do
            [ "$sel" = "$tag" ] && match=1 && break
        done
        [ $match -eq 0 ] && continue
    fi

    submit "$tag" "$root" "$state"
    count=$((count + 1))
done

echo ""
echo "$count jobs submitted. Monitor with: squeue -u \$USER"
echo "Results in: $OUTPUT_DIR/"
