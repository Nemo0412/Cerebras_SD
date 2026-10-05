#!/bin/bash
# Submit training sweep on the LOCAL 10K mixed dataset (mixed_train_10K.jsonl).
# Same loss combos as submit_regen_sweep.sh, just different data.
# Each train job auto-chains a tree+chain eval (--dependency=afterok).
#
# Tag suffix: "_10k" (parallel to "_regen") so ckpt dirs do not collide.
#
# Usage:
#   bash sdpo/slurm/submit_10k_sweep.sh                 # submit ALL
#   bash sdpo/slurm/submit_10k_sweep.sh q8_q06_klv4_10k # single tag
#   bash sdpo/slurm/submit_10k_sweep.sh --list          # list tasks
#   bash sdpo/slurm/submit_10k_sweep.sh --klv2          # filter KL+V2
#   bash sdpo/slurm/submit_10k_sweep.sh --v4            # filter all V4
#   bash sdpo/slurm/submit_10k_sweep.sh --klv4          # filter KL+V4
#   NO_EVAL=1 bash sdpo/slurm/submit_10k_sweep.sh ...   # skip auto-eval

set -e
cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
mkdir -p logs/sdpo_smalllm_q32_q06 logs/sdpo_smalllm_06b logs/smalllm_rollout

# ── Paths ─────────────────────────────────────────────────────────────────
TRAIN_10K=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_train_10K.jsonl
VAL_SMALL=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_val_80.jsonl

SLURM_NORO=sdpo/slurm/smalllm_06b_train.slurm
SLURM_RO=sdpo/slurm/smalllm_rollout_train.slurm
SLURM_EVAL=sdpo/slurm/smalllm_tree_eval.slurm
DS_CONFIG_8B_NORO=sdpo/sdpo_config_qwen3.json
DS_CONFIG_ZERO3=sdpo/sdpo_config_qwen3_bf16_zero3_2gpu.json

NUM_EPOCHS=${EPOCHS:-3}
MAX_SAMPLES=${MAX_SAMPLES:-""}

# ── Eval config (chain + tree, gamma=7 budget=128 top_k=4,3,2,1,1,1,1) ────
# NOTE: vars containing commas (BENCH, TOP_K) are passed via the sbatch
# process env (left side of sbatch call), picked up by --export=ALL.
# They cannot be inside --export=ALL,KEY=val,... — sbatch splits by comma
# regardless of backslashes.
EVAL_BASEPATH_8B=Qwen/Qwen3-8B
EVAL_BASEPATH_32B=Qwen/Qwen3-32B
EVAL_BENCH_VAL="mt_bench,gsm8k,humaneval"
EVAL_TOP_K_VAL="4,3,2,1,1,1,1"
EVAL_NUM_SAMPLES=80
EVAL_MAX_NEW_TOKENS=256
EVAL_GAMMA=7
EVAL_BUDGET=128
EVAL_BASELINE_GAMMA=7
EVAL_OUTPUT_DIR=smalllm_tree_eval_results

# ── Task table: "tag|target|anchor|aux|rollout" ────────────────────────────
TASKS=(
    # ─── 8B + 0.6B ───────────────────────────────────────────
    "q8_q06_klv2_10k       | 8b  | kl | v2   | noro"
    "q8_q06_klv2_10k_ro    | 8b  | kl | v2   | ro"
    "q8_q06_cev2_10k       | 8b  | ce | v2   | noro"
    "q8_q06_cev2_10k_ro    | 8b  | ce | v2   | ro"
    "q8_q06_klv4_10k       | 8b  | kl | v4   | noro"
    "q8_q06_klv4_10k_ro    | 8b  | kl | v4   | ro"
    "q8_q06_cev4_10k       | 8b  | ce | v4   | noro"
    "q8_q06_cev4_10k_ro    | 8b  | ce | v4   | ro"
    "q8_q06_kltv_10k       | 8b  | kl | tv   | noro"
    "q8_q06_kltv_10k_ro    | 8b  | kl | tv   | ro"
    "q8_q06_cetv_10k       | 8b  | ce | tv   | noro"
    "q8_q06_cetv_10k_ro    | 8b  | ce | tv   | ro"
    "q8_q06_kl_10k         | 8b  | kl | none | noro"
    "q8_q06_kl_10k_ro      | 8b  | kl | none | ro"
    "q8_q06_ce_10k         | 8b  | ce | none | noro"
    "q8_q06_ce_10k_ro      | 8b  | ce | none | ro"

    # ─── 32B + 0.6B ──────────────────────────────────────────
    "q32_q06_klv2_10k      | 32b | kl | v2   | noro"
    "q32_q06_klv2_10k_ro   | 32b | kl | v2   | ro"
    "q32_q06_cev2_10k      | 32b | ce | v2   | noro"
    "q32_q06_cev2_10k_ro   | 32b | ce | v2   | ro"
    "q32_q06_klv4_10k      | 32b | kl | v4   | noro"
    "q32_q06_klv4_10k_ro   | 32b | kl | v4   | ro"
    "q32_q06_cev4_10k      | 32b | ce | v4   | noro"
    "q32_q06_cev4_10k_ro   | 32b | ce | v4   | ro"
    "q32_q06_kltv_10k      | 32b | kl | tv   | noro"
    "q32_q06_kltv_10k_ro   | 32b | kl | tv   | ro"
    "q32_q06_cetv_10k      | 32b | ce | tv   | noro"
    "q32_q06_cetv_10k_ro   | 32b | ce | tv   | ro"
    "q32_q06_kl_10k        | 32b | kl | none | noro"
    "q32_q06_kl_10k_ro     | 32b | kl | none | ro"
    "q32_q06_ce_10k        | 32b | ce | none | noro"
    "q32_q06_ce_10k_ro     | 32b | ce | none | ro"
)

# ── --list ────────────────────────────────────────────────────────────────
if [ "$1" = "--list" ]; then
    echo "10K dataset training tasks:"
    for task in "${TASKS[@]}"; do
        IFS='|' read -r tag tgt anchor aux rollout <<< "$task"
        tag=$(echo "$tag" | xargs); tgt=$(echo "$tgt" | xargs)
        anchor=$(echo "$anchor" | xargs); aux=$(echo "$aux" | xargs)
        rollout=$(echo "$rollout" | xargs)
        printf "  %-26s target=%-4s anchor=%-2s aux=%-4s rollout=%s\n" \
            "$tag" "$tgt" "$anchor" "$aux" "$rollout"
    done
    exit 0
fi

# ── Filters ───────────────────────────────────────────────────────────────
FILTER=""
if [ "$1" = "--klv2" ]; then FILTER="klv2"; shift
elif [ "$1" = "--v4" ];   then FILTER="v4";   shift
elif [ "$1" = "--klv4" ]; then FILTER="klv4"; shift
fi

# ── submit train + eval ───────────────────────────────────────────────────
submit() {
    local tag=$1 tgt=$2 anchor=$3 aux=$4 rollout=$5

    if [ "$tgt" = "8b" ]; then
        BASEPATH=Qwen/Qwen3-8B
        EVAL_BASE=$EVAL_BASEPATH_8B
    else
        BASEPATH=Qwen/Qwen3-32B
        EVAL_BASE=$EVAL_BASEPATH_32B
    fi
    DRAFTPATH=Qwen/Qwen3-0.6B

    if [ "$rollout" = "ro" ]; then
        SCRIPT=$SLURM_RO
        DEEPSPEED_CONFIG=$DS_CONFIG_ZERO3
        SAVEROOT=/scratch/tx856/spec_reason/scratch/loss_train_smalllm_rollout
        LOGDIR=logs/smalllm_rollout
    else
        SCRIPT=$SLURM_NORO
        if [ "$tgt" = "32b" ]; then
            DEEPSPEED_CONFIG=$DS_CONFIG_ZERO3
            SAVEROOT=/scratch/tx856/spec_reason/scratch/loss_train_smalllm_q32_q06
            LOGDIR=logs/sdpo_smalllm_q32_q06
        else
            DEEPSPEED_CONFIG=$DS_CONFIG_8B_NORO
            SAVEROOT=/scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b
            LOGDIR=logs/sdpo_smalllm_06b
        fi
    fi

    # Submit training
    JOB_ID=$(sbatch \
        --job-name="$tag" \
        --output="${LOGDIR}/train_${tag}_%j.out" \
        --error="${LOGDIR}/train_${tag}_%j.err" \
        --export=ALL,TAG="$tag",BASEPATH="$BASEPATH",DRAFTPATH="$DRAFTPATH",ANCHOR="$anchor",AUX_LOSS="$aux",SIGMOID_COEF=0.1,T_MAX=0.1,T_MIN=0.1,LR=1e-6,NUM_EPOCHS="$NUM_EPOCHS",MAX_SAMPLES="$MAX_SAMPLES",TRAINPATH="$TRAIN_10K",TESTPATH="$VAL_SMALL",DEEPSPEED_CONFIG="$DEEPSPEED_CONFIG",SAVEROOT="$SAVEROOT" \
        "$SCRIPT" | awk '{print $4}')

    printf "  train  %-26s (%-4s %s+%-4s %-4s) -> %s\n" \
        "$tag" "$tgt" "$anchor" "$aux" "$rollout" "$JOB_ID"

    # Auto-chain eval (after train succeeds). Skip if NO_EVAL set.
    if [ -z "$NO_EVAL" ]; then
        local EVAL_TAG="${tag}_state2"
        local CKPT="$SAVEROOT/$tag/state_2"
        # BENCH + TOP_K contain commas — set as env vars of the sbatch process,
        # then --export=ALL forwards them. Everything without commas goes in
        # the --export list.
        local EVAL_JOB_ID=$(BENCH="$EVAL_BENCH_VAL" TOP_K="$EVAL_TOP_K_VAL" \
            sbatch \
            --job-name="ev_${tag}" \
            --output="logs/smalllm_rollout/eval_${EVAL_TAG}_%j.out" \
            --error="logs/smalllm_rollout/eval_${EVAL_TAG}_%j.err" \
            --constraint="a100|h100|h200" \
            --mem=128G --time=0-08:00:00 \
            --dependency=afterok:$JOB_ID \
            --export=ALL,TAG="$EVAL_TAG",BASEPATH="$EVAL_BASE",DRAFT_CKPT="$CKPT",GAMMA="$EVAL_GAMMA",BUDGET="$EVAL_BUDGET",MAX_NEW_TOKENS="$EVAL_MAX_NEW_TOKENS",NUM_SAMPLES="$EVAL_NUM_SAMPLES",BASELINE_GAMMA="$EVAL_BASELINE_GAMMA",RUN_BASELINE=1,OUTPUT_DIR="$EVAL_OUTPUT_DIR" \
            "$SLURM_EVAL" | awk '{print $4}')
        printf "  eval   %-26s (deps on %s)            -> %s\n" \
            "$EVAL_TAG" "$JOB_ID" "$EVAL_JOB_ID"
    fi
}

# ── Loop ──────────────────────────────────────────────────────────────────
SELECTED=("$@")

echo "10K sweep (epochs=$NUM_EPOCHS, samples=${MAX_SAMPLES:-full}, filter=${FILTER:-none}, eval=${NO_EVAL:+OFF}${NO_EVAL:-ON})"
echo "================================================================================"

count=0
for task in "${TASKS[@]}"; do
    IFS='|' read -r tag tgt anchor aux rollout <<< "$task"
    tag=$(echo "$tag" | xargs); tgt=$(echo "$tgt" | xargs)
    anchor=$(echo "$anchor" | xargs); aux=$(echo "$aux" | xargs)
    rollout=$(echo "$rollout" | xargs)

    if [ "$FILTER" = "klv2" ]; then
        [ "$anchor" = "kl" ] && [ "$aux" = "v2" ] || continue
    elif [ "$FILTER" = "v4" ]; then
        [ "$aux" = "v4" ] || continue
    elif [ "$FILTER" = "klv4" ]; then
        [ "$anchor" = "kl" ] && [ "$aux" = "v4" ] || continue
    fi

    if [ ${#SELECTED[@]} -gt 0 ]; then
        match=0
        for sel in "${SELECTED[@]}"; do
            [ "$sel" = "$tag" ] && { match=1; break; }
        done
        [ $match -eq 0 ] && continue
    fi

    submit "$tag" "$tgt" "$anchor" "$aux" "$rollout"
    count=$((count + 1))
done

echo ""
echo "$count train jobs submitted (each with chained eval). Monitor: squeue -u \$USER"
