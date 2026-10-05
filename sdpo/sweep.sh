#!/bin/bash
# SDPO Sigmoid Loss Hyperparameter Sweep
# α × T grid, all runs sequential, no manual intervention.
# Check wandb project 'sdpo_sigmoid' after completion.

# Continue on failure so one bad run doesn't kill the whole sweep

export HF_TOKEN=""

BASE="meta-llama/Llama-3.1-8B-Instruct"
DRAFT="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
TRAIN="data/mixed_train_10K.jsonl"
TEST="data/mixed_val.jsonl"
DS_CONFIG="sdpo_config.json"
SAMPLES=6800
EPOCHS=1
GAMMA=7

run() {
    local name=$1
    local alpha=$2
    local tmin=$3
    local tmax=$4
    echo "============================================"
    echo "Running: $name (alpha=$alpha, T_min=$tmin, T_max=$tmax)"
    echo "============================================"
    deepspeed main.py \
        --basepath $BASE \
        --draftpath $DRAFT \
        --trainpath $TRAIN \
        --testpath $TEST \
        --savedir "sdpo_sweep/$name" \
        --deepspeed_config $DS_CONFIG \
        --sigmoid_coef $alpha \
        --T_max $tmax --T_min $tmin \
        --gamma $GAMMA --num_epochs $EPOCHS \
        --max_train_samples $SAMPLES
}

# ── Baseline: pure EAGLE (α=0) ────────────────────────────────────
run "alpha0.00" 0.0 1.0 1.0

# ── α sweep × T sweep grid ────────────────────────────────────────
for alpha in 0.01 0.02 0.05 0.1 0.2 0.5 1.0 2.0; do
    for T in 0.1 0.5 1.0 2.0; do
        name="alpha${alpha}_T${T}"
        run "$name" $alpha $T $T
    done
done

echo "===== SWEEP COMPLETE (33 runs) ====="
echo "Check wandb project 'sdpo_sigmoid' for results."
