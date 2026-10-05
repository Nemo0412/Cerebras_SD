#!/bin/bash
# BAPO Hypothesis-Driven Sweep (10 runs)
#
# Each run tests ONE hypothesis against a shared default.
# ~1 epoch on 6800 samples per run for fast iteration.
#
# Priority axes:
#   1. w_fail    — the core contribution (boundary focus)
#   2. kl_coef   — stability vs signal
#   3. w_post    — post-boundary distillation (novel)
#   4. sampling  — greedy vs stochastic rollouts
#   5. ablation  — does boundary weighting matter at all?
#
# Usage:
#   bash bapo/sweep.sh

set -e

BASE=/scratch/xt2251/models/Llama-3.1-8B-Instruct
DRAFT=/scratch/xt2251/models/EAGLE3-LLaMA3.1-Instruct-8B
TRAIN=sdpo/data/mixed_train.jsonl
TEST=sdpo/data/mixed_val.jsonl
DS_CONFIG=bapo/bapo_config.json
CFG=sdpo/config.json
GAMMA=7
EPOCHS=1
MAX_SAMPLES=6800

run() {
    local tag=$1 wf=$2 wn=$3 wfail=$4 wpost=$5 kl=$6 samp=$7
    echo "=========================================="
    echo "  [$tag] w=($wf,$wn,$wfail,$wpost) kl=$kl $samp"
    echo "=========================================="
    deepspeed bapo/main.py \
        --basepath "$BASE" \
        --draftpath "$DRAFT" \
        --trainpath "$TRAIN" \
        --testpath "$TEST" \
        --deepspeed_config "$DS_CONFIG" \
        --config_path "$CFG" \
        --savedir "bapo_sweep/${tag}" \
        --mode finetune \
        --bapo_epochs "$EPOCHS" \
        --w_far "$wf" --w_near "$wn" --w_fail "$wfail" --w_post "$wpost" \
        --kl_coef "$kl" \
        --sampling "$samp" \
        --gamma "$GAMMA" \
        --max_train_samples "$MAX_SAMPLES"
}

# 0. Distill-only baseline (standard EAGLE3 KL — no BAPO at all)
echo "=========================================="
echo "  [distill_only] Phase 1 KL distillation baseline"
echo "=========================================="
deepspeed bapo/main.py \
    --basepath "$BASE" \
    --draftpath "$DRAFT" \
    --trainpath "$TRAIN" \
    --testpath "$TEST" \
    --deepspeed_config "$DS_CONFIG" \
    --config_path "$CFG" \
    --savedir "bapo_sweep/distill_only" \
    --mode finetune \
    --distill_epochs "$EPOCHS" --bapo_epochs 0 \
    --gamma "$GAMMA" \
    --max_train_samples "$MAX_SAMPLES"

#                    tag              w_far w_near w_fail w_post kl_coef  sampling
# ─────────────────────────────────────────────────────────────────────────────────
# 1. BAPO default
run  "default"       1.0   2.0   6.0   0.5   0.01    greedy

# 2. Ablation: uniform weights — does boundary weighting matter?
run  "uniform"       1.0   1.0   1.0   1.0   0.01    greedy

# 3–4. Boundary strength (most important axis)
run  "wfail_3"       1.0   2.0   3.0   0.5   0.01    greedy
run  "wfail_10"      1.0   2.0   10.0  0.5   0.01    greedy

# 5–6. KL regularization (stability vs freedom)
run  "kl_high"       1.0   2.0   6.0   0.5   0.1     greedy
run  "kl_low"        1.0   2.0   6.0   0.5   0.001   greedy

# 7–8. Post-boundary signal (your novel addition)
run  "no_post"       1.0   2.0   6.0   0.0   0.01    greedy
run  "post_strong"   1.0   2.0   6.0   2.0   0.01    greedy

# 9. Stochastic rollouts (more diverse but noisier)
run  "sample"        1.0   2.0   6.0   0.5   0.01    sample

# 10. Extreme: boundary-only (zero signal except at rejection token)
run  "boundary_only" 0.0   0.0   8.0   0.0   0.01    greedy

echo ""
echo "Sweep complete (10 runs). Results in bapo_sweep/"
echo "Compare mean_tau across runs in wandb project 'bapo'"
