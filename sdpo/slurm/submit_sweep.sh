#!/bin/bash
# Submit acceptance length loss sweep as separate SLURM jobs.
#
# Usage:
#   bash sdpo/slurm/submit_sweep.sh           # full dataset, 3 epochs
#   bash sdpo/slurm/submit_sweep.sh 1 6800    # 1 epoch, 6800 samples

set -e
cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization

SLURM_SCRIPT=sdpo/slurm/acclength_train.slurm
NUM_EPOCHS=${1:-3}
MAX_SAMPLES=${2:-""}

submit() {
    local tag=$1 aux=$2 coef=$3 tmax=$4 tmin=$5

    JOB_ID=$(sbatch \
        --job-name="al_${tag}" \
        --output="logs/sdpo_acclength/train_${tag}_%j.out" \
        --error="logs/sdpo_acclength/train_${tag}_%j.err" \
        --export=ALL,TAG="$tag",AUX_LOSS="$aux",SIGMOID_COEF="$coef",T_MAX="$tmax",T_MIN="$tmin",NUM_EPOCHS="$NUM_EPOCHS",MAX_SAMPLES="$MAX_SAMPLES" \
        "$SLURM_SCRIPT" | awk '{print $4}')

    echo "  Submitted $tag -> $JOB_ID"
}

echo "AccLength sweep (epochs=$NUM_EPOCHS, samples=${MAX_SAMPLES:-full})"
echo "================================================================"

# ─── Baselines ────────────────────────────────────────────────
#        tag                  aux_loss             coef   T_max  T_min

# 0. Eagle-only baseline (coef=0, aux loss ignored)
submit  "eagle_only"         "sigmoid"             0.0    1.0    1.0

# 1. Original sigmoid loss baseline
submit  "sigmoid_default"    "sigmoid"             0.1    5.0    1.0

# ─── Acceptance length loss: coef sweep ───────────────────────
submit  "al_coef_0.01"       "acceptance_length"   0.01   5.0    1.0
submit  "al_coef_0.05"       "acceptance_length"   0.05   5.0    1.0
submit  "al_coef_0.1"        "acceptance_length"   0.1    5.0    1.0
submit  "al_coef_0.2"        "acceptance_length"   0.2    5.0    1.0
submit  "al_coef_0.5"        "acceptance_length"   0.5    5.0    1.0
submit  "al_coef_1.0"        "acceptance_length"   1.0    5.0    1.0

# ─── Acceptance length loss: temperature sweep (coef=0.1) ─────
submit  "al_T_0.5"           "acceptance_length"   0.1    0.5    0.5
submit  "al_T_1.0"           "acceptance_length"   0.1    1.0    1.0
submit  "al_T_2.0"           "acceptance_length"   0.1    2.0    2.0
submit  "al_T_10.0"          "acceptance_length"   0.1   10.0   10.0

# ─── Acceptance length loss: temperature schedule ─────────────
submit  "al_sched_01_5"      "acceptance_length"   0.1    5.0    0.1
submit  "al_sched_1_10"      "acceptance_length"   0.1   10.0    1.0

# ─── Acceptance length loss: high coef + low T ────────────────
submit  "al_c0.5_T1"         "acceptance_length"   0.5    1.0    1.0
submit  "al_c1.0_T2"         "acceptance_length"   1.0    2.0    2.0

echo ""
echo "16 jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs in: logs/sdpo_acclength/"
echo "Wandb project: sdpo_sigmoid"
