#!/bin/bash
# Submit all 11 BAPO sweep runs as separate SLURM jobs.
#
# Usage:
#   bash bapo/slurm/submit_sweep.sh
#   bash bapo/slurm/submit_sweep.sh 6800    # custom max_samples

set -e
cd /scratch/xt2251/SDPO-Speculative-Decoding-Policy-Optimization

SLURM_SCRIPT=bapo/slurm/bapo_run.slurm
EPOCHS=${1:-3}
MAX_SAMPLES=""  # empty = use full 68K training set

submit() {
    local tag=$1 phase=$2 wf=$3 wn=$4 wfail=$5 wpost=$6 kl=$7 samp=$8 pcoef=${9:-0.0} ptau=${10:-1.0}

    JOB_ID=$(sbatch \
        --job-name="bapo_${tag}" \
        --export=ALL,TAG="$tag",PHASE="$phase",W_FAR="$wf",W_NEAR="$wn",W_FAIL="$wfail",W_POST="$wpost",KL_COEF="$kl",SAMPLING="$samp",PROD_COEF="$pcoef",PROD_TAU="$ptau",EPOCHS="$EPOCHS",MAX_SAMPLES="${MAX_SAMPLES}" \
        "$SLURM_SCRIPT" | awk '{print $4}')

    echo "  Submitted $tag -> $JOB_ID"
}

echo "Submitting BAPO sweep ($MAX_SAMPLES samples, $EPOCHS epoch each)"
echo "================================================================"

#        tag              phase   w_far w_near w_fail w_post kl     sampling
# ───────────────────────────────────────────────────────────────────────────
# 0. Distill-only baseline
submit  "distill_only"   distill  1.0   1.0   1.0   0.0   0.01   greedy

# 1. BAPO default
submit  "default"        bapo     1.0   2.0   6.0   0.5   0.01   greedy

# 2. Ablation: uniform weights
submit  "uniform"        bapo     1.0   1.0   1.0   1.0   0.01   greedy

# 3–4. Boundary strength
submit  "wfail_3"        bapo     1.0   2.0   3.0   0.5   0.01   greedy
submit  "wfail_10"       bapo     1.0   2.0   10.0  0.5   0.01   greedy

# 5–6. KL regularization
submit  "kl_high"        bapo     1.0   2.0   6.0   0.5   0.1    greedy
submit  "kl_low"         bapo     1.0   2.0   6.0   0.5   0.001  greedy

# 7–8. Post-boundary signal
submit  "no_post"        bapo     1.0   2.0   6.0   0.0   0.01   greedy
submit  "post_strong"    bapo     1.0   2.0   6.0   2.0   0.01   greedy

# 9. Stochastic rollouts
submit  "sample"         bapo     1.0   2.0   6.0   0.5   0.01   sample

# 10. Boundary-only (extreme)
submit  "boundary_only"  bapo     0.0   0.0   8.0   0.0   0.01   greedy

# 11–13. Product acceptance loss (differentiable E[L] surrogate)
#        tag              phase  wf   wn   wfail wpost kl    samp    pcoef ptau
submit  "prod_only"      bapo   0.0  0.0  0.0   0.0   0.01  greedy  1.0   1.0
submit  "prod_plus"      bapo   1.0  2.0  6.0   0.5   0.01  greedy  0.5   1.0
submit  "prod_tau_low"   bapo   1.0  2.0  6.0   0.5   0.01  greedy  0.5   0.3

echo ""
echo "14 jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs in: logs/bapo/"
echo "Wandb project: bapo"
