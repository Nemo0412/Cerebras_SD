#!/bin/bash
# Submit Gemma 4 31B target + E2B draft sweep.
#
# Usage:
#   bash sdpo/slurm/submit_sweep_gemma4.sh                # submit ALL
#   bash sdpo/slurm/submit_sweep_gemma4.sh g4_kl_l2k_6ep  # single
#   bash sdpo/slurm/submit_sweep_gemma4.sh --list         # list tasks
#
# Env vars:
#   EPOCHS=6 MAX_SAMPLES=60000 TRAINPATH=...   # override defaults
#   SLURM_DEPENDENCY=afterok:11629690          # wait for regen
#   GRPO_COEF=0.1 GRPO_MODE=sample DRAFTPATH=<state_2>  # stage-2 GRPO continuation

set -e
cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
mkdir -p logs/sdpo_gemma4

SLURM_SCRIPT=sdpo/slurm/gemma4_train.slurm
NUM_EPOCHS=${EPOCHS:-6}
MAX_SAMPLES=${MAX_SAMPLES:-60000}

# Default Gemma 4 60K-L2K regen output (sharegpt). Override via TRAINPATH env.
TRAINPATH_DEFAULT=${TRAINPATH:-/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/sharegpt_gemma4_31b_regen.jsonl}

# Format: tag|aux_loss|coef|tmax|tmin|baseline|trainpath
TASKS=(
    # ─── L2K 60K SFT (KL / KLV4 / KL+ALTV) ─────────────────────────
    "g4_e2b_kl_regendata_6ep         |none                 |0.0|0.1|0.1|eagle_only|"
    "g4_e2b_klv4_regendata_6ep       |acceptance_length_v4 |0.1|0.1|0.1|          |"
    "g4_e2b_kl_altv_regendata_6ep    |al_tv                |0.1|0.1|0.1|          |"
    # ─── Stage 1: KLV4 3ep SFT (for GRPO continuation) ─────────────
    "g4_e2b_klv4_3ep_sft_regendata   |acceptance_length_v4 |0.1|0.1|0.1|          |"
    # ─── Stage 2: KLV4 + GRPO sample (continuation from state_2) ───
    # Submit with: EPOCHS=3 GRPO_COEF=0.1 GRPO_MODE=sample GRPO_REWARD=hard
    #             DRAFTPATH=<saveroot>/g4_e2b_klv4_3ep_sft_regendata/state_2
    "g4_e2b_klv4_grpo_smp_regendata  |acceptance_length_v4 |0.1|0.1|0.1|          |"
)

if [ "$1" = "--list" ]; then
    echo "Available Gemma 4 tasks:"
    for task in "${TASKS[@]}"; do
        IFS='|' read -r tag aux coef tmax tmin bl tp <<< "$task"
        printf "  %-36s aux=%-22s coef=%-5s T=(%s->%s) %s\n" \
            "$(echo $tag|xargs)" "$(echo $aux|xargs)" "$coef" "$tmin" "$tmax" \
            "$([ -n "$(echo $bl|xargs)" ] && echo baseline=$(echo $bl|xargs))"
    done
    exit 0
fi

submit() {
    local tag=$1 aux=$2 coef=$3 tmax=$4 tmin=$5 bl=$6 tp=$7
    EXTRA_EXPORT=""
    [ -n "$bl" ] && EXTRA_EXPORT=",BASELINE=$bl"

    local TRAINPATH_USE="${tp:-$TRAINPATH_DEFAULT}"

    JOB_ID=$(sbatch \
        --job-name="g4_${tag}" \
        --output="logs/sdpo_gemma4/train_${tag}_%j.out" \
        --error="logs/sdpo_gemma4/train_${tag}_%j.err" \
        --time="${SLURM_TRAIN_TIME:-1-23:30:00}" \
        ${SLURM_DEPENDENCY:+--dependency="$SLURM_DEPENDENCY"} \
        --export=ALL,TAG="$tag",AUX_LOSS="$aux",SIGMOID_COEF="$coef",T_MAX="$tmax",T_MIN="$tmin",NUM_EPOCHS="$NUM_EPOCHS",MAX_SAMPLES="$MAX_SAMPLES",TRAINPATH="$TRAINPATH_USE",BASEPATH="${BASEPATH:-google/gemma-4-31B-it}",DRAFTPATH="${DRAFTPATH:-google/gemma-4-E2B-it}",GRPO_COEF="${GRPO_COEF:-0.0}",GRPO_MODE="${GRPO_MODE:-window}",GRPO_REWARD="${GRPO_REWARD:-hard}",GRPO_K_GROUPS="${GRPO_K_GROUPS:-8}",GRPO_M="${GRPO_M:-4}",GRPO_EPS="${GRPO_EPS:-0.2}",GRPO_SAMPLE_TEMP="${GRPO_SAMPLE_TEMP:-1.0}",GRPO_REWARD_ETA="${GRPO_REWARD_ETA:-1.0}",GRPO_REWARD_EPS="${GRPO_REWARD_EPS:-2.0}",SAVE_EPOCHS="${SAVE_EPOCHS:-2}",MAX_LEN="${MAX_LEN:-2048}",CONDA_ENV="${CONDA_ENV:-longreason}"${EXTRA_EXPORT} \
        "$SLURM_SCRIPT" | awk '{print $4}')
    echo "  Submitted $tag (data=${TRAINPATH_USE##*/}) -> $JOB_ID"
}

SELECTED=("$@")
echo "Gemma 4 sweep (epochs=$NUM_EPOCHS, samples=${MAX_SAMPLES:-full})"
echo "================================================================"
count=0
for task in "${TASKS[@]}"; do
    IFS='|' read -r tag aux coef tmax tmin bl tp <<< "$task"
    tag=$(echo "$tag" | xargs); aux=$(echo "$aux" | xargs)
    coef=$(echo "$coef" | xargs); tmax=$(echo "$tmax" | xargs)
    tmin=$(echo "$tmin" | xargs); bl=$(echo "$bl" | xargs); tp=$(echo "${tp:-}" | xargs)

    if [ ${#SELECTED[@]} -gt 0 ]; then
        match=0
        for sel in "${SELECTED[@]}"; do
            [ "$sel" = "$tag" ] && match=1 && break
        done
        [ $match -eq 0 ] && continue
    fi
    submit "$tag" "$aux" "$coef" "$tmax" "$tmin" "$bl" "$tp"
    count=$((count + 1))
done
echo ""
echo "$count jobs submitted."
