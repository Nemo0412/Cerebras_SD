#!/bin/bash
# Submit Qwen3-8B EAGLE3 sweep.
#
# Usage:
#   bash sdpo/slurm/submit_sweep_qwen3.sh                          # submit ALL
#   bash sdpo/slurm/submit_sweep_qwen3.sh q3_coef_0.1              # submit single task
#   bash sdpo/slurm/submit_sweep_qwen3.sh q3_coef_0.1 q3_coef_0.5  # submit multiple
#   bash sdpo/slurm/submit_sweep_qwen3.sh --list                   # list all tasks
#
# Environment variables (optional):
#   EPOCHS=1 MAX_SAMPLES=500 bash sdpo/slurm/submit_sweep_qwen3.sh q3_coef_0.1

set -e
cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization

mkdir -p logs/sdpo_qwen3

SLURM_SCRIPT=sdpo/slurm/qwen3_train.slurm
NUM_EPOCHS=${EPOCHS:-3}
MAX_SAMPLES=${MAX_SAMPLES:-""}

# ─── Task definitions ────────────────────────────────────────
# Format: "tag|aux_loss|coef|tmax|tmin|baseline"
# baseline empty = normal mode; can be 'eagle_only' or 'ce_only'
TASKS=(
    # Baselines
    "q3_eagle_only|sigmoid|0.0|1.0|1.0|"
    "q3_sigmoid|sigmoid|0.1|5.0|1.0|"
    # V2 coef sweep
    "q3_coef_0.01|acceptance_length_v2|0.01|5.0|1.0|"
    "q3_coef_0.05|acceptance_length_v2|0.05|5.0|1.0|"
    "q3_coef_0.1|acceptance_length_v2|0.1|5.0|1.0|"
    "q3_coef_0.2|acceptance_length_v2|0.2|5.0|1.0|"
    "q3_coef_0.5|acceptance_length_v2|0.5|5.0|1.0|"
    "q3_coef_1.0|acceptance_length_v2|1.0|5.0|1.0|"
    # V2 temperature sweep (coef=0.1)
    "q3_T_0.1|acceptance_length_v2|0.1|0.1|0.1|"
    "q3_T_0.5|acceptance_length_v2|0.1|0.5|0.5|"
    "q3_T_1.0|acceptance_length_v2|0.1|1.0|1.0|"
    "q3_T_2.0|acceptance_length_v2|0.1|2.0|2.0|"
    "q3_T_10.0|acceptance_length_v2|0.1|10.0|10.0|"
    # High coef + low T
    "q3_c0.5_T1|acceptance_length_v2|0.5|1.0|1.0|"
    "q3_c1.0_T2|acceptance_length_v2|1.0|2.0|2.0|"

    "q3_coef_0.2_T0.1|acceptance_length_v2|0.2|1.0|0.1|"
    "q3_coef_0.2_T0.5|acceptance_length_v2|0.2|0.5|0.1|"

    # ─── EAGLE-3 noro on AngelSlim init: KL / KLV4 / KLTV (L2K, 6ep, 60K) ──
    "q8_eagle3_kl_regen_l2k_6ep   |sigmoid              |0.0|0.1|0.1|eagle_only"
    "q8_eagle3_klv4_regen_l2k_6ep |acceptance_length_v4 |0.1|0.1|0.1|"
    "q8_eagle3_kltv_regen_l2k_6ep |tv                   |0.1|0.1|0.1|"

    # ─── EAGLE-3 GRPO (3 SFT + 3 RL pipelines) ────────────────────────
    # Submit each in 2 stages:
    #   Stage 1: SFT 3ep on AngelSlim init (EPOCHS=3, default DRAFTPATH).
    #   Stage 2: GRPO 3ep continuing from Stage 1 ckpt
    #     EPOCHS=3 GRPO_COEF=0.1 [GRPO_MODE=window|sample] [GRPO_REWARD=hard|eal] \
    #     DRAFTPATH=/scratch/.../<stage1>/state_2 \
    #     bash submit_sweep_qwen3.sh <stage2_tag>
    # Stage1 tags (3ep SFT)
    "q8_eagle3_klv4_3ep_sft       |acceptance_length_v4 |0.1|0.1|0.1|"
    "q8_eagle3_kleal_3ep_sft      |eal                  |0.1|0.1|0.1|"
    # Stage2 tags (3ep GRPO; submit with GRPO_* env + DRAFTPATH override)
    "q8_eagle3_klv4_grpo_w_al     |acceptance_length_v4 |0.1|0.1|0.1|"
    "q8_eagle3_kleal_grpo_w_al    |eal                  |0.1|0.1|0.1|"
    "q8_eagle3_kleal_grpo_w_eal   |eal                  |0.1|0.1|0.1|"
    "q8_eagle3_kleal_grpo_s_al    |eal                  |0.1|0.1|0.1|"
    # 6 RL (no SFT loss): pure GRPO from AngelSlim init
    "q8_eagle3_grpo_only_s_al     |none                 |0.0|0.1|0.1|eagle_only"

    # ─── CE-only baseline (hard cross-entropy with target argmax) ──
    "q3_ce_only|sigmoid|0.0|0.1|0.1|ce_only"

    # ─── E[L] with β = 1 − TV(p, q) per rollout position ────────────
    # SpS-marginal accept rate; cumprod across γ, sum over positions.
    "q8_eagle3_kl_altv_6ep        |al_tv                |0.1|0.1|0.1|"
    "q8_eagle3_altv_only_6ep      |al_tv                |1.0|0.1|0.1|aux_only"

    # ─── E[L] with β = 0.5 · exp(-KL(p || q)) per rollout position ──
    # KL-derived per-step accept rate; same cumprod-sum structure as al_tv.
    "q8_eagle3_kl_alkl_6ep        |al_kl                |0.1|0.1|0.1|"
    "q8_eagle3_alkl_only_6ep      |al_kl                |1.0|0.1|0.1|aux_only"

    # ─── L̃_EAL = Σ_j (γ − j) · KL(p || q): linear-weighted per-step KL ──
    "q8_eagle3_kl_wkl_6ep         |wkl                  |0.1|0.1|0.1|"
    "q8_eagle3_wkl_only_6ep       |wkl                  |1.0|0.1|0.1|aux_only"

    # ─── New losses + GRPO ─────────────────────────────────────────
    # 3+3 stage 2: continue from existing q8_eagle3_<loss>_only_6ep/state_2 (= 3ep SFT).
    # Submit with: EPOCHS=3 GRPO_COEF=0.1 GRPO_MODE=window GRPO_REWARD=hard
    #             DRAFTPATH=<save>/q8_eagle3_<loss>_only_6ep/state_2
    "q8_eagle3_altv_3plus3_grpo   |al_tv                |1.0|0.1|0.1|aux_only"
    "q8_eagle3_alkl_3plus3_grpo   |al_kl                |1.0|0.1|0.1|aux_only"
    "q8_eagle3_wkl_3plus3_grpo    |wkl                  |1.0|0.1|0.1|aux_only"
    # 6ep from base: aux loss + GRPO simultaneously (no SFT warm-start).
    # Submit with: EPOCHS=6 GRPO_COEF=0.1 GRPO_MODE=window GRPO_REWARD=hard
    "q8_eagle3_altv_grpo_6ep      |al_tv                |1.0|0.1|0.1|aux_only"
    "q8_eagle3_alkl_grpo_6ep      |al_kl                |1.0|0.1|0.1|aux_only"
    "q8_eagle3_wkl_grpo_6ep       |wkl                  |1.0|0.1|0.1|aux_only"

    # ─── EAGLE-3 trained on Qwen3-8B target-regenerated ShareGPT (120K) ─────
    # Format: tag | aux_loss | coef | T_max | T_min | baseline | trainpath
    # Trainpath column overrides the slurm-script default (mixed_train_10K.jsonl).
    # Submit with: EPOCHS=6 MAX_SAMPLES=60000 (matches small-LM l2k convention)
    "q8_eagle3_kl_regendata_6ep        |sigmoid              |0.0|0.1|0.1|eagle_only|/scratch/yf3005/gto_data/sharegpt_qwen3_8b_regen.jsonl"
    "q8_eagle3_klv4_regendata_6ep      |acceptance_length_v4 |0.1|0.1|0.1|          |/scratch/yf3005/gto_data/sharegpt_qwen3_8b_regen.jsonl"
    "q8_eagle3_kleal_regendata_6ep     |eal                  |0.1|0.1|0.1|          |/scratch/yf3005/gto_data/sharegpt_qwen3_8b_regen.jsonl"
    "q8_eagle3_kltv_regendata_6ep      |tv                   |0.1|0.1|0.1|          |/scratch/yf3005/gto_data/sharegpt_qwen3_8b_regen.jsonl"
    # New E[L]-style losses on regen data
    "q8_eagle3_kl_altv_regendata_6ep   |al_tv                |0.1|0.1|0.1|          |/scratch/yf3005/gto_data/sharegpt_qwen3_8b_regen.jsonl"
    "q8_eagle3_altv_only_regendata_6ep |al_tv                |1.0|0.1|0.1|aux_only  |/scratch/yf3005/gto_data/sharegpt_qwen3_8b_regen.jsonl"
    "q8_eagle3_kl_alkl_regendata_6ep   |al_kl                |0.1|0.1|0.1|          |/scratch/yf3005/gto_data/sharegpt_qwen3_8b_regen.jsonl"
    "q8_eagle3_alkl_only_regendata_6ep |al_kl                |1.0|0.1|0.1|aux_only  |/scratch/yf3005/gto_data/sharegpt_qwen3_8b_regen.jsonl"
    "q8_eagle3_kl_wkl_regendata_6ep    |wkl                  |0.1|0.1|0.1|          |/scratch/yf3005/gto_data/sharegpt_qwen3_8b_regen.jsonl"
    "q8_eagle3_wkl_only_regendata_6ep  |wkl                  |1.0|0.1|0.1|aux_only  |/scratch/yf3005/gto_data/sharegpt_qwen3_8b_regen.jsonl"
    # 6ep pure GRPO on regen data — submit with GRPO_COEF=0.1 GRPO_MODE=sample GRPO_REWARD=hard
    "q8_eagle3_grpo_only_s_al_regendata_6ep |none           |0.0|0.1|0.1|eagle_only|/scratch/yf3005/gto_data/sharegpt_qwen3_8b_regen.jsonl"
    # 3ep SFT stage-1 on regen data (submit with EPOCHS=3)
    "q8_eagle3_klv4_3ep_sft_regendata  |acceptance_length_v4 |0.1|0.1|0.1|          |/scratch/yf3005/gto_data/sharegpt_qwen3_8b_regen.jsonl"
    "q8_eagle3_kleal_3ep_sft_regendata |eal                  |0.1|0.1|0.1|          |/scratch/yf3005/gto_data/sharegpt_qwen3_8b_regen.jsonl"
    # 3+3 SFT→GRPO stage-2 — submit with EPOCHS=3 GRPO_COEF=0.1 + GRPO_MODE/REWARD + DRAFTPATH override
    "q8_eagle3_klv4_grpo_w_al_regendata  |acceptance_length_v4 |0.1|0.1|0.1|          |/scratch/yf3005/gto_data/sharegpt_qwen3_8b_regen.jsonl"
    "q8_eagle3_kleal_grpo_w_al_regendata |eal                  |0.1|0.1|0.1|          |/scratch/yf3005/gto_data/sharegpt_qwen3_8b_regen.jsonl"
    "q8_eagle3_kleal_grpo_w_eal_regendata|eal                  |0.1|0.1|0.1|          |/scratch/yf3005/gto_data/sharegpt_qwen3_8b_regen.jsonl"
    "q8_eagle3_kleal_grpo_s_al_regendata |eal                  |0.1|0.1|0.1|          |/scratch/yf3005/gto_data/sharegpt_qwen3_8b_regen.jsonl"

    # ─── New losses + GRPO on regen data ────────────────────────────
    # 6ep from base: aux + GRPO, regen data
    "q8_eagle3_altv_grpo_6ep_regendata   |al_tv                |1.0|0.1|0.1|aux_only  |/scratch/yf3005/gto_data/sharegpt_qwen3_8b_regen.jsonl"
    "q8_eagle3_alkl_grpo_6ep_regendata   |al_kl                |1.0|0.1|0.1|aux_only  |/scratch/yf3005/gto_data/sharegpt_qwen3_8b_regen.jsonl"
    "q8_eagle3_wkl_grpo_6ep_regendata    |wkl                  |1.0|0.1|0.1|aux_only  |/scratch/yf3005/gto_data/sharegpt_qwen3_8b_regen.jsonl"
    # 3+3 stage 2 from existing <loss>_only_regendata_6ep/state_2
    "q8_eagle3_altv_3plus3_grpo_regendata|al_tv                |1.0|0.1|0.1|aux_only  |/scratch/yf3005/gto_data/sharegpt_qwen3_8b_regen.jsonl"
    "q8_eagle3_alkl_3plus3_grpo_regendata|al_kl                |1.0|0.1|0.1|aux_only  |/scratch/yf3005/gto_data/sharegpt_qwen3_8b_regen.jsonl"
    "q8_eagle3_wkl_3plus3_grpo_regendata |wkl                  |1.0|0.1|0.1|aux_only  |/scratch/yf3005/gto_data/sharegpt_qwen3_8b_regen.jsonl"

    # ─── GRPO-only with new-loss reward (no SFT aux, no KL anchor) ──
    # total_loss = 0.1 · grpo_loss; reward formula matches the named loss.
    # Submit with: EPOCHS=6 GRPO_COEF=0.1 GRPO_MODE=window GRPO_REWARD=<reward>
    "q8_eagle3_grpo_only_w_altv_6ep    |none                 |0.0|0.1|0.1|aux_only"
    "q8_eagle3_grpo_only_w_alkl_6ep    |none                 |0.0|0.1|0.1|aux_only"
    "q8_eagle3_grpo_only_w_wkl_6ep     |none                 |0.0|0.1|0.1|aux_only"

    # ─── KL + aux + GRPO direct 6ep (analog of 3+3 SFT→GRPO but trained jointly) ──
    # baseline empty → default branch: total = eagle + 0.1·aux + 0.1·grpo
    # Submit with: EPOCHS=6 GRPO_COEF=0.1 + GRPO_MODE/REWARD overrides
    "q8_eagle3_klv4_w_al_6ep            |acceptance_length_v4 |0.1|0.1|0.1|"
    "q8_eagle3_kleal_w_al_6ep           |eal                  |0.1|0.1|0.1|"
    "q8_eagle3_kleal_w_eal_6ep          |eal                  |0.1|0.1|0.1|"
    "q8_eagle3_kleal_s_al_6ep           |eal                  |0.1|0.1|0.1|"

    # ─── KL + V8 (gap-vs-top2 instead of vs-max) ──────────────────
    "q8_eagle3_klv8_6ep                 |acceptance_length_v8 |0.1|0.1|0.1|"
    "q8_eagle3_klv8_3ep_sft             |acceptance_length_v8 |0.1|0.1|0.1|"
    "q8_eagle3_klv8_grpo_w_al           |acceptance_length_v8 |0.1|0.1|0.1|"
)

# ─── List mode ────────────────────────────────────────────────
if [ "$1" = "--list" ]; then
    echo "Available Qwen3 tasks:"
    for task in "${TASKS[@]}"; do
        IFS='|' read -r tag aux coef tmax tmin bl tp <<< "$task"
        tp_short=$(echo "$tp" | xargs)
        tp_short=${tp_short##*/}
        printf "  %-30s  aux=%-22s coef=%-5s T=(%s->%s)  %s  data=%s\n" \
            "$tag" "$aux" "$coef" "$tmin" "$tmax" "${bl:+baseline=$bl}" "${tp_short:-mixed_train_10K.jsonl}"
    done
    exit 0
fi

# ─── Submit function ──────────────────────────────────────────
submit() {
    local tag=$1 aux=$2 coef=$3 tmax=$4 tmin=$5 bl=$6 tp=$7

    EXTRA_EXPORT=""
    [ -n "$bl" ] && EXTRA_EXPORT=",BASELINE=$bl"

    # Trainpath resolution: task col 7 > env TRAINPATH > slurm-script default
    local TRAINPATH_USE="${tp:-${TRAINPATH:-}}"

    JOB_ID=$(sbatch \
        --job-name="q3_${tag}" \
        --output="logs/sdpo_qwen3/train_${tag}_%j.out" \
        --error="logs/sdpo_qwen3/train_${tag}_%j.err" \
        --time="${SLURM_TRAIN_TIME:-1-12:00:00}" \
        ${SLURM_CONSTRAINT:+--constraint="$SLURM_CONSTRAINT"} \
        ${SLURM_DEPENDENCY:+--dependency="$SLURM_DEPENDENCY"} \
        --export=ALL,TAG="$tag",AUX_LOSS="$aux",SIGMOID_COEF="$coef",T_MAX="$tmax",T_MIN="$tmin",NUM_EPOCHS="$NUM_EPOCHS",MAX_SAMPLES="$MAX_SAMPLES",TRAINPATH="$TRAINPATH_USE",DRAFTPATH="${DRAFTPATH:-AngelSlim/Qwen3-8B_eagle3}",GRPO_COEF="${GRPO_COEF:-0.0}",GRPO_MODE="${GRPO_MODE:-window}",GRPO_REWARD="${GRPO_REWARD:-hard}",GRPO_K_GROUPS="${GRPO_K_GROUPS:-8}",GRPO_M="${GRPO_M:-4}",GRPO_EPS="${GRPO_EPS:-0.2}",GRPO_SAMPLE_TEMP="${GRPO_SAMPLE_TEMP:-1.0}",CONDA_ENV="${CONDA_ENV:-longreason_eagle3}"${EXTRA_EXPORT} \
        "$SLURM_SCRIPT" | awk '{print $4}')

    echo "  Submitted $tag (data=${TRAINPATH_USE##*/}) -> $JOB_ID"
}

# ─── Determine which tasks to submit ─────────────────────────
SELECTED=("$@")

echo "Qwen3 sweep (epochs=$NUM_EPOCHS, samples=${MAX_SAMPLES:-full})"
echo "================================================================"

count=0
for task in "${TASKS[@]}"; do
    IFS='|' read -r tag aux coef tmax tmin bl tp <<< "$task"
    # Strip leading/trailing whitespace (TASKS lines pad fields for readability)
    tag=$(echo "$tag" | xargs)
    aux=$(echo "$aux" | xargs)
    coef=$(echo "$coef" | xargs)
    tmax=$(echo "$tmax" | xargs)
    tmin=$(echo "$tmin" | xargs)
    bl=$(echo "$bl" | xargs)
    tp=$(echo "${tp:-}" | xargs)

    if [ ${#SELECTED[@]} -gt 0 ]; then
        match=0
        for sel in "${SELECTED[@]}"; do
            if [ "$sel" = "$tag" ]; then match=1; break; fi
        done
        [ $match -eq 0 ] && continue
    fi

    submit "$tag" "$aux" "$coef" "$tmax" "$tmin" "$bl" "$tp"
    count=$((count + 1))
done

echo ""
echo "$count jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs in: logs/sdpo_qwen3/"
