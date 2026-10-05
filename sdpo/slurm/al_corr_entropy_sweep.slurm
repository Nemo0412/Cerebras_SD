#!/bin/bash

#SBATCH --job-name=al_ent_sweep
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100|h200"
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --mail-type=END
#SBATCH --mail-user=tx856@nyu.edu
#SBATCH --output=logs/al_ent/sweep_%j.out
#SBATCH --error=logs/al_ent/sweep_%j.err

cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
mkdir -p logs/al_ent

source ~/.bashrc
conda activate ${CONDA_ENV:-longreason}

echo "==== AL entropy/temperature sweep, SLURM_JOB_ID=$SLURM_JOB_ID ===="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
echo ""

declare -A CKPTS=(
    ["baseline"]="Qwen/Qwen3-0.6B"
    ["klv4_l2k_6ep"]="/scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/q8_q06_klv4_regen_l2k_6ep/state_5"
    ["alkl_only_6ep"]="/scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/q8_q06_alkl_only_regen_l2k_6ep/state_5"
    ["altv_only_6ep"]="/scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/q8_q06_altv_only_regen_l2k_6ep/state_5"
    ["kleal_l2k_grpo"]="/scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/q8_q06_kleal_l2k_grpo/state_2"
    ["eal_uni_l2k"]="/scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/q8_q06_eal_uni_regen_l2k/state_2"
)

CONFIGS=("greedy:greedy:1.0" "sampT05:sample:0.5" "sampT1:sample:1.0" "sampT2:sample:2.0")

for CKPT_TAG in "${!CKPTS[@]}"; do
    CKPT="${CKPTS[$CKPT_TAG]}"
    for CFG in "${CONFIGS[@]}"; do
        TAG=${CFG%%:*}
        REST=${CFG#*:}
        MODE=${REST%%:*}
        T=${REST##*:}
        OUT="diagnose_ent_${CKPT_TAG}_${TAG}.json"
        if [ -f "$OUT" ]; then
            echo "[SKIP] $OUT exists"
            continue
        fi
        echo ""
        echo "===== ${CKPT_TAG} / ${TAG} (mode=${MODE} T=${T}) ====="
        python sdpo/diagnose/al_corr_entropy_temperature.py \
            --base-model-path Qwen/Qwen3-8B \
            --draft-model-path "$CKPT" \
            --bench-name mt_bench \
            --num-samples 5 --gamma 7 --max-new-tokens 256 \
            --mode $MODE --temperature $T \
            --output "$OUT"
    done
done

echo ""
echo "==== ALL DONE ===="
ls -la diagnose_ent_*.json | tail -30
