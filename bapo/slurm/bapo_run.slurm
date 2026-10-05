#!/bin/bash

#SBATCH --job-name=bapo
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --constraint="a100|h100|h200"
#SBATCH --requeue
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=xt2251@nyu.edu
#SBATCH --output=logs/bapo/slurm_%j.out
#SBATCH --error=logs/bapo/slurm_%j.err

cd /scratch/xt2251/SDPO-Speculative-Decoding-Policy-Optimization

source /scratch/xt2251/miniconda3/etc/profile.d/conda.sh
conda activate spec

# These variables are set by the launcher scripts
# TAG, W_FAR, W_NEAR, W_FAIL, W_POST, KL_COEF, SAMPLING, PROD_COEF, PROD_TAU, PHASE, EPOCHS, MAX_SAMPLES

# Build optional flags
SAMPLE_FLAG=""
if [ -n "$MAX_SAMPLES" ]; then
    SAMPLE_FLAG="--max_train_samples $MAX_SAMPLES"
fi
PROD_COEF=${PROD_COEF:-0.0}
PROD_TAU=${PROD_TAU:-1.0}

echo "=============================="
echo "  BAPO run: $TAG"
echo "  w=($W_FAR,$W_NEAR,$W_FAIL,$W_POST) kl=$KL_COEF $SAMPLING"
echo "  prod_coef=$PROD_COEF prod_tau=$PROD_TAU"
echo "  phase=$PHASE epochs=$EPOCHS samples=${MAX_SAMPLES:-full}"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

if [ "$PHASE" = "distill" ]; then
    deepspeed bapo/main.py \
        --basepath /scratch/xt2251/models/Llama-3.1-8B-Instruct \
        --draftpath /scratch/xt2251/models/EAGLE3-LLaMA3.1-Instruct-8B \
        --trainpath sdpo/data/mixed_train.jsonl \
        --testpath sdpo/data/mixed_val.jsonl \
        --deepspeed_config bapo/bapo_config.json \
        --config_path sdpo/config.json \
        --savedir "bapo_sweep/${TAG}" \
        --mode finetune \
        --distill_epochs "$EPOCHS" --bapo_epochs 0 \
        --gamma 7 \
        $SAMPLE_FLAG
else
    deepspeed bapo/main.py \
        --basepath /scratch/xt2251/models/Llama-3.1-8B-Instruct \
        --draftpath /scratch/xt2251/models/EAGLE3-LLaMA3.1-Instruct-8B \
        --trainpath sdpo/data/mixed_train.jsonl \
        --testpath sdpo/data/mixed_val.jsonl \
        --deepspeed_config bapo/bapo_config.json \
        --config_path sdpo/config.json \
        --savedir "bapo_sweep/${TAG}" \
        --mode finetune \
        --bapo_epochs "$EPOCHS" \
        --w_far "$W_FAR" --w_near "$W_NEAR" --w_fail "$W_FAIL" --w_post "$W_POST" \
        --kl_coef "$KL_COEF" \
        --sampling "$SAMPLING" \
        --prod_coef "$PROD_COEF" --prod_tau "$PROD_TAU" \
        --gamma 7 \
        $SAMPLE_FLAG
fi
