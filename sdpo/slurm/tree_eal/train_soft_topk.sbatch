#!/bin/bash
#SBATCH --job-name=eal_soft_tr
#SBATCH --partition=h200_tandon
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=4-00:00:00
#SBATCH --output=/scratch/ll5914/logs/tree_eal/soft_topk_train_%j.out
#SBATCH --error=/scratch/ll5914/logs/tree_eal/soft_topk_train_%j.err

set -euo pipefail
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD

cd /home/ll5914/Cerebras_SD/SDPO-Speculative-Decoding-Policy-Optimization
mkdir -p /scratch/ll5914/logs/tree_eal/wandb /scratch/ll5914/loss_train_q8_q06/tree_eal_softk16

export HF_HOME=/scratch/ll5914/huggingface
export WANDB_MODE=offline
export WANDB_DIR=/scratch/ll5914/logs/tree_eal/wandb
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FAKE_CUDA=/scratch/ll5914/logs/tree_eal/fake_cuda
mkdir -p "${FAKE_CUDA}/bin"
cat > "${FAKE_CUDA}/bin/nvcc" << 'EOF'
#!/bin/sh
echo "Cuda compilation tools, release 12.6, V12.6.77"
EOF
chmod +x "${FAKE_CUDA}/bin/nvcc"
STUB=/scratch/ll5914/conda_envs/SVD/lib/python3.10/site-packages/torch/utils/cpp_extension.py
if [ ! -f "${STUB}" ]; then
  cat > "${STUB}" << EOF
import os
CUDA_HOME = os.environ.get("CUDA_HOME", "${FAKE_CUDA}")
EOF
fi
export CUDA_HOME="${FAKE_CUDA}"
export TRITON_CACHE_DIR=/scratch/ll5914/logs/tree_eal/triton
mkdir -p "${TRITON_CACHE_DIR}"

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used --format=csv -l 30 \
  > /scratch/ll5914/logs/tree_eal/gpuutil_${SLURM_JOB_ID}.csv &
UTIL_PID=$!
trap 'kill ${UTIL_PID} 2>/dev/null || true' EXIT

echo "HOST=$(hostname) JOB=${SLURM_JOB_ID}"
nvidia-smi -L

LOCK=/scratch/ll5914/loss_train_q8_q06/tree_eal_softk16/.train_lock
if ! mkdir "${LOCK}" 2>/dev/null; then
  echo "Another trainer already holds ${LOCK}: $(cat ${LOCK}/owner 2>/dev/null || true)"
  exit 1
fi
echo "${SLURM_JOB_ID} $(hostname)" > "${LOCK}/owner"
trap 'kill ${UTIL_PID} 2>/dev/null || true; rm -rf "${LOCK}"' EXIT

deepspeed --num_gpus=2 sdpo/main_small_lm_tree_v2.py \
  --basepath /scratch/ll5914/models/qwen3-8b \
  --draftpath /scratch/ll5914/models/Qwen3-0.6B \
  --trainpath /scratch/yf3005/gto_data/sharegpt_qwen3_8b_regen.jsonl \
  --testpath /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_val_80.jsonl \
  --deepspeed_config sdpo/sdpo_config_qwen3_bf16_zero2_h200.json \
  --savedir /scratch/ll5914/loss_train_q8_q06/tree_eal_softk16 \
  --top_k_per_depth 4,3,2,1,1,1,1 \
  --tree_budget 128 \
  --eal_mode soft_topk \
  --path_topk 16 \
  --temperature 1.0 \
  --anchor_kl_coef 0 \
  --eal_aux_coef 0 \
  --max_len 2048 \
  --num_epochs 6
