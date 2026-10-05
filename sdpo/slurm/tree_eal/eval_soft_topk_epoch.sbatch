#!/bin/bash
#SBATCH --job-name=eal_ev_ep
#SBATCH --partition=h200_tandon
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ll5914/logs/tree_eal/soft_topk_eval_epoch_%j.out
#SBATCH --error=/scratch/ll5914/logs/tree_eal/soft_topk_eval_epoch_%j.err

set -euo pipefail
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD

cd /home/ll5914/Cerebras_SD/SDPO-Speculative-Decoding-Policy-Optimization

EPOCH="${EPOCH:?EPOCH is required}"
ROOT=/scratch/ll5914/loss_train_q8_q06/tree_eal_softk16
DRAFT="${ROOT}/hf_epoch_${EPOCH}"
OUTDIR="${ROOT}/eval/epoch_${EPOCH}"
mkdir -p "${OUTDIR}"

export HF_HOME=/scratch/ll5914/huggingface
export TOKENIZERS_PARALLELISM=false

echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} EPOCH=${EPOCH} DRAFT=${DRAFT}"
nvidia-smi -L
ls -la "${DRAFT}"

python sdpo/eval_small_lm_tree.py \
  --base-model-path /scratch/ll5914/models/qwen3-8b \
  --draft-model-path "${DRAFT}" \
  --bench-name mt_bench \
  --gamma 7 \
  --top-k 4,3,2,1,1,1,1 \
  --budget 128 \
  --max-new-tokens 256 \
  --num-samples 80 \
  --temperature 1.0 \
  --seed 0 \
  --draft-mode argmax \
  --verify-mode simple \
  --tag "tree_eal_softk16_epoch${EPOCH}" \
  --output-dir "${OUTDIR}"

export EPOCH OUTDIR ROOT
python - << 'PY'
import json, os
epoch = int(os.environ["EPOCH"])
outdir = os.environ["OUTDIR"]
root = os.environ["ROOT"]
path = os.path.join(outdir, f"tree_eal_softk16_epoch{epoch}.json")
with open(path) as f:
    payload = json.load(f)
tree = payload["results"]["mt_bench"]["tree"]
row = {
    "epoch": epoch,
    "accept_length": tree["mean_alpha"],
    "median_accept_length": tree.get("median_al"),
    "tokens_per_sec": tree.get("tokens_per_sec"),
    "eval_json": path,
}
log_path = os.path.join(root, "epoch_metrics.jsonl")
with open(log_path, "a") as f:
    f.write(json.dumps(row) + "\n")
print(f"epoch {epoch} accept_length={row['accept_length']:.6f}")
PY
