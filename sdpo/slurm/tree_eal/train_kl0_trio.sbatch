#!/bin/bash
#SBATCH --job-name=kl0_trio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --partition=a100_tandon
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --output=/scratch/ll5914/logs/tree_eal/kl0_trio_%j.out
#SBATCH --error=/scratch/ll5914/logs/tree_eal/kl0_trio_%j.err
#
# Fair KL=0 trio on mix_sg_gsm_he_mt_4k5, 2 GPUs, 3 epochs (<8h for Tree EAL).
#   METHOD=tree_eal  -> sample_rsd Tree EAL, --anchor_kl_coef 0
#   METHOD=chain_eal -> al_tv Chain EAL, no KL anchor
#   METHOD=kl        -> pure position KL
set -euo pipefail
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-/scratch/ll5914/conda_envs/SVD}"

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
if [ ! -f "$REPO/sdpo/main_small_lm_tree_v2.py" ]; then
  REPO=/home/ll5914/Cerebras_SD/SDPO_dense_tv
fi
cd "$REPO"

METHOD=${METHOD:?set METHOD=tree_eal|chain_eal|kl}
TAG=${TAG:-kl0_${METHOD}_mix4k5}
TRAIN=${TRAIN:-/scratch/ll5914/loss_train_q8_q06/data/mix_sg_gsm_he_mt_4k5.jsonl}
TEST=${TEST:-/scratch/ll5914/repro_sdld/mixed_val_80.jsonl}
ROOT=${ROOT:-/scratch/ll5914/loss_train_q8_q06/${TAG}}
NUM_EPOCHS=${NUM_EPOCHS:-3}
LR=${LR:-1e-6}
SEED=${SEED:-42}
BASE=${BASE:-/scratch/ll5914/models/qwen3-8b}
DRAFT=${DRAFT:-/scratch/ll5914/models/Qwen3-0.6B}

if [ ! -f "$TRAIN" ]; then
  echo "MISSING train mix: $TRAIN (run prep_mix_sg_gsm_he_mt.sbatch first)"; exit 1
fi
NTRAIN=$(wc -l < "$TRAIN")
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} PART=${SLURM_JOB_PARTITION:-?} METHOD=$METHOD TAG=$TAG NTRAIN=$NTRAIN"
echo "REPO=$REPO HEAD=$(git rev-parse --short HEAD) start=$(date -Is)"
nvidia-smi -L
NGPU=$(nvidia-smi -L | wc -l)
if [ "$NGPU" -lt 2 ]; then echo "need 2 GPUs, got $NGPU"; exit 1; fi

export HF_HOME=${HF_HOME:-/scratch/ll5914/huggingface}
export WANDB_MODE=offline
export WANDB_DIR=${WANDB_DIR:-/scratch/ll5914/logs/tree_eal/wandb}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_HOME=${FAKE_CUDA:-/scratch/ll5914/logs/tree_eal/fake_cuda}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/scratch/ll5914/logs/tree_eal/triton}
export SDPO_DATALOADER_WORKERS=${SDPO_DATALOADER_WORKERS:-4}
mkdir -p "$TRITON_CACHE_DIR" "$WANDB_DIR" "$ROOT" /scratch/ll5914/logs/tree_eal

rm -rf "${ROOT}/state_"* "${ROOT}/hf_epoch_"* "${ROOT}/hf_best_al" "${ROOT}/hf_mid" \
       "${ROOT}/train_steps.jsonl" "${ROOT}/epoch_metrics.jsonl" "${ROOT}/mid" 2>/dev/null || true
mkdir -p "$ROOT"

PART=${SLURM_JOB_PARTITION:-a100}
DS="$ROOT/ds_train.json"
python3 - "$DS" "$METHOD" "$PART" "$NTRAIN" "$NUM_EPOCHS" <<'PY'
import json, sys, math, os
out, method, part, ntrain, nepoch = sys.argv[1:6]
ntrain, nepoch = int(ntrain), int(nepoch)
cands = [
    "sdpo/sdpo_config_qwen3_bf16_zero2_h200.json",
    "sdpo/sdpo_config_qwen3_bf16_zero2_h100.json",
    "sdpo/scripts/tree_eal/ds_zero2_a100_2gpu.json",
]
cfg = None
for p in cands:
    if os.path.isfile(p):
        cfg = json.load(open(p)); break
if cfg is None:
    raise SystemExit("no ds config template")
if method == "tree_eal":
    micro, accum = 1, 8
else:
    micro, accum = (2, 4) if part.startswith("a100") else (4, 2)
cfg["train_micro_batch_size_per_gpu"] = micro
cfg["gradient_accumulation_steps"] = accum
cfg["gradient_clipping"] = 5.0 if method == "tree_eal" else 1.0
eff = micro * accum * 2
opt_steps = max(100, math.ceil(ntrain / eff) * nepoch)
cfg["scheduler"]["params"]["warmup_max_lr"] = 1e-6
cfg["scheduler"]["params"]["warmup_num_steps"] = min(100, max(20, opt_steps // 20))
cfg["scheduler"]["params"]["total_num_steps"] = opt_steps + 10
json.dump(cfg, open(out, "w"), indent=2)
print(f"wrote {out} method={method} micro={micro} accum={accum} opt_steps≈{opt_steps}")
PY

PORT=$((29500 + SLURM_JOB_ID % 1000))
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used \
  --format=csv -l 30 > /scratch/ll5914/logs/tree_eal/gpuutil_${TAG}_${SLURM_JOB_ID}.csv &
UTIL_PID=$!
trap 'kill ${UTIL_PID} 2>/dev/null || true' EXIT

set +e
case "$METHOD" in
  tree_eal)
    deepspeed --num_gpus=2 --master_port=$PORT sdpo/main_small_lm_tree_v2.py \
      --basepath "$BASE" --draftpath "$DRAFT" \
      --trainpath "$TRAIN" --testpath "$TEST" \
      --deepspeed_config "$DS" --savedir "$ROOT" \
      --top_k_per_depth 4,3,2,1,1,1,1 --tree_budget 136 \
      --eal_mode sample_rsd --path_topk 4 --temperature 1.0 \
      --anchor_kl_coef 0.0 --eal_aux_coef 1.0 --eal_score_coef 0.0 \
      --eal_gradnorm_ratio 0.0 \
      --n_draws 1 --n_anchors 8 --anchor_chunk 8 \
      --lr "$LR" --max_len 2048 --num_epochs "$NUM_EPOCHS" \
      --save_every_steps 500 --seed "$SEED"
    rc=$?
    ;;
  chain_eal)
    deepspeed --num_gpus=2 --master_port=$PORT sdpo/main_small_lm.py \
      --basepath "$BASE" --draftpath "$DRAFT" \
      --trainpath "$TRAIN" --testpath "$TEST" \
      --deepspeed_config "$DS" --savedir "$ROOT" \
      --anchor none --aux_loss al_tv --aux_weight uniform \
      --sigmoid_coef 1.0 \
      --T_max 0.1 --T_min 0.1 \
      --prob_T_max 1.0 --prob_T_min 1.0 \
      --draft_T_max 1.0 --draft_T_min 1.0 \
      --gamma 7 --lr "$LR" --max_len 2048 --num_epochs "$NUM_EPOCHS" \
      --seed "$SEED"
    rc=$?
    ;;
  kl)
    deepspeed --num_gpus=2 --master_port=$PORT sdpo/main_small_lm.py \
      --basepath "$BASE" --draftpath "$DRAFT" \
      --trainpath "$TRAIN" --testpath "$TEST" \
      --deepspeed_config "$DS" --savedir "$ROOT" \
      --anchor kl --aux_loss none \
      --anchor_weight none --aux_weight pow08 \
      --sigmoid_coef 0.0 \
      --T_max 0.1 --T_min 0.1 \
      --prob_T_max 1.0 --prob_T_min 1.0 \
      --draft_T_max 1.0 --draft_T_min 1.0 \
      --gamma 7 --lr "$LR" --max_len 2048 --num_epochs "$NUM_EPOCHS" \
      --seed "$SEED"
    rc=$?
    ;;
  *)
    echo "unknown METHOD=$METHOD"; exit 2
    ;;
esac
set -e
echo "METHOD=$METHOD exit=$rc end=$(date -Is)"
exit $rc
