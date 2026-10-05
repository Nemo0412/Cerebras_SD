#!/bin/bash

#SBATCH --job-name=gemma4_train_eval
#SBATCH --time=1-23:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=192G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --constraint="h200"
#SBATCH --requeue
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=tx856@nyu.edu
#SBATCH --output=logs/sdpo_gemma4/train_%j.out
#SBATCH --error=logs/sdpo_gemma4/train_%j.err

cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
mkdir -p logs/sdpo_gemma4
# Fix ModuleNotFoundError: sdpo eval script uses `from sdpo.spec_decode_gemma4`
# which requires repo root on Python path. cd alone does not add cwd to sys.path
# when python is invoked with a relative script path.
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD"

source ~/.bashrc
conda activate ${CONDA_ENV:-longreason}

# Training pipeline runs on longreason env (deepspeed installed); gemma4 env
# is inference-only (vLLM 0.23.0 — no deepspeed). longreason transformers
# 5.5.3 supports Gemma 4.
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/scratch/tx856/.cache/torch_extensions}
mkdir -p "$TORCH_EXTENSIONS_DIR"

export TMPDIR=/tmp/sdpo_${SLURM_JOB_ID:-$$}
mkdir -p "$TMPDIR"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export HF_HOME="${HF_HOME:-/scratch/tx856/.huggingface}"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
export TORCH_HOME="$TMPDIR/torch"

# NCCL resilience: jobs 11975456/57/58 all crashed with 600 s ALLREDUCE timeout
# on different nodes (gh117, gh118); canary 11985764 hung at step 72 with both
# GPUs pinned at 100% util but no progress. The IB transport on these nodes is
# flaky for long collectives. Disable IB & P2P to fall back to TCP/SHM (slower
# but reliable). NCCL_TIMEOUT alone didn't take effect on the canary.
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_DEBUG=WARN

SIGMOID_COEF=${SIGMOID_COEF:-0.1}
T_MAX=${T_MAX:-0.1}
T_MIN=${T_MIN:-0.1}
PROB_T_MAX=${PROB_T_MAX:-1.0}
PROB_T_MIN=${PROB_T_MIN:-1.0}
DRAFT_T_MAX=${DRAFT_T_MAX:-1.0}
DRAFT_T_MIN=${DRAFT_T_MIN:-1.0}
EAL_WEIGHT=${EAL_WEIGHT:-0.5}
NUM_EPOCHS=${NUM_EPOCHS:-6}
BASEPATH=${BASEPATH:-google/gemma-4-31B-it}
DRAFTPATH=${DRAFTPATH:-google/gemma-4-E2B-it}
ANCHOR=${ANCHOR:-kl}
AUX_LOSS=${AUX_LOSS:-acceptance_length_v4}
ANCHOR_WEIGHT=${ANCHOR_WEIGHT:-none}
AUX_WEIGHT=${AUX_WEIGHT:-pow08}
GRPO_COEF=${GRPO_COEF:-0.0}
GRPO_K_GROUPS=${GRPO_K_GROUPS:-8}
GRPO_M=${GRPO_M:-4}
GRPO_EPS=${GRPO_EPS:-0.2}
GRPO_MODE=${GRPO_MODE:-window}
GRPO_SAMPLE_TEMP=${GRPO_SAMPLE_TEMP:-1.0}
GRPO_REWARD=${GRPO_REWARD:-hard}
GRPO_REWARD_ETA=${GRPO_REWARD_ETA:-1.0}
GRPO_REWARD_EPS=${GRPO_REWARD_EPS:-2.0}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-sdpo/sdpo_config_gemma4_bf16_zero2_offload.json}
SAVEROOT=${SAVEROOT:-/scratch/tx856/spec_reason/scratch/loss_train_gemma4}
MAX_LEN=${MAX_LEN:-2048}
SEED=${SEED:-42}
# SAVE_EPOCHS: 0-indexed epoch IDs. sbatch --export's comma-bug splits "2,5"
# into "2" + stray "5", so we resolve SAVE_EPOCHS INSIDE the script from
# NUM_EPOCHS instead of passing it through.
#   6ep  → "2,5"  (mid + last  — last useful for eval, mid for GRPO seed)
#   3ep  → "2"    (last only)
#   else → last epoch only
if [ -z "${SAVE_EPOCHS+x}" ]; then
    if [ "$NUM_EPOCHS" -ge 6 ]; then
        SAVE_EPOCHS="2,$((NUM_EPOCHS - 1))"
    else
        SAVE_EPOCHS=$((NUM_EPOCHS - 1))
    fi
fi

SAVE_EPOCHS_FLAG=""
if [ -n "$SAVE_EPOCHS" ]; then
    SAVE_EPOCHS_FLAG="--save_epochs $SAVE_EPOCHS"
fi

SAMPLE_FLAG=""
if [ -n "$MAX_SAMPLES" ]; then
    SAMPLE_FLAG="--max_train_samples $MAX_SAMPLES"
fi

BASELINE_FLAG=""
if [ -n "$BASELINE" ]; then
    BASELINE_FLAG="--baseline $BASELINE"
fi

LR_FLAG=""
if [ -n "$LR" ]; then
    LR_FLAG="--lr $LR"
fi

echo "=============================="
echo "  Gemma 4 Train: $TAG"
echo "  target=$BASEPATH draft=$DRAFTPATH"
echo "  anchor=$ANCHOR aux_loss=$AUX_LOSS"
echo "  coef=$SIGMOID_COEF T=($T_MIN→$T_MAX) lr=${LR:-default}"
echo "  grpo_coef=$GRPO_COEF mode=$GRPO_MODE reward=$GRPO_REWARD"
echo "  gamma=7 max_len=$MAX_LEN epochs=$NUM_EPOCHS samples=${MAX_SAMPLES:-full}"
echo "  trainpath=${TRAINPATH:-default}"
echo "  saveroot=$SAVEROOT"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

export MASTER_PORT=$((29500 + RANDOM % 1000))

deepspeed --master_port $MASTER_PORT sdpo/main_small_lm.py \
    --basepath "$BASEPATH" \
    --draftpath "$DRAFTPATH" \
    --trainpath "${TRAINPATH:?must set TRAINPATH}" \
    --testpath "${TESTPATH:-/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_val_80.jsonl}" \
    --deepspeed_config "$DEEPSPEED_CONFIG" \
    --savedir "${SAVEROOT}/${TAG}" \
    --sigmoid_coef "$SIGMOID_COEF" \
    --T_max "$T_MAX" --T_min "$T_MIN" \
    --prob_T_max "$PROB_T_MAX" --prob_T_min "$PROB_T_MIN" \
    --draft_T_max "$DRAFT_T_MAX" --draft_T_min "$DRAFT_T_MIN" \
    --eal_weight "$EAL_WEIGHT" \
    --anchor "$ANCHOR" \
    --aux_loss "$AUX_LOSS" \
    --anchor_weight "$ANCHOR_WEIGHT" \
    --aux_weight "$AUX_WEIGHT" \
    --grpo_coef "$GRPO_COEF" \
    --grpo_k_groups "$GRPO_K_GROUPS" \
    --grpo_m "$GRPO_M" \
    --grpo_eps "$GRPO_EPS" \
    --grpo_mode "$GRPO_MODE" \
    --grpo_sample_temp "$GRPO_SAMPLE_TEMP" \
    --grpo_reward "$GRPO_REWARD" \
    --grpo_reward_eta "$GRPO_REWARD_ETA" \
    --grpo_reward_eps "$GRPO_REWARD_EPS" \
    --gamma 7 \
    --max_len "$MAX_LEN" \
    --num_epochs "$NUM_EPOCHS" \
    --seed "$SEED" \
    $SAVE_EPOCHS_FLAG \
    $BASELINE_FLAG \
    $LR_FLAG \
    $SAMPLE_FLAG
TRAIN_RC=$?

echo ""
echo "=============================="
echo "  Train done rc=$TRAIN_RC"
echo "=============================="

# ────── EVAL section (greedy + sample_ratio T=1) ──────
# Skip eval if train failed.
if [ $TRAIN_RC -ne 0 ]; then
    echo "[EVAL] skipped: train failed (rc=$TRAIN_RC)"
    rm -rf "$TMPDIR" 2>/dev/null || true
    exit $TRAIN_RC
fi

CKPT_STATE="state_$((NUM_EPOCHS - 1))"
DRAFT_CKPT="${SAVEROOT}/${TAG}/${CKPT_STATE}"

if [ ! -f "$DRAFT_CKPT/pytorch_model.bin" ] && [ ! -f "$DRAFT_CKPT/model.safetensors" ]; then
    echo "[EVAL] ckpt not found at $DRAFT_CKPT — listing $SAVEROOT/$TAG/:"
    ls "$SAVEROOT/$TAG/" 2>&1
    # Try newest state_*
    DRAFT_CKPT=$(ls -d "$SAVEROOT/$TAG"/state_* 2>/dev/null | sort -V | tail -1)
    echo "[EVAL] fallback to newest: $DRAFT_CKPT"
fi

EVAL_BENCHES="mt_bench,gsm8k,humaneval,longwriter_top50,osr2_top50"
EVAL_OUTDIR=smalllm_tree_eval_results
mkdir -p "$EVAL_OUTDIR"

echo ""
echo "=============================="
echo "  EVAL: $TAG (ckpt=$CKPT_STATE)"
echo "  target=$BASEPATH draft=$DRAFT_CKPT"
echo "  benches=$EVAL_BENCHES"
echo "=============================="

# Greedy: target on cuda:0, draft on cuda:1 (loader already does device_map).
echo ""
echo "==== GREEDY (T=0, argmax/auto, γ=7 b=128) ===="
python sdpo/eval_gemma4_chain.py \
    --target-model "$BASEPATH" \
    --draft-model "$DRAFT_CKPT" \
    --bench-name "$EVAL_BENCHES" \
    --mode tree \
    --gamma 7 --top-k "4,3,2,1,1,1,1" --budget 128 \
    --max-new-tokens 256 \
    --temperature 0.0 \
    --draft-mode argmax \
    --verify-mode auto \
    --tag "${TAG}_greedy" \
    --output-dir "$EVAL_OUTDIR" \
    --num-samples 40
EVAL_GREEDY_RC=$?
echo "  greedy rc=$EVAL_GREEDY_RC"

# Sample ratio T=0.5 (tag `_smp_ratio` — matches standalone gemma4_eval.slurm)
echo ""
echo "==== SAMPLE_RATIO (T=0.5, sample/ratio, γ=7 b=128) ===="
python sdpo/eval_gemma4_chain.py \
    --target-model "$BASEPATH" \
    --draft-model "$DRAFT_CKPT" \
    --bench-name "$EVAL_BENCHES" \
    --mode tree \
    --gamma 7 --top-k "4,3,2,1,1,1,1" --budget 128 \
    --max-new-tokens 256 \
    --temperature 0.5 \
    --draft-mode sample \
    --verify-mode ratio \
    --tag "${TAG}_smp_ratio" \
    --output-dir "$EVAL_OUTDIR" \
    --num-samples 40
EVAL_SAMP_T05_RC=$?
echo "  smp_ratio (T=0.5) rc=$EVAL_SAMP_T05_RC"

# Sample ratio T=1 (tag `_smp_t1_ratio` — matches standalone gemma4_eval.slurm)
echo ""
echo "==== SAMPLE_RATIO (T=1, sample/ratio, γ=7 b=128) ===="
python sdpo/eval_gemma4_chain.py \
    --target-model "$BASEPATH" \
    --draft-model "$DRAFT_CKPT" \
    --bench-name "$EVAL_BENCHES" \
    --mode tree \
    --gamma 7 --top-k "4,3,2,1,1,1,1" --budget 128 \
    --max-new-tokens 256 \
    --temperature 1.0 \
    --draft-mode sample \
    --verify-mode ratio \
    --tag "${TAG}_smp_t1_ratio" \
    --output-dir "$EVAL_OUTDIR" \
    --num-samples 40
EVAL_SAMP_T1_RC=$?
echo "  smp_t1_ratio rc=$EVAL_SAMP_T1_RC"

echo ""
echo "==== ALL DONE  train=$TRAIN_RC greedy=$EVAL_GREEDY_RC smp_t05=$EVAL_SAMP_T05_RC smp_t1=$EVAL_SAMP_T1_RC ===="

rm -rf "$TMPDIR" 2>/dev/null || true
