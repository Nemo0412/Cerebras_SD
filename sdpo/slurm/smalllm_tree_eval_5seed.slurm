#!/bin/bash

#SBATCH --job-name=slm_5seed_eval
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100"
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --mail-type=END
#SBATCH --mail-user=tx856@nyu.edu
#SBATCH --output=logs/smalllm_rollout/eval_5seed_%j.out
#SBATCH --error=logs/smalllm_rollout/eval_5seed_%j.err

cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
mkdir -p logs/smalllm_rollout

source ~/.bashrc
conda activate ${CONDA_ENV:-longreason}

BASEPATH=${BASEPATH:-Qwen/Qwen3-8B}
CKPT_ROOT=${CKPT_ROOT:-/scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b}
CKPT_PREFIX=${CKPT_PREFIX:?must set CKPT_PREFIX, e.g. q17_q06_kl_l2k_6ep}
TAG_PREFIX=${TAG_PREFIX:?must set TAG_PREFIX, e.g. q17_kl_l2k_6ep}
TAG_SUFFIX=${TAG_SUFFIX:-_lwosr2_top50_greedy}
STATE=${STATE:-state_5}
BENCH=${BENCH:-longwriter_top50,osr2_top50}
GAMMA=${GAMMA:-7}
TOP_K=${TOP_K:-4,3,2,1,1,1,1}
BUDGET=${BUDGET:-128}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
NUM_SAMPLES=${NUM_SAMPLES:-40}
TEMP=${TEMP:-0.0}
SEED=${SEED:-0}
DRAFT_MODE=${DRAFT_MODE:-argmax}
VERIFY_MODE=${VERIFY_MODE:-auto}
OUTPUT_DIR=${OUTPUT_DIR:-smalllm_tree_eval_results}
SEEDS=${SEEDS:-0 1 2 3 4}

echo "=============================="
echo "  Small LM Tree Eval 5-seed bundle"
echo "  Target:      $BASEPATH"
echo "  Ckpt prefix: $CKPT_PREFIX (state=$STATE)"
echo "  Tag prefix:  $TAG_PREFIX (suffix=$TAG_SUFFIX)"
echo "  Bench:       $BENCH"
echo "  Tree:        gamma=$GAMMA top_k=$TOP_K budget=$BUDGET"
echo "  Seeds:       $SEEDS"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

for S in $SEEDS ; do
    TAG="${TAG_PREFIX}_s${S}${TAG_SUFFIX}"
    DRAFT_CKPT="${CKPT_ROOT}/${CKPT_PREFIX}_s${S}/${STATE}"
    if [ ! -f "${DRAFT_CKPT}/pytorch_model.bin" ]; then
        echo "  [SKIP] s$S: ${DRAFT_CKPT}/pytorch_model.bin not found"
        continue
    fi
    echo ""
    echo "==== s${S}  →  TAG=${TAG} ===="
    python sdpo/eval_small_lm_tree.py \
        --base-model-path "$BASEPATH" \
        --draft-model-path "$DRAFT_CKPT" \
        --bench-name "$BENCH" \
        --gamma "$GAMMA" \
        --top-k "$TOP_K" \
        --budget "$BUDGET" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMP" \
        --seed "$SEED" \
        --draft-mode "$DRAFT_MODE" \
        --verify-mode "$VERIFY_MODE" \
        --tag "$TAG" \
        --output-dir "$OUTPUT_DIR" \
        --num-samples "$NUM_SAMPLES"
    echo "==== done s${S} ===="
done

echo ""
echo "All seeds done."
