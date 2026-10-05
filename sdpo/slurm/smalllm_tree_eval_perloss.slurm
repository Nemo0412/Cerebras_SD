#!/bin/bash

#SBATCH --job-name=slm_perloss_eval
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100|h200"
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --mail-type=END
#SBATCH --mail-user=tx856@nyu.edu
#SBATCH --output=logs/smalllm_rollout/eval_perloss_%j.out
#SBATCH --error=logs/smalllm_rollout/eval_perloss_%j.err

cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
mkdir -p logs/smalllm_rollout

source ~/.bashrc
conda activate ${CONDA_ENV:-longreason}

BASEPATH=${BASEPATH:-Qwen/Qwen3-8B}
CKPT_ROOT=${CKPT_ROOT:-/scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b}
CKPT_PREFIX=${CKPT_PREFIX:?must set CKPT_PREFIX}
TAG_PREFIX=${TAG_PREFIX:?must set TAG_PREFIX}
TAG_BASE=${TAG_BASE:-_ep3}
STATE=${STATE:-state_5}
BENCH=${BENCH:-mt_bench,gsm8k,humaneval}
GAMMA=${GAMMA:-7}
TOP_K=${TOP_K:-4,3,2,1,1,1,1}
BUDGET=${BUDGET:-128}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
NUM_SAMPLES=${NUM_SAMPLES:-40}
EVAL_SEED=${EVAL_SEED:-0}
OUTPUT_DIR=${OUTPUT_DIR:-smalllm_tree_eval_results}
SEEDS=${SEEDS:-0 1 2}

# MODES: comma-separated "<name>:<draft_mode>:<verify_mode>:<temp>"
# default: greedy + sample_ratio T=0.5 + sample_ratio T=1.0
MODES=${MODES:-greedy:argmax:auto:0.0,samp_t05_smp_ratio:sample:ratio:0.5,samp_t1_smp_ratio:sample:ratio:1.0}

echo "=============================="
echo "  Small LM Tree Eval - per-loss bundle"
echo "  Target:      $BASEPATH"
echo "  CKPT_PREFIX: $CKPT_PREFIX (state=$STATE)"
echo "  TAG_PREFIX:  $TAG_PREFIX  TAG_BASE=$TAG_BASE"
echo "  Bench:       $BENCH"
echo "  Tree:        gamma=$GAMMA top_k=$TOP_K budget=$BUDGET"
echo "  SEEDS:       $SEEDS"
echo "  MODES:       $MODES"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

IFS=',' read -ra MODE_LIST <<< "$MODES"

for S in $SEEDS ; do
    DRAFT_CKPT="${CKPT_ROOT}/${CKPT_PREFIX}_s${S}/${STATE}"
    if [ ! -f "${DRAFT_CKPT}/pytorch_model.bin" ]; then
        echo "  [SKIP] s$S: ${DRAFT_CKPT}/pytorch_model.bin not found"
        continue
    fi
    for MODE_SPEC in "${MODE_LIST[@]}" ; do
        IFS=':' read -ra PARTS <<< "$MODE_SPEC"
        NAME=${PARTS[0]} DRAFT_MODE=${PARTS[1]} VERIFY_MODE=${PARTS[2]} TEMP=${PARTS[3]}
        TAG="${TAG_PREFIX}_s${S}${TAG_BASE}_${NAME}"
        echo ""
        echo "==== s${S}  ${NAME} (draft=${DRAFT_MODE} verify=${VERIFY_MODE} T=${TEMP})  → TAG=${TAG} ===="
        python sdpo/eval_small_lm_tree.py \
            --base-model-path "$BASEPATH" \
            --draft-model-path "$DRAFT_CKPT" \
            --bench-name "$BENCH" \
            --gamma "$GAMMA" \
            --top-k "$TOP_K" \
            --budget "$BUDGET" \
            --max-new-tokens "$MAX_NEW_TOKENS" \
            --temperature "$TEMP" \
            --seed "$EVAL_SEED" \
            --draft-mode "$DRAFT_MODE" \
            --verify-mode "$VERIFY_MODE" \
            --tag "$TAG" \
            --output-dir "$OUTPUT_DIR" \
            --num-samples "$NUM_SAMPLES"
        echo "==== done s${S} ${NAME} ===="
    done
done

echo ""
echo "All seeds × modes done."
