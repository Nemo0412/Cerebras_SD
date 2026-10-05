#!/bin/bash

#SBATCH --job-name=slm_evalseeds
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100"
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --mail-type=END
#SBATCH --mail-user=tx856@nyu.edu
#SBATCH --output=logs/smalllm_rollout/eval_evalseeds_%j.out
#SBATCH --error=logs/smalllm_rollout/eval_evalseeds_%j.err

cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
mkdir -p logs/smalllm_rollout

source ~/.bashrc
conda activate ${CONDA_ENV:-longreason}

BASEPATH=${BASEPATH:-Qwen/Qwen3-8B}
CKPT_ROOT=${CKPT_ROOT:-/scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b}
STATE=${STATE:-state_5}
BENCH=${BENCH:-mt_bench,gsm8k,humaneval}
GAMMA=${GAMMA:-7}
TOP_K=${TOP_K:-4,3,2,1,1,1,1}
BUDGET=${BUDGET:-128}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
NUM_SAMPLES=${NUM_SAMPLES:-40}
DRAFT_MODE=${DRAFT_MODE:-sample}
VERIFY_MODE=${VERIFY_MODE:-ratio}
OUTPUT_DIR=${OUTPUT_DIR:-smalllm_tree_eval_results}

# Required: comma-separated list of <ckpt_dir_name>:<tag_prefix>
# CKPT_PAIRS="q8_q06_altv_only_regen_l2k_6ep:altv_only,q8_q06_alkl_only_regen_l2k_6ep:alkl_only,..."
CKPT_PAIRS=${CKPT_PAIRS:?must set CKPT_PAIRS as ckpt_dir:tag_prefix comma list}
EVAL_SEEDS=${EVAL_SEEDS:-0 1 2 3 4}
TEMPS=${TEMPS:-0.5 1.0}
TAG_SUFFIX_BASE=${TAG_SUFFIX_BASE:-smp_ratio}

echo "=============================="
echo "  Small LM Tree Eval - eval seed loop"
echo "  Target:      $BASEPATH"
echo "  ckpts:       $CKPT_PAIRS"
echo "  eval seeds:  $EVAL_SEEDS"
echo "  temps:       $TEMPS"
echo "  draft_mode:  $DRAFT_MODE  verify_mode:  $VERIFY_MODE"
echo "  bench:       $BENCH"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "=============================="

IFS=',' read -ra PAIRS <<< "$CKPT_PAIRS"
for PAIR in "${PAIRS[@]}" ; do
    CKPT_DIR=$(echo $PAIR | cut -d: -f1)
    TAG_PREFIX=$(echo $PAIR | cut -d: -f2)
    DRAFT_CKPT="${CKPT_ROOT}/${CKPT_DIR}/${STATE}"
    if [ ! -f "${DRAFT_CKPT}/pytorch_model.bin" ]; then
        echo "  [SKIP] ${CKPT_DIR}: ${DRAFT_CKPT}/pytorch_model.bin not found"
        continue
    fi
    for T in $TEMPS ; do
        # Convert temp to suffix: 0.5 -> t05, 1.0 -> t1
        if [ "$T" = "0.5" ]; then T_SUFFIX="t05"; fi
        if [ "$T" = "1.0" ]; then T_SUFFIX="t1"; fi
        if [ -z "$T_SUFFIX" ]; then T_SUFFIX="t${T//./}"; fi
        for ES in $EVAL_SEEDS ; do
            TAG="${TAG_PREFIX}_evalseed${ES}_samp_${T_SUFFIX}_${TAG_SUFFIX_BASE}"
            echo ""
            echo "==== ${CKPT_DIR}  T=${T}  eval_seed=${ES}  → TAG=${TAG} ===="
            python sdpo/eval_small_lm_tree.py \
                --base-model-path "$BASEPATH" \
                --draft-model-path "$DRAFT_CKPT" \
                --bench-name "$BENCH" \
                --gamma "$GAMMA" \
                --top-k "$TOP_K" \
                --budget "$BUDGET" \
                --max-new-tokens "$MAX_NEW_TOKENS" \
                --temperature "$T" \
                --seed "$ES" \
                --draft-mode "$DRAFT_MODE" \
                --verify-mode "$VERIFY_MODE" \
                --tag "$TAG" \
                --output-dir "$OUTPUT_DIR" \
                --num-samples "$NUM_SAMPLES"
        done
    done
done

echo ""
echo "All done."
