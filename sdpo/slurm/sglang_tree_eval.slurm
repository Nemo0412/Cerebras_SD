#!/bin/bash
# Serving-engine eval of a small-LM draft checkpoint: SGLang STANDALONE
# speculative decoding (tree / chain) and the target alone, one request at a time.
# See sdpo/eval_sglang_tree.py.
#
#   sbatch --export=ALL,BASE=<target dir>,DRAFT_CKPT=<draft hf dir>,TAG=<name>,\
#          SGLANG_ENV=<venv with sglang> sdpo/slurm/sglang_tree_eval.slurm
#SBATCH --job-name=sgl_tree_eval
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/sglang_tree_eval_%j.log
set -uo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
BASE=${BASE:?set BASE to the target model path}
DRAFT_CKPT=${DRAFT_CKPT:?set DRAFT_CKPT to the draft model path}
TAG=${TAG:?set TAG}
SGLANG_ENV=${SGLANG_ENV:?set SGLANG_ENV to a venv/conda prefix with sglang}
OUT=${OUT:-smalllm_tree_eval_results}

export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# FlashInfer JIT-builds its ops with the system GCC, so they need that GCC's
# GLIBCXX; a conda env's older bundled libstdc++ would be loaded first and the
# server dies at CUDA graph capture. Preload the system one (a superset).
SYS_LIBSTDCXX=${SYS_LIBSTDCXX:-/usr/lib64/libstdc++.so.6}
if [ -e "$SYS_LIBSTDCXX" ]; then
  export LD_PRELOAD="$SYS_LIBSTDCXX${LD_PRELOAD:+:$LD_PRELOAD}"
fi
cd "$REPO"
echo "node=$(hostname) job=${SLURM_JOB_ID:-} start=$(date -Is) draft=$DRAFT_CKPT head=$(git rev-parse --short HEAD) LD_PRELOAD=${LD_PRELOAD:-}"

"$SGLANG_ENV/bin/python" sdpo/eval_sglang_tree.py \
  --base-model-path "$BASE" --draft-model-path "$DRAFT_CKPT" \
  --bench-name "${EVAL_BENCH:-mt_bench,gsm8k,humaneval}" \
  --num-samples "${EVAL_NUM_SAMPLES:-80}" --max-new-tokens "${MAX_NEW_TOKENS:-256}" \
  --temperature "${TEMP:-0.5}" \
  --num-steps "${NUM_STEPS:-7}" --eagle-topk "${EAGLE_TOPK:-4}" \
  --num-draft-tokens "${NUM_DRAFT_TOKENS:-29}" --chain-steps "${CHAIN_STEPS:-7}" \
  --modes "${MODES:-tree,chain,vanilla}" \
  --spec-algorithm "${SPEC_ALGORITHM:-STANDALONE}" \
  --extra-server-args "${EXTRA_SERVER_ARGS:-}" \
  --tag "${TAG}_sglang" --output-dir "$OUT"
rc=$?
echo "eval exit=$rc end=$(date -Is)"
exit $rc
