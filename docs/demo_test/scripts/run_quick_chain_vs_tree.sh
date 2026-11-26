#!/bin/bash
# Quick A/B: Chain EAL vs Sample Tree EAL on small data, tree ≤ ~1h.
# See docs/demo_test/README.md
# train=2048 alpaca-regen subset, val=16 mixed subset.
# GPU0=chain, GPU1,2,3=tree (parallel).
#
# Override with env: ROOT, REPO, PY, BASE, DRAFT, TRAIN, VAL
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
DEMO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
REPO=${REPO:-$(cd "$DEMO_DIR/../.." && pwd)}
ROOT=${ROOT:-/mnt/ssd_ext/lls/cerebras_sd}
PY=${PY:-/mnt/ssd_ext/xth/conda_envs/longreason}
DS=$PY/bin/deepspeed
PYTHON=$PY/bin/python
LOG=$ROOT/logs
TRAIN=${TRAIN:-$DEMO_DIR/data/quick_train_2048.jsonl}
VAL=${VAL:-$DEMO_DIR/data/quick_val_16.jsonl}
BASE=${BASE:-$ROOT/models/qwen3-8b}
DRAFT=${DRAFT:-$ROOT/models/Qwen3-0.6B}
DS_CHAIN=$REPO/sdpo/scripts/tree_eal/ds_zero2_a100_1gpu.json
DS_TREE=$REPO/sdpo/scripts/tree_eal/ds_zero2_a6000_3gpu.json
SAVE_CHAIN=$ROOT/checkpoints/quick_sample_chain_eal
SAVE_TREE=$ROOT/checkpoints/quick_sample_tree_eal
PROBE=$SCRIPT_DIR/probe_chain_tree_paths.py
PORT_CHAIN=${PORT_CHAIN:-29731}
PORT_TREE=${PORT_TREE:-29732}

export HF_HOME=$ROOT/hf
export WANDB_MODE=offline
export WANDB_DIR=$LOG/wandb
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$LOG" "$SAVE_CHAIN" "$SAVE_TREE"
cd "$REPO"
echo "[quick] start $(date -Is) train=$TRAIN val=$VAL" | tee "$LOG/quick_chain_vs_tree.log"

rm -rf "$SAVE_CHAIN" "$SAVE_TREE"
mkdir -p "$SAVE_CHAIN" "$SAVE_TREE"

# --- Chain: 2048 × 2ep (~20min) + online tree-path probe for curves ---
"$DS" --master_port "$PORT_CHAIN" --include "localhost:0" \
  sdpo/main_small_lm.py \
  --basepath "$BASE" --draftpath "$DRAFT" \
  --trainpath "$TRAIN" --testpath "$VAL" \
  --deepspeed_config "$DS_CHAIN" --savedir "$SAVE_CHAIN" \
  --lr 1e-6 --max_len 1024 --num_epochs 2 \
  --max_train_samples 2048 --seed 0 --gamma 7 \
  --T_max 0.1 --T_min 0.1 \
  --anchor none --aux_loss al_tv --sigmoid_coef 1.0 \
  --anchor_weight none --aux_weight uniform \
  --prob_T_max 1.0 --prob_T_min 1.0 \
  --top_k_per_depth 4,3,2,1,1,1,1 \
  --tree_path_probe_every 50 --tree_path_probe_topk 4 \
  --tree_path_probe_n_anchors 1 \
  > "$LOG/quick_sample_chain_eal_train.log" 2>&1 &
CHAIN_PID=$!
echo "[quick] chain pid=$CHAIN_PID" | tee -a "$LOG/quick_chain_vs_tree.log"

# --- Tree sample_rsd: 2048 × 1ep on 3gpu (~40min, <1h) ---
"$DS" --master_port "$PORT_TREE" --include "localhost:1,2,3" \
  sdpo/main_small_lm_tree_v2.py \
  --basepath "$BASE" --draftpath "$DRAFT" \
  --trainpath "$TRAIN" --testpath "$VAL" \
  --deepspeed_config "$DS_TREE" --savedir "$SAVE_TREE" \
  --top_k_per_depth 4,3,2,1,1,1,1 --tree_budget 0 \
  --temperature 1.0 --path_topk 4 --lr 1e-6 \
  --max_len 1024 --max_train_samples 2048 \
  --num_epochs 1 --seed 0 \
  --eal_mode sample_rsd \
  --n_anchors 4 --anchor_chunk 2 \
  --anchor_kl_coef 0.1 --eal_aux_coef 1.0 \
  --save_every_steps 0 \
  > "$LOG/quick_sample_tree_eal_train.log" 2>&1 &
TREE_PID=$!
echo "[quick] tree pid=$TREE_PID (3gpu, 2048×1ep)" | tee -a "$LOG/quick_chain_vs_tree.log"

wait $CHAIN_PID; echo "[quick] chain exit=$? $(date -Is)" | tee -a "$LOG/quick_chain_vs_tree.log"
wait $TREE_PID; echo "[quick] tree exit=$? $(date -Is)" | tee -a "$LOG/quick_chain_vs_tree.log"

# --- Strip chain draft_model. prefix ---
CHAIN_RAW=""
for d in "$SAVE_CHAIN"/state_[0-9]*; do
  case "$d" in *_stripped) continue ;; esac
  [ -d "$d" ] && [ -f "$d/config.json" ] && CHAIN_RAW=$d
done
CHAIN_CKPT=""
if [ -n "$CHAIN_RAW" ]; then
  CHAIN_CKPT="${CHAIN_RAW}_stripped"
  echo "[quick] stripping $CHAIN_RAW -> $CHAIN_CKPT" | tee -a "$LOG/quick_chain_vs_tree.log"
  "$PYTHON" - <<PY
import os, shutil, torch
from transformers import AutoConfig, AutoModelForCausalLM
ck, out, draft = "$CHAIN_RAW", "$CHAIN_CKPT", "$DRAFT"
os.makedirs(out, exist_ok=True)
sd = torch.load(os.path.join(ck, "pytorch_model.bin"), map_location="cpu")
out_sd = {}
for k, v in sd.items():
    nk = k
    for p in ("module.", "draft_model.", "draft.base.", "draft."):
        if nk.startswith(p):
            nk = nk[len(p):]
            break
    out_sd[nk] = v
cfg = AutoConfig.from_pretrained(draft, trust_remote_code=True)
m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
missing, unexpected = m.load_state_dict(out_sd, strict=False)
print("missing", len(missing), "unexpected", len(unexpected))
m.save_pretrained(out)
cfg.save_pretrained(out)
for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
             "generation_config.json"):
    src = os.path.join(draft, name)
    dst = os.path.join(out, name)
    if os.path.isfile(src) and not os.path.isfile(dst):
        shutil.copy2(src, dst)
print(out)
PY
fi

TREE_CKPT=""
for d in "$SAVE_TREE"/hf_epoch_* "$SAVE_TREE"/hf_mid; do
  [ -d "$d" ] && [ -f "$d/config.json" ] && TREE_CKPT=$d
done
# tree usually saves hf at epoch end as hf_epoch_0 etc — also try listing
if [ -z "$TREE_CKPT" ]; then
  TREE_CKPT=$(find "$SAVE_TREE" -maxdepth 1 -type d -name 'hf_*' | sort | tail -1)
fi
echo "[quick] chain_ckpt=$CHAIN_CKPT tree_ckpt=$TREE_CKPT" | tee -a "$LOG/quick_chain_vs_tree.log"

# --- Offline tree-path probe on val (fair compare) ---
if [ -n "$CHAIN_CKPT" ]; then
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$PROBE" \
    --basepath "$BASE" --draftpath "$CHAIN_CKPT" \
    --datapath "$VAL" --savedir "$SAVE_CHAIN" \
    --tag "quick_chain_final" --step 4096 --max_samples 16 \
    --path_topk 4 --n_anchors 4 \
    > "$LOG/quick_probe_chain.log" 2>&1
  echo "[quick] chain probe exit=$?" | tee -a "$LOG/quick_chain_vs_tree.log"
fi
if [ -n "$TREE_CKPT" ]; then
  CUDA_VISIBLE_DEVICES=1 "$PYTHON" "$PROBE" \
    --basepath "$BASE" --draftpath "$TREE_CKPT" \
    --datapath "$VAL" --savedir "$SAVE_TREE" \
    --tag "quick_tree_final" --step 2048 --max_samples 16 \
    --path_topk 4 --n_anchors 4 \
    > "$LOG/quick_probe_tree.log" 2>&1
  echo "[quick] tree probe exit=$?" | tee -a "$LOG/quick_chain_vs_tree.log"
fi

# --- Plot top4 paths ---
"$PYTHON" "$ROOT/plot_profile_chain_vs_tree.py" \
  --chain-tag quick_sample_chain_eal \
  --tree-tag quick_sample_tree_eal \
  --out "$ROOT/quick_chain_vs_tree_top4_paths.png" \
  --chain-gpus 1 --tree-gpus 3 --path-compare --xmax 5000 \
  | tee -a "$LOG/quick_chain_vs_tree.log"

"$PYTHON" "$ROOT/plot_profile_chain_vs_tree.py" \
  --chain-tag quick_sample_chain_eal \
  --tree-tag quick_sample_tree_eal \
  --out "$ROOT/quick_chain_vs_tree.png" \
  --chain-gpus 1 --tree-gpus 3 \
  | tee -a "$LOG/quick_chain_vs_tree.log"

# --- Summary ---
"$PYTHON" - <<'PY' | tee -a "$LOG/quick_chain_vs_tree.log"
import json, os
root = "/mnt/ssd_ext/lls/cerebras_sd/checkpoints"
print("\n=== QUICK COMPARE SUMMARY ===")
for tag in ["quick_sample_chain_eal", "quick_sample_tree_eal"]:
    ep = os.path.join(root, tag, "epoch_metrics.jsonl")
    pr = os.path.join(root, tag, "tree_path_probe.jsonl")
    print(f"\n[{tag}]")
    if os.path.isfile(ep):
        for line in open(ep):
            r = json.loads(line)
            keys = ["epoch", "train_analytic_accepted_length", "test_analytic_accepted_length",
                    "train_loss", "test_loss"]
            print(" epoch", {k: r.get(k) for k in keys if k in r})
    else:
        print("  no epoch_metrics")
    if os.path.isfile(pr):
        for line in open(pr):
            r = json.loads(line)
            print(" probe", {k: r.get(k) for k in
                ["tag","tree_probe_eal","path1_al","path2_al","path3_al","path4_al"]})
    else:
        print("  no tree_path_probe")
print("\nplots: quick_chain_vs_tree.png , quick_chain_vs_tree_top4_paths.png")
PY
echo "[quick] done $(date -Is)" | tee -a "$LOG/quick_chain_vs_tree.log"
