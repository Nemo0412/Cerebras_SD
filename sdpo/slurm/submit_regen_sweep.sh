#!/bin/bash
# Submit training sweep on target-matched regenerated ShareGPT datasets.
#   8B  target  -> sharegpt_qwen3_8b_regen.jsonl
#   32B target  -> sharegpt_qwen3_32b_regen.jsonl
#
# Defaults: max_len=1024, lr=1e-6, coef=0.1, T=0.1, 3 epochs.
# Each train job auto-chains a tree+chain eval (gamma=7, budget=128,
# top_k=4,3,2,1,1,1,1) via --dependency=afterok.
#
# Usage:
#   bash sdpo/slurm/submit_regen_sweep.sh                 # submit ALL
#   bash sdpo/slurm/submit_regen_sweep.sh q8_q06_klv2_ro  # single tag
#   bash sdpo/slurm/submit_regen_sweep.sh --list          # list tasks
#   bash sdpo/slurm/submit_regen_sweep.sh --klv2          # only KL+V2 combos
#   bash sdpo/slurm/submit_regen_sweep.sh --v4            # all V4 combos (8)
#   bash sdpo/slurm/submit_regen_sweep.sh --klv4          # only KL+V4 combos (4)
#   bash sdpo/slurm/submit_regen_sweep.sh --aw            # anchor-weight ablation (aux=none, 16 tasks)
#   bash sdpo/slurm/submit_regen_sweep.sh --xw            # aux-weight ablation (aw=none, 24 tasks)
#   bash sdpo/slurm/submit_regen_sweep.sh --weights       # both weight ablations (40 tasks)
#   bash sdpo/slurm/submit_regen_sweep.sh --ropo          # all on-policy rollout tasks (224 total)
#   bash sdpo/slurm/submit_regen_sweep.sh --ropow         # ropo weight-sweep (anchor_w × aux_w, 208 tasks)
#   bash sdpo/slurm/submit_regen_sweep.sh --soro          # soft-rollout tasks (16 total)
#   NO_EVAL=1 bash ... <tag>                              # skip auto-eval
#   MAX_LEN=1024 bash ... <tag>                           # override seq length
#   ANCHOR_WEIGHT=pow08 bash ... <tag>                    # override anchor_weight (if task col empty)
#     choices: none | uniform | pow08 | dec | inc
#       none    — per-token average on prefix (legacy behavior)
#       uniform — γ-window sum, each step weighted 1
#       pow08   — γ-window sum, weights [1, 0.8, 0.64, ...] (matches EAGLE-3 anchor)
#       dec     — [γ, γ-1, ..., 1] (linearly decreasing, early-heavy)
#       inc     — [1, 2, ..., γ]   (linearly increasing, late-heavy)
#   AUX_WEIGHT=dec bash ... <tag>                         # override aux_weight (if task col empty)
#     choices: uniform | pow08 | dec | inc
#       (same schemes as anchor_weight, but 'none' is not allowed — aux always
#        runs over the γ rollout window)
#   EPOCHS=5 bash ... <tag>                               # override num epochs
#   MAX_SAMPLES=2000 bash ... <tag>                       # subset training (debug)
#
# Resolution order for anchor_weight / aux_weight:
#   1. task's 6th/7th column (per-task explicit override)
#   2. env var ANCHOR_WEIGHT / AUX_WEIGHT (caller-set default)
#   3. script default (none / pow08)
#
# Tag naming: q<target>_q<draft>_<anchor><aux>_regen[_ro]
#   ro = rollout, no suffix = teacher-forced (non-rollout)

set -e
cd /scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization
mkdir -p logs/sdpo_smalllm_q32_q06 logs/sdpo_smalllm_06b logs/smalllm_rollout

# ── Paths ─────────────────────────────────────────────────────────────────
# Target-matched regen data: 8B target trained on 8B-regen; 32B target on 32B-regen.
TRAIN_REGEN_8B=/scratch/yf3005/gto_data/sharegpt_qwen3_8b_regen.jsonl
TRAIN_REGEN_32B=/scratch/yf3005/gto_data/sharegpt_qwen3_32b_regen.jsonl
VAL_SMALL=/scratch/tx856/spec_reason/accept_length/SDPO-Speculative-Decoding-Policy-Optimization/sdpo/data/mixed_val_80.jsonl

SLURM_NORO=sdpo/slurm/smalllm_06b_train.slurm         # teacher-forced
SLURM_RO=sdpo/slurm/smalllm_rollout_train.slurm       # on-policy rollout
SLURM_EVAL=sdpo/slurm/smalllm_tree_eval.slurm         # chain+tree eval
DS_CONFIG_8B_NORO=sdpo/sdpo_config_qwen3.json         # fp16 ZeRO-2
DS_CONFIG_ZERO3=sdpo/sdpo_config_qwen3_bf16_zero3_2gpu.json   # bf16 ZeRO-3

NUM_EPOCHS=${EPOCHS:-3}
MAX_SAMPLES=${MAX_SAMPLES:-""}
MAX_LEN=${MAX_LEN:-1024}
SIGMOID_COEF=${SIGMOID_COEF:-0.1}

# ── Eval config (chain+tree, gamma=7 budget=128 top_k=4,3,2,1,1,1,1) ──────
# NOTE: vars containing commas (BENCH, TOP_K) are forwarded via sbatch process env
# (ALL mode), NOT inside --export=ALL,KEY=val list — commas would corrupt parsing.
EVAL_BENCH_VAL="mt_bench,gsm8k,humaneval"
EVAL_TOP_K_VAL="4,3,2,1,1,1,1"
EVAL_NUM_SAMPLES=80
EVAL_MAX_NEW_TOKENS=256
EVAL_GAMMA=7
EVAL_BUDGET=128
EVAL_BASELINE_GAMMA=7
EVAL_OUTPUT_DIR=smalllm_tree_eval_results

# ── Task table: "tag|target|anchor|aux|rollout[|anchor_weight|aux_weight]" ─
# target  ∈ {8b, 32b}
# rollout ∈ {noro, ro, ropo, soro}
#   noro = teacher-forced (no rollout)
#   ro   = old rollout, target uses CLEAN argmax (context mismatch, kept for baseline)
#   ropo = rollout + on-policy target verify (draft's dirty chain re-forwarded
#          through target to get DIRTY-context target_argmax for β_k)
#   soro = soft rollout (expected embedding feedback), target stays CLEAN,
#          aux=v5 truncates cum_prod sum at first hard reject.
# anchor  ∈ {kl, ce}                     aux    ∈ {none, v2, v4, tv}
# anchor_weight default 'none', aux_weight default 'pow08' if columns absent.
# weight  ∈ {none, uniform, pow08, dec, inc} (anchor_weight='none' disables γ-window;
#   aux_weight cannot be 'none' since aux always has γ window).
TASKS=(
    # ─── 8B + 0.6B ───────────────────────────────────────────
    #   KL + V2 (primary)
    "q8_q06_klv2_regen       | 8b  | kl | v2   | noro"
    "q8_q06_klv2_regen_ro    | 8b  | kl | v2   | ro"
    #   CE + V2
    "q8_q06_cev2_regen       | 8b  | ce | v2   | noro"
    "q8_q06_cev2_regen_ro    | 8b  | ce | v2   | ro"
    #   KL + V4
    "q8_q06_klv4_regen       | 8b  | kl | v4   | noro"
    "q8_q06_klv4_regen_ro    | 8b  | kl | v4   | ro"
    #   CE + V4
    "q8_q06_cev4_regen       | 8b  | ce | v4   | noro"
    "q8_q06_cev4_regen_ro    | 8b  | ce | v4   | ro"
    #   KL + V5 (truncate at first reject, noro only)
    "q8_q06_klv5_regen       | 8b  | kl | v5   | noro"
    #   CE + V5
    "q8_q06_cev5_regen       | 8b  | ce | v5   | noro"
    #   KL + V6 (peaked at first reject, noro only)
    "q8_q06_klv6_regen       | 8b  | kl | v6   | noro"
    #   CE + V6
    "q8_q06_cev6_regen       | 8b  | ce | v6   | noro"
    #   KL + TV
    "q8_q06_kltv_regen       | 8b  | kl | tv   | noro"
    "q8_q06_kltv_regen_ro    | 8b  | kl | tv   | ro"
    #   CE + TV
    "q8_q06_cetv_regen       | 8b  | ce | tv   | noro"
    "q8_q06_cetv_regen_ro    | 8b  | ce | tv   | ro"
    #   KL only / CE only
    "q8_q06_kl_regen         | 8b  | kl | none | noro"
    "q8_q06_kl_regen_ro      | 8b  | kl | none | ro"
    "q8_q06_ce_regen         | 8b  | ce | none | noro"
    "q8_q06_ce_regen_ro      | 8b  | ce | none | ro"

    # ─── 8B + 0.6B — retrain with MAX_LEN=2048 (l2k suffix) ──────────
    #   Preserves original max_len=1024 checkpoints & eval JSONs under the
    #   original tag; this block writes to <tag>_l2k directories.
    "q8_q06_kl_regen_l2k     | 8b  | kl | none | noro"
    # noro + ZeRO-3 + BF16 control (vs default ZeRO-2 + FP16) to isolate
    # dtype/zero confounder when comparing to ro path. Submit with:
    #   DEEPSPEED_CONFIG_OVERRIDE=sdpo/sdpo_config_qwen3_bf16_zero3_2gpu.json
    "q8_q06_kl_regen_l2k_zero3 | 8b | kl | none | noro"
    "q8_q06_ce_regen_l2k     | 8b  | ce | none | noro"
    "q8_q06_klv4_regen_l2k   | 8b  | kl | v4   | noro"
    "q8_q06_cev4_regen_l2k   | 8b  | ce | v4   | noro"
    "q8_q06_klv5_regen_l2k   | 8b  | kl | v5   | noro"
    "q8_q06_cev5_regen_l2k   | 8b  | ce | v5   | noro"
    "q8_q06_klv6_regen_l2k   | 8b  | kl | v6   | noro"
    "q8_q06_cev6_regen_l2k   | 8b  | ce | v6   | noro"
    "q8_q06_kltv_regen_l2k   | 8b  | kl | tv   | noro"
    "q8_q06_cetv_regen_l2k   | 8b  | ce | tv   | noro"

    # ─── l2k + coef sweep on V4 — sigmoid_coef ∈ {0.5, 1.0} ──────────
    #   coef lives in col 8; tag + coef coupled — no env var needed.
    #   Default l2k uses coef=0.1; these extend the sweep to 0.5 and 1.0.
    "q8_q06_klv4_regen_l2k_c05 | 8b  | kl | v4 | noro | none | pow08 | 0.5"
    "q8_q06_cev4_regen_l2k_c05 | 8b  | ce | v4 | noro | none | pow08 | 0.5"
    "q8_q06_klv4_regen_l2k_c10 | 8b  | kl | v4 | noro | none | pow08 | 1.0"
    "q8_q06_cev4_regen_l2k_c10 | 8b  | ce | v4 | noro | none | pow08 | 1.0"

    # ─── 32B + 0.6B ──────────────────────────────────────────
    #   KL + V2 (primary)
    "q32_q06_klv2_regen      | 32b | kl | v2   | noro"
    "q32_q06_klv2_regen_ro   | 32b | kl | v2   | ro"
    #   CE + V2
    "q32_q06_cev2_regen      | 32b | ce | v2   | noro"
    "q32_q06_cev2_regen_ro   | 32b | ce | v2   | ro"
    #   KL + V4
    "q32_q06_klv4_regen      | 32b | kl | v4   | noro"
    "q32_q06_klv4_regen_ro   | 32b | kl | v4   | ro"
    #   CE + V4
    "q32_q06_cev4_regen      | 32b | ce | v4   | noro"
    "q32_q06_cev4_regen_ro   | 32b | ce | v4   | ro"
    #   KL + V5 (truncate at first reject, noro only)
    "q32_q06_klv5_regen      | 32b | kl | v5   | noro"
    #   CE + V5
    "q32_q06_cev5_regen      | 32b | ce | v5   | noro"
    #   KL + TV
    "q32_q06_kltv_regen      | 32b | kl | tv   | noro"
    "q32_q06_kltv_regen_ro   | 32b | kl | tv   | ro"
    #   CE + TV
    "q32_q06_cetv_regen      | 32b | ce | tv   | noro"
    "q32_q06_cetv_regen_ro   | 32b | ce | tv   | ro"
    #   KL only / CE only
    "q32_q06_kl_regen        | 32b | kl | none | noro"
    "q32_q06_kl_regen_ro     | 32b | kl | none | ro"
    "q32_q06_ce_regen        | 32b | ce | none | noro"
    "q32_q06_ce_regen_ro     | 32b | ce | none | ro"

    # ─── 8B + 0.6B — anchor_weight=v6 (dynamic per-window: accept=1, first_reject=1.5, after=0)
    "q8_q06_kl_awv6_regen    | 8b  | kl | none | noro | v6 | pow08"
    "q8_q06_ce_awv6_regen    | 8b  | ce | none | noro | v6 | pow08"
    "q8_q06_klv4_awv6_regen  | 8b  | kl | v4   | noro | v6 | pow08"
    "q8_q06_cev4_awv6_regen  | 8b  | ce | v4   | noro | v6 | pow08"

    # ─── 8B + 0.6B — EAL anchor / EAL aux family ────────────────
    # anchor=eal (window-level Σ cum_prod(P)); aux=eal (same on top of KL/CE prefix).
    # anchor=eal + aux=none: pure EAL anchor
    "q8_q06_eal_uni_regen    | 8b  | eal | none | noro | uniform | pow08"
    "q8_q06_eal_v6_regen     | 8b  | eal | none | noro | v6      | pow08"
    # KL prefix + EAL aux
    "q8_q06_kleal_pow_pow_regen | 8b  | kl  | eal | noro | pow08 | pow08"
    "q8_q06_kleal_v6_v6_regen   | 8b  | kl  | eal | noro | v6    | v6"
    "q8_q06_kleal_none_v6_regen | 8b  | kl  | eal | noro | none  | v6"
    # CE prefix + EAL aux
    "q8_q06_ceeal_pow_pow_regen | 8b  | ce  | eal | noro | pow08 | pow08"
    "q8_q06_ceeal_v6_v6_regen   | 8b  | ce  | eal | noro | v6    | v6"
    "q8_q06_ceeal_none_v6_regen | 8b  | ce  | eal | noro | none  | v6"
    # ─── l2k re-runs (max_len=2048, samples=60000, truncate) ──
    # EAL family (l2k):
    "q8_q06_eal_uni_regen_l2k       | 8b  | eal | none | noro | uniform | pow08"
    "q8_q06_eal_v6_regen_l2k        | 8b  | eal | none | noro | v6      | pow08"
    "q8_q06_kleal_pow_pow_regen_l2k | 8b  | kl  | eal  | noro | pow08   | pow08"
    "q8_q06_kleal_v6_v6_regen_l2k   | 8b  | kl  | eal  | noro | v6      | v6"
    "q8_q06_kleal_none_v6_regen_l2k | 8b  | kl  | eal  | noro | none    | v6"
    "q8_q06_ceeal_pow_pow_regen_l2k | 8b  | ce  | eal  | noro | pow08   | pow08"
    "q8_q06_ceeal_v6_v6_regen_l2k   | 8b  | ce  | eal  | noro | v6      | v6"
    "q8_q06_ceeal_none_v6_regen_l2k | 8b  | ce  | eal  | noro | none    | v6"
    # klv4 / kl rerun with new tag for variance check:
    "q8_q06_klv4_regen_l2k_rerun    | 8b  | kl  | v4   | noro"
    "q8_q06_kl_regen_l2k_rerun      | 8b  | kl  | none | noro"
    # GRPO continuation from best klv4_l2k ckpt. Columns:
    #   tag | target | anchor | aux | rollout | aw | xw | coef | grpo_coef | init_draft_path
    "q8_q06_klv4_l2k_grpo           | 8b | kl | v4 | noro | none | pow08 | 0.1 | 0.1 | /scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/q8_q06_klv4_regen_l2k/state_2"
    # KL-only continuation control (same init, same data config, no GRPO, no V4 aux, 3 epochs)
    "q8_q06_kl_l2k_cont             | 8b | kl | none | noro | none | pow08 | 0.1 | 0.0 | /scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/q8_q06_klv4_regen_l2k/state_2"
    # KL+V4 continuation control: identical to klv4_l2k_grpo except grpo_coef=0 (isolates GRPO)
    "q8_q06_klv4_l2k_cont           | 8b | kl | v4   | noro | none | pow08 | 0.1 | 0.0 | /scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/q8_q06_klv4_regen_l2k/state_2"
    # ─── 2k dataset config (max_len=2048, drop filter, ~57945 samples) — mirror l2k trio ──
    # Init from 2k klv4 ckpt (/q8_q06_klv4_regen_2k/state_2). Submit with: MAX_LEN=2048 (no MAX_SAMPLES)
    "q8_q06_klv4_2k_grpo            | 8b | kl | v4   | noro | none | pow08 | 0.1 | 0.1 | /scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/q8_q06_klv4_regen_2k/state_2"
    "q8_q06_klv4_2k_cont            | 8b | kl | v4   | noro | none | pow08 | 0.1 | 0.0 | /scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/q8_q06_klv4_regen_2k/state_2"
    "q8_q06_kl_2k_cont              | 8b | kl | none | noro | none | pow08 | 0.1 | 0.0 | /scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/q8_q06_klv4_regen_2k/state_2"
    # ─── KLV4 with T schedule (T_min=0.1 → T_max=2.0) — escape sigmoid saturation ──
    # Submit with: T_MIN=0.1 T_MAX=2.0 MAX_LEN=2048 MAX_SAMPLES=60000
    "q8_q06_klv4_regen_l2k_tsched   | 8b | kl | v4   | noro | none | pow08 | 0.1 | 0.0"
    # ─── GRPO with stronger RL signal (grpo_coef 0.1 → 0.5) ──
    "q8_q06_klv4_l2k_grpo_c05       | 8b | kl | v4   | noro | none | pow08 | 0.1 | 0.5 | /scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/q8_q06_klv4_regen_l2k/state_2"
    # ─── Pure GRPO (no anchor, no aux) from klv4_l2k init — isolate RL-only contribution ──
    "q8_q06_klv4_l2k_grpo_only      | 8b | none | none | noro | none | pow08 | 0.1 | 0.1 | /scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/q8_q06_klv4_regen_l2k/state_2"
    # ─── GRPO from KL+EAL (pow_pow) ckpt: same setup as klv4_l2k_grpo but EAL aux instead of V4 ──
    "q8_q06_kleal_l2k_grpo          | 8b | kl   | eal  | noro | pow08 | pow08 | 0.1 | 0.1 | /scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/q8_q06_kleal_pow_pow_regen_l2k/state_2"
    # ─── Pure GRPO (no anchor, no aux) from KL+EAL ckpt — RL-only from EAL warm-start ──
    "q8_q06_kleal_l2k_grpo_only     | 8b | none | none | noro | none  | pow08 | 0.1 | 0.1 | /scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/q8_q06_kleal_pow_pow_regen_l2k/state_2"
    # ─── GRPO group sweep: K=16 × M={4,8}, both KLV4+GRPO and GRPO-only, init=klv4_l2k ──
    # Submit with: GRPO_K_GROUPS=16 GRPO_M=<4|8> MAX_LEN=2048 MAX_SAMPLES=60000
    "q8_q06_klv4_l2k_grpo_k16m4     | 8b | kl   | v4   | noro | none  | pow08 | 0.1 | 0.1 | /scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/q8_q06_klv4_regen_l2k/state_2"
    "q8_q06_klv4_l2k_grpo_k16m8     | 8b | kl   | v4   | noro | none  | pow08 | 0.1 | 0.1 | /scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/q8_q06_klv4_regen_l2k/state_2"
    "q8_q06_klv4_l2k_grpo_only_k16m4 | 8b | none | none | noro | none | pow08 | 0.1 | 0.1 | /scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/q8_q06_klv4_regen_l2k/state_2"
    "q8_q06_klv4_l2k_grpo_only_k16m8 | 8b | none | none | noro | none | pow08 | 0.1 | 0.1 | /scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/q8_q06_klv4_regen_l2k/state_2"
    # ─── PEAL (penalty-augmented EAL): same setup as kleal_pow_pow_regen_l2k but penalty subtracts dominant distractors ──
    "q8_q06_klpeal_pow_pow_regen_l2k | 8b | kl  | peal | noro | pow08 | pow08 | 0.1 | 0.0"
    # ─── 8B+0.6B old rollout (ro) on full ShareGPT 8B regen (120,675 samples, max_len=2048 truncate) ──
    # Submit with: SLURM_TRAIN_TIME=3-00:00:00 MAX_LEN=2048 MAX_SAMPLES=120675
    "q8_q06_kl_regen_ro_full        | 8b | kl  | none | ro"
    "q8_q06_klv4_regen_ro_full      | 8b | kl  | v4   | ro"
    # ─── ro KL only with FP16+ZeRO-2 (matches noro 8B config) — isolates dtype/zero impact ──
    # Submit with: DEEPSPEED_CONFIG_OVERRIDE=sdpo/sdpo_config_qwen3.json
    "q8_q06_kl_regen_ro_fp16        | 8b | kl  | none | ro"
    # ─── noro full-data (120K samples truncate, max_len=2048) baselines for ro_full comparison ──
    # Submit with: MAX_LEN=2048 MAX_SAMPLES=120675
    "q8_q06_kl_regen_full           | 8b | kl  | none | noro"
    "q8_q06_klv4_regen_full         | 8b | kl  | v4   | noro"
    # ─── l2k KLV4 ro (60K truncate, fix-applied chain_anchor) ──
    # Submit with: MAX_LEN=2048 MAX_SAMPLES=60000
    "q8_q06_klv4_regen_ro_l2k       | 8b | kl  | v4   | ro"
    # ─── KL+V7 (top-2 aware sigmoid gap) baselines ──
    # Submit with: MAX_LEN=2048 MAX_SAMPLES=60000 (l2k) or 120675 (full)
    "q8_q06_klv7_regen_l2k          | 8b | kl  | v7   | noro"
    "q8_q06_klv7_regen_full         | 8b | kl  | v7   | noro"
    # ─── ro l2k with γ=2 (only 1 dirty rollout step beyond clean prefix) ──
    # Submit with: GAMMA=2 MAX_LEN=2048 MAX_SAMPLES=60000
    "q8_q06_klv4_regen_ro_l2k_g2    | 8b | kl  | v4   | ro"
    "q8_q06_kl_regen_ro_l2k_g2      | 8b | kl  | none | ro"
    # ─── 32B + 0.6B noro l2k baselines (KL only / KL+V4) ──
    # Submit with: SLURM_TRAIN_TIME=3-00:00:00 MAX_LEN=2048 MAX_SAMPLES=60000
    "q32_q06_kl_regen_l2k           | 32b | kl | none | noro"
    "q32_q06_klv4_regen_l2k         | 32b | kl | v4   | noro"
    # ─── 32B follow-ups: continuation from above l2k ckpt ──
    # Submit with: SLURM_TRAIN_TIME=2-00:00:00 SLURM_DEPENDENCY=afterok:<train_jid>
    "q32_q06_kl_l2k_cont            | 32b | kl | none | noro | none | pow08 | 0.1 | 0.0 | /scratch/tx856/spec_reason/scratch/loss_train_smalllm_q32_q06/q32_q06_kl_regen_l2k/state_2"
    "q32_q06_klv4_l2k_grpo_smp      | 32b | kl | v4   | noro | none | pow08 | 0.1 | 0.1 | /scratch/tx856/spec_reason/scratch/loss_train_smalllm_q32_q06/q32_q06_klv4_regen_l2k/state_2"
    # ─── ro l2k with truncate-at-first-reject (mitigate context mismatch) ──
    # Submit with: ROLLOUT_TRUNCATE_AT_REJECT=1 MAX_LEN=2048 MAX_SAMPLES=60000
    "q8_q06_klv4_regen_ro_l2k_trunc | 8b | kl  | v4   | ro"
    # CE only (no V4 aux) ro l2k with truncate
    "q8_q06_ce_regen_ro_l2k_trunc   | 8b | ce  | none | ro"
    # KL only ro l2k with truncate (companion to CE version)
    "q8_q06_kl_regen_ro_l2k_trunc   | 8b | kl  | none | ro"
    # CE only ro l2k with truncate + chain_anchor_coef=0 (isolate dtype/zero confounder)
    "q8_q06_ce_regen_ro_l2k_trunc_c0 | 8b | ce  | none | ro"
    # FP16 ZeRO-2 versions of trunc_c0 and ropo (verify dtype is the 0.016 gap source).
    # Submit with: DEEPSPEED_CONFIG_OVERRIDE=sdpo/sdpo_config_qwen3.json
    "q8_q06_ce_regen_ro_l2k_trunc_c0_fp16 | 8b | ce | none | ro"
    "q8_q06_klv4_regen_ropo_fp16 | 8b | kl | v4 | ropo"
    # ─── chain-mode rollout (γ controls chain length only; γ=1 should ≈ noro kl_l2k) ──
    # Submit with: ROLLOUT_LOSS_MODE=chain GAMMA=<n> MAX_LEN=2048 MAX_SAMPLES=60000
    #              DEEPSPEED_CONFIG_OVERRIDE=sdpo/sdpo_config_qwen3.json   (match noro fp16 zero2)
    "q8_q06_kl_regen_chain_g1   | 8b | kl | none | ro"
    "q8_q06_kl_regen_chain_g2   | 8b | kl | none | ro"
    "q8_q06_klv4_regen_chain_g2 | 8b | kl | v4   | ro"
    "q8_q06_kl_regen_chain_g7   | 8b | kl | none | ro"
    "q8_q06_klv4_regen_chain_g7 | 8b | kl | v4   | ro"
    # ─── Sample-mode GRPO from klv4_l2k init (multinomial sample at shared anchor) ──
    # Submit with: GRPO_MODE=sample (and optional GRPO_SAMPLE_TEMP=1.0)
    "q8_q06_klv4_l2k_grpo_smp       | 8b | kl   | v4   | noro | none | pow08 | 0.1 | 0.1 | /scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/q8_q06_klv4_regen_l2k/state_2"
    "q8_q06_klv4_l2k_grpo_only_smp  | 8b | none | none | noro | none | pow08 | 0.1 | 0.1 | /scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/q8_q06_klv4_regen_l2k/state_2"
    # ─── EAL-reward GRPO (window mode, reward = Σ cumprod(P_target) instead of hard τ) ──
    # Submit with: GRPO_REWARD=eal
    "q8_q06_klv4_l2k_grpo_eal       | 8b | kl   | v4   | noro | none | pow08 | 0.1 | 0.1 | /scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/q8_q06_klv4_regen_l2k/state_2"
    "q8_q06_klv4_l2k_grpo_only_eal  | 8b | none | none | noro | none | pow08 | 0.1 | 0.1 | /scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/q8_q06_klv4_regen_l2k/state_2"
    # ─── 6-epoch GRPO from base 0.6B (no SFT warm-start), KL+V4+GRPO ──
    # Submit with: EPOCHS=6 MAX_LEN=2048 MAX_SAMPLES=60000
    "q8_q06_klv4_l2k_grpo_base_6ep  | 8b | kl   | v4   | noro | none | pow08 | 0.1 | 0.1"
    # ─── 6-epoch GRPO from base 0.6B with sample mode (stochastic traces) ──
    # Submit with: EPOCHS=6 GRPO_MODE=sample MAX_LEN=2048 MAX_SAMPLES=60000
    "q8_q06_klv4_l2k_grpo_smp_base_6ep | 8b | kl | v4 | noro | none | pow08 | 0.1 | 0.1"
    # ─── l2k 6-epoch SFT ablations (start from base 0.6B) ──
    # Probe SFT ceiling: do 6 epochs unlock anything that 3 epochs missed?
    "q8_q06_kl_regen_l2k_6ep        | 8b | kl | none | noro | none | pow08 | 0.1 | 0.0"
    "q8_q06_klv4_regen_l2k_6ep      | 8b | kl | v4   | noro | none | pow08 | 0.1 | 0.0"
    "q8_q06_kltv_regen_l2k_6ep      | 8b | kl | tv   | noro | none | pow08 | 0.1 | 0.0"

    # ─── New E[L]-style losses (al_tv / al_kl / wkl), aux-only SFT 6ep ──
    "q8_q06_altv_only_regen_l2k_6ep | 8b | none | al_tv | noro | none | pow08 | 1.0 | 0.0"
    "q8_q06_alkl_only_regen_l2k_6ep | 8b | none | al_kl | noro | none | pow08 | 1.0 | 0.0"
    "q8_q06_wkl_only_regen_l2k_6ep  | 8b | none | wkl   | noro | none | pow08 | 1.0 | 0.0"

    # ─── New losses + GRPO (6ep from base, aux + GRPO together) ─────
    # Submit with: EPOCHS=6 GRPO_COEF=0.1 GRPO_MODE=window GRPO_REWARD=hard
    "q8_q06_altv_grpo_6ep           | 8b | none | al_tv | noro | none | pow08 | 1.0 | 0.1"
    "q8_q06_alkl_grpo_6ep           | 8b | none | al_kl | noro | none | pow08 | 1.0 | 0.1"
    "q8_q06_wkl_grpo_6ep            | 8b | none | wkl   | noro | none | pow08 | 1.0 | 0.1"

    # ─── KL + V8 (gap-vs-top2) ──────────────────────────────────────
    "q8_q06_klv8_regen_l2k_6ep      | 8b | kl   | v8    | noro | none | pow08 | 0.1 | 0.0"
    "q8_q06_klv8_regen_l2k_3ep      | 8b | kl   | v8    | noro | none | pow08 | 0.1 | 0.0"
    "q8_q06_klv8_l2k_grpo           | 8b | kl   | v8    | noro | none | pow08 | 0.1 | 0.1 | /scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/q8_q06_klv8_regen_l2k_3ep/state_2"
    # ─── GRPO from base 0.6B (no SFT warm-start), l2k, 3 epochs ──
    "q8_q06_klv4_l2k_grpo_base      | 8b | kl | v4   | noro | none | pow08 | 0.1 | 0.1"
    # ─── 2k re-runs (max_len=2048, drop filter, ~57945 samples) ──
    # Matches klv4_regen (4.752) / kl_regen (4.742) config for fair compare
    # to original "non-l2k SOTA" baseline.
    "q8_q06_eal_uni_regen_2k        | 8b  | eal | none | noro | uniform | pow08"
    "q8_q06_eal_v6_regen_2k         | 8b  | eal | none | noro | v6      | pow08"
    "q8_q06_kleal_pow_pow_regen_2k  | 8b  | kl  | eal  | noro | pow08   | pow08"
    "q8_q06_kleal_v6_v6_regen_2k    | 8b  | kl  | eal  | noro | v6      | v6"
    "q8_q06_kleal_none_v6_regen_2k  | 8b  | kl  | eal  | noro | none    | v6"
    "q8_q06_ceeal_pow_pow_regen_2k  | 8b  | ce  | eal  | noro | pow08   | pow08"
    "q8_q06_ceeal_v6_v6_regen_2k    | 8b  | ce  | eal  | noro | v6      | v6"
    "q8_q06_ceeal_none_v6_regen_2k  | 8b  | ce  | eal  | noro | none    | v6"
    "q8_q06_klv4_regen_2k           | 8b  | kl  | v4   | noro"
    "q8_q06_kl_regen_2k             | 8b  | kl  | none | noro"
    # l2k variants (max_len=2048, max_samples=60000)
    "q8_q06_kl_awv6_regen_l2k    | 8b  | kl | none | noro | v6 | pow08"
    "q8_q06_ce_awv6_regen_l2k    | 8b  | ce | none | noro | v6 | pow08"
    "q8_q06_klv4_awv6_regen_l2k  | 8b  | kl | v4   | noro | v6 | pow08"
    "q8_q06_cev4_awv6_regen_l2k  | 8b  | ce | v4   | noro | v6 | pow08"

    # ═══════════════════════════════════════════════════════════════════════
    # Weight-scheme ablation (non-rollout only, both targets)
    # columns: tag | target | anchor | aux | rollout | anchor_weight | aux_weight
    # ═══════════════════════════════════════════════════════════════════════

    # ─── Part 1: anchor weight sweep (aux=none) ────────────────────────
    # 32B + 0.6B — KL only × 4 anchor_weight
    "q32_q06_kl_awuni_regen  | 32b | kl | none | noro | uniform | pow08"
    "q32_q06_kl_awpow_regen  | 32b | kl | none | noro | pow08   | pow08"
    "q32_q06_kl_awdec_regen  | 32b | kl | none | noro | dec     | pow08"
    "q32_q06_kl_awinc_regen  | 32b | kl | none | noro | inc     | pow08"
    # 32B + 0.6B — CE only × 4 anchor_weight
    "q32_q06_ce_awuni_regen  | 32b | ce | none | noro | uniform | pow08"
    "q32_q06_ce_awpow_regen  | 32b | ce | none | noro | pow08   | pow08"
    "q32_q06_ce_awdec_regen  | 32b | ce | none | noro | dec     | pow08"
    "q32_q06_ce_awinc_regen  | 32b | ce | none | noro | inc     | pow08"
    # 8B + 0.6B — KL only × 4 anchor_weight
    "q8_q06_kl_awuni_regen   | 8b  | kl | none | noro | uniform | pow08"
    "q8_q06_kl_awpow_regen   | 8b  | kl | none | noro | pow08   | pow08"
    "q8_q06_kl_awdec_regen   | 8b  | kl | none | noro | dec     | pow08"
    "q8_q06_kl_awinc_regen   | 8b  | kl | none | noro | inc     | pow08"
    # 8B + 0.6B — CE only × 4 anchor_weight
    "q8_q06_ce_awuni_regen   | 8b  | ce | none | noro | uniform | pow08"
    "q8_q06_ce_awpow_regen   | 8b  | ce | none | noro | pow08   | pow08"
    "q8_q06_ce_awdec_regen   | 8b  | ce | none | noro | dec     | pow08"
    "q8_q06_ce_awinc_regen   | 8b  | ce | none | noro | inc     | pow08"

    # ─── Part 2: aux weight sweep (anchor_weight=none, anchor=KL) ──────
    # 32B + 0.6B — KL+V2 × 4 aux_weight
    "q32_q06_klv2_xwuni_regen | 32b | kl | v2 | noro | none | uniform"
    "q32_q06_klv2_xwpow_regen | 32b | kl | v2 | noro | none | pow08"
    "q32_q06_klv2_xwdec_regen | 32b | kl | v2 | noro | none | dec"
    "q32_q06_klv2_xwinc_regen | 32b | kl | v2 | noro | none | inc"
    # 32B + 0.6B — KL+V4 × 4
    "q32_q06_klv4_xwuni_regen | 32b | kl | v4 | noro | none | uniform"
    "q32_q06_klv4_xwpow_regen | 32b | kl | v4 | noro | none | pow08"
    "q32_q06_klv4_xwdec_regen | 32b | kl | v4 | noro | none | dec"
    "q32_q06_klv4_xwinc_regen | 32b | kl | v4 | noro | none | inc"
    # 32B + 0.6B — KL+V5 × 4
    "q32_q06_klv5_xwuni_regen | 32b | kl | v5 | noro | none | uniform"
    "q32_q06_klv5_xwpow_regen | 32b | kl | v5 | noro | none | pow08"
    "q32_q06_klv5_xwdec_regen | 32b | kl | v5 | noro | none | dec"
    "q32_q06_klv5_xwinc_regen | 32b | kl | v5 | noro | none | inc"
    # 32B + 0.6B — KL+TV × 4
    "q32_q06_kltv_xwuni_regen | 32b | kl | tv | noro | none | uniform"
    "q32_q06_kltv_xwpow_regen | 32b | kl | tv | noro | none | pow08"
    "q32_q06_kltv_xwdec_regen | 32b | kl | tv | noro | none | dec"
    "q32_q06_kltv_xwinc_regen | 32b | kl | tv | noro | none | inc"
    # ─── Part 2: aux weight sweep (anchor=CE) ──────────────────────────
    # 32B + 0.6B — CE+V2 × 4
    "q32_q06_cev2_xwuni_regen | 32b | ce | v2 | noro | none | uniform"
    "q32_q06_cev2_xwpow_regen | 32b | ce | v2 | noro | none | pow08"
    "q32_q06_cev2_xwdec_regen | 32b | ce | v2 | noro | none | dec"
    "q32_q06_cev2_xwinc_regen | 32b | ce | v2 | noro | none | inc"
    # 32B + 0.6B — CE+V4 × 4
    "q32_q06_cev4_xwuni_regen | 32b | ce | v4 | noro | none | uniform"
    "q32_q06_cev4_xwpow_regen | 32b | ce | v4 | noro | none | pow08"
    "q32_q06_cev4_xwdec_regen | 32b | ce | v4 | noro | none | dec"
    "q32_q06_cev4_xwinc_regen | 32b | ce | v4 | noro | none | inc"
    # 32B + 0.6B — CE+V5 × 4
    "q32_q06_cev5_xwuni_regen | 32b | ce | v5 | noro | none | uniform"
    "q32_q06_cev5_xwpow_regen | 32b | ce | v5 | noro | none | pow08"
    "q32_q06_cev5_xwdec_regen | 32b | ce | v5 | noro | none | dec"
    "q32_q06_cev5_xwinc_regen | 32b | ce | v5 | noro | none | inc"
    # 32B + 0.6B — CE+TV × 4
    "q32_q06_cetv_xwuni_regen | 32b | ce | tv | noro | none | uniform"
    "q32_q06_cetv_xwpow_regen | 32b | ce | tv | noro | none | pow08"
    "q32_q06_cetv_xwdec_regen | 32b | ce | tv | noro | none | dec"
    "q32_q06_cetv_xwinc_regen | 32b | ce | tv | noro | none | inc"

    # ─── Part 2: aux weight sweep — 8B + 0.6B ──────────────────────────
    # 8B + 0.6B — KL+V2 × 4
    "q8_q06_klv2_xwuni_regen  | 8b  | kl | v2 | noro | none | uniform"
    "q8_q06_klv2_xwpow_regen  | 8b  | kl | v2 | noro | none | pow08"
    "q8_q06_klv2_xwdec_regen  | 8b  | kl | v2 | noro | none | dec"
    "q8_q06_klv2_xwinc_regen  | 8b  | kl | v2 | noro | none | inc"
    # 8B + 0.6B — KL+V4 × 4
    "q8_q06_klv4_xwuni_regen  | 8b  | kl | v4 | noro | none | uniform"
    "q8_q06_klv4_xwpow_regen  | 8b  | kl | v4 | noro | none | pow08"
    "q8_q06_klv4_xwdec_regen  | 8b  | kl | v4 | noro | none | dec"
    "q8_q06_klv4_xwinc_regen  | 8b  | kl | v4 | noro | none | inc"
    # 8B + 0.6B — KL+V5 × 4
    "q8_q06_klv5_xwuni_regen  | 8b  | kl | v5 | noro | none | uniform"
    "q8_q06_klv5_xwpow_regen  | 8b  | kl | v5 | noro | none | pow08"
    "q8_q06_klv5_xwdec_regen  | 8b  | kl | v5 | noro | none | dec"
    "q8_q06_klv5_xwinc_regen  | 8b  | kl | v5 | noro | none | inc"
    # 8B + 0.6B — KL+TV × 4
    "q8_q06_kltv_xwuni_regen  | 8b  | kl | tv | noro | none | uniform"
    "q8_q06_kltv_xwpow_regen  | 8b  | kl | tv | noro | none | pow08"
    "q8_q06_kltv_xwdec_regen  | 8b  | kl | tv | noro | none | dec"
    "q8_q06_kltv_xwinc_regen  | 8b  | kl | tv | noro | none | inc"
    # 8B + 0.6B — CE+V2 × 4
    "q8_q06_cev2_xwuni_regen  | 8b  | ce | v2 | noro | none | uniform"
    "q8_q06_cev2_xwpow_regen  | 8b  | ce | v2 | noro | none | pow08"
    "q8_q06_cev2_xwdec_regen  | 8b  | ce | v2 | noro | none | dec"
    "q8_q06_cev2_xwinc_regen  | 8b  | ce | v2 | noro | none | inc"
    # 8B + 0.6B — CE+V4 × 4
    "q8_q06_cev4_xwuni_regen  | 8b  | ce | v4 | noro | none | uniform"
    "q8_q06_cev4_xwpow_regen  | 8b  | ce | v4 | noro | none | pow08"
    "q8_q06_cev4_xwdec_regen  | 8b  | ce | v4 | noro | none | dec"
    "q8_q06_cev4_xwinc_regen  | 8b  | ce | v4 | noro | none | inc"
    # 8B + 0.6B — CE+V5 × 4
    "q8_q06_cev5_xwuni_regen  | 8b  | ce | v5 | noro | none | uniform"
    "q8_q06_cev5_xwpow_regen  | 8b  | ce | v5 | noro | none | pow08"
    "q8_q06_cev5_xwdec_regen  | 8b  | ce | v5 | noro | none | dec"
    "q8_q06_cev5_xwinc_regen  | 8b  | ce | v5 | noro | none | inc"
    # 8B + 0.6B — CE+TV × 4
    "q8_q06_cetv_xwuni_regen  | 8b  | ce | tv | noro | none | uniform"
    "q8_q06_cetv_xwpow_regen  | 8b  | ce | tv | noro | none | pow08"
    "q8_q06_cetv_xwdec_regen  | 8b  | ce | tv | noro | none | dec"
    "q8_q06_cetv_xwinc_regen  | 8b  | ce | tv | noro | none | inc"

    # ═══════════════════════════════════════════════════════════════════════
    # On-policy rollout (ropo) — target re-forwards on draft's dirty chain
    # to get dirty-context target_argmax for β_k (k ≥ 1). Fixes context mismatch.
    # ═══════════════════════════════════════════════════════════════════════

    # 8B + 0.6B
    "q8_q06_klv2_regen_ropo  | 8b  | kl | v2   | ropo"
    "q8_q06_cev2_regen_ropo  | 8b  | ce | v2   | ropo"
    "q8_q06_klv4_regen_ropo  | 8b  | kl | v4   | ropo"
    "q8_q06_cev4_regen_ropo  | 8b  | ce | v4   | ropo"
    "q8_q06_kltv_regen_ropo  | 8b  | kl | tv   | ropo"
    "q8_q06_cetv_regen_ropo  | 8b  | ce | tv   | ropo"
    #   KL only / CE only — ropo = rollout anchor with dirty-context target
    "q8_q06_kl_regen_ropo    | 8b  | kl | none | ropo"
    "q8_q06_ce_regen_ropo    | 8b  | ce | none | ropo"

    # ─── l2k soro — MAX_LEN=2048, MAX_SAMPLES=60000 ──────────────────
    "q8_q06_klv4_regen_soro_l2k | 8b  | kl | v4   | soro"
    "q8_q06_cev4_regen_soro_l2k | 8b  | ce | v4   | soro"
    "q8_q06_klv5_regen_soro_l2k | 8b  | kl | v5   | soro"
    "q8_q06_cev5_regen_soro_l2k | 8b  | ce | v5   | soro"
    # kl/ce soro use rollout-anchor mode (per-step anchor on soft-rollout context)
    "q8_q06_kl_regen_soro_l2k   | 8b  | kl | none | soro"
    "q8_q06_ce_regen_soro_l2k   | 8b  | ce | none | soro"
    "q8_q06_kltv_regen_soro_l2k | 8b  | kl | tv   | soro"
    "q8_q06_cetv_regen_soro_l2k | 8b  | ce | tv   | soro"

    # 32B + 0.6B
    "q32_q06_klv2_regen_ropo | 32b | kl | v2   | ropo"
    "q32_q06_cev2_regen_ropo | 32b | ce | v2   | ropo"
    "q32_q06_klv4_regen_ropo | 32b | kl | v4   | ropo"
    "q32_q06_cev4_regen_ropo | 32b | ce | v4   | ropo"
    "q32_q06_kltv_regen_ropo | 32b | kl | tv   | ropo"
    "q32_q06_cetv_regen_ropo | 32b | ce | tv   | ropo"
    #   KL only / CE only — ropo
    "q32_q06_kl_regen_ropo   | 32b | kl | none | ropo"
    "q32_q06_ce_regen_ropo   | 32b | ce | none | ropo"

    # ═══════════════════════════════════════════════════════════════════════
    # Soft rollout (soro) — draft feeds softmax-weighted expected embedding
    # between rollout steps; target stays CLEAN (no verify). V5 aux truncates
    # cum_prod sum at first hard reject in the rollout chain.
    # ═══════════════════════════════════════════════════════════════════════

    # 8B + 0.6B
    "q8_q06_klv5_regen_soro  | 8b  | kl | v5   | soro"
    "q8_q06_cev5_regen_soro  | 8b  | ce | v5   | soro"
    "q8_q06_klv4_regen_soro  | 8b  | kl | v4   | soro"
    "q8_q06_cev4_regen_soro  | 8b  | ce | v4   | soro"
    "q8_q06_klv2_regen_soro  | 8b  | kl | v2   | soro"
    "q8_q06_cev2_regen_soro  | 8b  | ce | v2   | soro"
    "q8_q06_kl_regen_soro    | 8b  | kl | none | soro"
    "q8_q06_ce_regen_soro    | 8b  | ce | none | soro"

    # 32B + 0.6B
    "q32_q06_klv5_regen_soro | 32b | kl | v5   | soro"
    "q32_q06_cev5_regen_soro | 32b | ce | v5   | soro"
    "q32_q06_klv4_regen_soro | 32b | kl | v4   | soro"
    "q32_q06_cev4_regen_soro | 32b | ce | v4   | soro"
    "q32_q06_klv2_regen_soro | 32b | kl | v2   | soro"
    "q32_q06_cev2_regen_soro | 32b | ce | v2   | soro"
    "q32_q06_kl_regen_soro   | 32b | kl | none | soro"
    "q32_q06_ce_regen_soro   | 32b | ce | none | soro"
)

# ═══════════════════════════════════════════════════════════════════════════
# Ropo weight-sweep (generated): anchor_weight × aux_weight ablation
#
# For each (target, anchor, aw):
#   1 task with aux=none  (rolled-out anchor; only anchor_weight matters)
#   For each aux ∈ {v2, v4, tv}:
#     4 tasks with different aux_weight
#
# Tag: q<T>_q06_<anchor>[<aux>]_aw<AW>_xw<XW>_regen_ropo
#      AW/XW are 3-char short names (uni/pow/dec/inc)
#
# Total = 2 targets × 2 anchors × 4 aw × (1 + 3 × 4) = 208 tasks
# ═══════════════════════════════════════════════════════════════════════════
for _target in 8b 32b; do
    _tshort="${_target%b}"   # 8b→8, 32b→32 (match existing tag prefix)
    for _anchor in kl ce; do
        for _aw in uniform pow08 dec inc; do
            _awshort="${_aw:0:3}"
            # aux=none: rolled-out anchor, only anchor_weight varies
            TASKS+=("q${_tshort}_q06_${_anchor}_aw${_awshort}_regen_ropo | ${_target} | ${_anchor} | none | ropo | ${_aw} | pow08")
            # aux variants × aux_weight
            for _aux in v2 v4 tv; do
                for _xw in uniform pow08 dec inc; do
                    _xwshort="${_xw:0:3}"
                    TASKS+=("q${_tshort}_q06_${_anchor}${_aux}_aw${_awshort}_xw${_xwshort}_regen_ropo | ${_target} | ${_anchor} | ${_aux} | ropo | ${_aw} | ${_xw}")
                done
            done
        done
    done
done
unset _target _tshort _anchor _aw _aux _xw _awshort _xwshort

# ── --list: show all tasks ─────────────────────────────────────────────────
if [ "$1" = "--list" ]; then
    echo "Regen ShareGPT training tasks:"
    for task in "${TASKS[@]}"; do
        IFS='|' read -r tag tgt anchor aux rollout aw xw coef grpo_coef init_draft <<< "$task"
        tag=$(echo "$tag" | xargs); tgt=$(echo "$tgt" | xargs)
        anchor=$(echo "$anchor" | xargs); aux=$(echo "$aux" | xargs)
        rollout=$(echo "$rollout" | xargs)
        aw=$(echo "$aw" | xargs); xw=$(echo "$xw" | xargs)
        coef=$(echo "${coef:-}" | xargs)
        grpo_coef=$(echo "${grpo_coef:-}" | xargs)
        init_draft=$(echo "${init_draft:-}" | xargs)
        aw=${aw:-none}; xw=${xw:-pow08}; coef=${coef:-0.1}
        grpo_coef=${grpo_coef:-0.0}
        init_short=${init_draft##*/}
        init_short=${init_short:-base}
        printf "  %-32s tgt=%-4s anchor=%-3s aux=%-4s ro=%-4s aw=%-7s xw=%-7s coef=%s grpo=%s init=%s\n" \
            "$tag" "$tgt" "$anchor" "$aux" "$rollout" "$aw" "$xw" "$coef" "$grpo_coef" "$init_short"
    done
    exit 0
fi

# ── Filters ────────────────────────────────────────────────────────────────
# Basic: --klv2 / --v4 / --klv4 match loss combos.
# Weight ablation: --aw matches anchor-weight-sweep tasks (aux=none, aw≠none).
#                  --xw matches aux-weight-sweep tasks (aw=none, aux≠none, xw≠pow08 default).
#                  --weights matches BOTH weight-sweep sets.
FILTER=""
if [ "$1" = "--klv2" ]; then
    FILTER="klv2"; shift
elif [ "$1" = "--v4" ]; then
    FILTER="v4"; shift
elif [ "$1" = "--klv4" ]; then
    FILTER="klv4"; shift
elif [ "$1" = "--aw" ]; then
    FILTER="aw"; shift
elif [ "$1" = "--xw" ]; then
    FILTER="xw"; shift
elif [ "$1" = "--weights" ]; then
    FILTER="weights"; shift
elif [ "$1" = "--ropo" ]; then
    FILTER="ropo"; shift
elif [ "$1" = "--ropow" ]; then
    FILTER="ropow"; shift
elif [ "$1" = "--soro" ]; then
    FILTER="soro"; shift
fi

# ── submit helper ──────────────────────────────────────────────────────────
submit() {
    local tag=$1 tgt=$2 anchor=$3 aux=$4 rollout=$5 aw=$6 xw=$7 coef=$8
    local grpo_coef_col=$9 init_draft_col=${10}
    aw=${aw:-none}
    xw=${xw:-pow08}
    # coef: task col 8 > env SIGMOID_COEF > default 0.1
    coef=${coef:-$SIGMOID_COEF}
    # grpo_coef: task col 9 > env GRPO_COEF > default 0.0
    local GRPO_COEF_LOCAL=${grpo_coef_col:-${GRPO_COEF:-0.0}}

    # Resolve target model + target-matched regen data
    if [ "$tgt" = "8b" ]; then
        BASEPATH=Qwen/Qwen3-8B
        TRAINPATH=$TRAIN_REGEN_8B
    else
        BASEPATH=Qwen/Qwen3-32B
        TRAINPATH=$TRAIN_REGEN_32B
    fi
    # DRAFTPATH: task col 10 (init_draft) > env INIT_DRAFT_PATH > HF base
    if [ -n "$init_draft_col" ]; then
        DRAFTPATH="$init_draft_col"
    else
        DRAFTPATH=${INIT_DRAFT_PATH:-Qwen/Qwen3-0.6B}
    fi

    # Resolve slurm + deepspeed config + save root + on-policy / soft flag
    ONPOLICY=0
    SOFTRO=0
    if [ "$rollout" = "ro" ] || [ "$rollout" = "ropo" ] || [ "$rollout" = "soro" ]; then
        SCRIPT=$SLURM_RO
        DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG_OVERRIDE:-$DS_CONFIG_ZERO3}"
        SAVEROOT=/scratch/tx856/spec_reason/scratch/loss_train_smalllm_rollout
        LOGDIR=logs/smalllm_rollout
        [ "$rollout" = "ropo" ] && ONPOLICY=1
        [ "$rollout" = "soro" ] && SOFTRO=1
    else
        SCRIPT=$SLURM_NORO
        if [ "$tgt" = "32b" ]; then
            DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG_OVERRIDE:-$DS_CONFIG_ZERO3}"
            SAVEROOT=/scratch/tx856/spec_reason/scratch/loss_train_smalllm_q32_q06
            LOGDIR=logs/sdpo_smalllm_q32_q06
        else
            DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG_OVERRIDE:-$DS_CONFIG_8B_NORO}"
            SAVEROOT=/scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b
            LOGDIR=logs/sdpo_smalllm_06b
        fi
    fi

    # Submit training
    JOB_ID=$(sbatch \
        --job-name="$tag" \
        --output="${LOGDIR}/train_${tag}_%j.out" \
        --error="${LOGDIR}/train_${tag}_%j.err" \
        --time="${SLURM_TRAIN_TIME:-1-00:00:00}" \
        ${SLURM_CONSTRAINT:+--constraint="$SLURM_CONSTRAINT"} \
        ${SLURM_DEPENDENCY:+--dependency="$SLURM_DEPENDENCY"} \
        --export=ALL,TAG="$tag",BASEPATH="$BASEPATH",DRAFTPATH="$DRAFTPATH",ANCHOR="$anchor",AUX_LOSS="$aux",ANCHOR_WEIGHT="$aw",AUX_WEIGHT="$xw",ON_POLICY_TARGET="$ONPOLICY",SOFT_ROLLOUT="$SOFTRO",SIGMOID_COEF="$coef",GRPO_COEF="$GRPO_COEF_LOCAL",GRPO_K_GROUPS="${GRPO_K_GROUPS:-8}",GRPO_M="${GRPO_M:-4}",GRPO_EPS="${GRPO_EPS:-0.2}",GRPO_MODE="${GRPO_MODE:-window}",GRPO_SAMPLE_TEMP="${GRPO_SAMPLE_TEMP:-1.0}",GRPO_REWARD="${GRPO_REWARD:-hard}",GAMMA="${GAMMA:-7}",ROLLOUT_TRUNCATE_AT_REJECT="${ROLLOUT_TRUNCATE_AT_REJECT:-0}",ROLLOUT_LOSS_MODE="${ROLLOUT_LOSS_MODE:-chain}",T_MAX="${T_MAX:-0.1}",T_MIN="${T_MIN:-0.1}",LR=1e-6,NUM_EPOCHS="$NUM_EPOCHS",MAX_SAMPLES="$MAX_SAMPLES",MAX_LEN="$MAX_LEN",TRAINPATH="$TRAINPATH",TESTPATH="$VAL_SMALL",DEEPSPEED_CONFIG="$DEEPSPEED_CONFIG",SAVEROOT="$SAVEROOT" \
        "$SCRIPT" | awk '{print $4}')

    printf "  train  %-28s (%-4s %s+%-4s %-4s aw=%-7s xw=%-7s) -> %s\n" \
        "$tag" "$tgt" "$anchor" "$aux" "$rollout" "$aw" "$xw" "$JOB_ID"

    # Auto-chain eval (unless NO_EVAL=1)
    # Eval the LAST checkpoint (state_<NUM_EPOCHS-1>) so longer runs eval their final state.
    if [ -z "$NO_EVAL" ]; then
        local LAST_STATE=$((NUM_EPOCHS - 1))
        local EVAL_TAG="${tag}_state${LAST_STATE}"
        local CKPT="$SAVEROOT/$tag/state_${LAST_STATE}"
        # BENCH + TOP_K contain commas — set as env vars of the sbatch process,
        # then --export=ALL forwards them. Do NOT put commas inside --export list.
        local EVAL_JOB_ID=$(BENCH="$EVAL_BENCH_VAL" TOP_K="$EVAL_TOP_K_VAL" \
            sbatch \
            --job-name="ev_${tag}" \
            --output="logs/smalllm_rollout/eval_${EVAL_TAG}_%j.out" \
            --error="logs/smalllm_rollout/eval_${EVAL_TAG}_%j.err" \
            --constraint="a100|h100|h200" \
            --mem=128G --time=0-08:00:00 \
            --dependency=afterok:$JOB_ID \
            --export=ALL,TAG="$EVAL_TAG",BASEPATH="$BASEPATH",DRAFT_CKPT="$CKPT",GAMMA="$EVAL_GAMMA",BUDGET="$EVAL_BUDGET",MAX_NEW_TOKENS="$EVAL_MAX_NEW_TOKENS",NUM_SAMPLES="$EVAL_NUM_SAMPLES",BASELINE_GAMMA="$EVAL_BASELINE_GAMMA",RUN_BASELINE=1,OUTPUT_DIR="$EVAL_OUTPUT_DIR" \
            "$SLURM_EVAL" | awk '{print $4}')
        printf "  eval   %-28s (deps on %s)               -> %s\n" \
            "$EVAL_TAG" "$JOB_ID" "$EVAL_JOB_ID"
    fi
}

# ── Selection logic ───────────────────────────────────────────────────────
SELECTED=("$@")

echo "Regen ShareGPT sweep (epochs=$NUM_EPOCHS, samples=${MAX_SAMPLES:-full}, max_len=$MAX_LEN, filter=${FILTER:-none}, eval=${NO_EVAL:+OFF}${NO_EVAL:-ON})"
echo "  8B  target -> $(basename $TRAIN_REGEN_8B)"
echo "  32B target -> $(basename $TRAIN_REGEN_32B)"
echo "==================================================================="

count=0
for task in "${TASKS[@]}"; do
    IFS='|' read -r tag tgt anchor aux rollout aw xw coef grpo_coef init_draft <<< "$task"
    tag=$(echo "$tag" | xargs); tgt=$(echo "$tgt" | xargs)
    anchor=$(echo "$anchor" | xargs); aux=$(echo "$aux" | xargs)
    rollout=$(echo "$rollout" | xargs)
    aw=$(echo "$aw" | xargs); xw=$(echo "$xw" | xargs)
    coef=$(echo "${coef:-}" | xargs)
    grpo_coef=$(echo "${grpo_coef:-}" | xargs)
    init_draft=$(echo "${init_draft:-}" | xargs)
    # Resolution order for aw/xw:
    #   1. Task's 6th/7th column (explicit per-task override)
    #   2. ENV var ANCHOR_WEIGHT / AUX_WEIGHT (caller-set default)
    #   3. Script default (none / pow08)
    aw=${aw:-${ANCHOR_WEIGHT:-none}}
    xw=${xw:-${AUX_WEIGHT:-pow08}}

    # Apply filters
    case "$FILTER" in
        klv2)  [ "$anchor" = "kl" ] && [ "$aux" = "v2" ]   || continue ;;
        v4)    [ "$aux" = "v4" ]                           || continue ;;
        klv4)  [ "$anchor" = "kl" ] && [ "$aux" = "v4" ]   || continue ;;
        aw)    [ "$aux" = "none" ] && [ "$aw" != "none" ]  || continue ;;
        xw)    [ "$aw" = "none" ] && [ "$aux" != "none" ] \
               && [[ "$tag" == *"_xw"* ]]                  || continue ;;
        weights)
               { [ "$aux" = "none" ] && [ "$aw" != "none" ]; } \
               || { [[ "$tag" == *"_xw"* ]] && [ "$aw" = "none" ]; } \
               || continue ;;
        ropo)  [ "$rollout" = "ropo" ]                     || continue ;;
        ropow) [ "$rollout" = "ropo" ] && [[ "$tag" == *"_aw"* ]] || continue ;;
        soro)  [ "$rollout" = "soro" ]                     || continue ;;
    esac

    if [ ${#SELECTED[@]} -gt 0 ]; then
        match=0
        for sel in "${SELECTED[@]}"; do
            [ "$sel" = "$tag" ] && { match=1; break; }
        done
        [ $match -eq 0 ] && continue
    fi

    submit "$tag" "$tgt" "$anchor" "$aux" "$rollout" "$aw" "$xw" "$coef" "$grpo_coef" "$init_draft"
    count=$((count + 1))
done

echo ""
echo "$count train jobs submitted (each with chained eval). Monitor: squeue -u \$USER"
