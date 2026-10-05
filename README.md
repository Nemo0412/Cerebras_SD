# SDPO — Speculative-Decoding Loss Design

Train a small draft model so that, paired with a frozen large target, speculative decoding maximises mean acceptance length τ. The framework uses a single training script (`sdpo/main_small_lm.py`) parameterised by `--anchor` × `--aux_loss` to cover all loss variants, plus an optional `--grpo_coef > 0` continuation phase.

## Quick map

| Pipeline | Target | Draft | Train + eval slurm | Default data |
|---|---|---|---|---|
| **Small-LM Qwen3** | `Qwen/Qwen3-{8B,32B}` | `Qwen/Qwen3-{0.6B,1.7B,4B}` | `sdpo/slurm/smalllm_06b_train.slurm` (train) + `sdpo/slurm/smalllm_tree_eval.slurm` (eval) | `data/sharegpt_qwen3_{8b,32b}_regen.jsonl` (target-regenerated 120K; `.gitignore`d) |
| **Small-LM Gemma 4** | `google/gemma-4-31B-it` | `google/gemma-4-E2B-it` | `sdpo/slurm/gemma4_train.slurm` (train **and** eval in one job) | `sdpo/data/sharegpt_gemma4_31b_regen.jsonl` |

## Loss menu (`--anchor` × `--aux_loss`)

| Loss combo | `--anchor` | `--aux_loss` | `SIGMOID_COEF` | Other notes |
|---|---|---|---|---|
| **KL only** | `kl` | `none` | `0.0` (+ `--baseline eagle_only`) | KL anchor only |
| **CE only** | `ce` | `none` | `0.0` | hard-target CE anchor only |
| **KL + V4** | `kl` | `v4` | `0.1` | β = 2σ(gap/T), 0.8^t decay, cumprod |
| **KL + TV** | `kl` | `tv` | `0.1` | per-step TV distance |
| **ALTV** (preferred AL-family) | `none` | `al_tv` | `1.0` | E[L] with β = Σ_z min(p_z, q_z); SpS-marginal accept rate |
| **AL-KL** | `none` | `al_kl` | `1.0` | E[L] with β = ½·exp(−KL); log-space cumsum |
| **WKL** | `none` | `wkl` | `1.0` | Σ_j (γ−j)·KL_j |

`--baseline eagle_only` is shorthand for `--anchor kl --aux_loss none`; `--baseline ce_only` is shorthand for `--anchor ce --aux_loss none`. Argparse rejects unknown `--aux_loss` values, so pure-anchor runs must still pass a valid choice (`none`).

GRPO is layered on top by setting `--grpo_coef > 0`. Standard recipe is **3 ep SFT → 3 ep GRPO**: stage 1 is plain SFT (no GRPO); stage 2 sets `DRAFTPATH=<stage-1 state_2/>` and turns GRPO on.

---

## Qwen3 Small-LM recipe book

All commands use `sbatch --export=ALL` with **pre-exported env vars** to avoid the comma-splitting bug (sbatch splits `--export=KEY=v1,v2,...` on commas, so multi-value env vars must be pre-exported in the shell).

Shared eval setup:

```bash
export BENCH="mt_bench,gsm8k,humaneval,longwriter_top50,osr2_top50"
export TOP_K="4,3,2,1,1,1,1"
export GAMMA=7 BUDGET=128 MAX_NEW_TOKENS=256 NUM_SAMPLES=40 EVAL_SEED=0
```

### A) Q0.6B draft + Q8B target

Default DeepSpeed: `sdpo/sdpo_config_qwen3.json` (ZeRO-2). GPU: `--constraint="a100|h100|h200"`.

#### A.1 Train KL / KLV4 / **ALTV** (6 epoch)

```bash
TRAINDATA=data/sharegpt_qwen3_8b_regen.jsonl

# KL: anchor=kl, aux=none
TAG=q8_q06_kl_regen_l2k_6ep
sbatch --export=ALL,TAG=$TAG,SEED=42,\
BASEPATH=Qwen/Qwen3-8B,DRAFTPATH=Qwen/Qwen3-0.6B,\
ANCHOR=kl,AUX_LOSS=none,SIGMOID_COEF=0.1,T_MAX=0.1,T_MIN=0.1,\
PROB_T_MAX=1.0,PROB_T_MIN=0.1,ANCHOR_WEIGHT=none,AUX_WEIGHT=pow08,\
NUM_EPOCHS=6,MAX_LEN=2048,MAX_SAMPLES=60000,LR=1e-6,\
TRAINPATH=$TRAINDATA  sdpo/slurm/smalllm_06b_train.slurm

# KLV4: anchor=kl, aux=v4 (logit-space sigmoid gap)
TAG=q8_q06_klv4_l2k_6ep_s0
sbatch --export=ALL,TAG=$TAG,SEED=0,\
BASEPATH=Qwen/Qwen3-8B,DRAFTPATH=Qwen/Qwen3-0.6B,\
ANCHOR=kl,AUX_LOSS=v4,SIGMOID_COEF=0.1,T_MAX=0.1,T_MIN=0.1,\
PROB_T_MAX=1.0,PROB_T_MIN=0.1,ANCHOR_WEIGHT=none,AUX_WEIGHT=pow08,\
NUM_EPOCHS=6,MAX_LEN=2048,MAX_SAMPLES=60000,LR=1e-6,\
TRAINPATH=$TRAINDATA  sdpo/slurm/smalllm_06b_train.slurm

# ALTV (al_tv): anchor=none, aux=al_tv, coef=1.0 — the AL-family canonical
TAG=q8_q06_altv_only_regen_l2k_6ep
sbatch --export=ALL,TAG=$TAG,\
BASEPATH=Qwen/Qwen3-8B,DRAFTPATH=Qwen/Qwen3-0.6B,\
ANCHOR=none,AUX_LOSS=al_tv,SIGMOID_COEF=1.0,T_MAX=0.1,T_MIN=0.1,\
PROB_T_MAX=1.0,PROB_T_MIN=0.1,ANCHOR_WEIGHT=none,AUX_WEIGHT=pow08,\
NUM_EPOCHS=6,MAX_LEN=2048,MAX_SAMPLES=60000,LR=1e-6,\
TRAINPATH=$TRAINDATA  sdpo/slurm/smalllm_06b_train.slurm
```

#### A.2 KLV4 + GRPO continuation (3 epoch SFT → 3 epoch GRPO)

```bash
# Stage 1: KLV4 SFT 3 epoch (just NUM_EPOCHS=3 of the recipe above)
# Stage 2: GRPO continuation from state_2 of stage 1
SRC=/scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/q8_q06_klv4_regen_l2k/state_2
TAG=q8_q06_klv4_grpo_smp_s0
sbatch --export=ALL,TAG=$TAG,SEED=0,\
BASEPATH=Qwen/Qwen3-8B,DRAFTPATH=$SRC,\
ANCHOR=kl,AUX_LOSS=v4,SIGMOID_COEF=0.1,T_MAX=0.1,T_MIN=0.1,\
PROB_T_MAX=1.0,PROB_T_MIN=1.0,ANCHOR_WEIGHT=none,AUX_WEIGHT=pow08,\
GRPO_COEF=0.1,GRPO_K_GROUPS=8,GRPO_M=4,GRPO_EPS=0.2,\
GRPO_MODE=sample,GRPO_SAMPLE_TEMP=1.0,GRPO_REWARD=hard,\
NUM_EPOCHS=3,MAX_LEN=2048,MAX_SAMPLES=60000,LR=1e-6,\
TRAINPATH=data/sharegpt_qwen3_8b_regen.jsonl \
  sdpo/slurm/smalllm_06b_train.slurm
```

Note: `DRAFTPATH=<stage-1 state_2>` must point to a **prefix-stripped** ckpt. A prior eval pass on the stage-1 ckpt strips `draft_model.` automatically; if you skip that, `from_pretrained` silently loads random weights and stage 2 collapses (eagle_loss ~ 8.5, mean_tau ~ 0.14).

#### A.3 Eval (Q0.6B): greedy + sample-ratio T=1

```bash
TAG=q8_q06_altv_only_regen_l2k_6ep                  # any train TAG
CKPT=/scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/$TAG/state_5

# Greedy
export TAG_OUT=${TAG}_greedy BASEPATH=Qwen/Qwen3-8B DRAFT_CKPT=$CKPT \
       TEMP=0.0 DRAFT_MODE=argmax VERIFY_MODE=auto
sbatch --constraint="h100|h200" --export=ALL \
       --export=TAG=$TAG_OUT,BASEPATH,DRAFT_CKPT,TEMP,DRAFT_MODE,VERIFY_MODE,BENCH,TOP_K,GAMMA,BUDGET,MAX_NEW_TOKENS,NUM_SAMPLES \
       sdpo/slurm/smalllm_tree_eval.slurm

# Sample-ratio T=1 (multinomial draft + r ≤ p_target/p_draft verify — unbiased SpS)
export TAG_OUT=${TAG}_samp_t1_ratio TEMP=1.0 DRAFT_MODE=sample VERIFY_MODE=ratio
sbatch --constraint="h100|h200" --export=ALL \
       --export=TAG=$TAG_OUT,BASEPATH,DRAFT_CKPT,TEMP,DRAFT_MODE,VERIFY_MODE,BENCH,TOP_K,GAMMA,BUDGET,MAX_NEW_TOKENS,NUM_SAMPLES \
       sdpo/slurm/smalllm_tree_eval.slurm
```

Wider tree (γ=12) for ALTV-strength evaluation:

```bash
sbatch -J altv_g12_q06 \
       --export=ALL,TAG=q8_q06_altv_only_regen_l2k_6ep,DRAFT_CKPT=$CKPT,BASEPATH=Qwen/Qwen3-8B \
       sdpo/slurm/altv_g12_budget_sweep.slurm
```

This runs 5 benches × 3 budgets {128, 256, 512} × 2 modes (greedy + samp_t1_ratio) in one job with `--gamma 12 --top-k 5,4,3,2,1,1,1,1,1,1,1,1`.

### B) Q1.7B draft + Q8B target

Same target + train data as A; only `DRAFTPATH=Qwen/Qwen3-1.7B` and the tag prefix `q17_q06_*`.

```bash
# Q1.7B ALTV
TAG=q17_q06_altv_only_l2k_6ep
sbatch --export=ALL,TAG=$TAG,SEED=42,\
BASEPATH=Qwen/Qwen3-8B,DRAFTPATH=Qwen/Qwen3-1.7B,\
ANCHOR=none,AUX_LOSS=al_tv,SIGMOID_COEF=1.0,T_MAX=0.1,T_MIN=0.1,\
PROB_T_MAX=1.0,PROB_T_MIN=0.1,ANCHOR_WEIGHT=none,AUX_WEIGHT=pow08,\
NUM_EPOCHS=6,MAX_LEN=2048,MAX_SAMPLES=60000,LR=1e-6,\
TRAINPATH=data/sharegpt_qwen3_8b_regen.jsonl \
  sdpo/slurm/smalllm_06b_train.slurm

# Q1.7B KL / KLV4 / KLV4+GRPO — same as A but swap DRAFTPATH, AUX_LOSS, etc.
```

Eval same as A.3 with `q17_q06_*` tag (single GPU fits Q8B target + Q1.7B draft ≈ 20 GB).

### C) Q4B draft + Q32B target

Train data: `data/sharegpt_qwen3_32b_regen.jsonl`.
Required: `--constraint=h200` (2× H200 needed for 32B target).
DeepSpeed config: `sdpo/sdpo_config_qwen3_bf16_zero3_2gpu.json` (ZeRO-3).
Save root: `/scratch/tx856/spec_reason/scratch/loss_train_smalllm_q32_q4`.

```bash
# Q4B ALTV
TAG=q32_q4_altv_l2k_6ep_s0
sbatch --constraint=h200 \
  --export=ALL,TAG=$TAG,SEED=0,SAVE_EPOCHS=2,\
BASEPATH=Qwen/Qwen3-32B,DRAFTPATH=Qwen/Qwen3-4B,\
ANCHOR=none,AUX_LOSS=al_tv,SIGMOID_COEF=1.0,T_MAX=0.1,T_MIN=0.1,\
PROB_T_MAX=1.0,PROB_T_MIN=0.1,ANCHOR_WEIGHT=none,AUX_WEIGHT=pow08,\
NUM_EPOCHS=6,MAX_LEN=2048,MAX_SAMPLES=60000,LR=1e-6,\
DEEPSPEED_CONFIG=sdpo/sdpo_config_qwen3_bf16_zero3_2gpu.json,\
SAVEROOT=/scratch/tx856/spec_reason/scratch/loss_train_smalllm_q32_q4,\
TRAINPATH=data/sharegpt_qwen3_32b_regen.jsonl \
  sdpo/slurm/smalllm_06b_train.slurm
```

Eval same as A.3 with `BASEPATH=Qwen/Qwen3-32B`, `CKPT_ROOT=.../loss_train_smalllm_q32_q4`. `--constraint="h100|h200"` is fine for eval (Q32B target ≈ 64 GB fits 1× H100 80 GB tightly; H200 143 GB is safer).

### Eval modes — semantics

| `--verify-mode` | accept rule | unbiased? |
|---|---|---|
| `greedy` (T=0) | argmax(target) == drafted | strict deterministic |
| `simple` | r ≤ p_target(drafted) | **biased** (ignores draft probability) |
| `ratio` | r ≤ p_target(drafted) / p_draft(drafted) | unbiased SpS (Leviathan 2023) — recommended for sampling |

### D) `SAVE_EPOCHS` — choosing which checkpoints survive

```bash
SAVE_EPOCHS=""     # default: save every epoch
SAVE_EPOCHS=2      # save only state_2
SAVE_EPOCHS="2,5"  # save state_2 + state_5 — MUST pre-export the variable;
                   # sbatch's --export=KEY=2,5 splits on the comma.
```

---

## Reproduce ALTV sample-ratio T=1 (foreground, no SLURM)

The best ALTV checkpoints we ship are below. Each runs in one foreground process on a single GPU (large target = 1× H200 143 GB recommended; Q8B target also fits on 1× H100 80 GB). Tree config: **γ=7, top_k=[4,3,2,1,1,1,1], budget=128** — the standard SpS tree that the rest of the codebase uses; pair this with `--temperature 1.0 --draft-mode sample --verify-mode ratio` for unbiased sample-ratio decoding. Thinking mode is **on** by default (Qwen3 chat template `enable_thinking=True`), matching the reported numbers.

### Checkpoint download

The three best ALTV ckpts (gzipped) are mirrored on Google Drive:
**https://drive.google.com/drive/folders/1GamLIbjxISu2AFAj4a9FmzI18masF26K?usp=sharing**

| File | Size | Setup |
|---|---|---|
| `q8_q06_altv_state5.tar.gz` | 1.1 GB | Q0.6B draft + Q8B target |
| `q17_q06_altv_state5.tar.gz` | 3.0 GB | Q1.7B draft + Q8B target |
| `q32_q4_altv_s0_state2.tar.gz` | 6.1 GB | Q4B draft + Q32B target |

Extract anywhere and point `--draft-model-path` at the inner `state_{2,5}` directory:

```bash
tar xzf q8_q06_altv_state5.tar.gz
# → q8_q06_altv_only_regen_l2k_6ep/state_5/{config.json, pytorch_model.bin}
# Use:
#   --draft-model-path ./q8_q06_altv_only_regen_l2k_6ep/state_5
```

### Setup (one-time)

```bash
conda activate longreason
export HF_HOME=/scratch/tx856/.huggingface     # or your own HF cache
cd <repo root>
```

### Q0.6B draft + Q8B target  (mt+gsm+he avg α ≈ 5.901 over 5 eval seeds)

```bash
CKPT=/scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/q8_q06_altv_only_regen_l2k_6ep/state_5

CUDA_VISIBLE_DEVICES=0 python sdpo/eval_small_lm_tree.py \
    --base-model-path Qwen/Qwen3-8B \
    --draft-model-path $CKPT \
    --bench-name mt_bench,gsm8k,humaneval \
    --gamma 7 --top-k 4,3,2,1,1,1,1 --budget 128 \
    --max-new-tokens 256 --num-samples 40 \
    --temperature 1.0 --draft-mode sample --verify-mode ratio \
    --seed 0 \
    --tag q8_q06_altv_repro \
    --output-dir smalllm_tree_eval_results
```

### Q1.7B draft + Q8B target  (mt+gsm+he avg α ≈ 6.489 over 5 eval seeds)

```bash
CKPT=/scratch/tx856/spec_reason/scratch/loss_train_smalllm_06b/q17_q06_altv_only_l2k_6ep/state_5

CUDA_VISIBLE_DEVICES=0 python sdpo/eval_small_lm_tree.py \
    --base-model-path Qwen/Qwen3-8B \
    --draft-model-path $CKPT \
    --bench-name mt_bench,gsm8k,humaneval \
    --gamma 7 --top-k 4,3,2,1,1,1,1 --budget 128 \
    --max-new-tokens 256 --num-samples 40 \
    --temperature 1.0 --draft-mode sample --verify-mode ratio \
    --seed 0 \
    --tag q17_q06_altv_repro \
    --output-dir smalllm_tree_eval_results
```

### Q4B draft + Q32B target  (mt+gsm+he avg α ≈ 6.768 over 3 train seeds s0/s1/s2)

```bash
CKPT=/scratch/tx856/spec_reason/scratch/loss_train_smalllm_q32_q4/q32_q4_altv_l2k_6ep_s0/state_2

CUDA_VISIBLE_DEVICES=0 python sdpo/eval_small_lm_tree.py \
    --base-model-path Qwen/Qwen3-32B \
    --draft-model-path $CKPT \
    --bench-name mt_bench,gsm8k,humaneval \
    --gamma 7 --top-k 4,3,2,1,1,1,1 --budget 128 \
    --max-new-tokens 256 --num-samples 40 \
    --temperature 1.0 --draft-mode sample --verify-mode ratio \
    --seed 0 \
    --tag q32_q4_altv_repro \
    --output-dir smalllm_tree_eval_results
```

Notes:
- Q32B target fp16 ≈ 64 GB; with Q4B draft (~8 GB) and KV cache + tree activations, comfortable on H200 143 GB and tight on H100 80 GB.
- Single-seed eval reproduces ~one row of the seed-averaged headline. Sweep `--seed 0..4` and average to recover the multi-seed avg.
- Q4B headline 6.768 is the mean over the three trained seeds (`q32_q4_altv_l2k_6ep_s{0,1,2}/state_2`); the command above uses s0. Swap `_s0` → `_s1` / `_s2` to evaluate the other two.
- Tags above include `_repro` to avoid overwriting any existing eval JSON in `smalllm_tree_eval_results/`.

Per-bench JSON appears at `smalllm_tree_eval_results/<TAG>.json`; the `mean_alpha` field under `results.<bench>.tree` is the headline α.

---

## Gemma 4 recipe (E2B draft + 31B-IT target)

### Train + eval (one slurm job)

`sdpo/slurm/gemma4_train.slurm` runs deepspeed training, then on success runs `eval_gemma4_chain.py` greedy + sample-ratio T=1 on five benches.

```bash
SAVEROOT=/scratch/tx856/spec_reason/scratch/loss_train_gemma4
TRAINDATA=sdpo/data/sharegpt_gemma4_31b_regen.jsonl

# KL only (eagle_only baseline)
sbatch -J g4_kl_6ep --export=ALL,\
TAG=g4_e2b_kl_regendata_6ep,BASEPATH=google/gemma-4-31B-it,DRAFTPATH=google/gemma-4-E2B-it,\
AUX_LOSS=none,SIGMOID_COEF=0.0,T_MAX=0.1,T_MIN=0.1,\
NUM_EPOCHS=6,MAX_SAMPLES=60000,TRAINPATH=$TRAINDATA,\
BASELINE=eagle_only,MAX_LEN=2048,LR=1e-6,SAVEROOT=$SAVEROOT,SAVE_EPOCHS=2 \
  sdpo/slurm/gemma4_train.slurm

# KLV4
sbatch -J g4_klv4_6ep --export=ALL,\
TAG=g4_e2b_klv4_regendata_6ep,BASEPATH=google/gemma-4-31B-it,DRAFTPATH=google/gemma-4-E2B-it,\
AUX_LOSS=acceptance_length_v4,SIGMOID_COEF=0.1,T_MAX=0.1,T_MIN=0.1,\
NUM_EPOCHS=6,MAX_SAMPLES=60000,TRAINPATH=$TRAINDATA,\
MAX_LEN=2048,LR=1e-6,SAVEROOT=$SAVEROOT,SAVE_EPOCHS=2 \
  sdpo/slurm/gemma4_train.slurm

# KL + ALTV (al_tv as aux, KL anchor)
sbatch -J g4_kl_altv_6ep --export=ALL,\
TAG=g4_e2b_kl_altv_regendata_6ep,BASEPATH=google/gemma-4-31B-it,DRAFTPATH=google/gemma-4-E2B-it,\
AUX_LOSS=al_tv,SIGMOID_COEF=0.1,T_MAX=0.1,T_MIN=0.1,\
NUM_EPOCHS=6,MAX_SAMPLES=60000,TRAINPATH=$TRAINDATA,\
MAX_LEN=2048,LR=1e-6,SAVEROOT=$SAVEROOT,SAVE_EPOCHS=2 \
  sdpo/slurm/gemma4_train.slurm

# KLV4 3-ep SFT seed for GRPO
JID_SFT=$(sbatch --parsable -J g4_klv4_3ep_sft --export=ALL,\
TAG=g4_e2b_klv4_3ep_sft_regendata,BASEPATH=google/gemma-4-31B-it,DRAFTPATH=google/gemma-4-E2B-it,\
AUX_LOSS=acceptance_length_v4,SIGMOID_COEF=0.1,T_MAX=0.1,T_MIN=0.1,\
NUM_EPOCHS=3,MAX_SAMPLES=60000,TRAINPATH=$TRAINDATA,\
MAX_LEN=2048,LR=1e-6,SAVEROOT=$SAVEROOT \
  sdpo/slurm/gemma4_train.slurm)

# KLV4 GRPO continuation (depends on the SFT job)
sbatch --dependency=afterok:$JID_SFT -J g4_klv4_grpo_smp --export=ALL,\
TAG=g4_e2b_klv4_grpo_smp_regendata,BASEPATH=google/gemma-4-31B-it,\
DRAFTPATH=$SAVEROOT/g4_e2b_klv4_3ep_sft_regendata/state_2,\
AUX_LOSS=acceptance_length_v4,SIGMOID_COEF=0.1,T_MAX=0.1,T_MIN=0.1,\
NUM_EPOCHS=3,MAX_SAMPLES=60000,TRAINPATH=$TRAINDATA,\
MAX_LEN=2048,LR=1e-6,SAVEROOT=$SAVEROOT,\
GRPO_COEF=0.1,GRPO_MODE=sample,GRPO_REWARD=hard \
  sdpo/slurm/gemma4_train.slurm
```

Key Gemma-4-specific bits (set by `sdpo_config_gemma4_bf16_zero2_offload.json` and the loader):
- ZeRO-2 (not ZeRO-3) because Gemma 4 has **tied frozen target embeddings**; ZeRO-3 mid-step gather returns a stale shard the second time and `mean_tau` collapses to 0.
- No CPU optim offload — H200 143 GB fits both models + GPU AdamW (~80 GB / rank); the offload path needs a JIT C++ build that breaks on CUDA 13 ↔ torch CUDA 12.8 mismatch.
- Causal-form weights: the 31B + E2B checkpoints ship as `Gemma4ForConditionalGeneration` with `model.language_model.*` prefix. `_load_lm` in `small_lm_model.py` remaps to `Gemma4ForCausalLM` once and caches under `/scratch/tx856/.cache/gemma4_causal/<safe_model_id>/`.

### Regen data

Regenerate target responses on `humaneval → gsm8k → osr2 → longwriter → sharegpt` in one slurm job:

```bash
sbatch sdpo/slurm/regen_all_gemma4.slurm
```

Outputs `sdpo/data/{humaneval,gsm8k,osr2,longwriter,sharegpt}_gemma4_31b_regen.jsonl`. vLLM/torch/triton caches are pre-redirected to `/scratch/tx856/.cache/{vllm,torchinductor,triton,torch_extensions}` to avoid the home-quota exhaustion that killed the original run.

---

## Q32B target × Q0.6B draft — full-vocab / target-topK train + topK verify eval

Concrete commands for the 53×-ratio setup used in the ALTV loss-variant sweep. Every training run is 2 epochs on 60K samples of `sharegpt_qwen3_32b_regen.jsonl` with `--anchor none --aux_weight uniform`, ZeRO-3, 2× H100/H200. Every eval is `max_new=1024, num_samples=40, 3 seeds` with sample-ratio verify (`--verify-mode ratio`) and both draft-topK sampling and target-topK verify at `K=20` — see the `--draft-top-k` / `--target-top-k` args wired through `eval_small_lm_tree.py` and `tree_spec_decode.py`.

### Loss variants (all `--aux_loss <name> --anchor none --aux_weight uniform`)

| `--aux_loss` | β computation | β on draft |
|---|---|---|
| `al_tv` | `β = Σ min(target_p, draft_q)` over **full vocab** | full vocab, unnormalized |
| `al_tv_target_topk` (V2) | β on target's top-K only (target renormalized within K) | draft raw at target's top-K indices |
| `al_tv_target_topp` | dynamic K = min r s.t. Σ p_r ≥ P | draft raw at same nucleus |
| `al_tv_target_topk_topp` | union of top-K and top-P nuclei | draft raw at union |
| `al_tv_renewal` | full vocab β, anchor weighted by ρ̄_t forward recursion | full vocab |
| `al_tv_target_topk_renewal` | top-K β **and** ρ̄_t from that β | draft raw at top-K |

Set K with `--altv_topk N`, P with `--altv_topp F`. When both are set, `al_tv_target_topk_topp` takes their union per position.

### Training — raw deepspeed

```bash
BASEPATH=Qwen/Qwen3-32B
DRAFTPATH=Qwen/Qwen3-0.6B
TRAINPATH=data/sharegpt_qwen3_32b_regen.jsonl
TAG=q32_q06_altv_baseline_2ep                   # or _target_topk20_2ep, etc.
SAVEROOT=/scratch/tx856/spec_reason/scratch/loss_train_smalllm_q32_q06

# Full-vocab ALTV baseline
deepspeed --master_port 29500 sdpo/main_small_lm.py \
    --basepath $BASEPATH --draftpath $DRAFTPATH \
    --trainpath $TRAINPATH --testpath sdpo/data/mixed_val_80.jsonl \
    --deepspeed_config sdpo/sdpo_config_qwen3_bf16_zero3_2gpu.json \
    --savedir $SAVEROOT/$TAG \
    --anchor none --aux_loss al_tv \
    --anchor_weight none --aux_weight uniform \
    --sigmoid_coef 1.0 \
    --T_max 0.1 --T_min 0.1 \
    --prob_T_max 1.0 --prob_T_min 1.0 --draft_T_max 1.0 --draft_T_min 1.0 \
    --gamma 7 --max_len 2048 --num_epochs 2 --seed 42 --lr 1e-6 \
    --save_epochs 0,1 --max_train_samples 60000

# V2 target-topK — swap two lines above:
#     --aux_loss al_tv_target_topk \
#     --altv_topk 20 \
# top-P: --aux_loss al_tv_target_topp --altv_topp 0.99
# renewal (topK + ρ̄): --aux_loss al_tv_target_topk_renewal --altv_topk 20
```

### Training — sbatch

Both variants use the same slurm; only `AUX_LOSS`, `TAG_SUFFIX`, and (for topK variants) `ALTV_TOPK` change. `AUX_WEIGHT` defaults to `uniform` (weights = `[1, 1, ..., 1]`); override with e.g. `AUX_WEIGHT=pow08` if you want the geometric decay. Env vars are pre-exported so slurm's comma-splitter doesn't touch them.

```bash
# Full-vocab ALTV baseline
sbatch --export=ALL,AUX_LOSS=al_tv,TAG_SUFFIX=baseline_2ep \
    sdpo/slurm/altv_topk_train_q32_q06.slurm

# V2 target-topK K=20
sbatch --export=ALL,AUX_LOSS=al_tv_target_topk,TAG_SUFFIX=target_topk20_2ep,ALTV_TOPK=20 \
    sdpo/slurm/altv_topk_train_q32_q06.slurm

# top-P nucleus
sbatch --export=ALL,AUX_LOSS=al_tv_target_topp,TAG_SUFFIX=target_topp0p99_2ep,ALTV_TOPP=0.99 \
    sdpo/slurm/altv_topk_train_q32_q06.slurm
```

Continue-finetune from an existing ckpt (`altv_finetune_q32_q06.slurm`):

```bash
PRETRAIN=$SAVEROOT/q32_q06_altv_baseline_2ep/state_1_stripped   # see caveat below
sbatch --export=ALL,PRETRAIN=$PRETRAIN,AUX_LOSS=al_tv,TAG_SUFFIX=ft2ep_baseline \
    sdpo/slurm/altv_finetune_q32_q06.slurm
```

> **Ckpt-prefix caveat.** `main_small_lm.py` saves with `draft.base.` / `draft_model.` prefixes. `AutoModelForCausalLM.from_pretrained` on that dir silently random-inits the whole model — training starts with mean_τ ≈ 0.04. `eval_small_lm_tree.py` writes a prefix-stripped copy to `<state_N>_stripped/` on first use; **always continue-train from `_stripped`**, not the original.

### Eval — raw command with dtopk20 + ttopk20 verify

```bash
BASEPATH=Qwen/Qwen3-32B
CKPT=$SAVEROOT/q32_q06_altv_baseline_2ep/state_1_stripped        # or any trained ckpt
OUT=smalllm_tree_eval_results
BENCHES=mt_bench,gsm8k,humaneval

for SEED in 0 1 2; do
  for BENCH in mt_bench gsm8k humaneval; do
    # Chain (baseline-only, γ=7)
    FT="${TAG}_eseed${SEED}_chain_g7_samp_t1_ratio_dtopk20_ttopk20_max1024_${BENCH}_al"
    python sdpo/eval_small_lm_tree.py \
        --base-model-path $BASEPATH --draft-model-path $CKPT \
        --bench-name $BENCH --max-new-tokens 1024 --num-samples 40 \
        --output-dir $OUT --seed $SEED --tag $FT \
        --gamma 7 --top-k "4,3,2,1,1,1,1" --budget 128 \
        --baseline-only --baseline-gamma 7 \
        --temperature 1.0 --draft-mode sample --verify-mode ratio \
        --draft-top-k 20 --target-top-k 20

    # Tree (γ=7, top-k=[4,3,2,1,1,1,1], budget=128)
    FT="${TAG}_tree_eseed${SEED}_samp_t1_ratio_dtopk20_ttopk20_max1024_${BENCH}_al"
    python sdpo/eval_small_lm_tree.py \
        --base-model-path $BASEPATH --draft-model-path $CKPT \
        --bench-name $BENCH --max-new-tokens 1024 --num-samples 40 \
        --output-dir $OUT --seed $SEED --tag $FT \
        --gamma 7 --top-k "4,3,2,1,1,1,1" --budget 128 \
        --temperature 1.0 --draft-mode sample --verify-mode ratio \
        --draft-top-k 20 --target-top-k 20
  done
done
```

The chain block writes to a `*_chain_g7_*` filename, tree to a `*_tree_*` filename. Per-`(seed, bench, decode)` files let the pipeline SKIP-resume after preemption — the aggregate script (`smalllm_tree_eval_results/*.json`) reads all matching files across seeds.

### Eval — sbatch

For **max=1024** with 3 seeds × chain+tree × 3 benches in one job:

```bash
# max_new=1024, dtopk20+ttopk20 hardcoded in the slurm; pass CKPT + TAG_PREFIX
sbatch --export=ALL,DRAFT_CKPT=$CKPT,TAG_PREFIX=q32_q06_altv_baseline_2ep \
    sdpo/slurm/altv_max1024_all_q32.slurm
```

For **max=256** 3-seed eval (matches the ALTV big-table protocol) — uses the older per-seed slurm that stashes both chain and tree into the same JSON per seed:

```bash
sbatch --export=ALL,CKPT_TAG=q32_q06_altv_baseline_2ep \
    sdpo/slurm/altv_topk_eval_q32_q06.slurm
```

`CKPT_STATE` defaults to `state_1`; override with `--export=ALL,CKPT_TAG=...,CKPT_STATE=state_0`. `DRAFT_TOP_K` and `TARGET_TOP_K` default to `20` (set to `0` to disable either verify).

### File naming convention (used by the aggregate scripts)

```
{TAG}_eseed{SEED}_chain_g7_samp_t1_ratio_dtopk20_ttopk20_max1024_{BENCH}_al.json
{TAG}_tree_eseed{SEED}_samp_t1_ratio_dtopk20_ttopk20_max1024_{BENCH}_al.json
```

Fields: `samp_t1_ratio` = `T=1.0`, `draft-mode sample`, `verify-mode ratio`. Drop `_dtopk20_ttopk20` when `--draft-top-k 0 --target-top-k 0` (full-vocab verify). Drop `_max1024_` when `max_new=256`.

---

## Repository layout

```
sdpo/
  main_small_lm.py         entry point (deepspeed launchable)
  small_lm_model.py        SmallLMDraftModel — target/draft forward, loss menu
  spec_decode_gemma4.py    Gemma-4-specific tree decode (separate eval loader)
  eval_small_lm_tree.py    Qwen3 tree eval entry
  eval_gemma4_chain.py     Gemma 4 tree eval entry
  data/
    regen_*.py             target-regeneration utilities (vLLM)
    sharegpt_*_regen.jsonl regenerated training data
    mixed_val_80.jsonl     held-out validation set
  slurm/
    smalllm_06b_train.slurm           Qwen3 train (deepspeed)
    smalllm_tree_eval.slurm           Qwen3 tree eval
    gemma4_train.slurm                Gemma 4 train + auto-eval (one job)
    altv_g12_budget_sweep.slurm       ALTV γ=12 budget sweep per ckpt
    regen_all_gemma4.slurm            Gemma 4 regen (5 benches sequential)
  sdpo_config_qwen3.json                       ZeRO-2 (Qwen3 default)
  sdpo_config_qwen3_bf16_zero3_2gpu.json       ZeRO-3 (Q32B target)
  sdpo_config_gemma4_bf16_zero2_offload.json   ZeRO-2 (Gemma 4)
```

---

## Generating training data (target sequence regeneration)

The al_tv-family losses all consume a JSONL where each `assistant` turn is **regenerated by the target LLM** — so β measured during training is the same TV overlap the model sees at inference. Every trainpath cited above (`sharegpt_qwen3_{8b,32b}_regen.jsonl`, `sharegpt_gemma4_31b_regen.jsonl`) was produced this way.

### Output format

JSONL, one line per multi-turn conversation:

```json
{"id": "<original id>", "conversations": [
    {"from": "human",     "value": "<user prompt>"},
    {"from": "gpt",       "value": "<TARGET-regenerated assistant reply>"},
    {"from": "human",     "value": "<next user prompt>"},
    {"from": "gpt",       "value": "<TARGET-regenerated reply>"}
]}
```

Loss mask covers only `gpt` turns.

### Main script — `sdpo/data/regen_sharegpt.py`

vLLM-driven regeneration with per-turn context accumulation. Auto-detects the target's chat template via `AutoTokenizer.apply_chat_template` (works for Qwen3, Llama-3, Gemma alike); falls back to hand-written Llama-3 tags if no template is set. Stop tokens are auto-detected from the tokenizer.

### End-to-end recipe — `sdpo/slurm/regen_sharegpt.slurm`

```bash
sbatch --export=ALL,\
MODEL=Qwen/Qwen3-32B,\
DATA=/path/to/raw_sharegpt.json,\
OUTPUT=/scratch/<you>/sharegpt_qwen3_32b_regen.jsonl,\
TP=2,\
MAX_MODEL_LEN=8192,\
MAX_TOKENS=2048,\
BATCH_SIZE=200,\
TEMPERATURE=0.7,\
TOP_P=0.9,\
DTYPE=bfloat16 \
    sdpo/slurm/regen_sharegpt.slurm
```

| Env | Default | Meaning |
|---|---|---|
| **`MODEL`** | *required* | HF id / local path of the target LLM (e.g. `Qwen/Qwen3-{8B,32B}`, `google/gemma-4-31B-it`) |
| **`DATA`** | Aeala/ShareGPT snapshot | Raw ShareGPT-style JSON: `[{"id","conversations":[{"from","value"},...]}, ...]` |
| **`OUTPUT`** | *required* | Where to write the regenerated JSONL — this is the file you pass as `--trainpath` |
| `TP` | 2 | vLLM tensor-parallel size (Q0.6B/8B/32B: 2; Gemma 4 31B: 4) |
| `MAX_MODEL_LEN` | 8192 | vLLM context window |
| `MAX_TOKENS` | 2048 | Max new tokens per assistant turn |
| `BATCH_SIZE` | 200 | Prompts per vLLM batch |
| `MAX_SAMPLES` | 0 (= full) | Cap on source conversations (useful for smoke tests) |
| `TEMPERATURE` | 0.7 | vLLM sampling temperature |
| `TOP_P` | 0.9 | Nucleus |
| `DTYPE` | bfloat16 | Q3x targets need bf16 |

### Bench-style regeneration (mt_bench / gsm8k / humaneval / longwriter / osr2)

Not needed for training — al_tv trains on the shared `sharegpt_*_regen.jsonl`. Use these only when you want target-regenerated bench references for something else:

| Script | Use |
|---|---|
| `sdpo/data/prepare_bench_regen.py` + `sdpo/slurm/prepare_bench_regen.slurm` | Generic bench (mt_bench / gsm8k / humaneval) |
| `sdpo/data/prepare_osr2_regen.py` + `sdpo/slurm/prepare_osr2_regen.slurm` | OSR2 |
| `sdpo/data/prepare_longwriter_regen.py` + `sdpo/slurm/prepare_longwriter_regen.slurm` | LongWriter |
| `sdpo/slurm/regen_all_gemma4.slurm` | Gemma 4 — runs all 5 benches sequentially in one job |

### Existing target-regenerated corpora on scratch

| Target | Path | Samples | Avg seq (Qwen3 tokenizer) |
|---|---|---:|---:|
| Qwen3-8B | `data/sharegpt_qwen3_8b_regen.jsonl` | 120 675 | 2187 tok |
| Qwen3-32B | `data/sharegpt_qwen3_32b_regen.jsonl` | 120 675 | 2154 tok |
| Gemma 4 31B | `sdpo/data/sharegpt_gemma4_31b_regen.jsonl` | ~50 K | ~2 K tok |

### Pointing training at a new dataset

The Q32B slurms (`altv_topk_train_q32_q06.slurm`, `altv_finetune_q32_q06.slurm`, `altv_topk_train_q32_q06_9ep.slurm`, `altv_topk_train_q32_q17.slurm`, `altv_finetune_q32_q17.slurm`) accept a `TRAINPATH` env override — default is `data/sharegpt_qwen3_32b_regen.jsonl`. Pass your regen output through:

```bash
sbatch --export=ALL,AUX_LOSS=al_tv,TAG_SUFFIX=my_data_2ep,\
TRAINPATH=/scratch/<you>/sharegpt_qwen3_32b_regen.jsonl \
    sdpo/slurm/altv_topk_train_q32_q06.slurm
```


---

## Reproducing V2 target-topK ("target topK + full-vocab draft" training, dtopk20+ttopk20 verify)

The V2 setup that produced the strongest ALTV numbers to date. Train draft with `al_tv_target_topk K=20` (target renormalized inside its top-20; draft keeps its full-vocab softmax values at those indices), then run chain + tree spec-decoding with sample-ratio verify at `T=1`, both `--draft-top-k 20` and `--target-top-k 20`. Two hardware-tested pairs below.

### Files a collaborator needs

| Item | Where | How to get it |
|---|---|---|
| Repo | https://github.com/shawnyin128/SDPO-Speculative-Decoding-Policy-Optimization (branch `xth`) | `git clone -b xth …` — everything below runs from repo root |
| Target-regenerated training data | `sharegpt_qwen3_{8b,32b}_regen.jsonl` (~2 GB each) | Two options: (a) copy the files under `data/sharegpt_qwen3_{8b,32b}_regen.jsonl` (already in this repo checkout, `.gitignore`d) (grant them read); (b) regenerate from scratch — see the "Generating training data" section above |
| Base + draft weights | `Qwen/Qwen3-32B`, `Qwen/Qwen3-8B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-0.6B` | Public on Hugging Face — set `HF_HOME` and `huggingface-cli download` first-run, no access grant needed |
| Bench data | `sdpo/data/{mt_bench,gsm8k,humaneval,speed_qual_*}` | Already in the repo, no external download |

Grant read access on any scratch paths you send by pointing `--trainpath` at your own copy (env `TRAINPATH=...`, wired through in every training slurm below).

### Shared knobs (all training runs)

- `--aux_loss al_tv_target_topk --altv_topk 20` — V2 loss, target-topK=20
- `--anchor none --aux_weight uniform` — no anchor loss, weights = [1,1,…,1]
- `--gamma 7 --max_len 2048 --num_epochs 2 --lr 1e-6 --seed 42 --max_train_samples 60000` — standard hyperparameters
- DeepSpeed config picks by target: 8B → ZeRO-2 (`sdpo/sdpo_config_qwen3.json`); 32B → ZeRO-3 2-GPU (`sdpo/sdpo_config_qwen3_bf16_zero3_2gpu.json`).

### Save-path convention (what the training slurms do with ckpts)

Every Q32B training slurm resolves the save directory as:

```
${SAVEROOT}/${TAG}/state_{N}
```

- **`SAVEROOT`** — the top-level directory. Env-overridable in every training slurm; default is `/scratch/tx856/spec_reason/scratch/loss_train_smalllm_q32_{q06,q17}`, which is only writable by tx856. **A collaborator must set `SAVEROOT=/scratch/<their-netid>/...`** (any path they can write to) — otherwise the run fails at first save.
- **`TAG`** — auto-built from `TAG_SUFFIX`: the Q0.6B slurm uses `TAG=q32_q06_altv_${TAG_SUFFIX}`, the Q1.7B slurm uses `TAG=q32_q17_altv_${TAG_SUFFIX}`. The prefix is cosmetic (a historical hardcode); what actually matters is that `TAG_SUFFIX` is unique per run.
- **`state_N`** — one directory per epoch DeepSpeed saves. Controlled by `SAVE_EPOCHS` (pre-export before `sbatch --export=ALL` — sbatch splits `--export=…SAVE_EPOCHS=0,1,…` on the comma).

So for the Config A command below, `SAVEROOT=/scratch/<you>/loss_train_q8_q17` + `TAG_SUFFIX=target_topk20_2ep_q17` + `SAVE_EPOCHS=0,1` writes:

```
/scratch/<you>/loss_train_q8_q17/q32_q06_altv_target_topk20_2ep_q17/{state_0, state_1}
```

Point `--draft-model-path` (in the eval command) at `state_1` (the final).

### Config A · Qwen3-8B target + Qwen3-1.7B draft

The train slurm `sdpo/slurm/altv_topk_train_q32_q06.slurm` is env-configurable — override target/draft/config paths.

```bash
# Train (2ep, ~10 h on 2× H100/H200)
export SAVE_EPOCHS="0,1"    # pre-export to survive sbatch's comma splitter
sbatch --export=ALL,\
BASEPATH=Qwen/Qwen3-8B,\
DRAFTPATH=Qwen/Qwen3-1.7B,\
TRAINPATH=data/sharegpt_qwen3_8b_regen.jsonl,\
DS_CONFIG=sdpo/sdpo_config_qwen3.json,\
SAVEROOT=/scratch/<you>/loss_train_q8_q17,\
AUX_LOSS=al_tv_target_topk,\
ALTV_TOPK=20,\
AUX_WEIGHT=uniform,\
TAG_SUFFIX=target_topk20_2ep_q17 \
    sdpo/slurm/altv_topk_train_q32_q06.slurm
```

Resulting ckpt: `/scratch/<you>/loss_train_q8_q17/q32_q06_altv_target_topk20_2ep_q17/state_1`.

```bash
# Eval — max_new=1024, 3 seeds × chain + tree × mt_bench/gsm8k/humaneval, dtopk20+ttopk20, T=1 sample+ratio
CKPT=/scratch/<you>/loss_train_q8_q17/q32_q06_altv_target_topk20_2ep_q17/state_1
sbatch --export=ALL,\
BASEPATH=Qwen/Qwen3-8B,\
DRAFT_CKPT=$CKPT,\
TAG_PREFIX=q8_q17_altv_target_topk20_2ep \
    sdpo/slurm/altv_max1024_all_q32.slurm
```

For a 3-seed max_new=256 eval instead (matches the earlier V2 K sweep numbers):

```bash
sbatch --export=ALL,\
BASEPATH=Qwen/Qwen3-8B,\
CKPT_TAG=q32_q06_altv_target_topk20_2ep_q17,\
CKPT_STATE=state_1 \
    sdpo/slurm/altv_topk_eval_q32_q06.slurm
```

### Config B · Qwen3-32B target + Qwen3-0.6B draft

This is the original V2 setup — the slurm defaults already match, so almost nothing needs to be overridden.

```bash
export SAVE_EPOCHS="0,1"
sbatch --export=ALL,\
AUX_LOSS=al_tv_target_topk,\
ALTV_TOPK=20,\
AUX_WEIGHT=uniform,\
TRAINPATH=data/sharegpt_qwen3_32b_regen.jsonl,\
SAVEROOT=/scratch/<you>/loss_train_q32_q06,\
TAG_SUFFIX=target_topk20_2ep \
    sdpo/slurm/altv_topk_train_q32_q06.slurm
```

Resulting ckpt: `/scratch/<you>/loss_train_q32_q06/q32_q06_altv_target_topk20_2ep/state_1`.

```bash
# Eval — same shape as A
CKPT=/scratch/<you>/loss_train_q32_q06/q32_q06_altv_target_topk20_2ep/state_1
sbatch --export=ALL,\
DRAFT_CKPT=$CKPT,\
TAG_PREFIX=q32_q06_altv_target_topk20_2ep \
    sdpo/slurm/altv_max1024_all_q32.slurm
```

### Inference verify flags (both configs)

The eval slurm hardcodes `--draft-top-k 20 --target-top-k 20 --temperature 1.0 --draft-mode sample --verify-mode ratio` on both chain and tree, so the collaborator does not need to set anything at inference time beyond `DRAFT_CKPT` and `BASEPATH`. Numbers land in `smalllm_tree_eval_results/{TAG_PREFIX}_{seed,chain/tree}_..._al.json`.

### Sanity target (Config B, previously observed)

| Decode | max_new=256, 3 seeds | max_new=1024, 3 seeds |
|---|---:|---:|
| Chain | 2.907 ± 0.042 | (bench-scaled, see SPEED-Bench table above) |
| Tree | 5.727 ± 0.037 | – |

Anything within ± 0.05 of these on the standard mt_bench/gsm8k/humaneval mix confirms the reproduction is clean.
