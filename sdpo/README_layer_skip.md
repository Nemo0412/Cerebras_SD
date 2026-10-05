# Layer-Skip Draft Training for Speculative Decoding

Train a small LM (e.g., Qwen3-0.6B) so multiple intermediate transformer
layers can each act as an independent draft for a large target (e.g.,
Qwen3-32B). Each exit reuses the model's own `RMSNorm` + `lm_head`, with
per-exit losses summed and back-propagated through the shared head.

## Files

| Path | Role |
|---|---|
| `sdpo/layer_skip_model.py` | `LayerSkipBackbone` (multi-exit draft) + `LayerSkipDraftModel` (training wrapper with frozen target) |
| `sdpo/main_layer_skip.py` | Training entry point (multi-exit KL / V2 / TV / CE) |
| `sdpo/eval_layer_skip.py` | Speculative decoding eval, runs each exit independently |
| `sdpo/data_pipeline.py` | Shared ShareGPT data pipeline (also used by `main_small_lm.py`) |
| `sdpo/slurm/layer_skip_train.slurm` | SLURM training script (env-var driven) |
| `sdpo/slurm/layer_skip_eval.slurm` | SLURM eval script (env-var driven) |
| `sdpo/slurm/submit_sweep_layer_skip.sh` | Training sweep submitter (TASKS list) |
| `sdpo/slurm/submit_eval_layer_skip.sh` | Eval sweep submitter (untrained baseline + trained checkpoints) |

## Loss Options (per exit, then averaged)

```
--baseline eagle_only       # KL(target || draft_e)             pure distillation
--baseline ce_only          # -log q_e(target_argmax)           hard cross entropy
--aux_loss acceptance_length_v2  # KL + sigmoid_coef * L_acc_v2  (default)
--aux_loss tv               # KL + sigmoid_coef * L_tv
```

The default temperature `T_min = T_max = 0.1` (focused gradient on hard
positions). The default `sigmoid_coef = 0.1`.

## Step 1 — Baseline exit sweep (no training)

Find which intermediate exits already have decent acceptance α before any
training. This anchors the training upside.

```bash
cd /path/to/SDPO-Speculative-Decoding-Policy-Optimization

# Single exit at a time, parallel jobs (faster, uses less GPU per job)
for e in 16 20 24 26 28; do
    MAX_NEW_TOKENS=256 \
    EXIT_LAYERS=$e \
    BENCH=mt_bench \
    BASELINE_TAG=ls_orig_e${e} \
        bash sdpo/slurm/submit_eval_layer_skip.sh
done
```

Or all exits in one job (slower but single result file):

```bash
MAX_NEW_TOKENS=256 \
EXIT_LAYERS=4,8,12,16,20,24,28 \
BENCH=mt_bench,gsm8k,humaneval \
    bash sdpo/slurm/submit_eval_layer_skip.sh
```

Results land in `layerskip_eval_results/{tag}.json`, written incrementally
after each `(bench, exit)` pair so a walltime kill keeps partial data.

## Step 2 — Train on selected exits

The sweep script `submit_sweep_layer_skip.sh` defines a `TASKS` list with
pre-baked configurations. Format:

```
"tag|exit_layers|dataset|lr|loss|coef|tmin|tmax"
```

- `dataset`: `10k` (mixed_train_10K.jsonl) or `full` (mixed_train.jsonl)
- `loss`: `eagle_only` / `ce_only` / `v2` / `tv`
- `coef`/`tmin`/`tmax`: only applied when `loss == v2 | tv`

### List available tasks

```bash
bash sdpo/slurm/submit_sweep_layer_skip.sh --list
```

### Submit a single task

```bash
bash sdpo/slurm/submit_sweep_layer_skip.sh ls_deep_v2_full
```

### Submit several at once

```bash
bash sdpo/slurm/submit_sweep_layer_skip.sh \
    ls_deep_v2_full \
    ls_deep_v2_full_lr2e6 ls_deep_v2_full_lr3e6 \
    ls_e24_28_lr2e6 ls_e24_28_lr5e6
```

### Default sweep matrix

| Tag | Exits | Data | LR | Loss |
|---|---|---|---|---|
| `ls_deep_v2_full` | 20,24,26,28 | full 68K | 5e-6 | KL + V2 |
| `ls_deep_v2_full_lr1e6` | 20,24,26,28 | full 68K | 1e-6 | KL + V2 |
| `ls_deep_v2_full_lr2e6` | 20,24,26,28 | full 68K | 2e-6 | KL + V2 |
| `ls_deep_v2_full_lr3e6` | 20,24,26,28 | full 68K | 3e-6 | KL + V2 |
| `ls_deep_eagle_full` | 20,24,26,28 | full 68K | 5e-6 | KL only |
| `ls_e24_28_lr{2e6,5e6}` | 24,28 | 10K | 2e-6 / 5e-6 | KL + V2 |
| `ls_e26_28_lr{2e6,5e6}` | 26,28 | 10K | 2e-6 / 5e-6 | KL + V2 |
| `ls_e24_26_28_lr{2e6,5e6}` | 24,26,28 | 10K | 2e-6 / 5e-6 | KL + V2 |
| `ls_e20_24_28_lr{2e6,5e6}` | 20,24,28 | 10K | 2e-6 / 5e-6 | KL + V2 |
| `ls_all_v2_full` | 4,8,…,28 (all 7) | full 68K | 5e-6 | KL + V2 |

### Resource defaults (SLURM)

| Setting | Value |
|---|---|
| GPUs | 2 (`GPUS=2` env var; ZeRO-3 BF16 on H100/H200) |
| Memory | 256 GB |
| Walltime | 24 h |
| Constraint | `h100\|h200` |

Switch to 4 GPUs:

```bash
GPUS=4 bash sdpo/slurm/submit_sweep_layer_skip.sh ls_deep_v2_full
```

## Step 3 — Speculative decoding eval on trained checkpoints

```bash
MAX_NEW_TOKENS=256 \
EXIT_LAYERS=20,24,26,28 \
BENCH=mt_bench \
    bash sdpo/slurm/submit_eval_layer_skip.sh ls_deep_v2_full --no-baseline
```

Compare each exit's mean α against the untrained baseline from Step 1. The
training-time `mean_tau` (teacher-forced) overestimates real spec decoding
α by a wide margin — always confirm with this Step 3 eval before drawing
conclusions about a training run.

## Direct training command (no SLURM)

```bash
cd /path/to/SDPO-Speculative-Decoding-Policy-Optimization

deepspeed --num_gpus=2 --master_port=29501 sdpo/main_layer_skip.py \
    --basepath Qwen/Qwen3-32B \
    --draftpath Qwen/Qwen3-0.6B \
    --exit-layers 20,24,26,28 \
    --trainpath sdpo/data/mixed_train_10K.jsonl \
    --testpath sdpo/data/mixed_val_80.jsonl \
    --deepspeed_config sdpo/sdpo_config_qwen3_bf16_zero3_2gpu.json \
    --savedir /scratch/tx856/spec_reason/scratch/loss_train_layerskip/ls_test \
    --sigmoid_coef 0.1 --T_max 0.1 --T_min 0.1 \
    --aux_loss acceptance_length_v2 \
    --gamma 7 --max_len 1024 --num_epochs 3 --lr 1e-6
```

## Direct eval command (no SLURM)

```bash
python sdpo/eval_layer_skip.py \
    --base-model-path Qwen/Qwen3-32B \
    --draft-model-path /scratch/tx856/spec_reason/scratch/loss_train_layerskip/ls_test/state_2 \
    --exit-layers 20,24,26,28 \
    --tag ls_test \
    --bench-name mt_bench \
    --max-new-tokens 256
```

## Known footguns

1. **`sbatch --export` parses commas as variable separators.** When passing
   `EXIT_LAYERS=20,24,26,28` or `BENCH=mt_bench,gsm8k`, do NOT put it inside
   the `--export=` list. The submit scripts work around this by `export`-ing
   the variable in the parent shell and relying on `--export=ALL` to inherit
   it. If you write your own sbatch invocations, follow the same pattern.

2. **Final RMSNorm double-application.** HuggingFace Qwen3's
   `output_hidden_states=True` returns `hidden_states[N]` (the final entry)
   already passed through `model.norm`, while intermediate entries are raw.
   `LayerSkipBackbone.forward_with_exits()` applies `model.norm` only when
   `exit_layer < num_layers` to avoid double-norming the final exit, which
   would produce garbage logits.

3. **Teacher-forced `mean_tau` ≠ real spec decoding α.** The training loop
   reports per-exit `mean_tau` computed on target-greedy prefixes. This
   number can look great while real spec decoding α (Step 3) is unchanged
   or worse. Always validate with the actual eval before declaring victory.

4. **Resume safety.** If you re-submit a task after killing a previous run,
   delete `${SAVEROOT}/${TAG}/` first. Otherwise `find_latest_checkpoint`
   will resume from the old `state_*` and silently mix configurations.
