# Running Jobs Status - 2026-04-09

## Currently Running
| Job ID | Tag | Model | Status |
|--------|-----|-------|--------|
| 5799212 | sl06_lr_1e-7 | SmallLM 0.6B | RUNNING |
| 5799215 | sl06_T_0.5 | SmallLM 0.6B | RUNNING |
| 5799117 | sl_lr_1e-5 | SmallLM 1.7B | RUNNING |

## Pending (QOSGrpGRES - waiting for GPU quota)

### SmallLM 0.6B (Qwen3-8B target + Qwen3-0.6B draft)
| Job ID | Tag | Config |
|--------|-----|--------|
| 5799210 | sl06_coef_0.5 | coef=0.5, T=0.1, lr=1e-5, 10K |
| 5799213 | sl06_lr_1e-6 | coef=0.1, T=0.1, lr=1e-6, 10K |
| 5799214 | sl06_T_0.2 | coef=0.1, T=0.2, lr=1e-5, 10K |
| 5799216 | sl06_eo_lr_1e-6 | eagle_only, lr=1e-6, 10K |
| 5805055 | sl06_full_eagle_only | eagle_only, lr=1e-5, full 68K |
| 5805056 | sl06_full_coef_0.1 | coef=0.1, T=0.1, lr=1e-5, full 68K |
| 5805057 | sl06_full_coef_0.5 | coef=0.5, T=0.1, lr=1e-5, full 68K |
| 5805058 | sl06_full_lr_1e-6 | coef=0.1, T=0.1, lr=1e-6, full 68K |

### SmallLM 1.7B (Qwen3-8B target + Qwen3-1.7B draft)
| Job ID | Tag | Config |
|--------|-----|--------|
| 5805070 | sl_full_eagle_only | eagle_only, full 68K |
| 5805023 | sl_full_coef_0.5 | coef=0.5, T=1.0, full 68K |
| 5805024 | sl_full_T_0.1 | coef=0.1, T=0.1, full 68K |
| 5805025 | sl_full_lr_1e-6 | coef=0.1, T=0.1, lr=1e-6, full 68K |

### Qwen3-8B EAGLE3 - TV loss (pure TV, no eagle loss)
| Job ID | Tag | Config |
|--------|-----|--------|
| 5804368 | q3tv_eagle_only | eagle_only baseline, T=0.1, 10K |
| 5804369 | q3tv_coef_0.1 | TV coef=0.1, T=0.1, 10K |
| 5804370 | q3tv_coef_0.5 | TV coef=0.5, T=0.1, 10K |
| 5804371 | q3tv_coef_1.0 | TV coef=1.0, T=0.1, 10K |

### Qwen3-8B EAGLE3 - V4 loss (β=2σ, eagle+acc_length)
| Job ID | Tag | Config |
|--------|-----|--------|
| 5805145 | q3v4_T_0.1 | coef=0.1, T=0.1, 10K |
| 5805146 | q3v4_T_0.5 | coef=0.1, T=0.5, 10K |
| 5805147 | q3v4_coef_0.2_T_0.1 | coef=0.2, T=0.1, 10K |
| 5805148 | q3v4_coef_0.2_T_0.5 | coef=0.2, T=0.5, 10K |

## Eval Commands (run after training completes)

```bash
# SmallLM 1.7B eval
bash sdpo/slurm/submit_eval_smalllm.sh

# SmallLM 0.6B eval
bash sdpo/slurm/submit_eval_smalllm_06b.sh

# Qwen3 EAGLE3 (TV + V4) eval
bash sdpo/slurm/submit_eval_qwen3.sh --no-baseline q3tv
bash sdpo/slurm/submit_eval_qwen3.sh --no-baseline q3v4

# Check status
squeue -u $USER
bash sdpo/slurm/summarize_eval.sh logs/sdpo_qwen3
bash sdpo/slurm/summarize_eval.sh logs/sdpo_smalllm
bash sdpo/slurm/summarize_eval.sh logs/sdpo_smalllm_06b
```

## Check job status
```bash
# Quick status check
squeue -u $USER

# Check specific job
sacct -j <JOBID> --format=JobID,JobName,State,ExitCode,Reason -n

# Check completed jobs
sacct -u tx856 --starttime=2026-04-09 --format=JobID,JobName,State,Elapsed,ExitCode -n | head -30
```

## Previous completed results summary
See: bash sdpo/slurm/summarize_eval.sh logs/sdpo_qwen3
