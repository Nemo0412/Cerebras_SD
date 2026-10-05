"""Smoke test for SmallLMTreeTrainModel forward + backward.

Loads tiny target/draft (Qwen3-0.6B as both for fast smoke), constructs a
fake batch, runs forward, checks losses and gradients on draft params only.

Usage:
  python sdpo/diagnose/smoke_test_tree_train.py \\
    --target Qwen/Qwen3-0.6B --draft Qwen/Qwen3-0.6B
"""
import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "sdpo"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="Qwen/Qwen3-0.6B",
                   help="Use a SMALL target for the smoke test (default=draft itself).")
    p.add_argument("--draft", default="Qwen/Qwen3-0.6B")
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    from small_lm_tree_train_model import SmallLMTreeTrainModel

    print(f"[SMOKE] target={args.target}  draft={args.draft}")
    model = SmallLMTreeTrainModel(
        target_path=args.target, draft_path=args.draft,
        dtype=torch.bfloat16,
    ).to(args.device)
    model.train()

    # Fake input: random ids, full attention, loss_mask=1 on last ~half
    L = args.seq_len
    B = args.batch
    vocab = model.draft_model.config.vocab_size
    torch.manual_seed(0)
    input_ids = torch.randint(1, vocab, (B, L), device=args.device, dtype=torch.long)
    attention_mask = torch.ones(B, L, device=args.device, dtype=torch.long)
    loss_mask = torch.zeros(B, L, device=args.device, dtype=torch.long)
    loss_mask[:, L // 2:] = 1   # second half is "assistant"

    total_loss, _, metrics = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        anchor_kl_coef=1.0,
        eal_aux_coef=0.1,
    )
    print(f"[SMOKE] forward OK — metrics: {metrics}")
    print(f"[SMOKE] total_loss requires_grad={total_loss.requires_grad} "
          f"value={total_loss.item():.4f}")

    total_loss.backward()

    n_with_grad = 0
    n_total = 0
    for n, p in model.draft_model.named_parameters():
        if not p.requires_grad:
            continue
        n_total += 1
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            n_with_grad += 1
    print(f"[SMOKE] backward OK — draft params with non-zero grad: "
          f"{n_with_grad} / {n_total}")

    # Sanity: target params should NOT have grads
    n_tgt_grad = sum(1 for p in model.target_model.parameters()
                     if p.grad is not None and p.grad.abs().sum().item() > 0)
    print(f"[SMOKE] target params with grad (should be 0): {n_tgt_grad}")

    print("[SMOKE] PASS" if (n_with_grad > 0 and n_tgt_grad == 0)
          else "[SMOKE] FAIL")


if __name__ == "__main__":
    main()
