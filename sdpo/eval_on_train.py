"""
Evaluate draft model acceptance rate on training data.
Compares original vs trained draft on the same training samples.

Usage:
  python sdpo/eval_on_train.py \
    --basepath Qwen/Qwen3-8B \
    --original-draft Qwen/Qwen3-1.7B \
    --trained-draft /path/to/checkpoint/state_0 \
    --trainpath sdpo/data/mixed_train_10K.jsonl \
    --num-samples 50
"""

import argparse
import json
import os
import glob

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def load_draft_checkpoint(draft_path, dtype=torch.float16):
    """Load draft model, handling 'draft_model.' prefix from DeepSpeed."""
    config_path = os.path.join(draft_path, "config.json")
    if os.path.exists(config_path):
        # Checkpoint with possible prefix
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(draft_path)
        model = AutoModelForCausalLM.from_config(config)
        bin_files = glob.glob(os.path.join(draft_path, "*.bin"))
        if bin_files:
            state = {}
            for f in bin_files:
                state.update(torch.load(f, map_location="cpu"))
            # Strip draft_model. prefix if present
            if any(k.startswith("draft_model.") for k in state):
                state = {k.removeprefix("draft_model."): v for k, v in state.items()}
            model.load_state_dict(state, strict=True)
        model = model.to(dtype).cuda()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            draft_path, torch_dtype=dtype, device_map="auto")
    model.eval()
    return model


@torch.inference_mode()
def evaluate_draft(target, draft, tokenizer, samples, gamma=7, max_len=512):
    total_match = 0
    total_pos = 0
    total_tau = 0
    total_starts = 0
    step_matches = [0] * gamma
    step_counts = [0] * gamma

    for s in tqdm(samples, desc="Evaluating"):
        convs = s.get("conversations", [])
        if len(convs) < 2:
            continue

        roles = {"human": "user", "gpt": "assistant"}
        msgs = []
        for c in convs:
            role = roles.get(c["from"], c["from"])
            msgs.append({"role": role, "content": c["value"]})

        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False)
        ids = tokenizer(text, return_tensors="pt",
                        add_special_tokens=False).input_ids.cuda()
        if ids.shape[1] > max_len:
            ids = ids[:, :max_len]
        if ids.shape[1] < gamma + 2:
            continue

        t_logits = target(ids).logits[:, :-1, :].float()
        d_logits = draft(ids).logits[:, :-1, :].float()
        t_arg = t_logits.argmax(-1)
        d_arg = d_logits.argmax(-1)
        match = (t_arg == d_arg)  # [1, L-1]

        total_match += match.sum().item()
        total_pos += match.numel()

        # Acceptance length over gamma windows
        L = match.shape[1]
        for t in range(L - gamma + 1):
            tau = 0
            for k in range(gamma):
                if match[0, t + k]:
                    tau += 1
                    step_matches[k] += 1
                else:
                    break
                step_counts[k] += 1
            # Count remaining steps that didn't match
            for k2 in range(tau, gamma):
                step_counts[k2] += 1
            total_tau += tau
            total_starts += 1

    token_acc = total_match / total_pos if total_pos > 0 else 0
    mean_tau = total_tau / total_starts if total_starts > 0 else 0
    step_acc = [step_matches[k] / step_counts[k] if step_counts[k] > 0 else 0
                for k in range(gamma)]

    return {
        "token_acc": token_acc,
        "mean_tau": mean_tau,
        "total_windows": total_starts,
        "total_tokens": total_pos,
        "step_acc": step_acc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basepath", required=True, help="Target model")
    parser.add_argument("--original-draft", required=True, help="Original draft model")
    parser.add_argument("--trained-draft", required=True, help="Trained draft checkpoint")
    parser.add_argument("--trainpath", required=True, help="Training data JSONL")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--gamma", type=int, default=7)
    parser.add_argument("--max-len", type=int, default=512)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.basepath, trust_remote_code=True)

    # Load samples
    samples = []
    with open(args.trainpath) as f:
        for i, line in enumerate(f):
            if i >= args.num_samples:
                break
            samples.append(json.loads(line))
    print(f"Loaded {len(samples)} training samples")

    # Load target
    print(f"\nLoading target: {args.basepath}")
    target = AutoModelForCausalLM.from_pretrained(
        args.basepath, torch_dtype=torch.float16, device_map="auto")
    target.eval()

    # Eval original draft
    print(f"\nLoading original draft: {args.original_draft}")
    draft_orig = AutoModelForCausalLM.from_pretrained(
        args.original_draft, torch_dtype=torch.float16, device_map="auto")
    draft_orig.eval()

    print("\n--- Original Draft ---")
    r_orig = evaluate_draft(target, draft_orig, tokenizer, samples, args.gamma, args.max_len)
    print(f"  token_acc: {r_orig['token_acc']:.4f}")
    print(f"  mean_tau:  {r_orig['mean_tau']:.2f}")
    print(f"  step_acc:  {['%.3f' % a for a in r_orig['step_acc']]}")

    # Free memory
    del draft_orig
    torch.cuda.empty_cache()

    # Eval trained draft
    print(f"\nLoading trained draft: {args.trained_draft}")
    draft_trained = load_draft_checkpoint(args.trained_draft)

    print("\n--- Trained Draft ---")
    r_trained = evaluate_draft(target, draft_trained, tokenizer, samples, args.gamma, args.max_len)
    print(f"  token_acc: {r_trained['token_acc']:.4f}")
    print(f"  mean_tau:  {r_trained['mean_tau']:.2f}")
    print(f"  step_acc:  {['%.3f' % a for a in r_trained['step_acc']]}")

    # Comparison
    print(f"\n{'='*50}")
    print(f"  {'Metric':<15} {'Original':>10} {'Trained':>10} {'Delta':>10}")
    print(f"  {'-'*45}")
    print(f"  {'token_acc':<15} {r_orig['token_acc']:>10.4f} {r_trained['token_acc']:>10.4f} {r_trained['token_acc']-r_orig['token_acc']:>+10.4f}")
    print(f"  {'mean_tau':<15} {r_orig['mean_tau']:>10.2f} {r_trained['mean_tau']:>10.2f} {r_trained['mean_tau']-r_orig['mean_tau']:>+10.2f}")
    for k in range(args.gamma):
        d = r_trained['step_acc'][k] - r_orig['step_acc'][k]
        print(f"  {'step_'+str(k)+'_acc':<15} {r_orig['step_acc'][k]:>10.3f} {r_trained['step_acc'][k]:>10.3f} {d:>+10.3f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
