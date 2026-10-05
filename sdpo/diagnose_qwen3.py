"""Quick diagnostic for Qwen3 training setup."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset

# 1. Check chat template and SEP tokens
print("=" * 60)
print("1. Chat template check")
print("=" * 60)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)

msgs = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "The answer is 4."},
]
conversation = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
print(f"Conversation:\n{repr(conversation)}\n")

SEP_ASST = "<|im_start|>assistant\n"
SEP_USER = "<|im_start|>user\n"
print(f"SEP_USER found: {SEP_USER in conversation}")
print(f"SEP_ASST found: {SEP_ASST in conversation}")
print(f"Splits by SEP_USER: {len(conversation.split(SEP_USER))}")
print(f"Splits by SEP_ASST: {len(conversation.split(SEP_ASST))}")

# 2. Check loss_mask
print("\n" + "=" * 60)
print("2. Loss mask check")
print("=" * 60)

if not tokenizer.pad_token_id:
    tokenizer.pad_token_id = tokenizer.unk_token_id

ids = tokenizer(conversation, return_tensors="pt", add_special_tokens=False).input_ids[0]
print(f"Token count: {len(ids)}")

loss_mask = torch.ones_like(ids)
turns_raw = conversation.split(SEP_USER)
print(f"turns_raw count: {len(turns_raw)}")

if len(turns_raw) >= 2:
    turns_raw[1] = turns_raw[0] + SEP_USER + turns_raw[1]
    turns = turns_raw[1:]

    cur_len = 1
    loss_mask[:1] = 0
    for ti, turn in enumerate(turns):
        if turn == "":
            break
        turn_len = len(tokenizer(turn).input_ids)
        parts = turn.split(SEP_ASST)
        print(f"  Turn {ti}: turn_len={turn_len}, parts={len(parts)}")
        if len(parts) != 2:
            print(f"  WARNING: turn {ti} split by SEP_ASST gave {len(parts)} parts, expected 2")
            break
        parts[0] += SEP_ASST
        instr_len = len(tokenizer(parts[0]).input_ids) - 1
        if ti == 0:
            loss_mask[cur_len: cur_len + instr_len - 2] = 0
        else:
            loss_mask[cur_len - 3: cur_len + instr_len + 1] = 0
        cur_len += turn_len
        if ti != 0:
            cur_len += 3
    loss_mask[cur_len:] = 0

print(f"\nloss_mask sum: {loss_mask.sum().item()} / {len(loss_mask)} tokens")
print(f"loss_mask ratio: {loss_mask.float().mean().item():.3f}")

# Show token-by-token
decoded = [tokenizer.decode(ids[i:i+1]) for i in range(len(ids))]
print("\nToken-by-token (first 50):")
for i in range(min(50, len(ids))):
    print(f"  [{i:3d}] mask={loss_mask[i].item()} token={repr(decoded[i])}")

# 3. Check hidden states from AutoModelForCausalLM
print("\n" + "=" * 60)
print("3. Hidden states check")
print("=" * 60)
config = AutoConfig.from_pretrained("Qwen/Qwen3-8B")
print(f"Architecture: {config.architectures}")
print(f"Hidden size: {config.hidden_size}")
print(f"Num layers: {config.num_hidden_layers}")

# Check if output_hidden_states works
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B", torch_dtype=torch.float16, output_hidden_states=True,
    device_map="auto"
)
model.eval()

with torch.no_grad():
    test_ids = ids[:20].unsqueeze(0).to(model.device)
    outs = model(input_ids=test_ids)
    print(f"Has hidden_states: {outs.hidden_states is not None}")
    if outs.hidden_states is not None:
        print(f"Num hidden_states: {len(outs.hidden_states)}")
        for i in range(min(4, len(outs.hidden_states))):
            print(f"  hidden_states[{i}] shape: {outs.hidden_states[i].shape}")
    print(f"Logits shape: {outs.logits.shape}")

# 4. Check draft model vocab mapping
print("\n" + "=" * 60)
print("4. Vocab mapping check")
print("=" * 60)
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

draftpath = snapshot_download("AngelSlim/Qwen3-8B_eagle3")
sf_path = os.path.join(draftpath, "pytorch_model.bin")
state = load_file(sf_path, device="cpu")
if "d2t" in state and "t2d" in state:
    d2t = state["d2t"]
    t2d = state["t2d"]
    print(f"d2t shape: {d2t.shape}, t2d shape: {t2d.shape}")
    print(f"draft_vocab: {len(d2t)}, target_vocab coverage: {t2d.sum().item()}/{len(t2d)}")

    # Check what fraction of our test tokens are in draft vocab
    in_vocab = t2d[ids].float().mean().item()
    print(f"Test tokens in draft vocab: {in_vocab:.3f}")
else:
    print("WARNING: no d2t/t2d in checkpoint!")
