"""
Quick eval: compute mean_S and mean_tau on test set for any checkpoint.
Single GPU, no DeepSpeed.

Usage:
    # Original EAGLE3 (no fine-tune)
    python sdpo/experiments/eval_test_loss.py \
        --basepath /path/to/llama3-8b-instruct \
        --draftpath /path/to/EAGLE3-LLaMA3.1-Instruct-8B \
        --testpath sdpo/data/test.jsonl \
        --label "original_eagle3"

    # Fine-tuned checkpoint
    python sdpo/experiments/eval_test_loss.py \
        --basepath /path/to/llama3-8b-instruct \
        --draftpath sdpo_checkpoints/state_1 \
        --testpath sdpo/data/test.jsonl \
        --label "sdpo_0.003_ep2"
"""

import argparse
import json
import os
import shutil
import sys

import torch
import torch.nn.functional as F
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(script_dir, '..')
sys.path.insert(0, project_dir)
sys.path.insert(0, os.path.join(project_dir, 'sdpo'))
sys.path.insert(0, os.path.join(project_dir, 'traineagle3'))

from huggingface_hub import snapshot_download
from traineagle3.configs import EConfig
from traineagle3.cnets import Model
from datasets import load_dataset
from transformers import AutoTokenizer
from types import SimpleNamespace

SEP_ASST = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
SEP_USER = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
SYSTEM_MSG = ("You are a helpful, respectful and honest assistant. "
              "Always answer as helpfully as possible, while being safe.")


def load_samples(tokenizer, datapath, max_len, max_samples):
    import json as _json
    import random
    with open(datapath) as _f:
        raw = [_json.loads(line) for line in _f]
    random.Random(42).shuffle(raw)
    if max_samples:
        raw = raw[:max_samples]

    # Auto-detect format: ShareGPT ("conversations") or bench ("turns")
    if raw and 'turns' in raw[0] and 'conversations' not in raw[0]:
        converted = []
        for item in raw:
            convs = [{"from": "human", "value": item["turns"][0]},
                     {"from": "gpt", "value": "placeholder"}]
            converted.append({'id': str(item.get('question_id', '')), 'conversations': convs})
        raw = converted

    results = []
    roles = {"human": "user", "gpt": "assistant"}
    for item in raw:
        src = item['conversations']
        if not src:
            continue
        if roles.get(src[0]["from"]) != "user":
            src = src[1:]
        msgs = [{"role": "system", "content": SYSTEM_MSG}]
        for j, sent in enumerate(src):
            msgs.append({"role": roles[sent["from"]], "content": sent["value"]})
        conversation = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False)
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.unk_token_id
        ids = tokenizer(conversation, return_tensors="pt",
                        add_special_tokens=False).input_ids[0]
        if len(ids) > max_len:
            continue
        loss_mask = torch.ones_like(ids)
        turns_raw = conversation.split(SEP_USER)
        turns_raw[1] = turns_raw[0] + SEP_USER + turns_raw[1]
        turns = turns_raw[1:]
        cur_len = 1
        loss_mask[:1] = 0
        for ti, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(tokenizer(turn).input_ids)
            parts = turn.split(SEP_ASST)
            if len(parts) != 2:
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
        results.append({'input_ids': ids, 'loss_mask': loss_mask,
                        'attention_mask': torch.ones_like(ids)})
    return results


@torch.no_grad()
def eval_one_sample(model, sample, gamma, device, full2draft, t2d, d2t):
    input_ids = sample['input_ids'].unsqueeze(0).to(device)
    attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
    loss_mask = sample['loss_mask'].unsqueeze(0).to(device)

    hidden_states, target, loss_mask_3d, input_ids_shifted = model.dataprepare(
        input_ids, attention_mask, loss_mask
    )
    loss_mask_1d = loss_mask_3d.squeeze(-1).squeeze(0).bool()
    B, L, _ = hidden_states.shape

    hs_projected = model.fc(hidden_states.to(model.fc.weight.dtype))
    attn_mask = model._prepare_decoder_attention_mask(
        attention_mask, (B, L), hs_projected, 0)
    position_ids = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0)
    target_greedy = target.argmax(dim=-1)

    per_step_q = []
    per_step_acc = []
    cache_hidden = [[], []]
    cur_ids = input_ids_shifted
    cur_hs = hs_projected
    cur_tgt = target_greedy.clone()

    for idx in range(gamma):
        embeds = model.embed_tokens(cur_ids).to(cur_hs.dtype)
        layer_out, cache_hidden = model.midlayer(
            input_emb=embeds, hidden_states=cur_hs, cache_hidden=cache_hidden,
            attention_mask=attn_mask, position_ids=position_ids,
            past_key_value=None, output_attentions=False, use_cache=True,
        )
        cur_hs = layer_out[0]
        logits = model.lm_head(model.norm(cur_hs)).float()

        draft_d = logits.argmax(dim=-1)
        draft_full = draft_d + d2t[draft_d]
        tgt_in = t2d[cur_tgt]
        accept = (draft_full == cur_tgt) & tgt_in

        tgt_d = full2draft[cur_tgt].clamp(min=0)
        q = F.softmax(logits, dim=-1).gather(-1, tgt_d.unsqueeze(-1)).squeeze(-1)
        q = q * tgt_in.float()

        per_step_q.append(q.squeeze(0))
        per_step_acc.append(accept.squeeze(0))

        if idx < gamma - 1:
            cur_ids = cur_tgt
            cur_tgt = torch.cat([cur_tgt[:, 1:], torch.zeros_like(cur_tgt[:, :1])], dim=1)

    q_all = torch.stack(per_step_q, dim=1)  # [L, gamma]
    acc_all = torch.stack(per_step_acc, dim=1)

    log_q = torch.log(q_all.clamp(min=1e-10))
    S_t = torch.exp(torch.cumsum(log_q, dim=1))
    S_theta = S_t.sum(dim=1) / gamma

    valid_chain = torch.cumprod(acc_all.float(), dim=1)
    tau = valid_chain.sum(dim=1)

    m = loss_mask_1d
    return {
        'mean_S': S_theta[m].mean().item(),
        'mean_tau': tau[m].mean().item(),
        'n_positions': m.sum().item(),
        'step_acc': [acc_all[:, k][m].float().mean().item() for k in range(gamma)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--basepath', required=True)
    parser.add_argument('--draftpath', required=True)
    parser.add_argument('--testpath', required=True)
    parser.add_argument('--gamma', type=int, default=7)
    parser.add_argument('--max_len', type=int, default=2048)
    parser.add_argument('--max_samples', type=int, default=None,
                        help='None = use all test samples')
    parser.add_argument('--label', type=str, default='checkpoint')
    parser.add_argument('--config_path', type=str,
                        default=os.path.join(project_dir, 'sdpo', 'config.json'))
    args = parser.parse_args()

    device = torch.device('cuda:0')

    # Resolve HF repo IDs to local paths
    if not os.path.isdir(args.basepath):
        print(f"Downloading base model from HF: {args.basepath}")
        args.basepath = snapshot_download(args.basepath)
    if not os.path.isdir(args.draftpath):
        print(f"Downloading draft model from HF: {args.draftpath}")
        args.draftpath = snapshot_download(args.draftpath)

    tokenizer = AutoTokenizer.from_pretrained(args.basepath)
    draft_cfg = os.path.join(args.draftpath, "config.json")
    config = EConfig.from_pretrained(draft_cfg if os.path.exists(draft_cfg) else args.config_path)
    config.gradient_checkpointing = False

    train_ns = SimpleNamespace(
        bs=1, num_epochs=1, num_workers=0, max_len=args.max_len,
        config_path=args.config_path, gradient_checkpointing=False,
        eagle_coef=0, kl_coef=0, gamma=args.gamma, baseline=None,
    )
    ds_config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {"stage": 2},
    }

    model = Model(config, ds_config, train_ns, path=args.basepath,
                  load_emb=True, load_head=True)

    sf_path = os.path.join(args.draftpath, "model.safetensors")
    bin_path = os.path.join(args.draftpath, "pytorch_model.bin")
    if os.path.exists(sf_path):
        from safetensors.torch import load_file as sf_load
        state = sf_load(sf_path, device="cpu")
    elif os.path.exists(bin_path):
        state = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No weights in {args.draftpath}")
    for vk in ("d2t", "t2d"):
        state.pop(vk, None)
    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)

    draft_cache = os.path.join(args.draftpath, "cache.pt")
    if os.path.exists(draft_cache):
        shutil.copy(draft_cache, "cache.pt")
    model.scandata(args.testpath, args.basepath)
    model = model.to(device)
    model.eval()
    model.length = args.gamma

    # Build vocab mapping
    d2t = model.d2t.to(device)
    t2d = model.t2d.to(device)
    draft_ids = torch.arange(len(d2t), device=device)
    full_ids = draft_ids + d2t
    full2draft = torch.full((t2d.shape[0],), -1, dtype=torch.long, device=device)
    full2draft[full_ids] = draft_ids

    dataset = load_samples(tokenizer, args.testpath, args.max_len, args.max_samples)
    print(f"[{args.label}] Loaded {len(dataset)} test samples")

    all_S, all_tau, all_n = [], [], []
    all_step_acc = [[] for _ in range(args.gamma)]

    for i, sample in enumerate(dataset):
        if i % 100 == 0:
            print(f"  {i}/{len(dataset)}...")
        r = eval_one_sample(model, sample, args.gamma, device, full2draft, t2d, d2t)
        all_S.append(r['mean_S'] * r['n_positions'])
        all_tau.append(r['mean_tau'] * r['n_positions'])
        all_n.append(r['n_positions'])
        for k in range(args.gamma):
            all_step_acc[k].append(r['step_acc'][k])

    total_n = sum(all_n)
    mean_S = sum(all_S) / total_n
    mean_tau = sum(all_tau) / total_n

    print(f"\n{'='*50}")
    print(f"  Checkpoint: {args.label}")
    print(f"  Test samples: {len(dataset)}, positions: {total_n}")
    print(f"  mean_S  = {mean_S:.4f}")
    print(f"  mean_tau = {mean_tau:.4f} / {args.gamma}")
    print(f"  step_acc: {['%.4f' % np.mean(all_step_acc[k]) for k in range(args.gamma)]}")
    print(f"{'='*50}")

    result = {
        'label': args.label,
        'mean_S': mean_S,
        'mean_tau': mean_tau,
        'n_samples': len(dataset),
        'n_positions': total_n,
        'step_acc': [float(np.mean(all_step_acc[k])) for k in range(args.gamma)],
    }
    out_path = os.path.join(script_dir, f'test_loss_{args.label}.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
