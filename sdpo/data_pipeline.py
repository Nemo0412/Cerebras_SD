"""
Shared ShareGPT data pipeline.

Used by main_small_lm.py and main_layer_skip.py. Extracted verbatim from the
original inline implementation in main_small_lm.py so multiple entry points
can construct train/eval datasets without duplicating the tokenization +
loss-mask logic.
"""

import torch
from datasets import load_dataset


SYSTEM_MSG = (
    "You are a helpful, respectful and honest assistant. Always answer as "
    "helpfully as possible, while being safe.  Your answers should not "
    "include any harmful, unethical, racist, sexist, toxic, dangerous, or "
    "illegal content. Please ensure that your responses are socially "
    "unbiased and positive in nature.\n\nIf a question does not make any "
    "sense, or is not factually coherent, explain why instead of answering "
    "something not correct. If you don't know the answer to a question, "
    "please don't share false information."
)

SEP_TOKENS = {
    "llama": {
        "sep_asst": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "sep_user": "<|eot_id|><|start_header_id|>user<|end_header_id|>",
        "use_system": True,
    },
    "qwen": {
        "sep_asst": "<|im_start|>assistant\n",
        "sep_user": "<|im_start|>user\n",
        "use_system": False,
    },
}


def detect_model_family(tokenizer):
    name = getattr(tokenizer, 'name_or_path', '').lower()
    if 'qwen' in name:
        return 'qwen'
    return 'llama'


def build_dataset(tokenizer, datapath, max_len, max_samples=None, gamma=7):
    ds = load_dataset('json', data_files=datapath)['train']
    ds = ds.shuffle(seed=42)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    family = detect_model_family(tokenizer)
    sep_cfg = SEP_TOKENS[family]
    SEP_ASST = sep_cfg["sep_asst"]
    SEP_USER = sep_cfg["sep_user"]
    use_system = sep_cfg["use_system"]

    def preprocess(examples):
        out = {"attention_mask": [], "input_ids": [], "loss_mask": []}
        roles = {"human": "user", "gpt": "assistant"}

        for i in range(len(examples['id'])):
            src = examples['conversations'][i]
            if not src:
                continue
            if roles.get(src[0]["from"]) != "user":
                src = src[1:]

            msgs = []
            if use_system:
                msgs.append({"role": "system", "content": SYSTEM_MSG})
            for j, sent in enumerate(src):
                msgs.append({"role": roles[sent["from"]], "content": sent["value"]})

            try:
                conversation = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False,
                    enable_thinking=True)
            except TypeError:
                conversation = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False)
            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id = tokenizer.unk_token_id
            ids = tokenizer(conversation, return_tensors="pt",
                            add_special_tokens=False).input_ids[0]
            if len(ids) > max_len:
                if max_samples is not None:
                    ids = ids[:max_len]
                else:
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

            # Skip samples that won't produce rank-consistent training
            # signal. See main_small_lm.py for the three conditions.
            L = len(ids)
            if L <= gamma:
                continue
            if loss_mask[1:L - gamma + 1].sum().item() == 0:
                continue

            out["input_ids"].append(ids.tolist())
            out["loss_mask"].append(loss_mask.tolist())
            out["attention_mask"].append([1] * len(ids))
        return out

    ds = ds.map(preprocess, batched=True, num_proc=1,
                remove_columns=ds.column_names)
    return ds


class DataCollator:
    def __call__(self, features):
        max_len = max(len(f['input_ids']) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "loss_mask": []}
        for f in features:
            for key in batch:
                vals = f[key]
                if not isinstance(vals, list):
                    vals = vals.tolist()
                pad = max_len - len(vals)
                batch[key].append(torch.tensor(vals + [0] * pad, dtype=torch.long))
        return {k: torch.stack(v) for k, v in batch.items()}
