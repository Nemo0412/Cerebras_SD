#!/usr/bin/env python3
"""Probe a draft checkpoint under sample_rsd tree eval (top-4 path AL).

Does NOT import main_small_lm.py (that module runs training on import).
Writes <savedir>/tree_path_probe.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Prefer repo checkout that contains this file: docs/demo_test/scripts/ → repo root.
_REPO_CANDIDATES = [
    os.environ.get("CEREBRAS_SD_REPO", ""),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
    "/home/lls/workspace/Cerebras_SD",
]
REPO = next(
    (p for p in _REPO_CANDIDATES
     if p and os.path.isdir(os.path.join(p, "sdpo"))),
    _REPO_CANDIDATES[-1],
)
sys.path.insert(0, os.path.join(REPO, "sdpo"))
from small_lm_tree_v2_model import SmallLMTreeV2Model  # noqa: E402


SEP_ASST = "<|im_start|>assistant\n"
SEP_USER = "<|im_start|>user\n"


def build_dataset(tokenizer, datapath, max_len, max_samples=None, gamma=7, seed=0):
    ds = load_dataset("json", data_files=datapath)["train"]
    ds = ds.shuffle(seed=seed)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    roles = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}

    def preprocess(examples):
        out = {"attention_mask": [], "input_ids": [], "loss_mask": []}
        for i in range(len(examples["id"])):
            src = examples["conversations"][i]
            if not src:
                continue
            if roles.get(src[0]["from"]) != "user":
                src = src[1:]
            msgs = [{"role": roles[s["from"]], "content": s["value"]} for s in src]
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
                ids = ids[:max_len]
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
            L = len(ids)
            if L <= gamma or loss_mask[1:L - gamma + 1].sum().item() == 0:
                continue
            out["input_ids"].append(ids.tolist())
            out["loss_mask"].append(loss_mask.tolist())
            out["attention_mask"].append([1] * len(ids))
        return out

    return ds.map(preprocess, batched=True, num_proc=1,
                  remove_columns=ds.column_names)


class DataCollator:
    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "loss_mask": []}
        for f in features:
            for k in batch:
                vals = f[k]
                pad = max_len - len(vals)
                if k == "input_ids":
                    batch[k].append(vals + [0] * pad)
                else:
                    batch[k].append(vals + [0] * pad)
        return {k: torch.tensor(v) for k, v in batch.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basepath", required=True)
    ap.add_argument("--draftpath", required=True)
    ap.add_argument("--datapath", required=True)
    ap.add_argument("--savedir", required=True)
    ap.add_argument("--tag", default="probe")
    ap.add_argument("--step", type=int, default=None,
                    help="Optional training step to record in the jsonl row "
                         "(for offline probe curves).")
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--max_samples", type=int, default=64)
    ap.add_argument("--top_k_per_depth", default="4,3,2,1,1,1,1")
    ap.add_argument("--path_topk", type=int, default=4)
    ap.add_argument("--n_anchors", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    top_k = [int(x) for x in args.top_k_per_depth.split(",") if x.strip()]
    tok = AutoTokenizer.from_pretrained(args.basepath, trust_remote_code=True)
    ds = build_dataset(tok, args.datapath, args.max_len,
                       max_samples=args.max_samples, gamma=len(top_k),
                       seed=args.seed)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=DataCollator())

    model = SmallLMTreeV2Model(
        args.basepath, args.draftpath, top_k, dtype=torch.bfloat16,
        tree_budget=None,
    ).to(device)
    model.eval()
    model.srsd_tgt_cache = True

    sums = {"tree_probe_eal": 0.0}
    for i in range(1, args.path_topk + 1):
        sums[f"path{i}_al"] = 0.0
        sums[f"path{i}_eal"] = 0.0
    n = 0
    with torch.no_grad():
        for data in loader:
            _t, _a, m = model(
                input_ids=data["input_ids"].to(device),
                attention_mask=data["attention_mask"].to(device),
                loss_mask=data["loss_mask"].to(device),
                anchor_kl_coef=0.0,
                eal_aux_coef=1.0,
                temperature=args.temperature,
                eal_mode="sample_rsd",
                path_topk=args.path_topk,
                n_anchors=args.n_anchors,
                n_draws=1,
            )
            if m.get("num_valid", 0) <= 0:
                continue
            n += 1
            sums["tree_probe_eal"] += float(m["eal_mean"])
            for i in range(1, args.path_topk + 1):
                sums[f"path{i}_al"] += float(m.get(f"path{i}_al", 0.0))
                sums[f"path{i}_eal"] += float(m.get(f"path{i}_eal", 0.0))
            print(f"  probe {n}/{args.max_samples} eal={m['eal_mean']:.3f}",
                  flush=True)
    if n == 0:
        raise SystemExit("no valid probe batches")
    row = {"tag": args.tag, "draft": args.draftpath, "n_batches": n}
    if args.step is not None:
        row["step"] = int(args.step)
    for k, v in sums.items():
        row[k] = v / n
    os.makedirs(args.savedir, exist_ok=True)
    out = os.path.join(args.savedir, "tree_path_probe.jsonl")
    with open(out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))
    print("appended", out)


if __name__ == "__main__":
    main()
