"""One-time remap of Gemma 4 multimodal HF checkpoints → Gemma4ForCausalLM layout.

Multimodal checkpoint state_dict keys:
  model.language_model.embed_tokens.weight
  model.language_model.layers.<L>.self_attn.q_proj.weight
  ...
  model.vision_tower.<...>      (skip)
  model.audio_tower.<...>       (skip)
  model.multimodal_projector.<...>  (skip)
  model.embed_audio.<...>       (skip)
  model.embed_vision.<...>      (skip)

Remapped (Gemma4ForCausalLM layout):
  model.embed_tokens.weight
  model.layers.<L>.self_attn.q_proj.weight
  ...
  (no vision/audio/projector)

Writes the remapped checkpoint to:
  <cache_root>/<safe_name>/{config.json, generation_config.json, model.safetensors.index.json, *.safetensors, tokenizer files}

Idempotent — skips remap if cache_root/<safe_name>/config.json already exists.

Usage:
  python sdpo/data/remap_gemma4_to_causal.py \
      --model google/gemma-4-31B-it \
      --cache-root /scratch/tx856/.cache/gemma4_causal
"""
import argparse
import glob
import json
import os
import shutil

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from transformers import AutoConfig


SKIP_PREFIXES = (
    "model.vision_tower.",
    "model.audio_tower.",
    "model.multimodal_projector.",
    "model.embed_audio.",
    "model.embed_vision.",
)


def remap_key(k: str) -> str | None:
    if any(k.startswith(p) for p in SKIP_PREFIXES):
        return None
    if k.startswith("model.language_model."):
        return "model." + k[len("model.language_model."):]
    return k


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--cache-root",
                   default="/scratch/tx856/.cache/gemma4_causal")
    p.add_argument("--force", action="store_true")
    p.add_argument("--shard-bytes", type=int, default=int(5e9),
                   help="Max bytes per output safetensors shard.")
    args = p.parse_args()

    safe = args.model.replace("/", "_")
    out_dir = os.path.join(args.cache_root, safe)
    if os.path.exists(os.path.join(out_dir, "config.json")) and not args.force:
        print(f"[REMAP] cache hit at {out_dir}; skipping (use --force to redo)")
        return

    os.makedirs(out_dir, exist_ok=True)

    print(f"[REMAP] downloading {args.model} ...")
    local = snapshot_download(args.model, allow_patterns=[
        "*.safetensors", "*.json", "*.txt", "*.model"])

    # ── Read + remap state_dict ──
    shards = sorted(glob.glob(os.path.join(local, "model*.safetensors")))
    print(f"[REMAP] {len(shards)} source shards")
    state = {}
    for f in shards:
        st = load_file(f, device="cpu")
        for k, v in st.items():
            nk = remap_key(k)
            if nk is None:
                continue
            # de-dup: tied lm_head ↔ embed often shares storage; safe to overwrite.
            state[nk] = v
    print(f"[REMAP] {len(state)} kept keys after remap")

    # ── UNTIE lm_head from embed_tokens ──
    # Gemma 4 ships tied (tie_word_embeddings=True). Under DeepSpeed ZeRO-3,
    # a single nn.Parameter shared between embed_tokens (in TextModel) and
    # lm_head produces the wrong gather on a FROZEN target (target_lg saturates
    # at ±30 softcap on garbage argmax → mean_tau=0 in training).
    # Qwen3 Q0.6B/Q4B are also tied but are TRAINABLE drafts — the backward
    # pass clears ZeRO-3 gather state per step, masking the same bug. Gemma 4
    # 31B is a tied + frozen target, no backward → bug surfaces.
    # Fix: store lm_head.weight as a SEPARATE tensor copy in the cache and
    # set tie_word_embeddings=False in the config so ZeRO-3 partitions them
    # independently. Cost: +1 vocab × hidden bf16 (≈0.8 GB E2B, ≈2.8 GB 31B).
    embed_w = state.get("model.embed_tokens.weight")
    if embed_w is not None and "lm_head.weight" not in state:
        state["lm_head.weight"] = embed_w.clone()
        print(f"[REMAP] untied lm_head ← embed_tokens copy "
              f"({embed_w.shape}, {embed_w.dtype})")

    # ── Save shards ──
    cur, cur_bytes = {}, 0
    shard_idx = 0
    out_shards = []

    def flush():
        nonlocal cur, cur_bytes, shard_idx
        if not cur:
            return
        path = os.path.join(out_dir, f"model-{shard_idx:05d}.safetensors")
        save_file(cur, path)
        out_shards.append((path, list(cur.keys())))
        print(f"[REMAP] wrote {path} ({cur_bytes/1e9:.2f} GB, {len(cur)} keys)")
        cur, cur_bytes = {}, 0
        shard_idx += 1

    # Keep insertion order roughly: embed first, then layers
    for k in sorted(state.keys()):
        v = state[k]
        nb = v.numel() * v.element_size()
        if cur_bytes + nb > args.shard_bytes and cur:
            flush()
        cur[k] = v
        cur_bytes += nb
    flush()

    # ── Index file ──
    weight_map = {}
    total = 0
    for path, ks in out_shards:
        rel = os.path.basename(path)
        for k in ks:
            weight_map[k] = rel
            total += state[k].numel() * state[k].element_size()
    index = {
        "metadata": {"total_size": total},
        "weight_map": weight_map,
    }
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    # ── Config: convert text_config → top-level Gemma4ForCausalLM ──
    cfg = AutoConfig.from_pretrained(local)
    text_cfg = cfg.text_config.to_dict()
    text_cfg["architectures"] = ["Gemma4ForCausalLM"]
    text_cfg["_attn_implementation"] = "sdpa"
    text_cfg.pop("model_type", None)
    text_cfg["model_type"] = "gemma4_text"
    # Untie lm_head from embed_tokens (we added a separate copy above).
    text_cfg["tie_word_embeddings"] = False
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(text_cfg, f, indent=2)

    # ── Generation config + tokenizer files (copy) ──
    for fname in ("generation_config.json", "tokenizer.json",
                  "tokenizer_config.json", "tokenizer.model",
                  "special_tokens_map.json", "added_tokens.json",
                  "chat_template.jinja"):
        src = os.path.join(local, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out_dir, fname))

    print(f"[REMAP] done -> {out_dir}")


if __name__ == "__main__":
    main()
