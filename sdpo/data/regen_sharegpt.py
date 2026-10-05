"""Regenerate ShareGPT assistant turns with a target LLM via vLLM.

Adapted from GTO/ge_data/regeneratedata.py — same chat-format / batching
logic, but with argparse so the model, source dataset, and output path are
configurable from a slurm wrapper instead of hardcoded.

Output: JSONL where each line is {"id": str, "conversations": [{"from","value"}]}.
"""

import argparse
import gc
import json
import os
from typing import Dict, List

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as "
    "helpfully as possible, while being safe. Your answers should not include "
    "any harmful, unethical, racist, sexist, toxic, dangerous, or illegal "
    "content. Please ensure that your responses are socially unbiased and "
    "positive in nature.\n\nIf a question does not make any sense, or is not "
    "factually coherent, explain why instead of answering something not "
    "correct. If you don't know the answer to a question, please don't share "
    "false information."
)


class ShareGPTProcessor:
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int,
        max_model_len: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        gpu_memory_utilization: float,
        stop_token_ids: List[int],
        dtype: str,
        system_prompt: str,
    ):
        self.model_path = model_path
        self.system_prompt = system_prompt

        # Load tokenizer for model-agnostic chat templating (Llama-2 / Llama-3
        # / Qwen all use different chat formats; tokenizer.apply_chat_template
        # dispatches correctly). Falls back to hand-written Llama-3 tags
        # if tokenizer has no chat_template set.
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self._has_chat_template = bool(getattr(self.tokenizer, "chat_template", None))

        # Auto-detect stop tokens from tokenizer if caller passed empty list.
        if not stop_token_ids:
            eos = self.tokenizer.eos_token_id
            stop_token_ids = ([eos] if isinstance(eos, int) else list(eos)) if eos is not None else []
            # Llama-3 family: include <|eot_id|> (128009) explicitly since it
            # ends each turn; <|end_of_text|> (128001) ends the full sequence.
            eot = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if isinstance(eot, int) and eot != self.tokenizer.unk_token_id and eot not in stop_token_ids:
                stop_token_ids.append(eot)

        print(f"Initializing vLLM (TP={tensor_parallel_size}, dtype={dtype}, "
              f"max_model_len={max_model_len})...")
        print(f"stop_token_ids (auto-detected if was empty): {stop_token_ids}")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_token_ids=stop_token_ids,
        )

    def format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """Render a conversation as a single prompt string.

        Prepends `self.system_prompt` as a system message (caller passes only
        user/assistant turns). Uses `tokenizer.apply_chat_template` when the
        tokenizer has a chat_template (works for Llama-2 / Llama-3 / Qwen /
        etc.); falls back to hand-written Llama-3 tags otherwise.
        """
        with_sys = [{"role": "system", "content": self.system_prompt}] + messages
        if self._has_chat_template:
            try:
                return self.tokenizer.apply_chat_template(
                    with_sys, tokenize=False, add_generation_prompt=True,
                    enable_thinking=True)
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    with_sys, tokenize=False, add_generation_prompt=True)
        # Fallback: hand-written Llama-3 chat tags (legacy behavior).
        formatted = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{self.system_prompt}<|eot_id|>"
        )
        for msg in messages:
            formatted += (
                f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n"
                f"{msg['content']}<|eot_id|>"
            )
        formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return formatted

    def prepare_prompts_batch(self, batch: List[Dict]):
        prompts, meta = [], []
        n_skipped = 0
        for item_idx, item in enumerate(batch):
            history = []
            conv = item["conversations"]
            for turn_idx, turn in enumerate(conv):
                if turn["from"] in ("user", "human"):
                    messages = history + [{"role": "user", "content": turn["value"]}]
                    # Wrap format_conversation in try/except: some samples
                    # have malformed role alternation (e.g. consecutive
                    # human/human in raw ShareGPT) that Llama-2's strict
                    # chat template rejects. Skip the offending prompt
                    # instead of aborting the whole batch.
                    try:
                        rendered = self.format_conversation(messages)
                    except Exception:
                        n_skipped += 1
                    else:
                        prompts.append(rendered)
                        meta.append({"item_idx": item_idx, "turn_idx": turn_idx})
                    history.append({"role": "user", "content": turn["value"]})
                    # ShareGPT V4.3 uses `from="gpt"`; some other variants use
                    # `from="assistant"`. Accept both to keep history strictly
                    # alternating (Llama-2 chat template requires this).
                    if turn_idx + 1 < len(conv) and conv[turn_idx + 1]["from"] in ("assistant", "gpt"):
                        history.append({"role": "assistant",
                                        "content": conv[turn_idx + 1]["value"]})
        if n_skipped > 0:
            print(f"  prepare_prompts_batch: skipped {n_skipped} prompts due to chat-template error")
        return prompts, meta

    def process_batch(self, batch: List[Dict]) -> List[Dict]:
        prompts, meta = self.prepare_prompts_batch(batch)
        if not prompts:
            return batch
        outputs = self.llm.generate(prompts, self.sampling_params)
        response_map = {(m["item_idx"], m["turn_idx"]): o.outputs[0].text.strip()
                        for o, m in zip(outputs, meta)}
        out = []
        for item_idx, item in enumerate(batch):
            new_conv = []
            for turn_idx, turn in enumerate(item["conversations"]):
                if turn["from"] in ("user", "human"):
                    new_conv.append(turn)
                    key = (item_idx, turn_idx)
                    if key in response_map:
                        new_conv.append({"from": "assistant", "value": response_map[key]})
            out.append({
                "id": item.get("id", f"conversation_{item_idx}"),
                "conversations": new_conv,
            })
        return out

    def process_dataset(self, data_path: str, output_path: str, batch_size: int,
                        max_samples: int, checkpoint_every: int):
        print(f"Loading ShareGPT dataset from {data_path} ...")
        if os.path.isdir(data_path) or data_path.startswith("hf://"):
            ds = load_dataset(data_path)["train"]
        else:
            ds = load_dataset("json", data_files=data_path)["train"]
        ds = ds.shuffle(seed=42)
        if max_samples and max_samples > 0:
            ds = ds.select(range(min(max_samples, len(ds))))
        print(f"Dataset loaded: {len(ds)} conversations")

        items = []
        for i, item in enumerate(ds):
            if "id" not in item:
                item["id"] = f"sharegpt_{i:06d}"
            items.append(item)

        regenerated, written = [], 0
        with tqdm(total=len(items), desc="regen") as pbar:
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                try:
                    regenerated.extend(self.process_batch(batch))
                except Exception as e:
                    print(f"\nbatch error: {e!r}; passing through original turns")
                    for it in batch:
                        regenerated.append({"id": it.get("id", f"err_{i}"),
                                            "conversations": it["conversations"]})
                pbar.update(len(batch))

                if checkpoint_every and len(regenerated) - written >= checkpoint_every:
                    tmp = output_path.replace(".jsonl", f".ckpt{len(regenerated)}.jsonl")
                    _save_jsonl(regenerated, tmp)
                    written = len(regenerated)
                    print(f"\n  checkpoint: {written} written -> {tmp}")
                    gc.collect()
                    torch.cuda.empty_cache()

        _save_jsonl(regenerated, output_path)
        print(f"Done: {len(regenerated)} conversations -> {output_path}")


def _save_jsonl(data: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="HF id or local path of target")
    p.add_argument("--data", required=True, help="Source ShareGPT JSON / HF id")
    p.add_argument("--output", required=True, help="Output JSONL path")
    p.add_argument("--tensor-parallel-size", type=int, default=4)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float16", "bfloat16", "auto"])
    # Default empty → auto-detected from tokenizer.eos_token_id (+ <|eot_id|>
    # if model has it). Pass explicitly to override (e.g. "128001,128009" for
    # Llama-3, "2" for Llama-2 chat).
    p.add_argument("--stop-token-ids", type=_parse_int_list, default=[])
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--max-samples", type=int, default=0, help="0 = full dataset")
    p.add_argument("--checkpoint-every", type=int, default=10000)
    p.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print("=" * 30)
    print(f"  Regenerate ShareGPT with {args.model}")
    print(f"  SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID', 'n/a')}")
    print("=" * 30)
    print(f"Data:           {args.data}")
    print(f"Output:         {args.output}")
    print(f"TP:             {args.tensor_parallel_size}")
    print(f"max_tokens:     {args.max_tokens}")
    print(f"max_model_len:  {args.max_model_len}")
    print(f"stop_token_ids: {args.stop_token_ids}")

    proc = ShareGPTProcessor(
        model_path=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        gpu_memory_utilization=args.gpu_memory_utilization,
        stop_token_ids=args.stop_token_ids,
        dtype=args.dtype,
        system_prompt=args.system_prompt,
    )
    proc.process_dataset(
        data_path=args.data,
        output_path=args.output,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
