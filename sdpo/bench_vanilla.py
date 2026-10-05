"""Vanilla auto-regressive speed benchmark (target only, no draft).
Uses same bench+sample+max_new_tokens as eval_small_lm_tree.py for fair comparison.
"""
import argparse, json, os, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_questions(bench_name, data_dir="data"):
    path = os.path.join(data_dir, bench_name, "question.jsonl")
    if not os.path.exists(path): return []
    with open(path) as f:
        return [json.loads(l) for l in f]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target-model', required=True)
    p.add_argument('--bench-name', default='mt_bench,gsm8k,humaneval')
    p.add_argument('--num-samples', type=int, default=40)
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--data-dir', default='data')
    p.add_argument('--tag', default='vanilla_bench')
    p.add_argument('--output-dir', default='smalllm_tree_eval_results')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--enable-thinking', dest='enable_thinking', action='store_true', default=True)
    p.add_argument('--no-thinking', dest='enable_thinking', action='store_false')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    print(f"[VANILLA] Loading target {args.target_model}")
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=torch.float16,
        attn_implementation='sdpa').cuda()
    target.eval()
    tok = AutoTokenizer.from_pretrained(args.target_model)

    all_results = {}
    for bench in args.bench_name.split(','):
        questions = load_questions(bench, args.data_dir)
        if not questions:
            print(f"[VANILLA] {bench} not found, skip"); continue
        questions = questions[:args.num_samples]
        print(f"[VANILLA] {bench}: {len(questions)} samples, max_new_tokens={args.max_new_tokens}")

        # Warmup on one sample
        prompt = questions[0].get("turns", [questions[0].get("prompt", "")])[0] \
            if "turns" in questions[0] else questions[0].get("prompt", "")
        msgs = [{"role": "user", "content": prompt}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=args.enable_thinking)
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        with torch.inference_mode():
            target.generate(ids, max_new_tokens=16, do_sample=False, temperature=1.0)
        torch.cuda.synchronize()

        # Bench loop
        total_gen = 0
        total_time = 0.0
        n_printed = 0
        for qi, q in enumerate(questions):
            prompt = q.get("turns", [q.get("prompt", "")])[0] \
                if "turns" in q else q.get("prompt", "")
            msgs = [{"role": "user", "content": prompt}]
            try:
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=args.enable_thinking)
            except TypeError:
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.inference_mode():
                out = target.generate(
                    ids, max_new_tokens=args.max_new_tokens,
                    do_sample=(args.temperature > 0), temperature=max(args.temperature, 1e-6),
                    top_p=1.0, top_k=0,
                    pad_token_id=tok.eos_token_id if tok.pad_token_id is None else tok.pad_token_id,
                )
            torch.cuda.synchronize()
            elapsed = time.time() - t0
            n_gen = out.shape[1] - ids.shape[1]
            total_gen += n_gen
            total_time += elapsed
            if n_printed < 1:
                n_printed += 1
                gen_txt = tok.decode(out[0, ids.shape[1]:].tolist(), skip_special_tokens=True)
                print(f"  --- sample {qi} ({n_gen} tok in {elapsed:.2f}s) ---")
                print(f"  GEN: {gen_txt[:200]}")
        tps = total_gen / max(total_time, 1e-6)
        print(f"[VANILLA] {bench}  tok/s={tps:.2f}  total_time={total_time:.1f}s  gen={total_gen}")
        all_results[bench] = {"vanilla": {
            "tokens_per_sec": tps,
            "total_time": total_time,
            "total_gen": total_gen,
            "num_samples": len(questions),
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
        }}

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.tag}.json")
    with open(out_path, "w") as f:
        json.dump({"tag": args.tag, "results": all_results}, f, indent=2)
    print(f"[VANILLA] Saved to {out_path}")


if __name__ == "__main__":
    main()
