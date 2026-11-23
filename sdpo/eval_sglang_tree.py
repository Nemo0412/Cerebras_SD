"""
Tree speculative decoding eval on a real serving engine (SGLang), concurrency 1.

eval_small_lm_tree.py measures accept length with a hand-written HF transformers
loop, so its tok/s says little about a serving stack. This script starts an
SGLang server per mode, replays the same benchmark prompts one request at a time
and reads accept length and throughput back from the server:

  tree    : --speculative-algorithm STANDALONE, eagle_topk > 1 (draft token tree)
  chain   : same algorithm, eagle_topk = 1, num_steps = --chain-steps
  vanilla : target model only (the speedup denominator)

--spec-algorithm EAGLE3 / EAGLE serves an EAGLE draft head instead of a small LM.

Prompts are tokenized exactly like eval_small_lm_tree.py (chat template of the
target, first turn only, no extra special tokens) and sent as input_ids.

Differences from eval_small_lm_tree.py that make alpha not directly comparable:
  * SGLang builds the tree EAGLE-style: every step expands the top-k children of
    the top-k frontier nodes, then reranks all nodes by joint draft probability
    and verifies the best --num-draft-tokens. There is no per-depth branching
    list and no path-topk union.
  * The draft is always deterministic top-k (no sampled draft).
  * T > 0 verification is SGLang's tree speculative sampling
    (accept thresholds default to 1.0 = lossless), not the rsd/ratio rules here.
  * Only per-request counters come back, so alpha is completion_tokens /
    verify steps; there is no per-window median / percentiles.

Usage (the interpreter given by --sglang-python must have sglang installed; it
may be a different env from the one running this script):
  python sdpo/eval_sglang_tree.py \\
    --base-model-path /path/to/Llama-3.1-8B-Instruct \\
    --draft-model-path /path/to/draft/hf_epoch_2 \\
    --bench-name mt_bench,gsm8k,humaneval --num-samples 80 \\
    --num-steps 7 --eagle-topk 4 --num-draft-tokens 29 \\
    --temperature 0.5 --modes tree,chain,vanilla --tag l3_e2_sglang
"""

import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from transformers import AutoTokenizer


def load_questions(bench_name, data_dir="data"):
    path = os.path.join(data_dir, bench_name, "question.jsonl")
    if not os.path.exists(path):
        print(f"  Bench {bench_name} not found at {path}, skipping")
        return []
    with open(path) as f:
        return [json.loads(l) for l in f]


def build_input_ids(tokenizer, q, enable_thinking):
    prompt = q.get("turns", [q.get("prompt", "")])[0] \
        if "turns" in q else q.get("prompt", "")
    msgs = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking)
    except TypeError:
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
    return prompt, tokenizer(text, add_special_tokens=False).input_ids


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http_json(url, payload=None, timeout=600):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return json.loads(body) if body else None


def server_cmd(args, mode, port):
    cmd = [args.sglang_python, "-m", "sglang.launch_server",
           "--model-path", args.base_model_path,
           "--host", "127.0.0.1", "--port", str(port),
           "--dtype", args.dtype,
           "--max-running-requests", "1",
           # SGLang sizes the target KV pool from free GPU memory and gives the
           # draft a pool with the same token count, which then does not fit.
           # One request at a time needs only a small pool.
           "--max-total-tokens", str(args.max_total_tokens)]
    if mode != "vanilla":
        topk = args.eagle_topk if mode == "tree" else 1
        steps = args.num_steps if mode == "tree" else args.chain_steps
        ndt = args.num_draft_tokens if mode == "tree" else steps + 1
        cmd += ["--speculative-algorithm", args.spec_algorithm,
                "--speculative-draft-model-path", args.draft_model_path,
                "--speculative-num-steps", str(steps),
                "--speculative-eagle-topk", str(topk),
                "--speculative-num-draft-tokens", str(ndt)]
    cmd += shlex.split(args.extra_server_args)
    return cmd


class Server:
    """One SGLang server in its own process group, torn down on exit."""

    def __init__(self, cmd, port, log_path, startup_timeout):
        self.base = f"http://127.0.0.1:{port}"
        self.log_path = log_path
        self.log = open(log_path, "w")
        print(f"[SGL-EVAL] launch: {' '.join(shlex.quote(c) for c in cmd)}")
        print(f"[SGL-EVAL] server log: {log_path}")
        # The server JIT-compiles kernels with tools installed next to its
        # interpreter (ninja), which are not on PATH unless the env is activated.
        env = dict(os.environ)
        env["PATH"] = (os.path.dirname(os.path.abspath(cmd[0]))
                       + os.pathsep + env.get("PATH", ""))
        self.proc = subprocess.Popen(
            cmd, stdout=self.log, stderr=subprocess.STDOUT, env=env,
            start_new_session=True)
        self._wait_ready(startup_timeout)

    def _wait_ready(self, timeout):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"SGLang server exited with code {self.proc.returncode} "
                    f"during startup; see {self.log_path}")
            try:
                http_json(self.base + "/health_generate", timeout=30)
                print(f"[SGL-EVAL] server ready after {time.time() - t0:.0f}s")
                return
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                time.sleep(3)
        self.close()
        raise RuntimeError(
            f"SGLang server not ready after {timeout}s; see {self.log_path}")

    def generate(self, input_ids, sampling_params):
        return http_json(self.base + "/generate",
                         {"input_ids": input_ids,
                          "sampling_params": sampling_params})

    def close(self):
        if self.proc.poll() is None:
            for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
                try:
                    os.killpg(self.proc.pid, sig)
                    self.proc.wait(timeout=wait)
                    break
                except subprocess.TimeoutExpired:
                    continue
                except ProcessLookupError:
                    break
        self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def run_bench(server, tokenizer, questions, args, mode, bench):
    sampling = {"temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens}
    for i in range(min(args.warmup, len(questions))):
        _, ids = build_input_ids(tokenizer, questions[i], args.enable_thinking)
        server.generate(ids, {**sampling, "max_new_tokens": 32})

    rows = []
    for qi, q in enumerate(questions):
        prompt, ids = build_input_ids(tokenizer, q, args.enable_thinking)
        t0 = time.perf_counter()
        out = server.generate(ids, sampling)
        wall = time.perf_counter() - t0
        meta = out["meta_info"]
        n_tok = meta["completion_tokens"]
        if mode != "vanilla" and not meta.get("spec_verify_ct"):
            raise RuntimeError(
                f"mode={mode} but the server reported no spec_verify_ct "
                f"(meta_info keys: {sorted(meta)}); speculative decoding is "
                f"not active or this SGLang version does not report it")
        rows.append({
            "question_id": q.get("question_id", qi),
            "prompt_tokens": len(ids),
            "new_tokens": n_tok,
            "verify_steps": meta.get("spec_verify_ct", 0),
            "accepted_draft_tokens": meta.get("spec_accept_token_num"),
            "wall_time": wall,
            "server_latency": meta.get("e2e_latency"),
            "finish_reason": (meta.get("finish_reason") or {}).get("type"),
        })
        if qi < args.print_samples:
            print(f"\n  --- {mode} sample {qi} ---")
            print(f"  QUESTION: {prompt}")
            print(f"  GEN ({n_tok} tok):\n{out['text']}\n")

    tok = sum(r["new_tokens"] for r in rows)
    steps = sum(r["verify_steps"] for r in rows)
    wall = sum(r["wall_time"] for r in rows)
    stats = {
        "num_questions": len(rows),
        "new_tokens": tok,
        "verify_steps": steps,
        # Tokens emitted per target forward, the same definition as the
        # accepted_length of eval_small_lm_tree.py (accepted draft tokens + 1).
        "accepted_length": tok / steps if steps else None,
        "tokens_per_sec": tok / wall if wall else 0.0,
        "wall_time": wall,
        "per_question": rows,
    }
    al = stats["accepted_length"]
    print(f"  {bench} {mode}: "
          + (f"α={al:.3f}  " if al else "")
          + f"tok/s={stats['tokens_per_sec']:.1f}  n={len(rows)}")
    return stats


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base-model-path', required=True)
    p.add_argument('--draft-model-path', required=True)
    p.add_argument('--bench-name', default='mt_bench')
    p.add_argument('--data-dir', default='data')
    p.add_argument('--num-samples', type=int, default=40)
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--modes', default='tree,chain,vanilla',
                   help='Comma-separated subset of tree,chain,vanilla; one '
                        'server launch each.')
    p.add_argument('--spec-algorithm', default='STANDALONE',
                   choices=['STANDALONE', 'EAGLE', 'EAGLE3'],
                   help='STANDALONE: --draft-model-path is a small LM. '
                        'EAGLE / EAGLE3: it is a draft head for the target.')
    p.add_argument('--num-steps', type=int, default=7,
                   help='Tree depth (SGLang --speculative-num-steps).')
    p.add_argument('--eagle-topk', type=int, default=4,
                   help='Branching factor per step (--speculative-eagle-topk).')
    p.add_argument('--num-draft-tokens', type=int, default=29,
                   help='Tree nodes sent to the target per step, root included '
                        '(--speculative-num-draft-tokens).')
    p.add_argument('--chain-steps', type=int, default=7,
                   help='Draft length of the chain mode.')
    p.add_argument('--dtype', default='float16')
    p.add_argument('--max-total-tokens', type=int, default=65536,
                   help='KV cache pool size in tokens (target and draft each).')
    p.add_argument('--warmup', type=int, default=2,
                   help='Untimed short requests per bench before measuring.')
    p.add_argument('--print-samples', type=int, default=1)
    p.add_argument('--enable-thinking', dest='enable_thinking',
                   action='store_true', default=True)
    p.add_argument('--no-thinking', dest='enable_thinking', action='store_false')
    p.add_argument('--sglang-python', default=sys.executable,
                   help='Python interpreter of the env that has sglang.')
    p.add_argument('--extra-server-args', default='',
                   help='Extra flags appended to sglang.launch_server, as one '
                        'shell-quoted string.')
    p.add_argument('--port', type=int, default=0, help='0 = pick a free port.')
    p.add_argument('--startup-timeout', type=int, default=1200)
    p.add_argument('--tag', default='sglang_tree')
    p.add_argument('--output-dir', default='smalllm_tree_eval_results')
    args = p.parse_args()

    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    bad = [m for m in modes if m not in ('tree', 'chain', 'vanilla')]
    if bad:
        p.error(f"unknown modes: {bad}")
    benches = ['mt_bench', 'gsm8k', 'humaneval', 'qa', 'sum', 'alpaca', 'aime'] \
        if args.bench_name == 'all' else \
        [b.strip() for b in args.bench_name.split(',') if b.strip()]

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path, trust_remote_code=True)
    bench_qs = {}
    for b in benches:
        qs = load_questions(b, args.data_dir)
        if qs:
            bench_qs[b] = qs[:args.num_samples] if args.num_samples else qs
    if not bench_qs:
        sys.exit("[SGL-EVAL] no benchmark questions found")

    os.makedirs(args.output_dir, exist_ok=True)
    results = {b: {} for b in bench_qs}
    for mode in modes:
        port = args.port or free_port()
        log_path = os.path.join(args.output_dir, f"{args.tag}_{mode}_server.log")
        print(f"\n[SGL-EVAL] ===== mode={mode} =====")
        with Server(server_cmd(args, mode, port), port, log_path,
                    args.startup_timeout) as server:
            for b, qs in bench_qs.items():
                results[b][mode] = run_bench(server, tokenizer, qs, args, mode, b)

    print("\n" + "=" * 70)
    print(f"  SGLANG TREE SPEC DECODE SUMMARY (c=1): {args.tag}")
    for b, per_mode in results.items():
        print(f"\n  {b}:")
        base = per_mode.get('vanilla', {}).get('tokens_per_sec')
        for mode in modes:
            s = per_mode[mode]
            line = f"    {mode:8s}"
            if s['accepted_length']:
                line += f" α={s['accepted_length']:.3f}"
            line += f"  tok/s={s['tokens_per_sec']:.1f}"
            if base and mode != 'vanilla':
                s['speedup_vs_vanilla'] = s['tokens_per_sec'] / base
                line += f"  speedup={s['speedup_vs_vanilla']:.2f}x"
            print(line)
    print("=" * 70)

    out_path = os.path.join(args.output_dir, f"{args.tag}.json")
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"[SGL-EVAL] Saved to {out_path}")


if __name__ == "__main__":
    main()
