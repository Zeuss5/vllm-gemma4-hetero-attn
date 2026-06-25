"""Simple, reproducible decode-throughput benchmark against an OpenAI-compatible
vLLM server. Measures single-stream tok/s and a small concurrency sweep.

Usage:
  python bench/bench_decode.py --port 8001 --label baseline --out results/x.json
"""
import argparse
import json
import time
import threading
import urllib.request


PROMPT = (
    "You are a careful technical writer. Write a detailed, well-structured "
    "explanation of how modern GPUs execute attention in transformer models, "
    "covering memory hierarchy, tiling, and why FlashAttention is fast. "
    "Be thorough and continue until you have written several paragraphs."
)


def one_request(port, max_tokens, prompt=PROMPT):
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=body, headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    dt = time.perf_counter() - t0
    usage = out.get("usage", {})
    return {
        "latency_s": dt,
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
    }


def run_concurrency(port, n, max_tokens):
    results = [None] * n
    threads = []

    def worker(i):
        results[i] = one_request(port, max_tokens)

    t0 = time.perf_counter()
    for i in range(n):
        th = threading.Thread(target=worker, args=(i,))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    wall = time.perf_counter() - t0
    total_completion = sum(r["completion_tokens"] for r in results)
    return {
        "concurrency": n,
        "wall_s": wall,
        "total_completion_tokens": total_completion,
        "aggregate_tok_s": total_completion / wall,
        "per_req": results,
    }


def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--label", default="run")
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--single-reps", type=int, default=5)
    ap.add_argument("--concurrency", type=int, nargs="*", default=[1, 4, 8])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    MODEL = args.model

    print(f"== warmup ({args.warmup}) ==")
    for _ in range(args.warmup):
        one_request(args.port, 32)

    # Single-stream decode tok/s (the cleanest signal for attention backend).
    print(f"== single-stream x{args.single_reps}, max_tokens={args.max_tokens} ==")
    singles = [one_request(args.port, args.max_tokens) for _ in range(args.single_reps)]
    tok_s = [r["completion_tokens"] / r["latency_s"] for r in singles]
    tok_s_sorted = sorted(tok_s)
    median = tok_s_sorted[len(tok_s_sorted) // 2]
    for r, t in zip(singles, tok_s):
        print(f"   {r['completion_tokens']:4d} tok in {r['latency_s']:6.2f}s -> {t:6.2f} tok/s")
    print(f"   single-stream median: {median:.2f} tok/s")

    sweep = []
    for n in args.concurrency:
        res = run_concurrency(args.port, n, args.max_tokens)
        print(f"== concurrency={n}: {res['aggregate_tok_s']:.2f} tok/s aggregate "
              f"({res['total_completion_tokens']} tok in {res['wall_s']:.2f}s) ==")
        res.pop("per_req")
        sweep.append(res)

    summary = {
        "label": args.label,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "single_stream_tok_s_median": median,
        "single_stream_tok_s_all": tok_s,
        "concurrency_sweep": sweep,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print("wrote", args.out)
    return summary


if __name__ == "__main__":
    main()
