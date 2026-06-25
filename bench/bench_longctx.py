"""Long-context streaming benchmark: isolates prefill (attention-heavy, O(n^2))
and long-context decode, where the attention backend actually matters.

Reports TTFT (prefill latency) and decode TPOT/throughput separately.

Usage:
  python bench/bench_longctx.py --port 8001 --label plugin --input-len 6000 \
      --output-len 256 --out results/x.json
"""
import argparse
import json
import time
import threading
import urllib.request


def build_prompt(approx_tokens: int, nonce: int = 0) -> str:
    # Build a long passage. The nonce makes EVERY chunk unique per request so
    # vLLM's prefix cache cannot serve the prefill from cache -> we measure
    # real prefill work (where the attention backend matters).
    chunk = (
        "In session {n} section {i}, system {n} processes tensor block {k} with "
        "stride {j} and accumulates partial sums across the {k}th attention head, "
        "then normalizes the result before passing it to layer {j}. "
    )
    parts = []
    i = 0
    # ~40 tokens per chunk (measured); target token count / 40 chunks
    while i < approx_tokens // 40:
        parts.append(chunk.format(n=nonce, i=i, j=(i * 7) % 101, k=(i * 13 + nonce) % 100003))
        i += 1
    parts.append("\n\nNow write a detailed multi-paragraph technical summary of the above process:")
    return "".join(parts)


def stream_request(port, model, prompt, max_tokens):
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=body, headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    n_tok = 0
    last = t0
    prompt_tokens = None
    completion_tokens = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            ch = obj.get("choices") or []
            if ch and ch[0].get("text"):
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t0
                n_tok += 1
                last = now
            if obj.get("usage"):
                prompt_tokens = obj["usage"].get("prompt_tokens")
                completion_tokens = obj["usage"].get("completion_tokens")
    end = time.perf_counter()
    decode_time = max(last - (t0 + (ttft or 0)), 1e-6)
    gen = completion_tokens or n_tok
    decode_tok_s = (gen - 1) / decode_time if gen > 1 else 0.0
    return {
        "ttft_s": ttft,
        "total_s": end - t0,
        "decode_time_s": decode_time,
        "decode_tok_s": decode_tok_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": gen,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--label", default="run")
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    ap.add_argument("--input-len", type=int, default=6000)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--concurrency", type=int, nargs="*", default=[1, 8])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    nonce = [1000]
    def fresh_prompt():
        nonce[0] += 1
        return build_prompt(args.input_len, nonce[0])

    # warmup (also reports the real prompt token count)
    w = stream_request(args.port, args.model, fresh_prompt(), 16)
    print(f"prompt_tokens≈{w['prompt_tokens']}  (target input-len {args.input_len}, "
          f"prefix-cache defeated via per-request nonce)")

    print(f"== single-stream x{args.reps}: input≈{w['prompt_tokens']}tok, output={args.output_len} ==")
    singles = [stream_request(args.port, args.model, fresh_prompt(), args.output_len) for _ in range(args.reps)]
    def med(key):
        vals = sorted(s[key] for s in singles)
        return vals[len(vals) // 2]
    ttft_med = med("ttft_s"); dec_med = med("decode_tok_s")
    for s in singles:
        print(f"   TTFT {s['ttft_s']*1000:7.1f}ms  decode {s['decode_tok_s']:6.2f} tok/s  "
              f"({s['completion_tokens']} tok, total {s['total_s']:.2f}s)")
    print(f"   median TTFT: {ttft_med*1000:.1f} ms | median decode: {dec_med:.2f} tok/s")

    sweep = []
    for n in args.concurrency:
        results = [None] * n
        prompts = [fresh_prompt() for _ in range(n)]
        def worker(i):
            results[i] = stream_request(args.port, args.model, prompts[i], args.output_len)
        t0 = time.perf_counter()
        ths = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in ths: t.start()
        for t in ths: t.join()
        wall = time.perf_counter() - t0
        tot = sum(r["completion_tokens"] for r in results)
        agg = tot / wall
        ttfts = sorted(r["ttft_s"] for r in results)
        print(f"== concurrency={n}: {agg:.1f} tok/s aggregate, "
              f"median TTFT {ttfts[len(ttfts)//2]*1000:.1f}ms, wall {wall:.2f}s ==")
        sweep.append({"concurrency": n, "aggregate_tok_s": agg,
                      "median_ttft_ms": ttfts[len(ttfts)//2]*1000, "wall_s": wall})

    summary = {
        "label": args.label, "model": args.model,
        "input_len_target": args.input_len,
        "prompt_tokens": w["prompt_tokens"], "output_len": args.output_len,
        "median_ttft_ms": ttft_med * 1000,
        "median_decode_tok_s": dec_med,
        "single_runs": singles, "concurrency_sweep": sweep,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
