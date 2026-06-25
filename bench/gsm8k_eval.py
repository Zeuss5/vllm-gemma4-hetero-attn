"""GSM8K accuracy eval against an OpenAI-compatible vLLM server.

Standard 8-shot chain-of-thought, greedy, flexible last-number extraction
(matching the common lm-eval "gsm8k" flexible-extract protocol closely enough
for an A/B parity check between attention backends).

Usage:
  python bench/gsm8k_eval.py --port 8001 --label plugin --limit 1319 --out results/gsm8k_plugin.json
"""
import argparse
import json
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from datasets import load_dataset

# Canonical 8-shot CoT exemplars (Wei et al. / lm-eval gsm8k).
FEWSHOT = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.

"""

ANS_RE = re.compile(r"(-?[\d,]*\.?\d+)")


def extract_pred(text: str):
    # Prefer the number after "The answer is", else the last number seen.
    m = re.search(r"answer is\s*\$?(-?[\d,]*\.?\d+)", text, re.IGNORECASE)
    if not m:
        nums = ANS_RE.findall(text)
        if not nums:
            return None
        val = nums[-1]
    else:
        val = m.group(1)
    val = val.replace(",", "").rstrip(".")
    try:
        f = float(val)
        return int(f) if f.is_integer() else f
    except ValueError:
        return None


def gold_answer(ans: str):
    g = ans.split("####")[-1].strip().replace(",", "")
    try:
        f = float(g)
        return int(f) if f.is_integer() else f
    except ValueError:
        return None


def query(port, model, question, max_tokens):
    prompt = FEWSHOT + f"Q: {question}\nA:"
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "stop": ["\nQ:", "\n\nQ:"],
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=body, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    return out["choices"][0]["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--label", default="run")
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    ap.add_argument("--limit", type=int, default=1319)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ds = load_dataset("openai/gsm8k", "main", split="test")
    n = min(args.limit, len(ds))
    items = [(ds[i]["question"], gold_answer(ds[i]["answer"])) for i in range(n)]

    results = [None] * n
    lock = threading.Lock()
    done = [0]
    t0 = time.perf_counter()

    def work(i):
        q, gold = items[i]
        try:
            text = query(args.port, args.model, q, args.max_tokens)
            pred = extract_pred(text)
        except Exception as e:
            text, pred = f"<error: {e}>", None
        ok = (pred is not None and gold is not None and pred == gold)
        results[i] = {"i": i, "gold": gold, "pred": pred, "ok": ok}
        with lock:
            done[0] += 1
            if done[0] % 100 == 0:
                acc = sum(1 for r in results[:i+1] if r and r["ok"]) / done[0]
                print(f"  {done[0]}/{n}  running acc≈{acc:.3f}  "
                      f"({done[0]/(time.perf_counter()-t0):.1f} q/s)", flush=True)

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(ex.map(work, range(n)))

    correct = sum(1 for r in results if r["ok"])
    parsed = sum(1 for r in results if r["pred"] is not None)
    acc = correct / n
    elapsed = time.perf_counter() - t0
    summary = {
        "label": args.label, "n": n, "correct": correct, "accuracy": acc,
        "parsed_frac": parsed / n, "elapsed_s": elapsed,
        "max_tokens": args.max_tokens,
    }
    print(f"\n=== {args.label}: GSM8K acc = {acc:.4f} ({correct}/{n}), "
          f"parsed {parsed}/{n}, {elapsed:.0f}s ===")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
