# Results — Gemma-4-31B-it on 4× B200 (vLLM 0.23.0, CUDA 13, FlashInfer 0.6.12)

All numbers: TP=2, `max_model_len=8192`, `gpu_mem_util=0.90`, greedy. Single-stream
unless noted. "long" = 6716-token prompt with a per-request nonce so the prefix
cache cannot serve prefill. MTP draft = `google/gemma-4-31B-it-assistant`,
`num_speculative_tokens=4`.

## Headline

| Config | short decode | long decode | long prefill TTFT |
|---|---|---|---|
| **Baseline** stock vLLM (Triton, forced) | 118.4 tok/s | 96.8 tok/s | 367 ms |
| Plugin **text-only** (256→FlashInfer), no MTP | 114.0 tok/s | 112.8 tok/s | **216 ms** |
| Baseline **+ MTP** (all Triton) | 363 tok/s | 218 tok/s | 389 ms |
| **Plugin text-only + MTP** (256→FlashInfer / 512→Triton) | **454 tok/s** | 196 tok/s | **310 ms** |

The last row is the intended end state: PR #38891's ~80/20 split (50 sliding-window
layers on FlashInfer, 10 full-attention layers on Triton) running **with** MTP. It is
**1.25× faster than stock+MTP** on short/medium context and on prefill (and **3.85×**
over the original stock baseline). At long context, single-stream decode is dominated
by the 10 Triton head=512 layers + the MTP acceptance rate, so the attention win there
shows up mainly in prefill; throughput at concurrency-8 long is 467 vs 418 tok/s (1.12×).

### Speedups that are real
- **MTP is the big win:** 363/118 = **3.07×** (short), 218/96.8 = **2.25×** (long decode).
  Draft acceptance is excellent: mean acceptance length 3.8–5.0 / 5, 70–100% per-token.
- **Attention backend (FlashInfer vs Triton), text-only, no MTP:** prefill TTFT
  367→216 ms = **1.70×**; long decode 96.8→112.8 = **1.17×**; 8-way throughput
  350→496 tok/s = **1.42×**. The win **grows with context length** — at short context
  decode is weight-bandwidth bound, so the attention backend is ~neutral (even
  marginally slower).

## Accuracy parity (GSM8K, 1319 questions, 8-shot CoT, greedy)

| Config | GSM8K | eval wall |
|---|---|---|
| Baseline — stock Triton, no MTP | 81.43% (1074/1319) | 62 s |
| Plugin — 256→FlashInfer / 512→Triton + MTP | 81.65% (1077/1319) | 31 s |

+0.22% (3 questions) = noise. The attention-backend change and MTP do **not** degrade
accuracy, and the eval ran 2× faster. (`bench/gsm8k_eval.py`, `results/gsm8k_*.json`.)

## Two non-obvious findings (why the PR alone does nothing here)

1. **`mm_prefix` masks the PR.** `Gemma4ForConditionalGeneration` is multimodal with
   `use_bidirectional_attention="vision"` → vLLM sets `use_mm_prefix=True` on every
   layer → FlashAttention/FlashInfer are rejected (`supports_mm_prefix()==False`),
   so the per-layer selector lands on **Triton anyway** — identical to the old forced
   behavior. PR #38891's head-size logic never gets a chance. The plugin's opt-in
   `VLLM_GEMMA4_TEXT_ONLY_ATTN=1` clears the vision-bidirectional flag (a no-op for
   text inputs) and unlocks FlashInfer. **This is what actually delivers the speedup.**

2. **The head_dim=512 layers must land on Triton — and on B200 they don't by default.**
   PR #38891's design is ~80% of layers on a fast backend and the ~20% full-attention
   (head=512) layers *falling back* to Triton. On Blackwell that fallback silently
   breaks: FlashInfer reports `supports_head_size(512)==True`, so the per-layer selector
   routes the 512 layers to FlashInfer instead of Triton — and FlashInfer's only 512
   kernel (trtllm-gen) lacks the speculative-decode mask variant, so it **hard-crashes
   under MTP**. (FlashAttention can't help either — FA4 is disabled on Blackwell by TMEM
   limits → FA2, which caps at head≤256; and forcing FlexAttention hits a dynamo
   recompile-limit crash on MTP's variable shapes.) The plugin restores the intended
   behaviour by pinning head_size>256 → TRITON_ATTN per layer (`VLLM_GEMMA4_TEXT_ONLY_ATTN=1`
   installs the pin). No custom kernel is needed for this to work; a head=512 Blackwell
   kernel (see CUSTOM_KERNELS.md) would only be a further perf upside on the 20%.

## Recommended serve commands

**Recommended — text + MTP, ~80/20 fast attention (plugin installed):**
```bash
VLLM_GEMMA4_TEXT_ONLY_ATTN=1 vllm serve google/gemma-4-31B-it \
  --tensor-parallel-size 2 --max-model-len 8192 \
  --speculative-config '{"model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":4}'
# plugin auto-loads via its entry point: 256-dim layers -> FlashInfer, 512-dim -> Triton.
# 454 tok/s short-context single stream (3.85x over stock, 1.25x over stock+MTP).
```

Without MTP, the same command minus `--speculative-config` gives the 1.4–1.7×
long-context attention win.

Reproduce: `bench/bench_decode.py` (short) and `bench/bench_longctx.py` (long,
cache-defeating). Raw JSON in `results/`.
