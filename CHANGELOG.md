# Changelog

All notable changes to this project are documented here.

## [0.1.0] - 2026-06-25

Initial release.

### Added
- `vllm.general_plugins` entry point that backports vLLM
  [PR #38891](https://github.com/vllm-project/vllm/pull/38891): replaces
  `Gemma4Config.verify_and_update_config` so Gemma 4's heterogeneous-head-dim
  layers use per-layer attention backend selection instead of a forced global
  `TRITON_ATTN`.
- `VLLM_GEMMA4_TEXT_ONLY_ATTN=1` opt-in text-only fast path: clears
  `use_bidirectional_attention="vision"` (lossless for text inputs) so the
  ~80% sliding-window (head_dim=256) layers can run on FlashAttention/FlashInfer.
- Per-layer pin that keeps the ~20% full-attention (head_dim=512) layers on
  `TRITON_ATTN`, which is what lets the fast path coexist with MTP / speculative
  decoding on Blackwell (FlashInfer's only head=512 kernel lacks the spec-decode
  mask variant).
- `VLLM_GEMMA4_HETERO_ATTN=0` kill switch.
- Benchmarks (`bench/`) and an in-process patch test (`tests/verify_patch.py`).

### Validated (Gemma-4-31B-it, 4× B200, vLLM 0.23.0, CUDA 13, FlashInfer 0.6.12)
- 454 tok/s short-context single-stream with MTP (3.85× over stock, 1.25× over
  stock+MTP); 1.4–1.7× long-context prefill/throughput without MTP.
- GSM8K parity: 81.65% (plugin + MTP) vs 81.43% (stock) — within noise.
- See `RESULTS.md`.
