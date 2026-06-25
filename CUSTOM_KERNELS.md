# Custom kernels — optional perf upside on the 20% (B200)

**Not required for correctness.** The plugin already works end-to-end: ~80% of layers
(head_dim=256) run on FlashInfer and the ~20% full-attention layers (head_dim=512) are
pinned to Triton — exactly PR #38891's split — and this runs with MTP at 454 tok/s
(`RESULTS.md`). A custom kernel would only make that pinned 20% faster than Triton.

## The remaining slow spot, precisely

Gemma 4 has two attention shapes:
- **sliding-window layers**: `head_dim=256`, 50 of 60 layers
- **full-attention layers**: `head_dim=512`, 10 of 60 layers

On B200 + MTP, the 256-dim layers are fine (FlashInfer / FA2 both work), but the
**512-dim layers have no fast kernel that also supports speculative-decode masks**:

| backend | 512 + spec-decode on Blackwell |
|---|---|
| FlashAttention | FA4 disabled (TMEM limits) → FA2, which caps at head≤256 → unusable |
| FlashInfer | only `trtllm-gen` does 512; it lacks the `maskType=1,multiCtasKvMode=1` variant → crash |
| FlexAttention | `torch.compile` recompile-limit blow-up on variable spec shapes |
| Triton | works, but it's the slow one we're trying to beat |

So the 10 full-attention layers run on Triton. That's correct and fast enough that the
combined config already beats stock+MTP; **a SM100 head_dim=512 FMHA kernel with a
causal+tree(spec) mask is the only thing that would speed up those 10 layers further.**

## Kernel #1 (highest value): SM100 head_dim=512 FMHA with spec-decode mask

- **Scope:** decode/append attention, `head_dim_qk = head_dim_v = 512`, causal +
  speculative tree/staircase mask (`q_len` ∈ {1..num_spec+1}), paged KV (block 16),
  BF16 (and an fp8-KV variant later).
- **Why feasible:** only 10/60 layers and only the small spec `q_len` — a decode-shaped
  kernel (small Q tile, long KV) is the easy regime. The hard prefill case can stay on
  Triton.
- **How:** CUTLASS/CuTe-DSL FMHA for Blackwell, or a Triton kernel tiled for 512 with KV
  split across CTAs (the "multiCtasKvMode" FlashInfer couldn't provide). Register as a
  vLLM attention backend and have the plugin route **only** the 512-dim layers to it
  (the per-layer selector already keys on `head_size`, so this composes cleanly with
  PR #38891 — 256 → FlashInfer, 512 → our kernel).
- **Expected:** lifts the "fast attention + MTP" config from "crashes" to working, and
  should beat Triton on the 512 layers' prefill + long-context decode (where we already
  measured 1.4–1.7× for the 256 path).

## Kernel #2: fused split-KV decode for the 512 layers (perf, not correctness)

Even once #1 works, head_dim=512 decode is memory-bound on KV reads. A split-KV /
flash-decoding kernel (partition KV across SMs, online-softmax reduce) tuned for
`d=512` on B200's HBM3e keeps all SMs busy at low batch — the regime MTP runs in.

## Kernel #3: fuse the Gemma4 per-layer norms + RoPE into the attention prologue

Gemma 4 uses QK-norm and dual RoPE (`rope_theta` differs for sliding vs full layers,
`partial_rotary_factor=0.25` on full). Fusing RMSNorm(Q,K)+RoPE+paged-KV-write into the
attention entry kills a few launches/round-trips per layer per step — meaningful at
MTP's tiny per-step token counts.

## Suggested order
1. **Kernel #1** — it's the difference between "fast attention can't be used with MTP"
   and "it can." Everything else is incremental.
2. Wire the plugin to route `head_size==512` → custom backend, keep 256 on FlashInfer.
3. Kernel #2/#3 as profiling-guided follow-ups.

Scaffold for an out-of-tree backend lives next to the plugin (a backend class +
`get_attn_backend` hook) so it ships in the same pip package — still drag-and-drop.
