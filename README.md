# vllm-gemma4-hetero-attn

Drag-and-drop [vLLM](https://github.com/vllm-project/vllm) plugin that backports
**PR #38891 — "[Gemma4] Allow per-layer attention backend selection for
heterogeneous head dimensions"** without forking or editing vLLM.

## The problem

Gemma 4 uses **heterogeneous head dimensions**: sliding-window layers use
`head_dim=256` and full-attention layers use `global_head_dim=512`. Stock vLLM
(≤ 0.23.0) detects this and **force-pins `TRITON_ATTN` for all layers** to avoid
mixing backends:

```python
# vllm/model_executor/models/config.py — Gemma4Config.verify_and_update_config
if head_dim != global_head_dim and max_head_dim > 256 and backend is None:
    vllm_config.attention_config.backend = AttentionBackendEnum.TRITON_ATTN  # all layers -> Triton
```

For Gemma 4 31B (60 layers) that drops **every** layer onto Triton even though
the ~80% sliding-window layers can run FlashAttention/FlashInfer. The result is
the well-known "Gemma 4 is extremely slow on vLLM" symptom (vLLM issue #38887).

## What the plugin does

It replaces `Gemma4Config.verify_and_update_config` with the PR's behaviour:
**stop forcing a global backend** and let vLLM's per-layer, `@cache`-decorated
`get_attn_backend()` selector choose the best backend for each `head_size`:

| layer type      | head_size | selected backend (B200)                  |
|-----------------|-----------|------------------------------------------|
| sliding window  | 256 (~80% of layers) | FlashAttention / FlashInfer   |
| full attention  | 512 (~20% of layers) | Triton (see note below)       |

On Blackwell, FlashAttention runs as FA2 (FA4 is disabled by TMEM limits) so it
caps at `head_size ≤ 256`, and FlashInfer's only head=512 kernel lacks the
speculative-decode mask variant. The plugin therefore **pins the head=512
full-attention layers to Triton** (PR #38891's intended ~80/20 split), which is
also what lets the fast path coexist with MTP / speculative decoding without
crashing. It logs the split and warns on an incompatible explicit
`--attention-backend`. See `RESULTS.md` and `CUSTOM_KERNELS.md` for details.

## How it hooks in (no source edits)

The package registers a `vllm.general_plugins` entry point. vLLM calls every
such plugin from `load_general_plugins()`, which runs inside
`EngineArgs.__post_init__` (main process) **before** `VllmConfig.__post_init__`
→ `try_verify_and_update_config()`, and again inside the engine-core / worker
processes. So the monkeypatch is always in place before the Gemma 4 config is
verified.

## Important: the PR alone is often not enough (the `mm_prefix` gotcha)

Gemma 4 *multimodal* checkpoints (`Gemma4ForConditionalGeneration`) set
`use_bidirectional_attention="vision"`, which makes vLLM mark **every** attention
layer as `use_mm_prefix`. FlashAttention and FlashInfer don't support that, so the
per-layer selector falls back to Triton/Flex on all layers — and PR #38891's
head-size logic changes nothing. To actually move the text layers onto a fast
backend you must also clear that flag, which is **lossless for text-only serving**
(image-token spans never occur). Enable the plugin's opt-in lever:

```bash
VLLM_GEMMA4_TEXT_ONLY_ATTN=1 vllm serve google/gemma-4-31B-it ...
```

This unlocks FlashInfer for the sliding-window (head_dim=256) and full-attention
(head_dim=512) text layers. Do **not** use it if you serve image inputs.

See `RESULTS.md` for measured numbers (≈1.7× prefill, ≈1.4× throughput at long
context on B200) and `CUSTOM_KERNELS.md` for the head_dim=512 + MTP kernel gap.

## Install

```bash
# from GitHub
pip install "git+https://github.com/Zeuss5/vllm-gemma4-hetero-attn"

# or from a local checkout (editable)
git clone https://github.com/Zeuss5/vllm-gemma4-hetero-attn
pip install -e vllm-gemma4-hetero-attn
```

Requires an existing vLLM >= 0.23 install (vLLM is deliberately not a dependency
so this never reinstalls/forks your vLLM).

The plugin auto-loads via its entry point — look for `[gemma4-hetero-attn]`
lines in the log to confirm it's active. For the actual text-serving speedup,
add `VLLM_GEMMA4_TEXT_ONLY_ATTN=1` (see the `mm_prefix` gotcha above):

```bash
VLLM_GEMMA4_TEXT_ONLY_ATTN=1 vllm serve google/gemma-4-31B-it \
  --tensor-parallel-size 2 --max-model-len 8192 \
  --speculative-config '{"model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":4}'
```

## Environment knobs

| Env var | Default | Effect |
|---|---|---|
| `VLLM_GEMMA4_HETERO_ATTN` | `1` | `0`/`off` → don't patch (pure stock behavior) |
| `VLLM_GEMMA4_TEXT_ONLY_ATTN` | `0` | `1`/`on` → clear `use_bidirectional_attention` so FA/FlashInfer serve the text layers (text-only; see gotcha above) |

## Disable

```bash
VLLM_GEMMA4_HETERO_ATTN=0 vllm serve ...   # disable just this plugin
VLLM_PLUGINS="" vllm serve ...             # disable ALL general plugins
pip uninstall vllm-gemma4-hetero-attn
```

## Compatibility

Built and tested against vLLM 0.23.0 (CUDA 13.0, torch 2.11) on NVIDIA B200.
The patch is a no-op on non-heterogeneous configs and degrades gracefully if a
future vLLM removes/renames `Gemma4Config`.
