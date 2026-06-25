#!/usr/bin/env bash
# Convenience launcher used by the benchmarks. Portable: assumes `vllm` is on
# PATH (activate your venv first, or set $VLLM_BIN).
#
# Usage: bench/serve.sh <logfile> [extra vllm args...]
# Env knobs:
#   MODEL  (default google/gemma-4-31B-it)
#   DRAFT  (default google/gemma-4-31B-it-assistant)
#   MTP=1                       enable speculative decoding with $DRAFT
#   VLLM_GEMMA4_TEXT_ONLY_ATTN=1  enable the plugin's text-only fast path
#   GPUS   (default 0,1)        -> CUDA_VISIBLE_DEVICES
#   TP     (default 2)          tensor-parallel size
#   MAXLEN (default 8192)       max model len
set -e
LOG="${1:?usage: serve.sh <logfile> [extra vllm args...]}"; shift || true

MODEL="${MODEL:-google/gemma-4-31B-it}"
DRAFT="${DRAFT:-google/gemma-4-31B-it-assistant}"
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1}"
VLLM_BIN="${VLLM_BIN:-vllm}"

ARGS=(
  "$MODEL"
  --tensor-parallel-size "${TP:-2}"
  --max-model-len "${MAXLEN:-8192}"
  --gpu-memory-utilization 0.90
  --port "${PORT:-8001}"
)
if [ "${MTP:-0}" = "1" ]; then
  ARGS+=(--speculative-config "{\"model\": \"$DRAFT\", \"num_speculative_tokens\": ${NUM_SPEC:-4}}")
fi
ARGS+=("$@")

exec "$VLLM_BIN" serve "${ARGS[@]}" > "$LOG" 2>&1
