#!/usr/bin/env bash
# Robustly stop any vLLM server + its worker subprocesses and wait for GPU mem
# to actually free (workers are named VLLM::Worker_TP* and are NOT matched by
# `pkill -f "vllm serve"`).
set -u
pkill -9 -f "vllm serve" 2>/dev/null
pkill -9 -f "VLLM::" 2>/dev/null
# Kill anything still holding the GPUs.
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  kill -9 "$pid" 2>/dev/null
done
for i in $(seq 1 60); do
  total=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
  if [ "${total:-9999}" -lt 2000 ]; then
    echo "GPUs freed (total used=${total}MiB)"; exit 0
  fi
  sleep 2
done
echo "WARN: GPU memory not fully freed: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | paste -sd' ')"
exit 0
