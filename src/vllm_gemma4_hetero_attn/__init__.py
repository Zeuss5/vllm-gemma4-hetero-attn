# SPDX-License-Identifier: Apache-2.0
"""vllm-gemma4-hetero-attn

A drag-and-drop vLLM plugin that implements vLLM PR #38891
("[Gemma4] Allow per-layer attention backend selection for heterogeneous head
dimensions") without forking vLLM.

Stock vLLM (<= 0.23.0) force-pins ``TRITON_ATTN`` for *every* Gemma 4 attention
layer whenever the model declares heterogeneous head dimensions
(``head_dim`` != ``global_head_dim`` and ``max > 256``).  Gemma 4 31B uses
``head_dim=256`` for its sliding-window layers and ``global_head_dim=512`` for
its full-attention layers, so the override drops all 60 layers onto Triton even
though ~83% of them (the sliding-window layers) can run FlashAttention, and on
Blackwell (FA4) even the 512-dim full-attention layers can.

This plugin replaces ``Gemma4Config.verify_and_update_config`` with the PR's
behaviour: it stops forcing a global backend and lets vLLM's per-layer
``get_attn_backend()`` selector pick the best backend for each ``head_size``.
"""

from vllm_gemma4_hetero_attn._patch import apply_patch, register

__all__ = ["register", "apply_patch"]
__version__ = "0.1.0"
