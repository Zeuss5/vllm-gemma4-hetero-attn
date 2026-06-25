# SPDX-License-Identifier: Apache-2.0
"""The monkeypatch itself.

We replace ``vllm.model_executor.models.config.Gemma4Config.verify_and_update_config``
with a faithful re-implementation of vLLM PR #38891.  ``MODELS_CONFIG_MAP`` holds
the *class object* (not a bound copy of the method), and
``VllmConfig.try_verify_and_update_config`` looks the class up by architecture and
calls ``cls.verify_and_update_config(self)`` at call time, so swapping the
attribute on the class is sufficient for every Gemma 4 architecture
(``Gemma4ForCausalLM`` / ``Gemma4ForConditionalGeneration`` /
``Gemma4UnifiedForConditionalGeneration``).
"""

import os
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Use a name under the "vllm." namespace so the message inherits vLLM's
# configured log handler/level (a bare top-level logger would default to the
# root logger at WARNING and the INFO lines below would be invisible).
logger = init_logger("vllm.plugins.gemma4_hetero_attn")

_PATCH_FLAG = "_gemma4_hetero_attn_patched"
_SELECTOR_FLAG = "_gemma4_hetero_attn_selector_pinned"


def _install_full_attn_triton_pin() -> None:
    """Pin the full-attention (head_size > 256) layers to TRITON_ATTN.

    PR #38891 expects the full-attention layers to *fall back* to Triton while
    the ~80% sliding-window (head_dim=256) layers use a fast backend. On
    Blackwell that fallback doesn't happen on its own: FlashInfer reports
    ``supports_head_size(512)==True`` so the per-layer selector routes the 512
    layers to FlashInfer — whose only 512 kernel (trtllm-gen) lacks the
    speculative-decode mask variant and hard-crashes under MTP. We restore the
    intended behaviour by forcing TRITON_ATTN for head_size>256 layers when the
    selector would otherwise auto-pick. 256-dim layers are untouched.

    This runs in EVERY process (it is called from ``register()`` via
    ``load_general_plugins``), because per-layer backend selection happens in
    the worker processes, not where the config is verified. It is gated on
    TEXT-ONLY mode, which is Gemma 4-specific, so the head_size>256 rule is
    safely scoped to this server.
    """
    from vllm.platforms import current_platform

    platform_cls = type(current_platform)
    if getattr(platform_cls, _SELECTOR_FLAG, False):
        return
    if not hasattr(platform_cls, "get_attn_backend_cls"):
        return

    orig = platform_cls.get_attn_backend_cls

    def patched_get_attn_backend_cls(
        cls, selected_backend, attn_selector_config, num_heads=None
    ):
        if selected_backend is None:
            head_size = getattr(attn_selector_config, "head_size", None)
            if head_size is not None and head_size > 256:
                from vllm.v1.attention.backends.registry import AttentionBackendEnum

                logger.info_once(
                    "[gemma4-hetero-attn] Pinning head_size=%d (full-attention) "
                    "layers to TRITON_ATTN per PR #38891; sliding-window "
                    "(head_size<=256) layers keep their fast backend.",
                    head_size,
                )
                selected_backend = AttentionBackendEnum.TRITON_ATTN
        return orig.__func__(cls, selected_backend, attn_selector_config, num_heads)

    platform_cls.get_attn_backend_cls = classmethod(patched_get_attn_backend_cls)
    setattr(platform_cls, _SELECTOR_FLAG, True)
    logger.info(
        "[gemma4-hetero-attn] Installed full-attention->Triton per-layer pin on %s "
        "(head_size>256 -> TRITON_ATTN; head_size<=256 stays fast).",
        platform_cls.__name__,
    )


def _patched_verify_and_update_config(vllm_config: "VllmConfig") -> None:
    """Allow per-layer attention backend selection for Gemma 4 models with
    heterogeneous head dimensions (vLLM PR #38891).

    Gemma 4 uses different head dimensions for sliding-window (``head_dim``,
    typically 256) vs full-attention (``global_head_dim``, typically 512)
    layers.  Each ``Attention`` layer calls ``get_attn_backend()`` with its own
    ``head_size`` and the ``@cache``-decorated selector returns a distinct
    backend per unique configuration, so sliding-window layers (head_dim=256)
    automatically pick FlashAttention while full-attention layers
    (global_head_dim=512) pick the best backend that accepts that head size
    (FlashAttention 4 on Blackwell; otherwise a fallback such as Triton).

    Stock vLLM force-pinned ``TRITON_ATTN`` globally here, penalising the ~83%
    of layers that can run FlashAttention.  This plugin removes that override.
    """
    hf_text_config = vllm_config.model_config.hf_text_config

    # --- Text-only fast path (opt-in) -------------------------------------
    # Gemma 4 multimodal checkpoints set ``use_bidirectional_attention ==
    # "vision"``, which makes vLLM mark every attention layer as
    # ``use_mm_prefix`` (bidirectional over image-token spans). FlashAttention
    # and FlashInfer do not support that, so the per-layer selector is forced
    # onto TRITON/FLEX for ALL layers -> the PR #38891 backport has no effect
    # and Gemma 4 stays on the slow Triton path.
    #
    # For text-only serving there are never any image tokens, so the
    # bidirectional-vision attention is never exercised and clearing it is
    # numerically a no-op on text inputs — but it unlocks FlashAttention /
    # FlashInfer for the (causal) text layers, which is where the real
    # speedup comes from (see README benchmarks). This is opt-in because it
    # WOULD change behaviour for image inputs.
    if _text_only_enabled():
        bidir = getattr(hf_text_config, "use_bidirectional_attention", None)
        if bidir == "vision":
            hf_text_config.use_bidirectional_attention = None
            logger.warning(
                "[gemma4-hetero-attn] TEXT-ONLY mode: cleared "
                "use_bidirectional_attention='vision' so FlashAttention/"
                "FlashInfer can serve the (causal) text layers. This is a "
                "no-op for text inputs but DISABLES bidirectional attention "
                "over image tokens — do not use this mode for multimodal "
                "(image) requests. Unset VLLM_GEMMA4_TEXT_ONLY_ATTN to revert."
            )

    head_dim = getattr(hf_text_config, "head_dim", None)
    global_head_dim = getattr(hf_text_config, "global_head_dim", None)

    # Nothing heterogeneous to reason about -> behave like the no-op base class.
    if not (
        head_dim is not None
        and global_head_dim is not None
        and head_dim != global_head_dim
    ):
        return

    # Count sliding-window vs full-attention layers so the operator can see the
    # expected backend split in the logs.
    layer_types = getattr(hf_text_config, "layer_types", None) or []
    n_full = sum(1 for t in layer_types if t == "full_attention")
    n_sliding = len(layer_types) - n_full

    max_head = max(head_dim, global_head_dim)
    explicit_backend = vllm_config.attention_config.backend

    if explicit_backend is None and max_head > 256:
        # No user override and the larger head_dim exceeds FlashAttention's
        # classic kernel limit (head_size <= 256).  Per-layer selection routes
        # sliding-window layers to FlashAttention and full-attention layers to
        # whichever backend accepts head_size=max_head on this hardware.
        logger.info(
            "[gemma4-hetero-attn] Per-layer attention backend selection "
            "enabled (head_dim=%d, global_head_dim=%d): %d sliding-window "
            "layer(s) will use FlashAttention; %d full-attention layer(s) "
            "auto-select a head_size=%d-capable backend. No global "
            "TRITON_ATTN override is applied (vLLM PR #38891).",
            head_dim,
            global_head_dim,
            n_sliding,
            n_full,
            max_head,
        )
    else:
        logger.info(
            "[gemma4-hetero-attn] Gemma4 model has heterogeneous head "
            "dimensions (head_dim=%d, global_head_dim=%d).",
            head_dim,
            global_head_dim,
        )

    # If the user explicitly forced a single backend, warn when it cannot
    # handle the larger head dimension.  The per-layer selector would raise at
    # model-init time anyway; surfacing the conflict early is a clearer signal.
    if explicit_backend is not None:
        try:
            backend_cls = explicit_backend.get_class()
            supported = backend_cls.supports_head_size(max_head)
        except Exception:  # pragma: no cover - never block startup on a probe
            supported = True
        if not supported:
            logger.warning(
                "[gemma4-hetero-attn] Explicitly selected backend %s does not "
                "support head_size=%d (required by full-attention layers). "
                "Those layers will fail at runtime. Consider removing "
                "--attention-backend so each layer auto-selects the optimal "
                "backend.",
                explicit_backend.name,
                max_head,
            )


def apply_patch() -> bool:
    """Swap in the patched method. Idempotent; safe to call in every process.

    Returns ``True`` if the patch was (or had already been) applied, ``False``
    if Gemma4 support was not found in the installed vLLM.
    """
    try:
        from vllm.model_executor.models import config as _cfg
    except Exception as exc:  # pragma: no cover
        logger.warning("[gemma4-hetero-attn] Could not import vLLM config module: %s", exc)
        return False

    target = getattr(_cfg, "Gemma4Config", None)
    if target is None:
        logger.warning(
            "[gemma4-hetero-attn] vLLM build has no Gemma4Config; nothing to patch."
        )
        return False

    if getattr(target, _PATCH_FLAG, False):
        return True

    # Keep a handle on the original for debugging / unpatch.
    target._gemma4_orig_verify_and_update_config = target.verify_and_update_config
    target.verify_and_update_config = staticmethod(_patched_verify_and_update_config)
    setattr(target, _PATCH_FLAG, True)

    logger.info(
        "[gemma4-hetero-attn] Patched Gemma4Config.verify_and_update_config "
        "(vLLM PR #38891 backport) — Gemma 4 layers will no longer be force-"
        "pinned to TRITON_ATTN."
    )
    return True


def _text_only_enabled() -> bool:
    """Opt-in: clear Gemma 4's bidirectional-vision attention so FA/FlashInfer
    can serve the text layers. Enable with ``VLLM_GEMMA4_TEXT_ONLY_ATTN=1``."""
    val = os.environ.get("VLLM_GEMMA4_TEXT_ONLY_ATTN", "0").strip().lower()
    return val in ("1", "true", "on", "yes", "enable", "enabled")


def _disabled() -> bool:
    """Honour a kill-switch so the plugin can be A/B'd without uninstalling
    or disabling every other general plugin.  Set
    ``VLLM_GEMMA4_HETERO_ATTN=0`` (or ``false``/``off``/``no``) to skip."""
    val = os.environ.get("VLLM_GEMMA4_HETERO_ATTN", "1").strip().lower()
    return val in ("0", "false", "off", "no", "disable", "disabled")


def register() -> None:
    """Entry point invoked by vLLM's ``load_general_plugins()``."""
    if _disabled():
        logger.info(
            "[gemma4-hetero-attn] Disabled via VLLM_GEMMA4_HETERO_ATTN; "
            "leaving stock Gemma4Config behaviour in place."
        )
        return
    try:
        apply_patch()
    except Exception as exc:  # pragma: no cover - never break engine startup
        logger.warning("[gemma4-hetero-attn] Failed to apply patch: %s", exc)

    # In text-only mode the sliding-window layers move onto a fast backend, so
    # the full-attention (head_size>256) layers must be pinned to Triton (they
    # have no usable fast head=512 kernel under MTP on Blackwell). Installed
    # here so it also runs in the worker processes, where per-layer attention
    # backend selection actually happens.
    if _text_only_enabled():
        try:
            _install_full_attn_triton_pin()
        except Exception as exc:  # pragma: no cover
            logger.warning("[gemma4-hetero-attn] could not install Triton pin: %s", exc)
