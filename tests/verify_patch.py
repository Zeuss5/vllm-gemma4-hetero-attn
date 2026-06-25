"""In-process verification of the gemma4-hetero-attn plugin.

Run: python tests/verify_patch.py
"""
import types
import torch

from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.platforms import current_platform
from vllm.v1.attention.selector import AttentionSelectorConfig


def make_stub_vllm_config():
    """Minimal duck-typed stand-in for the bits Gemma4Config touches."""
    hf_text = types.SimpleNamespace(
        head_dim=256,
        global_head_dim=512,
        layer_types=(["sliding_attention"] * 50) + (["full_attention"] * 10),
    )
    attn = types.SimpleNamespace(backend=None)
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(hf_text_config=hf_text),
        attention_config=attn,
    )


def predict_backend(head_size: int) -> str:
    cfg = AttentionSelectorConfig(
        head_size=head_size,
        dtype=torch.bfloat16,
        kv_cache_dtype="auto",
        block_size=16,
    )
    try:
        path = current_platform.get_attn_backend_cls(
            selected_backend=None, attn_selector_config=cfg, num_heads=32
        )
        return path.rsplit(".", 1)[-1]
    except Exception as e:
        return f"<error: {type(e).__name__}: {e}>"


print("=" * 70)
print("1) Behaviour of STOCK Gemma4Config (force-pins TRITON_ATTN)")
print("=" * 70)
from vllm.model_executor.models.config import Gemma4Config

stub = make_stub_vllm_config()
Gemma4Config.verify_and_update_config(stub)
print("  attention_config.backend after stock verify :", stub.attention_config.backend)
stock_forced = stub.attention_config.backend

print()
print("=" * 70)
print("2) Load plugins (simulates EngineArgs.__post_init__ path)")
print("=" * 70)
from vllm.plugins import load_general_plugins

load_general_plugins()
patched = getattr(Gemma4Config, "_gemma4_hetero_attn_patched", False)
print("  Gemma4Config._gemma4_hetero_attn_patched :", patched)

print()
print("=" * 70)
print("3) Behaviour of PATCHED Gemma4Config (PR #38891 — no global override)")
print("=" * 70)
stub2 = make_stub_vllm_config()
Gemma4Config.verify_and_update_config(stub2)
print("  attention_config.backend after patched verify:", stub2.attention_config.backend)
patched_backend = stub2.attention_config.backend

print()
print("=" * 70)
print("4) Per-layer backend the selector will actually pick on this GPU")
print("=" * 70)
for hs, kind in [(256, "sliding-window"), (512, "full-attention")]:
    print(f"  head_size={hs:<4} ({kind:<14}) -> {predict_backend(hs)}")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
ok = True
if stock_forced != AttentionBackendEnum.TRITON_ATTN:
    print("  [WARN] stock did not force TRITON_ATTN as expected:", stock_forced)
    ok = False
else:
    print("  [OK] stock vLLM force-pins TRITON_ATTN on all layers")
if not patched:
    print("  [FAIL] plugin did not patch Gemma4Config"); ok = False
else:
    print("  [OK] plugin patched Gemma4Config via general_plugins entry point")
if patched_backend is not None:
    print("  [FAIL] plugin still set a global backend:", patched_backend); ok = False
else:
    print("  [OK] plugin leaves backend=None -> per-layer selection")
print()
print("RESULT:", "PASS" if ok else "FAIL")
