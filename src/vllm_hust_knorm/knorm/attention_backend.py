# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-HUST project
"""Attention wrapper that computes key L2 norms for Knorm.

Migrated from ``vllm/knorm/attention_backend.py`` in the archived
``vllm-hust`` tree (legacy PR #76). Norms from **all** attention layers
are collected during the ordinary attention forward and averaged after
the model forward completes, so scoring adds no extra compute pass.

torch and the host attention registry are imported lazily inside the
functions that need them: importing this module (and the whole plugin
package) stays side-effect free and CPU-safe, per the extension guide.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from .config import KnormConfig
from .norms import VALID_REDUCE_OPS

# ---------------------------------------------------------------------------
# Global buffer for pending per-token norms (still on the device).
# ---------------------------------------------------------------------------
_pending_norms: list[Any] = []
_original_forward: Callable[..., Any] | None = None


def get_pending_norms() -> list[Any]:
    """Return pending norm tensors (still on the device)."""
    return list(_pending_norms)


def clear_pending_norms() -> None:
    """Clear the pending norm buffer."""
    _pending_norms.clear()


def store_layer_norms(norms: Any) -> None:
    """Store per-token key norms from one attention layer.

    Args:
        norms: Float tensor of shape ``[num_tokens]``, on the device.
    """
    _pending_norms.append(norms.detach())


# ---------------------------------------------------------------------------
# Norm computation — device-agnostic over the leading token dimension.
# ---------------------------------------------------------------------------


def compute_key_norms(key: Any, reduce_op: str = "mean") -> Any:
    """Compute per-token importance score from a key tensor.

    Args:
        key: shape ``[num_tokens, num_kv_heads, head_size]``.
        reduce_op: 'mean', 'max', or 'sum' across heads (same rule as
            :func:`vllm_hust_knorm.knorm.norms.reduce_head_norms`).

    Returns:
        shape ``[num_tokens]`` float32. Lower norm = more important.
    """
    if reduce_op not in VALID_REDUCE_OPS:
        raise ValueError(
            f"reduce_op must be one of {VALID_REDUCE_OPS}, got {reduce_op!r}"
        )
    head_norms = key.float().norm(dim=-1)  # [num_tokens, num_kv_heads]
    if reduce_op == "max":
        return head_norms.max(dim=1).values
    if reduce_op == "sum":
        return head_norms.sum(dim=1)
    return head_norms.mean(dim=1)  # mean (default)


# ---------------------------------------------------------------------------
# Wrapper installation
# ---------------------------------------------------------------------------


def _resolve_impl_cls() -> type:
    """Resolve the live attention implementation class.

    Auto-detects the platform: Ascend NPU fills the ``CUSTOM`` slot via
    vllm-ascend; CUDA GPUs fall back to the built-in ``FLASH_ATTN``
    backend. Failure to resolve either is a hard error — scoring must
    not silently degrade to no-op.
    """
    try:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
    except ImportError as exc:  # pragma: no cover - host drift guard
        raise RuntimeError(
            "vllm-hust-knorm needs the vllm v1 attention registry "
            "(vllm.v1.attention.backends.registry), which is not "
            "importable in this process"
        ) from exc

    try:
        backend_cls = AttentionBackendEnum.CUSTOM.get_class()
        return backend_cls.get_impl_cls()
    except ValueError:
        backend_cls = AttentionBackendEnum.FLASH_ATTN.get_class()
        return backend_cls.get_impl_cls()


def install_attention_wrapper() -> str:
    """Wrap attention ``forward`` to collect key norms (idempotent).

    Returns a human-readable description of the wrapped implementation
    class. The wrapper stores per-layer norms only while the captured
    :class:`KnormConfig` snapshot is active.
    """
    global _original_forward

    if _original_forward is not None:
        return f"{_original_forward.__module__}.{_original_forward.__qualname__}"

    impl_cls = _resolve_impl_cls()
    _original_forward = impl_cls.forward  # type: ignore[attr-defined]
    config = KnormConfig()

    @functools.wraps(_original_forward)
    def _patched_forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
    ):
        if key is not None and config.is_active:
            key_norms = compute_key_norms(key, config.norm_reduce_op)
            store_layer_norms(key_norms)
        return _original_forward(
            self,
            layer=layer,
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )

    impl_cls.forward = _patched_forward  # type: ignore[attr-defined]
    return f"{impl_cls.__module__}.{impl_cls.__qualname__}"


# Legacy in-tree name kept as an alias for continuity with the archived
# host seams and provenance patches.
install_ascend_wrapper = install_attention_wrapper


def reset_for_tests() -> None:
    """Uninstall the wrapper and clear buffers (tests only)."""
    global _original_forward
    _pending_norms.clear()
    _original_forward = None


__all__ = [
    "clear_pending_norms",
    "compute_key_norms",
    "get_pending_norms",
    "install_ascend_wrapper",
    "install_attention_wrapper",
    "reset_for_tests",
    "store_layer_norms",
]
