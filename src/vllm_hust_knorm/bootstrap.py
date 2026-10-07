# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-HUST project
"""``vllm.general_plugins`` dynamic-load entry point.

After installation, vLLM imports this entry point in every process
(process 0, engine core and workers) during plugin loading, and calls
``register_plugins()``. Registration installs the runtime patches but
does not enable compression: only ``VLLM_KNORM_ENABLED=1`` (plus
prefix caching disabled) activates Knorm.

Failure mode: when this process is not a vllm host the hook is a no-op;
when the host conflicts with an in-tree ``vllm.knorm`` or lacks the
required surfaces, activation raises and vLLM logs the exception —
fail closed, never a silent half-install.
"""

from __future__ import annotations

from .core.hosts import detect_host

__all__ = ["detect_host", "register_plugins"]


def register_plugins() -> list[str]:
    """Register Knorm in vLLM hosts; no-op elsewhere."""
    if detect_host() is None:
        return []

    from .core.activation import activate

    info = activate()
    print(
        "[vllm-hust-knorm] runtime patches registered; activate with "
        "VLLM_KNORM_ENABLED=1 and --no-enable-prefix-caching "
        f"(status={info.get('status')})",
        flush=True,
    )
    return ["knorm"]
