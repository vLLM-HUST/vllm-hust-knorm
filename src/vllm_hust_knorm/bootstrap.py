# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-HUST project
"""Discovery hook: import-only registration in hosts, no-op elsewhere.

Candidate wrappers require a separate explicit development flag and source guards.
"""

from __future__ import annotations

from .core.hosts import detect_host

__all__ = ["detect_host", "register_plugins"]


def register_plugins() -> list[str]:
    """Register Knorm in vLLM hosts; no-op elsewhere."""
    if detect_host() is None:
        return []

    import os

    if os.environ.get("VLLM_KNORM_CANDIDATE") == "1":
        from .adapters.ascend.loading import install

        install()

    from .core.activation import activate

    info = activate()
    print(
        "[vllm-hust-knorm] knorm-import-only-registered; "
        "VLLM_KNORM_ENABLED=1 is unavailable pending host validation "
        f"(status={info.get('status')})",
        flush=True,
    )
    return ["knorm"]
