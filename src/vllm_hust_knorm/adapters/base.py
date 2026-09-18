# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-HUST project
"""Host adapter interface (same shape as the reference plugin)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..core.hosts import detect_host


class HostAdapter(ABC):
    """Binds the plugin to one concrete host stack.

    Subclasses declare ``host`` (the host identifier used by
    :mod:`vllm_hust_knorm.core.hosts`) and implement ``register`` to
    install their host patches. Registration must be idempotent and
    fail closed when the host stack is absent or unsupported.
    """

    host: str = ""

    def require_host(self) -> str:
        """Return the detected host id or raise with context."""
        detected = detect_host()
        if detected != self.host:
            raise RuntimeError(
                f"{type(self).__name__} binds to host {self.host!r}, but "
                f"this process provides {detected!r}"
            )
        return detected

    @abstractmethod
    def register(self) -> dict[str, Any]:
        """Install the host integration and return a status dict."""


__all__ = ["HostAdapter"]
