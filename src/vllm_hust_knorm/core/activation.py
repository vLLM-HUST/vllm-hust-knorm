# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-HUST project
"""Single import-only registration gate shared by bootstrap and facade.

Runtime activation is rejected until a verified host binding exists. Archived
legacy patches are never installed by this pipeline.
"""

from __future__ import annotations

from typing import Any

from .hosts import detect_host, require_host_surfaces, require_no_in_tree_knorm

_activated = False


def activate() -> dict[str, Any]:
    """Install the host integration once and return a status dict."""
    global _activated
    from ..config import validate_environment

    validate_environment()
    if _activated:
        return {"status": "already_activated", "extension_id": _extension_id()}

    require_no_in_tree_knorm()
    require_host_surfaces()

    from ..adapters.vllm_hust import VllmHustAdapter

    detected = detect_host()
    adapter = VllmHustAdapter()
    info = dict(adapter.register())
    info.setdefault("host", detected)
    info["status"] = "registered"
    _activated = True
    return info


def is_activated() -> bool:
    """Whether :func:`activate` succeeded in this process."""
    return _activated


def reset_for_tests() -> None:
    """Forget the activation state (tests only; does NOT uninstall
    patches — tests reinstall over fresh fakes instead)."""
    global _activated
    _activated = False


def _extension_id() -> str:
    from ..adapters.vllm_hust import EXTENSION_ID

    return EXTENSION_ID


__all__ = ["activate", "is_activated", "reset_for_tests"]
