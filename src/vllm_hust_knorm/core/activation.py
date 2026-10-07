# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-HUST project
"""Unified activation pipeline: guards → host adapter → register().

This is the single path that lights the plugin up inside a host
process, shared by two entry points:

- the ``vllm.general_plugins`` bootstrap hook (host-invoked), and
- the programmatic facade ``vllm_hust_knorm.activate()``.

Fail-closed semantics: an in-tree ``vllm.knorm`` conflict, a missing
host stack, or missing host surfaces raise immediately; nothing is
silently skipped. Registering patches does not enable compression —
``VLLM_KNORM_ENABLED`` still gates behavior, exactly like the reference
plugin's “register ≠ enable” split.
"""

from __future__ import annotations

from typing import Any

from .hosts import detect_host, require_host_surfaces, require_no_in_tree_knorm

_activated = False


def activate() -> dict[str, Any]:
    """Install the host integration once and return a status dict."""
    global _activated
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
