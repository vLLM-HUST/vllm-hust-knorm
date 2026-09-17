# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-HUST project
"""Host detection and activation guards (import-safe on any machine).

This module is the authority on "which host process may activate what":
the ``vllm.general_plugins`` bootstrap hook and the programmatic
``vllm_hust_knorm.core.activation.activate`` pipeline share the guards
defined here. Nothing in this module imports vllm or torch eagerly.
"""

from __future__ import annotations

import importlib.util

VLLM_HOST = "vllm"

# Host surfaces the runtime patches bind to. Kept in one place so the
# contract tests and HOST_CONTRACT.md stay in sync with the adapter.
REQUIRED_SURFACES = (
    "vllm.v1.core.single_type_kv_cache_manager:register_all_kvcache_specs",
    "vllm.v1.core.single_type_kv_cache_manager:FullAttentionManager",
    "vllm.v1.kv_cache_spec_registry:KVCacheSpecRegistry",
    "vllm.v1.core.kv_cache_utils:FreeQueue",
    "vllm.v1.core.block_pool:BlockPool",
    "vllm.v1.core.sched.scheduler:Scheduler",
    "vllm.v1.worker.gpu_model_runner:GPUModelRunner",
)


def module_importable(name: str) -> bool:
    """Whether *name* is importable (metadata probe only, no import)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


def detect_host() -> str | None:
    """Detect the host stack in this interpreter, or ``None``."""
    if module_importable("vllm"):
        return VLLM_HOST
    return None


def require_no_in_tree_knorm() -> None:
    """Fail closed when the host already ships an in-tree ``vllm.knorm``.

    vLLM-HUST builds of the 0.23 seam era embed the legacy in-tree Knorm
    module behind try-imports. Installing this plugin next to them would
    double-provide the feature; the guide forbids silently ignoring the
    conflict, so activation refuses with instructions.
    """
    if module_importable("vllm.knorm"):
        raise RuntimeError(
            "vllm-hust-knorm refuses to activate: this vLLM install "
            "already ships an in-tree 'vllm.knorm' module. Remove the "
            "in-tree module (use a knorm-free vLLM-HUST build) or "
            "uninstall one of the two implementations; running both is "
            "unsupported."
        )


def require_host_surfaces() -> None:
    """Fail closed when any host surface required by the patches is
    missing, naming every missing symbol."""
    missing: list[str] = []
    for surface in REQUIRED_SURFACES:
        module_name, _, attr = surface.partition(":")
        try:
            spec = importlib.util.find_spec(module_name)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            missing.append(surface)
    if missing:
        raise RuntimeError(
            "vllm-hust-knorm requires host surfaces that are not "
            "importable in this process: "
            + ", ".join(missing)
            + ". See HOST_CONTRACT.md for the verified host baselines; "
            "compatibility must not be assumed for other builds."
        )


__all__ = [
    "REQUIRED_SURFACES",
    "VLLM_HOST",
    "detect_host",
    "module_importable",
    "require_host_surfaces",
    "require_no_in_tree_knorm",
]
