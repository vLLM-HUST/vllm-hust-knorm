# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-HUST project
"""KNorm KV-cache compression for vLLM-HUST, as an installable plugin.

Importing this package is lazy and side-effect free: no torch, no vllm,
no device access, no global registry writes. The device and host paths
load only when the corresponding functions run.

Quick start::

    from vllm_hust_knorm import knorm

    knorm.KnormConfig()            # validated snapshot of VLLM_KNORM_*
    knorm.should_activate(False)   # single source of truth
    vllm_hust_knorm.activate()     # install host patches (host process)

Layering (see docs/architecture.md)::

    knorm.config / norms / eviction  pure, any process, CPU-testable
    knorm.attention_backend / hooks  torch paths, lazy imports
    knorm.manager                    host subclass, built lazily
    core                             host detection + activation pipeline
    adapters                         host patches (fail closed)
    bootstrap                        vllm.general_plugins hook

Provenance: migrated from the archived vllm-hust tree, legacy PRs #76,
#134 and #214 — see PROVENANCE.md and provenance/legacy-patches/.
"""

from ._version import __version__
from .knorm import KnormConfig, should_activate


def activate() -> dict:
    """Install the host integration (same pipeline as the bootstrap
    hook); raises RuntimeError outside a compatible vLLM-HUST host."""
    from .core.activation import activate as _activate

    return _activate()


__all__ = ["KnormConfig", "__version__", "activate", "should_activate"]
