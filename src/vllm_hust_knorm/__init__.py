# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-HUST project
"""Token-level KNorm configuration and import-only plugin registration.

Importing the package does not load torch, vLLM, or the legacy implementation.
Configure the CPU runtime explicitly::

    from vllm_hust_knorm import KNormConfig

    config = KNormConfig(mode="once", prune_after=4096, keep_ratio="0.5")

These values are illustrative, not evaluated model defaults. Host activation
remains unavailable until a concrete binding is verified. Historical source
and provenance are retained separately; legacy configuration is not mapped
onto token-level semantics.
"""

from ._version import __version__
from .config import KNormConfig


def activate() -> dict:
    """Register import-only discovery in a host; runtime enablement is rejected."""
    from .core.activation import activate as _activate

    return _activate()


def KnormConfig(*args, **kwargs):
    """Reject the former block-level configuration with migration guidance."""
    raise TypeError(
        "KnormConfig was removed: use KNormConfig(mode=..., prune_after=..., "
        "keep_ratio=...) with explicit token-level settings. Legacy block ratios "
        "and protection windows cannot be mapped automatically."
    )


def should_activate(*args, **kwargs):
    """Reject the legacy enablement predicate; host registration owns the gate."""
    raise RuntimeError(
        "should_activate was removed: activate() owns host registration; "
        "runtime enablement remains import_only pending host validation."
    )


__all__ = ["KNormConfig", "__version__", "activate"]
