# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-HUST project
"""Knorm configuration, read from ``VLLM_KNORM_*`` environment variables.

Migrated from the archived ``vllm-hust`` tree (legacy PRs #76/#134/#214,
see ``provenance/legacy-patches/``). The variable names, defaults and the
activation rule are kept compatible with the in-tree host seams, with one
deliberate divergence: as a separately installable plugin, Knorm is
**opt-in** — ``VLLM_KNORM_ENABLED`` defaults to ``0`` here, while the
in-tree module defaulted to ``1``. Enabling goes through the extension
manager (``vllm-hust-ext extension enable org.vllm-hust.knorm``) or by
exporting ``VLLM_KNORM_ENABLED=1`` explicitly.

Unlike the in-tree module (which read the cached ``vllm.envs`` registry),
this module reads ``os.environ`` directly so behavior is identical with or
without a host that registers the variables, and tests can drive it with
plain ``monkeypatch.setenv``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

ENV_ENABLED = "VLLM_KNORM_ENABLED"
ENV_COMPRESSION_RATIO = "VLLM_KNORM_COMPRESSION_RATIO"
ENV_WARMUP_TOKENS = "VLLM_KNORM_WARMUP_TOKENS"
ENV_SCORE_AGGREGATION = "VLLM_KNORM_SCORE_AGGREGATION"
ENV_NORM_REDUCE_OP = "VLLM_KNORM_NORM_REDUCE_OP"

# Plugin posture: disabled until explicitly enabled (see module docstring).
DEFAULT_ENABLED = False
DEFAULT_COMPRESSION_RATIO = 0.5
DEFAULT_WARMUP_TOKENS = 32
DEFAULT_SCORE_AGGREGATION = "min"
DEFAULT_NORM_REDUCE_OP = "mean"

VALID_SCORE_AGGREGATIONS = ("min", "mean", "max")
VALID_NORM_REDUCE_OPS = ("mean", "max", "sum")


def env_enabled() -> bool:
    raw = os.getenv(ENV_ENABLED)
    if raw is None:
        return DEFAULT_ENABLED
    try:
        return bool(int(raw))
    except ValueError:
        raise ValueError(
            f"{ENV_ENABLED} must be '0' or '1', got {raw!r}"
        ) from None


def env_compression_ratio() -> float:
    raw = os.getenv(ENV_COMPRESSION_RATIO)
    if raw is None:
        return DEFAULT_COMPRESSION_RATIO
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(
            f"{ENV_COMPRESSION_RATIO} must be a float in (0, 1], got {raw!r}"
        ) from None
    if not 0 < value <= 1:
        raise ValueError(
            f"{ENV_COMPRESSION_RATIO} must be in (0, 1], got {value}"
        )
    return value


def env_warmup_tokens() -> int:
    raw = os.getenv(ENV_WARMUP_TOKENS)
    if raw is None:
        return DEFAULT_WARMUP_TOKENS
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"{ENV_WARMUP_TOKENS} must be a non-negative integer, got {raw!r}"
        ) from None
    if value < 0:
        raise ValueError(
            f"{ENV_WARMUP_TOKENS} must be non-negative, got {value}"
        )
    return value


def env_score_aggregation() -> str:
    value = os.getenv(ENV_SCORE_AGGREGATION, DEFAULT_SCORE_AGGREGATION)
    if value not in VALID_SCORE_AGGREGATIONS:
        raise ValueError(
            f"{ENV_SCORE_AGGREGATION} must be one of "
            f"{VALID_SCORE_AGGREGATIONS}, got {value!r}"
        )
    return value


def env_norm_reduce_op() -> str:
    value = os.getenv(ENV_NORM_REDUCE_OP, DEFAULT_NORM_REDUCE_OP)
    if value not in VALID_NORM_REDUCE_OPS:
        raise ValueError(
            f"{ENV_NORM_REDUCE_OP} must be one of "
            f"{VALID_NORM_REDUCE_OPS}, got {value!r}"
        )
    return value


@dataclass
class KnormConfig:
    """Snapshot of the Knorm settings, validated at construction time."""

    compression_ratio: float = field(default_factory=env_compression_ratio)
    """Fraction of KV cache blocks to KEEP. 1.0 = no compression."""

    warmup_tokens: int = field(default_factory=env_warmup_tokens)
    """Tokens at the sequence start to always keep (attention sink)."""

    enabled: bool = field(default_factory=env_enabled)
    """Whether Knorm compression is enabled. Set ``VLLM_KNORM_ENABLED=0``
    to disable."""

    score_aggregation: str = field(default_factory=env_score_aggregation)
    """How to aggregate token norms within a block: 'min', 'mean', or
    'max'. Lower aggregated norm = more important = keep."""

    norm_reduce_op: str = field(default_factory=env_norm_reduce_op)
    """How to reduce norms across KV heads: 'mean', 'max', or 'sum'."""

    @property
    def is_active(self) -> bool:
        """Return True if compression should be applied."""
        return self.enabled and self.compression_ratio < 1.0


def should_activate(enable_prefix_caching: bool) -> bool:
    """Single source of truth for Knorm activation.

    Migrated verbatim in rule from the in-tree fix for legacy issue #163
    (PR #214): manager registration and the runner-side attention wrapper
    must consult the same predicate, otherwise a half-enabled state adds
    per-forward norm overhead with no consumer. Knorm is active only when
    ALL of:

    - ``VLLM_KNORM_ENABLED`` is true;
    - ``compression_ratio < 1.0`` (matches :attr:`KnormConfig.is_active`);
    - prefix caching is disabled (mutually exclusive; PR #134).
    """
    return (
        env_enabled()
        and env_compression_ratio() < 1.0
        and not enable_prefix_caching
    )


def keep_target_blocks(
    total_blocks: int, warmup_tokens: int, block_size: int, ratio: float
) -> int:
    """Blocks a request may keep under *ratio* (never below warmup)."""
    warmup_blocks = max(1, warmup_tokens // max(1, block_size))
    return max(warmup_blocks, math.ceil(total_blocks * ratio))


__all__ = [
    "DEFAULT_COMPRESSION_RATIO",
    "DEFAULT_ENABLED",
    "DEFAULT_NORM_REDUCE_OP",
    "DEFAULT_SCORE_AGGREGATION",
    "DEFAULT_WARMUP_TOKENS",
    "ENV_COMPRESSION_RATIO",
    "ENV_ENABLED",
    "ENV_NORM_REDUCE_OP",
    "ENV_SCORE_AGGREGATION",
    "ENV_WARMUP_TOKENS",
    "KnormConfig",
    "VALID_NORM_REDUCE_OPS",
    "VALID_SCORE_AGGREGATIONS",
    "env_compression_ratio",
    "env_enabled",
    "env_norm_reduce_op",
    "env_score_aggregation",
    "env_warmup_tokens",
    "keep_target_blocks",
    "should_activate",
]
