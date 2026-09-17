# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-HUST project
"""Pure eviction planning for Knorm.

Migrated from ``KnormFullAttentionManager`` in the archived ``vllm-hust``
tree (legacy PR #76, eviction semantics preserved through #134/#214).
The planning steps are factored out so the eviction policy is unit
testable on any machine; :mod:`vllm_hust_knorm.knorm.manager` applies the
plan against live host block tables.

Policy recap (Devoto et al., 2024): blocks whose keys have **high** L2
norms receive disproportionately low attention during decoding, so they
are safe to evict first. The warmup region at the sequence start (the
attention sink) is never evicted, and unscored blocks (never observed by
a forward pass) are evicted only after every scored candidate.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass


def num_skipped_tokens(
    num_computed_tokens: int,
    *,
    block_size: int,
    warmup_tokens: int,
    compression_ratio: float,
) -> int:
    """Tokens to evict from the prefix for a request of
    *num_computed_tokens* computed tokens.

    Returns 0 when nothing qualifies. The value is always a whole number
    of blocks.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if not 0 < compression_ratio <= 1:
        raise ValueError(
            f"compression_ratio must be in (0, 1], got {compression_ratio}"
        )

    total_blocks = -(-num_computed_tokens // block_size)  # ceil div
    warmup_blocks = max(1, warmup_tokens // block_size)
    if total_blocks <= warmup_blocks:
        return 0

    target_keep_blocks = max(
        warmup_blocks, math.ceil(total_blocks * compression_ratio)
    )
    evict_blocks = total_blocks - target_keep_blocks
    if evict_blocks <= 0:
        return 0
    return evict_blocks * block_size


@dataclass(frozen=True)
class EvictionDecision:
    """Which positions of a request's block list to evict.

    ``cached`` is parallel to ``indices`` and records whether the block
    carried a prefix-cache hash (freed at the tail of the free queue) or
    not (freed at the head for immediate reuse).
    """

    indices: tuple[int, ...]
    cached: tuple[bool, ...]


def plan_block_eviction(
    *,
    total_computed_tokens: int,
    num_allocated_blocks: int,
    block_size: int,
    warmup_tokens: int,
    compression_ratio: float,
    req_scores: dict[int, float],
    is_null: Callable[[int], bool],
    is_cached: Callable[[int], bool],
) -> EvictionDecision | None:
    """Plan which blocks of one request to evict, or ``None``.

    ``req_scores`` maps block index within the request to its importance
    score (lower = more important). ``is_null``/``is_cached`` are asked
    about block indices within the request's block list.
    """
    skipped = num_skipped_tokens(
        total_computed_tokens,
        block_size=block_size,
        warmup_tokens=warmup_tokens,
        compression_ratio=compression_ratio,
    )
    if skipped <= 0:
        return None

    num_skipped_blocks = min(skipped // block_size, num_allocated_blocks)
    warmup_blocks = max(1, warmup_tokens // block_size)
    target_evict = num_skipped_blocks - warmup_blocks
    if target_evict <= 0:
        return None

    scored: list[tuple[int, float]] = []
    unscored: list[int] = []
    for i in range(warmup_blocks, num_skipped_blocks):
        if i >= num_allocated_blocks or is_null(i):
            continue
        score = req_scores.get(i)
        if score is not None:
            scored.append((i, score))
        else:
            unscored.append(i)

    # Highest score (highest norm = least important) first; unscored
    # blocks follow every scored candidate.
    scored.sort(key=lambda item: item[1], reverse=True)
    candidates = [index for index, _ in scored] + unscored

    num_to_evict = min(target_evict, len(candidates))
    if num_to_evict <= 0:
        return None

    chosen = candidates[:num_to_evict]
    return EvictionDecision(
        indices=tuple(chosen),
        cached=tuple(is_cached(index) for index in chosen),
    )


__all__ = [
    "EvictionDecision",
    "num_skipped_tokens",
    "plan_block_eviction",
]
