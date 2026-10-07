# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-HUST project
"""Pure key-norm math shared by tests and the torch runtime path.

Migrated from ``vllm/knorm/attention_backend.py`` in the archived
``vllm-hust`` tree (legacy PR #76). The reduce-op selection is factored
out so it is unit-testable without torch; the device path in
:mod:`vllm_hust_knorm.knorm.attention_backend` applies the identical rule
to tensors.
"""

from __future__ import annotations

from collections.abc import Sequence

VALID_REDUCE_OPS = ("mean", "max", "sum")


def reduce_head_norms(head_norms: Sequence[float], reduce_op: str) -> float:
    """Reduce per-head L2 norms to one per-token score.

    Args:
        head_norms: L2 norm of each KV head for a single token.
        reduce_op: 'mean', 'max', or 'sum'.

    Lower score = lower norm = higher importance = keep.
    """
    if not head_norms:
        raise ValueError("head_norms must not be empty")
    if reduce_op == "max":
        return max(head_norms)
    if reduce_op == "sum":
        return float(sum(head_norms))
    if reduce_op == "mean":
        return sum(head_norms) / len(head_norms)
    raise ValueError(f"unknown reduce_op {reduce_op!r}")


def aggregate_block_scores(scores: Sequence[float], aggregation: str) -> float:
    """Aggregate token scores of one block into a block score.

    'min' (the default) marks a block important as soon as any of its
    tokens is important.
    """
    if not scores:
        raise ValueError("scores must not be empty")
    if aggregation == "min":
        return min(scores)
    if aggregation == "max":
        return max(scores)
    if aggregation == "mean":
        return sum(scores) / len(scores)
    raise ValueError(f"unknown aggregation {aggregation!r}")


__all__ = [
    "VALID_REDUCE_OPS",
    "aggregate_block_scores",
    "reduce_head_norms",
]
