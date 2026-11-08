# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-HUST project
"""Fail-closed scheduler/worker block-table replacement protocol.

The native cached-request protocol appends ``new_block_ids`` for running
requests. KNorm mutates existing positions, so an eviction needs an explicit
full-table replacement that survives scheduler/worker serialization.  The
marker is carried inside the existing integer-list payload and removed before
the native worker consumes it.
"""

from __future__ import annotations

from threading import Lock

# Block IDs are non-negative int32 values.  Reserve the minimum int32 value as
# a wire marker so dataclass/msgpack serializers preserve the signal without a
# host schema change.
FULL_REPLACEMENT_SENTINEL = -(2**31)

_dirty_requests: set[str] = set()
_dirty_lock = Lock()


def mark_block_table_dirty(request_id: str) -> None:
    """Require a full scheduler-to-worker table replacement for *request_id*."""
    with _dirty_lock:
        _dirty_requests.add(request_id)


def dirty_requests(request_ids: list[str]) -> set[str]:
    """Return dirty IDs present in this scheduler output without consuming them."""
    with _dirty_lock:
        return _dirty_requests.intersection(request_ids)


def clear_block_table_dirty(request_id: str) -> None:
    """Acknowledge that a replacement payload was constructed or is obsolete."""
    with _dirty_lock:
        _dirty_requests.discard(request_id)


def reset_for_tests() -> None:
    with _dirty_lock:
        _dirty_requests.clear()


def encode_full_replacement(
    block_ids: tuple[list[int], ...],
) -> tuple[list[int], ...]:
    """Encode a full multi-group table in the native ``new_block_ids`` shape."""
    if not block_ids:
        raise RuntimeError("KNorm cannot replace a block table with no cache groups")
    copied = tuple(list(group) for group in block_ids)
    if any(FULL_REPLACEMENT_SENTINEL in group for group in copied):
        raise RuntimeError("KNorm block table contains the reserved replacement marker")
    return ([FULL_REPLACEMENT_SENTINEL, *copied[0]], *copied[1:])


def decode_full_replacement(
    block_ids: tuple[list[int], ...],
) -> tuple[bool, tuple[list[int], ...]]:
    """Remove the wire marker and return ``(is_replacement, clean_payload)``."""
    if not block_ids:
        return False, block_ids
    marked = bool(block_ids[0]) and block_ids[0][0] == FULL_REPLACEMENT_SENTINEL
    marker_count = sum(
        value == FULL_REPLACEMENT_SENTINEL for group in block_ids for value in group
    )
    if not marked:
        if marker_count:
            raise RuntimeError("malformed KNorm block-table replacement payload")
        return False, block_ids
    if marker_count != 1:
        raise RuntimeError("ambiguous KNorm block-table replacement payload")
    return True, (list(block_ids[0][1:]), *(list(group) for group in block_ids[1:]))


__all__ = [
    "FULL_REPLACEMENT_SENTINEL",
    "clear_block_table_dirty",
    "decode_full_replacement",
    "dirty_requests",
    "encode_full_replacement",
    "mark_block_table_dirty",
]
