# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-HUST project
"""Wire-contract tests for scheduler/worker full-table replacement."""

from __future__ import annotations

import pickle

import pytest

from vllm_hust_knorm.knorm.block_table_sync import (
    FULL_REPLACEMENT_SENTINEL,
    clear_block_table_dirty,
    decode_full_replacement,
    dirty_requests,
    encode_full_replacement,
    mark_block_table_dirty,
)


def test_full_replacement_round_trips_all_cache_groups() -> None:
    original = ([1, 0, 3], [8, 9], [])

    marked, decoded = decode_full_replacement(encode_full_replacement(original))

    assert marked is True
    assert decoded == original


def test_marker_survives_scheduler_worker_pickle_boundary() -> None:
    encoded = encode_full_replacement(([1, 0, 2], [7]))

    transported = pickle.loads(pickle.dumps(encoded))
    marked, decoded = decode_full_replacement(transported)

    assert marked is True
    assert decoded == ([1, 0, 2], [7])


@pytest.mark.parametrize(
    "payload",
    [
        ([1, FULL_REPLACEMENT_SENTINEL],),
        (
            [FULL_REPLACEMENT_SENTINEL, 1],
            [FULL_REPLACEMENT_SENTINEL, 2],
        ),
    ],
)
def test_malformed_markers_fail_closed(payload: tuple[list[int], ...]) -> None:
    with pytest.raises(RuntimeError, match="malformed|ambiguous"):
        decode_full_replacement(payload)


def test_dirty_lifecycle_is_request_scoped() -> None:
    mark_block_table_dirty("r1")
    mark_block_table_dirty("r2")

    assert dirty_requests(["r2", "r3"]) == {"r2"}
    clear_block_table_dirty("r2")
    assert dirty_requests(["r1", "r2"]) == {"r1"}
    clear_block_table_dirty("r1")
