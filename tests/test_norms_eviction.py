# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-HUST project
"""Pure norm math and eviction planning — migrated semantics from the
in-tree KnormFullAttentionManager (legacy PR #76)."""

from __future__ import annotations

import math

import pytest

from vllm_hust_knorm.knorm.eviction import (
    EvictionDecision,
    num_skipped_tokens,
    plan_block_eviction,
)
from vllm_hust_knorm.knorm.norms import (
    aggregate_block_scores,
    reduce_head_norms,
)


class TestNorms:
    def test_reduce_ops(self):
        values = [1.0, 2.0, 3.0]
        assert reduce_head_norms(values, "mean") == pytest.approx(2.0)
        assert reduce_head_norms(values, "max") == 3.0
        assert reduce_head_norms(values, "sum") == 6.0

    def test_reduce_rejects_unknown_and_empty(self):
        with pytest.raises(ValueError):
            reduce_head_norms([1.0], "p50")
        with pytest.raises(ValueError):
            reduce_head_norms([], "mean")

    def test_aggregation_default_min_keeps_important_tokens(
        self,
    ):
        # One low-norm (important) token protects the whole block.
        assert aggregate_block_scores([5.0, 0.1, 8.0], "min") == 0.1
        assert aggregate_block_scores([5.0, 0.1, 8.0], "max") == 8.0
        assert aggregate_block_scores([5.0, 0.1, 8.0], "mean") == (
            pytest.approx(13.1 / 3)
        )


class TestNumSkippedTokens:
    RATIO = 0.5
    WARMUP = 32
    BLOCK = 16

    def test_below_warmup_region_never_skips(self):
        assert (
            num_skipped_tokens(
                self.WARMUP,
                block_size=self.BLOCK,
                warmup_tokens=self.WARMUP,
                compression_ratio=self.RATIO,
            )
            == 0
        )

    def test_half_of_a_64_block_prefix(self):
        # 64 blocks total, keep ceil(64*0.5)=32, warmup=2 → 30 evictable
        # blocks (60 skipped of 1024 computed tokens: 1024 → 64 blocks,
        # keep max(2, 32) = 32 → skip 32 blocks = 512 tokens).
        assert (
            num_skipped_tokens(
                1024,
                block_size=self.BLOCK,
                warmup_tokens=self.WARMUP,
                compression_ratio=self.RATIO,
            )
            == 512
        )

    def test_keep_target_never_below_warmup(self):
        # 4 blocks (64 tokens), warmup 2 blocks, ratio 0.1 → keep 2.
        assert (
            num_skipped_tokens(
                64,
                block_size=self.BLOCK,
                warmup_tokens=self.WARMUP,
                compression_ratio=0.1,
            )
            == 32
        )

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            num_skipped_tokens(
                10, block_size=0, warmup_tokens=8, compression_ratio=0.5
            )
        with pytest.raises(ValueError):
            num_skipped_tokens(
                10, block_size=16, warmup_tokens=8, compression_ratio=1.5
            )


class TestPlanBlockEviction:
    def make_plan(self, **overrides):
        kwargs = dict(
            total_computed_tokens=1024,
            num_allocated_blocks=64,
            block_size=16,
            warmup_tokens=32,
            compression_ratio=0.5,
            req_scores={},
            is_null=lambda _i: False,
            is_cached=lambda _i: False,
        )
        kwargs.update(overrides)
        return plan_block_eviction(**kwargs)

    def test_none_when_nothing_to_evict(self):
        assert self.make_plan(total_computed_tokens=32) is None
        assert self.make_plan(compression_ratio=1.0) is None

    def test_highest_norm_evicted_first_and_warmup_protected(self):
        # Warmup = 2 blocks; window is blocks [2, 64) with ratio 0.5 →
        # 30 candidates to evict, but scores make block 3 worst.
        scores = {2: 1.0, 3: 99.0, 4: 50.0}
        plan = self.make_plan(req_scores=scores)
        assert plan is not None
        assert plan.indices[0] == 3  # highest norm first
        assert 0 not in plan.indices and 1 not in plan.indices  # warmup
        assert len(plan.indices) == 30

    def test_unscored_blocks_evicted_after_scored(self):
        scores = {2: 10.0, 3: 20.0}
        plan = self.make_plan(req_scores=scores)
        # Scored candidates (worst first) precede every unscored one.
        assert plan.indices[0] == 3
        assert plan.indices[1] == 2
        assert 4 in plan.indices

    def test_null_blocks_are_skipped(self):
        plan = self.make_plan(is_null=lambda i: i < 10)
        assert plan is not None
        assert all(index >= 10 for index in plan.indices)

    def test_cached_flags_partition(self):
        plan = self.make_plan(is_cached=lambda i: i % 2 == 0)
        assert isinstance(plan, EvictionDecision)
        flags = dict(zip(plan.indices, plan.cached, strict=True))
        assert flags[2] is True and flags[3] is False

    def test_eviction_count_matches_keep_target(self):
        total_blocks = 64
        ratio = 0.25
        warmup_blocks = 2
        plan = self.make_plan(compression_ratio=ratio)
        keep = max(warmup_blocks, math.ceil(total_blocks * ratio))
        # Candidates exclude the warmup region, so the effective keep
        # set is the warmup blocks plus the keep target's remainder
        # (legacy semantics: target_evict = skipped - warmup).
        assert len(plan.indices) == total_blocks - keep - warmup_blocks


class TestManagerAgainstFakeHost:
    """The host-side apply path (uses the fake vllm from conftest)."""

    def test_manager_applies_plan_to_block_table(
        self, fake_host, monkeypatch
    ):
        import types

        from vllm_hust_knorm.adapters.vllm_hust.patches import install_patches
        from vllm_hust_knorm.knorm.manager import (
            get_knorm_manager_class,
            submit_block_scores,
        )

        # The manager frees uncached evicted blocks with
        # free_blocks(prepend=True), which the patch layer provides.
        assert install_patches()["already_installed"] is False

        monkeypatch.setenv("VLLM_KNORM_ENABLED", "1")
        monkeypatch.setenv("VLLM_KNORM_COMPRESSION_RATIO", "0.5")

        spec = fake_host["FullAttentionSpec"]()
        pool = fake_host["BlockPool"]()
        manager = get_knorm_manager_class()(spec, block_pool=pool)

        blocks = [
            types.SimpleNamespace(
                ref_cnt=1,
                is_null=False,
                block_hash=None,
                next_free_block=None,
                prev_free_block=None,
            )
            for _ in range(64)
        ]
        # Give a few blocks distinct scores; block 3 has the worst norm.
        manager.req_to_blocks["r1"] = blocks
        original_blocks = list(blocks)
        submit_block_scores({"r1": [(2, 1.0), (3, 99.0), (4, 5.0)]})

        manager.remove_skipped_blocks("r1", 1024)

        evicted = [i for i, b in enumerate(blocks) if b is manager._null_block]
        assert 3 in evicted and 0 not in evicted and 1 not in evicted
        assert len(evicted) == 30
        # Uncached evicted blocks were freed at the head of the queue,
        # worst (highest-norm) block first.
        head = pool.free_block_queue.fake_free_list_head.next_free_block
        assert head is original_blocks[3]
        assert pool.free_block_queue.num_free_blocks == 30

    def test_manager_free_cleans_scores(self, fake_host):
        from vllm_hust_knorm.knorm.manager import get_knorm_manager_class

        spec = fake_host["FullAttentionSpec"]()
        pool = fake_host["BlockPool"]()
        manager = get_knorm_manager_class()(spec, block_pool=pool)
        manager._block_scores["r1"] = {0: 1.0}
        manager.req_to_blocks["r1"] = []

        manager.free("r1")

        assert "r1" not in manager._block_scores
        assert manager.freed_requests == ["r1"]
