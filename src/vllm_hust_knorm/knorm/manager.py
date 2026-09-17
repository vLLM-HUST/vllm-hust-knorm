# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-HUST project
"""Knorm KV cache manager — evicts blocks based on key L2 norms.

Migrated from ``vllm/knorm/manager.py`` in the archived ``vllm-hust``
tree (legacy PR #76; eviction semantics preserved through #134/#214).
The manager subclasses the host ``FullAttentionManager``, so the class
is built lazily against the live host base (same pattern as the
reference plugin's ``_build_impl_cls``); importing this module never
imports vllm.
"""

from __future__ import annotations

from collections import defaultdict
from functools import cache
from typing import Any

from .config import KnormConfig
from .eviction import plan_block_eviction

# ---------------------------------------------------------------------------
# Global bridge: model runner (attention) → KV cache manager.
# Crosses the worker/scheduler boundary inside one engine process, same
# as the in-tree implementation.
# ---------------------------------------------------------------------------
_pending_block_scores: dict[str, list[tuple[int, float]]] = {}


def submit_block_scores(scores: dict[str, list[tuple[int, float]]]) -> None:
    """Store block-level importance scores from the model runner."""
    _pending_block_scores.update(scores)


def drain_block_scores() -> dict[str, list[tuple[int, float]]]:
    """Read and clear pending scores. Called by the manager each step."""
    result = dict(_pending_block_scores)
    _pending_block_scores.clear()
    return result


def reset_for_tests() -> None:
    """Clear the score bridge (tests only)."""
    _pending_block_scores.clear()


@cache
def get_knorm_manager_class() -> type:
    """Build ``KnormFullAttentionManager`` against the live host base.

    The host class is imported here, not at module import time, so the
    plugin package stays importable (and testable) without vllm.
    """
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

    class KnormFullAttentionManager(FullAttentionManager):
        """Full attention manager with Knorm-based KV cache compression.

        Overrides ``get_num_skipped_tokens`` and
        ``remove_skipped_blocks`` to evict blocks from the prefix based
        on the compression ratio. When the config is inactive it behaves
        identically to :class:`FullAttentionManager`.
        """

        def __init__(self, kv_cache_spec: Any, **kwargs: Any) -> None:
            super().__init__(kv_cache_spec, **kwargs)
            self._config = KnormConfig()
            # Per-request: block_index_in_request → importance score.
            # Lower score = lower norm = higher importance = keep.
            self._block_scores: dict[str, dict[int, float]] = defaultdict(dict)

        # ------------------------------------------------------------------
        # Public API for receiving norm data
        # ------------------------------------------------------------------

        def update_block_scores(
            self, scores: dict[str, list[tuple[int, float]]]
        ) -> None:
            """Ingest importance scores. Lower score = more important."""
            for req_id, block_scores in scores.items():
                req_dict = self._block_scores[req_id]
                for block_idx, score in block_scores:
                    if block_idx in req_dict:
                        req_dict[block_idx] = min(req_dict[block_idx], score)
                    else:
                        req_dict[block_idx] = score

        # ------------------------------------------------------------------
        # Overrides — core eviction logic
        # ------------------------------------------------------------------

        def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
            """Tokens to evict from the prefix based on the ratio."""
            if not self._config.is_active:
                return 0
            from .eviction import num_skipped_tokens

            return num_skipped_tokens(
                num_computed_tokens,
                block_size=self.block_size,
                warmup_tokens=self._config.warmup_tokens,
                compression_ratio=self._config.compression_ratio,
            )

        def remove_skipped_blocks(
            self, request_id: str, total_computed_tokens: int
        ) -> None:
            """Remove the least important blocks from the prefix.

            Drains the global score buffer, plans the eviction with the
            pure planner, then applies it to the live block table:
            evicted positions become the null block; cached blocks are
            freed at the tail of the free queue (LRU), uncached blocks
            at the head (immediate reuse).
            """
            # ── 1. Drain and ingest pending scores ──
            scores = drain_block_scores()
            if scores:
                self.update_block_scores(scores)

            if not self._config.is_active:
                return

            # ── 2. Plan the eviction against the live block list ──
            blocks = self.req_to_blocks[request_id]
            decision = plan_block_eviction(
                total_computed_tokens=total_computed_tokens,
                num_allocated_blocks=len(blocks),
                block_size=self.block_size,
                warmup_tokens=self._config.warmup_tokens,
                compression_ratio=self._config.compression_ratio,
                req_scores=self._block_scores.get(request_id, {}),
                is_null=lambda i: blocks[i] is self._null_block,
                is_cached=lambda i: blocks[i].block_hash is not None,
            )
            if decision is None:
                return

            # ── 3. Apply: replace with null_block and free ──
            removed_cached = []
            removed_uncached = []
            for index, was_cached in zip(
                decision.indices, decision.cached, strict=True
            ):
                block = blocks[index]
                if was_cached:
                    removed_cached.append(block)
                else:
                    removed_uncached.append(block)
                blocks[index] = self._null_block

            if removed_cached:
                self.block_pool.free_blocks(removed_cached)
            if removed_uncached:
                self.block_pool.free_blocks(removed_uncached, prepend=True)

            # ── 4. Clean up score records for evicted indices ──
            if request_id in self._block_scores:
                for index in decision.indices:
                    self._block_scores[request_id].pop(index, None)

        def free(self, request_id: str) -> None:
            """Free blocks and clean up score records."""
            super().free(request_id)
            self._block_scores.pop(request_id, None)

    return KnormFullAttentionManager


__all__ = [
    "drain_block_scores",
    "get_knorm_manager_class",
    "reset_for_tests",
    "submit_block_scores",
]
