# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-HUST project
"""Runtime hook functions for Knorm.

Migrated from ``vllm/knorm/hooks.py`` in the archived ``vllm-hust`` tree
(legacy PR #76). In-tree these were called explicitly from
``GPUModelRunner``; as a plugin they are invoked by the runner wrappers
installed in :mod:`vllm_hust_knorm.adapters.vllm_hust.patches`, at the
same lifecycle points (after the model forward / before the sampled
output escapes the worker).

torch is imported lazily; the pure aggregation over request/block
bookkeeping is factored into :func:`aggregate_block_scores` so it is
testable without a host or a device.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .attention_backend import clear_pending_norms, get_pending_norms


def aggregate_token_scores(
    token_norms: Sequence[float],
    req_ids: Sequence[str],
    num_scheduled: Sequence[int],
    positions: Sequence[int],
    block_size: int,
    score_aggregation: str = "min",
) -> dict[str, list[tuple[int, float]]]:
    """Aggregate per-token norms into per-(request, block) scores.

    Pure bookkeeping: every token contributes its norm to its block;
    repeated blocks combine with *score_aggregation* ('min' by default —
    one important token marks the whole block important).
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    scores: dict[str, dict[int, list[float]]] = {}
    idx = 0
    for req_id, num_toks in zip(req_ids, num_scheduled, strict=False):
        per_block: dict[int, list[float]] = {}
        for _ in range(int(num_toks)):
            if idx >= len(token_norms) or idx >= len(positions):
                break
            block_idx = int(positions[idx]) // block_size
            per_block.setdefault(block_idx, []).append(float(token_norms[idx]))
            idx += 1
        if per_block:
            scores[req_id] = per_block

    def combine(values: list[float]) -> float:
        if score_aggregation == "min":
            return min(values)
        if score_aggregation == "max":
            return max(values)
        return sum(values) / len(values)

    return {
        req_id: [(b, combine(vals)) for b, vals in blocks.items()]
        for req_id, blocks in scores.items()
    }


def collect_knorm_scores(runner: Any, input_batch: Any) -> None:
    """Collect key L2 norms after the model forward and aggregate per
    block.

    Stores the result on ``runner._knorm_scores`` as
    ``{req_id: [(block_idx, score), ...]}``, or ``None`` when nothing
    was collected this step.
    """
    if getattr(runner, "kv_cache_config", None) is None:
        runner._knorm_scores = None
        return

    norms_list = get_pending_norms()
    if not norms_list:
        runner._knorm_scores = None
        return

    import torch  # noqa: PLC0415 — deliberate lazy device dependency

    # Stack norms from all layers and average. Shape: [num_tokens].
    all_norms = torch.stack(norms_list).float().mean(dim=0)
    clear_pending_norms()

    token_norms = all_norms.cpu().numpy()
    num_tokens = len(token_norms)

    num_reqs = input_batch.num_reqs
    num_scheduled = input_batch.num_tokens_no_spec[:num_reqs]
    positions_np = runner.positions[:num_tokens].cpu().numpy()

    block_size = runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size

    from .config import KnormConfig

    config = KnormConfig()
    scores = aggregate_token_scores(
        token_norms,
        input_batch.req_ids,
        num_scheduled,
        positions_np,
        block_size,
        score_aggregation=config.score_aggregation,
    )
    runner._knorm_scores = scores or None


def attach_knorm_scores(runner: Any, result: Any) -> None:
    """Attach knorm block scores to a model runner output.

    The scores are later read by the ``Scheduler.update_from_output``
    wrapper and routed to :func:`vllm_hust_knorm.knorm.manager.
    submit_block_scores`.
    """
    knorm_scores = getattr(runner, "_knorm_scores", None)
    if not knorm_scores or result is None:
        return

    result.knorm_block_scores = knorm_scores  # type: ignore[attr-defined]


__all__ = [
    "aggregate_token_scores",
    "attach_knorm_scores",
    "collect_knorm_scores",
]
