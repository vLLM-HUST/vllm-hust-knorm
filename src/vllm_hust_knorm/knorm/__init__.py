# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-HUST project
"""Knorm KV-cache compression for vllm-hust (plugin-owned module).

Knorm (Devoto et al., 2024) evicts KV-cache blocks with high key L2
norms, which tend to receive lower attention scores during decoding.
Importing this package is lazy and side-effect free: torch, vllm and
device modules are only touched by the functions that need them.
"""

from .config import KnormConfig, should_activate
from .eviction import EvictionDecision, num_skipped_tokens, plan_block_eviction
from .norms import aggregate_block_scores, reduce_head_norms

__all__ = [
    "EvictionDecision",
    "KnormConfig",
    "aggregate_block_scores",
    "num_skipped_tokens",
    "plan_block_eviction",
    "reduce_head_norms",
    "should_activate",
]
