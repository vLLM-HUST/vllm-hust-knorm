# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-HUST project
"""Config parsing and the activation predicate (single source of truth).

The 8-combination matrix is migrated from the legacy host tests added
by PRs #134/#214 (legacy issue #163): ``should_activate`` and the
manager registration must agree on every combination.
"""

from __future__ import annotations

import pytest

from vllm_hust_knorm.knorm import config as cfg


class TestDefaults:
    def test_plugin_is_opt_in_by_default(self, clean_knorm_env):
        # Deliberate divergence from the in-tree module (default '1'):
        # an installed plugin must not change host behavior by default.
        assert cfg.env_enabled() is False

    def test_other_defaults_match_in_tree_module(self, clean_knorm_env):
        assert cfg.env_compression_ratio() == 0.5
        assert cfg.env_warmup_tokens() == 32
        assert cfg.env_score_aggregation() == "min"
        assert cfg.env_norm_reduce_op() == "mean"

    def test_snapshot_config(self, clean_knorm_env, monkeypatch):
        monkeypatch.setenv("VLLM_KNORM_ENABLED", "1")
        monkeypatch.setenv("VLLM_KNORM_COMPRESSION_RATIO", "0.25")
        snapshot = cfg.KnormConfig()
        assert snapshot.enabled is True
        assert snapshot.compression_ratio == 0.25
        assert snapshot.is_active is True

    def test_inactive_when_ratio_is_one(self, monkeypatch):
        monkeypatch.setenv("VLLM_KNORM_ENABLED", "1")
        monkeypatch.setenv("VLLM_KNORM_COMPRESSION_RATIO", "1.0")
        assert cfg.KnormConfig().is_active is False


class TestValidation:
    @pytest.mark.parametrize(
        ("variable", "raw"),
        [
            ("VLLM_KNORM_ENABLED", "maybe"),
            ("VLLM_KNORM_COMPRESSION_RATIO", "half"),
            ("VLLM_KNORM_COMPRESSION_RATIO", "0"),
            ("VLLM_KNORM_COMPRESSION_RATIO", "1.5"),
            ("VLLM_KNORM_WARMUP_TOKENS", "-4"),
            ("VLLM_KNORM_WARMUP_TOKENS", "many"),
            ("VLLM_KNORM_SCORE_AGGREGATION", "median"),
            ("VLLM_KNORM_NORM_REDUCE_OP", "p50"),
        ],
    )
    def test_invalid_values_fail_closed(self, monkeypatch, variable, raw):
        monkeypatch.setenv(variable, raw)
        with pytest.raises(ValueError, match=variable):
            cfg.KnormConfig()


class TestShouldActivateMatrix:
    """Migrated from tests/v1/test_kv_cache_spec_registry.py (PR #214)."""

    @pytest.mark.parametrize(
        ("enabled", "ratio", "prefix_caching", "expected"),
        [
            ("1", "0.5", False, True),   # fully active
            ("1", "0.5", True, False),   # prefix caching blocks
            ("1", "1.0", False, False),  # ratio=1.0 blocks
            ("1", "1.0", True, False),   # both block
            ("0", "0.5", False, False),  # disabled
            ("0", "0.5", True, False),   # disabled + prefix
            ("0", "1.0", False, False),  # all off
            ("0", "1.0", True, False),   # all off + prefix
        ],
    )
    def test_matrix(
        self, clean_knorm_env, monkeypatch, enabled, ratio, prefix_caching,
        expected,
    ):
        monkeypatch.setenv("VLLM_KNORM_ENABLED", enabled)
        monkeypatch.setenv("VLLM_KNORM_COMPRESSION_RATIO", ratio)
        assert cfg.should_activate(prefix_caching) is expected

    def test_unset_environment_is_inactive(self, clean_knorm_env):
        assert cfg.should_activate(False) is False
