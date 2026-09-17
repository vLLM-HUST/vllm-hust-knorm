# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-HUST project
"""Host-contract tests: the six idempotent patches against a fake host.

Coverage migrated from the legacy host tests of PRs #134/#214
(tests/v1/test_kv_cache_spec_registry.py): manager registration must
follow should_activate for every combination, and prefix caching must
keep the stock FullAttentionManager.
"""

from __future__ import annotations

import sys
import types

import pytest
from conftest import make_vllm_config

from vllm_hust_knorm.adapters.vllm_hust.patches import install_patches
from vllm_hust_knorm.knorm.manager import (
    drain_block_scores,
    get_knorm_manager_class,
)


def host_module(name: str):
    return sys.modules[name]


class TestInstallIdempotency:
    def test_install_reports_all_patches(self, fake_host):
        info = install_patches()
        assert info["already_installed"] is False
        assert all(info["patches"].values())

    def test_second_install_is_noop(self, fake_host):
        install_patches()
        info = install_patches()
        assert info["already_installed"] is True


class TestQueueAndFreeBlocks:
    def test_prependleft_n_inserts_at_head(self, fake_host):
        install_patches()
        queue = fake_host["FreeQueue"]()

        def block():
            return types.SimpleNamespace(next_free_block=None, prev_free_block=None)

        first, second = block(), block()
        queue.append_n([first])
        queue.prependleft_n([second])

        assert queue.fake_free_list_head.next_free_block is second
        assert second.next_free_block is first
        assert queue.num_free_blocks == 2

    def test_free_blocks_prepend_reuses_first(self, fake_host):
        install_patches()
        pool = fake_host["BlockPool"]()
        evicted = types.SimpleNamespace(ref_cnt=1, is_null=False, block_hash=None)
        cached = types.SimpleNamespace(ref_cnt=1, is_null=False, block_hash="h")

        pool.free_blocks([evicted], prepend=True)
        assert evicted.ref_cnt == 0
        assert pool.free_block_queue.num_free_blocks == 1
        assert pool.free_block_queue.fake_free_list_head.next_free_block is (evicted)

        pool.free_blocks([cached])  # default path still delegates
        assert cached.ref_cnt == 0
        assert pool.free_block_queue.num_free_blocks == 2

    def test_free_blocks_skips_wrap_when_host_has_prepend(self, fake_host):
        # 0.23-seam-era hosts already ship the in-tree signature.
        block_pool_cls = fake_host["BlockPool"]

        def free_blocks_with_prepend(self, ordered_blocks, prepend=False):
            pass

        block_pool_cls.free_blocks = free_blocks_with_prepend
        install_patches()
        assert block_pool_cls.free_blocks is free_blocks_with_prepend


class TestSpecRegistration:
    """P3 — migrated combination coverage from PRs #134/#214."""

    def run_registration(self, fake_host, config):
        registry = fake_host["registry"]
        registry._REGISTRY.clear()
        host_module(
            "vllm.v1.core.single_type_kv_cache_manager"
        ).register_all_kvcache_specs(config)
        return registry.get_manager_class(fake_host["FullAttentionSpec"]())

    def test_knorm_manager_registered_when_active(self, fake_host, monkeypatch, capsys):
        install_patches()
        monkeypatch.setenv("VLLM_KNORM_ENABLED", "1")
        monkeypatch.setenv("VLLM_KNORM_COMPRESSION_RATIO", "0.5")

        manager = self.run_registration(fake_host, make_vllm_config(False))

        assert manager is get_knorm_manager_class()
        # Server-side verification relies on this rg-able log marker.
        assert "knorm-manager-registered" in capsys.readouterr().out

    def test_prefix_caching_keeps_stock_manager_and_warns(self, fake_host, monkeypatch):
        install_patches()
        monkeypatch.setenv("VLLM_KNORM_ENABLED", "1")
        monkeypatch.setenv("VLLM_KNORM_COMPRESSION_RATIO", "0.5")

        manager = self.run_registration(fake_host, make_vllm_config(True))

        assert manager is fake_host["FullAttentionManager"]
        warnings = fake_host["logger"].warnings
        assert any("prefix caching" in message for message in warnings)

    def test_no_prefix_warning_when_knorm_disabled(self, fake_host, monkeypatch):
        install_patches()
        monkeypatch.setenv("VLLM_KNORM_ENABLED", "0")

        manager = self.run_registration(fake_host, make_vllm_config(True))

        assert manager is fake_host["FullAttentionManager"]
        assert fake_host["logger"].warnings == []

    def test_none_config_keeps_stock_manager(self, fake_host, monkeypatch):
        install_patches()
        monkeypatch.setenv("VLLM_KNORM_ENABLED", "1")

        manager = self.run_registration(fake_host, None)

        assert manager is fake_host["FullAttentionManager"]

    def test_full_matrix_agrees_with_should_activate(self, fake_host, monkeypatch):
        install_patches()
        from vllm_hust_knorm.knorm.config import should_activate

        cases = [
            ("1", "0.5", False),
            ("1", "0.5", True),
            ("1", "1.0", False),
            ("1", "1.0", True),
            ("0", "0.5", False),
            ("0", "0.5", True),
            ("0", "1.0", False),
            ("0", "1.0", True),
        ]
        for enabled, ratio, prefix_caching in cases:
            monkeypatch.setenv("VLLM_KNORM_ENABLED", enabled)
            monkeypatch.setenv("VLLM_KNORM_COMPRESSION_RATIO", ratio)

            manager = self.run_registration(fake_host, make_vllm_config(prefix_caching))

            if should_activate(prefix_caching):
                assert manager is get_knorm_manager_class(), (
                    enabled,
                    ratio,
                    prefix_caching,
                )
            else:
                assert manager is fake_host["FullAttentionManager"], (
                    enabled,
                    ratio,
                    prefix_caching,
                )

    def test_mla_spec_moves_together_with_full_attention(self, fake_host, monkeypatch):
        install_patches()
        monkeypatch.setenv("VLLM_KNORM_ENABLED", "1")
        registry = fake_host["registry"]
        registry._REGISTRY.clear()
        host_module(
            "vllm.v1.core.single_type_kv_cache_manager"
        ).register_all_kvcache_specs(make_vllm_config(False))

        manager = registry.get_manager_class(fake_host["MLAAttentionSpec"]())

        assert manager is get_knorm_manager_class()


class TestSchedulerRouting:
    def test_scores_routed_into_manager_bridge(self, fake_host):
        install_patches()
        scheduler = fake_host["Scheduler"]()
        output = types.SimpleNamespace(knorm_block_scores={"r1": [(0, 1.0)]})

        result = scheduler.update_from_output("sched-out", output)

        assert result == "host-result"
        assert drain_block_scores() == {"r1": [(0, 1.0)]}

    def test_output_without_scores_untouched(self, fake_host):
        install_patches()
        scheduler = fake_host["Scheduler"]()

        result = scheduler.update_from_output("sched-out", types.SimpleNamespace())

        assert result == "host-result"
        assert drain_block_scores() == {}


class TestRunnerPatches:
    def make_runner(self, fake_host, prefix_caching: bool):
        return fake_host["GPUModelRunner"](
            cache_config=types.SimpleNamespace(enable_prefix_caching=prefix_caching)
        )

    def test_active_runner_installs_attention_wrapper(
        self, fake_host, monkeypatch, capsys
    ):
        install_patches()
        monkeypatch.setenv("VLLM_KNORM_ENABLED", "1")

        runner = self.make_runner(fake_host, prefix_caching=False)

        assert runner._knorm_active is True
        assert runner._knorm_wrapper_installed is True
        from vllm_hust_knorm.knorm import attention_backend

        assert attention_backend._original_forward is not None
        # Server-side verification relies on this rg-able log marker.
        assert "knorm-wrapper-installed" in capsys.readouterr().out

    def test_inactive_runner_installs_nothing(self, fake_host, monkeypatch):
        install_patches()
        monkeypatch.setenv("VLLM_KNORM_ENABLED", "0")

        runner = self.make_runner(fake_host, prefix_caching=False)

        assert runner._knorm_active is False
        assert runner._knorm_wrapper_installed is False
        from vllm_hust_knorm.knorm import attention_backend

        assert attention_backend._original_forward is None

    def test_prefix_caching_blocks_wrapper(self, fake_host, monkeypatch):
        install_patches()
        monkeypatch.setenv("VLLM_KNORM_ENABLED", "1")

        runner = self.make_runner(fake_host, prefix_caching=True)

        assert runner._knorm_active is False
        assert runner._knorm_wrapper_installed is False

    def test_wrapped_attention_forward_stores_norms(self, fake_host, monkeypatch):
        torch = pytest.importorskip("torch")

        install_patches()
        monkeypatch.setenv("VLLM_KNORM_ENABLED", "1")
        self.make_runner(fake_host, prefix_caching=False)

        impl = fake_host["AttentionImpl"]()
        key = torch.zeros(3, 2, 4)
        result = impl.forward(
            layer="l",
            query=None,
            key=key,
            value=None,
            kv_cache=None,
            attn_metadata=None,
        )

        assert result == "attn-output"
        from vllm_hust_knorm.knorm import attention_backend

        norms = attention_backend.get_pending_norms()
        assert len(norms) == 1
        assert norms[0].shape == (3,)

    def test_sample_tokens_collects_and_attaches_when_wrapper_on(
        self, fake_host, monkeypatch
    ):
        install_patches()
        monkeypatch.setenv("VLLM_KNORM_ENABLED", "1")
        runner = self.make_runner(fake_host, prefix_caching=False)
        runner._knorm_scores = {"r1": [(0, 1.0)]}

        # No forward happened this step (kv_cache_config unset): collect
        # clears the stale scores and attach is a no-op.
        result = runner.sample_tokens(None)

        assert hasattr(result, "sampled_token_ids")
        assert not hasattr(result, "knorm_block_scores")
        assert runner._knorm_scores is None

    def test_sample_tokens_delegates_when_wrapper_off(self, fake_host):
        install_patches()
        runner = self.make_runner(fake_host, prefix_caching=False)
        runner._knorm_wrapper_installed = False
        runner._knorm_scores = {"r1": [(0, 1.0)]}

        result = runner.sample_tokens("grammar")

        assert runner.last_grammar_output == "grammar"
        # Collect was skipped, attach still ran (wrapper flag off →
        # nothing is attached; legacy guard semantics).
        assert not hasattr(result, "knorm_block_scores")


class TestAttachHook:
    def test_attach_sets_attribute(self):
        from vllm_hust_knorm.knorm.hooks import attach_knorm_scores

        runner = types.SimpleNamespace(_knorm_scores={"r1": [(0, 1.0)]})
        output = types.SimpleNamespace()

        attach_knorm_scores(runner, output)

        assert output.knorm_block_scores == {"r1": [(0, 1.0)]}

    def test_attach_noop_without_scores(self):
        from vllm_hust_knorm.knorm.hooks import attach_knorm_scores

        runner = types.SimpleNamespace(_knorm_scores=None)
        output = types.SimpleNamespace()

        attach_knorm_scores(runner, output)

        assert not hasattr(output, "knorm_block_scores")
