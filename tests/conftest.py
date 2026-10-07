# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-HUST project
"""Test fixtures: a fake vLLM-HUST host and a clean KNorm environment.

The fake host reproduces exactly the surfaces documented in
HOST_CONTRACT.md (see ``vllm_hust_knorm.core.hosts.REQUIRED_SURFACES``)
so the patch layer, the activation pipeline and the host-contract
behavior are testable on any machine without vllm or a device.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from typing import Any

import pytest

KNORM_ENV_VARS = (
    "VLLM_KNORM_ENABLED",
    "VLLM_KNORM_COMPRESSION_RATIO",
    "VLLM_KNORM_WARMUP_TOKENS",
    "VLLM_KNORM_SCORE_AGGREGATION",
    "VLLM_KNORM_NORM_REDUCE_OP",
)

# Modules the plugin may import from the host; restored after each test.
HOST_MODULE_PREFIXES = ("vllm",)


def make_module(name: str) -> types.ModuleType:
    """Create a fake module that ``importlib.util.find_spec`` resolves.

    The spec carries a loader (like a real module); loader-less specs
    mean namespace packages and are treated as "not present" by the
    plugin's conflict guards (see ``real_module_present``).
    """
    module = types.ModuleType(name)
    loader = importlib.machinery.SourceFileLoader(name, f"/nonexistent/{name}.py")
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=loader)
    module.__package__ = name
    return module


def register_module(name: str) -> types.ModuleType:
    """Register *name* (and its parents) in ``sys.modules``."""
    parts = name.split(".")
    for depth in range(1, len(parts) + 1):
        partial = ".".join(parts[:depth])
        if partial not in sys.modules:
            sys.modules[partial] = make_module(partial)
    return sys.modules[name]


class _FakeLogger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning_once(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)


def build_fake_host() -> dict[str, Any]:
    """Assemble the fake vllm host tree and return its key objects."""
    host: dict[str, Any] = {}

    vllm = register_module("vllm")
    vllm.__path__ = []  # mark as package for submodule lookups

    logger_module = register_module("vllm.logger")
    fake_logger = _FakeLogger()
    logger_module.init_logger = lambda _name: fake_logger
    host["logger"] = fake_logger

    # --- vllm.v1.kv_cache_interface -----------------------------------
    kci = register_module("vllm.v1.kv_cache_interface")

    class FullAttentionSpec:
        block_size: int = 16

    class MLAAttentionSpec(FullAttentionSpec):
        pass

    kci.FullAttentionSpec = FullAttentionSpec
    kci.MLAAttentionSpec = MLAAttentionSpec
    host["FullAttentionSpec"] = FullAttentionSpec
    host["MLAAttentionSpec"] = MLAAttentionSpec

    # --- vllm.v1.kv_cache_spec_registry --------------------------------
    registry_module = register_module("vllm.v1.kv_cache_spec_registry")

    class KVCacheSpecMetadata:
        def __init__(self, spec, manager_class, uniform_type_base_spec):
            self.kvcache_spec_cls = spec
            self.manager_class = manager_class
            self.uniform_type_base_spec = uniform_type_base_spec

    class KVCacheSpecRegistry:
        _REGISTRY: dict[type, KVCacheSpecMetadata] = {}

        @classmethod
        def register(cls, spec, manager_class, uniform_type_base_spec=None):
            if uniform_type_base_spec is None:
                uniform_type_base_spec = spec
            assert issubclass(spec, uniform_type_base_spec)
            existing = cls._REGISTRY.get(spec)
            if existing is not None:
                same = (
                    existing.manager_class is manager_class
                    and existing.uniform_type_base_spec is uniform_type_base_spec
                )
                assert same, f"Conflicting registration for {spec.__name__}"
            cls._REGISTRY[spec] = KVCacheSpecMetadata(
                spec, manager_class, uniform_type_base_spec
            )

        @classmethod
        def get_manager_class(cls, spec_instance):
            for base in type(spec_instance).__mro__:
                if base in cls._REGISTRY:
                    return cls._REGISTRY[base].manager_class
            return None

    registry_module.KVCacheSpecRegistry = KVCacheSpecRegistry
    registry_module._REGISTRY_KVCACHESPEC_LIST = KVCacheSpecRegistry._REGISTRY
    host["registry"] = KVCacheSpecRegistry

    # --- vllm.v1.core.kv_cache_utils / free-block queue ----------------
    # Mirrors the real host: FreeKVCacheBlockQueue with a native
    # prepend_n (current vllm-hust main). The 0.23-seam era named it
    # FreeQueue and lacked head insertion — the legacy variant is
    # exposed for the fallback test via make_legacy_queue().
    kcu = register_module("vllm.v1.core.kv_cache_utils")

    class FreeKVCacheBlockQueue:
        def __init__(self):
            self.fake_free_list_head = types.SimpleNamespace(next_free_block=None)
            self.num_free_blocks = 0
            self.blocks: list = []
            self.prepend_calls: list[list] = []

        def append_n(self, blocks):
            for i in range(len(blocks) - 1):
                blocks[i].next_free_block = blocks[i + 1]
                blocks[i + 1].prev_free_block = blocks[i]
            if blocks:
                head = self.fake_free_list_head
                if head.next_free_block is None:
                    head.next_free_block = blocks[0]
                    blocks[0].prev_free_block = head
                    blocks[-1].next_free_block = None
                self.blocks.extend(blocks)
            self.num_free_blocks += len(blocks)

        def prepend_n(self, blocks):
            self.prepend_calls.append(list(blocks))
            if not blocks:
                return
            if self.fake_free_list_head.next_free_block is None:
                self.append_n(blocks)
                return
            first, rest = blocks[0], blocks[1:]
            for i in range(len(blocks) - 1):
                blocks[i].next_free_block = blocks[i + 1]
                blocks[i + 1].prev_free_block = blocks[i]
            old_first = self.fake_free_list_head.next_free_block
            self.fake_free_list_head.next_free_block = first
            first.prev_free_block = self.fake_free_list_head
            blocks[-1].next_free_block = old_first
            old_first.prev_free_block = blocks[-1]
            self.num_free_blocks += len(blocks)
            self.blocks.extend(rest or [])

    def make_legacy_queue():
        """0.23-seam-era FreeQueue: no head insertion at all."""

        class FreeQueue:
            def __init__(self):
                self.fake_free_list_head = types.SimpleNamespace(next_free_block=None)
                self.num_free_blocks = 0
                self.blocks: list = []

            append_n = FreeKVCacheBlockQueue.append_n

        return FreeQueue

    kcu.FreeKVCacheBlockQueue = FreeKVCacheBlockQueue
    host["FreeKVCacheBlockQueue"] = FreeKVCacheBlockQueue
    host["make_legacy_queue"] = make_legacy_queue

    # --- vllm.v1.core.block_pool ---------------------------------------
    bp = register_module("vllm.v1.core.block_pool")

    class BlockPool:
        def __init__(self):
            self.free_block_queue = FreeKVCacheBlockQueue()
            self.null_block = types.SimpleNamespace(is_null=True, ref_cnt=99)
            self.freed: list[list] = []

        def free_blocks(self, ordered_blocks):
            blocks_list = list(ordered_blocks)
            for block in blocks_list:
                block.ref_cnt -= 1
            freed = [b for b in blocks_list if b.ref_cnt == 0 and not b.is_null]
            if freed:
                self.free_block_queue.append_n(freed)
            self.freed.append([("append", b) for b in freed])

    bp.BlockPool = BlockPool
    host["BlockPool"] = BlockPool

    # --- vllm.v1.core.single_type_kv_cache_manager ---------------------
    stm = register_module("vllm.v1.core.single_type_kv_cache_manager")

    class FullAttentionManager:
        def __init__(self, kv_cache_spec, block_pool=None, **kwargs):
            self.block_size = kv_cache_spec.block_size
            self.block_pool = block_pool
            self.req_to_blocks: dict[str, list] = {}
            self._null_block = block_pool.null_block if block_pool is not None else None
            self.freed_requests: list[str] = []

        def free(self, request_id):
            self.freed_requests.append(request_id)

    def register_all_kvcache_specs(vllm_config=None):
        KVCacheSpecRegistry.register(
            FullAttentionSpec,
            FullAttentionManager,
            uniform_type_base_spec=FullAttentionSpec,
        )
        KVCacheSpecRegistry.register(
            MLAAttentionSpec,
            FullAttentionManager,
            uniform_type_base_spec=FullAttentionSpec,
        )

    stm.FullAttentionManager = FullAttentionManager
    stm.register_all_kvcache_specs = register_all_kvcache_specs
    host["FullAttentionManager"] = FullAttentionManager
    host["register_all_kvcache_specs"] = register_all_kvcache_specs

    # --- vllm.v1.engine.core --------------------------------------------
    # Real hosts bind register_all_kvcache_specs with a top-level
    # import here and call it at EngineCore construction (engine/core.py
    # line ~59/~266 on the verified builds) — a second binding site the
    # P3 patch must also cover.
    engine_core = register_module("vllm.v1.engine.core")
    engine_core.register_all_kvcache_specs = register_all_kvcache_specs
    host["engine_core_module"] = engine_core

    # --- vllm.v1.core.sched.scheduler -----------------------------------
    sched_mod = register_module("vllm.v1.core.sched.scheduler")

    class Scheduler:
        def update_from_output(self, scheduler_output, model_runner_output):
            return "host-result"

    sched_mod.Scheduler = Scheduler
    host["Scheduler"] = Scheduler

    # --- vllm.v1.worker.gpu_model_runner --------------------------------
    runner_mod = register_module("vllm.v1.worker.gpu_model_runner")

    class GPUModelRunner:
        def __init__(self, cache_config=None):
            self.cache_config = cache_config
            self.input_batch = None
            self.kv_cache_config = None
            self.positions = None

        def sample_tokens(self, grammar_output=None):
            result = types.SimpleNamespace(sampled_token_ids=[1])
            self.last_grammar_output = grammar_output
            return result

    runner_mod.GPUModelRunner = GPUModelRunner
    host["GPUModelRunner"] = GPUModelRunner

    # --- vllm.v1.attention.backends.registry ----------------------------
    attn_registry = register_module("vllm.v1.attention.backends.registry")

    class AttentionImpl:
        calls: list[dict] = []

        def forward(
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output=None,
            output_scale=None,
            output_block_scale=None,
        ):
            type(self).calls.append({"key": key})
            return "attn-output"

    class Backend:
        @staticmethod
        def get_impl_cls():
            return AttentionImpl

    class _BackendEnumEntry:
        def __init__(self, cls):
            self._cls = cls

        def get_class(self):
            return self._cls

    class AttentionBackendEnum:
        CUSTOM = _BackendEnumEntry(Backend)
        FLASH_ATTN = _BackendEnumEntry(Backend)

    attn_registry.AttentionBackendEnum = AttentionBackendEnum
    host["AttentionImpl"] = AttentionImpl

    # --- vllm.distributed (missing on purpose style: raising getter) ----
    distributed = register_module("vllm.distributed")

    def get_pp_group():
        raise RuntimeError("no distributed group in tests")

    distributed.get_pp_group = get_pp_group

    return host


@pytest.fixture
def clean_knorm_env(monkeypatch):
    for var in KNORM_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def is_host_module(name: str) -> bool:
    """Only the host tree — never the plugin (``vllm_hust_knorm``)."""
    return name == "vllm" or name.startswith("vllm.")


@pytest.fixture
def fake_host(monkeypatch, clean_knorm_env):
    """Fresh fake host modules + pristine plugin globals per test."""
    from vllm_hust_knorm.core import activation
    from vllm_hust_knorm.knorm.attention_backend import reset_for_tests
    from vllm_hust_knorm.knorm.manager import (
        get_knorm_manager_class,
    )
    from vllm_hust_knorm.knorm.manager import (
        reset_for_tests as reset_manager,
    )

    host = build_fake_host()

    # Ensure a plugin-loaded host is assumed.
    import vllm  # the fake top package

    assert vllm is not None

    yield host

    reset_for_tests()
    reset_manager()
    get_knorm_manager_class.cache_clear()
    activation.reset_for_tests()


@pytest.fixture(autouse=True)
def restore_sys_modules():
    """Drop fake host modules after each test."""
    yield
    for name in list(sys.modules):
        if is_host_module(name):
            del sys.modules[name]


def make_vllm_config(enable_prefix_caching: bool):
    return types.SimpleNamespace(
        cache_config=types.SimpleNamespace(enable_prefix_caching=enable_prefix_caching)
    )
