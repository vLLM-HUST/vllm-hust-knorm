"""Contracts against the pinned host classes; only device execution is excluded."""

import os
from types import SimpleNamespace

import pytest

if os.environ.get("KNORM_RUN_HOST_TESTS") != "1":
    pytest.skip("requires the pinned patched host environment", allow_module_level=True)

import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.request import Request as HostRequest
from vllm.v1.worker.block_table import MultiGroupBlockTable
from vllm.v1.worker.gpu_input_batch import CachedRequestState

from vllm_hust_knorm.adapters.ascend.allocator import HostAllocator
from vllm_hust_knorm.adapters.ascend.cache import (
    PAGE_BYTES,
    SIDECAR_BYTES,
    allocate_shared_cache,
    groups_from_specs,
    shared_cache_config,
)
from vllm_hust_knorm.adapters.ascend.candidate import verify_sources
from vllm_hust_knorm.adapters.ascend.metadata import (
    physical_metadata,
    replace_block_tables,
)
from vllm_hust_knorm.adapters.ascend.scheduler import bind_layer_allocations
from vllm_hust_knorm.adapters.ascend.worker import (
    RemoteCache,
    RemoteWorker,
    WorkerControl,
)
from vllm_hust_knorm.config import KNormConfig
from vllm_hust_knorm.runtime.controller import Controller
from vllm_hust_knorm.runtime.state import Layout, Request
from vllm_hust_knorm.runtime.transaction import Transactions


@pytest.fixture(autouse=True)
def pageable_host_buffers(monkeypatch):
    # Real buffers and copy methods remain in use; pinning requires a device.
    monkeypatch.setattr("vllm.v1.utils.PIN_MEMORY", False)
    monkeypatch.setattr("vllm.v1.worker.gpu_input_batch.PIN_MEMORY", False)


def host_cache(blocks=160):
    specs = {
        f"model.layers.{i}.self_attn.attn": FullAttentionSpec(
            block_size=128, num_kv_heads=8, head_size=128, dtype=torch.float16
        )
        for i in range(48)
    }
    groups = groups_from_specs(specs)
    reserve = 1024 * 1024
    config = shared_cache_config(
        groups, blocks * (PAGE_BYTES + SIDECAR_BYTES) + reserve, reserve
    )
    manager = KVCacheManager(
        config,
        max_model_len=512,
        scheduler_block_size=128,
        hash_block_size=128,
        enable_caching=False,
    )
    allocator = HostAllocator(manager.coordinator)
    return config, manager, allocator


def request_pair(length=256, req_id="request"):
    host = HostRequest(
        request_id=req_id,
        prompt_token_ids=list(range(length)),
        sampling_params=SamplingParams(max_tokens=16),
        pooling_params=None,
    )
    state = Request(req_id, length, semantic_length=length, phase="decode")
    state.tokens = list(range(length))
    return host, state


def test_fixed_host_source_contract():
    assert set(verify_sources()) == {"vllm", "vllm_ascend"}


def test_shared_backing_has_one_capacity_for_48_groups():
    config, manager, _ = host_cache()
    backing, sidecar, aliases = allocate_shared_cache(config, "cpu")
    assert len(config.kv_cache_groups) == len(aliases) == 48
    assert config.num_blocks == 160
    assert backing.numel() * backing.element_size() == 160 * PAGE_BYTES
    assert sidecar.numel() * sidecar.element_size() == 160 * SIDECAR_BYTES
    assert len({k.data_ptr() for k, v in aliases.values()}) == 1
    assert len({v.data_ptr() for k, v in aliases.values()}) == 1
    assert all(k.is_contiguous() and v.is_contiguous() for k, v in aliases.values())
    assert manager.block_pool.get_num_free_blocks() == 159
    assert all(
        m.block_pool is manager.block_pool
        for m in manager.coordinator.single_type_managers
    )


def test_real_allocation_uses_layer_lengths_and_reuses_released_pages():
    _, manager, allocator = host_cache()
    host, state = request_pair()
    states = {state.request_id: state}
    bind_layer_allocations(manager.coordinator, states)
    assert manager.allocate_slots(host, 256) is not None
    allocator.adopt(state, dict.fromkeys(range(48), 256))
    host.num_computed_tokens = 256
    old = state.layers[2].pages
    state.layers[2] = Layout(127, old[:1])
    allocator.install(state)
    allocator.release_once((state.instance, 2), old[1:], "compression")
    manager.allocate_slots(host, 1)
    assert len(manager.get_block_ids(host.request_id)[2]) == 1
    assert len(manager.get_block_ids(host.request_id)[0]) == 3
    allocator.adopt(
        state, {layer: layout.length for layer, layout in state.layers.items()}
    )
    allocator.release_request(state.instance)
    assert manager.block_pool.get_num_free_blocks() == 159
    other, replacement = request_pair(req_id="other")
    manager.allocate_slots(other, 256)
    allocator.adopt(replacement, dict.fromkeys(range(48), 256))
    with pytest.raises(RuntimeError, match="stale"):
        allocator.validate((replacement.instance, 2), old)


def runner_for(state, config, max_reqs=2):
    table = MultiGroupBlockTable(
        max_reqs, 512, False, torch.device("cpu"), [128] * 48, [128] * 48, [4] * 48
    )
    blocks = tuple([p.number for p in state.layers[layer].pages] for layer in range(48))
    table.add_row(blocks, 0)
    from vllm.v1.worker.gpu_input_batch import InputBatch

    batch = InputBatch(
        max_reqs, 512, 128, torch.device("cpu"), 2048, [128] * 48, [128] * 48, [4] * 48
    )
    cached = CachedRequestState(
        state.request_id,
        list(state.tokens),
        [],
        SamplingParams(max_tokens=16),
        None,
        blocks,
        state.semantic_length,
        [],
    )
    batch.add_request(cached)
    batch.refresh_metadata()
    return SimpleNamespace(input_batch=batch, requests={state.request_id: cached})


@pytest.mark.parametrize(
    "lose_ack", [None, "move", "install", "commit", "install_receipt", "commit_receipt"]
)
def test_real_pool_transaction_and_complete_worker_table_replacement(lose_ack):
    config, manager, allocator = host_cache()
    host, state = request_pair()
    manager.allocate_slots(host, 256)
    allocator.adopt(state, dict.fromkeys(range(48), 256))
    runner = runner_for(state, config)
    backing, positions, _ = allocate_shared_cache(config, "cpu")
    # Keys prefer the first half; no per-head aggregation enters selection.
    for layout in state.layers.values():
        for index in range(256):
            page = layout.pages[index // 128].number
            backing[0, page, index % 128].fill_(float(index))
            backing[1, page, index % 128].fill_(float(index))
            positions[page, index % 128].fill_(index)
    control = WorkerControl(runner, backing, positions, 1024 * 1024, lambda: None)
    control.register(state)
    failed = False
    order = []

    def rpc(operation, *args):
        nonlocal failed
        result = control.dispatch(operation, *args)
        order.append(operation)
        if lose_ack == f"{operation}_receipt" and not failed:
            failed = True
            return {**result, "instance": "stale-instance"}
        if operation == lose_ack and not failed:
            failed = True
            raise RuntimeError("lost control ACK")
        return result

    transactions = Transactions(
        allocator, RemoteCache(rpc), RemoteWorker(rpc, allocator)
    )
    controller = Controller(KNormConfig("once", 256, "0.5"), transactions)
    before = manager.block_pool.get_num_free_blocks()
    if lose_ack:
        with pytest.raises(RuntimeError, match="lost control ACK|acknowledgement"):
            controller.before_decode(state)
        assert manager.block_pool.get_num_free_blocks() == before
    controller.before_decode(state)
    assert state.layers[0].length == 256 and state.layers[2].length == 128
    assert manager.block_pool.get_num_free_blocks() == before + 46
    assert "commit" in order
    assert len(manager.get_block_ids(state.request_id)[2]) == 1
    table = runner.input_batch.block_table[2]
    assert table.num_blocks_per_row[0] == 1
    assert table.get_cpu_tensor()[0, 1:].count_nonzero() == 0
    assert table.get_device_tensor(1)[0, 1:].count_nonzero() == 0
    plan = next(iter(transactions.records.values())).plan
    for layer in plan.layers:
        rpc("move", plan.transaction, layer.packing)
    assert len(state.events) == 1
    from copy import deepcopy

    old_state = deepcopy(state)
    transactions.retire(state)
    assert manager.block_pool.get_num_free_blocks() == 159
    with pytest.raises(RuntimeError, match="retired"):
        control.register(old_state)
    with pytest.raises(RuntimeError, match="retired"):
        rpc("move", plan.transaction, plan.layers[0].packing)


def test_real_block_table_metadata_retains_semantic_positions():
    config, manager, allocator = host_cache()
    host, state = request_pair()
    manager.allocate_slots(host, 256)
    allocator.adopt(state, dict.fromkeys(range(48), 256))
    runner = runner_for(state, config)
    state.layers[2] = Layout(100, state.layers[2].pages[:1])
    replace_block_tables(runner, state, state.layers)
    table = runner.input_batch.block_table[2]
    common = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([257], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=1,
        max_query_len=1,
        max_seq_len=257,
        block_table_tensor=table.get_device_tensor(1),
        slot_mapping=torch.tensor([0]),
        positions=torch.tensor([256]),
        is_prefilling=torch.tensor([False]),
    )
    common.num_input_tokens = 1
    converted = physical_metadata(runner, 2, common, {state.request_id: state})
    assert converted.seq_lens.tolist() == [101]
    assert converted.seq_lens_cpu.tolist() == [101]
    assert converted.positions.tolist() == [256]
    assert not converted.is_prefilling.item()
    assert converted.slot_mapping.item() == state.layers[2].pages[0].number * 128 + 100


def scheduler_config():
    from transformers import Qwen2Config
    from vllm.config import CacheConfig, DeviceConfig, SchedulerConfig, VllmConfig

    config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        cache_config=CacheConfig(block_size=128, enable_prefix_caching=False),
        scheduler_config=SchedulerConfig(
            is_encoder_decoder=False,
            max_model_len=512,
            max_num_batched_tokens=128,
            max_num_seqs=2,
            enable_chunked_prefill=True,
            async_scheduling=False,
        ),
    )
    # Scheduler-only model description: no model registry or model process starts.
    config.model_config = SimpleNamespace(
        hf_text_config=Qwen2Config(
            num_hidden_layers=48,
            num_key_value_heads=8,
            num_attention_heads=40,
            hidden_size=5120,
        ),
        dtype=torch.float16,
        enforce_eager=True,
        quantization=None,
        uses_mrope=False,
        is_encoder_decoder=False,
        is_diffusion=False,
        is_multimodal_model=False,
        max_model_len=512,
        enable_sleep_mode=False,
        enable_return_routed_experts=False,
        return_sampling_mask=False,
        runner_type="generate",
        enable_trace_replay=False,
        skip_tokenizer_init=True,
        multimodal_config=None,
        original_max_model_len=512,
    )
    config.cache_config.num_gpu_blocks = 256
    return config


def test_real_scheduler_constructs_and_chunks_prefill():
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.structured_output import StructuredOutputManager

    config = scheduler_config()
    cache, _, _ = host_cache(256)
    scheduler = Scheduler(config, cache, StructuredOutputManager(config), 128)
    host, _ = request_pair()
    scheduler.add_request(host)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {host.request_id: 128}
    assert host.num_computed_tokens == 128
    assert len(scheduler.kv_cache_manager.get_block_ids(host.request_id)) == 48


def test_planner_wrappers_are_idempotent_and_disabled_path_is_native(monkeypatch):
    import vllm.v1.core.kv_cache_utils as planner

    from vllm_hust_knorm.adapters.ascend.candidate import configure
    from vllm_hust_knorm.adapters.ascend.loading import install

    config = scheduler_config()
    specs = {
        f"model.layers.{i}.self_attn.attn": FullAttentionSpec(
            block_size=128, num_kv_heads=8, head_size=128, dtype=torch.float16
        )
        for i in range(48)
    }
    original = planner.get_kv_cache_groups
    disabled = original(config, specs)
    install()
    installed = planner.get_kv_cache_groups
    install()
    assert planner.get_kv_cache_groups is installed
    assert planner.get_kv_cache_groups(config, specs) == disabled
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "LBNHC")
    monkeypatch.setenv("VLLM_KNORM_CANDIDATE", "1")
    configure(config, KNormConfig("repeat", 256, "0.5"), 4 * 1024 * 1024)
    groups = planner.get_kv_cache_groups(config, specs)
    assert len(groups) == 48
    result = planner.get_kv_cache_config_from_groups(config, groups, 256 * PAGE_BYTES)
    assert len(result.kv_cache_tensors) == 1
    full = planner.get_kv_cache_configs(config, [specs], [256 * PAGE_BYTES])
    assert full == [result]
    scheduler_cache = planner.generate_scheduler_kv_cache_config(full)
    assert len(scheduler_cache.kv_cache_groups) == 48
    with pytest.raises(MemoryError, match="reserved workspace"):
        planner.get_kv_cache_configs(config, [specs], [193 * PAGE_BYTES])
    config.cache_config.enable_prefix_caching = True
    with pytest.raises(RuntimeError, match="scope"):
        planner.get_kv_cache_groups(config, specs)


def test_real_input_batch_reorders_physical_metadata():
    from vllm.v1.worker.gpu_input_batch import InputBatch

    config, manager, allocator = host_cache(256)
    host, first = request_pair(req_id="first")
    manager.allocate_slots(host, 256)
    allocator.adopt(first, dict.fromkeys(range(48), 256))
    host2, second = request_pair(length=128, req_id="second")
    second.phase = "prefill"
    manager.allocate_slots(host2, 128)
    allocator.adopt(second, dict.fromkeys(range(48), 64))
    batch = InputBatch(
        2, 512, 128, torch.device("cpu"), 2048, [128] * 48, [128] * 48, [4] * 48
    )
    runner = SimpleNamespace(input_batch=batch, requests={})
    for state in (first, second):
        cached = CachedRequestState(
            state.request_id,
            list(range(state.prompt_length)),
            [],
            SamplingParams(max_tokens=16),
            None,
            tuple([p.number for p in layout.pages] for layout in state.layers.values()),
            state.semantic_length if state is first else 64,
            [],
        )
        runner.requests[state.request_id] = cached
        batch.add_request(cached)
    first.layers[2] = Layout(100, first.layers[2].pages[:1])
    replace_block_tables(runner, first, first.layers)
    batch.swap_states(0, 1)
    starts = torch.tensor([0, 32, 33], dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=starts,
        query_start_loc_cpu=starts,
        seq_lens=torch.tensor([96, 257], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=33,
        max_query_len=32,
        max_seq_len=257,
        block_table_tensor=batch.block_table[2].get_device_tensor(2),
        slot_mapping=torch.zeros(33, dtype=torch.int64),
        positions=torch.tensor(list(range(64, 96)) + [256]),
        is_prefilling=torch.tensor([True, False]),
    )
    common.num_input_tokens = 33
    physical = physical_metadata(
        runner, 2, common, {s.request_id: s for s in (first, second)}
    )
    assert physical.seq_lens.tolist() == [96, 101]
    assert physical.positions is common.positions
    assert physical.is_prefilling is common.is_prefilling
    assert physical.slot_mapping[-1] == first.layers[2].pages[0].number * 128 + 100
    assert physical.slot_mapping[0] == second.layers[2].pages[0].number * 128 + 64


class ForwardBoundary:
    """CPU executor boundary; Scheduler and EngineCore.step remain real."""

    def __init__(self, engine, cache):
        from vllm.v1.worker.gpu_input_batch import InputBatch

        self.engine = engine
        self.input_batch = InputBatch(
            2, 512, 128, torch.device("cpu"), 2048, [128] * 48, [128] * 48, [4] * 48
        )
        self.requests = {}
        self.execute_model_state = None
        self.inputs = []
        self.operations = []
        self.samples = 0
        backing, positions, _ = allocate_shared_cache(cache, "cpu")
        self.knorm_control = WorkerControl(
            self, backing, positions, 4 * 1024 * 1024, lambda: None
        )

    def collective_rpc(self, method, args):
        assert method == "knorm_control"
        self.operations.append(args[0])
        return [self.knorm_control.dispatch(*args)]

    def execute_model(self, output, non_block=False):
        from concurrent.futures import Future

        self.output = output
        self.operations.append("forward")
        for item in output.scheduled_new_reqs:
            self.requests[item.req_id] = CachedRequestState(
                item.req_id,
                item.prompt_token_ids,
                [],
                item.sampling_params,
                torch.Generator().manual_seed(7),
                item.block_ids,
                item.num_computed_tokens,
                [],
            )
        for row, req_id in enumerate(output.scheduled_cached_reqs.req_ids):
            cached = self.requests[req_id]
            blocks = output.scheduled_cached_reqs.new_block_ids[row]
            if blocks is not None:
                for old, new in zip(cached.block_ids, blocks, strict=True):
                    old.extend(new)
            cached.num_computed_tokens = (
                output.scheduled_cached_reqs.num_computed_tokens[row]
            )
        for req_id in list(self.input_batch.req_ids):
            self.input_batch.remove_request(req_id)
        for req_id in output.num_scheduled_tokens:
            cached = self.requests[req_id]
            self.input_batch.add_request(cached)
        self.input_batch.refresh_metadata()
        self.knorm_control.prepare_known_inputs()
        for req_id, count in output.num_scheduled_tokens.items():
            state = self.knorm_control.states[req_id]
            row = self.input_batch.req_id_to_index[req_id]
            tokens = self.input_batch.token_ids_cpu[
                row, state.semantic_length : state.semantic_length + count
            ].tolist()
            self.inputs.append((req_id, state.semantic_length, tokens))
            for layout in state.layers.values():
                for offset, token in enumerate(tokens):
                    index = layout.length + offset
                    page, slot = layout.pages[index // 128].number, index % 128
                    self.knorm_control.cache.keys[page, slot].fill_(
                        state.semantic_length + offset
                    )
                    self.knorm_control.cache.values[page, slot].fill_(token)
        self.execute_model_state = object()
        if non_block:
            result = Future()
            result.set_result(None)
            return result
        return None

    def sample_tokens(self, grammar):
        from vllm.v1.outputs import ModelRunnerOutput

        self.samples += 1
        ids, tokens = [], []
        for req_id in self.output.num_scheduled_tokens:
            host = self.engine.scheduler.requests[req_id]
            ids.append(req_id)
            sampled = (
                [1000 + self.samples]
                if host.num_computed_tokens >= host.num_prompt_tokens
                else []
            )
            tokens.append(sampled)
            self.requests[req_id].output_token_ids.extend(sampled)
        self.execute_model_state = None
        return ModelRunnerOutput(
            ids, dict(zip(ids, range(len(ids)), strict=True)), tokens
        )


def engine_with_cpu_forward(monkeypatch, mode):
    from queue import Queue

    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.structured_output import StructuredOutputManager

    from vllm_hust_knorm.adapters.ascend.candidate import configure

    config = scheduler_config()
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "LBNHC")
    monkeypatch.setenv("VLLM_KNORM_CANDIDATE", "1")
    configure(config, KNormConfig(mode, 256, "0.5"), 4 * 1024 * 1024)
    cache, _, _ = host_cache(256)
    engine = EngineCore.__new__(EngineCore)
    engine.vllm_config = config
    engine.log_stats = False
    engine.engines_running = False
    engine.batch_queue = None
    engine.aborts_queue = Queue()
    engine.scheduler = Scheduler(config, cache, StructuredOutputManager(config), 128)
    engine.model_executor = ForwardBoundary(engine, cache)
    return engine


def step_if_work(engine):
    from vllm.v1.engine.core import EngineCoreProc

    assert EngineCoreProc.has_work(engine)
    return engine.step()


@pytest.mark.parametrize("mode", ["once", "repeat"])
def test_real_engine_control_precedes_schedule_and_replays_known_tokens(
    monkeypatch, mode
):
    import time

    engine = engine_with_cpu_forward(monkeypatch, mode)
    host, _ = request_pair()
    engine.scheduler.add_request(host)
    engine.step()
    state = engine.knorm_binding.states[host.request_id]
    assert state.phase == "prefill" and state.semantic_length == 128
    assert not state.events
    engine.step()
    assert state.semantic_length == 256 and not state.events
    engine.step()
    assert state.semantic_length == 257 and len(state.events) == 1
    assert state.layers[0].length == 257 and state.layers[2].length == 129
    device = engine.model_executor
    assert device.operations.index("commit") < len(device.operations) - 1
    known, delivered = list(state.tokens), list(host.output_token_ids)
    pending = host.all_token_ids[-1]
    generator = device.requests[host.request_id].generator
    rng = generator.get_state().clone()
    engine.scheduler.running.remove(host)
    engine.scheduler._preempt_request(host, time.monotonic())
    assert not state.layers
    assert engine.scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 255
    replay_start, sample_count = len(device.inputs), device.samples
    assert engine.scheduler.get_num_unfinished_requests() == 1
    step_if_work(engine)
    replayed = [
        token for _, _, tokens in device.inputs[replay_start:-1] for token in tokens
    ]
    assert replayed == known
    assert device.samples == sample_count + 1
    assert list(host.output_token_ids[:-1]) == delivered
    assert host.all_token_ids[-2] == pending
    assert device.requests[host.request_id].output_token_ids[-2] == pending
    assert torch.equal(generator.get_state(), rng)
    assert device.requests[host.request_id].generator is generator
    assert state.layers[0].length == 258 and state.layers[2].length == 130
    assert len(state.events) == 1
    assert device.inputs[-1][2] == [pending]
    engine.aborts_queue.put([host.request_id])
    engine.step()
    assert host.request_id not in engine.scheduler.requests
    assert host.request_id not in engine.knorm_binding.states
    assert engine.scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 255


def test_real_engine_mixes_chunked_prefill_and_compressed_decode(monkeypatch):
    engine = engine_with_cpu_forward(monkeypatch, "repeat")
    engine.vllm_config.additional_config["knorm_candidate"]["policy"]["prune_after"] = (
        128
    )
    first, _ = request_pair(length=128, req_id="first")
    second, _ = request_pair(length=256, req_id="second")
    engine.scheduler.add_request(first)
    engine.scheduler.add_request(second)
    engine.step()
    engine.step()
    states = engine.knorm_binding.states
    assert states["first"].semantic_length == 129
    assert states["first"].layers[2].length == 65
    assert states["second"].phase == "prefill"
    assert not states["second"].events
    for _ in range(3):
        engine.step()
    assert states["second"].phase == "decode"
    assert len(states["second"].events) == 1
    assert states["first"].layers[0].length == states["first"].semantic_length
    assert states["second"].layers[0].length == states["second"].semantic_length


def test_repeat_history_and_cancel_before_first_forward(monkeypatch):
    import time

    engine = engine_with_cpu_forward(monkeypatch, "repeat")
    engine.vllm_config.additional_config["knorm_candidate"]["policy"]["prune_after"] = (
        128
    )
    host, _ = request_pair()
    engine.scheduler.add_request(host)
    for _ in range(4):
        engine.step()
    state = engine.knorm_binding.states[host.request_id]
    assert [event.boundary for event in state.events] == [256, 257]
    previous = tuple(state.events)
    engine.scheduler.running.remove(host)
    engine.scheduler._preempt_request(host, time.monotonic())
    step_if_work(engine)
    assert tuple(state.events) == previous
    # A queued request can be retired before a worker row ever exists.
    waiting, _ = request_pair(req_id="waiting")
    engine.scheduler.add_request(waiting)
    engine.knorm_binding.before_schedule()
    from vllm.v1.request import RequestStatus

    engine.scheduler.finish_requests(
        [waiting.request_id], RequestStatus.FINISHED_ABORTED
    )
    assert "waiting" not in engine.knorm_binding.states


@pytest.mark.parametrize("partial_apply", [False, True])
def test_host_retirement_cancels_plan_and_rejects_reused_request_messages(
    partial_apply,
):
    from copy import deepcopy

    config, manager, allocator = host_cache()
    host, state = request_pair()
    manager.allocate_slots(host, 256)
    allocator.adopt(state, dict.fromkeys(range(48), 256))
    runner = runner_for(state, config)
    backing, positions, _ = allocate_shared_cache(config, "cpu")
    backing.zero_()
    for layout in state.layers.values():
        for index in range(256):
            positions[layout.pages[index // 128].number, index % 128].fill_(index)
    control = WorkerControl(runner, backing, positions, 4 * 1024 * 1024, lambda: None)
    control.register(state)
    seen = 0

    def rpc(operation, *args):
        nonlocal seen
        result = control.dispatch(operation, *args)
        if operation == "move":
            seen += 1
            if partial_apply and seen == 1:
                raise RuntimeError("interrupted movement")
        return result

    transactions = Transactions(
        allocator, RemoteCache(rpc), RemoteWorker(rpc, allocator)
    )
    plan = transactions.plan(state, {2: 128, 3: 128})
    stale = deepcopy(state)
    before = manager.block_pool.get_num_free_blocks()
    if partial_apply:
        with pytest.raises(RuntimeError, match="interrupted movement"):
            transactions.execute(state, plan)
    else:
        transactions.abort(state)
    assert manager.block_pool.get_num_free_blocks() == before
    transactions.retire(state)
    assert manager.block_pool.get_num_free_blocks() == 159
    replacement_host, replacement = request_pair(req_id=host.request_id)
    manager.allocate_slots(replacement_host, 256)
    allocator.adopt(replacement, dict.fromkeys(range(48), 256))
    control.register(replacement)
    with pytest.raises(RuntimeError, match="retired"):
        control.register(stale)
    with pytest.raises(RuntimeError, match="stale"):
        control.dispatch("begin", stale, plan)
    assert control.states[host.request_id].instance == replacement.instance


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("occupied_pages", [200, 230])
def test_replay_admission_failure_remains_recoverable_or_cancellable(
    monkeypatch, cancel, occupied_pages
):
    import time

    from vllm.v1.request import RequestStatus

    engine = engine_with_cpu_forward(monkeypatch, "once")
    host, _ = request_pair()
    engine.scheduler.add_request(host)
    engine.step()
    engine.step()
    engine.scheduler.running.remove(host)
    engine.scheduler._preempt_request(host, time.monotonic())
    pool = engine.scheduler.kv_cache_manager.block_pool
    occupied = pool.get_new_blocks(occupied_pages)
    step_if_work(engine)
    assert host.request_id in engine.knorm_binding.checkpoints
    assert pool.get_num_free_blocks() == 255 - occupied_pages
    if cancel:
        engine.scheduler.finish_requests(
            [host.request_id], RequestStatus.FINISHED_ABORTED
        )
        assert host.request_id not in engine.knorm_binding.states
        assert host.request_id not in engine.model_executor.knorm_control.sampling
    pool.free_blocks(occupied)
    if not cancel:
        step_if_work(engine)
        assert host.request_id not in engine.knorm_binding.checkpoints
        assert engine.knorm_binding.states[host.request_id].semantic_length == 257
    else:
        assert engine.scheduler.get_num_unfinished_requests() == 0
        step_if_work(engine)  # Drain the native finished-request notification.
        from vllm.v1.engine.core import EngineCoreProc

        assert not EngineCoreProc.has_work(engine)


@pytest.mark.parametrize("enabled", [False, True])
def test_pinned_runner_initialization_call_and_forwarding(monkeypatch, enabled):
    import ast
    from contextlib import contextmanager
    from copy import deepcopy

    from vllm_hust_knorm.adapters.ascend import hooks, loading
    from vllm_hust_knorm.adapters.ascend.candidate import configure

    roots = verify_sources()
    source = roots["vllm_ascend"] / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(source.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "initialize_kv_cache_tensors"
    )
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "initialize_kv_cache_tensors"
    )
    # Execute the pinned call expression and signature without importing the
    # device module. Only the original allocation body is replaced by a recorder.
    original = deepcopy(method)
    original.decorator_list = []
    original.returns = None
    for arg in original.args.args:
        arg.annotation = None
    original.body = ast.parse(
        "return (self, kv_cache_config, kernel_block_sizes, "
        "kv_cache_allocation_context)"
    ).body
    namespace = {}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[original], type_ignores=[])),
            str(source),
            "exec",
        ),
        namespace,
    )
    cls = type("AllocationBoundary", (), {method.name: namespace[method.name]})
    config = scheduler_config()
    if enabled:
        config.cache_config.kv_cache_layout = "LBNHC"
        monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "LBNHC")
        monkeypatch.setenv("VLLM_KNORM_CANDIDATE", "1")
        configure(config, KNormConfig("once", 256, "0.5"), 1024 * 1024)
    loading.install()
    loading.wrap(
        cls,
        method.name,
        loading.initialize_cache,
        tuple(a.arg for a in original.args.args),
    )
    runner = cls()
    runner.vllm_config = config
    cache, _, _ = host_cache()
    entered = []

    @contextmanager
    def allocation_context():
        entered.append("enter")
        yield
        entered.append("exit")

    result = object()

    def allocate(actual_runner, actual_cache, workspace):
        assert actual_runner is runner and actual_cache is cache
        assert workspace == 1024 * 1024 and entered == ["enter"]
        return result

    monkeypatch.setattr(hooks, "initialize_cache", allocate)
    context = allocation_context()
    sizes = [128] * 48
    arguments = dict(
        self=runner,
        kv_cache_config=cache,
        kernel_block_sizes=sizes,
        kv_cache_allocation_context=context,
    )
    expression = compile(ast.Expression(call), str(source), "eval")
    returned = eval(expression, arguments)
    if enabled:
        assert returned is result and entered == ["enter", "exit"]
        for invalid in (None, [], [64] * 48, [128] * 47):
            arguments["kernel_block_sizes"] = invalid
            with pytest.raises(RuntimeError, match="kernel block size 128"):
                eval(expression, arguments)
    else:
        assert returned == (runner, cache, sizes, context)
        assert returned[2] is sizes and returned[3] is context
        assert not entered
        assert runner.initialize_kv_cache_tensors(cache, sizes, context) == returned
        assert runner.initialize_kv_cache_tensors(cache) == (runner, cache, None, None)


@pytest.mark.parametrize("removed_row", [0, 1])
@pytest.mark.parametrize("swapped", [False, True])
@pytest.mark.parametrize("lost_ack", [None, "install", "commit"])
def test_transaction_receipts_cover_holes_swaps_and_page_reuse(
    removed_row, swapped, lost_ack
):
    config, manager, allocator = host_cache(384)
    states = []
    for req_id in ("first", "middle", "target"):
        host, state = request_pair(req_id=req_id)
        manager.allocate_slots(host, 256)
        allocator.adopt(state, dict.fromkeys(range(48), 256))
        states.append(state)
    runner = runner_for(states[0], config, max_reqs=3)
    batch = runner.input_batch
    for state in (states[2], states[1]) if swapped else states[1:]:
        cached = CachedRequestState(
            state.request_id,
            list(state.tokens),
            [],
            SamplingParams(max_tokens=16),
            torch.Generator().manual_seed(7),
            tuple([p.number for p in state.layers[layer].pages] for layer in range(48)),
            state.semantic_length,
            [],
        )
        runner.requests[state.request_id] = cached
        batch.add_request(cached)
    if swapped:
        batch.swap_states(1, 2)
    batch.refresh_metadata()
    batch.block_table.commit_block_table(3)
    backing, positions, _ = allocate_shared_cache(config, "cpu")
    control = WorkerControl(runner, backing, positions, 4 * 1024 * 1024, lambda: None)
    for state in states:
        control.register(state)
    removed = next(
        state
        for state in states
        if batch.req_id_to_index[state.request_id] == removed_row
    )
    control.dispatch("forget", removed.instance)
    allocator.release_request(removed.instance)
    target = states[2]
    assert batch.req_id_to_index["target"] == 2 and batch.num_reqs == 2
    indices = dict(batch.req_id_to_index)
    generators = dict(batch.generators)
    for layout in target.layers.values():
        for index in range(256):
            page, slot = layout.pages[index // 128].number, index % 128
            backing[0, page, slot].fill_(2 if index < 128 else 1)
            backing[1, page, slot].fill_(index)
            positions[page, slot].fill_(index)
    expected = {
        layer: [p.number for p in (layout.pages if layer < 2 else layout.pages[1:])]
        for layer, layout in target.layers.items()
    }
    released = {target.layers[layer].pages[0].number for layer in range(2, 48)}
    before = manager.block_pool.get_num_free_blocks()
    receipts = []
    failed = False

    def assert_tables():
        assert batch.req_id_to_index == indices and batch.generators == generators
        for group, blocks in expected.items():
            table = batch.block_table[group]
            cpu = table.get_cpu_tensor()[2]
            device = table.get_device_tensor(3)[2]
            assert cpu.tolist() == blocks + [0] * (len(cpu) - len(blocks))
            assert torch.equal(cpu, device)

    def rpc(operation, *args):
        nonlocal failed
        result = control.dispatch(operation, *args)
        if operation in ("install", "commit"):
            # The receipt is observed before RemoteWorker can release any page.
            assert_tables()
            assert manager.block_pool.get_num_free_blocks() == before
            receipts.append(result["status"])
            if operation == lost_ack and not failed:
                failed = True
                raise RuntimeError("lost control ACK")
        return result

    transactions = Transactions(
        allocator, RemoteCache(rpc), RemoteWorker(rpc, allocator)
    )
    controller = Controller(KNormConfig("once", 256, "0.5"), transactions)
    if lost_ack:
        with pytest.raises(RuntimeError, match="lost control ACK"):
            controller.before_decode(target)
        assert manager.block_pool.get_num_free_blocks() == before
    controller.before_decode(target)
    assert "APPLIED" in receipts and "COMMITTED" in receipts
    assert manager.block_pool.get_num_free_blocks() == before + 46
    reused = manager.block_pool.get_new_blocks(before + 46)
    assert released <= {block.block_id for block in reused}
    assert_tables()
    manager.block_pool.free_blocks(reused)


@pytest.mark.parametrize("pause", ["PAUSED_ALL", "PAUSED_NEW"])
def test_checkpoint_work_detection_and_running_admission(monkeypatch, pause):
    import time

    from vllm.v1.core.sched.interface import PauseState
    from vllm.v1.engine.core import EngineCoreProc
    from vllm.v1.request import RequestStatus

    engine = engine_with_cpu_forward(monkeypatch, "once")
    scheduler = engine.scheduler
    first, _ = request_pair(length=128, req_id="first")
    second, _ = request_pair(length=128, req_id="second")
    scheduler.add_request(first)
    scheduler.add_request(second)
    for _ in range(3):
        step_if_work(engine)
    for host in (first, second):
        scheduler.running.remove(host)
        scheduler._preempt_request(host, time.monotonic())
    assert len(engine.knorm_binding.checkpoints) == 2
    assert not scheduler.waiting and not scheduler.running
    assert scheduler.get_num_unfinished_requests() == 2
    scheduler.set_pause_state(getattr(PauseState, pause))
    assert not EngineCoreProc.has_work(engine)
    scheduler.set_pause_state(PauseState.UNPAUSED)
    scheduler.max_num_running_reqs = 1
    step_if_work(engine)
    assert scheduler.running == [first]
    assert list(engine.knorm_binding.checkpoints) == [second.request_id]
    assert scheduler.get_num_unfinished_requests() == 2
    step_if_work(engine)
    assert scheduler.running == [first]
    assert second.request_id in engine.knorm_binding.checkpoints
    scheduler.finish_requests([first.request_id], RequestStatus.FINISHED_ABORTED)
    step_if_work(engine)
    assert scheduler.running == [second]
    assert not engine.knorm_binding.checkpoints
    assert scheduler.get_num_unfinished_requests() == 1
    scheduler.finish_requests([second.request_id], RequestStatus.FINISHED_ABORTED)
    step_if_work(engine)
    assert scheduler.get_num_unfinished_requests() == 0
    assert not EngineCoreProc.has_work(engine)
