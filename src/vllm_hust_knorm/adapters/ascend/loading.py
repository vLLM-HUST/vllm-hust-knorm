"""Explicit, process-local wrappers for the pinned host's existing boundaries."""

import importlib.abc
import importlib.machinery
import inspect
import sys
from contextlib import nullcontext
from functools import wraps
from pathlib import Path

from .guard import settings, validate

_finder = None
_verified_files = set()


def wrap(owner, name, callback, parameters):
    original = getattr(owner, name)
    if getattr(original, "_knorm_callback", None) is callback:
        return
    code = getattr(original, "__code__", None)
    if code is None or Path(code.co_filename).resolve() not in _verified_files:
        raise RuntimeError(f"unexpected effective host method: {owner}.{name}")
    if not set(parameters) <= inspect.signature(original).parameters.keys():
        raise RuntimeError(f"unsupported host signature: {owner}.{name}")

    @wraps(original)
    def wrapped(*args, **kwargs):
        return callback(original, *args, **kwargs)

    wrapped._knorm_callback = callback
    setattr(owner, name, wrapped)


def planner_groups(original, vllm_config, kv_cache_spec):
    if settings(vllm_config):
        from .hooks import groups

        return groups(vllm_config, kv_cache_spec)
    return original(vllm_config, kv_cache_spec)


def planner_config(original, vllm_config, kv_cache_groups, available_memory):
    if settings(vllm_config):
        from .hooks import cache_config

        return cache_config(vllm_config, kv_cache_groups, available_memory)
    return original(vllm_config, kv_cache_groups, available_memory)


def step(original, engine):
    from .hooks import before_schedule

    before_schedule(engine)
    try:
        return original(engine)
    finally:
        if hasattr(engine, "knorm_binding"):
            engine.knorm_binding.inflight = None


def abort_boundary(original, engine, *args, **kwargs):
    binding = getattr(engine, "knorm_binding", None)
    if binding is not None and binding.inflight is not None:
        binding.after_forward(binding.inflight)
        binding.inflight = None
    return original(engine, *args, **kwargs)


def free_blocks(original, scheduler, request):
    binding = getattr(scheduler, "knorm_binding", None)
    if binding is None or not binding.free(request):
        return original(scheduler, request)


def budget(original, worker, available_memory):
    if settings(worker.vllm_config):
        validate(worker.vllm_config)
        return available_memory
    return original(worker, available_memory)


def control(worker, operation, *args):
    if not settings(worker.vllm_config):
        raise RuntimeError("KNorm candidate control is disabled")
    return worker.model_runner.knorm_control.dispatch(operation, *args)


def initialize_cache(
    original,
    runner,
    kv_cache_config,
    kernel_block_sizes=None,
    kv_cache_allocation_context=None,
):
    if not settings(runner.vllm_config):
        return original(
            runner,
            kv_cache_config,
            kernel_block_sizes=kernel_block_sizes,
            kv_cache_allocation_context=kv_cache_allocation_context,
        )
    from .hooks import initialize_cache as allocate

    validate(runner.vllm_config)
    if (
        kernel_block_sizes is None
        or len(kernel_block_sizes) != len(kv_cache_config.kv_cache_groups)
        or any(size != 128 for size in kernel_block_sizes)
    ):
        raise RuntimeError("candidate requires kernel block size 128 for every group")
    from vllm import envs

    if (
        runner.vllm_config.cache_config.kv_cache_layout != "LBNHC"
        or envs.VLLM_KV_CACHE_LAYOUT != "LBNHC"
    ):
        raise RuntimeError("negotiated cache and attention layouts must be LBNHC")
    with kv_cache_allocation_context or nullcontext():
        return allocate(
            runner, kv_cache_config, settings(runner.vllm_config)["workspace_bytes"]
        )


def update_states(original, runner, scheduler_output):
    result = original(runner, scheduler_output)
    if hasattr(runner, "knorm_control"):
        runner.knorm_control.prepare_known_inputs()
    return result


def bind_builders(runner):
    """Wrap each real builder instance, after Ascend has created its groups."""
    from .metadata import physical_metadata

    for group, attention_groups in enumerate(runner.attn_groups):
        for attention_group in attention_groups:
            for builder in attention_group.metadata_builders:
                if getattr(builder, "_knorm_runner", None) is runner:
                    continue
                if hasattr(builder, "_knorm_runner"):
                    raise RuntimeError("metadata builder shared between runners")
                original = builder.build
                if "common_attn_metadata" not in inspect.signature(original).parameters:
                    raise RuntimeError("unsupported attention metadata builder")

                @wraps(original)
                def build(*args, _original=original, _group=group, **kwargs):
                    bound = inspect.signature(_original).bind(*args, **kwargs)
                    common = physical_metadata(
                        runner,
                        _group,
                        bound.arguments["common_attn_metadata"],
                        runner.knorm_control.states,
                    )
                    bound.arguments["common_attn_metadata"] = common
                    # The runner later exposes the persistent slot buffer to the
                    # forward context, so replace it together with the metadata.
                    table = runner.input_batch.block_table[_group]
                    table.slot_mapping.gpu[: len(common.slot_mapping)].copy_(
                        common.slot_mapping
                    )
                    return _original(*bound.args, **bound.kwargs)

                builder.build = build
                builder._knorm_runner = runner


def build_metadata(original, runner, *args, **kwargs):
    if hasattr(runner, "knorm_control"):
        bind_builders(runner)
    return original(runner, *args, **kwargs)


def patch_module(module):
    name = module.__name__
    if name.endswith("patch_kv_cache_utils"):
        # Ascend replaces planner globals on import. Wrap those effective
        # functions again, retaining its native path for disabled configs.
        patch_module(sys.modules["vllm.v1.core.kv_cache_utils"])
    elif name == "vllm.v1.core.kv_cache_utils":
        wrap(
            module,
            "get_kv_cache_groups",
            planner_groups,
            ("vllm_config", "kv_cache_spec"),
        )
        wrap(
            module,
            "get_kv_cache_config_from_groups",
            planner_config,
            ("vllm_config", "kv_cache_groups", "available_memory"),
        )
    elif name == "vllm.v1.engine.core":
        wrap(module.EngineCore, "step", step, ("self",))
        wrap(module.EngineCore, "_process_aborts_queue", abort_boundary, ("self",))
    elif name == "vllm.v1.core.sched.scheduler":
        wrap(module.Scheduler, "_free_request_blocks", free_blocks, ("self", "request"))
    elif name == "vllm_ascend.worker.worker":
        cls = module.NPUWorker
        wrap(
            cls,
            "_scale_kv_cache_memory_for_multi_group",
            budget,
            ("self", "available_memory"),
        )
        previous = getattr(cls, "knorm_control", None)
        if previous is not None and previous is not control:
            raise RuntimeError("worker control method collision")
        cls.knorm_control = control
    elif name == "vllm_ascend.worker.model_runner_v1":
        cls = module.NPUModelRunner
        wrap(
            cls,
            "initialize_kv_cache_tensors",
            initialize_cache,
            (
                "self",
                "kv_cache_config",
                "kernel_block_sizes",
                "kv_cache_allocation_context",
            ),
        )
        wrap(cls, "_update_states", update_states, ("self", "scheduler_output"))
        wrap(cls, "_build_attention_metadata", build_metadata, ("self",))


TARGETS = (
    "vllm.v1.core.kv_cache_utils",
    "vllm.v1.core.sched.scheduler",
    "vllm.v1.engine.core",
    "vllm_ascend.worker.model_runner_v1",
    "vllm_ascend.worker.worker",
    "vllm_ascend.patch.platform.patch_kv_cache_utils",
)


class Loader:
    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        patch_module(module)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"missing pinned host module {fullname}")
        spec.loader = Loader(spec.loader)
        return spec


def install():
    global _finder, _verified_files
    import json

    from .candidate import verify_sources

    roots = verify_sources()
    lock = json.loads(Path(__file__).with_name("source-lock.json").read_text())
    _verified_files = {
        (roots[relative.split("/")[0]] / relative).resolve() for relative in lock
    }
    if _finder is None:
        _finder = Finder()
        sys.meta_path.insert(0, _finder)
    for name in TARGETS:
        if name in sys.modules:
            patch_module(sys.modules[name])
