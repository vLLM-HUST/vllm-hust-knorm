"""Small host extension callbacks; disabled configurations retain native paths."""

from .guard import settings, validate


def before_schedule(engine):
    if settings(engine.vllm_config):
        from .engine import EngineBinding

        if not hasattr(engine, "knorm_binding"):
            engine.knorm_binding = EngineBinding(engine)
        engine.knorm_binding.before_schedule()


def scheduled(engine, output):
    if hasattr(engine, "knorm_binding"):
        engine.knorm_binding.scheduled(output)


def after_forward(engine, output):
    if hasattr(engine, "knorm_binding"):
        engine.knorm_binding.after_forward(output)


def initialize_cache(runner, config, workspace_bytes):
    import torch
    from vllm.v1.worker.utils import bind_kv_cache

    from .cache import allocate_shared_cache
    from .worker import WorkerControl

    backing, positions, aliases = allocate_shared_cache(config, runner.device)

    def synchronize():
        torch.npu.synchronize(runner.device)

    runner.knorm_control = WorkerControl(
        runner, backing, positions, workspace_bytes, synchronize
    )
    runner.knorm_backing = backing
    bind_kv_cache(
        aliases, runner.compilation_config.static_forward_context, runner.kv_caches, 1
    )
    return aliases


def groups(config, specs):
    from .cache import groups_from_specs

    validate(config)
    return groups_from_specs(specs)


def cache_config(config, layer_groups, memory):
    from .cache import shared_cache_config

    validate(config)
    result = shared_cache_config(
        layer_groups, memory, settings(config)["workspace_bytes"]
    )
    required = 48 * ((config.model_config.max_model_len + 127) // 128) + 1
    if result.num_blocks < required:
        raise MemoryError(
            "reserved workspace and sidecar leave insufficient full-prefill capacity"
        )
    return result
