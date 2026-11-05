"""Explicit candidate configuration, separate from public activation eligibility."""

from ...config import KNormConfig

CORE_COMMIT = "ced6857afa0ea7b2e3f0846a62e1394e90f15607"
ASCEND_COMMIT = "c09d3e20b9c45c80b34ae3781440d90904a44023"


def settings(config):
    return config.additional_config.get("knorm_candidate")


def validate(config):
    import torch

    from .candidate import verify_sources

    verify_sources()
    options = settings(config)
    if not options:
        raise RuntimeError("explicit candidate configuration required")
    if (
        options["core_commit"] != CORE_COMMIT
        or options["ascend_commit"] != ASCEND_COMMIT
    ):
        raise RuntimeError("unsupported host source identities")
    policy = KNormConfig.from_mapping(options["policy"])
    model = config.model_config
    hf = model.hf_text_config
    parallel = config.parallel_config
    if (
        hf.model_type != "qwen2"
        or hf.num_hidden_layers != 48
        or hf.num_key_value_heads != 8
        or hf.hidden_size != 5120
        or hf.num_attention_heads != 40
        or model.dtype != torch.float16
        or not model.enforce_eager
        or model.quantization is not None
        or config.use_v2_model_runner
        or config.scheduler_config.async_scheduling
        or parallel.tensor_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
        or config.cache_config.enable_prefix_caching
        or config.speculative_config is not None
        or config.kv_transfer_config is not None
        or config.cache_config.kv_offloading_size is not None
        or config.cache_config.kv_sharing_fast_prefill
        or config.cache_config.block_size != 128
        or parallel.num_ubatches > 1
        or config.cache_config.num_gpu_blocks_override is not None
        or config.attention_config.hisparse_config is not None
        or getattr(model, "original_max_model_len", None) == -1
    ):
        raise RuntimeError("host configuration outside the candidate binding scope")
    if (
        parallel.decode_context_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
        or getattr(model, "is_multimodal_model", False)
        or model.enable_sleep_mode
        or getattr(config, "lora_config", None) is not None
    ):
        raise RuntimeError("unsupported parallelism, model or cache lifecycle")
    if config.cache_config.kv_cache_layout not in (None, "LBNHC"):
        raise RuntimeError("candidate binding requires LBNHC layout")
    if config.cache_config.cache_dtype not in {"auto", "float16"}:
        raise RuntimeError("FP16 KV required")
    if options["workspace_bytes"] <= 0:
        raise ValueError("explicit positive workspace budget required")
    return policy
