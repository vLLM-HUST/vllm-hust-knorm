"""One physical backing and one host allocation authority for all layers."""

import re

BLOCK_SIZE = 128
HEADS = 8
DIMENSION = 128
LAYERS = 48
PAGE_BYTES = 2 * BLOCK_SIZE * HEADS * DIMENSION * 2
SIDECAR_BYTES = BLOCK_SIZE * HEADS * 8


def groups_from_specs(specs):
    import torch
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

    indexed = {}
    for name, spec in specs.items():
        match = re.fullmatch(r"model\.layers\.(\d+)\.self_attn\.attn", name)
        if match is None:
            raise ValueError(f"unsupported attention layer name: {name}")
        layer = int(match[1])
        if layer in indexed or type(spec) is not FullAttentionSpec:
            raise ValueError("one ordinary full-attention spec per layer required")
        if spec.sliding_window or spec.attention_chunk_size or spec.non_causal:
            raise ValueError("ordinary causal full attention required")
        if (
            spec.block_size,
            spec.num_kv_heads,
            spec.head_size,
            spec.dtype,
            spec.page_size_bytes,
        ) != (BLOCK_SIZE, HEADS, DIMENSION, torch.float16, PAGE_BYTES):
            raise ValueError("unsupported cache geometry or quantization")
        indexed[layer] = KVCacheGroupSpec(layer_names=[name], kv_cache_spec=spec)
    if set(indexed) != set(range(LAYERS)):
        raise ValueError("Qwen2.5-14B requires exactly 48 attention layers")
    return [indexed[layer] for layer in range(LAYERS)]


def shared_cache_config(groups, available_memory, workspace_bytes):
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor

    if workspace_bytes <= 0 or len(groups) != LAYERS:
        raise ValueError("explicit workspace reservation and 48 groups required")
    names = [group.layer_names[0] for group in groups]
    blocks = (available_memory - workspace_bytes) // (PAGE_BYTES + SIDECAR_BYTES)
    if blocks < LAYERS + 1:
        raise MemoryError("cache cannot hold one page per layer plus the null page")
    return KVCacheConfig(
        num_blocks=blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=blocks * PAGE_BYTES,
                layers=names,
                layer_stride=0,
                block_stride=PAGE_BYTES,
            )
        ],
        kv_cache_groups=groups,
        kv_cache_layout="LBNHC",
    )


def allocate_shared_cache(config, device):
    import torch

    if config.kv_cache_layout != "LBNHC":
        raise ValueError("only contiguous token-major LBNHC is supported")
    backing = torch.empty(
        (2, config.num_blocks, BLOCK_SIZE, HEADS, DIMENSION),
        dtype=torch.float16,
        device=device,
    )
    backing[:, 0].zero_()
    positions = torch.full(
        (config.num_blocks, BLOCK_SIZE, HEADS), -1, dtype=torch.int64, device=device
    )
    aliases = {
        group.layer_names[0]: (backing[0], backing[1])
        for group in config.kv_cache_groups
    }
    return backing, positions, aliases
