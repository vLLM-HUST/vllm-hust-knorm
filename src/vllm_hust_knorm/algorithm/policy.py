"""Pure policy; checking a boundary is a runtime responsibility."""

from vllm_hust_knorm.config import KNormConfig


def target_length(config: KNormConfig, layer: int, length: int) -> int:
    if layer < 2 or length < config.prune_after:
        return length
    return config.budget(length)


def page_count(length: int, block_size: int) -> int:
    if length < 0 or block_size <= 0:
        raise ValueError("invalid length or block size")
    return (length + block_size - 1) // block_size
