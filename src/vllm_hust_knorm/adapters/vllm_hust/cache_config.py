"""Host-neutral per-layer group description; physical pages share one pool."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CacheGroup:
    layer: int
    block_size: int


def cache_groups(layers, block_size=128):
    if layers < 1 or block_size < 1:
        raise ValueError("invalid cache geometry")
    return tuple(CacheGroup(layer, block_size) for layer in range(layers))
