"""Simple materialized reference independent of the movement executor."""

from .selection import select


def compress(keys, values, positions, keep):
    chosen = select(keys, positions, keep)
    return tuple(
        tuple((keys[h][i], values[h][i], positions[h][i]) for i in indices)
        for h, indices in enumerate(chosen)
    )
