"""Deterministic original-page selection and disjoint donor/hole pairing."""

from dataclasses import dataclass

from .policy import page_count


@dataclass(frozen=True)
class Move:
    head: int
    source: tuple[int, int]
    target: tuple[int, int]


@dataclass(frozen=True)
class Packing:
    blocks: tuple[int, ...]
    released: tuple[int, ...]
    moves: tuple[Move, ...]
    keep: int


def pack(blocks, length, block_size, selected):
    blocks = tuple(blocks)
    if len(blocks) != page_count(length, block_size) or len(set(blocks)) != len(blocks):
        raise ValueError("source table must be compact and unique")
    if not selected:
        raise ValueError("no heads")
    keep = len(selected[0])
    for indices in selected:
        if len(indices) != keep or len(set(indices)) != keep:
            raise ValueError("unequal head budgets or duplicate selection")
        if any(i < 0 or i >= length for i in indices):
            raise ValueError("selection outside valid prefix")
    scores = [0] * len(blocks)
    outside = [0] * len(blocks)
    remainder = keep % block_size
    for indices in selected:
        for i in indices:
            scores[i // block_size] += 1
            outside[i // block_size] += i % block_size >= remainder
    chosen = sorted(
        sorted(range(len(blocks)), key=lambda b: (-scores[b], b))[
            : page_count(keep, block_size)
        ]
    )
    if remainder:
        last = min(chosen, key=lambda b: (outside[b], b))
        chosen.remove(last)
        chosen.append(last)
    target_blocks = tuple(blocks[b] for b in chosen)
    targets = [(target_blocks[i // block_size], i % block_size) for i in range(keep)]
    target_set = set(targets)
    moves = []
    for head, indices in enumerate(selected):
        sources = [(blocks[i // block_size], i % block_size) for i in sorted(indices)]
        source_set = set(sources)
        donors = [s for s in sources if s not in target_set]
        holes = [t for t in targets if t not in source_set]
        for source, target in zip(donors, holes, strict=True):
            moves.append(Move(head, source, target))
    return Packing(
        target_blocks,
        tuple(b for b in blocks if b not in target_blocks),
        tuple(moves),
        keep,
    )
