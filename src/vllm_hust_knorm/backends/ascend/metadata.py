"""Identity-indexed physical metadata, independent of batch row permutations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerMetadata:
    instances: tuple[str, ...]
    block_tables: tuple[tuple[int, ...], ...]
    lengths: tuple[int, ...]
    slot_mapping: tuple[tuple[int, ...], ...]
    semantic_positions: tuple[tuple[int, ...], ...]
    max_length: int


def build_metadata(requests, layer, query_counts, block_size):
    tables, lengths, slots, positions = [], [], [], []
    identities = tuple(r.instance for r in requests)
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate batch request")
    for request in requests:
        if request.pending or request.phase not in {"prefill", "decode", "replay"}:
            raise RuntimeError("layout is not forward-ready")
        q = query_counts[request.instance]
        if q < 1 or (request.phase == "decode" and q != 1):
            raise ValueError("invalid query count")
        layout = request.layers[layer]
        if len(layout.pages) * block_size < layout.length + q:
            raise RuntimeError(
                "append pages must be reserved before metadata construction"
            )
        table = tuple(p.number for p in layout.pages)
        tables.append(table)
        lengths.append(layout.length + q)
        slots.append(
            tuple(
                table[i // block_size] * block_size + i % block_size
                for i in range(layout.length, layout.length + q)
            )
        )
        positions.append(
            tuple(range(request.semantic_length, request.semantic_length + q))
        )
    return LayerMetadata(
        identities,
        tuple(tables),
        tuple(lengths),
        tuple(slots),
        tuple(positions),
        max(lengths, default=0),
    )


def device_metadata(metadata, device):
    import torch

    # A host binding must translate these into its exact attention metadata type.
    return {
        "lengths_cpu": torch.tensor(metadata.lengths, dtype=torch.int32),
        "lengths_device": torch.tensor(
            metadata.lengths, dtype=torch.int32, device=device
        ),
        "slot_mapping": tuple(
            torch.tensor(row, dtype=torch.int64, device=device)
            for row in metadata.slot_mapping
        ),
        "block_tables": tuple(
            torch.tensor(row, dtype=torch.int32, device=device)
            for row in metadata.block_tables
        ),
        "max_length": metadata.max_length,
    }
