"""Bounded, per-head K/V/sidecar moves; donors never overlap holes."""


def move_tensor(keys, values, positions, packing):
    # Scalar addressing is deliberately a correctness candidate, not a tuned kernel.
    for move in packing.moves:
        source = (*move.source, move.head)
        target = (*move.target, move.head)
        keys[target].copy_(keys[source])
        values[target].copy_(values[source])
        positions[target].copy_(positions[source])
