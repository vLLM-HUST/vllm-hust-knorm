"""CPU oracle: FP32 square and sequential FP32 accumulation, position ties."""

import math
import struct


def fp32(value):
    try:
        return struct.unpack("f", struct.pack("f", value))[0]
    except OverflowError as exc:
        raise ValueError("non-finite K score") from exc


def score(vector):
    total = 0.0
    for value in vector:
        value = fp32(value)
        total = fp32(total + fp32(value * value))
    if not math.isfinite(total):
        raise ValueError("non-finite K score")
    return total


def select(keys, positions, keep):
    """Inputs [head][token][dim], [head][token]; return logical indices."""
    if not keys or len(keys) != len(positions):
        raise ValueError("head shape mismatch")
    length = len(keys[0])
    if not 0 <= keep <= length:
        raise ValueError("invalid keep count")
    result = []
    for vectors, pos in zip(keys, positions, strict=True):
        if len(vectors) != length or len(pos) != length or len(set(pos)) != length:
            raise ValueError("length mismatch or duplicate original positions")
        scores = [score(v) for v in vectors]
        result.append(
            tuple(sorted(range(length), key=lambda i: (scores[i], pos[i]))[:keep])
        )
    return tuple(result)
