"""Explicit configuration for the token-level implementation (no legacy mapping)."""

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class KNormConfig:
    mode: str
    prune_after: int
    keep_ratio: Fraction | str | float

    def __post_init__(self):
        if self.mode not in {"once", "repeat"}:
            raise ValueError("mode must be once or repeat")
        if type(self.prune_after) is not int or self.prune_after < 1:
            raise ValueError("prune_after must be a positive integer")
        try:
            ratio = Fraction(str(self.keep_ratio))
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError("keep_ratio must be finite and in (0, 1]") from exc
        if not 0 < ratio <= 1:
            raise ValueError("keep_ratio must be in (0, 1]")
        object.__setattr__(self, "keep_ratio", ratio)

    @classmethod
    def from_mapping(cls, values: Mapping):
        expected = {"mode", "prune_after", "keep_ratio"}
        if set(values) != expected:
            raise ValueError(
                f"explicit {sorted(expected)} required; legacy/unknown keys rejected"
            )
        return cls(**values)

    def budget(self, length: int) -> int:
        if length < 0:
            raise ValueError("negative length")
        ratio = self.keep_ratio
        return (length * ratio.numerator + ratio.denominator - 1) // ratio.denominator


def validate_environment():
    """Single gate for entry points, checked even after earlier registration."""
    import os

    legacy = {
        "VLLM_KNORM_COMPRESSION_RATIO",
        "VLLM_KNORM_WARMUP_TOKENS",
        "VLLM_KNORM_SCORE_AGGREGATION",
        "VLLM_KNORM_NORM_REDUCE_OP",
    }
    present = legacy.intersection(os.environ)
    if present:
        raise ValueError(f"legacy KNorm configuration rejected: {sorted(present)}")
    enabled = os.environ.get("VLLM_KNORM_ENABLED", "0")
    if enabled not in {"0", "1"}:
        raise ValueError("VLLM_KNORM_ENABLED must be 0 or 1")
    if enabled == "1":
        from .adapters.vllm_hust.engine import require_verified_binding

        require_verified_binding()
