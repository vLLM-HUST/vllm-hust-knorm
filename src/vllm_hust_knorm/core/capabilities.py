"""Explicit first-version scope; eligibility is separate from runtime correctness."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeScope:
    model: str = "Qwen/Qwen2.5-14B-Instruct"
    device: str = "Ascend910B2"
    dtype: str = "float16"
    world_size: int = 1
    eager: bool = True
    synchronous: bool = True
    attention: str = "rope_full"
    apc: bool = False
    speculative: bool = False
    offload: bool = False
    multimodal: bool = False
    quantized_kv: bool = False

    def validate(self):
        if self != RuntimeScope():
            raise RuntimeError("configuration outside the KNorm first-version scope")
