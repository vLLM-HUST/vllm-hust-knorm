from dataclasses import dataclass

from ..algorithm.packing import Packing
from .state import Event, Layout


@dataclass(frozen=True)
class LayerPlan:
    layer: int
    old: Layout
    new: Layout
    packing: Packing


@dataclass(frozen=True)
class Plan:
    instance: str
    transaction: str
    expected_version: int
    target_version: int
    layers: tuple[LayerPlan, ...]
    event: Event


@dataclass
class Transaction:
    plan: Plan
    status: str = "PLANNED"
