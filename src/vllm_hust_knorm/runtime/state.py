from dataclasses import dataclass, field
from uuid import uuid4


@dataclass(frozen=True)
class Page:
    number: int
    generation: int


@dataclass(frozen=True)
class Layout:
    length: int
    pages: tuple[Page, ...]


@dataclass(frozen=True)
class Event:
    boundary: int
    # Original token positions per layer and head; independent of physical IDs.
    selections: tuple[tuple[int, tuple[tuple[int, ...], ...]], ...]
    policy_version: str = "knorm-fp32-v1"


@dataclass
class Request:
    request_id: str
    prompt_length: int
    instance: str = field(default_factory=lambda: uuid4().hex)
    semantic_length: int = 0
    layers: dict[int, Layout] = field(default_factory=dict)
    version: int = 0
    once_checked: bool = False
    checked_boundary: int | None = None
    phase: str = "prefill"
    events: list[Event] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    pending: str | None = None
