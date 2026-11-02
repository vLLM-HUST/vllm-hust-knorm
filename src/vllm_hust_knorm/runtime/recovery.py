"""Replay committed choices at original semantic boundaries, without resampling."""

from dataclasses import dataclass

from .state import Event


@dataclass(frozen=True)
class Checkpoint:
    tokens: tuple[int, ...]
    semantic_length: int
    events: tuple[Event, ...]
    once_checked: bool
    checked_boundary: int | None
    phase: str


def preempt(request, transactions):
    if request.pending:
        transactions.execute(request, transactions.records[request.pending].plan)
    checkpoint = Checkpoint(
        tuple(request.tokens),
        request.semantic_length,
        tuple(request.events),
        request.once_checked,
        request.checked_boundary,
        request.phase,
    )
    if len(checkpoint.tokens) != checkpoint.semantic_length:
        raise RuntimeError("complete known input-token history required for replay")
    transactions.retire(request, "preempted")
    return checkpoint


def replay(request, checkpoint, transactions, runner):
    if request.phase != "preempted" or request.layers:
        raise RuntimeError("replay requires a detached request")
    if any(event.policy_version != "knorm-fp32-v1" for event in checkpoint.events):
        raise RuntimeError("unsupported replay policy version")
    request.semantic_length = 0
    request.tokens.clear()
    request.events.clear()
    request.once_checked = False
    request.checked_boundary = None
    request.phase = "replay"
    events = {e.boundary: e for e in checkpoint.events}
    if len(events) != len(checkpoint.events):
        raise RuntimeError("duplicate replay boundary")
    try:
        prompt = min(request.prompt_length, checkpoint.semantic_length)
        runner.prefill(request, checkpoint.tokens[:prompt])
        for boundary in range(prompt, checkpoint.semantic_length + 1):
            if request.semantic_length != boundary:
                raise RuntimeError("replay runner advanced incorrect semantic position")
            if boundary in events:
                request.phase = "replay"
                choices = dict(events[boundary].selections)
                plan = transactions.plan(
                    request, {layer: len(h[0]) for layer, h in choices.items()}, choices
                )
                transactions.execute(request, plan)
            if boundary < checkpoint.semantic_length:
                request.phase = "replay"
                runner.decode_known(request, checkpoint.tokens[boundary])
        if tuple(request.tokens) != checkpoint.tokens:
            raise RuntimeError("replay runner changed known token history")
        request.events = list(checkpoint.events)
        request.once_checked = checkpoint.once_checked
        request.checked_boundary = checkpoint.checked_boundary
        request.phase = checkpoint.phase
    except Exception:
        # No automatic retry loop: caller must explicitly admit a later retry.
        transactions.retire(request, "preempted")
        raise
