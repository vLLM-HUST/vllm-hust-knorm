"""New architecture tests; legacy tests are historical regressions only."""

import math
import random

import pytest

from vllm_hust_knorm.algorithm.packing import pack
from vllm_hust_knorm.algorithm.selection import select
from vllm_hust_knorm.config import KNormConfig
from vllm_hust_knorm.runtime.engine import Simulation
from vllm_hust_knorm.runtime.recovery import preempt, replay
from vllm_hust_knorm.runtime.state import Layout


def kv(layer, head, token, position):
    return ((float((position * (head + 3) + layer) % 13),), (float(token), float(head)))


def populated(mode="repeat", ratio="0.5", threshold=4, length=12):
    sim = Simulation(KNormConfig(mode, threshold, ratio), capacity=128, block_size=4)
    request = sim.create("a", length)
    sim.append(request, range(length), kv)
    return sim, request


def identities(sim, request):
    return {
        layer: [
            set(zip(map(tuple, k[h]), map(tuple, v[h]), p[h], strict=True))
            for h in range(sim.cache.heads)
        ]
        for layer, layout in request.layers.items()
        for k, v, p in [sim.cache.read(layout)]
    }


def test_exact_budget_and_configuration():
    assert KNormConfig("once", 1, "0.14").budget(100) == 14
    assert KNormConfig("repeat", 1, "0.5").budget(4097) == 2049
    for value in ("nan", "inf", "0", "1.1", "-0.1"):
        with pytest.raises(ValueError):
            KNormConfig("once", 1, value)
    with pytest.raises(ValueError):
        KNormConfig.from_mapping({"compression_ratio": 0.5})
    with pytest.raises(ValueError):
        KNormConfig("unknown", 1, ".5")


def test_selection_ties_nonfinite_and_independent_heads():
    keys = [[[2], [1], [-1]], [[0], [3], [1]]]
    assert select(keys, [[2, 1, 0], [2, 1, 0]], 2) == ((2, 1), (0, 2))
    with pytest.raises(ValueError, match="non-finite"):
        select([[[math.nan]]], [[0]], 1)


def test_random_packing_and_kv_pairing():
    rng = random.Random(217)
    for _ in range(400):
        size, length, heads = rng.randint(1, 8), rng.randint(1, 80), rng.randint(1, 5)
        keep = rng.randint(1, length)
        blocks = rng.sample(range(500), math.ceil(length / size))
        selected = [rng.sample(range(length), keep) for _ in range(heads)]
        plan = pack(blocks, length, size, selected)
        data = {
            (b, slot, h): (i, h, -i)
            for i in range(length)
            for b, slot in [(blocks[i // size], i % size)]
            for h in range(heads)
        }
        sources = {(m.source, m.head) for m in plan.moves}
        targets = {(m.target, m.head) for m in plan.moves}
        assert not sources & targets
        assert len(targets) == len(plan.moves)
        for move in plan.moves:
            data[(*move.target, move.head)] = data[(*move.source, move.head)]
        for head in range(heads):
            result = [data[plan.blocks[i // size], i % size, head] for i in range(keep)]
            assert set(result) == {(i, head, -i) for i in selected[head]}
        assert set(plan.blocks) | set(plan.released) == set(blocks)
        assert not set(plan.blocks) & set(plan.released)
        counts = [
            sum(i // size == b for h in selected for i in h) for b in range(len(blocks))
        ]
        chosen = sorted(range(len(blocks)), key=lambda b: (-counts[b], b))[
            : math.ceil(keep / size)
        ]
        assert set(plan.blocks) == {blocks[b] for b in chosen}


@pytest.mark.parametrize("fault", ["partial", "APPLIED", "COMMIT"])
def test_transaction_retry_and_recycled_donors(fault):
    sim, request = populated()
    original = identities(sim, request)
    plan = sim.transactions.plan(request, {2: 6, 3: 6})
    if fault == "partial":
        assert plan.layers[0].packing.moves
        sim.cache.fail_after = 1
    else:
        sim.worker.lose_ack = fault
    with pytest.raises(RuntimeError):
        sim.transactions.execute(request, plan)
    assert len(sim.pool.free) == 116  # No release before both consumers ACK.
    with pytest.raises(RuntimeError, match="cannot roll back"):
        sim.transactions.abort(request)
    sim.transactions.execute(request, plan)
    assert request.semantic_length == 12
    assert [r.length for r in request.layers.values()] == [12, 12, 6, 6]
    result = identities(sim, request)
    for layer in (2, 3):
        for head in range(2):
            assert result[layer][head] <= original[layer][head]
    other = sim.create("b", 4)
    sim.append(other, [100] * 4, kv)
    before = identities(sim, other)
    sim.transactions.execute(request, plan)
    assert identities(sim, other) == before
    assert len(request.events) == 1
    assert sim.worker.layouts[request.instance] == (request.version, request.layers)


def test_once_threshold_chunked_prefill_and_no_repeat_boundary():
    sim = Simulation(KNormConfig("once", 10, ".5"), block_size=4)
    r = sim.create("a", 8)
    sim.append(r, range(4), kv)
    sim.controller.before_decode(r)
    assert not r.once_checked
    sim.append(r, range(4, 8), kv)
    sim.controller.before_decode(r)
    assert r.once_checked and not r.events
    for i in range(8, 14):
        sim.append(r, [i], kv)
    assert not r.events
    sim, r = populated()
    sim.controller.before_decode(r)
    first = dict(r.layers)
    sim.controller.before_decode(r)
    assert r.layers == first
    sim.append(r, [12], kv)
    assert r.layers[2].length == 7 and r.semantic_length == 13
    sim.controller.before_decode(r)
    assert r.layers[2].length == 4


def test_deletion_without_page_reduction_and_noop():
    sim, r = populated(ratio=".75", length=4)
    pages = r.layers[2].pages
    sim.controller.before_decode(r)
    assert r.layers[2] == Layout(3, pages)
    sim, r = populated(ratio="1")
    sim.controller.before_decode(r)
    assert not r.events and r.once_checked


def test_cancel_and_aba_stale_plan():
    sim, r = populated()
    plan = sim.transactions.plan(r, {2: 6})
    old = r.layers[2].pages
    sim.transactions.abort(r)
    with pytest.raises(RuntimeError, match="aborted"):
        sim.transactions.execute(r, plan)
    sim.finish(r)
    assert len(sim.pool.free) == 128
    new = sim.create("a", 12)
    sim.append(new, range(12), kv)
    assert new.instance != r.instance
    with pytest.raises(RuntimeError, match="stale"):
        sim.pool.validate((new.instance, 2), old)
    with pytest.raises(RuntimeError, match="unknown"):
        sim.transactions.execute(r, plan)


def test_replay_preserves_choices_and_once_flags():
    sim, r = populated()
    for i in range(12, 16):
        sim.append(r, [i], kv)
    sim.controller.before_decode(r)
    expected = identities(sim, r)
    checkpoint = preempt(r, sim.transactions)
    assert not sim.pool.owners

    class Runner:
        def prefill(self, request, tokens):
            sim.append(request, tokens, kv)

        def decode_known(self, request, token):
            sim.append(request, [token], kv)

    replay(r, checkpoint, sim.transactions, Runner())
    assert identities(sim, r) == expected
    assert r.once_checked == checkpoint.once_checked
    assert r.checked_boundary == checkpoint.checked_boundary
    assert tuple(r.events) == checkpoint.events
    sim.controller.before_decode(r)
    assert tuple(r.events) == checkpoint.events


def test_pool_admission_is_atomic_and_busy_cancel_retains_pages():
    sim = Simulation(KNormConfig("repeat", 4, ".5"), capacity=3, block_size=4)
    r = sim.create("a", 4)
    with pytest.raises(MemoryError):
        sim.append(r, range(4), kv)
    assert len(sim.pool.free) == 3
    sim, r = populated()
    sim.worker.busy.add(r.instance)
    with pytest.raises(RuntimeError, match="in flight"):
        sim.finish(r)
    assert len(sim.pool.owners) == 12
    sim.worker.busy.clear()
    sim.finish(r)
    assert not sim.pool.owners


def test_new_activation_rejects_enable_and_legacy(fake_host, monkeypatch):
    from vllm_hust_knorm.core.activation import activate

    activate()
    monkeypatch.setenv("VLLM_KNORM_ENABLED", "1")
    with pytest.raises(RuntimeError, match="import_only"):
        activate()
    monkeypatch.setenv("VLLM_KNORM_ENABLED", "0")
    monkeypatch.setenv("VLLM_KNORM_COMPRESSION_RATIO", ".5")
    with pytest.raises(ValueError, match="legacy"):
        activate()


def test_tensor_candidate_matches_cpu_when_torch_available():
    torch = pytest.importorskip("torch")
    from vllm_hust_knorm.backends.ascend.selection import select_tensor

    keys = torch.tensor(
        [[[2.0], [1.0], [-1.0]], [[0.0], [3.0], [1.0]]], dtype=torch.float16
    )
    positions = torch.tensor([[2, 1, 0], [2, 1, 0]])
    assert select_tensor(keys, positions, 2).tolist() == [[2, 1], [0, 2]]


def test_compression_matches_reference_exactly():
    from vllm_hust_knorm.algorithm.reference import compress

    sim, r = populated()
    expected = {}
    for layer in (2, 3):
        keys, values, positions = sim.cache.read(r.layers[layer])
        expected[layer] = [set(head) for head in compress(keys, values, positions, 6)]
    sim.controller.before_decode(r)
    actual = identities(sim, r)
    for layer in expected:
        assert actual[layer] == expected[layer]


def test_replay_once_and_admission_failure_can_retry():
    sim, r = populated(mode="once")
    sim.append(r, [12], kv)
    checkpoint = preempt(r, sim.transactions)

    class FailingRunner:
        def prefill(self, request, tokens):
            raise MemoryError("replay admission")

    with pytest.raises(MemoryError):
        replay(r, checkpoint, sim.transactions, FailingRunner())
    assert r.phase == "preempted" and not sim.pool.owners

    class Runner:
        def prefill(self, request, tokens):
            sim.append(request, tokens, kv)

        def decode_known(self, request, token):
            sim.append(request, [token], kv)

    replay(r, checkpoint, sim.transactions, Runner())
    for token in range(13, 20):
        sim.append(r, [token], kv)
    assert len(r.events) == 1 and r.once_checked


def test_batch_metadata_uses_identity_and_separates_semantic_positions():
    from vllm_hust_knorm.backends.ascend.metadata import build_metadata

    sim, a = populated(ratio=".75")
    sim.controller.before_decode(a)  # N=9, S=12, capacity=12 in compressed layers
    b = sim.create("b", 7)
    sim.append(b, range(2), kv)
    query_counts = {a.instance: 1, b.instance: 2}
    metadata = build_metadata([b, a], 2, query_counts, 4)
    assert metadata.instances == (b.instance, a.instance)
    assert metadata.lengths == (4, 10)
    assert metadata.semantic_positions == ((2, 3), (12,))
    assert metadata.slot_mapping[1] == (a.layers[2].pages[2].number * 4 + 1,)
    reverse = build_metadata([a, b], 2, query_counts, 4)
    assert reverse.block_tables == metadata.block_tables[::-1]


def test_new_modules_import_without_heavy_dependencies():
    import subprocess
    import sys

    code = """
import sys
import vllm_hust_knorm.algorithm.reference
import vllm_hust_knorm.runtime.engine
import vllm_hust_knorm.backends.ascend.cache
import vllm_hust_knorm.backends.ascend.metadata
assert not {'torch', 'torch_npu', 'vllm'}.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_release_ack_loss_is_idempotent_after_page_reuse(monkeypatch):
    sim, r = populated()
    plan = sim.transactions.plan(r, {2: 6, 3: 6})
    release = sim.pool.release_once
    calls = []

    def lose_first(owner, pages, transaction):
        release(owner, pages, transaction)
        if not calls:
            calls.append(owner)
            raise RuntimeError("lost release ACK")

    monkeypatch.setattr(sim.pool, "release_once", lose_first)
    with pytest.raises(RuntimeError, match="release ACK"):
        sim.transactions.execute(r, plan)
    assert sim.transactions.records[plan.transaction].status == "ACKNOWLEDGED"
    other = sim.create("b", 4)
    sim.append(other, [99] * 4, kv)
    expected = identities(sim, other)
    sim.transactions.execute(r, plan)
    assert identities(sim, other) == expected
    for layer, layout in other.layers.items():
        sim.pool.validate((other.instance, layer), layout.pages)


def test_workspace_exhaustion_does_not_mark_boundary_or_modify_kv():
    sim, r = populated()
    original = identities(sim, r)
    sim.cache.workspace_limit = 1
    with pytest.raises(MemoryError, match="workspace"):
        sim.controller.before_decode(r)
    assert identities(sim, r) == original
    assert r.pending is None and not r.once_checked


def test_random_selection_matches_independent_integer_oracle():
    rng = random.Random(921)
    for _ in range(100):
        length, keep = 40, rng.randint(1, 40)
        keys = [
            [[rng.randint(-5, 5) for _ in range(8)] for _ in range(length)]
            for _ in range(4)
        ]
        positions = [rng.sample(range(1000), length) for _ in range(4)]
        expected = tuple(
            tuple(
                sorted(
                    range(length),
                    key=lambda i: (sum(x * x for x in keys[h][i]), positions[h][i]),
                )[:keep]
            )
            for h in range(4)
        )
        assert select(keys, positions, keep) == expected


def test_cancel_after_partial_apply_discards_all_worker_references():
    sim, r = populated()
    plan = sim.transactions.plan(r, {2: 6, 3: 6})
    sim.cache.fail_after = 1
    with pytest.raises(RuntimeError):
        sim.transactions.execute(r, plan)
    sim.finish(r)
    assert r.instance not in sim.worker.layouts
    assert not sim.pool.owners and not sim.transactions.records
    with pytest.raises(RuntimeError, match="unknown"):
        sim.transactions.execute(r, plan)


def test_scope_and_reserved_pages():
    from vllm_hust_knorm.core.capabilities import RuntimeScope
    from vllm_hust_knorm.runtime.allocator import PagePool

    RuntimeScope().validate()
    for field, value in (
        ("apc", True),
        ("world_size", 2),
        ("eager", False),
        ("synchronous", False),
        ("dtype", "bfloat16"),
    ):
        with pytest.raises(RuntimeError):
            RuntimeScope(**{field: value}).validate()
    pool = PagePool(4, reserved=(0,))
    assert {p.number for p in pool.allocate(("a", 2), 3)} == {1, 2, 3}
    with pytest.raises(MemoryError):
        pool.allocate(("b", 3), 1)


@pytest.mark.parametrize("mode", ["once", "repeat"])
@pytest.mark.parametrize("cancel", [False, True])
def test_before_apply_failure_preserves_check_until_success(mode, cancel, monkeypatch):
    sim, request = populated(mode=mode)
    original = identities(sim, request)
    quiesce = sim.worker.quiesce
    calls = 0

    def fail_before_apply(instance):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("control failure before APPLY")
        quiesce(instance)

    monkeypatch.setattr(sim.worker, "quiesce", fail_before_apply)
    with pytest.raises(RuntimeError, match="before APPLY"):
        sim.controller.before_decode(request)
    transaction = sim.transactions.records[request.pending]
    assert transaction.status == "PLANNED"
    assert not request.once_checked and request.checked_boundary is None
    assert identities(sim, request) == original
    if cancel:
        sim.transactions.abort(request)
        assert not request.once_checked and request.checked_boundary is None
    monkeypatch.setattr(sim.worker, "quiesce", quiesce)
    sim.controller.before_decode(request)
    assert request.layers[2].length == 6
    assert len(request.events) == 1
    assert request.once_checked and request.checked_boundary == 12
    sim.controller.before_decode(request)
    assert request.layers[2].length == 6 and len(request.events) == 1


def test_aborting_later_plan_preserves_previous_completed_check(monkeypatch):
    sim, request = populated()
    sim.controller.before_decode(request)
    sim.append(request, [12], kv)
    previous = (request.once_checked, request.checked_boundary)
    assert previous == (True, 12)
    original = identities(sim, request)
    quiesce = sim.worker.quiesce
    calls = 0

    def fail_before_apply(instance):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("control failure before APPLY")
        quiesce(instance)

    monkeypatch.setattr(sim.worker, "quiesce", fail_before_apply)
    with pytest.raises(RuntimeError, match="before APPLY"):
        sim.controller.before_decode(request)
    sim.transactions.abort(request)
    assert (request.once_checked, request.checked_boundary) == previous
    assert identities(sim, request) == original
    monkeypatch.setattr(sim.worker, "quiesce", quiesce)
    sim.controller.before_decode(request)
    assert request.layers[2].length == 4
    assert request.checked_boundary == 13 and len(request.events) == 2
    sim.controller.before_decode(request)
    assert len(request.events) == 2


@pytest.mark.parametrize("mode", ["once", "repeat"])
@pytest.mark.parametrize("fault", ["partial", "APPLIED", "COMMIT"])
def test_pending_apply_retry_consumes_check_once(mode, fault):
    sim, request = populated(mode=mode)
    if fault == "partial":
        sim.cache.fail_after = 1
    else:
        sim.worker.lose_ack = fault
    with pytest.raises(RuntimeError):
        sim.controller.before_decode(request)
    assert not request.once_checked and request.checked_boundary is None
    sim.controller.before_decode(request)
    assert request.once_checked and request.checked_boundary == 12
    assert request.layers[2].length == 6 and len(request.events) == 1
    sim.controller.before_decode(request)
    assert request.layers[2].length == 6 and len(request.events) == 1
