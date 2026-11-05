"""Forward-only transactions. Records live until explicit request retirement."""

from uuid import uuid4

from ..algorithm.packing import pack
from .protocol import LayerPlan, Plan, Transaction
from .state import Event, Layout


class Transactions:
    def __init__(self, pool, cache, worker):
        self.pool, self.cache, self.worker = pool, cache, worker
        self.records = {}

    def plan(self, request, targets, recorded=None):
        if request.pending or request.phase not in {"decode", "replay"}:
            raise RuntimeError("request cannot enter PLAN")
        self.worker.quiesce(request.instance)
        self.cache.synchronize()
        layers, selections = [], []
        for layer, keep in sorted(targets.items()):
            old = request.layers[layer]
            self.pool.validate((request.instance, layer), old.pages)
            if not 0 < keep < old.length or layer < 2:
                raise ValueError("invalid compression target")
            indices, positions = self.cache.choose(
                old, keep, None if recorded is None else recorded[layer]
            )
            packing = pack(
                tuple(p.number for p in old.pages),
                old.length,
                self.cache.block_size,
                indices,
            )
            by_number = {p.number: p for p in old.pages}
            new = Layout(keep, tuple(by_number[b] for b in packing.blocks))
            layers.append(LayerPlan(layer, old, new, packing))
            selections.append(
                (
                    layer,
                    tuple(
                        tuple(positions[h][i] for i in chosen)
                        for h, chosen in enumerate(indices)
                    ),
                )
            )
        if not layers:
            raise ValueError("empty transaction")
        plan = Plan(
            request.instance,
            uuid4().hex,
            request.version,
            request.version + 1,
            tuple(layers),
            Event(request.semantic_length, tuple(selections)),
        )
        transaction = Transaction(plan)
        self.records[plan.transaction] = transaction
        request.pending = plan.transaction
        request.phase = "transaction"
        return plan

    def execute(self, request, plan):
        transaction = self.records.get(plan.transaction)
        if (
            transaction is None
            or transaction.plan != plan
            or plan.instance != request.instance
        ):
            raise RuntimeError("unknown or conflicting transaction")
        if transaction.status == "COMMITTED":
            return transaction.status  # Never touch recycled donor pages again.
        if transaction.status == "ABORTED":
            raise RuntimeError("aborted transaction")
        if request.pending != plan.transaction:
            raise RuntimeError("transaction does not own request")
        if request.version not in {plan.expected_version, plan.target_version}:
            raise RuntimeError("stale layout version")
        for layer in plan.layers:
            if transaction.status != "ACKNOWLEDGED":
                self.pool.validate((request.instance, layer.layer), layer.old.pages)
            expected = (
                layer.new if request.version == plan.target_version else layer.old
            )
            if request.layers[layer.layer] != expected:
                raise RuntimeError("scheduler layout mismatch")
        self.worker.quiesce(request.instance)
        if transaction.status in {"PLANNED", "APPLYING"}:
            begin = getattr(self.cache, "begin", None)
            if begin is not None:
                begin(request, plan)
            transaction.status = "APPLYING"  # From here rollback is forbidden.
            for layer in plan.layers:
                self.cache.move(layer.packing)
            self.cache.synchronize()
            layouts = dict(request.layers)
            layouts.update({p.layer: p.new for p in plan.layers})
            self.worker.install(request, plan.target_version, layouts)
            transaction.status = "APPLIED"
        if transaction.status == "APPLIED":
            request.layers.update({p.layer: p.new for p in plan.layers})
            request.version = plan.target_version
            self.worker.commit(request.instance, plan.target_version)
            transaction.status = "ACKNOWLEDGED"
        # Device and both metadata consumers have acknowledged the new version.
        for layer in plan.layers:
            retained = set(layer.new.pages)
            self.pool.release_once(
                (request.instance, layer.layer),
                tuple(p for p in layer.old.pages if p not in retained),
                plan.transaction,
            )
        request.events.append(plan.event)
        # Consume the check only on completion, including pending retries.
        request.once_checked = True
        request.checked_boundary = plan.event.boundary
        transaction.status = "COMMITTED"
        request.pending = None
        request.phase = "decode"
        return transaction.status

    def abort(self, request):
        transaction = self.records[request.pending]
        if transaction.status != "PLANNED":
            raise RuntimeError("cannot roll back after APPLY; retry or retire request")
        transaction.status = "ABORTED"
        request.pending = None
        request.phase = "decode"

    def retire(self, request, phase="finished"):
        # Host must stop future forwards before calling; synchronization may fail,
        # in which case pages deliberately remain owned and cannot be reused.
        self.worker.quiesce(request.instance)
        self.cache.synchronize()
        self.worker.forget(request.instance)
        self.pool.release_request(request.instance)
        self.pool.forget_releases(request.instance)
        request.layers.clear()
        request.pending = None
        request.phase = phase
        self.records = {
            k: t for k, t in self.records.items() if t.plan.instance != request.instance
        }
