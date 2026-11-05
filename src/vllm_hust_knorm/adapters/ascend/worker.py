"""Synchronous control RPC consumer for the Ascend runner's persistent state."""

from copy import deepcopy

from ...backends.ascend.cache import AscendCache
from .metadata import replace_block_tables


class WorkerControl:
    def __init__(self, runner, backing, positions, workspace_bytes, synchronize):
        self.runner = runner
        self.cache = AscendCache(
            backing[0],
            backing[1],
            positions,
            workspace_bytes,
            cpu_validation=backing.device.type == "cpu",
        )
        self.synchronize = synchronize
        self.states = {}
        self.instances = {}
        self.committed = {}
        self.sampling = {}
        self.replay_input = None
        self.transactions = {}
        self.retired = {}

    def register(self, request):
        if request.instance in self.retired:
            if (
                request.phase != "replay"
                or request.request_id not in self.sampling
                or request.version <= self.retired[request.instance]
            ):
                raise RuntimeError("retired worker request instance")
            del self.retired[request.instance]
        previous = self.instances.get(request.request_id)
        if previous is not None and previous != request.instance:
            raise RuntimeError("request ID reused before old worker state retired")
        current = self.states.get(request.request_id)
        if current is not None and (
            current.version > request.version
            or current.semantic_length > request.semantic_length
        ):
            raise RuntimeError("stale worker registration")
        self.instances[request.request_id] = request.instance
        self.states[request.request_id] = deepcopy(request)

    def dispatch(self, operation, *args):
        self.synchronize()
        receipt = "ACK"
        if operation == "choose":
            layout = args[0]
            if not any(
                layout in state.layers.values() for state in self.states.values()
            ):
                raise RuntimeError("unknown source layout or stale page generations")
            return self.cache.choose(*args)
        if operation == "begin":
            request, plan = args
            existing = self.transactions.get(plan.transaction)
            if existing is not None:
                if existing[0] != plan:
                    raise RuntimeError("conflicting worker plan")
                return "ACK"
            current = self.states.get(request.request_id)
            if (
                current is None
                or current.instance != plan.instance
                or current.version != plan.expected_version
                or any(current.layers[p.layer] != p.old for p in plan.layers)
            ):
                raise RuntimeError("stale worker plan or page generation")
            self.transactions[plan.transaction] = (plan, "APPLYING")
        elif operation == "move":
            transaction, packing = args
            if transaction not in self.transactions:
                raise RuntimeError("unknown or retired worker transaction")
            plan, status = self.transactions[transaction]
            if status == "COMMITTED":
                return "ACK"
            if packing not in [p.packing for p in plan.layers]:
                raise RuntimeError("movement does not belong to the fixed plan")
            self.cache.move(packing)
        elif operation == "synchronize":
            pass
        elif operation == "register":
            self.register(*args)
        elif operation == "advance":
            request, count = args
            state = self.states[request.request_id]
            if (
                state.instance != request.instance
                or state.version != request.version
                or state.semantic_length != request.semantic_length
                or state.layers != request.layers
            ):
                raise RuntimeError("stale forward completion")
            for layout in state.layers.values():
                for offset in range(count):
                    index = layout.length + offset
                    page = layout.pages[index // 128].number
                    self.cache.positions[page, index % 128].fill_(
                        state.semantic_length + offset
                    )
        elif operation == "discard_sampling":
            (req_id,) = args
            self.sampling.pop(req_id, None)
        elif operation == "checkpoint":
            (instance,) = args
            state = next(s for s in self.states.values() if s.instance == instance)
            host = self.runner.requests[state.request_id]
            self.sampling[state.request_id] = host.generator
        elif operation == "replay_forward":
            req_id, known, output = args
            self.replay_input = (req_id, known)
            try:
                result = self.runner.execute_model(output)
                if result is not None:
                    raise RuntimeError(
                        "unexpected replay output; sampling is forbidden"
                    )
            finally:
                self.runner.execute_model_state = None
                self.replay_input = None
        elif operation == "restore_sampling":
            req_id, known = args
            host = self.runner.requests[req_id]
            host.generator = self.sampling.pop(req_id)
            host.output_token_ids[:] = known[len(host.prompt_token_ids) :]
            batch = self.runner.input_batch
            row = batch.req_id_to_index[req_id]
            batch.token_ids_cpu[row, : len(known)] = known
            batch.num_tokens_no_spec[row] = len(known)
            batch.is_token_ids[row, : len(known)] = True
            if hasattr(batch, "num_tokens"):
                batch.num_tokens[row] = len(known)
            if host.generator is not None:
                batch.generators[row] = host.generator
        elif operation == "install":
            request, version, layouts = args
            current = self.states.get(request.request_id)
            if current is None or current.instance != request.instance:
                raise RuntimeError("unknown worker request instance")
            if current.version > version:
                raise RuntimeError("stale worker layout version")
            if current.version == version and current.layers != layouts:
                raise RuntimeError("conflicting worker layout version")
            record = self.transactions.get(request.pending)
            if record is None or record[0].target_version != version:
                raise RuntimeError("installation requires a known transaction")
            plan = record[0]
            expected = dict(current.layers)
            expected.update({p.layer: p.new for p in plan.layers})
            if layouts != expected:
                raise RuntimeError("installation differs from planned layout")
            replace_block_tables(self.runner, request, layouts)
            current.layers = deepcopy(layouts)
            current.version = version
            receipt = {
                "instance": request.instance,
                "version": version,
                "transaction": request.pending,
                "status": "APPLIED",
            }
        elif operation == "commit":
            instance, version = args
            current = next(
                (s for s in self.states.values() if s.instance == instance), None
            )
            if current is None or current.version != version:
                raise RuntimeError("commit version mismatch")
            self.committed[instance] = version
            for transaction, (plan, _) in tuple(self.transactions.items()):
                if plan.instance == instance and plan.target_version == version:
                    self.transactions[transaction] = (plan, "COMMITTED")
                    receipt = {
                        "instance": instance,
                        "version": version,
                        "transaction": transaction,
                        "status": "COMMITTED",
                    }
        elif operation == "forget":
            (instance,) = args
            for req_id, state in tuple(self.states.items()):
                if state.instance == instance:
                    self.retired[instance] = state.version
                    batch = self.runner.input_batch
                    previous = getattr(batch, "prev_req_id_to_index", None)
                    if previous is not None:
                        previous.pop(req_id, None)
                    batch.remove_request(req_id)
                    self.runner.requests.pop(req_id, None)
                    del self.states[req_id]
                    del self.instances[req_id]
            self.committed.pop(instance, None)
            self.retired.setdefault(instance, -1)
            self.transactions = {
                key: record
                for key, record in self.transactions.items()
                if record[0].instance != instance
            }
        else:
            raise ValueError(f"unknown KNorm control operation {operation}")
        self.synchronize()
        return receipt

    def prepare_known_inputs(self):
        if self.replay_input is None:
            return
        req_id, known = self.replay_input
        batch = self.runner.input_batch
        row = batch.req_id_to_index[req_id]
        batch.token_ids_cpu[row, : len(known)] = known
        batch.num_tokens_no_spec[row] = len(known)
        batch.is_token_ids[row, : len(known)] = True
        if hasattr(batch, "num_tokens"):
            batch.num_tokens[row] = len(known)
        host = self.runner.requests[req_id]
        host.output_token_ids[:] = known[len(host.prompt_token_ids) :]


class RemoteCache:
    block_size = 128

    def __init__(self, rpc):
        self.rpc = rpc

    def choose(self, layout, keep, recorded=None):
        return self.rpc("choose", layout, keep, recorded)

    def begin(self, request, plan):
        self.plan = plan
        self.rpc("begin", request, plan)

    def move(self, packing):
        self.rpc("move", self.plan.transaction, packing)

    def synchronize(self):
        self.rpc("synchronize")


class RemoteWorker:
    def __init__(self, rpc, allocator):
        self.rpc, self.allocator = rpc, allocator
        self.pending = {}

    def quiesce(self, instance):
        self.rpc("synchronize")

    def install(self, request, version, layouts):
        receipt = self.rpc("install", request, version, layouts)
        self.check_receipt(receipt, request, version, "APPLIED")
        self.pending[request.instance] = request

    def commit(self, instance, version):
        request = self.pending[instance]
        self.allocator.install(request)
        receipt = self.rpc("commit", instance, version)
        self.check_receipt(receipt, request, version, "COMMITTED")

    def forget(self, instance):
        self.rpc("forget", instance)
        self.pending.pop(instance, None)

    @staticmethod
    def check_receipt(receipt, request, version, status):
        expected = {
            "instance": request.instance,
            "version": version,
            "transaction": request.pending,
            "status": status,
        }
        if receipt != expected:
            raise RuntimeError("stale or conflicting worker acknowledgement")
