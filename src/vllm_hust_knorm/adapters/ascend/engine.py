"""Control transactions between completed forwards and the next host schedule."""

from types import MethodType

from ...runtime.controller import Controller
from ...runtime.recovery import preempt
from ...runtime.state import Request
from ...runtime.transaction import Transactions
from .allocator import HostAllocator
from .guard import validate
from .scheduler import bind_layer_allocations
from .worker import RemoteCache, RemoteWorker


class EngineBinding:
    def __init__(self, engine):
        policy = validate(engine.vllm_config)
        self.engine = engine
        self.states = {}
        self.host_requests = {}
        self.checkpoints = {}
        self.preempting = set()
        self.inflight = None
        self.allocator = HostAllocator(engine.scheduler.kv_cache_manager.coordinator)
        self.transactions = Transactions(
            self.allocator,
            RemoteCache(self.rpc),
            RemoteWorker(self.rpc, self.allocator),
        )
        self.controller = Controller(policy, self.transactions)
        bind_layer_allocations(self.allocator.coordinator, self.states)
        engine.scheduler.knorm_binding = self
        unfinished = engine.scheduler.get_num_unfinished_requests

        def count_unfinished(scheduler):
            from vllm.v1.core.sched.interface import PauseState

            pending = (
                len(self.checkpoints)
                if scheduler.pause_state == PauseState.UNPAUSED
                else 0
            )
            return unfinished() + pending

        engine.scheduler.get_num_unfinished_requests = MethodType(
            count_unfinished, engine.scheduler
        )
        schedule = engine.scheduler.schedule

        def scheduled(scheduler, *args, **kwargs):
            output = schedule(*args, **kwargs)
            self.scheduled(output)
            self.inflight = output
            return output

        engine.scheduler.schedule = MethodType(scheduled, engine.scheduler)
        original = engine.scheduler._preempt_request

        def preempt_request(scheduler, request, *args, **kwargs):
            state = self.states.get(request.request_id)
            if state is not None:
                self.allocator.adopt(state, self.lengths(state))
                self.rpc("checkpoint", state.instance)
                self.checkpoints[request.request_id] = preempt(state, self.transactions)
            self.preempting.add(request.request_id)
            try:
                result = original(request, *args, **kwargs)
            finally:
                self.preempting.discard(request.request_id)
            if request.request_id in self.checkpoints:
                scheduler.waiting.remove_request(request)
            return result

        engine.scheduler._preempt_request = MethodType(
            preempt_request, engine.scheduler
        )

    def lengths(self, state):
        return {
            layer: state.layers[layer].length if state.layers else 0
            for layer in range(len(self.allocator.coordinator.single_type_managers))
        }

    def rpc(self, operation, *args):
        replies = self.engine.model_executor.collective_rpc(
            "knorm_control", args=(operation, *args)
        )
        if len(replies) != 1:
            raise RuntimeError("candidate control requires exactly one worker")
        return replies[0]

    def before_schedule(self):
        from vllm.v1.core.sched.interface import PauseState

        scheduler = self.engine.scheduler
        for req_id, host in scheduler.requests.items():
            previous = self.host_requests.get(req_id)
            if previous is not None and previous is not host:
                raise RuntimeError("request ID reused before prior instance retirement")
            if req_id not in self.states:
                self.states[req_id] = Request(req_id, host.num_prompt_tokens)
                self.host_requests[req_id] = host
            state = self.states[req_id]
            if req_id in self.checkpoints:
                from .recovery import restore_request

                if (
                    scheduler.pause_state != PauseState.UNPAUSED
                    or len(scheduler.running) >= scheduler.max_num_running_reqs
                ):
                    continue
                try:
                    restore_request(self, host, state, self.checkpoints[req_id])
                except MemoryError:
                    continue
                del self.checkpoints[req_id]
            self.controller.before_decode(state)

    def scheduled(self, output):
        for req_id in output.num_scheduled_tokens:
            state = self.states[req_id]
            lengths = {
                layer: state.layers[layer].length if state.layers else 0
                for layer in range(len(self.allocator.coordinator.single_type_managers))
            }
            self.allocator.adopt(state, lengths)
            self.rpc("register", state)

    def after_forward(self, output):
        for req_id, count in output.num_scheduled_tokens.items():
            state = self.states[req_id]
            host = self.host_requests[req_id]
            self.rpc("advance", state, count)
            state.semantic_length += count
            state.tokens = list(host.all_token_ids[: state.semantic_length])
            lengths = {
                layer: layout.length + count for layer, layout in state.layers.items()
            }
            self.allocator.adopt(state, lengths)
            state.phase = (
                "decode" if state.semantic_length >= state.prompt_length else "prefill"
            )
            self.rpc("register", state)

    def free(self, host):
        state = self.states.get(host.request_id)
        if state is None:
            return False
        if host.request_id in self.preempting:
            return False
        if host.request_id in self.checkpoints:
            self.rpc("discard_sampling", host.request_id)
            del self.checkpoints[host.request_id]
            del self.states[host.request_id]
            del self.host_requests[host.request_id]
            return True
        self.allocator.adopt(state, self.lengths(state))
        self.transactions.retire(state)
        self.states.pop(host.request_id)
        self.host_requests.pop(host.request_id)
        return True
