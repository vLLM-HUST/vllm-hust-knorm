"""Executable CPU integration harness; no production host is implicitly selected."""

from ..algorithm.policy import page_count
from .allocator import PagePool
from .controller import Controller
from .simulation import MemoryCache, MemoryWorker
from .state import Layout, Request
from .transaction import Transactions


class Simulation:
    def __init__(self, config, capacity=1024, block_size=128, heads=2, layers=4):
        self.pool = PagePool(capacity)
        self.cache = MemoryCache(block_size, heads)
        self.worker = MemoryWorker()
        self.transactions = Transactions(self.pool, self.cache, self.worker)
        self.controller = Controller(config, self.transactions)
        self.layer_count = layers
        self.requests = {}

    def create(self, request_id, prompt_length):
        if prompt_length < 1:
            raise ValueError("prompt must be nonempty")
        request = Request(request_id, prompt_length)
        request.layers = {i: Layout(0, ()) for i in range(self.layer_count)}
        self.requests[request.instance] = request
        return request

    def append(self, request, tokens, kv):
        """kv(layer, head, token, semantic_position) returns actual K/V vectors."""
        if request.pending or request.phase not in {"prefill", "decode", "replay"}:
            raise RuntimeError("forward forbidden in current phase")
        tokens = tuple(tokens)
        if not tokens:
            return
        if request.phase == "decode" and len(tokens) != 1:
            raise ValueError("only single-token decode is supported")
        if (
            request.phase == "prefill"
            and request.semantic_length + len(tokens) > request.prompt_length
        ):
            raise ValueError("prefill exceeds original prompt")
        if request.phase == "decode":
            self.controller.before_decode(request)
        self.worker.quiesce(request.instance)
        old = request.layers or {i: Layout(0, ()) for i in range(self.layer_count)}
        counts = {
            layer: page_count(v.length + len(tokens), self.cache.block_size)
            - len(v.pages)
            for layer, v in old.items()
        }
        if sum(counts.values()) > len(self.pool.free):
            raise MemoryError("all-layer append admission failed")
        new = {}
        try:
            for layer, layout in old.items():
                pages = layout.pages + self.pool.allocate(
                    (request.instance, layer), counts[layer]
                )
                new[layer] = Layout(layout.length + len(tokens), pages)
                for j, token in enumerate(tokens):
                    index = layout.length + j
                    for head in range(self.cache.heads):
                        position = request.semantic_length + j
                        key, value = kv(layer, head, token, position)
                        self.cache.write(
                            pages[index // self.cache.block_size].number,
                            index % self.cache.block_size,
                            head,
                            key,
                            value,
                            position,
                        )
            self.worker.install(request, request.version + 1, new)
            self.worker.commit(request.instance, request.version + 1)
        except Exception:
            request.phase = "failed"  # An unknown write/install must never generate.
            raise
        request.layers = new
        request.version += 1
        request.tokens.extend(tokens)
        request.semantic_length += len(tokens)
        if (
            request.phase == "prefill"
            and request.semantic_length == request.prompt_length
        ):
            request.phase = "decode"

    def finish(self, request):
        self.transactions.retire(request)
        self.requests.pop(request.instance, None)
