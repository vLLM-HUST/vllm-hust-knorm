"""Executable CPU cache and worker, including partial-write/ACK fault injection."""

from copy import deepcopy


class MemoryCache:
    def __init__(self, block_size, heads, workspace_limit=64 * 1024 * 1024):
        if block_size <= 0 or heads <= 0:
            raise ValueError("invalid cache geometry")
        self.workspace_limit = workspace_limit
        self.block_size = block_size
        self.heads = heads
        self.data = {}
        self.fail_after = None

    def write(self, page, slot, head, key, value, position):
        self.data[page, slot, head] = (tuple(key), tuple(value), position)

    def read(self, layout):
        rows = [
            [
                self.data[
                    layout.pages[i // self.block_size].number, i % self.block_size, h
                ]
                for i in range(layout.length)
            ]
            for h in range(self.heads)
        ]
        return tuple([[row[j] for row in head] for head in rows] for j in range(3))

    def choose(self, layout, keep, recorded=None):
        from ..algorithm.selection import select

        # CPU oracle admission covers scalar score/index/position records.
        if layout.length * self.heads * 256 > self.workspace_limit:
            raise MemoryError("selection workspace budget exceeded before APPLY")
        keys, _, positions = self.read(layout)
        if recorded is None:
            indices = select(keys, positions, keep)
        else:
            if len(recorded) != self.heads:
                raise ValueError("replay head mismatch")
            indices = tuple(
                tuple(pos.index(p) for p in chosen)
                for pos, chosen in zip(positions, recorded, strict=True)
            )
            if any(len(h) != keep for h in indices):
                raise ValueError("replay budget mismatch")
        return indices, positions

    def move(self, packing):
        for i, move in enumerate(packing.moves):
            if self.fail_after == i:
                self.fail_after = None
                raise RuntimeError("injected partial APPLY")
            self.data[move.target + (move.head,)] = self.data[
                move.source + (move.head,)
            ]

    def synchronize(self):
        pass


class MemoryWorker:
    def __init__(self):
        self.layouts = {}
        self.committed = {}
        self.busy = set()
        self.lose_ack = None

    def quiesce(self, instance):
        if instance in self.busy:
            raise RuntimeError("forward still in flight")

    def install(self, request, version, layouts):
        old = self.layouts.get(request.instance)
        if old is not None and old[0] > version:
            raise RuntimeError("stale worker layout")
        if old is not None and old[0] == version and old[1] != layouts:
            raise RuntimeError("conflicting worker layout")
        self.layouts[request.instance] = (version, deepcopy(layouts))
        if self.lose_ack == "APPLIED":
            self.lose_ack = None
            raise RuntimeError("lost APPLIED acknowledgement")

    def commit(self, instance, version):
        if self.layouts[instance][0] != version:
            raise RuntimeError("commit version mismatch")
        self.committed[instance] = version
        if self.lose_ack == "COMMIT":
            self.lose_ack = None
            raise RuntimeError("lost COMMIT acknowledgement")

    def forget(self, instance):
        self.quiesce(instance)
        self.layouts.pop(instance, None)
        self.committed.pop(instance, None)
