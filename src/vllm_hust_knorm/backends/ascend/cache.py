"""Candidate contiguous FP16 [page, slot, head, dim] cache contract."""

from .movement import move_tensor
from .selection import select_tensor


def workspace_bytes(length, heads, dimension):
    if min(length, heads, dimension) < 0:
        raise ValueError("negative geometry")
    # Conservative admission estimate: K gather, FP32 components/scores,
    # stable-sort indices and position sidecars. Vendor sort scratch is unknown.
    return length * heads * (2 * dimension + 128)


class AscendCache:
    def __init__(
        self, keys, values, positions, workspace_limit, *, cpu_validation=False
    ):
        import torch

        if keys.shape != values.shape or keys.ndim != 4:
            raise ValueError("K/V layout mismatch")
        if positions.shape != keys.shape[:3] or positions.dtype != torch.int64:
            raise ValueError("position sidecar must be int64 [page, slot, head]")
        if keys.dtype != torch.float16 or values.dtype != torch.float16:
            raise ValueError("only nonquantized FP16 KV is supported")
        if any(
            t.device != keys.device or not t.is_contiguous()
            for t in (keys, values, positions)
        ):
            raise ValueError("cache must be contiguous on one device")
        if keys.device.type != "npu" and not (
            cpu_validation and keys.device.type == "cpu"
        ):
            raise RuntimeError(
                "Ascend NPU required (CPU allowed only for explicit validation)"
            )
        if keys.shape[1] != 128 and not cpu_validation:
            raise ValueError("candidate Ascend page size is 128")
        self.keys, self.values, self.positions = keys, values, positions
        self.block_size, self.heads = keys.shape[1:3]
        self.workspace_limit = workspace_limit

    def choose(self, layout, keep, recorded=None):
        import torch

        needed = workspace_bytes(layout.length, self.heads, self.keys.shape[-1])
        if needed > self.workspace_limit:
            raise MemoryError("selection workspace budget exceeded before APPLY")
        logical = torch.arange(layout.length, device=self.keys.device)
        table = torch.tensor([p.number for p in layout.pages], device=self.keys.device)
        page, slot = table[logical // self.block_size], logical % self.block_size
        positions = self.positions[page, slot].transpose(0, 1)
        sidecar = positions.tolist()
        if recorded is None:
            keys = self.keys[page, slot].transpose(0, 1)
            indices = select_tensor(keys, positions, keep).tolist()
        else:
            if len(recorded) != self.heads:
                raise ValueError("replay head mismatch")
            indices = [
                [pos.index(p) for p in chosen]
                for pos, chosen in zip(sidecar, recorded, strict=True)
            ]
        if any(len(h) != keep for h in indices):
            raise ValueError("replay budget mismatch")
        return tuple(tuple(h) for h in indices), sidecar

    def move(self, packing):
        move_tensor(self.keys, self.values, self.positions, packing)

    def synchronize(self):
        if self.keys.device.type == "npu":
            import torch

            torch.npu.synchronize(self.keys.device)
