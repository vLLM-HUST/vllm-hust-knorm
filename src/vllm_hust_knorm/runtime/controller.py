"""Compression runs before allocation, once per semantic decode boundary."""

from ..algorithm.policy import target_length


class Controller:
    def __init__(self, config, transactions):
        self.config, self.transactions = config, transactions

    def before_decode(self, request):
        if request.pending:
            return self.transactions.execute(
                request, self.transactions.records[request.pending].plan
            )
        if request.phase != "decode":
            return None
        if request.checked_boundary == request.semantic_length:
            return None
        if self.config.mode == "once" and request.once_checked:
            return None
        targets = {
            layer: target_length(self.config, layer, layout.length)
            for layer, layout in request.layers.items()
        }
        targets = {
            layer: keep
            for layer, keep in targets.items()
            if keep < request.layers[layer].length
        }
        plan = self.transactions.plan(request, targets) if targets else None
        if plan:
            return self.transactions.execute(request, plan)
        request.once_checked = True
        request.checked_boundary = request.semantic_length
        return None
