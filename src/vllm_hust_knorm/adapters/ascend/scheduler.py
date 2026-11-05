"""Per-layer length translation at the real manager allocation boundary."""

import inspect
from types import MethodType


def bind_layer_allocations(coordinator, requests):
    for layer, manager in enumerate(coordinator.single_type_managers):
        previous = getattr(manager, "_knorm_states", None)
        if previous is requests:
            continue
        if previous is not None:
            raise RuntimeError("manager already bound to different request states")
        manager._knorm_states = requests
        for name in ("get_num_blocks_to_allocate", "allocate_new_blocks"):
            original = getattr(manager, name)
            signature = inspect.signature(original)

            def build(original, signature, layer):
                def translated(manager, *args, **kwargs):
                    bound = signature.bind(*args, **kwargs)
                    state = requests.get(bound.arguments["request_id"])
                    if state is not None and layer in state.layers:
                        removed = state.semantic_length - state.layers[layer].length
                        for parameter in (
                            "num_tokens",
                            "num_tokens_main_model",
                            "total_computed_tokens",
                        ):
                            if parameter in bound.arguments:
                                bound.arguments[parameter] = max(
                                    0, bound.arguments[parameter] - removed
                                )
                    return original(*bound.args, **bound.kwargs)

                return translated

            setattr(
                manager, name, MethodType(build(original, signature, layer), manager)
            )
