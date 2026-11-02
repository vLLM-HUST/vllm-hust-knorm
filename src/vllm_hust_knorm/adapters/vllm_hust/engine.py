"""Production boundary is intentionally unavailable until a host binding is verified."""


def require_verified_binding():
    raise RuntimeError(
        "KNorm remains import_only: no verified vLLM-HUST/Ascend binding; "
        "The CPU simulation runtime is available through runtime.engine.Simulation only"
    )
