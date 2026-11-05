"""Check the installed public entry point and isolated candidate loader."""

import importlib.metadata
import os
import sys

from test_without_devices import isolate_devices


def main():
    isolate_devices()
    os.environ.pop("VLLM_KNORM_CANDIDATE", None)
    import vllm_hust_knorm

    assert "torch" not in sys.modules
    assert "vllm_ascend.worker.model_runner_v1" not in sys.modules
    entry = next(
        ep
        for ep in importlib.metadata.entry_points(group="vllm.general_plugins")
        if ep.name == "vllm-hust-knorm"
    )
    assert entry.load()() == ["knorm"]
    assert vllm_hust_knorm.activate()["status"] == "already_activated"
    from vllm_hust_knorm.adapters.ascend.candidate import verify_sources
    from vllm_hust_knorm.adapters.ascend.loading import install

    verify_sources()
    install()
    install()
    assert "vllm_ascend.worker.model_runner_v1" not in sys.modules
    os.environ["VLLM_KNORM_ENABLED"] = "1"
    try:
        vllm_hust_knorm.activate()
    except RuntimeError:
        pass
    else:
        raise AssertionError("public runtime eligibility must remain blocked")
    print("Public entry remains import-only; candidate loader checks passed.")


if __name__ == "__main__":
    main()
