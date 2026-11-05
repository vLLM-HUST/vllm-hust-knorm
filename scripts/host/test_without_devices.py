"""Run host contracts without allowing a device backend to initialize."""

import importlib.abc
import importlib.util
import os
import sys
from pathlib import Path


class DeviceImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"torch_npu", "acl", "triton.backends"}:
            raise ImportError(
                f"device backend import forbidden in CPU validation: {fullname}"
            )
        return None


def isolate_devices():
    sys.dont_write_bytecode = True
    root = Path(sys.prefix).parent.parent
    if root.is_relative_to("/root/knorm"):
        os.environ.update(
            XDG_CACHE_HOME=str(root / "cache"),
            TMPDIR=str(root / "tmp"),
            HF_HOME=str(root / "cache/huggingface"),
            VLLM_CACHE_ROOT=str(root / "cache/vllm"),
        )
    os.environ.update(
        TORCH_DEVICE_BACKEND_AUTOLOAD="0",
        VLLM_NO_USAGE_STATS="1",
        VLLM_USE_V2_MODEL_RUNNER="0",
        VLLM_PLUGINS="",
        KNORM_RUN_HOST_TESTS="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )
    # Availability probes should see the deliberately excluded device backend as
    # unavailable. An actual import still raises, rather than creating a stub.
    find_spec = importlib.util.find_spec

    def cpu_find_spec(name, package=None):
        if name in {"torch_npu", "acl", "triton.backends"}:
            return None
        return find_spec(name, package)

    importlib.util.find_spec = cpu_find_spec
    sys.meta_path.insert(0, DeviceImportsForbidden())


def main():
    isolate_devices()
    if sys.argv[1:] == ["--probe-imports"]:
        import importlib
        import json
        import traceback

        results = {}
        for name in (
            "vllm_ascend.patch.platform.patch_kv_cache_utils",
            "vllm_ascend.worker.model_runner_v1",
        ):
            try:
                importlib.import_module(name)
            except Exception:
                results[name] = {
                    "status": "unavailable",
                    "traceback": traceback.format_exc(),
                }
            else:
                results[name] = {"status": "imported"}
        print(json.dumps(results, indent=2))
        return
    import pytest

    raise SystemExit(pytest.main(sys.argv[1:]))


if __name__ == "__main__":
    main()
