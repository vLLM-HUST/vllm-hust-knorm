"""Explicit development entry point for the fixed host sources."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path

from .guard import ASCEND_COMMIT, CORE_COMMIT, validate


def verify_sources():
    lock = json.loads(Path(__file__).with_name("source-lock.json").read_text())
    roots = {}
    for package in ("vllm", "vllm_ascend"):
        spec = importlib.util.find_spec(package)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"missing candidate host package {package}")
        roots[package] = Path(spec.origin).parent.parent
    for relative, expected in lock.items():
        root = roots[relative.split("/")[0]]
        path = root / relative
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise RuntimeError(f"candidate host source mismatch: {relative}")
    return roots


def configure(vllm_config, policy, workspace_bytes):
    """Opt in a resolved config and propagate the guarded loader to child workers."""
    verify_sources()
    if type(workspace_bytes) is not int or workspace_bytes <= 0:
        raise ValueError("workspace_bytes must be a positive integer")
    if os.environ.get("VLLM_KV_CACHE_LAYOUT", "LBNHC") != "LBNHC":
        raise RuntimeError("candidate requires VLLM_KV_CACHE_LAYOUT=LBNHC")
    options = {
        "core_commit": CORE_COMMIT,
        "ascend_commit": ASCEND_COMMIT,
        "policy": {
            "mode": policy.mode,
            "prune_after": policy.prune_after,
            "keep_ratio": str(policy.keep_ratio),
        },
        "workspace_bytes": workspace_bytes,
    }
    previous = vllm_config.additional_config.get("knorm_candidate")
    if previous is not None and previous != options:
        raise RuntimeError("conflicting candidate configuration")
    vllm_config.additional_config["knorm_candidate"] = options
    try:
        validate(vllm_config)
    except Exception:
        if previous is None:
            vllm_config.additional_config.pop("knorm_candidate")
        raise
    from .loading import install

    os.environ["VLLM_KV_CACHE_LAYOUT"] = "LBNHC"
    os.environ["VLLM_KNORM_CANDIDATE"] = "1"
    install()
    return vllm_config
