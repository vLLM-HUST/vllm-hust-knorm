# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-HUST project
"""Packaging gates: the wheel and the sdist must both ship the manifest
(extension guide §14.1), and the wheel must carry both entry points."""

from __future__ import annotations

import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from vllm_hust_knorm._version import __version__

REPO = Path(__file__).resolve().parents[1]


def build_distributions(tmp_path: Path) -> Path:
    pytest.importorskip("build")
    out = tmp_path / "dist"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--outdir",
            str(out),
            "--no-isolation",
            str(REPO),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return out


def test_wheel_and_sdist_ship_the_manifest(tmp_path: Path):
    out = build_distributions(tmp_path)

    (wheel_path,) = out.glob("*.whl")
    with zipfile.ZipFile(wheel_path) as wheel:
        names = wheel.namelist()
    assert any(
        name.endswith("manifests/vllm-hust-extension-v0.3.json") for name in names
    ), names
    assert any(name.endswith("manifests/__init__.py") for name in names), names

    (sdist_path,) = out.glob("*.tar.gz")
    with tarfile.open(sdist_path) as sdist:
        sdist_names = sdist.getnames()
    assert any(
        name.endswith("manifests/vllm-hust-extension-v0.3.json") for name in sdist_names
    ), sdist_names


def test_wheel_metadata_carries_entry_points_and_version(tmp_path: Path):
    out = build_distributions(tmp_path)

    (wheel_path,) = out.glob("*.whl")
    with zipfile.ZipFile(wheel_path) as wheel:
        (metadata_name,) = [n for n in wheel.namelist() if n.endswith("METADATA")]
        metadata = wheel.read(metadata_name).decode("utf-8")
        (entry_points_name,) = [
            n for n in wheel.namelist() if n.endswith("entry_points.txt")
        ]
        entry_points = wheel.read(entry_points_name).decode("utf-8")

    assert f"Version: {__version__}" in metadata
    assert "[vllm_hust.extension_bundles]" in entry_points
    assert "org.vllm-hust.knorm = vllm_hust_knorm.manifests" in entry_points
    assert "[vllm.general_plugins]" in entry_points
    assert (
        "vllm-hust-knorm = vllm_hust_knorm.bootstrap:register_plugins" in entry_points
    )


def test_installed_wheel_public_configuration_and_activation_gate(tmp_path: Path):
    out = build_distributions(tmp_path)
    (wheel_path,) = out.glob("*.whl")
    environment = tmp_path / "installed"
    subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
    python = environment / "bin" / "python"
    installed = subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            str(wheel_path),
        ],
        capture_output=True,
        text=True,
    )
    assert installed.returncode == 0, installed.stdout + installed.stderr
    code = """
import os
import sys
import vllm_hust_knorm as package
from vllm_hust_knorm import KNormConfig, KnormConfig, should_activate
from vllm_hust_knorm.runtime.engine import Simulation

for name in tuple(os.environ):
    if name.startswith("VLLM_KNORM_"):
        del os.environ[name]
assert "site-packages" in package.__file__
assert not {"torch", "torch_npu", "vllm", "vllm_hust_knorm.knorm"}.intersection(
    sys.modules
)
for mode in ("once", "repeat"):
    config = KNormConfig(mode=mode, prune_after=4, keep_ratio="0.5")
    assert config.budget(9) == 5
    sim = Simulation(config, block_size=4)
    request = sim.create("installed", 8)
    sim.append(request, range(8), lambda l, h, t, p: ((float(p),), (float(t),)))
    sim.controller.before_decode(request)
    assert [layout.length for layout in request.layers.values()] == [8, 8, 4, 4]
for call, exception, message in (
    (lambda: KNormConfig(), TypeError, "required positional arguments"),
    (lambda: KnormConfig(), TypeError, "use KNormConfig"),
    (lambda: KnormConfig(compression_ratio=0.5), TypeError, "use KNormConfig"),
    (lambda: should_activate(False), RuntimeError, "should_activate was removed"),
    (lambda: KNormConfig(mode="once", prune_after=4, keep_ratio="0.5",
                        warmup_tokens=32), TypeError, "warmup_tokens"),
):
    try:
        call()
    except exception as error:
        assert message in str(error)
    else:
        raise AssertionError("legacy or implicit configuration was accepted")
os.environ["VLLM_KNORM_ENABLED"] = "1"
try:
    package.activate()
except RuntimeError as error:
    assert "import_only" in str(error)
else:
    raise AssertionError("runtime enablement was accepted")
os.environ["VLLM_KNORM_ENABLED"] = "0"
os.environ["VLLM_KNORM_COMPRESSION_RATIO"] = "0.5"
try:
    package.activate()
except ValueError as error:
    assert "legacy" in str(error)
else:
    raise AssertionError("legacy environment was accepted")
"""
    result = subprocess.run(
        [str(python), "-I", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
