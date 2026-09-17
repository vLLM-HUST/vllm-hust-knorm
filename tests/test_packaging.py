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
        name.endswith("manifests/vllm-hust-extension-v0.2.json")
        for name in names
    ), names
    assert any(
        name.endswith("manifests/__init__.py") for name in names
    ), names

    (sdist_path,) = out.glob("*.tar.gz")
    with tarfile.open(sdist_path) as sdist:
        sdist_names = sdist.getnames()
    assert any(
        name.endswith("manifests/vllm-hust-extension-v0.2.json")
        for name in sdist_names
    ), sdist_names


def test_wheel_metadata_carries_entry_points_and_version(tmp_path: Path):
    out = build_distributions(tmp_path)

    (wheel_path,) = out.glob("*.whl")
    with zipfile.ZipFile(wheel_path) as wheel:
        (metadata_name,) = [
            n for n in wheel.namelist() if n.endswith("METADATA")
        ]
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
        "vllm-hust-knorm = vllm_hust_knorm.bootstrap:register_plugins"
        in entry_points
    )
