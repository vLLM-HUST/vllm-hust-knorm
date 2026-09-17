# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-HUST project
"""Manifest 0.2 consistency: fields, IDs, version sync, entry points,
and the guide's rule that discovery must not import implementations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_hust_knorm._version import __version__

SRC = Path(__file__).resolve().parents[1] / "src"
MANIFEST_PATH = (
    SRC / "vllm_hust_knorm" / "manifests" / "vllm-hust-extension-v0.2.json"
)
EXTENSION_ID = "org.vllm-hust.knorm"


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


class TestManifestFields:
    def test_schema_and_identity(self, manifest):
        assert manifest["schema_version"] == "0.2-experimental"
        assert manifest["extension_id"] == EXTENSION_ID
        assert manifest["extension_version"] == __version__

    def test_kind_host_runtime_lifecycle(self, manifest):
        assert manifest["kind"] == "in_process_plugin"
        assert manifest["host"]["provider"] == "vllm"
        assert manifest["host"]["name"] == "vllm"
        assert manifest["runtime"]["type"] == "python"
        assert manifest["runtime"]["isolation"] == "trusted_in_process"
        assert manifest["lifecycle_owner"] == "vllm"

    def test_components_are_unique_and_minimal(self, manifest):
        component_ids = [c["component_id"] for c in manifest["components"]]
        assert len(component_ids) == len(set(component_ids))
        # Only the attention collector touches the device; the manager
        # stays on the CPU scheduler plane.
        by_id = {c["component_id"]: c for c in manifest["components"]}
        assert by_id["kv-compression-manager"]["permissions"] == []
        assert by_id["attention-norm-collector"]["permissions"] == [
            "device_access"
        ]

    def test_implementation_carrier_is_active(self, manifest):
        carriers = manifest["implementation"]
        assert len(carriers) == 1
        assert carriers[0]["status"] == "active"
        assert carriers[0]["module"] == "vllm_hust_knorm.bootstrap"
        assert carriers[0]["object"] == "register_plugins"

    def test_activation_environment_covers_enable_switch(self, manifest):
        environment = manifest["activation"]["environment"]
        assert environment["VLLM_KNORM_ENABLED"] == "1"

    def test_requires_services_empty_for_in_process_plugin(self, manifest):
        assert manifest["requires_services"] == []

    def test_exactly_one_manifest_file(self):
        manifests_dir = MANIFEST_PATH.parent
        json_files = list(manifests_dir.glob("*.json"))
        assert json_files == [MANIFEST_PATH]

    def test_implementation_refs_resolve(self, manifest):
        import importlib

        for component in manifest["components"]:
            ref = component["implementation_ref"]
            module_name, _, attr = ref.partition(":")
            module = importlib.import_module(module_name)
            assert getattr(module, attr) is not None, ref


class TestEntryPointConsistency:
    def test_pyproject_declares_both_entry_points(self):
        import tomllib

        pyproject = tomllib.loads(
            (Path(__file__).resolve().parents[1] / "pyproject.toml")
            .read_text(encoding="utf-8")
        )
        general = pyproject["project"]["entry-points"]["vllm.general_plugins"]
        bundles = pyproject["project"]["entry-points"][
            "vllm_hust.extension_bundles"
        ]
        # Runtime hook and manifest locator both present.
        assert general == {
            "vllm-hust-knorm": "vllm_hust_knorm.bootstrap:register_plugins"
        }
        # Registration name must equal the extension_id (guide §5.2).
        assert bundles == {EXTENSION_ID: "vllm_hust_knorm.manifests"}

    def test_installed_distribution_matches(self):
        import importlib.metadata as md

        try:
            distribution = md.distribution("vllm-hust-knorm")
        except md.PackageNotFoundError:  # pragma: no cover - non-installed
            pytest.skip("vllm-hust-knorm is not installed in this env")

        bundles = [
            ep
            for ep in distribution.entry_points
            if ep.group == "vllm_hust.extension_bundles"
        ]
        general = [
            ep
            for ep in distribution.entry_points
            if ep.group == "vllm.general_plugins"
        ]
        assert [ep.name for ep in bundles] == [EXTENSION_ID]
        assert bundles[0].value == "vllm_hust_knorm.manifests"
        assert [ep.name for ep in general] == ["vllm-hust-knorm"]
        assert distribution.version == __version__


class TestDiscoveryPurity:
    def test_import_manifests_package_does_not_import_torch_or_vllm(self):
        import subprocess
        import sys

        code = (
            "import sys; import vllm_hust_knorm.manifests as m; "
            "assert 'torch' not in sys.modules; "
            "assert 'vllm' not in sys.modules; print('ok')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
