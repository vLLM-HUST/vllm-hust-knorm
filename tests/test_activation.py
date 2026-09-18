# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-HUST project
"""Bootstrap and activation pipeline: no-op outside hosts, fail closed
on conflicts and missing surfaces, idempotent on re-entry."""

from __future__ import annotations

import pytest

import vllm_hust_knorm.bootstrap as bootstrap
from vllm_hust_knorm.core import activation


class TestBootstrap:
    def test_noop_without_vllm_host(self, monkeypatch):
        monkeypatch.setattr(bootstrap, "detect_host", lambda: None)
        assert bootstrap.register_plugins() == []
        assert activation.is_activated() is False

    def test_registers_in_fake_host(self, fake_host, capsys):
        result = bootstrap.register_plugins()

        assert result == ["knorm"]
        assert activation.is_activated() is True
        stdout = capsys.readouterr().out
        assert "[vllm-hust-knorm]" in stdout
        assert "VLLM_KNORM_ENABLED=1" in stdout

    def test_fail_closed_on_in_tree_knorm(self, fake_host):
        import sys

        from conftest import register_module

        register_module("vllm.knorm")

        with pytest.raises(RuntimeError, match="in-tree"):
            bootstrap.register_plugins()
        del sys.modules["vllm.knorm"]

    def test_stale_namespace_dir_is_not_an_in_tree_conflict(self, fake_host):
        # Real-host regression (910B2 checkout): an emptied vllm/knorm/
        # directory with only __pycache__ left resolves as a namespace
        # package (loader=None) — that must NOT trip the conflict guard.
        import importlib.machinery
        import sys

        namespace_spec = importlib.machinery.ModuleSpec("vllm.knorm", loader=None)
        namespace_spec.submodule_search_locations = ["/nonexistent/vllm/knorm"]
        stale = type(sys)("vllm.knorm")
        stale.__spec__ = namespace_spec
        sys.modules["vllm.knorm"] = stale

        try:
            result = bootstrap.register_plugins()
        finally:
            del sys.modules["vllm.knorm"]

        assert result == ["knorm"]

    def test_fail_closed_on_missing_surface(self, fake_host):
        import sys

        del sys.modules["vllm.v1.core.sched.scheduler"]

        with pytest.raises(RuntimeError, match="sched.scheduler"):
            activation.activate()


class TestActivationPipeline:
    def test_activate_installs_patches_once(self, fake_host):
        first = activation.activate()
        second = activation.activate()

        assert first["status"] == "registered"
        assert first["extension_id"] == "org.vllm-hust.knorm"
        assert first["patches"]["spec_registration"] is True
        assert second["status"] == "already_activated"

    def test_facade_activate_matches_pipeline(self, fake_host):
        import vllm_hust_knorm

        info = vllm_hust_knorm.activate()

        assert info["status"] == "registered"
        assert activation.is_activated() is True


class TestImportPurity:
    def test_package_import_has_no_host_side_effects(self, clean_knorm_env):
        # Importing the plugin must not import vllm or torch.
        import subprocess
        import sys

        code = (
            "import sys, vllm_hust_knorm as p; "
            "assert p.__version__; "
            "assert 'vllm' not in sys.modules; "
            "assert 'torch' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env={
                "PATH": __import__("os").environ.get("PATH", ""),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            check=False,
        )
        assert result.returncode == 0, result.stderr
