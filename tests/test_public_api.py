"""Public token configuration and explicit rejection of legacy entry points."""

import subprocess
import sys

import pytest

import vllm_hust_knorm as knorm


def test_root_exports_explicit_token_configuration():
    from vllm_hust_knorm import KNormConfig
    from vllm_hust_knorm.config import KNormConfig as Config

    assert KNormConfig is Config
    assert "KNormConfig" in knorm.__all__
    for mode in ("once", "repeat"):
        config = KNormConfig(mode=mode, prune_after=4, keep_ratio="0.5")
        assert config.mode == mode and config.prune_after == 4
        assert config.budget(9) == 5
    with pytest.raises(TypeError, match="required positional arguments"):
        KNormConfig()


def test_legacy_public_names_reject_with_migration_guidance():
    from vllm_hust_knorm import KnormConfig, should_activate

    assert "KnormConfig" not in knorm.__all__
    assert "should_activate" not in knorm.__all__
    for kwargs in ({}, {"compression_ratio": 0.5, "warmup_tokens": 32}):
        with pytest.raises(TypeError, match="use KNormConfig"):
            KnormConfig(**kwargs)
    with pytest.raises(RuntimeError, match="should_activate was removed.*import_only"):
        should_activate(False)


@pytest.mark.parametrize("parameter", ["compression_ratio", "warmup_tokens"])
def test_token_configuration_rejects_legacy_parameters(parameter):
    settings = {"mode": "once", "prune_after": 4, "keep_ratio": "0.5"}
    settings[parameter] = 1
    with pytest.raises(TypeError, match=f"unexpected keyword argument '{parameter}'"):
        knorm.KNormConfig(**settings)
    with pytest.raises(ValueError, match="legacy/unknown keys rejected"):
        knorm.KNormConfig.from_mapping(settings)


def test_public_configuration_import_is_lazy():
    code = """
import sys
from vllm_hust_knorm import KNormConfig, KnormConfig, should_activate
KNormConfig(mode="once", prune_after=4, keep_ratio="0.5")
assert not {"torch", "torch_npu", "vllm", "vllm_hust_knorm.knorm"}.intersection(
    sys.modules
)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
