# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-HUST project
"""Bind Knorm into the vLLM-HUST (vllm v1) host process."""

from __future__ import annotations

from typing import Any

from ..base import HostAdapter

EXTENSION_ID = "org.vllm-hust.knorm"


class VllmHustAdapter(HostAdapter):
    host = "vllm"

    def register(self) -> dict[str, Any]:
        """Install the runtime patches; the environment still gates
        activation (``VLLM_KNORM_ENABLED``), mirroring the reference
        plugin's “register ≠ enable” split."""
        self.require_host()
        from .patches import install_patches

        info = dict(install_patches())
        info.update(
            {
                "host": self.host,
                "extension_id": EXTENSION_ID,
                "integration": "vllm.general_plugins runtime patches",
                "activation_env": "VLLM_KNORM_ENABLED=1",
                "usage": (
                    "VLLM_KNORM_ENABLED=1 VLLM_KNORM_COMPRESSION_RATIO=0.5 "
                    "vllm serve MODEL --no-enable-prefix-caching"
                ),
            }
        )
        return info


__all__ = ["EXTENSION_ID", "VllmHustAdapter"]
