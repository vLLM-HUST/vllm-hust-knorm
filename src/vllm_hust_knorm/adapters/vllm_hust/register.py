"""Import-only registration. Legacy patches are no longer installed."""

from typing import Any

from ..base import HostAdapter

EXTENSION_ID = "org.vllm-hust.knorm"


class VllmHustAdapter(HostAdapter):
    host = "vllm"

    def register(self) -> dict[str, Any]:
        self.require_host()
        return {
            "host": self.host,
            "extension_id": EXTENSION_ID,
            "integration": "import_only",
            "patches": {},
            "activation_env": "VLLM_KNORM_ENABLED=1 (currently rejected)",
        }
