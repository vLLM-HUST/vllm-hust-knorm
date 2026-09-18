# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-HUST project
"""Cross-cutting host layer: detection, guards, activation pipeline."""

from .activation import activate, is_activated
from .hosts import VLLM_HOST, detect_host

__all__ = ["VLLM_HOST", "activate", "detect_host", "is_activated"]
