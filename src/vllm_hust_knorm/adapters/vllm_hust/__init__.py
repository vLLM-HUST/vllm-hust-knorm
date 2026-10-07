# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-HUST project
"""Adapter for the vLLM-HUST host (vllm v1 core)."""

from .register import EXTENSION_ID, VllmHustAdapter

__all__ = ["EXTENSION_ID", "VllmHustAdapter"]
