# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-HUST project
"""Host adapters. Heavy host imports happen inside ``register()`` only."""

from .base import HostAdapter

__all__ = ["HostAdapter"]
