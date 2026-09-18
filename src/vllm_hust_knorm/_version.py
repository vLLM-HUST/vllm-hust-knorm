# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-HUST project
"""Single source of truth for the distribution version.

Must stay in sync with ``extension_version`` in
``manifests/vllm-hust-extension-v0.2.json``; ``tests/test_manifest.py``
enforces it. PyPI forbids re-uploading the same version, so bump this
after every code change that is published (see
docs/packaging-and-release.md).
"""

__version__ = "0.1.0"
