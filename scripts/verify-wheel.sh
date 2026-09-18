#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-HUST project
#
# Wheel content gate from the vLLM-HUST packaging and release guide:
# the manifest must ship in the wheel (or vllm-hust-ext cannot discover
# the extension) and both entry-point groups must be registered.
set -euo pipefail

wheel=${1:?usage: scripts/verify-wheel.sh dist/package.whl}
python_bin=${PYTHON:-python3}
entries=$($python_bin -m zipfile -l "$wheel")

for required in \
  'vllm_hust_knorm/_version.py' \
  'vllm_hust_knorm/bootstrap.py' \
  'vllm_hust_knorm/core/hosts.py' \
  'vllm_hust_knorm/adapters/vllm_hust/patches.py' \
  'vllm_hust_knorm/knorm/manager.py' \
  'vllm_hust_knorm/manifests/__init__.py' \
  'vllm_hust_knorm/manifests/vllm-hust-extension-v0.2.json' \
  '.dist-info/entry_points.txt' \
  '.dist-info/METADATA'; do
  grep -q "$required" <<<"$entries" || {
    echo "FAIL: wheel is missing $required"
    exit 1
  }
done

for forbidden in 'provenance' 'tests/' 'docs/'; do
  if grep -q "$forbidden" <<<"$entries"; then
    echo "FAIL: wheel contains non-runtime content: $forbidden"
    exit 1
  fi
done

entry_file=$($python_bin - "$wheel" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as archive:
    name = next(n for n in archive.namelist()
                if n.endswith('.dist-info/entry_points.txt'))
    print(archive.read(name).decode())
PY
)
grep -q 'vllm.general_plugins' <<<"$entry_file"
grep -q 'bootstrap:register_plugins' <<<"$entry_file"
grep -q 'vllm_hust.extension_bundles' <<<"$entry_file"
grep -q 'org.vllm-hust.knorm = vllm_hust_knorm.manifests' <<<"$entry_file"
echo "PASS: vllm-hust-knorm wheel verified"
