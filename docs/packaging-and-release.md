# Packaging and release

Follows the vLLM-HUST guide
([bidkv-packaging-and-release-guide](https://github.com/vLLM-HUST/vllm-hust-docs/blob/main/operations/bidkv-packaging-and-release-guide.md))
and the extension author guide's test matrix; the concrete workflow is
adapted from the reference plugin
[vllm-ascend-quantized-kv-cache-hust](https://github.com/vLLM-HUST/vllm-ascend-quantized-kv-cache-hust).

## Versioning

- Single source: `src/vllm_hust_knorm/_version.py`
  (`__version__ = "0.1.0"`).
- Must equal `extension_version` in
  `src/vllm_hust_knorm/manifests/vllm-hust-extension-v0.2.json`
  (enforced by `tests/test_manifest.py`).
- PyPI forbids re-uploading a version: bump after every published code
  change; the release workflow reads the version from metadata, never
  hardcodes it.

## Build and artifact verification (local)

```bash
git status --short          # record a clean commit
git rev-parse HEAD
rm -rf dist
python -m build --no-isolation          # or: uv build --no-sources --out-dir dist
bash scripts/verify-wheel.sh dist/*.whl
python -m zipfile -l dist/vllm_hust_knorm-0.1.0-py3-none-any.whl
```

The wheel must contain `manifests/vllm-hust-extension-v0.2.json`,
`manifests/__init__.py`, dist-info and both entry-point groups
(`vllm.general_plugins`, `vllm_hust.extension_bundles`).

Smoke test in an isolated venv (no host, no device):

```bash
python -m venv .release-smoke
.release-smoke/bin/python -m pip install --no-deps dist/*.whl
.release-smoke/bin/python - <<'PY'
import json, pathlib, importlib.metadata as im
import vllm_hust_knorm, vllm_hust_knorm.manifests as m

assert im.version("vllm-hust-knorm") == vllm_hust_knorm.__version__
eps = im.entry_points()
assert [e.name for e in eps.select(group="vllm_hust.extension_bundles")] \
    == ["org.vllm-hust.knorm"]
manifest = json.loads((pathlib.Path(m.__file__).parent
    / "vllm-hust-extension-v0.2.json").read_text())
assert manifest["extension_version"] == vllm_hust_knorm.__version__
print("smoke OK")
PY
```

## Test matrix before a release

- `pytest -q` (manifest/packaging/config/eviction/activation/host
  contract, incl. the 8-combination activation matrix).
- `ruff check .`
- Clean-install gate from the extension guide §14.3: fresh venv,
  install manager wheel + plugin wheel, `extension validate`, `enable`,
  `run --dry-run`, `disable`, `forget`, `pip uninstall`.
- NPU end-to-end acceptance with matched baseline; record commit/wheel
  hash and evidence in HOST_CONTRACT.md. **No non-dev PyPI release
  before this passes** (manifest schema `0.2-experimental` gate).

## Publish

CI (`.github/release.yml`) on a `v*` tag:

1. build job: record provenance, `uv build --no-sources`, wheel
   verification, artifact upload;
2. publish job (environment `pypi`): `uv publish --check-url
   https://pypi.org/simple/vllm-hust-knorm/` with `UV_PUBLISH_TOKEN`
   (per-project token, username `__token__`, `pypi-` prefix), then a
   no-cache smoke install of the exact tagged version.

Token handling: CI secret only; never in the repo, command history or
logs; unset after use. Files on PyPI are immutable — a bad release is
fixed by a new version or a yank per policy.
