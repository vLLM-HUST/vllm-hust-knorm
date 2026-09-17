# vLLM-HUST KNorm

Owner-maintained KNorm KV-cache compression for vLLM-HUST, shipped as a
runtime-loadable extension bundle. KNorm (Devoto et al., 2024) evicts
KV-cache blocks whose keys have high L2 norms — they receive
disproportionately low attention during decoding, so evicting them
compresses the cache with minimal quality loss. The implementation is
pure PyTorch and works on CUDA GPUs and Ascend NPUs.

**Status: experimental (manifest `0.2-experimental`).** The runtime
integration is verified statically and against a simulated host; NPU
end-to-end acceptance is pending and recorded in
[HOST_CONTRACT.md](HOST_CONTRACT.md) before any release.

Migrated from the archived vLLM-HUST tree (legacy PRs #76/#134/#214 —
see [PROVENANCE.md](PROVENANCE.md)); as a plugin it is **opt-in**:
installing it does not change host behavior until Knorm is explicitly
enabled.

## Layout

```text
src/vllm_hust_knorm/
├── bootstrap.py            vllm.general_plugins 动态加载入口
├── core/                   宿主探测、守卫、统一激活管线
├── knorm/                  config / norms / eviction（纯逻辑，CPU 可测）
│   ├── attention_backend / hooks / manager（torch 与宿主路径，惰性导入）
├── adapters/vllm_hust/     六个幂等宿主补丁（见 HOST_CONTRACT.md）
└── manifests/              vllm-hust-extension-v0.2.json
provenance/legacy-patches/  三个 legacy PR 的逐 commit patch 存档
```

## Quick start

```bash
pip install vllm-hust-knorm vllm-hust-ext
vllm-hust-ext extension list                      # org.vllm-hust.knorm
vllm-hust-ext extension enable org.vllm-hust.knorm
vllm-hust-ext run --dry-run -- vllm serve MODEL --no-enable-prefix-caching
vllm-hust-ext run -- vllm serve MODEL --no-enable-prefix-caching
```

Without the extension manager:

```bash
pip install vllm-hust-knorm
VLLM_KNORM_ENABLED=1 VLLM_KNORM_COMPRESSION_RATIO=0.5 \
  vllm serve MODEL --no-enable-prefix-caching
```

Prefix caching and KNorm are mutually exclusive (legacy PR #134): with
`--enable-prefix-caching` the plugin keeps the stock
`FullAttentionManager` and logs a warning.

### Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `VLLM_KNORM_ENABLED` | `0` | Master switch (plugin posture: opt-in). |
| `VLLM_KNORM_COMPRESSION_RATIO` | `0.5` | Fraction of blocks to **keep**; `1.0` disables compression. |
| `VLLM_KNORM_WARMUP_TOKENS` | `32` | Attention-sink tokens never evicted. |
| `VLLM_KNORM_SCORE_AGGREGATION` | `min` | Token→block score aggregation (`min`/`mean`/`max`). |
| `VLLM_KNORM_NORM_REDUCE_OP` | `mean` | Head-norm reduction (`mean`/`max`/`sum`). |

## Rollback

```bash
vllm-hust-ext extension disable org.vllm-hust.knorm
# stop old processes, start new ones, then:
vllm-hust-ext extension forget org.vllm-hust.knorm
pip uninstall vllm-hust-knorm
```

## Development

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
pytest -q && ruff check .
```

See [docs/how-to-run.md](docs/how-to-run.md),
[docs/architecture.md](docs/architecture.md) and
[docs/packaging-and-release.md](docs/packaging-and-release.md).
