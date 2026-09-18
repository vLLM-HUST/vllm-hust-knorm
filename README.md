# vLLM-HUST KNorm

Owner-maintained KNorm KV-cache compression for vLLM-HUST, shipped as a
runtime-loadable extension bundle. KNorm (Devoto et al., 2024) evicts
KV-cache blocks whose keys have high L2 norms — they receive
disproportionately low attention during decoding, so evicting them
compresses the cache with minimal quality loss. The implementation is
pure PyTorch with no device-specific kernels; it has been verified on
Ascend 910B2 (eager mode) and makes no CUDA-specific assumptions, but
CUDA hosts are untested.

**Status: experimental (manifest `0.2-experimental`).** Serving-level
integration passed end-to-end on a real Ascend 910B2 (2026-09-18, eager
mode — see the verified-environment table below and
[HOST_CONTRACT.md](HOST_CONTRACT.md)). Not yet verified: graph mode,
matched-baseline throughput, multi-card. No non-dev release before
those pass.

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
pytest -q && ruff check . && ruff format --check .
```

See [docs/index.md](docs/index.md) for the full documentation index —
in particular [docs/how-to-run.md](docs/how-to-run.md)（服务器验证
workflow）、[docs/development.md](docs/development.md)、
[docs/architecture.md](docs/architecture.md) and
[docs/packaging-and-release.md](docs/packaging-and-release.md).

## 已验证环境

2026-09-18 真机端到端验收（910B2，eager 模式；证据与日志路径见
[docs/how-to-run.md](docs/how-to-run.md) §11）：

| 项目 | 已验证值 |
|---|---|
| 插件 commit / 安装方式 | `45e9157`，editable（验收发布需换 wheel 重跑） |
| vLLM-HUST commit | `f18cf803c5`（detached；静态基线 `main@8344e107`） |
| vLLM-Ascend-HUST / triton-ascend | `17ed0571d`（`sync/upstream-main-20260908-latest`）/ `ef6c29210` |
| 设备 / 模型 | Ascend 910B2 单卡；Qwen2.5-14B-Instruct BF16 |
| 关键参数 | `--enforce-eager --gpu-memory-utilization 0.85 --max-model-len 8192 --no-enable-prefix-caching`，`VLLM_VERSION=0.23.1` |
| 安装无副作用 / 关闭对照 | READY 50s，HTTP 200，仅 bootstrap 标记 ✓ |
| 激活标记（manager + wrapper 同现） | `knorm-manager-registered`（5 spec）+ `knorm-wrapper-installed` ✓ |
| 请求 / 淘汰路径 | 短请求 HTTP 200；6640-token 长上下文触发淘汰零错误、摘要正确 ✓ |
| prefix-caching 互斥对照 | 告警出现、manager 不重定向、HTTP 200 ✓ |
| matched-baseline 吞吐（≥3 runs） | **pending** |
| graph 模式 / 多卡 / 长稳 | **pending** |

宿主栈注意事项：该组合需要 `_triton_compat.py` gluon 热修（宿主自身
bug，与插件无关；对照实验与重放步骤见 how-to-run.md §10）。

