# vLLM-HUST KNorm

Token-level KNorm KV-cache compression with a CPU reference runtime and a
candidate Ascend backend. Each layer and KV head independently retains the
lowest key-norm tokens. The first two layers remain uncompressed. Surviving
K/V pairs are packed into selected original pages, with independent allocation
and reclamation per layer.

**Status: experimental, import-only.** The algorithm and simulated runtime can
be exercised on CPU. A concrete vLLM-HUST/Ascend host binding and device
validation remain outstanding. Installing the plugin does not install legacy
compression patches. `VLLM_KNORM_ENABLED=1` is rejected, including outside ECPA
qualification; there is no supported serving command for this implementation.
The manifest remains `0.3-experimental` with implementation status `import_only`.

## Public configuration and CPU example

Install the checkout into an isolated environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
```

```python
from vllm_hust_knorm import KNormConfig
from vllm_hust_knorm.runtime.engine import Simulation

# Illustrative values for a small CPU example, not evaluated model defaults.
config = KNormConfig(mode="once", prune_after=4, keep_ratio="0.5")
sim = Simulation(config, block_size=4)
request = sim.create("example", prompt_length=8)


def kv(layer, head, token, position):
    return ((float(position),), (float(token),))


sim.append(request, range(8), kv)
sim.controller.before_decode(request)
assert [layout.length for layout in request.layers.values()] == [8, 8, 4, 4]
sim.finish(request)
```

`Simulation` uses supplied K/V vectors; it does not run a model or sample output.

| Parameter | Required value | Meaning |
| --- | --- | --- |
| `mode` | `"once"` or `"repeat"` | Check once after complete prefill, or at each decode boundary. |
| `prune_after` | Positive integer | Trigger threshold applied to each layer's current effective KV length. |
| `keep_ratio` | Ratio in `(0, 1]` | Retain `ceil(ratio * effective_length)` tokens per KV head. Decimal strings preserve the intended budget. |

All three parameters are explicit. There are no evaluated model defaults,
prefix protection windows, or head/layer score aggregation options.

### Migration from the block-level implementation

The package root exports `KNormConfig` (capital `N`). The former `KnormConfig`
name raises `TypeError` with migration guidance; it no longer constructs a
block-level configuration. The root `should_activate` helper raises
`RuntimeError`; `activate()` is the shared host registration entry point and
currently only registers import-only discovery. Package import does not load
torch, vLLM, or the historical implementation.

The old parameters `compression_ratio`, `warmup_tokens`, `score_aggregation`,
and `norm_reduce_op` have no automatic token-level mapping. New configuration
rejects them. Host registration rejects the legacy environment variables
`VLLM_KNORM_COMPRESSION_RATIO`, `VLLM_KNORM_WARMUP_TOKENS`,
`VLLM_KNORM_SCORE_AGGREGATION`, and `VLLM_KNORM_NORM_REDUCE_OP`.
`VLLM_KNORM_ENABLED` defaults to `0`; setting it to `1` raises an `import_only`
error pending host validation.

Historical modules remain under `knorm/` for provenance and regression tests.
They are not imported or installed by the public configuration/registration path.

## Layout

```text
src/vllm_hust_knorm/
├── config.py               explicit token-level configuration
├── algorithm/              selection, budgets and original-page packing
├── runtime/                state, page ownership, transactions and replay
├── backends/ascend/        candidate device operations and metadata
├── core/                   host guards and import-only registration
├── adapters/vllm_hust/     host contracts and archived legacy wrappers
├── bootstrap.py            vllm.general_plugins discovery hook
├── manifests/              static extension manifest
└── knorm/                  historical block-level implementation
provenance/legacy-patches/  archived upstream changes
```

## Development and validation

```bash
pytest -q
ruff check .
ruff format --check .
python -m build --no-isolation
bash scripts/verify-wheel.sh dist/*.whl
```

The commands above validate the CPU runtime and packaging.
[The documentation index](docs/index.md) links to architecture and historical
device evidence. The old serving workflow
and host contract describe the historical implementation; their commands do
not enable the current runtime. Device quality, numerical tolerances and
performance require separate validation on a selected host stack.

## Historical device evidence

以下仅记录旧块级实现，不代表当前 token 运行时通过宿主或 NPU 验证。

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

## Canonical MOD metadata

Repository identity, directly responsible maintainers, advisor status, default-off
activation, rollback, scope, and evidence qualification are recorded in
[`MOD_METADATA.json`](MOD_METADATA.json). `advisor_status: unknown` is not the
same as confirmed `none`. Performance statements remain limited to the workloads
and evidence labels recorded there; they are not general online claims.
