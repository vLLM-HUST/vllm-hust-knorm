# vLLM-HUST KNorm

[English](README.md) | **简体中文**

面向 token 的 KNorm KV 缓存压缩插件，提供 CPU 参考运行时和候选 Ascend 后端。
各层、各 KV head 独立保留 K 向量范数最低的 token，前两层不压缩。
保留的 K/V 对紧致存放到选定的原有页中，各层独立分配和回收页。

**当前状态：实验性，仅支持导入注册（import-only）。** 算法和模拟运行时可在 CPU 上运行，
现有受保护的开发候选绑定面向固定 vLLM 0.30.0 / Ascend 0.30.0 源码，
参见[宿主环境准备](host-patches/README.md)。宿主契约可在无设备条件下测试，
原生 runner 初始化与设备验证仍待完成。安装插件不会安装旧压缩补丁。
即使不经过 ECPA 资格检查，`VLLM_KNORM_ENABLED=1` 也会被拒绝；
当前实现尚无受支持的推理服务启动命令。manifest 仍为 `0.3-experimental`，
实现状态为 `import_only`。

## 公开配置与 CPU 示例

在隔离环境中安装当前仓库：

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
```

```python
from vllm_hust_knorm import KNormConfig
from vllm_hust_knorm.runtime.engine import Simulation

# 以下数值仅用于小型 CPU 示例，并非经过模型评测的默认配置。
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

`Simulation` 使用调用者提供的 K/V 向量，不运行模型，也不采样输出。

| 参数 | 必填值 | 含义 |
| --- | --- | --- |
| `mode` | `"once"` 或 `"repeat"` | 在完整 prefill 后检查一次，或在每个 decode 边界检查。 |
| `prune_after` | 正整数 | 对各层当前有效 KV 长度应用的触发阈值。 |
| `keep_ratio` | `(0, 1]` 内的比例 | 每个 KV head 保留 `ceil(ratio * effective_length)` 个 token。使用十进制字符串可准确表达预期保留预算。 |

三个参数都必须显式指定。目前没有经过模型评测的默认值，
也不提供前缀保护窗口或跨 head、跨层的分数聚合选项。

### 从块级实现迁移

包根导出 `KNormConfig`（`N` 为大写）。旧名称 `KnormConfig` 会抛出带迁移说明的
`TypeError`，不再构造块级配置。包根的 `should_activate` 辅助函数会抛出
`RuntimeError`；`activate()` 是统一的宿主注册入口，目前仅注册供发现的 import-only 信息。
导入包不会加载 torch、vLLM 或历史实现。

旧参数 `compression_ratio`、`warmup_tokens`、`score_aggregation` 和 `norm_reduce_op`
无法自动映射为 token 级配置，新配置会拒绝这些参数。宿主注册会拒绝旧环境变量
`VLLM_KNORM_COMPRESSION_RATIO`、`VLLM_KNORM_WARMUP_TOKENS`、
`VLLM_KNORM_SCORE_AGGREGATION` 和 `VLLM_KNORM_NORM_REDUCE_OP`。
`VLLM_KNORM_ENABLED` 默认为 `0`；宿主验证完成前，将其设为 `1` 会触发
`import_only` 错误。

历史模块保留在 `knorm/` 下，用于来源追溯和回归测试。
公开配置与注册路径不会导入或安装这些模块。

## 目录结构

```text
src/vllm_hust_knorm/
├── config.py               显式 token 级配置
├── algorithm/              选择、保留预算与原页紧致填充
├── runtime/                状态、页所有权、事务与重放
├── backends/ascend/        候选设备操作与元数据
├── core/                   宿主检查与 import-only 注册
├── adapters/ascend/        固定宿主的规划、分配、worker 与重放绑定
├── adapters/vllm_hust/     宿主契约与保留的旧包装器
├── bootstrap.py            vllm.general_plugins 发现钩子
├── manifests/              静态扩展 manifest
└── knorm/                  历史块级实现
provenance/legacy-patches/  上游变更存档
```

## 开发与验证

```bash
pytest -q
ruff check .
ruff format --check .
python -m build --no-isolation
bash scripts/verify-wheel.sh dist/*.whl
```

以上命令用于验证 CPU 运行时和打包。
[文档索引](docs/index.md) 提供架构说明和历史设备证据的链接。
旧推理服务流程描述的是历史实现，其中的命令不能启用当前运行时。
设备上的质量、数值容差和性能需要在选定的宿主栈上单独验证。

## 历史设备证据

以下仅记录旧块级实现，不代表当前 token 运行时通过宿主或 NPU 验证。

2026-09-18 真机端到端验收（910B2，eager 模式；证据与日志路径见
[docs/how-to-run.md](docs/how-to-run.md) §11）：

| 项目 | 已验证值 |
| --- | --- |
| 插件 commit / 安装方式 | `45e9157`，editable（验收发布需换 wheel 重跑） |
| vLLM-HUST commit | `f18cf803c5`（detached；静态基线 `main@8344e107`） |
| vLLM-Ascend-HUST / triton-ascend | `17ed0571d`（`sync/upstream-main-20260908-latest`）/ `ef6c29210` |
| 设备 / 模型 | Ascend 910B2 单卡；Qwen2.5-14B-Instruct BF16 |
| 关键参数 | `--enforce-eager --gpu-memory-utilization 0.85 --max-model-len 8192 --no-enable-prefix-caching`，`VLLM_VERSION=0.23.1` |
| 安装无副作用 / 关闭对照 | READY 50s，HTTP 200，仅 bootstrap 标记 ✓ |
| 激活标记（manager + wrapper 同现） | `knorm-manager-registered`（5 spec）+ `knorm-wrapper-installed` ✓ |
| 请求 / 淘汰路径 | 短请求 HTTP 200；6640-token 长上下文触发淘汰零错误、摘要正确 ✓ |
| prefix-caching 互斥对照 | 告警出现、manager 不重定向、HTTP 200 ✓ |
| matched-baseline 吞吐（≥3 runs） | **待验证** |
| graph 模式 / 多卡 / 长稳 | **待验证** |

宿主栈注意事项：该组合需要 `_triton_compat.py` gluon 热修（宿主自身
bug，与插件无关；对照实验与重放步骤见 how-to-run.md §10）。

## 规范 MOD 元数据

仓库身份、直接负责的维护者、顾问状态、默认关闭的激活方式、回退、适用范围及证据资格
记录在 [`MOD_METADATA.json`](MOD_METADATA.json) 中。
`advisor_status: unknown` 不等同于已确认的 `none`。
性能声明仅适用于其中记录的工作负载和证据标签，不代表普遍的线上性能承诺。
