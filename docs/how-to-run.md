# 服务器验证 workflow（How to run）

在 Ascend 910B（或 CUDA GPU）服务器上对 `vllm-hust-knorm` 做端到端
验证。目标：证明"插件被宿主真正加载、KNorm 真正生效、可关闭、可
回退、性能结论有 matched baseline"——"进程启动成功"不算验收。

> 静态/模拟验证与宿主基线约束见 [HOST_CONTRACT.md](../HOST_CONTRACT.md)；
> 本文档跑通后，把证据（commit、日志摘录、吞吐）回填其"已验证的
> 宿主基线"一节。

## 1. 环境准备

```bash
# 宿主环境：knorm-free 的 vllm-hust（见 HOST_CONTRACT.md 基线），
# 例如 conda 环境 vllm-hust-dev
conda activate vllm-hust-dev

# 在宿主所在环境安装本插件（开发装法）
cd /path/to/vllm-hust-knorm
python -m pip install -e .

# 或安装构建产物（验收必须用 wheel，见 packaging-and-release.md）
python -m pip install --no-cache-dir dist/vllm_hust_knorm-*.whl
```

确认动态插件入口存在：

```bash
python -c "from importlib.metadata import entry_points; \
  print([e for e in entry_points(group='vllm.general_plugins') if e.name == 'vllm-hust-knorm'])"
```

## 2. 安装无副作用（必测）

```bash
# 不设置任何 VLLM_KNORM_* 环境变量启动
vllm serve /path/to/model --max-model-len 8192 2>&1 | tee /tmp/vllm-knorm-off.log
```

成功标准：服务正常、请求正常、日志**没有**
`knorm-manager-registered` / `knorm-wrapper-installed` 标记——插件
默认不改变宿主行为（`VLLM_KNORM_ENABLED` 缺省 0）。

## 3. 进程内快速验证（可选，不起服务）

```bash
python -c 'from vllm_hust_knorm.bootstrap import register_plugins; print(register_plugins())'
# 期望输出包含: [vllm-hust-knorm] runtime patches registered ... 返回 ['knorm']
```

若报 `refuses to activate: ... in-tree 'vllm.knorm'`，说明宿主自带
树内 knorm（0.23 seam 世代），按 HOST_CONTRACT.md 换 knorm-free 宿主。

## 4. 启动 KNorm 服务

先 `--enforce-eager` 排除图捕获变量：

```bash
VLLM_KNORM_ENABLED=1 \
VLLM_KNORM_COMPRESSION_RATIO=0.5 \
VLLM_KNORM_WARMUP_TOKENS=32 \
VLLM_LOGGING_LEVEL=INFO \
vllm serve /path/to/model \
  --served-model-name knorm-model \
  --no-enable-prefix-caching \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.6 \
  --enforce-eager \
  2>&1 | tee /tmp/vllm-knorm-eager.log
```

通过后再去掉 `--enforce-eager` 复跑（Ascend ACL Graph / CUDA Graph
路径），日志存 `/tmp/vllm-knorm.log`。

## 5. 日志验证（三个标记 + 一条禁项）

```bash
rg -n 'vllm-hust-knorm|knorm-manager-registered|knorm-wrapper-installed|\
prefix caching is enabled|ERROR|Traceback' /tmp/vllm-knorm.log
```

成功标准：

| 检查项 | 期望 |
|---|---|
| `runtime patches registered` | 出现（bootstrap，每个进程一次） |
| `knorm-manager-registered` | 出现（scheduler 进程，列出被重定向的 spec，如 `FullAttentionSpec, MLAAttentionSpec`） |
| `knorm-wrapper-installed` | 出现（worker 进程，注明包装的 attention impl 类） |
| `Knorm KV compression is disabled because prefix caching` | **不出现**（本实验 prefix caching 已关） |
| `ERROR` / `Traceback` | 无 KNorm 相关条目 |

若 `knorm-manager-registered` 未出现但 `knorm-wrapper-installed`
出现（或反之），即半启用态——立即停服并提 issue（这正是 legacy
issue #163 的症状，本插件设计上不可能出现）。

## 6. 请求验证

另一终端：

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "knorm-model",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0,
    "max_tokens": 64
  }' -w '\nHTTP %{http_code}\n'
```

成功标准：HTTP 200，输出为连贯文本。再用长上下文请求（数 k token
prompt + 长 max_tokens）确认压缩生效后生成质量仍可接受——KNorm 的
质量主张（Devoto et al., 2024）以任务评测为准，不接受"能出 token
即通过"。

## 7. 关闭与回退对照（必测）

```bash
# 7.1 显式关闭：同命令但 VLLM_KNORM_ENABLED=0 → 新进程
VLLM_KNORM_ENABLED=0 vllm serve /path/to/model \
  --no-enable-prefix-caching --max-model-len 32768 \
  2>&1 | tee /tmp/vllm-knorm-disabled.log
# 期望：无任何 knorm-* 标记，行为与第 2 步一致

# 7.2 prefix caching 互斥：开启 prefix caching 且 Knorm 开着
VLLM_KNORM_ENABLED=1 vllm serve /path/to/model \
  --enable-prefix-caching --max-model-len 32768 \
  2>&1 | tee /tmp/vllm-knorm-prefix.log
# 期望：日志出现 "Knorm KV compression is disabled because prefix
# caching is enabled"，且无 knorm-manager-registered 标记
```

## 8. 吞吐对照（matched baseline，验收核心）

按 legacy issue #163 的复验协议，**同模型、同请求集、同并发、同输入
输出长度、同硬件、同启动参数**，base（`VLLM_KNORM_ENABLED=0`）与
knorm 各 ≥3 次独立运行，同卡空闲时段执行：

```bash
# 用 vllm-hust 基准工具或 benchmark_serving；关键固定项示例：
#   --gpu-memory-utilization 0.6 --max-model-len 32768 --enforce-eager
#   相同 random-seed / request-count / input-len / output-len / 并发
```

记录 `output_throughput`、`total_output_tokens` 与输出正确性（token
数应与 base 同量级——若 knorm 侧输出 token 数腰斩，是质量回归信号，
参见 legacy issue #163 的"early EOS"事故）。历史参考（910B2，
legacy #163 修复后）：base 同源对照 output_throughput 123.25 tok/s、
total_output_tokens 25124。结果回填 HOST_CONTRACT.md。

## 9. 卸载与回滚

```bash
# 停服后（进程内补丁不支持热卸载）：
pip uninstall vllm-hust-knorm
# entry point 随包消失，vllm 不再调用钩子；
# 残留的 VLLM_KNORM_ENABLED=1 成为无害 no-op，行为回到装包前。
vllm serve /path/to/model   # 复跑第 2 步确认
```

Extension manager 路径的 disable/forget 流程见
[README.md](../README.md)。

## 10. 故障排查（事故记录风格）

| 症状 | 根因 | 处置 |
|---|---|---|
| `refuses to activate ... in-tree 'vllm.knorm'` | 宿主自带树内 knorm（0.23 seam 世代），双实现冲突 | 换 knorm-free 宿主构建；不要强行共存 |
| `requires host surfaces that are not importable: ...` | 宿主版本缺补丁挂接点（见列表） | 核对 HOST_CONTRACT.md 基线；缺表面的宿主不支持，勿改宽 version_range 绕过 |
| 无 `runtime patches registered` | entry point 未注册 / `VLLM_PLUGINS` 白名单未含本插件 | 检查第 1 步命令输出；`VLLM_PLUGINS=vllm-hust-knorm` 显式放行 |
| `knorm-manager-registered` 缺失但 env 已开 | `--enable-prefix-caching` 忘了关，或 ratio=1.0 | 看日志是否走了 prefix 告警分支；检查启动参数 |
| 半启用态（两标记只出现其一） | 理论上不可能（should_activate 单一真源） | 立即停服提 issue，附完整日志与两个 commit（宿主+插件） |
| 请求输出 token 数异常腰斩 | 压缩质量回归 / early EOS 复发（legacy #163 症状） | 停用并提 issue；用第 8 步协议复现并记录 |
| `VLLM_KNORM_COMPRESSION_RATIO` 等取值报错 | env 值非法（fail-closed 校验） | 按报错信息修正取值范围（ratio ∈ (0,1]，见 README 表） |
