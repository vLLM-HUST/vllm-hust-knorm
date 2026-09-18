# Host contract

本插件定位为 vLLM-HUST（vllm v1 核心）的进程内 KV-cache 压缩扩展，
manifest 宿主标识为：

```json
{"host": {"provider": "vllm", "name": "vllm", "version_range": ">=0.23",
          "api_range": ">=1,<2"}}
```

## 宿主表面（六个幂等补丁的挂接点）

插件经 `vllm.general_plugins` 在**每个进程**（process 0、engine core、
worker）被加载，安装以下补丁（全部带 marker、可重入、默认惰性 ——
只有 `VLLM_KNORM_ENABLED=1` 时才改变行为）：

| # | 表面 | 作用 | 来源 |
| - | ---- | ---- | ---- |
| P1 | 空闲队列类（`FreeKVCacheBlockQueue`，旧构建为 `FreeQueue`）的队头插入 `prependleft_n`/`prepend_n` | 纯新增的队头插入（宿主已原生支持时跳过） | legacy PR #76 |
| P2 | `vllm.v1.core.block_pool.BlockPool.free_blocks(prepend=)` | 未缓存驱逐块回到空闲队列头部；宿主已原生支持时跳过包裹 | legacy PR #76 |
| P3 | `register_all_kvcache_specs`（定义模块 + 一切已导入的顶层绑定站点） | 注册后按 `should_activate()` 将 `FullAttentionSpec`（及同组的 `MLAAttentionSpec` 等）重定向到 `KnormFullAttentionManager`；prefix caching 开启时告警并保持原样 | legacy PR #134/#214 |
| P4 | `vllm.v1.core.sched.scheduler.Scheduler.update_from_output` | 把 `model_runner_output.knorm_block_scores` 路由进 manager 分数桥 | legacy PR #76 |
| P5 | `vllm.v1.worker.gpu_model_runner.GPUModelRunner.__init__` | 以 `should_activate()` 计算 `_knorm_active`（与 P3 同一真源，杜绝 legacy issue #163 的半启用态）；激活时在首个 forward 前安装注意力包装 | legacy PR #214 |
| P6 | `vllm.v1.worker.gpu_model_runner.GPUModelRunner.sample_tokens` | forward 后收集逐块分数并附着到输出 | legacy PR #76/#214 |

宿主 registry（`KVCacheSpecRegistry._ensure_registered`）以惰性 import
解析 `register_all_kvcache_specs`。**真机教训（910B2，2026-09-18）**：
`vllm.v1.engine.core` 在模块顶层 `from … import register_all_kvcache_specs`
并在 EngineCore 构造时调用这个顶层绑定——只包裹定义模块的属性会被
绕过（症状：`knorm-wrapper-installed` 出现而 `knorm-manager-registered`
缺失）。P3 因此扫描 `sys.modules`，对每个已导入且持有该符号的
`vllm.*` 模块重复包裹（fix `495a653`）。重注册经模块私有
`_REGISTRY_KVCACHESPEC_LIST`（pop-then-register，与 legacy 宿主测试同一
惯用法），因为公开 `register` 拒绝冲突更新。

另需 `vllm.v1.attention.backends.registry.AttentionBackendEnum`
（`CUSTOM` → `FLASH_ATTN` 回退解析 impl 类）与
`vllm.distributed.get_pp_group`（取 last rank；单进程缺失时按 last
rank 处理）。

## 已验证的宿主基线

- **静态/模拟验证**：vLLM-HUST `main@8344e10772e03d4c54c21f90335b7250ff776336`
  （knorm 已从树内移除、上述六表面齐全）。补丁行为由
  `tests/test_host_contract.py` 在复刻这些表面的模拟宿主上覆盖。
- **拒绝共存**：vLLM-HUST 0.23 seam 世代（seam commit
  `5c994cdc029dfebe318ca745a39920473033038b`）内树自带 `vllm.knorm`；
  本插件启动时检测到即 **fail closed**（拒绝激活并给出指引）。
- **NPU/真机验收（910B2，2026-09-18）**：host `vllm-hust@f18cf803c5`
  （detached，"sync vllm-project/vllm@bfb443a6"）+ `vllm-ascend-hust@17ed0571d`
  （branch `sync/upstream-main-20260908-latest`）+ triton-ascend
  `ef6c29210`，Qwen2.5-14B-Instruct BF16、`--enforce-eager`、
  `--gpu-memory-utilization 0.85`、max_model_len 8192。结果（日志
  `/tmp/vllm-knorm-*.log`，插件 `45e9157`）：安装无副作用 ✓、三标记
  同现 ✓、HTTP 200 ✓、**6640-token 长上下文触发淘汰路径零错误** ✓、
  关闭/prefix 对照 ✓。**未验**：graph 模式、matched-baseline 吞吐、
  多卡、分数透传的数值核查。历史基准（legacy #163 修复后
  123.25 tok/s）不可与本插件混谈。真机另发现宿主栈自身 bug：
  `vllm_ascend/_triton_compat.py` 对 triton <3.6 安装空 gluon stub，
  而本机 triton-ascend fork 的 runtime 硬依赖真实
  `gluon.nvidia`——已证与插件无关（卸载插件对照组同样崩溃），见
  how-to-run.md §10 的热修记录。

另：真机宿主的 `remove_skipped_blocks` 签名已含
`num_prompt_tokens`（R-SWA 中段淘汰语义，head-prefix 淘汰忽略之，
fix `45e9157`）；`FreeKVCacheBlockQueue` 原生 `prepend_n` 存在时
P1/P2 自动跳过（fix `15b1cc0`）。

对其他 commit 或发行版的兼容性**不应仅凭 `host.version_range` 推断**，
必须重跑集成测试。

## 行为边界

- 安装 ≠ 启用：`VLLM_KNORM_ENABLED` 插件侧默认 `0`（与树内模块默认
  `1` 的差异是刻意的插件姿态）；manifest 的
  `activation.environment` 由 `vllm-hust-ext run` 注入。
- prefix caching 与 Knorm 互斥（PR #134）：开启时保持宿主原生
  manager 并 `warning_once`。
- 异步调度（async scheduling）路径为尽力而为：分数附着在
  `sample_tokens` 返回对象上，是否被宿主异步管线透传取决于宿主版本；
  真机验收以同步调度为准。
- 卸载后必须重启进程才能回退（进程内补丁不支持热卸载）。
- **可观测性标记**（服务器验证依据，见 docs/how-to-run.md §5）：
  bootstrap 打 `runtime patches registered`；P3 成功重定向打
  `knorm-manager-registered`；P5 安装包装打
  `knorm-wrapper-installed`。两个激活标记必须同现，只出现其一即
  半启用态（legacy issue #163 症状），设计上已被
  `should_activate` 单一真源排除。
