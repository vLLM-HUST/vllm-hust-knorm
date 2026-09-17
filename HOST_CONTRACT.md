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
| P1 | `vllm.v1.core.kv_cache_utils.FreeQueue.prependleft_n` | 纯新增的队头插入（宿主缺失时补上） | legacy PR #76 |
| P2 | `vllm.v1.core.block_pool.BlockPool.free_blocks(prepend=)` | 未缓存驱逐块回到空闲队列头部；宿主已原生支持时跳过包裹 | legacy PR #76 |
| P3 | `vllm.v1.core.single_type_kv_cache_manager.register_all_kvcache_specs` | 注册后按 `should_activate()` 将 `FullAttentionSpec`（及同组的 `MLAAttentionSpec` 等）重定向到 `KnormFullAttentionManager`；prefix caching 开启时告警并保持原样 | legacy PR #134/#214 |
| P4 | `vllm.v1.core.sched.scheduler.Scheduler.update_from_output` | 把 `model_runner_output.knorm_block_scores` 路由进 manager 分数桥 | legacy PR #76 |
| P5 | `vllm.v1.worker.gpu_model_runner.GPUModelRunner.__init__` | 以 `should_activate()` 计算 `_knorm_active`（与 P3 同一真源，杜绝 legacy issue #163 的半启用态）；激活时在首个 forward 前安装注意力包装 | legacy PR #214 |
| P6 | `vllm.v1.worker.gpu_model_runner.GPUModelRunner.sample_tokens` | forward 后收集逐块分数并附着到输出 | legacy PR #76/#214 |

宿主 registry（`KVCacheSpecRegistry._ensure_registered`）以惰性 import
解析 `register_all_kvcache_specs`，而 general plugins 在首次注册之前
加载，因此模块属性包裹生效；重注册经模块私有
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
- **NPU/真机验收**：pending。目标 910B2（历史基准：修复后
  output_throughput 123.25 tok/s、total_output_tokens 25124，
  gpu_memory_utilization=0.6、max_model_len=32768 —— 见 legacy
  issue #163 的 matched-baseline 记录）。真机通过前不发布非 dev 版本。

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
