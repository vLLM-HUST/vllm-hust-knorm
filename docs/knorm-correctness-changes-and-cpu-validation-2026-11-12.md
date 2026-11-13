# KNorm 重构与正确性修复进展报告

报告日期：2026-11-12。代码分支：`fix/knorm-correctness`。验证版本：`fb5ff2727652d2bf0ae3976c8e52fc16d585cdbc`。

## 当前进展

已完成 token 级压缩语义重构、共享物理页池与逐层分配管理、带确认的完整块表同步、异常重试与历史重放，并建立了固定 vLLM 0.30.0 / Ascend 0.30.0 的候选插件绑定。本次新增的 CPU 对照实验复现了原有同步故障，并验证了新实现的修复效果。真实 Ascend 端到端验证尚未完成。

**本次实现通过了针对 issue #3 核心故障机制的 CPU 回归：完整块表替换、worker 确认、确认后释放旧页，以及旧页复用后不再被原请求引用。** 实验不仅检查最终块表相等，还检查实际 K/V 内容、逐 head 保留集合、确认丢失后的重试，以及错误版本回执不能触发提前释放。

本次验证覆盖插件事务和 worker 控制代码，宿主块表对象及 RPC 传输采用 CPU 测试替身。真实 Ascend 运行环境中的端到端正确性、模型任务得分和性能评测列为下一阶段工作。

## 1. 比较对象和 issue 状态

| 对象 | 固定版本 | 含义 |
| --- | --- | --- |
| 原插件化实现 | `1efd0103a348ba63716341b9d25141780e4689de` | token 级重构前的基线，保留块级淘汰策略。 |
| 本报告验证对象 | `fb5ff2727652d2bf0ae3976c8e52fc16d585cdbc` | 包含 `57b0b31` 的 token 级语义重构和 `fb5ff27` 的候选 Ascend 宿主绑定。 |
| issue 原始复现 | `vllm-hust@363794235231c032de407fbe759d693962ced81a` | 原宿主内实现；其故障机制由旧插件化 manager 延续。 |

[Issue #3](https://github.com/vLLM-HUST/vllm-hust-knorm/issues/3) 的问题是：scheduler 把已淘汰块的位置改为 null block 并释放物理块，但非恢复请求的 worker 只追加新块，已有位置没有替换。物理块号被复用后，旧引用继续存在。

截至本次核查，issue 确实已经关闭，但关闭证据属于另一条代码路径：

- [2026-11-08 的进展评论](https://github.com/vLLM-HUST/vllm-hust-knorm/issues/3#issuecomment-6726980195) 记录 PR #9（合并提交 `42475fcbd9ec79d4adcab26389d91a76e0c4bd7b`）通过整数列表中的 marker 传递完整块表，并保留普通请求原有更新方式。
- [2026-11-09 的关闭评论](https://github.com/vLLM-HUST/vllm-hust-knorm/issues/3#issuecomment-6738691234) 及 [PR #10](https://github.com/vLLM-HUST/vllm-hust-knorm/pull/10) 记录 Ascend Model Runner V2 适配，合并提交为 `58ca9d9167e4cd496f25282136ee2937d4ca5c59`。贡献者报告了 Qwen3-8B 在 910B2 上普通调度和异步调度的验证。

**issue 的关闭对应上述另一组提交，不能作为 `fb5ff27` 的设备验证证据。** 本报告使用下面独立运行的 CPU 实验说明当前分支的修复效果，不沿用其他版本的 NPU 结论。

## 2. 分支价值与定位

两条路径解决的问题范围不同。关闭 issue 的提交在原块级实现上补齐了同步和 Ascend runner 适配；当前分支进一步重构压缩语义、物理布局和恢复机制。固定到 `58ca9d9` 的源码仍调用块级淘汰 planner、将淘汰位置替换为 null block，并在评分阶段跨层平均后聚合成块分数。这一判断来自实际源码，而不只是 PR 标题：[manager](https://github.com/vLLM-HUST/vllm-hust-knorm/blob/58ca9d9167e4cd496f25282136ee2937d4ca5c59/src/vllm_hust_knorm/knorm/manager.py)、[hooks](https://github.com/vLLM-HUST/vllm-hust-knorm/blob/58ca9d9167e4cd496f25282136ee2937d4ca5c59/src/vllm_hust_knorm/knorm/hooks.py)。

| 项目目标 | 关闭 issue 的块级路径解决了什么 | 当前分支增加的工作 |
| --- | --- | --- |
| 消除旧表引用 | 补齐完整块表更新，并有公开记录的设备验证。 | 在新的 token 级布局下独立完成同步、确认与页复用回归。 |
| 每层、每 head 独立保留 token | 修复同步后，原块级选择粒度仍然存在。 | 取消跨层、跨 head 的选择聚合，直接实现各自的低范数 token 集合。 |
| 精确执行保留比例 | 仍以块级淘汰策略决定保留集合。 | 按实际有效长度计算 token 预算，再按物理页取整；支持一次和重复压缩。 |
| 支撑逐层不同的缓存历史 | issue 的关闭不能证明另一种内存布局已实现。 | 实现共享页池、逐层长度与写入位置、原页填洞，以及带压缩事件的历史重放。 |

一个具体例子是本次实验中的两个 head：head 0 应保留原位置 128–255，head 1 应保留 0–127。跨 head 聚合成一个块分数，再统一保留或删除整块，无法表达这两套不同的选择；即使块表同步完全正确，也不会自动得到这种 token 级行为。当前分支为实现这一目标进行了必要的布局和调度改造。

因此，**如果项目目标只是修复旧块级实现的同步 bug，已经合并的修复可以作为可用的解决路径，无须以同一 bug 为理由重复重构；如果目标是实现上述逐层、逐 head 的 token 级 KNorm，那么本分支仍是独立且必要的实现工作。** Issue 关闭不等于这一更完整的算法目标已经完成。

当前分支的阶段价值是：把目标算法语义、物理页回收和异常恢复实现为可测试的代码，并已经取得 CPU 正确性证据。它的成熟度尚不能等同于已记录设备验证的路径。逐 token 选择、K/V 搬移、位置元数据及控制同步也会带来开销，**目前没有证据表明该分支吞吐更高或任务得分更好**。下一步应在相同模型、数据集和压缩预算下评估任务得分、实际 KV 占用、吞吐与 TPOT，判断细粒度选择的收益是否抵消额外成本。已有同步修复和测试经验可以作为参考，不需要否定或重复计算其贡献。

## 3. 相较原插件化实现的主要改动

| 方面 | 原实现 | 本次实现及影响 |
| --- | --- | --- |
| 选择粒度 | 先归约 head 范数、跨层平均，再聚合成块分数，按块淘汰。 | 每层、每个 **KV head** 独立选择低 K 范数 token，并保留配对 V；head 之间可以保留不同原始 token。 |
| 前两层与保护窗口 | 保护序列开头的 warmup 块，参与评分的层被聚合。 | 层 0、1 完全不压缩；其余层不设置固定前缀或近期 token 保护窗口。保护层和保护序列位置是不同概念。 |
| 保留预算 | 用已计算 token 数换算块数；旧 planner 在前缀候选区间淘汰，受 warmup 与块粒度影响。 | 用该层当前有效 KV 长度 `N_l` 计算 `K_l = ceil(keep_ratio × N_l)`，再单独计算物理页数 `ceil(K_l / block_size)`。 |
| 触发时机 | 在宿主管理器的 skipped-block 清理路径中执行。 | `once` 在完整 prefill 后、第一次 decode 前检查一次；`repeat` 在每次 decode 前按当前有效长度检查，同一边界最多一次。`prune_after` 是触发阈值，不是永久保护长度。 |
| 物理布局 | 在原逻辑表中填 null block，留下空洞。 | 各层独立块表与有效长度；优先保留幸存项较多的原物理页，逐 head 填洞，使有效 KV 占据紧致前缀。同层 heads 共用页表，但槽位对应的原 token 可以不同。 |
| 页池 | 原宿主块分配、归还路径。 | 所有层使用一个共享物理页池，按请求实例和层管理所有权；页代际用于识别块号复用后的旧引用。 |
| 块表同步 | 淘汰位置没有通过非 resumed 请求的 append-only 协议传到 worker。 | 独立控制路径发送所有层的完整 layout；替换 worker 请求状态、清空并重建块表行，确认完成后才回收旧页。 |
| 批内行空洞 | 原实现没有本轮新增的完整替换过程。 | 完整替换复制到最后一个活跃行，不能只复制 `num_reqs` 行；同时清除较长旧行的尾部，避免留存旧页号。 |
| 失败与重试 | 旧淘汰流程直接修改表并归还页，没有新的事务确认协议。 | PLAN/APPLY/COMMIT 分阶段；确认携带请求实例、事务、版本和状态。APPLY 前可取消，开始搬移后采用向前重试或终止请求，不能假装回滚旧 KV。 |
| 位置与恢复 | 历史已计算长度与含空洞的块表相关联。 | 区分原始语义位置和逐层压缩后的物理位置；抢占恢复重放已知输入及压缩事件，不重新采样已输出 token。 |
| 宿主与启用 | 历史代码针对旧 vLLM-HUST 接口；基线当时也已有限制启用的保护措施。 | 候选接入固定 vLLM 0.30.0 / Ascend 0.30.0。运行时包装放在插件内；另外保留两项宿主依赖声明补丁。公开入口继续 `import_only`，并非已提供可正式启用的服务插件。 |

旧路径仍保留在 `src/vllm_hust_knorm/knorm/` 中，用于来源追踪与历史回归；公开注册路径不会安装它。本次负对照使用的 `knorm/manager.py` 与 `1efd010` 中的内容相同，SHA-256 为 `567707f15273d1de83cc7399f2998326f4d101704a9607c8a857bdc5db80b5a0`。

算法规则符合性、块表同步正确性和模型任务得分是三个独立的验证目标；本次验证前两者的所列 CPU 路径，模型任务得分需要另行评测。

代码入口：[旧淘汰逻辑](../src/vllm_hust_knorm/knorm/eviction.py)、[旧范数聚合](../src/vllm_hust_knorm/knorm/hooks.py)、[新选择逻辑](../src/vllm_hust_knorm/algorithm/selection.py)、[保留原页与填洞](../src/vllm_hust_knorm/algorithm/packing.py)、[事务](../src/vllm_hust_knorm/runtime/transaction.py)、[worker 控制与回执](../src/vllm_hust_knorm/adapters/ascend/worker.py)、[块表替换](../src/vllm_hust_knorm/adapters/ascend/metadata.py)。

## 4. 修复原问题所依赖的执行顺序

当前事务的关键顺序是：

1. 阻止请求在压缩期间继续 forward，并等待此前操作完成。
2. 固定保留 token、搬移计划和目标 layout，搬移配对的 K、V 及原位置记录。
3. 向 worker 安装**所有层的完整 layout**，更新请求状态与各 cache group 的块表行，等待同步及 APPLIED 回执。
4. 更新 scheduler 的 layout 和宿主管理器状态，核对 worker 的 COMMITTED 回执。
5. 只有上述确认完成，才将不再引用的物理页放回共享池。

确认前，两侧可以处于受控的过渡状态，但此时禁止 forward，也不能复用旧页；因此不要求事务内部每个瞬间都完全相等。真正的约束是：**允许后续 forward 或旧页复用之前，所有消费者必须已经确认相同的新映射。**

## 5. 本次实际运行的 CPU 实验

### 5.1 旧路径负对照：复现原 issue 的状态差异

执行当前仓库保留的旧 `KnormFullAttentionManager.remove_skipped_blocks()`，使用已有的 fake-host 基类和分配池，设置 8 个块、块大小 128、1024 token、比例 0.5。随后明确模拟原 worker 的非 resumed 追加操作。

实际输出：

```text
scheduler final: [1, 0, 0, 0, 5, 6, 7, 8, 2]
worker final:    [1, 2, 3, 4, 5, 6, 7, 8, 2]
mismatches:     [1, 2, 3]
recycled page:  2
```

这与 issue 的关键状态一致：块 2 已经成为追加的新块，却仍被 worker 当作旧位置的块引用。负对照预期出现不一致，实验断言确实捕获到了它。

此项是旧 manager 加协议模型的 CPU 复现，没有启动旧版 vLLM 服务，也没有模拟其完整异步时间线。

### 5.2 当前路径：真实插件控制代码，加 CPU 块表与 RPC 替身

实际调用的插件代码包括 `Transactions`、`RemoteCache`、`RemoteWorker`、`WorkerControl`、`AscendCache`、张量选择和搬移代码，以及 `replace_block_tables()`。只把宿主表对象、scheduler 宿主管理器安装动作、RPC 传输和设备同步替换为明确的 CPU 测试实现；页池使用仓库的 `PagePool`。

实验配置：

- 48 层、每层 2 个 KV head、向量维度 2、FP16 KV、块大小 128、初始每层 256 token。这是受控测试数据，**不是 Qwen2.5-14B 的完整模型几何或推理**。
- 总计 160 个物理页，页 0 保留，最初占用 96 页，空闲 63 页。
- 层 0、1 保留 256 token；其余 46 层压缩为每 head 128 token。head 0 的低范数集合为原位置 128–255，head 1 为 0–127，用于强制验证不同 head 的独立选择及 K/V 配对。
- 唯一活跃请求放在 batch 行 2，`num_reqs=1`，故意构造前方存在空行的情况。CPU 表和“设备表”用两份独立的 CPU 张量表示，检测是否确实复制了目标行并清空旧尾部。
- 旧页全部归还后，把所有空闲页分配给另一个请求实例，并将 K、V、位置全部覆写为 `-999`。这保证 46 个已释放页都被实际复用，而不只是检查空闲计数。

在正常执行之外，分别注入安装确认丢失、提交确认丢失、安装回执版本错误、提交回执版本错误。每次都检查：确认失败时没有旧页被释放；重试成功后两侧完整表一致；页复用后旧请求没有读到外来内容；重复执行已提交事务不会再次搬移或释放。

实际结果：

| 情景 | 有效确认前空闲页 | 提交后空闲页 | 释放且被另一实例复用的页 | 48 组完整表及尾部检查 | 外来 KV 读取 | attention 最大绝对误差 |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| 正常执行 | 63 | 109 | 46 | 通过 | 0 | 0.0 |
| 安装确认丢失后重试 | 63 | 109 | 46 | 通过 | 0 | 0.0 |
| 提交确认丢失后重试 | 63 | 109 | 46 | 通过 | 0 | 0.0 |
| 安装回执版本错误后重试 | 63 | 109 | 46 | 通过 | 0 | 0.0 |
| 提交回执版本错误后重试 | 63 | 109 | 46 | 通过 | 0 | 0.0 |

attention 检查从 worker 的“设备表”镜像读取压缩后 KV，对每层每 head 计算单 query 的 `softmax(QKᵀ / sqrt(d))V`；参考值直接由预先确定的幸存位置集合构造，**没有调用生产 `select()` 生成参考答案**。每个情景比较 96 次，五个情景合计 480 次，均完全一致。实际计算使用 CPU float64，KV 输入仍为 FP16。

这里比较的是**相同保留集合的物理布局与独立参考计算**，不要求压缩后的 attention 等于未压缩 attention，也没有证明模型输出文本或任务得分保持不变。此单 query 检查不包含额外的位置掩码、滑动窗口或 NPU attention kernel。

### 5.3 现有回归测试复跑

| 本次执行 | 结果 | 说明 |
| --- | --- | --- |
| 独立临时环境运行完整 `pytest -q -ra` | **118 passed, 3 skipped，4.15 秒** | 包括打包与安装态检查；跳过固定真实宿主模块及两项需要 torch 的测试。 |
| 已有 torch 环境运行 `tests/test_token_core.py` | **33 passed，1.01 秒** | 包括 tensor 选择、随机 packing、独立整数 oracle、事务故障、页复用、取消、历史重放和触发时机。 |
| 补跑旧 attention wrapper 的 torch 测试 | **1 passed, 21 deselected，1.92 秒** | 仅补足 torch 环境覆盖，不把历史 wrapper 测试当作新宿主绑定证明。 |

上述用例有重叠，**不能把通过数量相加当作独立用例总数**。完整套件中的固定真实宿主模块这次没有运行；本报告不把测试替身结果等同于该模块或真实设备验证通过。

## 6. 复现方法与证据留存

普通套件使用独立 Python 3.12.0 环境、pytest 9.2.0，并安装当前 checkout；打包测试还构建 wheel，在另一独立环境检查安装后的公开入口。专用实验使用 PyTorch `2.12.0+cu130`，全部张量与运算均在 CPU 上，没有加载模型。此环境不等同于目标宿主的 torch 2.10.0 / torch-npu 组合。

普通测试可在仓库根目录复跑：

```bash
python3 -m venv /tmp/knorm-cpu-review
/tmp/knorm-cpu-review/bin/python -m pip install -e '.[dev]'
/tmp/knorm-cpu-review/bin/python -m pytest -q -ra
```

以下命令在已安装 pytest 和 PyTorch 的环境中执行，工作目录为仓库根目录：

```bash
PYTHONPATH=src python -m pytest -q -ra tests/test_token_core.py
PYTHONPATH=src python -m pytest -q tests/test_host_contract.py -k wrapped_attention_forward_stores_norms
```

专用实验脚本完整保存在本报告附录中，避免依赖可能被清理的 `/tmp` 文件。可以从 Markdown 提取后运行：

```bash
python3 - <<'PY'
from pathlib import Path
document = Path('docs/knorm-correctness-changes-and-cpu-validation-2026-11-12.md').read_text()
section = document.rsplit('<!-- BEGIN CPU EXPERIMENT -->', 1)[1]
script = section.split('```python\n', 1)[1].split('\n```', 1)[0]
Path('/tmp/knorm-correctness-validation-20261112.py').write_text(script + '\n')
PY
PYTHONPATH=src python /tmp/knorm-correctness-validation-20261112.py > /tmp/knorm-correctness-validation-20261112.json
```

脚本应退出码为 0，JSON 中旧路径出现 `[1, 2, 3]` 的差异，当前五种情景全部得到 `full_tables_equal: true`、`foreign_kv_reads: 0`、`attention_max_abs_error: 0.0`。负对照出现差异是预期结果，不是当前代码测试失败。

测试汇总和实验指标已保存在正文，实验源码保存在附录。复现会生成 JSON 结果，无须依赖原测试机器的临时目录。

- 实验脚本 SHA-256：`da3950dcb9fb10bb6454511d29a23cb5718d19cb268f53cb61a53d1e0eff7f74`。
- 本次 JSON SHA-256：`bb035b18d83199e4dac1e0211f6456ebcd2d400315b0d5bf3588f21d93e17c2a`。换用不同 Python/torch 版本时，JSON 中环境字段会变化，不要求该摘要一致。

## 7. 后续验证计划

在已完成的 CPU 正确性回归基础上，下一阶段将在固定宿主与 NPU 环境中开展以下验证：

1. **端到端运行正确性。** 验证实际安装包、Ascend runner、原生算子和跨进程控制路径；在压缩及物理页复用过程中，核对 scheduler、worker 和各层设备块表的一致性。
2. **数值与请求生命周期。** 对照同一保留集合检查 FP16 KV 搬移和 attention 数值结果，覆盖压缩后追加 token、多请求、batch 行调整、分块 prefill、once/repeat、取消及抢占重放。
3. **模型效果与运行效率。** 在相同模型、数据集和压缩预算下评测任务得分、实际 KV 占用、吞吐与 TPOT，量化细粒度压缩的收益和额外开销。

首轮设备验证以当前候选支持范围为准：Qwen2.5-14B、FP16、单卡、同步调度、V1 runner、eager、关闭 APC。原 issue 中的 Qwen3-8B、BF16、异步调度和图模式不属于当前支持范围。设备验证完成前，公开启用状态保持 `import_only`。

## 附录：本次 CPU 对照实验完整脚本

该脚本是本报告的可复现实验附件，不是插件运行依赖；宿主测试替身在代码中明确标注。

<details>
<summary>展开完整实验脚本</summary>

<!-- BEGIN CPU EXPERIMENT -->

```python
"""CPU evidence for the local KNorm branch; host tables and RPC are doubles."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))


def legacy_reproduction():
    from conftest import build_fake_host
    from vllm_hust_knorm.adapters.vllm_hust.patches import install_patches
    from vllm_hust_knorm.knorm.manager import get_knorm_manager_class

    host = build_fake_host()
    install_patches()
    os.environ.update(VLLM_KNORM_ENABLED="1", VLLM_KNORM_COMPRESSION_RATIO="0.5")
    spec = host["FullAttentionSpec"]()
    spec.block_size = 128
    pool = host["BlockPool"]()
    pool.null_block.block_id = 0
    manager = get_knorm_manager_class()(spec, block_pool=pool)
    blocks = [SimpleNamespace(block_id=i, ref_cnt=1, is_null=False,
                              block_hash=None, next_free_block=None,
                              prev_free_block=None) for i in range(1, 9)]
    manager.req_to_blocks["legacy"] = blocks
    worker = [b.block_id for b in blocks]
    manager.remove_skipped_blocks("legacy", 1024)
    scheduler_after_eviction = [b.block_id for b in blocks]
    recycled = pool.free_block_queue.fake_free_list_head.next_free_block.block_id
    # Explicit model of the historical non-resumed append-only worker protocol.
    scheduler = scheduler_after_eviction + [recycled]
    worker.extend([recycled])
    mismatches = [i for i, (a, b) in enumerate(zip(scheduler, worker)) if a != b]
    assert scheduler == [1, 0, 0, 0, 5, 6, 7, 8, 2]
    assert worker == [1, 2, 3, 4, 5, 6, 7, 8, 2]
    assert mismatches == [1, 2, 3] and worker[1] == worker[-1] == recycled
    for name in ("VLLM_KNORM_ENABLED", "VLLM_KNORM_COMPRESSION_RATIO"):
        os.environ.pop(name, None)
    return dict(scheduler=scheduler, worker=worker, mismatches=mismatches,
                recycled_page=recycled, negative_control="expected mismatch")


def current_reproduction(fault):
    import torch
    from vllm_hust_knorm.adapters.ascend.worker import (
        RemoteCache, RemoteWorker, WorkerControl,
    )
    from vllm_hust_knorm.runtime.allocator import PagePool
    from vllm_hust_knorm.runtime.state import Layout, Request
    from vllm_hust_knorm.runtime.transaction import Transactions

    # CPU copies model host/device separation, including an uncondensed row hole.
    class Table:
        def __init__(self, blocks):
            self.cpu = torch.zeros((3, 4), dtype=torch.int64)
            self.device = self.cpu.clone()
            self.cpu[2, :len(blocks)] = torch.tensor(blocks)
            self.device.copy_(self.cpu)

        def clear_row(self, row):
            self.cpu[row].zero_()

        def add_row(self, blocks, row):
            self.cpu[row, :len(blocks)] = torch.tensor(blocks)

        def commit_block_table(self, rows):
            self.device[:rows].copy_(self.cpu[:rows])

    pool = PagePool(160, reserved=(0,))
    request = Request("current", 256, semantic_length=256, phase="decode",
                      tokens=list(range(256)))
    request.layers = {
        layer: Layout(256, pool.allocate((request.instance, layer), 2))
        for layer in range(48)
    }
    backing = torch.zeros((2, 160, 128, 2, 2), dtype=torch.float16)
    positions = torch.full((160, 128, 2), -1, dtype=torch.int64)
    tables = tuple([p.number for p in layout.pages] for layout in request.layers.values())
    runner = SimpleNamespace(
        requests={request.request_id: SimpleNamespace(block_ids=tables)},
        input_batch=SimpleNamespace(req_id_to_index={request.request_id: 2},
                                    num_reqs=1, block_table=[Table(b) for b in tables]),
    )
    for layout in request.layers.values():
        for i in range(256):
            page, slot = layout.pages[i // 128].number, i % 128
            for head in range(2):
                backing[0, page, slot, head, 0] = 1 if (i >= 128) == (head == 0) else 2
                backing[1, page, slot, head] = torch.tensor([i, head])
                positions[page, slot, head] = i
    control = WorkerControl(runner, backing, positions, 16 * 1024 * 1024, lambda: None)
    control.register(request)
    initial_free = len(pool.free)
    calls = []
    injected = False
    scheduler_tables = dict(request.layers)

    def check_tables(layouts):
        expected = tuple([p.number for p in layouts[l].pages] for l in sorted(layouts))
        assert runner.requests[request.request_id].block_ids == expected
        for table, blocks in zip(runner.input_batch.block_table, expected):
            padded = blocks + [0] * (4 - len(blocks))
            assert table.cpu[2].tolist() == padded
            assert table.device[2].tolist() == padded

    def rpc(operation, *args):
        nonlocal injected
        result = control.dispatch(operation, *args)
        if operation in {"install", "commit"}:
            check_tables(control.states[request.request_id].layers)
            assert len(pool.free) == initial_free
            calls.append(operation)
            if not injected and fault == operation:
                injected = True
                raise RuntimeError("injected lost acknowledgement")
            if not injected and fault == operation + "_receipt":
                injected = True
                result = dict(result, version=result["version"] + 1)
        return result

    def install_scheduler(state):
        scheduler_tables.clear()
        scheduler_tables.update(state.layers)

    worker = RemoteWorker(rpc, SimpleNamespace(install=install_scheduler))
    transactions = Transactions(pool, RemoteCache(rpc), worker)
    original_release = pool.release_once
    release_count = 0

    def checked_release(owner, pages, transaction):
        nonlocal release_count
        assert calls[-1] == "commit"
        assert control.committed[request.instance] == request.version
        assert scheduler_tables == request.layers
        check_tables(request.layers)
        release_count += len(pages)
        original_release(owner, pages, transaction)

    pool.release_once = checked_release
    plan = transactions.plan(request, {l: 128 for l in range(2, 48)})
    if fault:
        try:
            transactions.execute(request, plan)
        except RuntimeError as exc:
            assert "acknowledgement" in str(exc)
        else:
            raise AssertionError("fault was not observed")
        assert injected and len(pool.free) == initial_free and release_count == 0
    assert transactions.execute(request, plan) == "COMMITTED"
    assert release_count == 46 and len(pool.free) == initial_free + 46
    assert [v.length for v in request.layers.values()] == [256, 256] + [128] * 46
    check_tables(request.layers)

    # Give every free physical page to a second owner and overwrite with a sentinel.
    recycled = pool.allocate(("other-request-instance", 0), len(pool.free))
    released = {p.number for l in plan.layers for p in l.old.pages if p not in l.new.pages}
    recycled_ids = [p.number for p in recycled]
    assert released <= set(recycled_ids)
    backing[:, recycled_ids] = -999
    positions[recycled_ids] = -999
    snapshot = backing.clone()
    assert transactions.execute(request, plan) == "COMMITTED"
    assert torch.equal(backing, snapshot) and release_count == 46
    check_tables(request.layers)

    # Independent oracle: explicitly known survivor sets, not production select().
    # Attention is one query over the retained set, no positional/sliding mask.
    errors = []
    for layer, layout in request.layers.items():
        page_ids = runner.input_batch.block_table[layer].device[2].tolist()
        assert not (set(page_ids[:len(layout.pages)]) & set(recycled_ids))
        for head in range(2):
            expected_positions = (list(range(256)) if layer < 2 else
                                  list(range(128, 256)) if head == 0 else list(range(128)))
            actual = []
            for i in range(layout.length):
                page, slot = page_ids[i // 128], i % 128
                pos = int(positions[page, slot, head])
                k = backing[0, page, slot, head].double()
                v = backing[1, page, slot, head].double()
                expected_k = 1 if (pos >= 128) == (head == 0) else 2
                assert k.tolist() == [expected_k, 0]
                assert v.tolist() == [pos, head]
                actual.append((pos, k, v))
            assert sorted(p for p, _, _ in actual) == expected_positions
            query = torch.tensor([0.2, -0.1], dtype=torch.float64)
            keys = torch.stack([k for _, k, _ in actual])
            values = torch.stack([v for _, _, v in actual])
            reference_keys = torch.tensor([
                [1 if (p >= 128) == (head == 0) else 2, 0]
                for p in expected_positions], dtype=torch.float64)
            reference_values = torch.tensor([[p, head] for p in expected_positions],
                                            dtype=torch.float64)
            actual_out = torch.softmax(keys @ query / (2 ** 0.5), dim=0) @ values
            reference_out = torch.softmax(reference_keys @ query / (2 ** 0.5), dim=0) @ reference_values
            errors.append(float((actual_out - reference_out).abs().max()))
    assert max(errors) < 1e-12
    return dict(fault=fault, groups=48, heads=2, active_row=2, num_reqs=1,
                released_pages=release_count, donor_pages_reused=len(released),
                free_pages_before_ack=initial_free, free_pages_after_commit=initial_free + 46,
                full_tables_equal=True, foreign_kv_reads=0, duplicate_commit="no mutation",
                attention_max_abs_error=max(errors), attention_comparisons=len(errors))


if __name__ == "__main__":
    import platform
    import torch
    result = dict(python=platform.python_version(), torch=torch.__version__,
                  device="cpu", legacy=legacy_reproduction(),
                  current=[current_reproduction(f) for f in
                           (None, "install", "commit", "install_receipt", "commit_receipt")])
    print(json.dumps(result, indent=2, ensure_ascii=False))
```

</details>
