# 开发指南（Development）

本文面向要改这个插件的贡献者：仓库结构、必须遵守的分层纪律、测试
策略、提交前自查。服务器上的端到端验证见
[how-to-run.md](how-to-run.md)；发布流程见
[packaging-and-release.md](packaging-and-release.md)。

## 1. 环境与常用命令

```bash
python -m pip install -e ".[dev]"   # pytest + ruff + build + setuptools/wheel

pytest -q                                 # 全量单测（CPU，无需 NPU/vllm/torch）
ruff check .                              # lint
ruff format --check .                     # 格式
python -m build --no-isolation            # 构建 wheel + sdist（纯 Python）
bash scripts/verify-wheel.sh dist/*.whl   # wheel 内容/清单/entry point 校验
```

CI（`.github/extension-ci.yml`，push main / PR 触发）：Python
3.10 / 3.12 / 3.14 矩阵，依次跑 `ruff check` → `ruff format
--check` → `pytest -q` → `build` → `verify-wheel.sh` → 隔离 venv
冒烟安装 → manifest 校验。**CI 没有 NPU 阶段**——设备验证在授权
机器上按 [how-to-run.md](how-to-run.md) 手工跑，证据回填
[HOST_CONTRACT.md](../HOST_CONTRACT.md)。

## 2. 仓库结构

```
src/vllm_hust_knorm/
├── knorm/                    # 从 legacy 移植的 KNorm 本体
│   ├── config.py             #   KnormConfig + should_activate（唯一真源）
│   ├── norms.py / eviction.py#   纯数学（CPU 可测，零依赖）
│   ├── attention_backend.py / hooks.py   # torch 路径（惰性导入）
│   └── manager.py            #   KnormFullAttentionManager（惰性基于宿主基类）
├── core/
│   ├── hosts.py              # 宿主探测 + fail-closed 守卫（REQUIRED_SURFACES）
│   └── activation.py         # 统一激活管线（bootstrap 与 facade 共用）
├── adapters/
│   ├── base.py               # HostAdapter（宿主 import 只在方法内）
│   └── vllm_hust/patches.py  # 六个幂等宿主补丁 P1–P6（见 HOST_CONTRACT.md）
├── bootstrap.py              # vllm.general_plugins 钩子（无宿主时 no-op）
├── manifests/                # vllm-hust-extension-v0.2.json（发现用，import 安全）
└── _version.py               # 唯一版本源（与 manifest extension_version 同值）
```

## 3. 分层纪律（改代码前必读）

四条硬规则，违反会被测试抓住（`test_activation.py::TestImportPurity`、
`test_manifest.py::TestDiscoveryPurity` 以子进程强制）：

1. **导入卫生**：import 本包不得拉起 `vllm` / `torch`。重依赖只在
   使用点函数体内惰性导入；缺依赖时报带调用语境的精确错误，不是
   import 级联堆栈。
2. **宿主 import 只在 register/patch 方法体内**：适配器模块顶层绝不
   import 宿主；`KnormFullAttentionManager` 经
   `get_knorm_manager_class()` 在激活时才组合（`@cache`）。
3. **激活入口唯一**：点亮插件的逻辑只存在于
   `core/activation.py::activate`；`bootstrap.register_plugins` 与
   包门面 `vllm_hust_knorm.activate()` 都是它的薄封装。改注册语义
   只改一处。
4. **单一真源**：`knorm/config.py::should_activate` 是激活判定的
   唯一出处，P3（manager 注册）与 P5（runner 包装）必须共用——
   这是 legacy issue #163 半启用态的根治点，别在任何一侧内联环境
   判断。

其余约定：

- **fail-closed 全库一致**：未知值/非法配置/缺表面一律
  `ValueError`/`RuntimeError` 并给出清单。遇到"要不要猜一个默认"，
  答案永远是抛错。
- **出处分明**：从 legacy 移植的代码在模块 docstring 标注来源
  PR；新逻辑不冒充 legacy（迁移映射见
  [PROVENANCE.md](../PROVENANCE.md)）。
- **可观测性标记**：改变激活状态的关键路径必须打 `[vllm-hust-knorm]`
  前缀 + `knorm-*-registered/installed` 日志标记，供服务器验证
  rg 断言（见 how-to-run.md §5）；新增标记同步补测试与文档。
- ruff 规则集 `B,E,F,I,SIM,UP` + `ruff format`（CI 双门禁）。

## 4. 测试策略（没有 NPU 怎么开发）

- **纯核心**（config/norms/eviction）：CPU 直接测，包括 8 组合
  激活矩阵（PR #214 移植）与驱逐规划语义（warmup 保护、scored
  优先、cached/uncached 分流）。
- **宿主契约**（`test_host_contract.py`）：`tests/conftest.py` 的
  模拟 vllm 宿主复刻 HOST_CONTRACT.md 的六个表面，覆盖 P1–P6 的
  行为、幂等与 fail-closed——无需安装 vllm。
- **torch 路径**：`pytest.importorskip("torch")` 门控，有 torch 的
  环境自动加测。
- **打包/发现**：manifest 一致性、wheel/sdist 内容、entry point、
  发现不 import 实现。
- **设备数值/端到端**：不属于 pytest——按
  [how-to-run.md](how-to-run.md) 在授权机器上跑，证据回填
  HOST_CONTRACT.md。

## 5. 提交前自查

- [ ] 新代码遵守 §3 四条纪律（尤其：没有新的顶层重导入）
- [ ] `pytest -q` 全绿；`ruff check . && ruff format --check .` 干净
- [ ] 动了激活语义？确认 P3/P5 仍共用 `should_activate`，并更新
      激活矩阵测试
- [ ] 动了宿主表面？同步 HOST_CONTRACT.md、`core/hosts.py::
      REQUIRED_SURFACES` 与模拟宿主（conftest）
- [ ] 代码改动已递增 `_version.py`（发布相关；见
      packaging-and-release.md）
- [ ] 涉及设备路径：在授权机器跑过 how-to-run.md 的验证 workflow
      并把证据（commit、日志摘录、吞吐数字）回填 HOST_CONTRACT.md
