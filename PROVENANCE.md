# Provenance

Source archive:
[intellistream/vllm-hust-legacy-20260831](https://github.com/intellistream/vllm-hust-legacy-20260831)

Primary history (per-commit patches archived under
[provenance/legacy-patches/](provenance/legacy-patches/)):

- [PR #76: initial KNorm module](https://github.com/intellistream/vllm-hust-legacy-20260831/pull/76)
  — merged `973fb5b752dc7b1a0545c85bd06464e43ca18261` (2026-07-03).
  `vllm/knorm/` module (config, attention wrapper, hooks, manager),
  env variables, v1 core integration, free-queue prepend support.
- [PR #134: prefix-cache safety](https://github.com/intellistream/vllm-hust-legacy-20260831/pull/134)
  — merged `dfe9ee8aaddfa0b90ba87b0ef944eb8ab35f9683` (2026-07-20).
  Knorm disabled with prefix caching; registration coverage.
- [PR #214: unified activation and correctness repair](https://github.com/intellistream/vllm-hust-legacy-20260831/pull/214)
  — merged `f680fd3e9b201b9b2eb57354f6dee2926825e4b9` (2026-08-15).
  `should_activate()` as the single source of truth fixing the
  half-enabled state of legacy issue #163 (78% throughput regression);
  all runner-side Knorm paths guarded.

## Migration map (legacy in-tree → this plugin)

| Legacy file | Plugin location | Notes |
| --- | --- | --- |
| `vllm/knorm/config.py` | `src/vllm_hust_knorm/knorm/config.py` | env read moved from `vllm.envs` to `os.environ`; `VLLM_KNORM_ENABLED` now defaults to `0` (plugin opt-in posture); values validated fail-closed |
| `vllm/knorm/attention_backend.py` | `.../knorm/attention_backend.py` + `norms.py` | torch imports made lazy; reduce-op math factored pure |
| `vllm/knorm/hooks.py` | `.../knorm/hooks.py` | aggregation factored pure (`aggregate_token_scores`) |
| `vllm/knorm/manager.py` | `.../knorm/manager.py` + `eviction.py` | host base imported lazily; eviction planning factored pure |
| `vllm/envs.py` (VLLM_KNORM_*) | plugin reads `os.environ` directly | same names/defaults as the 0.23 host registration |
| `vllm/v1/core/single_type_kv_cache_manager.py` | adapter patch P3 | same `should_activate` rule |
| `vllm/v1/core/sched/scheduler.py` | adapter patch P4 | identical routing |
| `vllm/v1/worker/gpu_model_runner.py` | adapter patches P5/P6 | wrapper install moved to runner `__init__` (before first forward); score collect/attach moved to a `sample_tokens` wrapper — same lifecycle points |
| `vllm/v1/core/block_pool.py` + `kv_cache_utils.py` | adapter patches P1/P2 | identical implementations |
| `tests/v1/test_kv_cache_spec_registry.py` (knorm cases) | `tests/test_host_contract.py` | 8-combination coverage matrix ported |
| `tests/ci/test_knorm_runner_static.py` | guard semantics covered by `tests/test_host_contract.py::TestRunnerPatches` | static-AST checks replaced by behavioral tests against the fake host |
| `.github/workflows/scripts/*` (Ascend E2E infra from PR #134) | not migrated | legacy-repo CI plumbing; plugin CI is CPU-only per the extension guide, NPU acceptance runs out-of-band |

Host-side reference: vLLM-HUST `main@8344e107` (knorm-free surfaces)
and 0.23 seam `5c994cdc` (in-tree module the plugin refuses to
double-provide) — details in [HOST_CONTRACT.md](HOST_CONTRACT.md).
