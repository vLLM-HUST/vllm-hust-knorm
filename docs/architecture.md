# Architecture

KNorm as an extension bundle: one implementation, two activation
surfaces, zero host-file edits.

```text
┌────────────────────────────────────────────────────────────┐
│ vLLM-HUST host process (scheduler / engine core / worker)  │
│                                                            │
│  vllm.general_plugins ──► bootstrap.register_plugins()     │
│                              │                             │
│                              ▼                             │
│                    core.activation.activate()              │
│              guards: no in-tree vllm.knorm, surfaces OK    │
│                              │                             │
│                              ▼                             │
│              adapters/vllm_hust/patches (P1–P6, idempotent)│
│   FreeQueue / BlockPool / spec registration / Scheduler /  │
│   GPUModelRunner __init__ + sample_tokens                  │
└────────────────────────────────────────────────────────────┘
              activation still gated by VLLM_KNORM_ENABLED=1
                              │
┌───────────────────────── knorm package ────────────────────┴──┐
│ config    KnormConfig + should_activate()（唯一真源）          │
│ norms     纯 key-norm 数学（reduce / aggregate）               │
│ eviction  纯驱逐规划（warmup 保护、scored 优先于 unscored）     │
│ attention_backend  torch 路径：包装 impl.forward 收集逐层范数   │
│ hooks     collect（forward 后聚合到块）/ attach（附着到输出）    │
│ manager   KnormFullAttentionManager（惰性基于宿主基类构建）     │
└───────────────────────────────────────────────────────────────┘
```

## Layering rules (extension guide)

- **Import purity**: importing `vllm_hust_knorm` (and its `manifests`
  package) imports neither vllm nor torch; enforced by
  `tests/test_activation.py::TestImportPurity` and
  `tests/test_manifest.py::TestDiscoveryPurity`. Discovery
  (`vllm_hust.extension_bundles` → `manifests` module) reads the JSON
  without touching any implementation module.
- **Register ≠ enable**: `register_plugins()` installs patches in every
  host process, but behavior changes only under
  `VLLM_KNORM_ENABLED=1` (default `0`) **and** prefix caching
  disabled — both via `should_activate()`, the same predicate for
  manager registration (P3) and runner activation (P5), which is the
  fix for legacy issue #163's half-enabled state.
- **Fail closed**: in-tree `vllm.knorm` present, missing host
  surfaces, or invalid `VLLM_KNORM_*` values all abort activation
  loudly; nothing silently degrades.

## Data flow (per decode step)

1. P5 installed the attention wrapper before the first forward; during
   the forward each layer's `forward` computes per-token key L2 norms
   (mean over KV heads by default) and stores them (device tensors).
2. P6 (`sample_tokens` entry, last PP rank) averages the per-layer
   norms, maps tokens → (request, block) with `min` aggregation, and
   stores `runner._knorm_scores`.
3. P6 attaches the scores to the model runner output; P4
   (`Scheduler.update_from_output`) routes them into the manager
   bridge (`submit_block_scores`).
4. On scheduling, `KnormFullAttentionManager.remove_skipped_blocks`
   drains the bridge and plans evictions: keep
   `max(warmup, ceil(total × ratio))` blocks, evict the highest-norm
   scored blocks first, unscored last; cached evictions free at the
   free-queue tail, uncached at the head (immediate reuse).
