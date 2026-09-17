# How to run

## Prerequisites

- A knorm-free vLLM-HUST build that carries the surfaces listed in
  [HOST_CONTRACT.md](../HOST_CONTRACT.md) (statically verified against
  vLLM-HUST `main@8344e107`). Hosts shipping an in-tree `vllm.knorm`
  (0.23 seam era) are refused at startup.
- CUDA GPU or Ascend NPU (torch + platform stack installed by the
  host, not by this plugin).

## Install and enable (extension manager path)

```bash
pip install vllm-hust-knorm vllm-hust-ext

vllm-hust-ext extension list
vllm-hust-ext extension inspect org.vllm-hust.knorm   # disabled, compatible
vllm-hust-ext extension enable org.vllm-hust.knorm

vllm-hust-ext run --dry-run -- vllm serve MODEL --no-enable-prefix-caching
vllm-hust-ext run -- vllm serve MODEL --no-enable-prefix-caching
```

The manager injects `VLLM_KNORM_ENABLED=1` (plus the ratio/warmup
defaults from the manifest's `activation.environment`).

## Install and enable (plain pip path)

```bash
pip install vllm-hust-knorm
export VLLM_KNORM_ENABLED=1
export VLLM_KNORM_COMPRESSION_RATIO=0.5   # fraction of blocks to keep
export VLLM_KNORM_WARMUP_TOKENS=32
vllm serve MODEL --no-enable-prefix-caching
```

## Runtime evidence (acceptance bar)

"Process started" is not acceptance. Check:

1. Startup log contains `[vllm-hust-knorm] runtime patches registered`.
2. The scheduler uses `KnormFullAttentionManager` (prefix-cache
   warning must **not** appear when prefix caching is off).
3. Requests complete and outputs match the uncompressed baseline for
   your quality bar; compression is observable via reduced KV usage /
   increased capacity at the same memory.
4. With `VLLM_KNORM_ENABLED=0` (or `--enable-prefix-caching`) a fresh
   process falls back to the stock manager and the warning/behavior
   matches HOST_CONTRACT.md.

NPU acceptance: run the matched-baseline protocol from legacy issue
#163 (same model, request set, concurrency, lengths, hardware and
launch flags; ≥3 runs) and record evidence in HOST_CONTRACT.md before
releasing.

## Disable and roll back

```bash
vllm-hust-ext extension disable org.vllm-hust.knorm
# stop old processes and start new ones (no hot unload), then:
vllm-hust-ext extension forget org.vllm-hust.knorm
pip uninstall vllm-hust-knorm
vllm-hust-ext extension list   # gone
```
