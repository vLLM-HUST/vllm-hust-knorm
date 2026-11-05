# Fixed-host development environment

This is an experimental binding, not a serving recipe. Public eligibility
remains `import_only`. The target is Qwen2.5-14B-Instruct with FP16 KV, eager
Ascend V1, synchronous single-rank execution and private layer pages.

The source archives must match:

| Source | Commit | Archive SHA-256 |
| --- | --- | --- |
| vLLM | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` | `5309f930285a26629e3e49a8d022f81c0f99b2490e24fa86dddbd2cd4986dfee` |
| vLLM Ascend | `c09d3e20b9c45c80b34ae3781440d90904a44023` | `8d3a91782da1120681624a79fdc3c38b93d0a20f13cf88b966e8a47e8437b35a` |

Runtime source files remain unchanged. The two declaration patches are:

- Ascend's runtime and build FastAPI ranges are aligned with core's
  `fastapi[standard]>=0.133.0,<0.137.0`.
- Core's OpenCV requirement becomes `>=4.11.0.86,<4.12.0`. OpenCV 4.13 requires
  NumPy 2, while vendor Triton Ascend 3.6.0.post2 requires NumPy 1.26.4.
  This environment resolves to OpenCV 4.11.0.86 and NumPy 1.26.4. The candidate
  is text-only; this is not a qualification of upstream multimodal features.

Use the PyPI index plus the vendor Ascend index shown in the scripts. The
runtime lock contains exact versions and distribution hashes. Build dependencies
are frozen separately because empty core declares torch 2.13.0, whereas the
Ascend runtime requires torch 2.10.0. The former's CUDA Python dependencies stay
inside the build environment; the runtime uses the ARM64 CPU torch distribution
with the installed torch-npu package. No driver is installed by these scripts.

## Reproduce without devices

On the authorized development server, place the plugin checkout in
`/root/knorm/host-0.30.0/src/knorm` and the two original GitHub source archives in
`/root/knorm/host-0.30.0/artifacts`, named `<repository>-<commit>.tar.gz`.
The scripts only prepare task-owned paths. A CPython 3.12 prefix is copied,
including its headers, excluding shared site-packages. The prefix below is the
one used in this development run; it is read-only input.

```bash
python3 scripts/host/prepare_host.py /root/knorm/host-0.30.0 \
  --python-prefix /usr/local/python3.12.13
/root/knorm/host-0.30.0/env/build/bin/python scripts/host/build_host.py \
  /root/knorm/host-0.30.0
```

`prepare_host.py --resolve` explicitly re-resolves the combined declarations
instead of installing the checked-in lock. Review a changed lock before using it.
Build concurrency is limited to two. Empty core and Ascend Python-only wheels
use `VLLM_TARGET_DEVICE=empty`, `COMPILE_CUSTOM_KERNELS=0` and
`SOC_VERSION=ascend910b2`. Archive builds receive the local metadata suffix
`+knorm`; source identities are checked by hashes, not by spoofing a host version.
This is packaging validation: the Ascend wheel has no custom native operators.

Run from the checkout with the installed plugin (clear inherited `PYTHONPATH`):

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 \
  /root/knorm/host-0.30.0/env/runtime/bin/python \
  scripts/host/test_without_devices.py -q -p no:cacheprovider tests/test_ascend_host.py
env -u PYTHONPATH /root/knorm/host-0.30.0/env/runtime/bin/python \
  scripts/host/check_install.py
env -u PYTHONPATH /root/knorm/host-0.30.0/env/runtime/bin/python \
  scripts/host/test_without_devices.py --probe-imports
```

The test entry blocks `torch_npu`, `acl` and Triton device-backend imports,
reports those packages unavailable to availability probes, and disables plugin
autoloading. It does not fabricate device modules. Host buffers use pageable CPU
memory. The real core engine/scheduler and cache objects remain in use; a named
CPU executor supplies deterministic KV and sampled outputs instead of a model.
The separate import probe reports real Ascend modules that cannot be loaded
without device dependencies. Those are unverified, not skipped integration wins.

`record_environment.py ROOT [--label build]` records Python, installed package
versions/origins and archive/lock/wheel hashes without importing their runtimes.
It writes only under `ROOT/artifacts`.

## Explicit candidate entry

For a resolved host configuration, the development API is:

```python
from vllm_hust_knorm import KNormConfig
from vllm_hust_knorm.adapters.ascend.candidate import configure

configure(vllm_config, KNormConfig("once", threshold, ratio), workspace_bytes)
```

All policy values and the scratch reservation are explicit. Call before creating
an engine or worker. This verifies the installed source files and installs the
loader locally; `VLLM_KNORM_CANDIDATE=1` propagates installation through the
normal general-plugin hook in child processes. Do not filter out the
`vllm-hust-knorm` general plugin in those processes. No wrappers act without the
matching `additional_config` payload. `VLLM_KV_CACHE_LAYOUT` must be `LBNHC`;
conflicting layouts/configurations are rejected. `VLLM_KNORM_ENABLED=1` remains
unavailable. Native Ascend initialization, kernels, synchronization, full model
recovery, numerical tolerance, quality and performance still require device
qualification before this entry can become a supported runtime interface.

## Native dependency preparation without devices

The native target keeps CANN Toolkit, 910B Ops and NNAL at **9.2.1**, and
FLA NPU A2 at **26.9.1**. See the [release installation guide](https://docs.vllm.ai/projects/ascend/en/v0.30.0/getting_started/installation.html)
and [compatibility matrix](https://docs.vllm.ai/projects/ascend/en/v0.30.0/community/versioning_policy.html).
The release image validates HDK 26.1.1; that is not a statement of the minimum
compatible driver. Compatibility with an existing 26.0.rc1 driver is unconfirmed.
Private user-space libraries do not remove the host driver requirement.

`environment/native-components.json` pins public URLs and archive SHA-256 values.
The preparation script additionally verifies each provided embedded SHA-256.
The optional `test-ops` payload uses the vendor's all-zero marker for an absent
checksum; it remains covered by the verified parent archive and is reported as
such. The optional non-Makeself `mindstudio-memscope` installer is retained
without unpacking or execution. The script never executes installer code, including during
extraction: these wrappers log outside the project even with `--noexec --extract`.
The nested packages are inspection payloads, **not an installed SDK**. The audited
root installer writes `/etc/Ascend` and `/var/log/ascend_seclog` regardless of the
chosen install prefix. Run it only through the private-root installation below.
Official Conda packages also invoke this installer from their post-link hooks;
their environment prefix alone does not confine those writes.

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  /root/knorm/host-0.30.0/env/runtime/bin/python \
  scripts/host/prepare_native_dependencies.py /root/knorm/host-0.30.0 \
  --download --extract
```

Keep at least 20 GiB free before downloads; allow roughly 25 GiB for archives,
nested packages and expanded inspection files together. The script reuses
verified completed extractions and refuses to overwrite incomplete destinations.
Temporary extraction directories from failed attempts remain available for
inspection; remove only identified task-owned temporary files if space requires it.

FLA NPU is required by the A2 standard worker patch import chain even when the
selected model does not use gated delta attention. Its release wheel digest is
verified against the [upstream release API](https://api.github.com/repos/flashserve/flash-linear-attention-npu/releases/tags/v26.9.1-release).
The normalized vendor index uses `flash-linear-attention-npu-a2/`; the underscore
URL in the pinned Dockerfile returns 404. The actual wheel matches the release
API digest, not the stale commented Dockerfile digest. `expecttest==0.3.0` is its
only additional missing Python dependency in the prepared runtime.

Install the additive requirements with the existing runtime constraints. This
uses normal dependency resolution, preserves the old lock and does not rebuild
Triton or load the NPU backend:

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  UV_CACHE_DIR=/root/knorm/host-0.30.0/cache/uv \
  TMPDIR=/root/knorm/host-0.30.0/tmp \
  /root/knorm/host-0.30.0/env/build/bin/uv pip install \
  --python /root/knorm/host-0.30.0/env/runtime/bin/python \
  --constraint host-patches/environment/runtime.lock \
  --requirement host-patches/environment/native-python.lock

env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  /root/knorm/host-0.30.0/env/runtime/bin/python \
  scripts/host/inspect_native_dependencies.py /root/knorm/host-0.30.0
```

Inspection uses distribution metadata, `pip check`, compiler version commands,
ELF headers and a C++17/Python/NUMA compile/link check whose output is never
loaded. It does not import torch, torch_npu or ACL. The recorded library
candidates are a static search, not a dynamic loader or device compatibility test.
Existing runtime CMake/Ninja and system GCC/NUMA are used without modifying them.

`install_private_sdk.py` copies the system utilities and private Python into
`ROOT/toolchains/sdk-root`, then runs the unchanged, checksum-verified vendor
installers in that root filesystem. It requires Linux aarch64, chroot permission
and Landlock support. No host directories are mounted. The root contains only
the four standard null/zero/random character devices; `/proc` and `/sys` are
empty. The installer retains only filesystem ownership/mode capabilities;
mount, chroot, ptrace and device-creation capabilities are dropped. Landlock
provides an additional path restriction, but its ABI 1 metadata limitations
are why the separate root filesystem is required.

Toolkit uses the vendor's `atc,devtools,nnrt` selection, which supplies the
runtime, Ascend C compiler, CMake support and Python compilation tools. Optional
profilers, PyPTO and simulator packages are not selected. Ops and NNAL use their
normal full component installation. No compatibility checks are disabled, and
the absence of a driver inside this offline root remains visible in the logs.
Allow at least 20 GiB free after obtaining the archives and inspection payloads.

```bash
TASK_ROOT=/root/knorm/host-0.30.0
TASK_PYTHON="$TASK_ROOT/env/runtime/bin/python"
env -u PYTHONPATH "$TASK_PYTHON" scripts/host/install_private_sdk.py "$TASK_ROOT" --prepare --probe
env -u PYTHONPATH "$TASK_PYTHON" scripts/host/install_private_sdk.py "$TASK_ROOT" --install toolkit
env -u PYTHONPATH "$TASK_PYTHON" scripts/host/install_private_sdk.py "$TASK_ROOT" --install 910b-ops
env -u PYTHONPATH "$TASK_PYTHON" scripts/host/install_private_sdk.py "$TASK_ROOT" --install nnal --publish
```

Successful installations are recorded and cannot be silently overwritten.
Failed attempts retain separate logs. The final `toolchains/ascend` symlink
exposes the vendor-installed tree while preserving its absolute installation
prefix; it does not assemble files from inspection payloads. Registry files,
shell files and logs modified by the vendor remain inside `sdk-root`.

Install `environment/native-compiler.lock` with the original runtime constraints
using the same `uv pip install` command as above. This adds the pinned `absl-py`
dependency needed by CANN's `schedule_search`; the SDK supplies its own Python
compilation modules through `PYTHONPATH`, without replacing runtime packages.

Generate process-local exports and verify the installed SDK:

```bash
env -u PYTHONPATH "$TASK_PYTHON" scripts/host/native_build_environment.py "$TASK_ROOT" \
  --cann "$TASK_ROOT/toolchains/ascend/cann-9.2.1" \
  --atb "$TASK_ROOT/toolchains/ascend/nnal/atb/9.2.1/atb/cxx_abi_1" \
  > "$TASK_ROOT/runs/private-sdk/native-env.sh"
source "$TASK_ROOT/runs/private-sdk/native-env.sh"
"$TASK_PYTHON" scripts/host/verify_installed_sdk.py "$TASK_ROOT"
```

The generator checks CANN 9.2.1 metadata, Ascend C support and private paths,
replaces inherited search paths and limits build parallelism to two. ATB's ABI
directory must match the runtime torch; verification checks this with a CPU
torch import while device backends are forbidden and autoload is disabled.
The fixed runtime uses ABI 1. Verification also checks combined Python
requirements, ELF dependency names and a C++17 SDK header/link probe whose
output is never loaded or executed. Missing driver runtime libraries are
reported separately; link stubs are never added to `LD_LIBRARY_PATH`.
`LDFLAGS` supplies `-rpath-link` with the installed `lib64` before `devlib`, so
compiler checks can resolve indirect driver references without selecting SDK
stubs in place of real user-space libraries. A small CMake project verifies the
host's device-ID API compile/link check; it never runs the resulting executable.
The verifier also compares the runtime and original artifacts with
`artifacts/native-dependencies-environment.json` from the preceding preparation.

Review and source the generated file within the build shell; never append it
to a global shell startup file. Native extension builds must use **env/runtime**
with torch 2.10.0 / torch-npu 2.10.0.post7, not the empty-core torch 2.13 environment.

The next native build is a separate action after SDK readiness and any required
CANN Python component installation are verified. Set `SOC_VERSION=ascend910b2`
explicitly so the build does not invoke automatic device discovery, then build
the pinned Ascend source with the runtime interpreter, `COMPILE_CUSTOM_KERNELS=1`
and `python -m build --wheel --no-isolation`. Do not use `build_host.py` as proof
of native compilation: that script deliberately keeps custom kernels disabled.
No native custom-op build, runner import or device validation is established by
the preparation commands above.
