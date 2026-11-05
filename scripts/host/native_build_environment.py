"""Print process-scoped exports for a verified private native build environment."""

import argparse
import importlib.metadata
import os
import re
import shlex
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--cann", type=Path, required=True)
    parser.add_argument("--atb", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to("/root/knorm"):
        parser.error("server preparation is confined to /root/knorm")
    cann, atb = args.cann.absolute(), args.atb.absolute()
    for prefix in (cann, atb):
        if not prefix.resolve().is_relative_to(root / "toolchains"):
            parser.error(
                "CANN and ATB must resolve inside the private toolchains directory"
            )
        if "unpacked" in prefix.resolve().relative_to(root).parts:
            parser.error("inspection payloads are not an installed SDK")
    version = cann / "compiler/version.info"
    if not version.is_file() or not re.search(
        r"^Version=9\.2\.1$", version.read_text(), re.MULTILINE
    ):
        parser.error("requires installed CANN 9.2.1 compiler metadata")
    if atb.name not in {"cxx_abi_0", "cxx_abi_1"}:
        parser.error("select the ATB cxx_abi directory matching the runtime torch")
    atb_version = atb.parents[1] / "version.info"
    if not atb_version.is_file() or not re.search(
        r"^\s*Ascend-cann-atb\s*:\s*9\.2\.1$", atb_version.read_text(), re.MULTILINE
    ):
        parser.error("requires installed ATB 9.2.1 metadata")
    runtime = root / "env/runtime"
    site = runtime / "lib/python3.12/site-packages"
    versions = {
        package.metadata["Name"].lower().replace("_", "-"): package.version
        for package in importlib.metadata.distributions(path=[str(site)])
    }
    for name, version in (("torch", "2.10.0"), ("torch-npu", "2.10.0.post7")):
        if versions.get(name) != version:
            parser.error(f"native build requires {name}=={version}")
    for required in (
        cann / "include/acl/acl.h",
        cann / "lib64/libascendcl.so",
        atb / "include/atb/atb_infer.h",
        atb / "lib/libatb.so",
        runtime / "bin/python",
        site / "torch/include/torch/extension.h",
        cann / "tools/tikcpp/ascendc_kernel_cmake/ascendc.cmake",
        cann / "tools/ccec_compiler/bin/bisheng",
    ):
        if not required.is_file() or not required.resolve().is_relative_to(root):
            parser.error(f"missing private build prerequisite: {required}")
    values = dict(
        PATH=f"{runtime}/bin:{cann}/bin:{cann}/tools/ccec_compiler/bin:/usr/bin:/bin",
        PYTHONPATH=f"{cann}/python/site-packages:"
        f"{cann}/opp/built-in/op_impl/ai_core/tbe",
        LD_LIBRARY_PATH=f"{atb}/lib:{cann}/lib64:{site}/torch/lib:"
        f"{site}/torch_npu/lib:{root}/python/lib:"
        f"{cann}/lib64/plugin/opskernel:{cann}/lib64/plugin/nnengine:"
        f"{cann}/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64",
        LIBRARY_PATH=f"{atb}/lib:{cann}/lib64:{cann}/lib64/stub:{cann}/devlib",
        # Resolve indirect driver dependencies for compiler link checks only.
        # rpath-link is not embedded as a runtime search path in the output.
        LDFLAGS=f"-Wl,-rpath-link,{cann}/lib64:{cann}/devlib",
        CPATH=f"{atb}/include:{cann}/include",
        CMAKE_PREFIX_PATH=f"{site}/torch/share/cmake:{site}/pybind11/share/cmake:"
        f"{cann}/lib64/cmake:{cann}/toolkit/tools/tikicpulib/lib/cmake",
        TOOLCHAIN_HOME=str(cann / "toolkit"),
        ASCEND_HOME_PATH=str(cann),
        ASCEND_TOOLKIT_HOME=str(cann),
        ASCEND_AICPU_PATH=str(cann),
        ASCEND_OPP_PATH=str(cann / "opp"),
        ATB_HOME_PATH=str(atb),
        TORCH_NPU_PATH=str(site / "torch_npu"),
        VIRTUAL_ENV=str(runtime),
        PYTHONNOUSERSITE="1",
        TORCH_DEVICE_BACKEND_AUTOLOAD="0",
        VLLM_PLUGINS="",
        VLLM_USE_V2_MODEL_RUNNER="0",
        SOC_VERSION="ascend910b2",
        MAX_JOBS="2",
        CMAKE_BUILD_PARALLEL_LEVEL="2",
        CARGO_BUILD_JOBS="2",
        TMPDIR=str(root / "tmp"),
        XDG_CACHE_HOME=str(root / "cache"),
        UV_CACHE_DIR=str(root / "cache/uv"),
        PIP_CACHE_DIR=str(root / "cache/pip"),
        HF_HOME=str(root / "cache/huggingface"),
        VLLM_CACHE_ROOT=str(root / "cache/vllm"),
        TORCH_EXTENSIONS_DIR=str(root / "cache/torch-extensions"),
        TRITON_CACHE_DIR=str(root / "cache/triton"),
        ASCEND_PROCESS_LOG_PATH=str(root / "runs/ascend"),
    )
    inherited = {
        key
        for key in os.environ
        if key.startswith(("ASCEND_", "ATB_", "CANN_", "TRITON_", "HCCL_"))
    }
    inherited.update({"LD_PRELOAD", "PYTHONHOME", "PYTHONUSERBASE", "TOOLCHAIN_HOME"})
    print("unset " + " ".join(shlex.quote(key) for key in sorted(inherited)))
    for key, value in values.items():
        print(f"export {key}={shlex.quote(value)}")


if __name__ == "__main__":
    main()
