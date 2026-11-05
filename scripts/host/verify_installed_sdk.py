"""Check the installed SDK, its Python dependencies and host link interfaces.

Run with the exports from native_build_environment.py. Device backends and
device files are excluded. Compiled link probes are never executed or loaded.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import sysconfig
from pathlib import Path

from private_install_process import confine
from test_without_devices import DeviceImportsForbidden


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to("/root/knorm"):
        parser.error("verification requires the private task root")
    cann, atb = (Path(os.environ[key]) for key in ("ASCEND_HOME_PATH", "ATB_HOME_PATH"))
    for path in (cann, atb, Path(sys.prefix)):
        if not path.resolve().is_relative_to(root):
            parser.error(f"external dependency prefix: {path}")
    confine(root, allow_cpu_topology=True)
    sys.meta_path.insert(0, DeviceImportsForbidden())
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    destination = root / "runs/private-sdk"
    commands = []
    failures = []

    def run(command):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        commands.append(
            dict(
                command=command,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        )
        if result.returncode:
            failures.append(command)
        return result.stdout

    # CPU torch import only, with autoload disabled and backend imports denied.
    import torch

    abi = int(torch.compiled_with_cxx11_abi())
    if atb.name != f"cxx_abi_{abi}":
        failures.append("ATB C++ ABI differs from the installed torch")
    site = Path(sysconfig.get_path("purelib"))
    versions = {}
    for path in (
        cann / "compiler/version.info",
        cann / "opp/version.info",
        atb.parents[1] / "version.info",
    ):
        if not path.is_file():
            failures.append(f"missing version metadata: {path}")
            continue
        versions[str(path)] = path.read_text()
        if not re.search(
            r"^\s*(?:(?:Version|version)=|Ascend-cann-atb\s*:\s*)9\.2\.1$",
            path.read_text(),
            re.MULTILINE,
        ):
            failures.append(f"unexpected component version: {path}")
    required = [
        cann / f"lib64/lib{name}.so"
        for name in (
            "ascendcl",
            "register",
            "platform",
            "ascendalog",
            "opapi",
        )
    ]
    required += [
        cann / "lib64/libtiling_api.a",
        atb / "lib/libatb.so",
        site / "torch_npu/lib/libtorch_npu.so",
        site / "torch/lib/libtorch_cpu.so",
    ]
    required += [
        cann / "include/acl/acl.h",
        atb / "include/atb/atb_infer.h",
        cann / "tools/tikcpp/ascendc_kernel_cmake/ascendc.cmake",
        cann / "tools/tikcpp/tikcfw/kernel_operator.h",
    ]
    identities = {}
    for path in required:
        if not path.is_file() or not path.resolve().is_relative_to(root):
            failures.append(f"missing or external prerequisite: {path}")
            continue
        with path.open("rb") as stream:
            identities[str(path)] = dict(
                resolved=str(path.resolve()),
                sha256=hashlib.file_digest(stream, "sha256").hexdigest(),
            )
    run([sys.executable, "-m", "pip", "check"])
    run([str(cann / "tools/ccec_compiler/bin/bisheng"), "--version"])
    archive_headers = run(
        ["/usr/bin/readelf", "-h", str(cann / "lib64/libtiling_api.a")]
    )
    if set(re.findall(r"Machine:\s*(.*)", archive_headers)) != {"AArch64"}:
        failures.append("tiling_api static archive architecture mismatch")
    source = destination / "sdk_headers.cpp"
    source.write_text(
        "#include <Python.h>\n#include <numa.h>\n#include <acl/acl.h>\n"
        "#include <atb/atb_infer.h>\n"
        'extern "C" int check_headers() { atb::Context *context = nullptr; '
        "return atb::CreateContext(&context) + aclrtGetDeviceCount(nullptr) + "
        "static_cast<int>(atb::Utils::GetAtbVersion().size()); }\n"
    )
    run(
        [
            "/usr/bin/g++",
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-Wl,--no-undefined",
            f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
            f"-I{sysconfig.get_path('include')}",
            f"-I{cann}/include",
            f"-I{atb}/include",
            str(source),
            f"-L{cann}/lib64",
            f"-L{atb}/lib",
            "-latb",
            "-lascendcl",
            "-lnuma",
            "-o",
            str(destination / "sdk_headers.so"),
        ]
    )
    project = destination / "acl-cmake"
    project.mkdir(exist_ok=True)
    (project / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\n"
        "project(private_sdk_link LANGUAGES CXX)\n"
        "include(CheckCXXSourceCompiles)\n"
        'set(CMAKE_REQUIRED_INCLUDES "$ENV{ASCEND_HOME_PATH}/include")\n'
        "set(CMAKE_REQUIRED_LIBRARIES ascendcl)\n"
        'set(CMAKE_REQUIRED_LINK_OPTIONS "-L$ENV{ASCEND_HOME_PATH}/lib64")\n'
        'check_cxx_source_compiles("#include <acl/acl_rt.h>\n'
        'int main() { int32_t id=0; return aclrtGetPhyDevIdByUserDevId(0, &id); }" '
        "HAVE_DEVICE_ID_MAPPING)\n"
        "if(NOT HAVE_DEVICE_ID_MAPPING)\n"
        'message(FATAL_ERROR "CANN device-ID interface did not compile/link")\n'
        "endif()\n"
    )
    run(
        [
            str(Path(sys.prefix) / "bin/cmake"),
            "--fresh",
            "-S",
            str(project),
            "-B",
            str(project / "build"),
            "-G",
            "Ninja",
        ]
    )

    # Resolve ELF dependency names statically; never invoke ldd/dlopen or run a
    # vendor binary to test runtime initialization. Driver stubs are link-only.
    search = [Path(p) for p in os.environ["LD_LIBRARY_PATH"].split(":") if p]
    search += [Path(p) for p in ("/usr/lib64", "/usr/lib", "/lib64", "/lib")]
    allowed = [root, Path("/usr/lib64"), Path("/usr/lib")]
    stubs = [cann / "lib64/stub", cann / "devlib", cann / "aarch64-linux/devlib"]
    queue = [path for path in required if path.is_file() and ".so" in path.name]
    graph, unresolved = {}, {}
    while queue:
        path = queue.pop().resolve()
        if str(path) in graph:
            continue
        headers = run(["/usr/bin/readelf", "-h", "-d", str(path)])
        if "AArch64" not in headers:
            failures.append(f"unexpected ELF architecture: {path}")
        needed = re.findall(r"\(NEEDED\).*\[(.*?)\]", headers)
        rpath = re.findall(r"\((?:RUNPATH|RPATH)\).*\[(.*?)\]", headers)
        local = [
            Path(
                p.replace("${ORIGIN}", str(path.parent)).replace(
                    "$ORIGIN", str(path.parent)
                )
            )
            for group in rpath
            for p in group.split(":")
            if p
        ]
        resolved = {}
        for name in needed:
            candidate = next(
                (
                    directory / name
                    for directory in search + local
                    if (directory / name).is_file()
                    and any(
                        (directory / name).resolve().is_relative_to(base)
                        for base in allowed
                    )
                ),
                None,
            )
            if candidate:
                resolved[name] = str(candidate.resolve())
                queue.append(candidate)
            else:
                stub = next(
                    (
                        directory / name
                        for directory in stubs
                        if (directory / name).is_file()
                    ),
                    None,
                )
                unresolved[name] = dict(link_stub=str(stub) if stub else None)
                if name != "libascend_hal.so" or stub is None:
                    failures.append(f"unresolved user-space library: {name}")
                resolved[name] = None
        graph[str(path)] = resolved
    distributions = {
        d.metadata["Name"]: dict(version=d.version, path=str(d.locate_file("")))
        for d in importlib.metadata.distributions()
    }
    baseline_path = root / "artifacts/native-dependencies-environment.json"
    baseline = json.loads(baseline_path.read_text())

    def normalize(name):
        return re.sub(r"[-_.]+", "-", name).lower()

    before = {
        normalize(name): data["version"] for name, data in baseline["packages"].items()
    }
    after = {
        normalize(d.metadata["Name"]): d.version
        for d in importlib.metadata.distributions(path=[str(site)])
    }
    changes = {
        name: [version, after.get(name)]
        for name, version in before.items()
        if after.get(name) != version
    }
    added = {name: version for name, version in after.items() if name not in before}
    if changes or added != {"absl-py": "2.5.1"}:
        failures.append("unexpected runtime changes relative to dependency baseline")
    artifacts = {}
    for relative, expected in baseline["artifacts"].items():
        with (root / relative).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        artifacts[relative] = actual
        if actual != expected:
            failures.append(f"previous artifact changed: {relative}")
    forbidden = sorted(
        name for name in sys.modules if name.split(".")[0] in {"torch_npu", "acl"}
    )
    if forbidden:
        failures.append("device module imported")
    report = dict(
        torch_cxx11_abi=abi,
        atb=str(atb),
        versions=versions,
        files=identities,
        commands=commands,
        elf_dependencies=graph,
        runtime_unresolved=unresolved,
        distributions=distributions,
        runtime_delta=dict(changed_or_removed=changes, added=added),
        previous_artifacts=artifacts,
        device_modules=forbidden,
        failures=failures,
        link_probe_executed=False,
        driver_runtime_tested=False,
    )
    (destination / "installed-sdk.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "torch_cxx11_abi",
                    "runtime_unresolved",
                    "device_modules",
                    "failures",
                )
            }
        )
    )
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
