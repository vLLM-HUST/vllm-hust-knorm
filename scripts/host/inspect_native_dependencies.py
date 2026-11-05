"""Record native build prerequisites using metadata and ELF headers only."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to("/root/knorm") or not Path(sys.prefix).is_relative_to(
        root
    ):
        raise ValueError("use the private runtime under /root/knorm")
    destination = root / "runs/dependencies"
    destination.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update(
        PATH=f"{root}/env/runtime/bin:/usr/bin:/bin",
        LD_LIBRARY_PATH=str(root / "python/lib"),
        PYTHONNOUSERSITE="1",
        TORCH_DEVICE_BACKEND_AUTOLOAD="0",
        TMPDIR=str(root / "tmp"),
        XDG_CACHE_HOME=str(root / "cache"),
    )
    commands = []

    def run(command):
        result = subprocess.run(
            command, env=environment, capture_output=True, text=True
        )
        commands.append(
            dict(
                command=command,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        )
        return result

    distributions = {}
    for package in importlib.metadata.distributions():
        name = re.sub(r"[-_.]+", "-", package.metadata["Name"]).lower()
        distributions[name] = dict(
            version=package.version,
            path=str(package.locate_file("")),
            origin=package.read_text("direct_url.json"),
            wheel=package.read_text("WHEEL"),
        )
    lock = root / "src/knorm/host-patches/environment/runtime.lock"
    mismatches = []
    for name, version in re.findall(
        r"^([\w.-]+)==([^\s;]+)", lock.read_text(), re.MULTILINE
    ):
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        actual = distributions.get(normalized, {}).get("version")
        if actual != version:
            mismatches.append(dict(package=name, expected=version, actual=actual))
    tools = {}
    for name, args_ in (
        ("gcc", ["--version"]),
        ("g++", ["--version"]),
        ("cmake", ["--version"]),
        ("ninja", ["--version"]),
        ("readelf", ["--version"]),
    ):
        executable = shutil.which(name, path=environment["PATH"])
        tools[name] = executable
        if executable:
            run([executable, *args_])
    run([sys.executable, "-m", "pip", "check"])
    run(["/usr/bin/getconf", "GNU_LIBC_VERSION"])
    run([tools["readelf"], "--version-info", "/usr/lib64/libstdc++.so.6"])
    header = Path(sysconfig.get_path("include")) / "Python.h"
    if not header.resolve().is_relative_to(root):
        raise ValueError(f"Python header escaped private prefix: {header}")
    source = destination / "compiler_headers.cpp"
    source.write_text(
        "#include <Python.h>\n#include <numa.h>\n#include <optional>\n"
        'extern "C" int headers_link() { std::optional<int> n{1}; '
        "return *n + numa_available(); }\n"
    )
    output = destination / "compiler_headers.so"
    run(
        [
            tools["g++"],
            "-std=c++17",
            "-shared",
            "-fPIC",
            f"-I{header.parent}",
            str(source),
            "-lnuma",
            "-o",
            str(output),
        ]
    )
    # Only compile/link this CPU probe; do not load or run the resulting library.
    roots = [root / "toolchains/unpacked", Path(sysconfig.get_path("purelib"))]
    basenames = {
        "libascendcl.so",
        "libatb.so",
        "libtorch_npu.so",
        "libhccl.so",
        "libnnopbase.so",
        "libtriton.so",
        "bisheng",
        "ccec",
    }
    selected, candidates, headers = [], {}, []
    for search in roots:
        for path in search.rglob("*"):
            if path.is_file():
                if ".so" in path.name:
                    candidates.setdefault(path.name, []).append(str(path.resolve()))
                if path.name in basenames:
                    selected.append(path)
                if path.name in {"acl.h", "kernel_operator.h", "atb_infer.h"}:
                    headers.append(str(path))
    selected.extend([Path("/usr/lib64/libnuma.so.1"), output])
    libraries = []
    for path in sorted(set(selected)):
        if not path.exists():
            continue
        with path.open("rb") as stream:
            if stream.read(4) != b"\x7fELF":
                continue
        result = run([tools["readelf"], "-h", "-d", str(path)])
        needed = re.findall(r"\(NEEDED\).*?\[(.*?)\]", result.stdout)
        machine = re.search(r"Machine:\s*(.*)", result.stdout)
        libraries.append(
            dict(
                path=str(path),
                machine=machine.group(1) if machine else None,
                needed=needed,
                candidates={
                    name: candidates.get(name, [])
                    + [
                        str(p.resolve())
                        for folder in ("/usr/lib64", "/lib64")
                        if (p := Path(folder) / name).exists()
                    ]
                    for name in needed
                },
            )
        )
    record = dict(
        python=sys.version,
        executable=sys.executable,
        machine=platform.machine(),
        python_header=str(header),
        free_bytes=shutil.disk_usage(root).free,
        tools=tools,
        runtime_lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),
        lock_mismatches=mismatches,
        packages=distributions,
        libraries=libraries,
        headers=headers,
        commands=commands,
        device_imports=[
            name for name in ("torch", "torch_npu", "acl") if name in sys.modules
        ],
    )
    (destination / "static-inventory.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                key: record[key]
                for key in (
                    "executable",
                    "python_header",
                    "free_bytes",
                    "tools",
                    "lock_mismatches",
                    "device_imports",
                )
            },
            indent=2,
        )
    )
    print(f"ELF records: {len(libraries)}; report: {destination}/static-inventory.json")
    if mismatches or record["device_imports"] or any(c["exit_code"] for c in commands):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
