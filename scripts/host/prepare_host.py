"""Prepare independent Python environments and verified host source archives."""

import argparse
import hashlib
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

from build_host import apply_patch

ARCHIVES = dict(
    [
        (
            "vllm-ced6857afa0ea7b2e3f0846a62e1394e90f15607",
            "5309f930285a26629e3e49a8d022f81c0f99b2490e24fa86dddbd2cd4986dfee",
        ),
        (
            "vllm-ascend-c09d3e20b9c45c80b34ae3781440d90904a44023",
            "8d3a91782da1120681624a79fdc3c38b93d0a20f13cf88b966e8a47e8437b35a",
        ),
    ]
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--python-prefix", type=Path, required=True)
    parser.add_argument(
        "--resolve",
        action="store_true",
        help="re-resolve the combined input before installing",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to("/root/knorm"):
        raise ValueError("server preparation is confined to /root/knorm")
    for relative in (
        "python/bin",
        "python/lib",
        "env",
        "cache",
        "tmp",
        "src",
        "runs",
        "artifacts/wheels",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    if not (root / "python/bin/python3.12").exists():
        source = args.python_prefix.resolve()
        shutil.copy2(source / "bin/python3.12", root / "python/bin/python3.12")
        for library in (source / "lib").glob("libpython3.12*"):
            if library.is_file():
                shutil.copy2(library, root / "python/lib" / library.name)
        shutil.copytree(
            source / "lib/python3.12",
            root / "python/lib/python3.12",
            ignore=shutil.ignore_patterns("site-packages", "__pycache__"),
        )
        shutil.copytree(source / "include", root / "python/include")
    os.environ.pop("PYTHONPATH", None)
    os.environ.update(
        PYTHONNOUSERSITE="1",
        TORCH_DEVICE_BACKEND_AUTOLOAD="0",
        VLLM_PLUGINS="",
        TMPDIR=str(root / "tmp"),
        XDG_CACHE_HOME=str(root / "cache"),
        UV_CACHE_DIR=str(root / "cache/uv"),
        PIP_CACHE_DIR=str(root / "cache/pip"),
        CARGO_HOME=str(root / "cache/cargo"),
        RUSTUP_HOME=str(root / "cache/rustup"),
        MAX_JOBS="2",
        CMAKE_BUILD_PARALLEL_LEVEL="2",
        CARGO_BUILD_JOBS="2",
    )
    for kind in ("build", "runtime"):
        if not (root / f"env/{kind}/bin/python").exists():
            subprocess.run(
                [
                    str(root / "python/bin/python3.12"),
                    "-m",
                    "venv",
                    str(root / f"env/{kind}"),
                ],
                check=True,
            )
    plugin = root / "src/knorm"
    for name, digest in ARCHIVES.items():
        archive = root / "artifacts" / f"{name}.tar.gz"
        if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
            raise ValueError(f"source archive mismatch: {name}")
        source = root / "src" / name
        if not source.exists():
            with tarfile.open(archive) as stream:
                stream.extractall(root / "src", filter="data")
        project = "vllm-ascend" if "ascend" in name else "vllm"
        for patch in sorted(
            (plugin / "host-patches" / f"{project}-0.30.0").glob("*.patch")
        ):
            apply_patch(source, patch)
    subprocess.run(
        [
            str(root / "env/build/bin/python"),
            "-m",
            "pip",
            "install",
            "-r",
            str(plugin / "host-patches/environment/build.lock"),
        ],
        check=True,
    )
    uv = str(root / "env/build/bin/uv")
    runtime = str(root / "env/runtime/bin/python")
    core, ascend = [root / "src" / name for name in ARCHIVES]
    requirements = root / "artifacts/runtime.in"
    requirements.write_text(
        (core / "requirements/common.txt").read_text()
        + "\n"
        + (ascend / "requirements.txt").read_text()
    )
    lock = root / "artifacts/runtime.lock"
    index = [
        "--index",
        "https://mirrors.huaweicloud.com/ascend/repos/pypi",
        "--index-strategy",
        "unsafe-best-match",
    ]
    if args.resolve:
        subprocess.run(
            [
                uv,
                "pip",
                "compile",
                "--no-config",
                "--python",
                runtime,
                "--generate-hashes",
                *index,
                str(requirements),
                "-o",
                str(lock),
            ],
            check=True,
        )
    else:
        shutil.copy2(plugin / "host-patches/environment/runtime.lock", lock)
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--python",
            runtime,
            *index,
            "--require-hashes",
            "-r",
            str(lock),
        ],
        check=True,
    )
    subprocess.run([runtime, "-m", "pip", "check"], check=True)


if __name__ == "__main__":
    main()
