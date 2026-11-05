"""Build the pinned, patched sources in a task-owned environment without devices."""

import argparse
import os
import subprocess
from pathlib import Path


def apply_patch(source, patch):
    command = ["git", "apply", "--check", str(patch)]
    if subprocess.run(command, cwd=source, capture_output=True).returncode == 0:
        subprocess.run(["git", "apply", str(patch)], cwd=source, check=True)
    else:
        subprocess.run(
            ["git", "apply", "--reverse", "--check", str(patch)], cwd=source, check=True
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    plugin = root / "src/knorm"
    core = root / "src/vllm-ced6857afa0ea7b2e3f0846a62e1394e90f15607"
    ascend = root / "src/vllm-ascend-c09d3e20b9c45c80b34ae3781440d90904a44023"
    os.environ.pop("PYTHONPATH", None)
    os.environ.update(
        PYTHONNOUSERSITE="1",
        TORCH_DEVICE_BACKEND_AUTOLOAD="0",
        VLLM_PLUGINS="",
        VLLM_TARGET_DEVICE="empty",
        SOC_VERSION="ascend910b2",
        COMPILE_CUSTOM_KERNELS="0",
        MAX_JOBS="2",
        CMAKE_BUILD_PARALLEL_LEVEL="2",
        CARGO_BUILD_JOBS="2",
        TMPDIR=str(root / "tmp"),
        XDG_CACHE_HOME=str(root / "cache"),
        PIP_CACHE_DIR=str(root / "cache/pip"),
        UV_CACHE_DIR=str(root / "cache/uv"),
        CARGO_HOME=str(root / "cache/cargo"),
        RUSTUP_HOME=str(root / "cache/rustup"),
        SETUPTOOLS_SCM_PRETEND_VERSION="0.30.0+knorm",
    )
    for project, source in (("vllm", core), ("vllm-ascend", ascend)):
        for patch in sorted(
            (plugin / "host-patches" / f"{project}-0.30.0").glob("*.patch")
        ):
            apply_patch(source, patch)
    python = root / "env/build/bin/python"
    runtime = root / "env/runtime/bin/python"
    # Each build uses its own declared torch generation with autoload disabled.
    subprocess.run(
        [
            str(root / "env/build/bin/uv"),
            "pip",
            "install",
            "--python",
            str(runtime),
            "build==1.6.1",
            "pytest==8.4.2",
            "setuptools-scm==10.3.4",
            "wheel==0.48.0",
        ],
        check=True,
    )
    for source, interpreter in ((core, python), (ascend, runtime)):
        with (root / "runs" / f"{source.name}-build.log").open("w") as log:
            subprocess.run(
                [
                    str(interpreter),
                    "-m",
                    "build",
                    "--wheel",
                    "--no-isolation",
                    "--outdir",
                    str(root / "artifacts/wheels"),
                ],
                cwd=source,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    os.environ.pop("SETUPTOOLS_SCM_PRETEND_VERSION")
    subprocess.run(
        [
            str(python),
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(root / "artifacts/wheels"),
            str(plugin),
        ],
        check=True,
    )
    wheels = sorted(str(p) for p in (root / "artifacts/wheels").glob("*.whl"))
    subprocess.run(
        [
            str(root / "env/build/bin/uv"),
            "pip",
            "install",
            "--python",
            str(runtime),
            "--index",
            "https://mirrors.huaweicloud.com/ascend/repos/pypi",
            "--index-strategy",
            "unsafe-best-match",
            "--reinstall-package",
            "vllm",
            "--reinstall-package",
            "vllm-ascend",
            "--reinstall-package",
            "vllm-hust-knorm",
            *wheels,
        ],
        check=True,
    )
    subprocess.run([str(runtime), "-m", "pip", "check"], check=True)


if __name__ == "__main__":
    main()
