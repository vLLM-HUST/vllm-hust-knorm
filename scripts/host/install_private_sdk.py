"""Install the pinned SDK in a private root filesystem without device access.

The vendor installers run unchanged. Their absolute registry/log paths resolve
inside the private root. No host directories are mounted into it. The published
SDK symlink preserves the logical install prefix used inside that root.
"""

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

from private_install_process import confine


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(root, jail):
    marker = jail / ".prepared.json"
    if marker.exists():
        if not (jail / "usr/share").exists():
            shutil.copytree("/usr/share", jail / "usr/share", symlinks=True)
        return
    if jail.exists():
        raise ValueError(f"refusing incomplete private root: {jail}")
    if shutil.disk_usage(root).free < 20 * 1024**3:
        raise ValueError("at least 20 GiB free required before SDK installation")
    jail.mkdir(parents=True)
    for name in (
        "usr/bin",
        "usr/sbin",
        "usr/lib",
        "usr/lib64",
        "usr/libexec",
        "usr/share",
    ):
        # Copy, never hardlink shared files: installer chmod/chown must stay local.
        shutil.copytree("/" + name, jail / name, symlinks=True)
    for name in ("bin", "sbin", "lib", "lib64"):
        (jail / name).symlink_to("usr/" + name)
    for name in ("etc", "dev", "proc", "sys", "tmp", "var/log", "root", "run"):
        (jail / name).mkdir(parents=True, exist_ok=True)
    for name in ("os-release", "openEuler-release", "ld.so.cache", "nsswitch.conf"):
        source = Path("/etc") / name
        if source.exists():
            shutil.copy2(source, jail / "etc" / name)
    (jail / "etc/passwd").write_text("root:x:0:0:root:/root:/bin/bash\n")
    (jail / "etc/group").write_text("root:x:0:\n")
    for name in ("profile", "bashrc"):
        (jail / "etc" / name).touch()
    for name, minor in (("null", 3), ("zero", 5), ("random", 8), ("urandom", 9)):
        os.mknod(jail / "dev" / name, stat.S_IFCHR | 0o666, os.makedev(1, minor))
    private = jail / root.relative_to("/")
    shutil.copytree(root / "python", private / "python", symlinks=True)
    for name in ("tmp", "cache/pip", "toolchains/ascend", "artifacts/toolchains"):
        (private / name).mkdir(parents=True, exist_ok=True)
    # Remove privilege-granting mode bits from private copies only.
    for directory, _, files in os.walk(jail):
        for name in files:
            path = Path(directory) / name
            mode = path.lstat().st_mode
            if stat.S_ISREG(mode) and mode & (stat.S_ISUID | stat.S_ISGID):
                path.chmod(stat.S_IMODE(mode) & ~stat.S_ISUID & ~stat.S_ISGID)
    marker.write_text(json.dumps({"root": str(root), "driver_exposed": False}) + "\n")


def enter(jail, root):
    # subprocess closes inherited descriptors before executing the installer.
    # Landlock restricts access before chroot; chroot confines chmod/chown and
    # absolute symlinks, which Landlock ABI 1 alone cannot restrict.
    confine(jail)
    os.chroot(jail)
    os.chdir(root)
    libc = ctypes.CDLL(None, use_errno=True)
    # Archive ownership/mode restoration needs these filesystem capabilities.
    # Never retain mount, chroot, device creation, ptrace or file-handle access.
    filesystem_caps = {0, 1, 3, 4}
    for capability in set(range(41)) - filesystem_caps:
        if libc.prctl(24, capability, 0, 0, 0) != 0:  # PR_CAPBSET_DROP
            raise OSError(ctypes.get_errno(), "could not drop capability")

    class Header(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]

    class Caps(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_uint32)
            for name in ("effective", "permitted", "inheritable")
        ]

    header = Header(0x20080522, 0)
    caps = (Caps * 2)()
    caps[0].effective = caps[0].permitted = sum(1 << c for c in filesystem_caps)
    if libc.capset(ctypes.byref(header), ctypes.byref(caps)) != 0:
        raise OSError(ctypes.get_errno(), "could not clear capabilities")


def environment(root):
    env = {
        key: os.environ[key] for key in ("HOME", "USER", "LOGNAME") if key in os.environ
    }
    env.update(
        PATH=f"{root}/env/sdk-installer/bin:{root}/python/bin:"
        "/usr/bin:/usr/sbin:/bin:/sbin",
        LD_LIBRARY_PATH=f"{root}/python/lib",
        LANG="C",
        LC_ALL="C",
        TZ="UTC",
        PYTHONNOUSERSITE="1",
        PIP_NO_INDEX="1",
        PIP_DISABLE_PIP_VERSION_CHECK="1",
        PIP_CACHE_DIR=f"{root}/cache/pip",
        TMPDIR=f"{root}/tmp",
        OMP_NUM_THREADS="2",
        MAX_JOBS="2",
    )
    return env


def run(root, jail, command, log):
    original = log
    attempt = 1
    while log.exists():
        log = original.with_stem(f"{original.stem}-{attempt}")
        attempt += 1
    env = environment(root)
    cann = root / "toolchains/ascend/cann-9.2.1"
    if (jail / cann.relative_to("/")).is_dir():
        env.update(
            ASCEND_HOME_PATH=str(cann),
            ASCEND_TOOLKIT_HOME=str(cann),
            ASCEND_OPP_PATH=str(cann / "opp"),
        )
    with log.open("w") as stream:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=stream,
            env=env,
            preexec_fn=lambda: enter(jail, root),
            close_fds=True,
            check=False,
        )
    print(
        json.dumps(
            {"command": command, "exit_code": result.returncode, "log": str(log)}
        ),
        flush=True,
    )
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    return log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--install", choices=("toolkit", "910b-ops", "nnal"))
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to("/root/knorm") or root == Path("/root/knorm"):
        parser.error("use an isolated task directory below /root/knorm")
    if os.getuid() != 0 or os.uname().machine != "aarch64":
        parser.error("requires Linux aarch64 and permission to create a private chroot")
    jail = root / "toolchains/sdk-root"
    if not jail.resolve().is_relative_to(root):
        parser.error("private root escaped the task directory")
    logs = root / "runs/private-sdk"
    logs.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        prepare(root, jail)
        installer_env = root / "env/sdk-installer"
        if not (jail / installer_env.relative_to("/") / "bin/pip").exists():
            run(
                root,
                jail,
                [str(root / "python/bin/python3.12"), "-m", "venv", str(installer_env)],
                logs / "installer-python.log",
            )
    if not (jail / ".prepared.json").is_file():
        parser.error("prepare the private root first")
    if args.probe:
        probe = (
            "import ctypes,os,pathlib; "
            "p=pathlib.Path; assert sorted(x.name for x in p('/dev').iterdir()) == "
            "['null','random','urandom','zero']; "
            "assert not list(p('/proc').iterdir()); "
            "assert not list(p('/sys').iterdir()); "
            "libc=ctypes.CDLL(None,use_errno=True); "
            "assert libc.prctl(39,0,0,0,0)==1; "
            "assert all(libc.prctl(23,c,0,0,0)==int(c in {0,1,3,4}) "
            "for c in range(41)); "
            "assert libc.chroot(b'/')==-1 and ctypes.get_errno()==1; "
            "p('/etc/isolation-check').write_text('private root only\\n'); "
            "print('private /etc write; no device/proc/sys exposure; '"
            "'only filesystem capabilities retained; chroot denied')"
        )
        run(
            root,
            jail,
            [str(root / "python/bin/python3.12"), "-c", probe],
            logs / "isolation.log",
        )
        if (
            not (jail / "etc/isolation-check").is_file()
            or Path("/etc/isolation-check").exists()
        ):
            raise RuntimeError("private filesystem boundary check failed")
    if args.install:
        manifest = (
            Path(__file__).parents[2]
            / "host-patches/environment/native-components.json"
        )
        filename = f"Ascend-cann-{args.install}_9.2.1_linux-aarch64.run"
        entry = next(
            x
            for x in json.loads(manifest.read_text())["artifacts"]
            if x["filename"] == filename
        )
        source = root / "artifacts/toolchains" / filename
        if digest(source) != entry["sha256"]:
            raise ValueError("vendor installer digest mismatch")
        record = logs / f"{args.install}-installed.json"
        if record.exists():
            raise ValueError(
                "installation already recorded; inspect it instead of overwriting"
            )
        target = jail / source.relative_to("/")
        if not target.exists():
            shutil.copy2(source, target)
        if digest(target) != entry["sha256"]:
            raise ValueError("private installer copy digest mismatch")
        command = [
            "/bin/bash",
            str(source),
            "--install",
            "--quiet",
            f"--install-path={root}/toolchains/ascend",
        ]
        if args.install == "toolkit":
            # Vendor-supported selection covers runtime, Ascend C and dev tools.
            # Optional profiler/PyPTO installers are outside this build's needs.
            command.append("--whitelist=atc,devtools,nnrt")
        log = run(root, jail, command, logs / f"{args.install}-install.log")
        record.write_text(
            json.dumps(
                dict(
                    entry,
                    command=command,
                    exit_code=0,
                    log=str(log),
                    method="vendor_installer_in_private_chroot",
                ),
                indent=2,
            )
            + "\n"
        )
    if args.publish:
        for component in ("toolkit", "910b-ops", "nnal"):
            if not (logs / f"{component}-installed.json").is_file():
                raise ValueError(f"missing successful installer record: {component}")
        prefix = root / "toolchains/ascend"
        installed = jail / prefix.relative_to("/")
        if prefix.is_symlink() and prefix.resolve() == installed:
            return
        if prefix.exists() or prefix.is_symlink():
            raise ValueError("refusing to overwrite an existing SDK prefix")
        prefix.symlink_to(installed)


if __name__ == "__main__":
    main()
