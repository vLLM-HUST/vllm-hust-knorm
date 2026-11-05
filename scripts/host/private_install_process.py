"""Run audited SDK installers with project-only filesystem write access."""

import argparse
import ctypes
import os
from pathlib import Path


class Ruleset(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class PathRule(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def confine(root, *, allow_cpu_topology=False):
    libc = ctypes.CDLL(None, use_errno=True)

    def checked(result):
        if result < 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))
        return result

    abi = checked(libc.syscall(444, 0, 0, 1))
    # Use the filesystem rights supported since Landlock ABI 1. Metadata changes
    # such as chmod are not covered; installer hooks must also be audited.
    handled = (1 << 13) - 1
    ruleset = Ruleset(handled)
    descriptor = checked(
        libc.syscall(444, ctypes.byref(ruleset), ctypes.sizeof(ruleset), 0)
    )
    read_execute = (1 << 0) | (1 << 2) | (1 << 3)
    entries = [(root, handled & ~(1 << 6) & ~(1 << 11))]
    entries += [
        (Path(p), read_execute)
        for p in ("/usr", "/etc", "/proc", "/bin", "/sbin", "/lib", "/lib64")
    ]
    if allow_cpu_topology:
        entries.append((Path("/sys/devices/system/cpu"), read_execute))
    entries += [
        (Path(p), (1 << 1) | (1 << 2))
        for p in ("/dev/null", "/dev/zero", "/dev/random", "/dev/urandom")
    ]
    try:
        for path, rights in entries:
            if not path.exists():
                continue
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = PathRule(rights, fd)
                checked(libc.syscall(445, descriptor, 1, ctypes.byref(rule), 0))
            finally:
                os.close(fd)
        checked(libc.prctl(38, 1, 0, 0, 0))  # PR_SET_NO_NEW_PRIVS
        checked(libc.syscall(446, descriptor, 0))
    finally:
        os.close(descriptor)
    return abi


def check_boundary(root):
    # O_WRONLY without O_TRUNC does not alter the file even if the policy fails.
    try:
        fd = os.open("/etc/passwd", os.O_WRONLY)
    except PermissionError:
        pass
    else:
        os.close(fd)
        raise RuntimeError("shared-file write protection failed")
    try:
        with os.scandir("/dev"):
            pass
    except PermissionError:
        pass
    else:
        raise RuntimeError("device directory access was not denied")
    probe = root / "tmp" / f"private-install-{os.getpid()}"
    probe.write_text("private write check\n")
    probe.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to("/root/knorm"):
        parser.error("installation is confined to /root/knorm")
    (root / "tmp").mkdir(parents=True, exist_ok=True)
    abi = confine(root)
    check_boundary(root)
    print(
        f"Landlock ABI {abi}: private write/device-directory checks passed", flush=True
    )
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if command:
        os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    main()
