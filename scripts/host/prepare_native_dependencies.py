"""Download and unpack pinned native dependencies without executing installers."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

CHUNK = 8 * 1024 * 1024


class PayloadReader:
    """Give tar/gzip a seekable view starting after the shell header."""

    def __init__(self, stream, offset):
        self.stream = stream
        self.offset = offset

    def read(self, size=-1):
        return self.stream.read(size)

    def tell(self):
        return self.stream.tell() - self.offset

    def seek(self, offset, whence=0):
        if whence == 0:
            offset += self.offset
        return self.stream.seek(offset, whence) - self.offset


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def extract_run(path, destination):
    """Verify the vendor's embedded payload checksum before offline extraction.

    The Makeself wrapper logs outside the project even with --noexec --extract.
    Read its single gzip/tar payload directly; never run any installer shell code.
    This produces inspection files, not a registered CANN installation.
    """
    with path.open("rb") as stream:
        header = stream.read(65536).decode("utf-8", errors="replace")
        values = {}
        for key in ("skip", "filesizes", "SHA"):
            match = re.search(rf'^{key}="([^"\n]+)"$', header, re.MULTILINE)
            if match is None:
                raise ValueError(f"unsupported Makeself header: {path}: {key}")
            values[key] = match[1]
        lines, size = int(values["skip"]), int(values["filesizes"])
        if not re.fullmatch(r"[0-9a-f]{64}", values["SHA"]):
            raise ValueError(f"missing payload SHA-256: {path}")
        stream.seek(0)
        for _ in range(lines):
            if not stream.readline():
                raise ValueError(f"truncated header: {path}")
        offset = stream.tell()
        if path.stat().st_size != offset + size:
            raise ValueError(f"payload size mismatch: {path}")
        digest = hashlib.sha256()
        while chunk := stream.read(CHUNK):
            digest.update(chunk)
        embedded = values["SHA"] != "0" * 64
        if embedded and digest.hexdigest() != values["SHA"]:
            raise ValueError(f"payload checksum mismatch: {path}")
        record = {"archive_sha256": sha256(path), "payload_sha256": digest.hexdigest()}
        if not embedded:
            # Makeself uses all zeros to indicate no embedded checksum. The
            # enclosing archive has already been checked against the manifest.
            record["embedded_sha256"] = "not_provided; covered by parent archive"
        marker = destination / ".payload-verified.json"
        if destination.exists():
            if marker.exists() and json.loads(marker.read_text()) == record:
                return record
            raise ValueError(
                f"refusing to overwrite incomplete extraction: {destination}"
            )
        temporary = destination.with_name(
            destination.name + f".extracting-{os.getpid()}"
        )
        temporary.mkdir(parents=True)
        stream.seek(offset)
        with tarfile.open(fileobj=PayloadReader(stream, offset), mode="r:*") as archive:
            archive.extractall(temporary, filter="data")
        (temporary / marker.name).write_text(json.dumps(record, indent=2) + "\n")
        temporary.rename(destination)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to("/root/knorm"):
        raise ValueError("server preparation is confined to /root/knorm")
    manifest = (
        Path(__file__).parents[2] / "host-patches/environment/native-components.json"
    )
    entries = json.loads(manifest.read_text())["artifacts"]
    artifacts = root / "artifacts/toolchains"
    logs = root / "runs/dependencies"
    for directory in (artifacts, logs, root / "toolchains/unpacked"):
        directory.mkdir(parents=True, exist_ok=True)
    results = []
    for entry in entries:
        path = artifacts / entry["filename"]
        result = dict(entry)
        try:
            if not path.exists() and args.download:
                if shutil.disk_usage(root).free < 20 * 1024**3:
                    raise OSError(
                        "keep at least 20 GiB free before dependency downloads"
                    )
                partial = path.with_suffix(path.suffix + ".partial")
                command = [
                    "curl",
                    "--fail",
                    "--location",
                    "--retry",
                    "2",
                    "--connect-timeout",
                    "30",
                    "--max-time",
                    "1800",
                    "--silent",
                    "--show-error",
                    "-H",
                    "Referer: https://www.hiascend.com/",
                    "--output",
                    str(partial),
                    entry["url"],
                ]
                with (logs / (path.name + ".download.log")).open("w") as log:
                    subprocess.run(command, stdout=log, stderr=log, check=True)
                if sha256(partial) != entry["sha256"]:
                    raise ValueError(f"download checksum mismatch: {path.name}")
                partial.rename(path)
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"archive checksum mismatch: {path.name}")
            result.update(status="downloaded_verified", bytes=path.stat().st_size)
            if args.extract and path.suffix == ".run":
                destination = root / "toolchains/unpacked" / path.stem
                result["payload"] = extract_run(path, destination)
                result["extracted_path"] = str(destination)
                result["status"] = "downloaded_verified_unpacked_not_installed"
                result["components"] = {}
                for child in sorted((destination / "run_package").glob("*.run")):
                    if child.name == "mindstudio-memscope_26.2.0_aarch64.run":
                        result["components"][child.name] = {
                            "archive_sha256": sha256(child),
                            "status": "retained_optional_profiler_not_unpacked",
                            "reason": "non-Makeself installer; not a build dependency",
                        }
                        continue
                    target = destination / "components" / child.stem
                    result["components"][child.name] = extract_run(child, target)
        except (
            OSError,
            ValueError,
            subprocess.SubprocessError,
            tarfile.TarError,
        ) as exc:
            result.update(status="blocked", error=str(exc))
        results.append(result)
        print(json.dumps(result), flush=True)
    (logs / "native-preparation.json").write_text(json.dumps(results, indent=2) + "\n")
    if any(item["status"] == "blocked" for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
