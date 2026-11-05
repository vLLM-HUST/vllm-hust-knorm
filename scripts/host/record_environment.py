"""Record installed distribution origins without importing device packages."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--label", default="runtime")
    args = parser.parse_args()
    root = args.root.resolve()
    packages = {}
    for distribution in importlib.metadata.distributions():
        direct = distribution.read_text("direct_url.json")
        packages[distribution.metadata["Name"]] = {
            "version": distribution.version,
            "origin": json.loads(direct)
            if direct
            else "package index (see resolver lock)",
        }
    artifacts = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for pattern in (
            "artifacts/*.tar.gz",
            "artifacts/*.lock",
            "artifacts/wheels/*.whl",
        )
        for path in root.glob(pattern)
    }
    record = {
        "python": sys.version,
        "executable": sys.executable,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "artifacts": artifacts,
        "packages": dict(sorted(packages.items())),
    }
    destination = root / "artifacts" / f"{args.label}-environment.json"
    destination.write_text(json.dumps(record, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
