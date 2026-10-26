"""Verify published file checksums without launching a simulation or sampler."""

from pathlib import Path
import argparse
import hashlib
import json
import sys


ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    manifest = ROOT / "RELEASE_MANIFEST_SHA256.json"
    if not manifest.is_file():
        print("Release manifest is missing.", file=sys.stderr)
        return 1
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    failures = []
    total_bytes = 0
    for name, expected in entries.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            failures.append({"file": name, "reason": "missing or invalid path"})
            continue
        if sha256(path) != expected:
            failures.append({"file": name, "reason": "checksum mismatch; check Git LFS download or local edits"})
        total_bytes += path.stat().st_size
    print(json.dumps({"passed": not failures, "files": len(entries), "bytes": total_bytes,
                      "failures": failures}, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
