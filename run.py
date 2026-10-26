"""Command-line interface for the Lab B calibration and reduction pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import uuid


ROOT = Path(__file__).resolve().parent
ACTIONS = ("verify", "start", "resume", "status", "stop", "export")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=ACTIONS)
    parser.add_argument(
        "--workdir",
        type=Path,
        help="New run directory for start. Relative paths are resolved from the repository root.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the command without executing it."
    )
    args = parser.parse_args(argv)
    if args.workdir is not None and args.action != "start":
        parser.error("--workdir is only valid for start; subsequent actions use the saved run.")
    return args


def resolve_workdir(repository: Path, value: Path | None) -> Path:
    """Resolve a fresh run location without allowing writes to published inputs/results."""
    repository = repository.resolve()
    runs = repository / "runs"
    if value is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        workdir = runs / run_id
    else:
        value = value.expanduser()
        workdir = value if value.is_absolute() else repository / value
    workdir = workdir.resolve()
    if workdir == repository or repository.is_relative_to(workdir):
        raise ValueError("The run directory must not contain the repository.")
    if workdir.is_relative_to(repository) and not workdir.is_relative_to(runs):
        raise ValueError("Run directories inside the repository must be under runs/.")
    if workdir == runs:
        raise ValueError("Choose a new subdirectory of runs/, rather than runs/ itself.")
    return workdir


def build_command(args, repository: Path = ROOT, interpreter: str = sys.executable):
    repository = repository.resolve()
    script = repository / "03_workflow" / "run_pipeline.py"
    command = [interpreter, "-u", str(script), args.action]
    if args.action == "start":
        command.extend(["--workdir", str(resolve_workdir(repository, args.workdir))])
    return command


def main(argv=None):
    args = parse_args(argv)
    try:
        command = build_command(args)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    if args.dry_run:
        print(subprocess.list2cmdline(command))
        return 0
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    return subprocess.run(command, cwd=ROOT / "03_workflow", env=env, check=False).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
