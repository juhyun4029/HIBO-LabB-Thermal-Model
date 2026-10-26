"""Manage foreground sampling, isolated run output, checkpoints, and resume."""
from pathlib import Path
import argparse
import contextlib
import datetime
import hashlib
import importlib.metadata
import json
import os
import sys
import threading
import traceback
import zipfile

ROOT = Path(__file__).resolve().parent
PACKAGE_ID = "HIBO_LabB_SingleSP_Initial_v3"
RESULTS_NAME = "single_sp_initial"
STOP_NAME = "CASE5_STOP_REQUESTED"  # The sampler reads only this package's root.
LOCK_PATH = ROOT / "CASE5_RUN.lock"
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
sys.path.insert(0, str(ROOT / "src"))


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify():
    manifest = ROOT / "PACKAGE_MANIFEST_SHA256.json"
    if not manifest.is_file():
        raise RuntimeError("Package manifest is missing. Extract the complete Case 5 ZIP into a new folder.")
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(entries, dict) or not entries:
        raise RuntimeError("Invalid package manifest.")
    for relative, expected in entries.items():
        path = (ROOT / relative.replace("\\", "/")).resolve()
        if not path.is_relative_to(ROOT):
            raise RuntimeError("Manifest path leaves this package.")
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError("Missing or changed package file: " + relative)
    return sha256(manifest)


def runtime_versions():
    packages = ("numpy", "scipy", "pandas", "numba", "arviz", "arviz-base",
                "arviz-stats", "llvmlite", "arviz-plots", "xarray-einstats",
                "joblib", "threadpoolctl", "scikit-learn", "matplotlib", "xarray")
    try:
        versions = {name: importlib.metadata.version(name) for name in packages}
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("A required dependency is missing. Run SETUP.bat in this Case 5 folder.") from error
    return dict(python=".".join(map(str, sys.version_info[:3])), packages=versions)


def settings():
    result = json.loads((ROOT / "CASE5_SETTINGS.json").read_text(encoding="utf-8"))
    if any(key in result for key in ("output_dir", "resume", "smoke", "period")):
        raise RuntimeError("Reserved controller option in CASE5_SETTINGS.json.")
    return result


class RunLock:
    """OS-owned advisory lock, released automatically after process death.

    The file stays in place to avoid unlink/recreate races. A PID is descriptive;
    it never decides lock ownership and no process is signaled.
    """
    def __init__(self):
        self.handle = None

    def acquire(self):
        self.handle = LOCK_PATH.open("a+b")
        self.handle.seek(0, os.SEEK_END)
        if self.handle.tell() == 0:
            self.handle.write(b" ")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.handle.close()
            self.handle = None
            raise RuntimeError("Case 5 is already running or exporting in this folder. No second process was started.") from error
        return self

    def release(self):
        if self.handle is not None:
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *unused):
        self.release()


def is_active():
    if not LOCK_PATH.exists():
        return False
    lock = RunLock()
    try:
        lock.acquire()
    except RuntimeError:
        return True
    lock.release()
    return False


def read_json(path, default=None):
    if not path.is_file():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_state(folder):
    import numpy as np
    for name in ("checkpoint.npz", "settling_checkpoint.npz", "warmup_checkpoint.npz"):
        path = folder / name
        if path.is_file():
            with np.load(path, allow_pickle=False) as archive:
                result = json.loads(str(archive["checkpoint_state_json"]))
            result["checkpoint_kind"] = name
            return result
    return dict(status="not_started")


def new_identity(manifest_hash, runtime, config, smoke=False):
    from diagnostic_parallel import diagnostic_definition
    return dict(package_id=PACKAGE_ID, experiment="Single-setpoint July Lab B; v3 numerical-stability update",
                created_utc=utc_now(), package_manifest_sha256=manifest_hash,
                runtime=runtime, diagnostic_definition=diagnostic_definition(),
                smoke_only=bool(smoke), old_run_imported=False,
                initial_draw_count=0, seed=int(config["seed"]))


def check_identity(folder, manifest_hash, runtime, allow_smoke=False):
    identity = read_json(folder / "run_identity.json")
    if (identity.get("package_id") != PACKAGE_ID
            or (identity.get("smoke_only") and not allow_smoke)
            or identity.get("old_run_imported") is not False
            or identity.get("package_manifest_sha256") != manifest_hash):
        raise RuntimeError("This is not this package's own unchanged Case 5 run. Case 1-4 and foreign checkpoints cannot be imported or resumed.")
    if identity.get("runtime") != runtime:
        raise RuntimeError("Python or dependency versions changed since this Case 5 run started. Restore its recorded environment before resuming.")
    return identity


class Tee:
    def __init__(self, console, log, mutex):
        self.console, self.log, self.mutex = console, log, mutex

    def write(self, message):
        with self.mutex:
            self.console.write(message)
            self.log.write(message)
            self.log.flush()
        return len(message)

    def flush(self):
        with self.mutex:
            self.console.flush()
            self.log.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


@contextlib.contextmanager
def console_log(folder):
    with (folder / "console.log").open("a", encoding="utf-8", buffering=1) as handle:
        mutex = threading.RLock()
        with contextlib.redirect_stdout(Tee(sys.stdout, handle, mutex)), \
                contextlib.redirect_stderr(Tee(sys.stderr, handle, mutex)):
            yield


def sampler_arguments(folder, config, smoke=False, resume=False, max_retained=None):
    result = ["july", "--output-dir", str(folder)]
    chosen = dict(config)
    if max_retained is not None:
        chosen["max_retained"] = max_retained
    if smoke:
        chosen.update(warmup=8, max_retained=8, minimum_retained=2000,
                      particles=16, max_leapfrog=4, residue_substeps=1)
        result.append("--smoke")
    for name, value in chosen.items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                result.append(flag)
        else:
            result.extend((flag, str(value)))
    if resume:
        result.append("--resume")
    return result


def save_results_zip(folder, smoke=False, reason="manual_export"):
    """Always archive diagnostic checkpoints, including a failed settling gate."""
    try:
        from july_export import export_run
        exported = export_run(folder)
    except Exception as error:
        exported = dict(status="posterior_export_failed", error=type(error).__name__ + ": " + str(error))
        atomic_json(folder / "export_error.json", dict(created_utc=utc_now(), **exported))
        traceback.print_exc()
    stage = read_json(folder / "status.json").get("stage", "initialization_or_interrupted")
    atomic_json(folder / "result_bundle.json", dict(
        package_id=PACKAGE_ID, created_utc=utc_now(), reason=reason, sampler_stage=stage,
        posterior_export=exported, smoke_only=bool(smoke),
        diagnostic_only_if_no_retained_draws=True,
        note="Discarded warmup and settling are diagnostic records, not posterior samples."))
    destination = ROOT / ("Case5_smoke_results.zip" if smoke else "Initial_MCMC_internal.zip")
    temporary = destination.with_name(destination.stem + ".tmp.zip")
    provenance = ("PACKAGE_INFO.json", "CASE5_SETTINGS.json", "PACKAGE_MANIFEST_SHA256.json",
                  "requirements.txt", "run_case5.py", "SAMPLER_DESIGN.md", "PROBLEM_DIAGNOSIS.md")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for path in sorted(folder.rglob("*")):
            if path.is_file() and not path.name.startswith(".") and not path.name.endswith((".tmp", ".tmp.npz")):
                archive.write(path, path.relative_to(ROOT).as_posix())
        for relative in provenance:
            path = ROOT / relative
            if path.is_file():
                archive.write(path, "package/" + relative)
        for path in sorted((ROOT / "src").glob("*.py")):
            archive.write(path, "package/src/" + path.name)
    temporary.replace(destination)
    print(json.dumps(dict(results_zip=str(destination), stage=stage, posterior_export=exported), indent=2), flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "resume", "stop", "status", "export", "smoke", "verify"))
    parser.add_argument("--max-retained", type=int, default=None,
                        help="Absolute finite retained draws per chain; no automatic second run.")
    args = parser.parse_args()
    smoke = args.action == "smoke"
    folder = ROOT / "results" / ("smoke_case5" if smoke else RESULTS_NAME)
    if args.action == "stop":
        if not is_active():
            print("No active Case 5 process in this package folder. No stop file was created.")
            return
        (ROOT / STOP_NAME).write_text("Graceful Case 5 stop requested\n", encoding="utf-8")
        print("Case 5 only: finish the active transition, save a coherent checkpoint, then export Initial_MCMC_internal.zip.")
        return
    if args.action == "status":
        current = checkpoint_state(folder)
        latest = next((row for row in reversed(current.get("history", [])) if "physical" in row), {})
        status = read_json(folder / "status.json")
        print(json.dumps(dict(package_id=PACKAGE_ID, active_in_this_folder=is_active(),
                              stage=status.get("stage", "not_started"), chains=4,
                              retained_per_chain=current.get("retained_per_chain", 0),
                              first_passing_N=current.get("N"),
                              discarded_stage_completed=current.get("completed"),
                              checkpoint_kind=current.get("checkpoint_kind"),
                              latest_diagnostics={key: latest[key] for key in ("physical", "ordered", "passed") if key in latest},
                              last_sampler_status=status), indent=2))
        return
    manifest_hash = verify()
    runtime = runtime_versions()
    config = settings()
    if args.action == "verify":
        from model import Model
        from diagnostic_parallel import diagnostic_definition
        model = Model("july")
        if model.ndim != 55 or len(model.names) != 59 or len(model.d["y"]) != 44640 or len(model.idx) != 6048:
            raise AssertionError("Unexpected model dimensions, July minute data length, or training likelihood count.")
        print(json.dumps(dict(package_id=PACKAGE_ID, checksums="passed", runtime=runtime,
                              n_parameters=55, n_reported=59, input_minutes=44640,
                              train_likelihood_points=6048, physical_input_seconds=60,
                              integration_substep_seconds=30, likelihood_minutes=5,
                              settings=config, diagnostic_definition=diagnostic_definition()), indent=2))
        return
    if args.max_retained is not None and args.max_retained < 5000 and not smoke:
        raise ValueError("Production budget must be at least 5,000 draws per chain.")
    with RunLock():
        if args.action == "export":
            check_identity(folder, manifest_hash, runtime)
            if not folder.is_dir():
                raise RuntimeError("No Case 5 results exist yet.")
            with console_log(folder):
                save_results_zip(folder)
            return
        if args.action in ("start", "smoke") and folder.exists() and any(folder.iterdir()):
            raise RuntimeError("This result folder is not empty. START never resumes or overwrites. Use RESUME_CASE5.bat only for this package's interrupted Case 5 run.")
        if args.action == "resume":
            check_identity(folder, manifest_hash, runtime)
            if not (folder / "configuration.json").is_file():
                raise RuntimeError("Initialization did not finish; no resumable configuration exists. Preserve these diagnostics and extract the Case 5 ZIP into a new folder for a fresh start.")
            if read_json(folder / "status.json").get("stage") in ("complete", "settling_gate_failed", "search_budget_reached"):
                raise RuntimeError("This Case 5 run reached a terminal stage. Export its results; this command does not restart a completed or failed experiment.")
        folder.mkdir(parents=True, exist_ok=True)
        if args.action != "resume":
            atomic_json(folder / "run_identity.json", new_identity(manifest_hash, runtime, config, smoke))
        atomic_json(folder / "environment.json", runtime)
        atomic_json(ROOT / "CASE5_PROCESS.json", dict(package_id=PACKAGE_ID, pid=os.getpid(), started_utc=utc_now(), action=args.action))
        stop = ROOT / STOP_NAME
        if stop.exists():
            stop.unlink()
        with console_log(folder):
            reason = "normal_return"
            print("INITIAL MCMC: fresh single-setpoint experiment; old samples imported: 0. Foreground execution; no automatic resume or restart.", flush=True)
            if smoke:
                print("SMOKE ONLY: tiny execution test; no inference or convergence claim.", flush=True)
            try:
                from tuned_profile_sampler import main as sample
                sample(sampler_arguments(folder, config, smoke, args.action == "resume", args.max_retained))
            except BaseException as error:
                reason = "interrupted" if isinstance(error, KeyboardInterrupt) else "controller_or_sampler_error"
                atomic_json(folder / "controller_failure.json", dict(created_utc=utc_now(), type=type(error).__name__, message=str(error)))
                traceback.print_exc()
                raise
            finally:
                print("Preparing the diagnostic/result ZIP from saved coherent checkpoints. Do not close this terminal yet.", flush=True)
                try:
                    save_results_zip(folder, smoke, reason)
                except Exception as export_error:
                    print("Result ZIP could not be completed: " + str(export_error), file=sys.stderr, flush=True)
                    print("Saved results remain in " + str(folder) + ". Run EXPORT_RESULTS.bat after resolving the file/disk error.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        sys.exit(1)
