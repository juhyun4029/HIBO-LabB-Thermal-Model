"""Rank-normalized R-hat, bulk ESS, and 5%/95% quantile ESS diagnostics.

Tail ESS is the minimum ESS of indicators below pooled q05 and q95. Explicit
quantile calls avoid dependence on library tail-ESS defaults. Diagnostics use
all retained production draws and a random stream separate from sampling."""
import copy
import os
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError, version
import platform
import sys
import numpy as np
import arviz as az
import xarray as xr

DEFINITION = {
    "schema": "hibo-diagnostics-5-95-v1",
    "rhat_method": "rank",
    "rhat_description": "Maximum rank-normalized split and folded-split R-hat",
    "bulk_ess_method": "bulk",
    "tail_ess_method": "minimum_of_two_explicit_quantile_ess",
    "tail_quantiles": [0.05, 0.95],
    "relative_ess": False,
    "production_draw_policy": "All retained post-warmup draws; no thinning or posthoc discard",
    "thresholds": {"rhat_strictly_below": 1.10, "bulk_ess_at_least": 400, "tail_ess_at_least": 400},
    "coordinate_policy": "Both all 59 reported physical marginals and all 55 ordered coordinates must pass",
}


def diagnostic_definition():
    return copy.deepcopy(DEFINITION)


def runtime_environment():
    packages = {}
    for name in ("numpy", "scipy", "numba", "llvmlite", "arviz", "arviz-stats", "arviz-base", "xarray", "pandas", "scikit-learn", "matplotlib"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "python_executable": sys.executable,
            "platform": platform.platform(), "machine": platform.machine(),
            "packages": packages,
            "thread_environment": {key: os.environ.get(key) for key in
                ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "HIBO_DIAGNOSTIC_WORKERS")}}


def serial_matrix(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 3 or values.shape[0] < 2 or values.shape[1] < 4 or values.shape[2] < 1:
        raise ValueError("Diagnostics require (chain>=2, draw>=4, parameter>=1)")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite retained draws; diagnostics must not pass")
    data = xr.Dataset({"theta": (("chain", "draw", "parameter"), values)})
    rhat = np.asarray(az.rhat(data, method="rank")["theta"])
    bulk = np.asarray(az.ess(data, method="bulk", relative=False)["theta"])
    lower = np.asarray(az.ess(data, method="quantile", prob=0.05, relative=False)["theta"])
    upper = np.asarray(az.ess(data, method="quantile", prob=0.95, relative=False)["theta"])
    return np.stack([rhat, bulk, np.minimum(lower, upper)])


def diagnostic_matrix(values, workers=None):
    values = np.asarray(values, dtype=float)
    if values.ndim != 3 or values.shape[-1] < 1:
        raise ValueError("Diagnostics require a nonempty chain/draw/parameter array")
    workers = int(os.environ.get("HIBO_DIAGNOSTIC_WORKERS", "4")) if workers is None else int(workers)
    if workers <= 1:
        return serial_matrix(values)
    groups = np.array_split(np.arange(values.shape[-1]), min(workers, values.shape[-1]))
    with ThreadPoolExecutor(len(groups)) as pool:
        parts = list(pool.map(lambda idx: serial_matrix(values[:, :, idx]), groups))
    return np.concatenate(parts, axis=1)
