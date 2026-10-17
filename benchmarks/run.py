"""Time benchmark cases of docs/design.md section 12 and print the timings as JSON."""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import scipy

import gspice
from gspice import _spectra
from gspice.legacy import gp_interp

NPIX = 2400
NGUARD = 20
# The chunk size of design section 8.
CHUNK = _spectra.CHUNK // NPIX
ENVIRONMENT = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMPY_MADVISE_HUGEPAGE",
)


def _covariance():
    pixels = np.arange(NPIX)
    K = np.exp(-((pixels[:, np.newaxis] - pixels) ** 2) / (2 * 15**2))
    V = np.random.default_rng(0).standard_normal((NPIX, 30))
    return K + V @ V.T / 30 + np.eye(NPIX)


def _timed(times, name, function, *args):
    start = time.perf_counter()
    result = function(*args)
    times[name].append(time.perf_counter() - start)
    # A caller that drops a large result frees it after the clock has stopped.
    return result


def _entry(case, name, parameters, times):
    # The first run of each entry is a warm-up: in a new process, its first writes to
    # new memory cost page faults.
    stages = {
        stage: {
            "min": min(seconds[1:]),
            "median": statistics.median(seconds[1:]),
            "times": seconds[1:],
        }
        for stage, seconds in times.items()
    }
    return {"case": case, "name": name, "parameters": parameters, "stages": stages}


def _pixelwise(C, repeats):
    rng = np.random.default_rng(1)
    pixelwise = partial(gspice.pixelwise, nguard=NGUARD)
    # 1e5 spectra peak at about 8.5 GB (design section 8).
    for nspec in (4000, 20_000, 100_000):
        flux = rng.standard_normal((nspec, NPIX), dtype=np.float32)
        ivar = np.ones((nspec, NPIX), dtype=np.float32)
        ivar[rng.random((nspec, NPIX), dtype=np.float32) < 0.02] = 0.0
        times = defaultdict(list)
        for _ in range(repeats + 1):
            # A new model, whose cache the first two stages fill.
            model = gspice.Model(np.zeros((1, NPIX)), (C,), (nspec,))
            _timed(times, "precision", model._precision, 0)
            _timed(times, "guard_weights", model._guard_weights, 0, NGUARD)
            _timed(times, "pixelwise", pixelwise, model, flux, ivar)
        times["total"] = [sum(seconds) for seconds in zip(*times.values())]
        yield {"nspec": nspec, "npix": NPIX, "nguard": NGUARD}, times


def _infill(C, repeats):
    # |G|, the patterns and the number of spectra. A unique pattern at |G| 1200 costs
    # tens of milliseconds, so there are fewer spectra.
    entries = [
        (11, "unique", 1000),
        (11, "shared", CHUNK),
        (30, "unique", 1000),
        (30, "shared", CHUNK),
        (100, "unique", 1000),
        (100, "shared", CHUNK),
        (1200, "unique", 20),
        (1200, "shared", CHUNK),
    ]
    # A model caches P, so it is not part of the time.
    model = gspice.Model(np.zeros((1, NPIX)), (C,), (1,))
    model._precision(0)
    rng = np.random.default_rng(2)
    pixels = np.arange(NPIX)
    for size, patterns, nspec in entries:
        # The inputs exist before the clock starts: a run of size pixels per spectrum,
        # or one (npix,) mask for all, and ivar = 1, so that G = M = T = replace.
        if patterns == "unique":
            first = rng.choice(NPIX - size + 1, nspec, replace=False)[:, np.newaxis]
        else:
            first = rng.integers(NPIX - size + 1)
        replace = (pixels >= first) & (pixels < first + size)
        flux = rng.standard_normal((nspec, NPIX), dtype=np.float32)
        ivar = np.ones((nspec, NPIX), dtype=np.float32)
        infill = partial(gspice.infill, model, replace=replace)
        times = defaultdict(list)
        for _ in range(repeats + 1):
            _timed(times, "infill", infill, flux, ivar)
        parameters = {
            "nspec": nspec,
            "npix": NPIX,
            "G_size": size,
            "patterns": patterns,
            "M": "G",
            "T": "G",
        }
        yield parameters, times


def _legacy(C, repeats):
    nspec = 4000
    X = np.random.default_rng(3).standard_normal((nspec, NPIX))
    times = defaultdict(list)
    for _ in range(repeats + 1):
        _timed(times, "total", gp_interp, X, C, NGUARD)
    yield {"nspec": nspec, "npix": NPIX, "nguard": NGUARD}, times


CASES = {
    1: ("pixelwise", _pixelwise),
    2: ("infill", _infill),
    7: ("legacy gp_interp", _legacy),
}


def _git(directory, *arguments):
    return subprocess.run(
        ["git", *arguments],
        cwd=directory,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _commit(path):
    # The commit of the tree that holds path, and whether path differs from it. A new
    # module that is not yet added counts; results written elsewhere do not.
    try:
        return {
            "sha": _git(path.parent, "rev-parse", "HEAD"),
            "dirty": bool(_git(path.parent, "status", "--porcelain", "--", path.name)),
        }
    except (OSError, subprocess.CalledProcessError):
        return None


def _read(path):
    try:
        return Path(path).read_text()
    except OSError:
        return ""


def _cpu_model():
    for line in _read("/proc/cpuinfo").splitlines():
        if line.startswith("model name"):
            return line.partition(":")[2].strip()
    return platform.processor()


def _memory_bytes():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None


def _load_average():
    try:
        return os.getloadavg()
    except (AttributeError, OSError):
        return None


def _blas_lapack(module):
    dependencies = module.show_config(mode="dicts")["Build Dependencies"]
    return {
        # The OpenBLAS configuration names the kernel, such as SkylakeX or Haswell.
        library: {
            key: dependencies[library].get(key)
            for key in ("name", "version", "openblas configuration")
        }
        for library in ("blas", "lapack")
    }


def _setup(repeats):
    # PYTHONPATH can import gspice from another tree than the script's.
    package = Path(gspice.__file__).parent
    return {
        "commits": {
            "gspice": _commit(package),
            "script": _commit(Path(__file__).resolve()),
        },
        "machine": {
            "cpu": _cpu_model(),
            "cpu_count": os.cpu_count(),
            "memory_bytes": _memory_bytes(),
            "platform": platform.platform(),
            # Huge pages change the time of memory-bound stages.
            "transparent_hugepage": _read(
                "/sys/kernel/mm/transparent_hugepage/enabled"
            ).strip()
            or None,
        },
        "environment": {name: os.environ.get(name) for name in ENVIRONMENT},
        "libraries": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "gspice": gspice.__version__,
            "gspice_path": str(package),
            "blas_lapack": {"numpy": _blas_lapack(np), "scipy": _blas_lapack(scipy)},
        },
        "repeats": repeats,
        "warm_up_runs": 1,
    }


def _report(entry):
    parameters = ", ".join(
        f"{key} {value}" for key, value in entry["parameters"].items()
    )
    lines = [f"case {entry['case']}, {entry['name']}: {parameters}"]
    for stage, stats in entry["stages"].items():
        lines.append(
            f"  {stage:<14} min {stats['min']:9.4f} s  median {stats['median']:9.4f} s"
        )
    print("\n".join(lines), file=sys.stderr)


def _arguments():
    cases = ", ".join(f"{case} ({name})" for case, (name, _) in CASES.items())
    parser = argparse.ArgumentParser(
        description=__doc__, epilog="Times are in seconds; the summary goes to stderr."
    )
    parser.add_argument(
        "cases", nargs="*", type=int, metavar="CASE", help=f"{cases}; default: all"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        metavar="N",
        help="timed repeats of each entry, at least 5 (default: %(default)s)",
    )
    args = parser.parse_args()
    # Design section 12: the minimum and median of at least 5 repeats.
    if args.repeats < 5:
        parser.error(
            f"argument --repeats: expected an integer >= 5, got {args.repeats}"
        )
    if unknown := sorted(set(args.cases) - CASES.keys()):
        parser.error(f"argument CASE: expected one of {list(CASES)}, got {unknown}")
    return args


def _main():
    args = _arguments()
    setup = _setup(args.repeats)
    load_before = _load_average()
    C = _covariance()
    results = []
    for case in args.cases or CASES:
        name, run = CASES[case]
        for parameters, times in run(C, args.repeats):
            entry = _entry(case, name, parameters, times)
            _report(entry)
            results.append(entry)
    setup["load_average"] = {"before": load_before, "after": _load_average()}
    print(json.dumps({"setup": setup, "results": results}, indent=2))


if __name__ == "__main__":
    _main()
