import argparse
import contextlib
import hashlib
import importlib
import io
import json
import os
import platform
import subprocess
import sys
import warnings
import zipfile
from functools import partial
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import cho_factor, cho_solve

import _golden

COMMIT = "1fbab9f2ef28c148d58c9b0e6ea90f248f646dfe"
SEEDS = {
    "E": 202203,
    "MAT": 202204,
    "PARAMS": 202205,
    "D1": 202201,
    "D2": 202202,
    "D3": 202206,
    "D4": 202207,
}
THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
# Z differs between the 2022 code, a direct solve and 11 Linux builds by up to 8.2e-10
# (D3) and 1.0e-7 (D4), relative, where |Z| >= 1: 10 times less than this margin.
MIN_MARGIN = 1e-6
SU, MU, DJ = "gspice.specUtils", "gspice.matrixUtils", "gspice.djs_maskinterp"
# Every other function is in specUtils.
MODULES = {
    "cholesky_inv": MU,
    "submatrix_inv": MU,
    "submatrix_inv_mult": MU,
    "maskinterp": DJ,
    "maskinterp1": DJ,
    "average_bilinear": DJ,
}
# Spectral cases go to golden_<dataset>_<group>.npz; the default group is "scale".
GROUPS = {
    "cholesky_inv": "gaussian_estimate",
    "gaussian_estimate": "gaussian_estimate",
    "gp_interp": "gp_interp",
    "get_chimask": "masks",
    "covar_iter_mask": "masks",
}

# Ranges are (low, high): lengths of masked blocks and shifted blocks, masked pixels
# of the "many" rows, S/N. "boundary" holds maxbadpix and maxbadpix + 1.
SPECTRAL = {
    "D1": {
        "npix": 50,
        "nspec": 2000,
        "nsubset": 64,
        "blocks": (5, 12),
        "many": (15, 30),
        "shifts": (20, 30),
        "boundary": (8, 9),
        "badcols": [([17], 0.30)],
        "snr": (1.0, 10.0),
    },
    "D2": {
        "npix": 300,
        "nspec": 5000,
        "nsubset": 96,
        "blocks": (10, 40),
        "many": (70, 150),
        "shifts": (20, 60),
        "boundary": (64, 65),
        "badcols": [([123, 124], 0.25), ([251], 0.15)],
        "snr": (0.4, 4.0),
    },
}
# D3 is D1 at a realistic S/N, so its covariance is ill conditioned. D4 has realistic
# S/N on 150 pixels, so get_chimask (nguard=20) conditions on most of them; it has mask
# cases only.
SPECTRAL["D3"] = dict(SPECTRAL["D1"], snr=(12.0, 120.0))
SPECTRAL["D4"] = {
    "npix": 150,
    "nspec": 1500,
    "nsubset": 64,
    "blocks": (10, 30),
    "many": (40, 80),
    "shifts": (20, 40),
    "boundary": (64, 65),
    "badcols": [([60, 61], 0.25)],
    "snr": (30.0, 300.0),
}
MATRICES = {
    "M50a": (50, 1e3),
    "M50b": (50, 1e6),
    "M300a": (300, 1e3),
    "M300b": (300, 1e6),
}


def make_spectral(name):
    config = SPECTRAL[name]
    npix, nspec = config["npix"], config["nspec"]
    rng = np.random.default_rng(SEEDS[name])
    x = (np.arange(npix) + 0.5) / npix
    pix = np.arange(npix, dtype=np.float64)
    # Mean: continuum times absorption lines. Variation: 4 modes, SE kernel, white.
    cont = 1.0 + 0.25 * x - 0.15 * x**2
    nlines = max(4, npix // 12)
    centers = rng.uniform(0.03, 0.97, nlines) * npix
    width = rng.uniform(1.0, 3.5, nlines)
    depth = rng.uniform(0.05, 0.6, nlines)
    profile = np.exp(-0.5 * ((pix[None, :] - centers[:, None]) / width[:, None]) ** 2)
    mean = cont * np.prod(1.0 - depth[:, None] * profile, axis=0)
    wave = np.sin(2 * np.pi * 1.5 * x + 0.4)
    modes = np.stack([mean * (x - 0.5), np.gradient(mean), mean - cont, wave])
    lag = pix[:, None] - pix[None, :]
    kse = 0.01**2 * np.exp(-0.5 * (lag / 4.0) ** 2) + 1e-12 * np.eye(npix)
    coeff = rng.standard_normal((nspec, 4)) * np.array([0.05, 0.4, 0.15, 0.01])
    y = coeff @ modes + rng.standard_normal((nspec, npix)) @ np.linalg.cholesky(kse).T
    y = y + 0.002 * rng.standard_normal((nspec, npix))
    amp = np.exp(0.25 * rng.standard_normal(nspec))
    snr = 10 ** rng.uniform(
        np.log10(config["snr"][0]), np.log10(config["snr"][1]), nspec
    )
    ivar_a = (snr / amp) ** 2
    ivar_b = 1.0 / (1.0 + 0.4 * np.cos(2 * np.pi * 0.8 * x + 0.3)) ** 2
    sigma = 1.0 / np.sqrt(ivar_a[:, None] * ivar_b[None, :])
    flux = amp[:, None] * (mean[None, :] + y) + sigma * rng.standard_normal(y.shape)

    order = iter(rng.permutation(nspec))
    sizes = {"edge_first": 6, "edge_last": 6, "edge_both": 2, "block": 5}
    sizes.update(single_good=1, many_masked=3 if npix == 50 else 4)
    sizes.update(shifted=3 if npix == 50 else 6, clean=20, boundary=2)
    special = {k: np.sort([next(order) for _ in range(n)]) for k, n in sizes.items()}

    # Artifacts that ivar does not know about: shifted blocks, spikes, bad columns.
    flux0 = flux.copy()
    for s in special["shifted"]:
        length = int(rng.integers(config["shifts"][0], config["shifts"][1] + 1))
        shift = int(rng.choice([-3, -2, -1, 1, 2, 3]))
        i0 = int(rng.integers(max(0, -shift), npix - length - max(0, shift) + 1))
        flux[s, i0 : i0 + length] = flux0[s, i0 + shift : i0 + shift + length]
    spikes = []
    for _ in range(nspec // 10):
        s, p = int(rng.integers(nspec)), int(rng.integers(npix))
        w = int(rng.integers(1, 3))
        a = float(10 ** rng.uniform(np.log10(5.0), 2.0))
        flux[s, p : p + w] += a * sigma[s, p : p + w]
        spikes.append((s, p, min(p + w, npix) - p))
    badcol_rows = []
    for cols, fraction in config["badcols"]:
        hit = np.flatnonzero(rng.random(nspec) < fraction)
        a = rng.uniform(6.0, 30.0, hit.size)
        for c in cols:
            flux[hit, c] += a * sigma[hit, c]
        badcol_rows.append(hit[:10])

    # Mask flags: 1 sparse, 2 edge, 4 block, 8 many. mask_alt adds 16 on spikes and
    # 32 on a bad column, where ivar is not 0.
    mask = np.zeros((nspec, npix), dtype=np.uint8)
    sparse_rows = rng.random(nspec) < 0.30
    mask[(rng.random((nspec, npix)) < 0.01) & sparse_rows[:, None]] |= 1
    for s in special["edge_first"]:
        mask[s, : int(rng.integers(1, 6))] |= 2
    for s in special["edge_last"]:
        mask[s, npix - int(rng.integers(1, 6)) :] |= 2
    for s in special["edge_both"]:
        mask[s, : int(rng.integers(1, 4))] |= 2
        mask[s, npix - int(rng.integers(1, 4)) :] |= 2
    for s in special["block"]:
        length = int(rng.integers(config["blocks"][0], config["blocks"][1] + 1))
        i0 = int(rng.integers(2, npix - length - 2))
        mask[s, i0 : i0 + length] |= 4
    good_pixel = int(rng.integers(npix // 4, 3 * npix // 4))
    mask[special["single_good"][0], :] |= 8
    mask[special["single_good"][0], good_pixel] = 0
    for s in special["many_masked"]:
        count = int(rng.integers(config["many"][0], config["many"][1] + 1))
        mask[s, rng.choice(npix, size=count, replace=False)] |= 8
    # Rows with maxbadpix and maxbadpix + 1 masked pixels: covar_iter_mask keeps the
    # first and rejects the second.
    for s, count in zip(special["boundary"], config["boundary"], strict=True):
        mask[s, :] = 0
        mask[s, rng.choice(npix, size=count, replace=False)] = 8
    mask[special["clean"], :] = 0
    mask[special["shifted"], :] = 0
    flux[(mask & 11) != 0] = 0.0

    # Flux on a per-spectrum dyadic grid finer than sigma / 4, stored as two bytes.
    exponent = np.ceil(np.log2(4.0 / sigma.min(axis=1))).astype(np.intc)
    counts = np.rint(flux / np.ldexp(1.0, -exponent)[:, None])
    counts = counts.astype(np.int64)
    stored = {"flux_hi": (counts >> 8).astype(np.int8)}
    stored.update(flux_lo=(counts & 255).astype(np.uint8))
    stored.update(flux_exp=exponent.astype(np.int8), ivar_a=ivar_a, ivar_b=ivar_b)
    stored["mask"] = mask
    rebuilt = stored["flux_hi"].astype(np.int64) * 256 + stored["flux_lo"]
    assert np.array_equal(rebuilt, counts), name
    if name == "D1":
        stored["mask_alt"] = mask.copy()
        for s, p, w in spikes:
            stored["mask_alt"][s, p : p + w] |= 16
        stored["mask_alt"][badcol_rows[0], 17] |= 32

    # Fixed row subset for (nspec, npix) outputs: the special rows, some spike and
    # bad-column rows, random rows.
    rows = [v for k, v in special.items() if k != "clean"] + [special["clean"][:4]]
    must = np.unique(np.concatenate(rows))
    spike_rows = np.unique([s for s, _, _ in spikes])[:10]
    extra = [r for r in [*spike_rows, *badcol_rows[0][:5]] if r not in must]
    subset = np.unique(np.concatenate([must, np.asarray(extra, dtype=np.int64)]))
    rest = np.setdiff1d(np.arange(nspec), subset)
    fill = rng.choice(rest, size=config["nsubset"] - subset.size, replace=False)
    stored["rows_subset"] = np.sort(np.concatenate([subset, fill])).astype(np.int32)
    if name == "D1":
        stored["ivar_zero"] = make_ivar_zero(mask, stored["rows_subset"], special)
    return stored, special


def make_ivar_zero(mask, rows_subset, special):
    # Unmasked pixels with ivar 0, from their own random stream so no other D1 array
    # changes.
    rng = np.random.default_rng([SEEDS["D1"], 7])
    nspec, npix = mask.shape
    good = mask == 0
    zero = np.zeros(mask.shape, dtype=np.uint8)
    rows = rng.random(nspec) < 0.25
    zero[(rng.random((nspec, npix)) < 0.01) & rows[:, None] & good] |= 1
    clean = [int(r) for r in rows_subset if good[r].all()]
    pick = rng.choice(clean, size=9, replace=False)
    zero[pick[0:3], 0] |= 2
    zero[pick[3:6], npix - 1] |= 2
    for r in pick[6:9]:
        i0 = int(rng.integers(3, npix - 7))
        zero[r, i0 : i0 + 4] |= 4
    edges = [*special["edge_first"][:3], *special["edge_last"][:3]]
    for r in [*special["block"], *edges]:
        bad = np.flatnonzero(~good[r])
        if bad.min() > 0:
            zero[r, bad.min() - 1] |= 8
        if bad.max() < npix - 1:
            zero[r, bad.max() + 1] |= 8
    zero[special["single_good"], :] = 0
    zero[~good] = 0
    return zero


def make_edge():
    # Rows: 0 and 9 clean; 1 fully masked; 2 one good pixel; 3 first pixels; 4 last
    # pixels; 5 interior block; 6 alternating; 7 first, last and sparse; 8 only the two
    # end pixels good.
    rng = np.random.default_rng(SEEDS["E"])
    shape = (10, 24)
    flux = 1.0 + 0.1 * np.sin(np.arange(24) / 3.0)[None, :]
    flux = flux + 0.05 * rng.standard_normal(shape)
    ivar = rng.uniform(50.0, 400.0, shape)
    mask = np.zeros(shape, dtype=np.int32)
    mask[1, :] = 1
    mask[2, :] = 2
    mask[2, 7] = 0
    mask[3, :3] = 5
    mask[4, -4:] = 128
    mask[5, 8:14] = 1
    mask[6, ::2] = 1
    mask[7, [0, 5, 11, 23]] = 3
    mask[8, 1:-1] = 1
    ivar[mask != 0] = 0.0
    zero = np.zeros(shape, dtype=np.uint8)
    # Unmasked pixels with ivar 0 (ivar_z): ends, next to masked runs, a block.
    columns = {0: [0, 12], 3: [3, 23], 4: [19], 5: [3, 4, 7, 14]}
    columns.update({6: [9], 7: [6], 8: [23], 9: [10, 11, 12]})
    for r, cols in columns.items():
        zero[r, cols] = 1
    return {"flux": flux, "ivar": ivar, "mask": mask, "ivar_zero": zero}


def make_matrices():
    # t[|i - j|] (Gaussian kernel) + 3 rank-one terms + a diagonal that sets the
    # condition number.
    rng = np.random.default_rng(SEEDS["MAT"])
    stored = {}
    for name, (n, target) in MATRICES.items():
        t = np.exp(-0.5 * (np.arange(n) / 3.0) ** 2)
        V = np.stack([rng.standard_normal(n) * np.sqrt(w / n) for w in (20, 10, 5)])
        u = rng.uniform(0.0, 1.0, n)
        eig = np.linalg.eigvalsh(_golden.build_spd_matrix(t, V, np.zeros(n)))
        d = (eig[-1] - target * max(eig[0], 0.0)) / (target - 1.25)
        for _ in range(4):
            eig = np.linalg.eigvalsh(_golden.build_spd_matrix(t, V, d * (1 + 0.5 * u)))
            d *= eig[-1] / eig[0] / target
        stored.update(
            {name + "__t": t, name + "__V": V, name + "__d": d * (1 + 0.5 * u)}
        )
        stored["Y__" + name] = rng.standard_normal((n, 5))
    return stored


def write_npz(path, arrays):
    # The np.savez_compressed layout with sorted members and a fixed timestamp, so a
    # regeneration in the same environment is byte-identical.
    with zipfile.ZipFile(path, "w") as archive:
        for key in sorted(arrays):
            buffer = io.BytesIO()
            array = np.ascontiguousarray(arrays[key])
            np.lib.format.write_array(buffer, array, allow_pickle=False)
            info = zipfile.ZipInfo(key + ".npy", date_time=(2022, 8, 17, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, buffer.getvalue(), compresslevel=9)


def dump_json(value, indent=0):
    # Indented JSON that keeps each list of scalars on one line.
    pad = " " * indent
    if isinstance(value, dict) and value:
        items = [
            f"{pad} {json.dumps(k)}: {dump_json(v, indent + 1)}"
            for k, v in value.items()
        ]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(value, list) and any(isinstance(v, (dict, list)) for v in value):
        items = [f"{pad} {dump_json(v, indent + 1)}" for v in value]
        return "[\n" + ",\n".join(items) + "\n" + pad + "]"
    return json.dumps(value)


def normwise(a, b):
    scale = np.abs(b).max()
    error = np.abs(np.asarray(a) - b).max()
    return float(error / scale if scale > 0 else error)


def elementwise(a, b):
    # Equal entries agree, zeros included (the norms of gp_interp with nguard=50).
    diff = np.abs(a - b)
    return float(np.max(diff / np.where(diff > 0, np.abs(b), 1.0)))


def solve(M, rows, columns, lower=True):
    return cho_solve(cho_factor(M[np.ix_(rows, rows)], lower=lower), columns)


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def output(name, storage="full", index=None, **extra):
    spec = {"name": name} if index is None else {"name": name, "index": index}
    return dict(spec, storage=storage, **extra)


def check(prop, output=None, **extra):
    return {"output": output, "prop": prop, **extra}


OUTPUTS = {
    "maskinterp": [output("ynew", "masked")],
    "maskinterp1": [output("ynew")],
    "average_bilinear": [output("interp", "masked")],
    "standard_scale": [
        output("refscale", "full", 1),
        output("refmean", "full", 2),
        output("Dvec", "scaled", 0),
    ],
    "covar": [output("cov", "triu", 0), output("refmean", "full", 1)],
    "cholesky_inv": [output("Minv", "triu")],
    "submatrix_inv": [output("Ainv")],
    "submatrix_inv_mult": [output("AinvY")],
    "gp_interp": [output("pred", "full", 0), output("predvar", "onerow", 1)],
    "get_chimask": [output("chimask", "packbits")],
    "covar_iter_mask": [output("cov", "triu", 0), output("finalmask", "packbits", 1)],
}


def file_of(function, dataset):
    if dataset == "E":
        return "golden_E.npz"
    if dataset.startswith("M"):
        return f"golden_linalg_{dataset[1:-1]}.npz"
    return f"golden_{dataset}_{GROUPS.get(function, 'scale')}.npz"


class Generator:
    def __init__(self, committed):
        self.committed = committed
        self.files = {}
        self.inputs = {}
        self.cases = []
        self.actual = {}
        self.passed = {}
        self.margins = {}
        self.rng = np.random.default_rng(SEEDS["PARAMS"])

    def dataset(self, name, stored):
        path = f"inputs_{name}.npz"
        if self.committed is not None and (self.committed.directory / path).exists():
            # Committed inputs rebuild to the manifest hashes on every platform. A fresh
            # build differs from them only by rounding, unless the definition of the
            # dataset changed. Masks are integer functions of the random stream, so
            # equal masks also mean equal special rows of make_spectral.
            fresh = _golden.build_inputs(stored)
            for key, value in self.committed.inputs(name).items():
                close = value.shape == fresh[key].shape
                close = close and np.allclose(value, fresh[key], rtol=1e-6, atol=0)
                assert close, f"{name}/{key} differs from {path}: pass --new-inputs"
            stored = self.committed.file(path)
        self.files[path] = stored
        self.inputs[name] = _golden.build_inputs(stored)

    def resolve(self, spec, stored=None):
        lookups = (self.inputs, self.actual, stored or {})
        return _golden.resolve(spec, *(lookup.__getitem__ for lookup in lookups))

    def call(self, cid, function, args, file, stored):
        if stored:
            arrays = {f"{cid}__in__{k}": v for k, v in stored.items()}
            self.files.setdefault(file, {}).update(arrays)
        kwargs = {name: self.resolve(spec, stored) for name, spec in args.items()}
        self.passed[cid] = kwargs
        copies = {
            k: v.copy() if isinstance(v, np.ndarray) else v for k, v in kwargs.items()
        }
        baseline = getattr(sys.modules[MODULES.get(function, SU)], function)
        silent = contextlib.redirect_stdout(io.StringIO())
        with warnings.catch_warnings(record=True) as caught, silent:
            warnings.simplefilter("always")
            try:
                result, error = baseline(**copies), None
            except (AssertionError, NameError, ValueError) as e:
                result, error = None, e
        return copies, result, error, sorted({w.category.__name__ for w in caught})

    def golden(self, cid, args, outputs=None, stored=None):
        function, dataset = cid.split(".")[:2]
        file = file_of(function, dataset)
        _, result, error, _ = self.call(cid, function, args, file, stored)
        assert error is None, f"{cid} raised {error!r}"
        kwargs = self.passed[cid]
        store = self.files.setdefault(file, {})
        self.actual[cid] = {}
        entries = []
        for spec in outputs or OUTPUTS[function]:
            value = result[spec["index"]] if "index" in spec else result
            value = self.actual[cid][spec["name"]] = np.asarray(value)
            entries.append(dict(spec, dtype=str(value.dtype), shape=list(value.shape)))
            key = f"{cid}__out__{spec['name']}"
            storage = spec["storage"]
            if storage == "full":
                store[key] = value
            elif storage == "triu":
                store[key] = value[np.triu_indices(value.shape[0])]
            elif storage == "packbits":
                store[key] = np.packbits((value != 0).ravel())
            elif storage == "masked":
                store[key] = value[kwargs["mask"] != 0]
            elif storage == "scaled":
                # The values standard_scale interpolates, from which Dvec is rebuilt.
                mask = kwargs.get("mask")
                bad = kwargs["ivar"] == 0 if mask is None else np.where(mask == 0, 0, 1)
                maskinterp = sys.modules[DJ].maskinterp
                with quiet():
                    interp = maskinterp(yval=kwargs["spec"], mask=bad, axis=0)
                store[key] = interp[bad != 0]
            elif storage == "onerow":
                store[key] = value[0]
            else:
                if storage == "rows":
                    rows = self.inputs[dataset]["rows_subset"]
                    store[key + "__rows"] = value[rows]
                    store[key + "__rowidx"] = rows.astype(np.int32)
                store[key + "__rownorm"], store[key + "__colnorm"] = _golden.norms(
                    value
                )
        case = {"id": cid, "kind": "golden", "module": MODULES.get(function, SU)}
        case.update(function=function, file=file, args=args, outputs=entries)
        self.cases.append(case)
        return kwargs, self.actual[cid]

    def behaviour(self, cid, args, checks=(), file=None, stored=None, **expect):
        function = cid.split(".")[1]
        copies, result, error, warns = self.call(cid, function, args, file, stored)
        assert (error is None) == bool(checks), f"{cid}: raised {error!r}"
        if error is not None:
            expect["raises"] = type(error).__name__
            # Python and numpy word their own errors differently across versions.
            if type(error) in (AssertionError, ValueError):
                expect["message"] = str(error).split(" (")[0]
        else:
            values = [_golden.property_value(result, c, copies) for c in checks]
            expect["checks"] = [dict(c, value=v) for c, v in zip(checks, values)]
            if warns:
                expect["warns"] = warns
        case = {"id": cid, "kind": "behaviour", "module": MODULES.get(function, SU)}
        case.update(function=function, **({"file": file} if file else {}))
        self.cases.append(dict(case, args=args, expect=expect))

    def accuracy(self, **errors):
        self.cases[-1]["baseline_accuracy"] = errors

    def error(self, name, reference):
        # The metrics of check_output for a normwise output, because the rule of
        # RTOL_OVERRIDES applies to them.
        case = self.cases[-1]
        spec = next(spec for spec in case["outputs"] if spec["name"] == name)
        value = self.actual[case["id"]][name]
        reference = np.reshape(reference, value.shape)
        errors = [normwise(value, reference)]
        if spec["storage"] == "rows":
            rows = self.inputs[case["id"].split(".")[1]]["rows_subset"]
            errors.append(normwise(value[rows], reference[rows]))
        if spec["storage"] in ("rows", "norms"):
            pairs = zip(_golden.norms(value), _golden.norms(reference), strict=True)
            errors += [elementwise(a, b) for a, b in pairs]
        if "derived" in spec:
            # How far the reference of the whole-array check lies from a direct solve.
            derived = spec["derived"]
            left = np.asarray(self.passed[case["id"]][derived["left"]], np.float64)
            whole = left @ self.actual[derived["golden"]][derived["output"]]
            errors.append(normwise(whole.reshape(value.shape), reference))
        return max(errors)

    def margin(self, cid, chimask, flux, ivar, mask, nsigma):
        # Recompute Z as get_chimask does: a mask reproduces on another platform only
        # if no |Z| lies close to nsigma.
        specutils = sys.modules[SU]
        with quiet():
            chi = _golden.zscore(partial(getattr, specutils), flux, ivar, mask)
        flag = np.abs(chi) >= nsigma
        replay = specutils.dilate(flag, np.array([[1, 1, 1]])).astype(flag.dtype)
        assert _golden.same(replay, chimask), cid
        margin = float(np.min(np.abs(np.abs(chi) - nsigma))) / float(nsigma)
        assert margin > MIN_MARGIN, f"{cid}: |Z| within {margin:.1e} of nsigma"
        self.margins[cid] = min(margin, self.margins.get(cid, np.inf))
        return chi


def cases_edge(g):
    flux, ivar, mask = ({"input": "E", "key": k} for k in ("flux", "ivar", "mask"))
    for axis in (0, 1):
        args = {"yval": flux, "mask": mask, "axis": {"value": axis}}
        g.golden(f"maskinterp.E.axis{axis}", args)
    g.golden("average_bilinear.E", {"yval": flux, "mask": mask})
    for r in range(10):
        args = {"yval": dict(flux, row=r), "mask": dict(mask, row=r)}
        g.golden(f"maskinterp1.E.row{r}", args)
    scale_args = {"spec": flux, "ivar": ivar, "mask": mask}
    g.golden("standard_scale.E.with_fully_masked", scale_args)
    for tag, ivar_key in (("no_full", "ivar"), ("ivar0", "ivar_z")):
        for name, m in (
            ("mask", dict(mask, drop_rows=[1])),
            ("maskNone", {"value": None}),
        ):
            args = {"spec": dict(flux, drop_rows=[1]), "mask": m}
            args["ivar"] = {"input": "E", "key": ivar_key, "drop_rows": [1]}
            g.golden(f"standard_scale.E.{tag}.{name}", args)
    checks = [check("nan_index", 1), check("nan_index", 2), check("nan_index", 0)]
    g.behaviour("behaviour.standard_scale.fully_masked_nan", scale_args, checks)
    args = {"yval": dict(flux, row=2), "mask": dict(mask, row=2)}
    g.behaviour("behaviour.maskinterp1.single_good_constant", args, [check("tolist")])
    args = {"yval": flux, "mask": mask, "axis": {"value": 0}}
    checks = [check("row_unchanged", argument="yval", row=1)]
    g.behaviour("behaviour.maskinterp.fully_masked_row_unchanged", args, checks)
    args = {"yval": {"array": [0, 10, 3, 7, 1, 4], "dtype": "int64"}}
    args["mask"] = {"array": [1, 0, 1, 0, 1, 0], "dtype": "int64"}
    checks = [check("dtype"), check("tolist")]
    g.behaviour("behaviour.maskinterp1.integer_input_truncates", args, checks)


def cases_matrices(g):
    M = {name: {"input": "MAT", "key": name} for name in MATRICES}
    Minv = {name: {"golden": f"cholesky_inv.{name}", "output": "Minv"} for name in M}
    for name in MATRICES:
        kwargs, out = g.golden(f"cholesky_inv.{name}", {"M": M[name]})
        expected = cho_solve(cho_factor(kwargs["M"]), np.eye(len(kwargs["M"])))
        g.accuracy(Minv=normwise(out["Minv"], expected))

    # The references use upper factors: at cond 1e6 their own error is about 1e-11, and
    # these are the factors the overrides in test_golden.py were derived with.
    scattered50 = np.sort(g.rng.choice(50, 10, replace=False)).tolist()
    scattered300 = np.sort(g.rng.choice(300, 200, replace=False)).tolist()
    contiguous = list(range(20, 31))
    for tag, name, removed, bruteforce in (
        ("contig", "M50a", contiguous, False),
        ("scattered", "M50b", scattered50, False),
        ("single", "M50a", [17], False),
        ("bruteforce", "M50a", contiguous, True),
        ("keep100", "M300a", scattered300, False),
    ):
        args = {"M": M[name], "Minv": Minv[name], "imask": {"stored": "imask"}}
        args["bruteforce"] = {"value": bruteforce}
        stored = {"imask": np.array(indicator(MATRICES[name][0], removed, 0))}
        kwargs, out = g.golden(f"submatrix_inv.{name}.{tag}", args, stored=stored)
        k = np.flatnonzero(kwargs["imask"])
        expected = solve(kwargs["M"], k, np.eye(k.size), lower=False)
        g.accuracy(Ainv=normwise(out["Ainv"], expected))
    args = {"M": M["M50a"], "Minv": Minv["M50a"]}
    args["imask"] = {"array": [1] * 50, "dtype": "int64"}
    kwargs, out = g.golden("submatrix_inv.M50a.nr0", args)
    expected = solve(kwargs["M"], np.arange(50), np.eye(50), lower=False)
    g.accuracy(Ainv=normwise(out["Ainv"], expected))
    checks = [check("is_argument", argument="Minv")]
    g.behaviour("behaviour.submatrix_inv.nr0_returns_input", args, checks)

    contiguous = list(range(18, 29))
    scattered30 = np.sort(g.rng.choice(300, 30, replace=False)).tolist()
    for tag, name, removed, columns, pad, bruteforce in (
        ("contig.col.padT", "M50a", contiguous, [23], True, False),
        ("contig.col.padF", "M50a", contiguous, [23], False, False),
        ("contig.cols11.padT", "M50a", contiguous, contiguous, True, False),
        ("contig.dense.padT", "M50a", contiguous, None, True, False),
        ("contig.dense.padF", "M50a", contiguous, None, False, False),
        ("scattered.dense.padT", "M50b", scattered50, None, True, False),
        ("scattered.cols.padF", "M50b", scattered50, scattered50[:3], False, False),
        ("single.col.padT", "M50a", [17], [17], True, False),
        ("single.dense.padF", "M50a", [17], None, False, False),
        ("nr0.dense", "M50a", [], None, True, False),
        ("bruteforce.dense", "M50a", contiguous, None, True, True),
        ("bruteforce.col", "M50b", scattered50, scattered50[:1], False, True),
        ("contig41.col.padT", "M300a", list(range(130, 171)), [150], True, False),
        ("scattered.dense.padT", "M300b", scattered30, None, True, False),
    ):
        n = MATRICES[name][0]
        dense = {"input": "MAT", "key": "Y__" + name}
        Y = dense if columns is None else dict(M[name], cols=columns)
        args = {"M": M[name], "Minv": Minv[name], "imask": {"stored": "imask"}, "Y": Y}
        args.update(MinvY={"stored": "MinvY"}, pad={"value": pad})
        args["bruteforce"] = {"value": bruteforce}
        MinvY = g.actual[f"cholesky_inv.{name}"]["Minv"] @ g.resolve(Y)
        stored = {"imask": np.array(indicator(n, removed, 0)), "MinvY": MinvY}
        kwargs, out = g.golden(f"submatrix_inv_mult.{name}.{tag}", args, stored=stored)
        k = np.flatnonzero(kwargs["imask"])
        AinvY = out["AinvY"][k] if removed and len(out["AinvY"]) == n else out["AinvY"]
        expected = solve(kwargs["M"], k, kwargs["Y"][k], lower=False)
        g.accuracy(AinvY=normwise(AinvY, expected))
        if tag == "nr0.dense":
            nr0 = args, stored
    cid = "behaviour.submatrix_inv_mult.nr0_returns_input"
    checks = [check("is_argument", argument="MinvY")]
    g.behaviour(cid, nr0[0], checks, file="golden_linalg_50.npz", stored=nr0[1])


def gp_bruteforce(cov, nguard, i0, i1):
    # A Cholesky solve with S_kk for every predicted pixel.
    npix = cov.shape[0]
    W = np.zeros((npix, i1 + 1 - i0))
    var = np.zeros(i1 + 1 - i0)
    for c, i in enumerate(range(i0, i1 + 1)):
        guard = np.arange(max(0, i - nguard), min(i + nguard + 1, npix))
        k = np.setdiff1d(np.arange(npix), guard)
        var[c] = cov[i, i]
        if k.size:
            W[k, c] = solve(cov, k, cov[k, i])
            var[c] -= cov[i, k] @ W[k, c]
    return W, var


def indicator(npix, indices, value=1.0):
    v = np.full(npix, 1.0 - value)
    v[np.asarray(indices, dtype=int)] = value
    return v.tolist()


def cases_spectral(g, ds, special):
    nspec, npix = g.inputs[ds]["flux"].shape
    flux, ivar, mask = ({"input": ds, "key": k} for k in ("flux", "ivar", "mask"))
    D1 = ds == "D1"
    for axis in (0, 1) if ds != "D3" else ():
        g.golden(
            f"maskinterp.{ds}.axis{axis}",
            {"yval": flux, "mask": mask, "axis": {"value": axis}},
        )
    if D1:
        g.golden("average_bilinear.D1", {"yval": flux, "mask": mask})
    g.golden(f"standard_scale.{ds}.mask", {"spec": flux, "ivar": ivar, "mask": mask})
    if D1:
        g.golden("standard_scale.D1.maskNone", {"spec": flux, "ivar": ivar})
        args = {"spec": flux, "ivar": ivar, "mask": {"input": ds, "key": "mask_alt"}}
        g.golden("standard_scale.D1.mask_alt", args)
        for name, m in (("mask", mask), ("maskNone", {"value": None})):
            args = {"spec": flux, "ivar": {"input": ds, "key": "ivar_z"}, "mask": m}
            g.golden(f"standard_scale.D1.ivar0.{name}", args)
    Dvec = {"golden": f"standard_scale.{ds}.mask", "output": "Dvec"}
    cov_spec = {"golden": f"covar.{ds}.scaled", "output": "cov"}
    cov = g.golden(f"covar.{ds}.scaled", {"spec": Dvec})[1]["cov"]
    if D1:
        g.golden("covar.D1.raw", {"spec": flux})
        g.golden("covar.D1.checkmean", {"spec": Dvec, "checkmean": {"value": True}})
        args = {"spec": dict(Dvec, astype="float32")}
        g.behaviour("behaviour.covar.float32_assert", args)

    if ds == "D2":
        # Size: the D2 covariance inverse is not stored, so D2 runs only the bruteforce
        # branch of gaussian_estimate, which ignores it.
        covinv = {"value": None}
        icond = [
            indicator(npix, range(145, 156), 0),
            indicator(npix, range(135, 166), 0),
        ]
        configs = [("one.guard5", [150]), ("many21.guard5", list(range(140, 161)))]
        paths = (True,)
    else:
        cid = f"cholesky_inv.{ds}.cov"
        Minv = g.golden(cid, {"M": cov_spec})[1]["Minv"]
        g.accuracy(Minv=normwise(Minv, cho_solve(cho_factor(cov), np.eye(npix))))
        covinv = {"golden": cid, "output": "Minv"}
        predicted = np.sort(g.rng.choice(npix, 6, replace=False))
        rest = np.setdiff1d(np.arange(npix), predicted)
        conditioning = np.sort(g.rng.choice(rest, 30, replace=False))
        icond = [indicator(npix, range(20, 31), 0), indicator(npix, range(10, 41), 0)]
        icond.append(indicator(npix, conditioning))
        configs = [("one.guard5", [25]), ("many21.guard5", list(range(15, 36)))]
        configs.append(("scattered", predicted.tolist()))
        paths = (False,) if ds == "D3" else (False, True)
        if ds == "D3":
            icond, configs = icond[1:2], configs[1:2]
    for (tag, predicted), conditioning in zip(configs, icond, strict=True):
        for bruteforce in paths:
            twin = f"gaussian_estimate.{ds}.{tag}"
            cid = twin + (".bruteforce" if bruteforce else "")
            args = {"icond": {"array": conditioning, "dtype": "float64"}}
            args["ipred"] = {"array": indicator(npix, predicted), "dtype": "float64"}
            args.update(cov=cov_spec, Dvec=Dvec, covinv=covinv)
            args["bruteforce"] = {"value": bruteforce}
            names = ["predcovar", "predkstar", "kstar"]
            if not bruteforce:
                names.insert(0, "predoverD")
            outputs = [output(name, "full", i) for i, name in enumerate(names)]
            variants = [(cid, args, outputs)]
            predkstar = outputs[names.index("predkstar")]
            if nspec * len(predicted) > 4000 and len(predicted) > 1:
                predkstar["storage"] = "rows"
            # predkstar = Dvec @ predoverD, of this case or of its default-path twin.
            # Without a twin, the predkstar of Dvec = I holds the weights in full.
            if not bruteforce or twin in g.actual:
                predkstar["derived"] = {
                    "left": "Dvec",
                    "golden": twin,
                    "output": "predoverD",
                }
            elif predkstar["storage"] == "rows":
                full = [dict(spec, storage="full") for spec in outputs]
                eye = dict(args, Dvec={"eye": npix})
                variants.insert(0, (cid + ".weights", eye, full))
                predkstar["derived"] = {
                    "left": "Dvec",
                    "golden": cid + ".weights",
                    "output": "predkstar",
                }
            for cid, args, outputs in variants:
                kwargs = g.golden(cid, args, outputs)[0]
                k, ks = np.flatnonzero(kwargs["icond"]), np.flatnonzero(kwargs["ipred"])
                weights = solve(cov, k, cov[np.ix_(k, ks)])
                errors = {}
                if not bruteforce:
                    predoverD = np.zeros((npix, ks.size))
                    predoverD[k] = weights
                    errors["predoverD"] = g.error("predoverD", predoverD)
                predcovar = cov[np.ix_(ks, ks)] - cov[np.ix_(ks, k)] @ weights
                errors["predcovar"] = g.error("predcovar", predcovar)
                errors["predkstar"] = g.error(
                    "predkstar", kwargs["Dvec"][:, k] @ weights
                )
                g.accuracy(**errors)

    # gp_interp(spec=I) returns its weights W; each case on data also checks pred
    # against spec @ W. Size: on D2 only nguard=20 stores the row subset.
    ranges = [(nguard, None) for nguard in ((0, 1, 5, 20, 50) if D1 else (5, 20))]
    if D1:
        # irange [10, 30] as the array [10, 30, 0]: the 2022 length assert is inverted.
        ranges.append((5, [10, 30, 0]))
    for nguard, irange in ranges:
        cid = f"gp_interp.{ds}.ng{nguard}" + (".irange10_30" if irange else "")
        args = {"spec": {"eye": npix}, "cov": cov_spec, "nguard": {"value": nguard}}
        if irange:
            args["irange"] = {"array": irange, "dtype": "int64"}
        weights = g.golden(cid + ".weights", args)[1]
        i0, i1 = irange[:2] if irange else (0, npix - 1)
        W, var = gp_bruteforce(cov, nguard, i0, i1)
        errors = {"pred": normwise(weights["pred"], W)}
        errors["predvar"] = elementwise(weights["predvar"][0], var)
        g.accuracy(**errors)
        derived = {"left": "spec", "golden": cid + ".weights", "output": "pred"}
        storage = "norms" if ds == "D2" and nguard != 20 else "rows"
        outputs = [output("pred", storage, 0, derived=derived), OUTPUTS["gp_interp"][1]]
        kwargs = g.golden(cid, dict(args, spec=Dvec), outputs)[0]
        g.accuracy(pred=g.error("pred", kwargs["spec"] @ W), predvar=errors["predvar"])
    if D1:
        gp_args = {"spec": Dvec, "cov": cov_spec, "nguard": {"value": 5}}
        args = dict(gp_args, irange={"array": [10, 30], "dtype": "int64"})
        g.behaviour("behaviour.gp_interp.irange_two_elements_assert", args)
        args = dict(gp_args, irange={"array": [5, 4, 0], "dtype": "int64"})
        g.behaviour("behaviour.gp_interp.irange_empty", args)
        args = dict(gp_args, bruteforce={"value": True})
        g.behaviour("behaviour.gp_interp.bruteforce_unpack", args)
        checks = [check("dtype", 0), check("shape", 0), check("dtype", 1)]
        checks += [check("shape", 1), check("rows_equal", 1)]
        g.behaviour("behaviour.gp_interp.predvar_shape_identical_rows", gp_args, checks)
    cases_masks(g, ds, special)


def cases_masks(g, ds, special):
    nspec, npix = g.inputs[ds]["flux"].shape
    data = {key: {"input": ds, "key": key} for key in ("flux", "ivar", "mask")}
    D1 = ds == "D1"
    for nsigma in (20, 8, 6, 4) if D1 else (20, 6):
        cid = f"get_chimask.{ds}.nsigma{nsigma}"
        kwargs, out = g.golden(cid, dict(data, nsigma={"value": nsigma}))
        chi = g.margin(cid, out["chimask"], **kwargs)
        if nsigma == 6:
            chi6 = chi
    if D1:
        args = dict(data, nsigma={"value": 6})
        g.behaviour(
            "behaviour.get_chimask.dtype_bool", args, [check("dtype"), check("shape")]
        )
        # A tie at the largest |Z| below 15 of a clean spectrum: 2022 Python flags
        # |Z| >= nsigma.
        largest = np.abs(chi6[special["clean"]]).max(axis=1)
        s = int(special["clean"][int(np.argmax(largest * (largest < 15)))])
        p = int(np.argmax(np.abs(chi6[s])))
        tie = float(np.abs(chi6[s, p]))
        above = float(np.nextafter(tie, np.inf))
        for suffix, nsigma in (("", tie), ("_nextafter", above)):
            args = dict(data, nsigma={"value": nsigma})
            tie_spec = {"index": [s, p], "nextafter": bool(suffix)}
            checks = [check("at", index=[s, p])]
            g.behaviour(
                f"behaviour.get_chimask.ge_tie{suffix}", args, checks, tie=tie_spec
            )
        for name, m in (("mask", data["mask"]), ("maskNone", {"value": None})):
            cid = f"get_chimask.D1.ivar0.{name}.nsigma6"
            ivar_z = {"input": ds, "key": "ivar_z"}
            args = dict(data, ivar=ivar_z, mask=m, nsigma={"value": 6})
            kwargs, out = g.golden(cid, args)
            g.margin(cid, out["chimask"], **kwargs)

    variants = [("default", {}, [20, 8, 6], 64)]
    if D1:
        extra = {"nsigma": {"array": [10.0, 5.5], "dtype": "float64"}}
        variants.append(
            ("nsig10_5p5.maxbad8", dict(extra, maxbadpix={"value": 8}), [10.0, 5.5], 8)
        )
    for tag, extra, nsigmas, maxbadpix in variants:
        cid = f"covar_iter_mask.{ds}.{tag}"
        kwargs, out = g.golden(cid, dict(data, **extra))
        # Replay the 2022 loop with its own get_chimask to store the mask of every pass.
        specutils = sys.modules[SU]
        store = g.files[file_of("covar_iter_mask", ds)]
        rows = np.where(np.sum(kwargs["mask"] != 0, axis=1) <= maxbadpix)[0]
        flux, ivar, mask = (kwargs[key][rows] for key in ("flux", "ivar", "mask"))
        extras = []
        with quiet():
            passes = list(
                _golden.chimask_passes(specutils.get_chimask, flux, ivar, mask, nsigmas)
            )
        for n, (nsigma, (this, chimask)) in enumerate(zip(nsigmas, passes)):
            g.margin(cid, chimask, flux, ivar, this, nsigma)
            name = f"pass{n}_chimask"
            out[name] = chimask
            store[f"{cid}__out__{name}"] = np.packbits(chimask.ravel())
            shape, nsigma = list(chimask.shape), float(nsigma)
            spec = output(name, "packbits", dtype="bool", shape=shape, nsigma=nsigma)
            extras.append(spec)
        finalmask = np.ones((nspec, npix))
        finalmask[rows] = np.logical_or(this, chimask)
        with quiet():
            Dvec = specutils.standard_scale(flux, ivar, finalmask[rows])[0]
            replay = specutils.covar(Dvec)[0]
        assert _golden.same(finalmask, out["finalmask"]), cid
        assert _golden.same(replay, out["cov"]), cid
        out["kept_rows"] = store[f"{cid}__out__kept_rows"] = rows.astype(np.int32)
        extras.append(output("kept_rows", dtype="int32", shape=[int(rows.size)]))
        g.cases[-1]["extras"] = extras
    if D1:
        checks = [check("dtype", 1), check("values", 1), check("shape", 1)]
        checks += [check("dtype", 0), check("shape", 0)]
        g.behaviour("behaviour.covar_iter_mask.output_types", data, checks)
        args = dict(data, mask={"value": None})
        g.behaviour("behaviour.covar_iter_mask.mask_none", args)
        args = dict(data, maxbadpix={"value": -1})
        g.behaviour("behaviour.covar_iter_mask.not_enough_spectra", args)


def verify(g, directory):
    golden = _golden.Golden(directory)
    for case in golden.manifest["cases"]:
        cid = case["id"]
        for name, value in golden.arguments(case).items():
            assert _golden.same(value, g.passed[cid][name]), f"{cid}: argument {name}"
        if case["kind"] != "golden":
            continue
        decoded = golden.expected(case)
        for spec in case["outputs"] + case.get("extras", []):
            actual, expected = g.actual[cid][spec["name"]], decoded[spec["name"]]
            if isinstance(expected, dict):
                if "rows" not in expected:
                    continue
                expected, actual = expected["rows"], actual[expected["rowidx"]]
            assert _golden.same(expected, actual), (
                f"{cid}: {spec['name']} does not decode"
            )


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", repo, *args], capture_output=True, check=True
    ).stdout


def main():
    parser = argparse.ArgumentParser(description="Write the golden files.")
    parser.add_argument("--baseline", type=Path, required=True, help="holds gspice/")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--new-inputs", action="store_true", help="build every dataset from its seed"
    )
    options = parser.parse_args()
    # The outputs, and inputs_MAT.npz, differ in the last bits between 1 and 2 threads.
    assert all(os.environ.get(v) == "2" for v in THREADS), f"set {THREADS} to 2"
    baseline, out = options.baseline.resolve(), options.out.resolve()
    repo = Path(__file__).resolve().parents[2]
    sources = {}
    listing = git(repo, "ls-tree", "--name-only", COMMIT, "gspice/").decode().split()
    for path in [path for path in listing if path.endswith(".py")]:
        sources[path] = hashlib.sha256(
            git(repo, "show", f"{COMMIT}:{path}")
        ).hexdigest()
        extracted = hashlib.sha256((baseline / path).read_bytes()).hexdigest()
        assert extracted == sources[path], f"{baseline / path} differs from {COMMIT}"
    sys.path.insert(0, str(baseline))
    for module in (SU, MU, DJ):
        with quiet():
            path = Path(importlib.import_module(module).__file__).resolve()
        assert baseline in path.parents, f"{module} is not imported from --baseline"

    reuse = not options.new_inputs and (out / "manifest.json").exists()
    assert reuse or options.new_inputs or not any(out.glob("inputs_*.npz")), (
        f"{out} holds inputs files but no manifest.json: pass --new-inputs"
    )
    g = Generator(_golden.Golden(out) if reuse else None)
    g.dataset("E", make_edge())
    cases_edge(g)
    g.dataset("MAT", make_matrices())
    cases_matrices(g)
    for ds in ("D1", "D2", "D3", "D4"):
        stored, special = make_spectral(ds)
        g.dataset(ds, stored)
        (cases_masks if ds == "D4" else cases_spectral)(g, ds, special)

    out.mkdir(parents=True, exist_ok=True)
    for name, arrays in g.files.items():
        write_npz(out / name, arrays)
    config = np.show_config(mode="dicts")
    blas = config["Build Dependencies"]["blas"]
    environment = {"python": platform.python_version(), "numpy": np.__version__}
    environment["scipy"] = scipy.__version__
    environment["blas"] = blas.get("openblas configuration", blas["name"])
    environment["cpu_features"] = config["SIMD Extensions"]["found"]
    hashes = {
        name: {k: _golden.array_sha256(v) for k, v in sorted(inputs.items())}
        for name, inputs in g.inputs.items()
    }
    manifest = {"baseline": {"commit": COMMIT, "sha256": sources}}
    manifest.update(
        environment=environment, seeds=SEEDS, datasets=hashes, cases=g.cases
    )
    (out / "manifest.json").write_text(dump_json(manifest) + "\n", encoding="utf-8")
    verify(g, out)

    print(f"smallest relative margin of |Z| to nsigma: {min(g.margins.values()):.1e}")
    sizes = [path.stat().st_size for path in out.glob("*.npz")]
    print(f"{len(sizes)} files, {sum(sizes)} bytes, largest {max(sizes)}")


if __name__ == "__main__":
    main()
