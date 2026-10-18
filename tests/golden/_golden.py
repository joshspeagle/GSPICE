import dataclasses
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def array_sha256(a):
    a = np.ascontiguousarray(a)
    digest = hashlib.sha256(a.dtype.str.encode())
    digest.update(repr(tuple(int(n) for n in a.shape)).encode())
    digest.update(a.tobytes())
    return digest.hexdigest()


def build_spd_matrix(t, V, d):
    # M[i, j] = t[|i - j|] + sum_m V[m, i] V[m, j] + d[i] [i == j], summed in a fixed
    # order of correctly rounded elementwise operations, so every platform gets the
    # same bits.
    n = t.shape[0]
    idx = np.arange(n)
    M = t[np.abs(idx[:, np.newaxis] - idx[np.newaxis, :])]
    for row in V:
        M = M + np.multiply.outer(row, row)
    M[idx, idx] = M[idx, idx] + d
    return np.ascontiguousarray(M)


def build_inputs(stored):
    if "flux_hi" in stored:
        # Flux is an integer times a power of two, ivar the product of two factors: both
        # are exact.
        # A C int exponent: numpy 1.x has no ldexp loop for int64 on Windows.
        scale = np.ldexp(1.0, -stored["flux_exp"].astype(np.intc))
        counts = stored["flux_hi"].astype(np.int32) * 256 + stored["flux_lo"]
        mask = stored["mask"].astype(np.int32)
        ivar = stored["ivar_a"][:, np.newaxis] * stored["ivar_b"][np.newaxis, :]
        ivar[mask != 0] = 0.0
        inputs = {
            "flux": counts.astype(np.float64) * scale[:, np.newaxis],
            "ivar": ivar,
            "mask": mask,
            "rows_subset": stored["rows_subset"].astype(np.int64),
        }
        if "mask_alt" in stored:
            inputs["mask_alt"] = stored["mask_alt"].astype(np.int32)
    elif "flux" in stored:
        inputs = {key: stored[key] for key in ("flux", "ivar", "mask")}
    else:
        names = [key[:-3] for key in stored if key.endswith("__t")]
        inputs = {
            name: build_spd_matrix(
                stored[name + "__t"], stored[name + "__V"], stored[name + "__d"]
            )
            for name in names
        }
        inputs.update({"Y__" + name: stored["Y__" + name] for name in names})
    if "ivar_zero" in stored:
        inputs["ivar_z"] = np.where(stored["ivar_zero"] != 0, 0.0, inputs["ivar"])
    return inputs


def resolve(spec, inputs, golden, stored):
    if "value" in spec:
        return spec["value"]
    if "array" in spec:
        value = np.array(spec["array"], dtype=spec["dtype"])
    elif "eye" in spec:
        value = np.eye(spec["eye"])
    elif "input" in spec:
        value = inputs(spec["input"])[spec["key"]]
    elif "golden" in spec:
        value = golden(spec["golden"])[spec["output"]]
    else:
        value = stored(spec["stored"])
    return select(value, spec)


def select(value, spec):
    if "drop_rows" in spec:
        value = np.delete(value, spec["drop_rows"], axis=0)
    if "row" in spec:
        value = value[spec["row"]]
    if "cols" in spec:
        value = value[:, spec["cols"]]
    if "astype" in spec:
        value = value.astype(spec["astype"])
    return np.array(value, order="C", copy=True)


def arrays(result):
    # The public arrays of a Prediction or a Repaired.
    # blocks is set only by repair.
    fields = dataclasses.fields(result)
    names = [f.name for f in fields if f.name[0] != "_" and f.name != "blocks"]
    return tuple(getattr(result, name) for name in names)


def same(a, b):
    if isinstance(a, tuple):
        return isinstance(b, tuple) and len(a) == len(b) and all(map(same, a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    return a == b


def checked(function, *args, **kwargs):
    # The result of function, after a check that it changes no input array and that no
    # output array shares memory with one.
    inputs = [value for value in (*args, *kwargs.values()) if value is not None]
    pristine = [value.copy() for value in inputs]
    result = function(*args, **kwargs)
    for value, copy in zip(inputs, pristine, strict=True):
        assert same(value, copy)
    for output in result if isinstance(result, tuple) else (result,):
        assert not any(np.shares_memory(output, value) for value in inputs)
    return result


def norms(a):
    a = np.asarray(a, dtype=np.float64)
    return np.linalg.norm(a, axis=1), np.linalg.norm(a, axis=0)


def zscore(specutils, flux, ivar, mask):
    # The calls of get_chimask in its order: masks and ties depend on the last bit of Z.
    Dvec = specutils("standard_scale")(spec=flux, ivar=ivar, mask=mask)[0]
    cov = specutils("covar")(spec=Dvec)[0]
    pred, predvar = specutils("gp_interp")(spec=Dvec, cov=cov, nguard=20)
    return (Dvec - pred) / np.sqrt(predvar)


def chimask_passes(get_chimask, flux, ivar, mask, nsigmas):
    # The 2022 covar_iter_mask loop: each pass starts from the input mask and the mask
    # of the previous pass only.
    chimask = np.zeros(mask.shape)
    for nsigma in nsigmas:
        this = np.logical_or(chimask, mask != 0)
        chimask = get_chimask(flux=flux, ivar=ivar, mask=this, nsigma=nsigma)
        yield this, chimask


PROPERTIES = {
    "dtype": lambda part, check, kwargs: str(np.asarray(part).dtype),
    "shape": lambda part, check, kwargs: list(np.shape(part)),
    "nan_index": lambda part, check, kwargs: np.flatnonzero(np.isnan(part)).tolist(),
    "values": lambda part, check, kwargs: np.unique(part).tolist(),
    "rows_equal": lambda part, check, kwargs: bool(np.all(part == part[:1])),
    "tolist": lambda part, check, kwargs: np.asarray(part).tolist(),
    "at": lambda part, check, kwargs: part[tuple(check["index"])].item(),
    "is_argument": lambda part, check, kwargs: part is kwargs[check["argument"]],
    "row_unchanged": lambda part, check, kwargs: same(
        part[check["row"]], kwargs[check["argument"]][check["row"]]
    ),
}


def property_value(result, check, kwargs):
    part = result if check["output"] is None else result[check["output"]]
    return PROPERTIES[check["prop"]](part, check, kwargs)


class Golden:
    def __init__(self, directory=Path(__file__).parent):
        self.directory = Path(directory)
        text = (self.directory / "manifest.json").read_text(encoding="utf-8")
        self.manifest = json.loads(text)
        self.cases = {case["id"]: case for case in self.manifest["cases"]}
        self._files = {}
        self._inputs = {}
        self._expected = {}

    def file(self, name):
        if name not in self._files:
            with np.load(self.directory / name) as archive:
                self._files[name] = dict(archive)
        return self._files[name]

    def inputs(self, dataset):
        if dataset not in self._inputs:
            inputs = build_inputs(self.file(f"inputs_{dataset}.npz"))
            for key, digest in self.manifest["datasets"][dataset].items():
                assert array_sha256(inputs[key]) == digest, (
                    f"{dataset}/{key} does not rebuild"
                )
            self._inputs[dataset] = inputs
        return self._inputs[dataset]

    def arguments(self, case):
        def golden(cid):
            return self.expected(self.cases[cid])

        def stored(name):
            return self.file(case["file"])[f"{case['id']}__in__{name}"]

        return {
            name: resolve(spec, self.inputs, golden, stored)
            for name, spec in case["args"].items()
        }

    def expected(self, case):
        if case["id"] not in self._expected:
            kwargs = self.arguments(case)
            decoded = {}
            for output in case["outputs"] + case.get("extras", []):
                decoded[output["name"]] = self._decode(case, output, kwargs, decoded)
            self._expected[case["id"]] = decoded
        return self._expected[case["id"]]

    def _decode(self, case, output, kwargs, decoded):
        stored = self.file(case["file"])
        key = f"{case['id']}__out__{output['name']}"
        storage = output["storage"]
        shape = tuple(output["shape"])
        if storage == "full":
            return stored[key]
        if storage == "onerow":
            return np.broadcast_to(stored[key], shape)
        if storage == "triu":
            full = np.zeros(shape, dtype=stored[key].dtype)
            i, j = np.triu_indices(shape[0])
            full[i, j] = stored[key]
            full[j, i] = stored[key]
            return full
        if storage == "packbits":
            bits = np.unpackbits(stored[key], count=math.prod(shape)).reshape(shape)
            return bits.astype(output["dtype"])
        if storage == "masked":
            full = np.array(kwargs["yval"], dtype=np.float64)
            full[kwargs["mask"] != 0] = stored[key]
            return full
        if storage == "scaled":
            # Dvec = (spec, interpolated at bad pixels) * refscale - refmean
            mask = kwargs.get("mask")
            bad = kwargs["ivar"] == 0 if mask is None else mask != 0
            full = np.array(kwargs["spec"], dtype=np.float64)
            full[bad] = stored[key]
            full = full * decoded["refscale"][:, np.newaxis]
            return full - decoded["refmean"].reshape(1, -1)
        parts = ("rows", "rowidx", "rownorm", "colnorm")
        return {p: stored[f"{key}__{p}"] for p in parts if f"{key}__{p}" in stored}
