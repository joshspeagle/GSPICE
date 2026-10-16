import importlib
import math
import warnings
from functools import partial

import numpy as np
import pytest

from _golden import Golden, chimask_passes, norms, property_value, zscore

GOLDEN = Golden()

PUBLIC_2022 = {
    "gspice.specUtils": (
        "standard_scale",
        "covar",
        "get_chimask",
        "gaussian_estimate",
        "gp_interp",
        "covar_iter_mask",
    ),
    "gspice.matrixUtils": ("cholesky_inv", "submatrix_inv", "submatrix_inv_mult"),
    "gspice.djs_maskinterp": ("maskinterp1", "maskinterp", "average_bilinear"),
}

RTOL = 1e-12
# An override is the next power of ten above 10 times the difference, in the comment,
# between the 2022 result and a float64 Cholesky solve, measured with the metrics of
# check_output (manifest "baseline_accuracy"). That difference estimates where a correct
# new implementation lands. Across 24 Linux builds (numpy 1.26.4 and 2.5.4, scipy 1.11.4
# and 1.18.1, 1 and 2 threads, seven OpenBLAS kernels) the 2022 results moved by up to
# 2.7 times that difference where it is at least 3e-14. The smallest margin of
# tolerance over observed spread is 4.5 (gp_interp.D1 predvar at the default 1e-12). A
# direct Cholesky solve and the precision route keep a margin of at least 6.
RTOL_OVERRIDES = {
    # Looser than 1e-10. cond 1e6: the 2022 results are as accurate as a direct solve
    # (both about 1e-11 against extended precision), and the tolerance is about 10 times
    # the float64 limit cond * u = 1.1e-10, where u = 2^-53 is the unit roundoff.
    ("cholesky_inv.M50b", "Minv"): 1e-9,  # 1.9e-11
    ("cholesky_inv.M300b", "Minv"): 1e-9,  # 1.7e-11
    ("submatrix_inv.M50b.scattered", "Ainv"): 1e-9,  # 1.1e-11
    ("submatrix_inv_mult.M50b.scattered.cols.padF", "AinvY"): 1e-9,  # 1.0e-11
    ("submatrix_inv_mult.M50b.bruteforce.col", "AinvY"): 1e-9,  # 1.1e-11
    ("submatrix_inv_mult.M300b.scattered.dense.padT", "AinvY"): 1e-9,  # 2.5e-11
    # Looser than 1e-10. D3 (cond 8.6e4), against extended precision: the 2022 predvar
    # is wrong by up to 7.6e-10 and predcovar by 1.5e-11, a direct solve by 1.3e-13 and
    # 7e-15.
    ("gaussian_estimate.D3.many21.guard5", "predcovar"): 1e-9,  # 1.5e-11
    ("gp_interp.D3.ng5.weights", "predvar"): 1e-8,  # 7.6e-10
    ("gp_interp.D3.ng5", "predvar"): 1e-8,  # 7.6e-10
    ("gp_interp.D3.ng20.weights", "predvar"): 1e-8,  # 2.7e-10
    ("gp_interp.D3.ng20", "predvar"): 1e-8,  # 2.7e-10
    # M50b with cond 1e6; M300a (cond 1e3) with 41 of 300 rows removed.
    ("submatrix_inv_mult.M50b.scattered.dense.padT", "AinvY"): 1e-10,  # 6.6e-12
    ("submatrix_inv_mult.M300a.contig41.col.padT", "AinvY"): 1e-11,  # 4.6e-13
    # D3.
    ("cholesky_inv.D3.cov", "Minv"): 1e-11,  # 4.0e-13
    ("gaussian_estimate.D3.many21.guard5", "predoverD"): 1e-11,  # 7.7e-13
    ("gaussian_estimate.D3.many21.guard5", "predkstar"): 1e-10,  # 2.6e-12
    ("gp_interp.D3.ng5.weights", "pred"): 1e-11,  # 6.8e-13
    ("gp_interp.D3.ng5", "pred"): 1e-10,  # 1.4e-12
    ("gp_interp.D3.ng20.weights", "pred"): 1e-11,  # 7.4e-13
    ("gp_interp.D3.ng20", "pred"): 1e-10,  # 1.6e-12
}

# normwise: max|a - b| <= rtol * max|b| + atol, for solves and matrices.
# elementwise: |a - b| <= rtol * |b| + atol, for scales, variances, interpolated values.
MODE = {
    ("maskinterp", "ynew"): "elementwise",
    ("maskinterp1", "ynew"): "elementwise",
    ("average_bilinear", "interp"): "elementwise",
    ("standard_scale", "refscale"): "elementwise",
    ("standard_scale", "refmean"): "normwise",
    ("standard_scale", "Dvec"): "elementwise",
    ("covar", "cov"): "normwise",
    ("covar", "refmean"): "normwise",
    ("cholesky_inv", "Minv"): "normwise",
    ("submatrix_inv", "Ainv"): "normwise",
    ("submatrix_inv_mult", "AinvY"): "normwise",
    ("gaussian_estimate", "predoverD"): "normwise",
    ("gaussian_estimate", "predcovar"): "normwise",
    ("gaussian_estimate", "predkstar"): "normwise",
    ("gaussian_estimate", "kstar"): "exact",
    ("gp_interp", "pred"): "normwise",
    ("gp_interp", "predvar"): "elementwise",
    ("get_chimask", "chimask"): "exact",
    ("covar_iter_mask", "cov"): "normwise",
    ("covar_iter_mask", "finalmask"): "exact",
}

# atol = factor * max|basis| over the compared entries; the basis is the expected value
# or the spec argument. Interpolation and scaling are elementwise arithmetic: an
# independent implementation agrees to 3 eps max|b|. The refmean of centred data (a
# Dvec) is rounding noise; the refmean of raw flux gets no atol.
ATOL = {
    ("maskinterp", "ynew"): (1e-14, "expected"),
    ("maskinterp1", "ynew"): (1e-14, "expected"),
    ("average_bilinear", "interp"): (1e-14, "expected"),
    ("standard_scale", "Dvec"): (1e-13, "expected"),
    ("covar", "refmean"): (1e-13, "spec"),
}

CHANGED = {
    "behaviour.gp_interp.irange_two_elements_assert": "D-01",
    "behaviour.gp_interp.irange_empty": "D-18",
    "standard_scale.E.with_fully_masked": "Q1",
    "behaviour.standard_scale.fully_masked_nan": "Q1",
    "behaviour.covar_iter_mask.mask_none": "D-14",
}


@pytest.fixture(scope="session")
def implementation():
    # The manifest names the 2022 modules, which now live in gspice.legacy.
    return lambda module, function: getattr(
        importlib.import_module(module.replace("gspice.", "gspice.legacy.", 1)),
        function,
    )


def case_params(kind, keep=lambda case: True):
    params = []
    for case in GOLDEN.manifest["cases"]:
        if case["kind"] == kind and keep(case):
            reason = CHANGED.get(case["id"])
            marks = pytest.mark.xfail(strict=True, reason=reason) if reason else ()
            params.append(pytest.param(case["id"], marks=marks))
    return params


def assert_close(actual, expected, mode, rtol, atol, label):
    a, b = np.asarray(actual), np.asarray(expected)
    assert a.shape == b.shape, f"{label}: shape {a.shape}, expected {b.shape}"
    if mode == "exact":
        np.testing.assert_array_equal(a, b, err_msg=label)
        return
    if mode == "normwise":
        atol += rtol * np.max(np.abs(b), where=np.isfinite(b), initial=0.0)
        rtol = 0.0
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=label)


def atol_of(case, output, expected, kwargs):
    if (case["function"], output["name"]) not in ATOL:
        return 0.0
    factor, basis = ATOL[(case["function"], output["name"])]
    if basis == "spec" and case["args"]["spec"].get("output") != "Dvec":
        return 0.0
    values = np.asarray(
        kwargs["spec"] if basis == "spec" else expected, dtype=np.float64
    )
    if output["storage"] == "masked":
        values = values[kwargs["mask"] != 0]
    values = values[np.isfinite(values)]
    return factor * np.abs(values).max() if values.size else 0.0


def check_output(case, output, actual, expected, kwargs, mode):
    label = f"{case['id']}:{output['name']}"
    actual = np.asarray(actual)
    assert actual.dtype == output["dtype"], f"{label}: dtype {actual.dtype}"
    assert list(actual.shape) == output["shape"], f"{label}: shape {actual.shape}"
    rtol = RTOL_OVERRIDES.get((case["id"], output["name"]), RTOL)
    storage = output["storage"]
    if storage in ("rows", "norms"):
        if "rows" in expected:
            rows = actual[expected["rowidx"]]
            assert_close(rows, expected["rows"], mode, rtol, 0.0, label + " rows")
        for name, value in zip(("rownorm", "colnorm"), norms(actual), strict=True):
            assert_close(
                value, expected[name], "elementwise", rtol, 0.0, f"{label} {name}"
            )
    else:
        atol = atol_of(case, output, expected, kwargs)
        assert_close(actual, expected, mode, rtol, atol, label)
        if storage == "masked":
            unmasked = kwargs["mask"] == 0
            assert_close(
                actual[unmasked], kwargs["yval"][unmasked], "exact", 0, 0, label
            )
    if "derived" in output:
        derived = output["derived"]
        right = GOLDEN.expected(GOLDEN.cases[derived["golden"]])[derived["output"]]
        left = np.asarray(kwargs[derived["left"]], dtype=np.float64)
        whole = (left @ right).reshape(actual.shape)
        assert_close(actual, whole, "normwise", rtol, 0.0, label + " whole array")


def test_every_2022_public_function_has_a_golden_case():
    covered = {
        (case["module"], case["function"])
        for case in GOLDEN.manifest["cases"]
        if case["kind"] == "golden"
    }
    assert {(m, f) for m, fs in PUBLIC_2022.items() for f in fs} <= covered


def test_rtol_overrides_follow_the_baseline_accuracy_of_the_manifest():
    required = {}
    for case in GOLDEN.manifest["cases"]:
        for name, error in case.get("baseline_accuracy", {}).items():
            assert math.isfinite(error), f"{case['id']}:{name}: accuracy {error}"
            rtol = float(f"1e{math.ceil(math.log10(max(10 * error, RTOL)))}")
            if rtol > RTOL:
                required[(case["id"], name)] = (rtol, error)
    table = "\n".join(
        f'    ("{cid}", "{name}"): {rtol:.0e},  # {error:.1e}'
        for (cid, name), (rtol, error) in required.items()
    )
    assert RTOL_OVERRIDES == {key: rtol for key, (rtol, _) in required.items()}, (
        "the manifest requires\n" + table
    )


@pytest.mark.parametrize("case_id", case_params("golden"))
def test_golden_case_matches_2022_outputs(case_id, implementation):
    case = GOLDEN.cases[case_id]
    kwargs = GOLDEN.arguments(case)
    pristine = {k: v.copy() for k, v in kwargs.items() if isinstance(v, np.ndarray)}
    result = implementation(case["module"], case["function"])(**kwargs)
    for name, value in pristine.items():
        assert kwargs[name].tobytes() == value.tobytes(), f"input {name} was modified"
    outputs = case["outputs"]
    assert isinstance(result, tuple) == ("index" in outputs[0])
    values = result if isinstance(result, tuple) else (result,)
    assert len(values) == len(outputs)
    expected = GOLDEN.expected(case)
    for output in outputs:
        mode = MODE[(case["function"], output["name"])]
        actual = values[output.get("index", 0)]
        check_output(case, output, actual, expected[output["name"]], kwargs, mode)


@pytest.mark.parametrize(
    "case_id", case_params("golden", lambda case: "extras" in case)
)
def test_covar_iter_mask_passes_match_2022(case_id, implementation):
    case = GOLDEN.cases[case_id]
    kwargs = GOLDEN.arguments(case)
    expected = GOLDEN.expected(case)
    kept = expected["kept_rows"]
    flux, ivar, mask = (kwargs[key][kept] for key in ("flux", "ivar", "mask"))
    get_chimask = implementation("gspice.specUtils", "get_chimask")
    outputs = [extra for extra in case["extras"] if "nsigma" in extra]
    nsigmas = [output["nsigma"] for output in outputs]
    passes = chimask_passes(get_chimask, flux, ivar, mask, nsigmas)
    for output, (_, chimask) in zip(outputs, passes, strict=True):
        check_output(case, output, chimask, expected[output["name"]], kwargs, "exact")


@pytest.mark.parametrize("case_id", case_params("behaviour"))
def test_2022_behaviour_holds(case_id, implementation):
    case = GOLDEN.cases[case_id]
    kwargs = GOLDEN.arguments(case)
    expect = case["expect"]
    function = implementation(case["module"], case["function"])
    if "tie" in expect:
        kwargs["nsigma"] = tie_nsigma(implementation, kwargs, expect["tie"])
    if "raises" in expect:
        with pytest.raises(Exception) as raised:
            function(**kwargs)
        assert expect["raises"] in [c.__name__ for c in type(raised.value).__mro__]
        assert expect.get("message", "") in str(raised.value)
        return
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = function(**kwargs)
    for check in expect["checks"]:
        assert property_value(result, check, kwargs) == check["value"], check
    assert set(expect.get("warns", [])) <= {w.category.__name__ for w in caught}


def tie_nsigma(implementation, kwargs, tie):
    # A tie exists only bit for bit, so |Z| comes from the implementation's own
    # functions. The records hold only for a get_chimask that computes Z with these
    # calls, as the 2022 one does.
    specutils = partial(implementation, "gspice.specUtils")
    z = zscore(specutils, kwargs["flux"], kwargs["ivar"], kwargs["mask"])
    z = abs(float(z[tuple(tie["index"])]))
    return float(np.nextafter(z, np.inf)) if tie["nextafter"] else z
