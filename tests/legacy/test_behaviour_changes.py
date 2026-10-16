import ast
import importlib
import pickle
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse

import gspice
from _golden import Golden, same
from gspice.legacy import (
    cholesky_inv,
    covar,
    covar_iter_mask,
    get_chimask,
    gp_interp,
    maskinterp,
    maskinterp1,
    matrixUtils,
    standard_scale,
    submatrix_inv,
    submatrix_inv_mult,
)

# The names that the 2022 gspice/__init__.py imported.
NAMES_2022 = [
    "cholesky_inv",
    "submatrix_inv",
    "submatrix_inv_mult",
    "standard_scale",
    "covar",
    "get_chimask",
    "covar_iter_mask",
    "gaussian_estimate",
    "gp_interp",
    "maskinterp1",
    "maskinterp",
]

# pickle.dumps(gspice.standard_scale, protocol=4) with the package of 1fbab9f.
PICKLE_2022 = (
    b"\x80\x04\x95'\x00\x00\x00\x00\x00\x00\x00\x8c\x10gspice.specUtils\x94"
    b"\x8c\x0estandard_scale\x94\x93\x94."
)

GOLDEN = Golden()
GP_INTERP_D1 = GOLDEN.arguments(GOLDEN.cases["gp_interp.D1.ng5"])
SPEC, COV = GP_INTERP_D1["spec"], GP_INTERP_D1["cov"]
GP_INTERP_D2 = GOLDEN.arguments(GOLDEN.cases["gp_interp.D2.ng5"])
EYE = np.eye(3)
INV = {"M": EYE, "Minv": EYE, "imask": EYE[0]}
MULT = INV | {"Y": EYE, "MinvY": EYE}
INTERP = {"yval": EYE, "mask": EYE, "axis": 0}

# Each 2022 assert of gspice.legacy, with arguments that fail it, and its message.
ASSERTS = [
    (covar, {"spec": np.ones((3, 2), np.float32)}, "spec.dtype must be float64"),
    (gp_interp, {"spec": EYE, "cov": EYE[:, :2]}, "covariance must be square"),
    (gp_interp, {"spec": EYE[:, :2], "cov": EYE}, "incompatible dimensions"),
    (
        covar_iter_mask,
        {"flux": EYE[:2], "ivar": EYE[:2], "mask": np.zeros((2, 3))},
        "Not enough good spectra",
    ),
    (submatrix_inv, INV | {"M": EYE[:, :2]}, "M must be"),
    (submatrix_inv, INV | {"imask": EYE[0, :2]}, "imask"),
    (submatrix_inv_mult, MULT | {"M": EYE[:, :2]}, "M must be"),
    (submatrix_inv_mult, MULT | {"imask": EYE[0, :2]}, "imask"),
    (submatrix_inv_mult, MULT | {"Y": EYE[0]}, "Y must be"),
    (submatrix_inv_mult, MULT | {"MinvY": EYE[0]}, "MinvY must be"),
    (maskinterp, INTERP | {"yval": EYE[0]}, "yval"),
    (maskinterp, INTERP | {"mask": EYE[0]}, "mask"),
    (maskinterp, INTERP | {"mask": EYE[:2]}, "yval, mask"),
    (maskinterp, INTERP | {"axis": 2}, "axis"),
]


@pytest.mark.parametrize(
    "module", ["specUtils", "matrixUtils", "djs_maskinterp", "gspiceMain"]
)
def test_2022_module_paths_no_longer_import(module):
    # In 1fbab9f both worked, except gspice.gspiceMain before an import of the module.
    # Without the attribute, `from gspice import <module>` and the star import fail.
    with pytest.raises(ModuleNotFoundError) as raised:
        importlib.import_module(f"gspice.{module}")
    assert raised.value.name == f"gspice.{module}"
    assert not hasattr(gspice, module)


@pytest.mark.parametrize("name", NAMES_2022)
def test_top_level_2022_name_is_the_legacy_function(name):
    assert getattr(gspice, name) is getattr(gspice.legacy, name)


def test_2022_pickle_of_standard_scale_no_longer_loads():
    with pytest.raises(ModuleNotFoundError, match="gspice.specUtils"):
        pickle.loads(PICKLE_2022)


@pytest.mark.parametrize(("function", "kwargs", "message"), ASSERTS)
def test_2022_assert_raises_an_error_that_is_also_a_value_error(
    function, kwargs, message
):
    # 1fbab9f raised a plain AssertionError; under python -O it returned a result or
    # raised another error.
    with pytest.raises(ValueError, match=message) as raised:
        function(**kwargs)
    assert isinstance(raised.value, AssertionError)


@pytest.mark.parametrize(
    ("function", "kwargs", "argument"),
    [(submatrix_inv, INV, "Minv"), (submatrix_inv_mult, MULT, "MinvY")],
)
def test_submatrix_functions_copy_their_input_when_nothing_is_removed(
    function, kwargs, argument
):
    # 1fbab9f returned Minv, or MinvY, itself: a later change to the result changed
    # the caller's array. The golden cases nr0 check the values.
    result = function(**kwargs | {"imask": np.ones(3)})
    assert not np.shares_memory(result, kwargs[argument])


def test_padded_submatrix_inv_mult_keeps_a_sparse_minvy():
    # 1fbab9f returned a csr_matrix equal to the result for the dense MinvY.
    M = EYE + np.ones((3, 3))
    Minv = np.linalg.inv(M)
    kwargs = {"M": M, "Minv": Minv, "imask": np.array([1, 0, 1]), "Y": EYE}
    result = submatrix_inv_mult(**kwargs, MinvY=scipy.sparse.csr_matrix(Minv))
    assert isinstance(result, scipy.sparse.csr_matrix)
    np.testing.assert_equal(result.toarray(), submatrix_inv_mult(**kwargs, MinvY=Minv))


def legacy_trees():
    return {
        path.name: ast.parse(path.read_text(encoding="utf-8"))
        for path in Path(gspice.legacy.__file__).parent.glob("*.py")
    }


def test_legacy_has_no_assert_statement():
    # python -O removes asserts, so a check written as one would not run there.
    for name, tree in legacy_trees().items():
        assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree)), name


def test_legacy_prints_only_the_mean_check_of_covar():
    # 1fbab9f also printed progress, timings, shapes and notices from 10 other calls.
    callers = [
        getattr(statement, "name", "<module>")
        for tree in legacy_trees().values()
        for statement in tree.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "print"
    ]
    assert callers == ["covar"]


def test_legacy_checks_run_under_python_o():
    # Under python -O, 1fbab9f skipped the assert and returned a float32 covariance.
    code = (
        "import numpy as np; from gspice.legacy import covar; "
        "covar(np.ones((3, 2), np.float32))"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", code], capture_output=True, text=True, check=False
    )
    assert "spec.dtype must be float64" in result.stderr


def test_empty_cholesky_inv_returns_an_empty_inverse_without_dpotri(monkeypatch):
    # 1fbab9f returned the same array, and dpotri printed "** On entry to DPOTRI
    # parameter number  4 had an illegal value" on stdout.
    monkeypatch.setattr(matrixUtils.lapack, "dpotri", None)
    inverse = cholesky_inv(np.zeros((0, 0)))
    assert (inverse.shape, inverse.dtype) == ((0, 0), np.float64)


@pytest.mark.parametrize(
    "irange",
    [
        np.array([10, 30]),
        [10, 30],
        (10, 30),
        np.array([10, 30, 0], np.uint64),
        [np.uint64(10), 30],
    ],
)
def test_integer_irange_equals_the_2022_form(irange):
    # 1fbab9f raised AssertionError for a 2-element array, AttributeError for a list or
    # a tuple, and IndexError (numpy 2) or TypeError (numpy 1.26) for a uint64 array.
    # The golden cases use [10, 30, 0].
    expected = gp_interp(SPEC, COV, 5, irange=np.array([10, 30, 0]))
    np.testing.assert_equal(gp_interp(SPEC, COV, 5, irange=irange), expected)


@pytest.mark.parametrize(
    ("irange", "error"),
    [
        (np.array([-1, 49, 0]), ValueError),
        (np.array([5, 4, 0]), ValueError),
        (np.array([10, 50, 0]), ValueError),
        (np.array([10]), ValueError),
        (np.array([[10, 30, 0]]), ValueError),
        (np.array([10.0, 30.0, 0.0]), TypeError),
        (np.array([True, False, False]), TypeError),
        ([np.True_, 30], TypeError),
    ],
)
def test_invalid_irange_is_rejected(irange, error):
    # 1fbab9f returned a (2000, 51) result whose first column is 0 for [-1, 49, 0],
    # raised UnboundLocalError, IndexError or TypeError for the other arrays, and raised
    # AttributeError for the list.
    with pytest.raises(error, match="irange"):
        gp_interp(SPEC, COV, 5, irange=irange)


@pytest.mark.parametrize("irange", [None, [0, 0]])
def test_spectra_without_pixels_are_rejected(irange):
    # 1fbab9f raised UnboundLocalError from a print at its end, and AttributeError for
    # the list.
    with pytest.raises(ValueError, match="spec"):
        gp_interp(np.zeros((3, 0)), np.zeros((0, 0)), irange=irange)


def test_negative_nguard_is_rejected():
    # 1fbab9f predicted each pixel from itself: on D1, pred = X to 3.8e-13.
    with pytest.raises(ValueError, match="nguard"):
        gp_interp(SPEC, COV, nguard=-1)


@pytest.mark.parametrize(
    "nguard", [np.uint8(5), np.uint64(5), np.int8(100), np.array(5), True]
)
def test_integer_nguard_equals_the_python_integer(nguard):
    # Under numpy 2, 1fbab9f computed the windows in the type of nguard: np.uint8(5)
    # predicted pixels 0 to 4 from themselves, np.int8(100) pixels 27 to 49, and
    # np.uint64(5) raised TypeError. A 0-d array and True worked, and still do.
    expected = gp_interp(SPEC, COV, int(nguard))
    np.testing.assert_equal(gp_interp(SPEC, COV, nguard), expected)


@pytest.mark.parametrize(
    ("nguard", "irange"),
    [(np.int8(5), np.array([126, 130, 0])), (np.uint8(5), np.array([254, 258, 0]))],
)
def test_fixed_width_nguard_works_past_the_range_of_its_type(nguard, irange):
    # Under numpy 2, 1fbab9f raised OverflowError on D2 at pixel 128 for np.int8 and at
    # pixel 256 for np.uint8.
    spec, cov = GP_INTERP_D2["spec"], GP_INTERP_D2["cov"]
    expected = gp_interp(spec, cov, 5, irange=irange)
    np.testing.assert_equal(gp_interp(spec, cov, nguard, irange=irange), expected)


@pytest.mark.parametrize("nguard", [np.True_, np.array([5])])
def test_numpy_bool_or_array_nguard_raises_type_error(nguard):
    # 1fbab9f read np.True_ as 1 and raised ValueError for np.array([5]).
    with pytest.raises(TypeError, match="nguard"):
        gp_interp(SPEC, COV, nguard=nguard)


# The 2022 expression yval * 0 + value computes inf * 0.
@pytest.mark.filterwarnings("ignore:invalid value encountered in multiply")
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_non_finite_bad_entries_get_the_value_of_the_only_good_entry(value):
    # 1fbab9f returned NaN at the 23 bad entries of E row 2, and without the non-finite
    # values it returned the value of entry 7 everywhere (record maskinterp1.E.row2).
    case = GOLDEN.cases["maskinterp1.E.row2"]
    kwargs = GOLDEN.arguments(case)
    kwargs["yval"][kwargs["mask"] != 0] = value
    expected = GOLDEN.expected(case)["ynew"]
    assert same(maskinterp1(**kwargs), expected)


@pytest.mark.filterwarnings("ignore:invalid value encountered in multiply")
@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_infinite_only_good_entry_keeps_the_2022_values(value):
    # 1fbab9f returned [value, nan, value, nan]: inf * 0 is NaN at the good entry.
    result = maskinterp1(np.array([1.0, value, -2.0, np.nan]), np.array([1, 0, 1, 1]))
    np.testing.assert_array_equal(result, [value, np.nan, value, value])


def test_object_array_with_one_good_entry_keeps_the_2022_values():
    # 1fbab9f returned yval * 0 + Fraction(2), the value of the good entry.
    yval = np.array([Fraction(1, 3), Fraction(2)], dtype=object)
    result = maskinterp1(yval, np.array([1, 0]))
    assert result.dtype == object
    assert result.tolist() == [Fraction(2), Fraction(2)]


def test_negative_zero_only_good_entry_keeps_the_2022_signs():
    # 1fbab9f returned [0.0, -0.0, -0.0, nan], which is yval * 0 + -0.0.
    result = maskinterp1(np.array([3.0, -0.0, -2.0, np.nan]), np.array([1, 0, 1, 1]))
    assert same(result, np.array([0.0, -0.0, -0.0, -0.0]))


def test_get_chimask_ignores_nan_flux_at_bad_pixels_of_a_one_good_pixel_spectrum():
    # Row 1930 of D1 has one good pixel. 1fbab9f raised ValueError, "array must not
    # contain infs or NaNs", in the Cholesky factorization of the covariance.
    case = GOLDEN.cases["get_chimask.D1.nsigma6"]
    kwargs = GOLDEN.arguments(case)
    kwargs["flux"][1930, kwargs["mask"][1930] != 0] = np.nan
    expected = GOLDEN.expected(case)["chimask"]
    np.testing.assert_array_equal(get_chimask(**kwargs), expected)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_non_finite_ivar_at_bad_pixels_leaves_standard_scale_unchanged(value, dtype):
    # 1fbab9f returned NaN in every value of Dvec and refmean, and in the refscale of the
    # 281 spectra of D1 with a bad pixel.
    kwargs = GOLDEN.arguments(GOLDEN.cases["standard_scale.D1.mask"])
    spec, ivar = kwargs["spec"].astype(dtype), kwargs["ivar"].astype(dtype)
    mask = kwargs["mask"]
    expected = standard_scale(spec, ivar, mask)
    ivar[mask != 0] = value
    for actual, wanted in zip(standard_scale(spec, ivar, mask), expected, strict=True):
        assert same(actual, wanted)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_refscale_of_finite_input_equals_the_2022_expression(dtype):
    # The refscale of 1fbab9f, specUtils.py lines 27 and 34 to 36, on D1 with mask_alt,
    # which also flags pixels whose ivar is not 0.
    kwargs = GOLDEN.arguments(GOLDEN.cases["standard_scale.D1.mask_alt"])
    spec, ivar = kwargs["spec"].astype(dtype), kwargs["ivar"].astype(dtype)
    mask = kwargs["mask"]
    wt = 1 - np.where(mask == 0, 0, 1)
    expected = np.sqrt(np.sum(ivar * wt, axis=1) / np.sum(wt, axis=1))
    assert same(standard_scale(spec, ivar, mask)[1], expected)


@pytest.mark.parametrize("fill", [1.0, 0.5, np.nan])
def test_fully_masked_spectrum_is_rejected(fill):
    # Row 1 of E is bad at every pixel by the 2022 mask rule, which also counts 0.5 and
    # NaN as bad. For each fill, 1fbab9f returned NaN in its refscale and in every value
    # of Dvec and refmean, as the record behaviour.standard_scale.fully_masked_nan holds
    # for a fill of 1.
    kwargs = GOLDEN.arguments(GOLDEN.cases["standard_scale.E.with_fully_masked"])
    kwargs["mask"] = kwargs["mask"].astype(float)
    kwargs["mask"][1] = fill
    with pytest.raises(ValueError, match="^mask: no good pixel in rows 1$"):
        standard_scale(**kwargs)


def test_rejection_names_the_first_10_rows_and_the_count():
    # 1fbab9f returned NaN in every value of Dvec, refscale and refmean.
    rows = r"0, 1, 2, 3, 4, 5, 6, 7, 8, 9, \.\.\. \(12 rows\)$"
    with pytest.raises(ValueError, match=f"^ivar: no good pixel in rows {rows}"):
        standard_scale(np.zeros((12, 3)), np.zeros((12, 3)))


def test_one_dimensional_input_keeps_the_maskinterp_error():
    # 1fbab9f failed the 2-D assert of maskinterp, which runs before the check for
    # spectra without a good pixel.
    with pytest.raises(AssertionError, match="^yval: expected a 2-D array"):
        standard_scale(np.zeros(3), np.zeros(3))


@pytest.mark.parametrize(("maxbadpix", "row3_bad"), [(64, 0), (47, 48)])
@pytest.mark.parametrize("nsigma", [[6], [6, 6]])
def test_spectrum_that_clipping_masks_completely_is_rejected(
    nsigma, maxbadpix, row3_bad
):
    # Row 11 of D1 is good only at pixels 24 to 26, and a 20 sigma spike at pixel 25
    # makes the first pass flag all three. With maxbadpix 47, row 3, given 48 bad
    # pixels, and row 1930 are dropped, so row 11 is row 10 of the kept spectra.
    # 1fbab9f returned a covariance of NaN for nsigma [6], and for [6, 6] raised
    # ValueError, "array must not contain infs or NaNs", in the second pass.
    kwargs = GOLDEN.arguments(GOLDEN.cases["covar_iter_mask.D1.default"])
    flux, ivar, mask = kwargs["flux"], kwargs["ivar"], kwargs["mask"]
    mask[11] = 1
    mask[11, 24:27] = 0
    mask[3, :row3_bad] = 1
    ivar[mask != 0] = 0
    flux[11, 25] += 20 / np.sqrt(ivar[11, 25])
    with pytest.raises(ValueError, match="^nsigma: no good pixel in rows 11$"):
        covar_iter_mask(**kwargs, nsigma=nsigma, maxbadpix=maxbadpix)


@pytest.mark.parametrize(
    ("nsigma", "name"), [([6], "mask"), ([], "mask"), ([6], "ivar")]
)
def test_kept_spectrum_without_a_good_pixel_in_mask_is_rejected(nsigma, name):
    # maxbadpix 64 keeps the fully masked row 5, whose ivar is 0 everywhere. 1fbab9f
    # raised ValueError, "array must not contain infs or NaNs", in the first pass,
    # UnboundLocalError for an empty nsigma (D-32), and AxisError for mask=None (D-14).
    kwargs = GOLDEN.arguments(GOLDEN.cases["covar_iter_mask.D1.default"])
    kwargs["mask"][5] = 1
    kwargs["ivar"][5] = 0
    if name == "ivar":
        kwargs["mask"] = None
    with pytest.raises(ValueError, match=f"^{name}: no good pixel in rows 5$"):
        covar_iter_mask(**kwargs, nsigma=nsigma)


def test_empty_nsigma_without_a_fully_masked_spectrum_still_raises_unbound_local_error():
    # D-32 stays: 1fbab9f raised UnboundLocalError, as no pass binds thismask.
    kwargs = GOLDEN.arguments(GOLDEN.cases["covar_iter_mask.D1.default"])
    with pytest.raises(UnboundLocalError):
        covar_iter_mask(**kwargs, nsigma=[])


def test_mask_none_names_ivar_when_too_few_spectra_are_kept():
    with pytest.raises(ValueError, match="bad pixels in ivar"):
        covar_iter_mask(EYE[:2], EYE[:2], None)


def test_mask_none_marks_the_pixels_where_ivar_is_0():
    # ivar is 0 exactly where the mask of D1 is not 0, and maxbadpix 8 drops 8 spectra.
    # 1fbab9f raised numpy's AxisError (record behaviour.covar_iter_mask.mask_none).
    kwargs = GOLDEN.arguments(GOLDEN.cases["covar_iter_mask.D1.nsig10_5p5.maxbad8"])
    expected = covar_iter_mask(**kwargs)
    actual = covar_iter_mask(**kwargs | {"mask": None})
    for value, wanted in zip(actual, expected, strict=True):
        assert same(value, wanted)
