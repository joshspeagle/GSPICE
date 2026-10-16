import ast
import importlib
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import gspice
from _golden import Golden
from gspice.legacy import (
    covar,
    covar_iter_mask,
    gp_interp,
    maskinterp,
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
