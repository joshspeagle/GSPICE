import importlib
import pickle
import subprocess
import sys

import numpy as np
import pytest

import gspice
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
