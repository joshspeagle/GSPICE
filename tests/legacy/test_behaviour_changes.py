import importlib
import pickle

import pytest

import gspice

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
