from functools import partial

import numpy as np
import pytest

from gspice import Model, legacy, pixelwise, scale, simulate, snr, unscale

SPECTRA = {"flux": np.ones((2, 3)), "ivar": np.ones((2, 3)), "mask": np.zeros((2, 3))}
UNSCALE = {"x": np.zeros((2, 3)), "refscale": np.ones(2), "refmean": np.zeros(3)}
MODEL = Model(np.zeros((1, 3)), (np.eye(3),), (10,))
FLUX_IVAR_MASK = pytest.mark.parametrize(
    "function",
    [scale, snr, partial(pixelwise, MODEL)],
    ids=["scale", "snr", "pixelwise"],
)
# Finite only where long double is wider than float64, as on Linux x86-64; where long
# double is float64, it parses to inf.
BEYOND = np.longdouble("1e400")
WIDE = np.isfinite(BEYOND)


@FLUX_IVAR_MASK
@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("flux", np.ones(3)),
        ("flux", np.ones((2, 3, 1))),
        ("flux", np.ones((2, 0))),
        ("flux", [[1.0, 1.0, 1.0], [1.0]]),
        ("ivar", np.ones((2, 2))),
        ("mask", np.zeros((3, 3))),
    ],
)
def test_a_wrong_shape_is_rejected(function, name, value):
    with pytest.raises(ValueError, match=f"^{name}: "):
        function(**(SPECTRA | {name: value}))


@FLUX_IVAR_MASK
@pytest.mark.parametrize(
    ("name", "dtype"),
    [
        ("flux", bool),
        ("flux", complex),
        ("ivar", bool),
        ("ivar", object),
        ("mask", complex),
        ("mask", str),
    ],
)
def test_a_wrong_dtype_is_rejected(function, name, dtype):
    with pytest.raises(TypeError, match=f"^{name}: "):
        function(**(SPECTRA | {name: SPECTRA[name].astype(dtype)}))


@FLUX_IVAR_MASK
@pytest.mark.parametrize("mask", [SPECTRA["mask"], None])
@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("flux", np.nan),
        ("flux", np.inf),
        ("flux", BEYOND),
        ("ivar", np.nan),
        ("ivar", np.inf),
        ("ivar", -1.0),
        ("ivar", BEYOND),
    ],
)
@pytest.mark.filterwarnings("error")
def test_a_non_finite_or_negative_value_at_a_good_pixel_is_rejected(
    function, mask, name, value
):
    arrays = {"flux": np.ones((2, 3)), "ivar": np.ones((2, 3))}
    arrays[name] = np.ones((2, 3), np.asarray(value).dtype)
    arrays[name][1, 2] = value
    with pytest.raises(ValueError, match=f"^{name}: "):
        function(**arrays, mask=mask)


@FLUX_IVAR_MASK
def test_spectra_without_a_good_pixel_are_counted_and_the_first_10_named(function):
    # 12 rows without a good pixel: 7 by mask, 4 by ivar 0 and row 12 by both.
    mask = np.zeros((15, 3), bool)
    ivar = np.ones((15, 3))
    mask[[0, 2, 5, 7, 8, 11, 13]] = True
    ivar[[1, 4, 10, 14]] = 0.0
    mask[12, :2] = True
    ivar[12, 2] = 0.0
    rows = r"0, 1, 2, 4, 5, 7, 8, 10, 11, 12, \.\.\. \(12 rows\)$"
    with pytest.raises(ValueError, match=f"^mask, ivar: .* rows {rows}"):
        function(np.ones((15, 3)), ivar, mask)


# Four pixels, one more than MODEL has.
WIDER = {"flux": np.ones((2, 4)), "ivar": np.ones((2, 4)), "mask": np.zeros((2, 4))}


@pytest.mark.parametrize(
    ("changes", "error", "name"),
    [
        pytest.param({"model": 5}, TypeError, "model", id="model=5"),
        pytest.param(WIDER, ValueError, "flux", id="npix=4"),
        pytest.param({"nguard": -1}, ValueError, "nguard", id="nguard=-1"),
        pytest.param({"nguard": 2.0}, TypeError, "nguard", id="nguard=2.0"),
        pytest.param({"progress": 5}, TypeError, "progress", id="progress=5"),
    ],
)
def test_an_invalid_argument_of_pixelwise_is_rejected(changes, error, name):
    with pytest.raises(error, match=f"^{name}: "):
        pixelwise(**({"model": MODEL} | SPECTRA | changes))


@pytest.mark.parametrize(
    ("changes", "error", "name"),
    [
        pytest.param({"nspec": -1}, ValueError, "nspec", id="nspec=-1"),
        pytest.param({"nspec": 1.5}, TypeError, "nspec", id="nspec=1.5"),
        pytest.param({"npix": 0}, ValueError, "npix", id="npix=0"),
        pytest.param({"rng": None}, TypeError, "rng", id="rng=None"),
        pytest.param({"rng": 0}, TypeError, "rng", id="rng=0"),
        pytest.param({"artifacts": -0.1}, ValueError, "artifacts", id="artifacts=-0.1"),
        pytest.param({"artifacts": 1.5}, ValueError, "artifacts", id="artifacts=1.5"),
        pytest.param(
            {"artifacts": np.nan}, ValueError, "artifacts", id="artifacts=nan"
        ),
        pytest.param({"gaussian": 1}, TypeError, "gaussian", id="gaussian=1"),
    ],
)
def test_an_invalid_argument_of_simulate_is_rejected(changes, error, name):
    arguments = {"nspec": 2, "npix": 3, "rng": np.random.default_rng(0)}
    with pytest.raises(error, match=f"^{name}: "):
        simulate(**(arguments | changes))


CENTER = {
    "scale": lambda center: scale(SPECTRA["flux"], SPECTRA["ivar"], center=center),
    "unscale": lambda center: unscale(**UNSCALE, center=center),
}


@pytest.mark.parametrize("call", CENTER.values(), ids=list(CENTER))
@pytest.mark.parametrize(
    ("center", "error"),
    [
        ("1", TypeError),
        (True, TypeError),
        (np.array(1.0), TypeError),
        (np.nan, ValueError),
        (np.inf, ValueError),
        (10**400, ValueError),
    ],
)
def test_a_center_that_is_not_a_finite_real_number_is_rejected(call, center, error):
    with pytest.raises(error, match="^center: "):
        call(center)


@pytest.mark.parametrize(
    ("name", "value", "error"),
    [
        ("x", np.zeros(3), ValueError),
        ("x", np.zeros((2, 3), complex), TypeError),
        ("refscale", np.ones(3), ValueError),
        ("refscale", np.array([1.0, 0.0]), ValueError),
        ("refscale", np.array([1.0, -1.0]), ValueError),
        ("refscale", np.array([np.nan, 1.0]), ValueError),
        ("refscale", np.ones(2, complex), TypeError),
        ("refmean", np.zeros(2), ValueError),
        ("refmean", np.array([0.0, np.nan, 0.0]), ValueError),
        pytest.param(
            "refmean",
            np.array([0.0, BEYOND, 0.0]),
            ValueError,
            marks=pytest.mark.filterwarnings("error"),
        ),
        ("refmean", np.zeros(3, complex), TypeError),
        ("var", np.ones(2), ValueError),
        ("var", -np.ones(3), ValueError),
        ("var", np.full((2, 3), np.nan), ValueError),
        ("var", np.ones(3, complex), TypeError),
    ],
)
def test_an_invalid_argument_of_unscale_is_rejected(name, value, error):
    with pytest.raises(error, match=f"^{name}: "):
        unscale(**(UNSCALE | {name: value}))


@pytest.mark.parametrize("refmean", [np.zeros(2), np.array([0.0, np.nan, 0.0])])
def test_an_invalid_refmean_of_scale_is_rejected(refmean):
    with pytest.raises(ValueError, match="^refmean: "):
        scale(SPECTRA["flux"], SPECTRA["ivar"], refmean=refmean)


@pytest.mark.parametrize(
    ("name", "value", "old"),
    [
        ("flux", np.nan, lambda Dvec, refscale, refmean: np.isnan(Dvec[:, 1]).all()),
        ("ivar", np.nan, lambda Dvec, refscale, refmean: np.isnan(refscale).all()),
        # The mean ivar of each row is (1 - 0.5 + 1) / 3.
        ("ivar", -0.5, lambda Dvec, refscale, refmean: np.all(refscale == 0.5**0.5)),
        # 2022 computed in long double, where the value is finite.
        pytest.param(
            "flux",
            BEYOND,
            lambda Dvec, refscale, refmean: np.isfinite(Dvec).all(),
            marks=pytest.mark.skipif(not WIDE, reason="long double is float64"),
        ),
    ],
)
def test_scale_rejects_a_value_for_which_2022_returned_a_result(name, value, old):
    arrays = {"flux": np.ones((2, 3)), "ivar": np.ones((2, 3))}
    arrays[name] = np.ones((2, 3), np.asarray(value).dtype)
    arrays[name][:, 1] = value
    assert old(*legacy.standard_scale(arrays["flux"], arrays["ivar"]))
    with pytest.raises(ValueError, match=f"^{name}: "):
        scale(**arrays)


def test_scale_of_zero_spectra_raises_where_2022_returned_a_nan_mean():
    empty = np.zeros((0, 3))
    with pytest.warns(RuntimeWarning):
        assert np.isnan(legacy.standard_scale(empty, empty)[2]).all()
    with pytest.raises(ValueError, match="^flux: "):
        scale(empty, empty)
