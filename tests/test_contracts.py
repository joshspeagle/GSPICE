import dataclasses
from functools import partial

import numpy as np
import pytest

from gspice import (
    Model,
    finite_sample_factor,
    fit,
    infill,
    legacy,
    pixelwise,
    scale,
    simulate,
    snr,
    unscale,
)

SPECTRA = {"flux": np.ones((2, 3)), "ivar": np.ones((2, 3)), "mask": np.zeros((2, 3))}
UNSCALE = {"x": np.zeros((2, 3)), "refscale": np.ones(2), "refmean": np.zeros(3)}
MODEL = Model(np.zeros((1, 3)), (np.eye(3),), (10,))
REJECT_EMPTY = {
    "scale": scale,
    "snr": snr,
    "pixelwise": partial(pixelwise, MODEL),
    "infill": partial(infill, MODEL),
}
# fit drops a spectrum without a good pixel, which the others reject.
FLUX_IVAR_MASK = pytest.mark.parametrize(
    "function", [*REJECT_EMPTY.values(), fit], ids=[*REJECT_EMPTY, "fit"]
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
        ("mask", [[0, 0, 0], [0]]),
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


@pytest.mark.parametrize(
    "function", list(REJECT_EMPTY.values()), ids=list(REJECT_EMPTY)
)
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
# The checks that pixelwise and infill share.
MODEL_ARGUMENTS = [
    pytest.param({"model": 5}, TypeError, "model", id="model=5"),
    pytest.param(WIDER, ValueError, "flux", id="npix=4"),
    pytest.param({"progress": 5}, TypeError, "progress", id="progress=5"),
    pytest.param({"inflate": 1}, TypeError, "inflate", id="inflate=1"),
    # Design section 4.3: f_s is derived only for mean-centred, unweighted models.
    *(
        pytest.param(
            {"model": dataclasses.replace(MODEL, **kind), "inflate": True},
            ValueError,
            "inflate",
            id=f"inflate+{name}",
        )
        for name, kind in (
            ("center", {"center": 0.9}),
            ("weighted", {"weighted": True}),
        )
    ),
]


@pytest.mark.parametrize(
    ("changes", "error", "name"),
    [
        *MODEL_ARGUMENTS,
        pytest.param({"nguard": -1}, ValueError, "nguard", id="nguard=-1"),
        pytest.param({"nguard": 2.0}, TypeError, "nguard", id="nguard=2.0"),
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


@pytest.mark.parametrize(
    ("changes", "error", "name"),
    [
        *MODEL_ARGUMENTS,
        # An integer array would read as pixel indices (D-70).
        pytest.param(
            {"replace": np.ones((2, 3), int)}, TypeError, "replace", id="replace=int"
        ),
        pytest.param(
            {"replace": [[True, False, True], [True]]},
            ValueError,
            "replace",
            id="replace=ragged",
        ),
        pytest.param(
            {"replace": np.ones((2, 2), bool)},
            ValueError,
            "replace",
            id="replace=(2, 2)",
        ),
        pytest.param(
            {"replace": np.ones(4, bool)}, ValueError, "replace", id="replace=(4,)"
        ),
        pytest.param(
            {"ignore": np.ones((2, 2), bool)}, ValueError, "ignore", id="ignore=(2, 2)"
        ),
        pytest.param(
            {"ignore": np.ones(4, bool)}, ValueError, "ignore", id="ignore=(4,)"
        ),
        pytest.param({"covariance": 1}, TypeError, "covariance", id="covariance=1"),
        pytest.param({"rng": 0}, TypeError, "rng", id="rng=0"),
        pytest.param(
            {"rng": np.random.RandomState(0)}, TypeError, "rng", id="rng=RandomState"
        ),
    ],
)
def test_an_invalid_argument_of_infill_is_rejected(changes, error, name):
    with pytest.raises(error, match=f"^{name}: "):
        infill(**({"model": MODEL} | SPECTRA | changes))


@pytest.mark.parametrize(
    ("s", "error"),
    [(1.5, TypeError), (True, TypeError), (-1, ValueError), (2, ValueError)],
)
def test_an_invalid_spectrum_of_repaired_covariance_is_rejected(s, error):
    result = infill(MODEL, **SPECTRA, covariance=True)
    with pytest.raises(error, match="^s: "):
        result.covariance(s)


@pytest.mark.parametrize(
    ("changes", "error", "name"),
    [
        pytest.param({"nsigma": 5}, TypeError, "nsigma", id="nsigma=5"),
        pytest.param({"nsigma": ["6"]}, TypeError, "nsigma", id="nsigma=['6']"),
        pytest.param({"nsigma": [[6]]}, ValueError, "nsigma", id="nsigma=[[6]]"),
        pytest.param({"nsigma": [6, -1]}, ValueError, "nsigma", id="nsigma=[6,-1]"),
        pytest.param(
            {"nsigma": [6, np.inf]}, ValueError, "nsigma", id="nsigma=[6,inf]"
        ),
        pytest.param({"maxbadpix": -1}, ValueError, "maxbadpix", id="maxbadpix=-1"),
        pytest.param({"nguard": -1}, ValueError, "nguard", id="nguard=-1"),
        pytest.param({"cumulative": 1}, TypeError, "cumulative", id="cumulative=1"),
        pytest.param({"center": np.nan}, ValueError, "center", id="center=nan"),
        pytest.param({"weights": np.ones(3)}, ValueError, "weights", id="weights=(3,)"),
        pytest.param(
            {"weights": np.array([1.0, -1.0])}, ValueError, "weights", id="weights<0"
        ),
        pytest.param(
            {"weights": np.array([1.0, np.inf])},
            ValueError,
            "weights",
            id="weights=inf",
        ),
        pytest.param(
            {"weights": np.ones(2, complex)}, TypeError, "weights", id="weights=complex"
        ),
        pytest.param({"progress": 5}, TypeError, "progress", id="progress=5"),
    ],
)
def test_an_invalid_argument_of_fit_is_rejected(changes, error, name):
    with pytest.raises(error, match=f"^{name}: "):
        fit(**(SPECTRA | changes))


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


def test_finite_sample_factor_has_its_contract_value():
    # Design section 11.
    value = finite_sample_factor(60000, 2400, 1)
    assert abs(value - 1.0851274682919) <= 1e-12
    assert finite_sample_factor(60000, 2400) == value


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ((30.0, 10), TypeError, "n_train: expected an integer dtype, got float64"),
        ((30, True), TypeError, "n_cond: expected an integer dtype, got bool"),
        ((30, 10, "2"), TypeError, "n_pred: expected an integer dtype, got <U1"),
        ((0, 0), ValueError, r"n_train: expected integers from 1 to 2\^53 - 1, got 0"),
        (
            ([30, 30], [[10, 10], [10, -1]]),
            ValueError,
            r"n_cond: expected integers from 0 to 2\^53 - 1, got -1 at index \(1, 1\)",
        ),
        (
            (30, 10, 0),
            ValueError,
            r"n_pred: expected integers from 1 to 2\^53 - 1, got 0",
        ),
        (
            (2**53, 10),
            ValueError,
            r"n_train: expected integers from 1 to 2\^53 - 1, got 9007199254740992",
        ),
        (
            ([30, 31], [10, 10, 10]),
            ValueError,
            (
                r"n_train, n_cond, n_pred: expected shapes that broadcast together, "
                r"got \(2,\), \(3,\) and \(\)"
            ),
        ),
        (
            ([200, 30], 27),
            ValueError,
            (
                r"n_train, n_cond, n_pred: expected n_train - n_cond - n_pred - 2 > 0, "
                r"got 30 - 27 - 1 - 2 = 0 at index \(1,\)"
            ),
        ),
        # In uint8 arithmetic, 10 - 10 - 1 - 2 wraps around to 253.
        (
            (np.uint8(10), np.uint8(10), np.uint8(1)),
            ValueError,
            r"n_train, n_cond, n_pred: expected .*, got 10 - 10 - 1 - 2 = -3",
        ),
    ],
)
def test_an_invalid_argument_of_finite_sample_factor_is_rejected(
    arguments, error, message
):
    with pytest.raises(error, match=f"^{message}$"):
        finite_sample_factor(*arguments)
