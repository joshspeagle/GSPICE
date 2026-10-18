import dataclasses
import subprocess
import sys
import tracemalloc
from functools import partial

import numpy as np
import pytest

from gspice import (
    Blocks,
    Calibration,
    LowRank,
    Model,
    _model,
    _predict,
    _spectra,
    calibration,
    finite_sample_factor,
    fit,
    fit_blocks,
    infill,
    legacy,
    noise_audit,
    pixelwise,
    repair,
    scale,
    simulate,
    snr,
    unscale,
)
from gspice._linalg import precision

SPECTRA = {"flux": np.ones((2, 3)), "ivar": np.ones((2, 3)), "mask": np.zeros((2, 3))}
UNSCALE = {"x": np.zeros((2, 3)), "refscale": np.ones(2), "refmean": np.zeros(3)}
MODEL = Model(np.zeros((1, 3)), (np.eye(3),), (10,))
REJECT_EMPTY = {
    "scale": scale,
    "snr": snr,
    "pixelwise": partial(pixelwise, MODEL),
    "infill": partial(infill, MODEL),
    "fit_blocks": partial(fit_blocks, MODEL),
    "repair": partial(repair, MODEL),
    "calibration": partial(calibration, MODEL, rng=np.random.default_rng(0)),
    "noise_audit": partial(noise_audit, MODEL),
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
# The checks of the functions that take a model and spectra.
MODEL_ARGUMENTS = [
    pytest.param({"model": 5}, TypeError, "model", id="model=5"),
    pytest.param(WIDER, ValueError, "flux", id="npix=4"),
    pytest.param({"progress": 5}, TypeError, "progress", id="progress=5"),
]
# The checks of inflate.
INFLATE_ARGUMENTS = [
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
            ("low-rank", {"cov": (LowRank(np.eye(3, 1), [1.0], np.ones(3)),)}),
        )
    ),
]
# The checks of nguard.
NGUARD_ARGUMENTS = [
    pytest.param({"nguard": -1}, ValueError, "nguard", id="nguard=-1"),
    pytest.param({"nguard": 2.0}, TypeError, "nguard", id="nguard=2.0"),
]


@pytest.mark.parametrize(
    ("changes", "error", "name"),
    [*MODEL_ARGUMENTS, *INFLATE_ARGUMENTS, *NGUARD_ARGUMENTS],
)
def test_an_invalid_argument_of_pixelwise_is_rejected(changes, error, name):
    with pytest.raises(error, match=f"^{name}: "):
        pixelwise(**({"model": MODEL} | SPECTRA | changes))


@pytest.mark.parametrize(
    ("changes", "error", "name"), [*MODEL_ARGUMENTS, *NGUARD_ARGUMENTS]
)
def test_an_invalid_argument_of_noise_audit_is_rejected(changes, error, name):
    arguments = {"model": MODEL} | SPECTRA | changes
    # Zero spectra, so that no chunk runs.
    for key in ("flux", "ivar", "mask"):
        arguments[key] = arguments[key][:0]
    with pytest.raises(error, match=f"^{name}: "):
        noise_audit(**arguments)


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
        *INFLATE_ARGUMENTS,
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


# Every row has one good pixel, so that no default size of calibration is at most half
# of the median good-pixel count.
ONE_GOOD = np.array([[True, True, False]] * 2)


@pytest.mark.parametrize(
    ("changes", "error", "name"),
    [
        *MODEL_ARGUMENTS,
        pytest.param({"sizes": 4}, TypeError, "sizes", id="sizes=4"),
        pytest.param({"sizes": []}, ValueError, "sizes", id="sizes=[]"),
        pytest.param({"sizes": [1, 0]}, ValueError, r"sizes\[1\]", id="sizes=[1,0]"),
        pytest.param({"sizes": [2.0]}, TypeError, r"sizes\[0\]", id="sizes=[2.0]"),
        pytest.param({"sizes": [True]}, TypeError, r"sizes\[0\]", id="sizes=[True]"),
        pytest.param({"mask": ONE_GOOD}, ValueError, "sizes", id="one-good-pixel"),
        pytest.param({"layout": "block"}, ValueError, "layout", id="layout=block"),
        pytest.param({"layout": None}, ValueError, "layout", id="layout=None"),
        pytest.param(
            {"layout": np.array("run")}, ValueError, "layout", id="layout=array"
        ),
        pytest.param({"rng": None}, TypeError, "rng", id="rng=None"),
        pytest.param(
            {"rng": np.random.RandomState(0)}, TypeError, "rng", id="rng=RandomState"
        ),
    ],
)
def test_an_invalid_argument_of_calibration_is_rejected(changes, error, name):
    arguments = {"model": MODEL, **SPECTRA, "rng": np.random.default_rng(0)}
    with pytest.raises(error, match=f"^{name}: "):
        calibration(**(arguments | changes))


# Three spectra and two columns.
CALIBRATION = Calibration(np.ones((3, 2)), 1, 1)


@pytest.mark.parametrize(
    ("method", "arguments", "error", "name"),
    [
        ("ratio", {"where": [1, 0, 1]}, TypeError, "where"),
        ("ratio", {"where": np.ones(2, bool)}, ValueError, "where"),
        ("tail", {"where": np.ones((3, 1), bool)}, ValueError, "where"),
        ("tail", {"q": 0}, ValueError, "q"),
        ("tail", {"q": 1.0}, ValueError, "q"),
        ("tail", {"q": "0.9"}, TypeError, "q"),
        ("qq", {"i": 1.0}, TypeError, "i"),
        ("qq", {"i": np.True_}, TypeError, "i"),
        ("qq", {"i": -1}, ValueError, "i"),
        ("qq", {"i": 2}, ValueError, "i"),
        ("qq", {"where": [True] * 4}, ValueError, "where"),
    ],
)
def test_an_invalid_argument_of_a_calibration_method_is_rejected(
    method, arguments, error, name
):
    with pytest.raises(error, match=f"^{name}: "):
        getattr(CALIBRATION, method)(**arguments)


@pytest.mark.parametrize(
    ("changes", "error", "name"),
    [
        *MODEL_ARGUMENTS,
        *NGUARD_ARGUMENTS,
        pytest.param({"width": 0}, ValueError, "width", id="width=0"),
        pytest.param({"width": 4}, ValueError, "width", id="width=4"),
        pytest.param({"width": 2.0}, TypeError, "width", id="width=2.0"),
        pytest.param({"stride": 0}, ValueError, "stride", id="stride=0"),
        pytest.param({"stride": True}, TypeError, "stride", id="stride=True"),
        pytest.param({"nguard": 2**63}, ValueError, "nguard", id="nguard=2^63"),
        pytest.param({"percentile": 0}, ValueError, "percentile", id="percentile=0"),
        pytest.param(
            {"percentile": 100}, ValueError, "percentile", id="percentile=100"
        ),
        pytest.param(
            {"percentile": np.nan}, ValueError, "percentile", id="percentile=nan"
        ),
        pytest.param(
            {"percentile": "50"}, TypeError, "percentile", id="percentile='50'"
        ),
    ],
)
def test_an_invalid_argument_of_fit_blocks_is_rejected(changes, error, name):
    # Valid otherwise: two spectra suffice at the percentile 50.
    arguments = {"model": MODEL} | SPECTRA | {"width": 2, "percentile": 50}
    with pytest.raises(error, match=f"^{name}: "):
        fit_blocks(**(arguments | changes))


# MODEL with one block of all pixels.
BLOCKED = dataclasses.replace(MODEL, blocks=Blocks([0], 3, 0, [[1.0]]))


@pytest.mark.parametrize(
    ("changes", "error", "name"),
    [
        *MODEL_ARGUMENTS,
        *INFLATE_ARGUMENTS,
        *NGUARD_ARGUMENTS,
        *(
            pytest.param({name: value}, error, name, id=f"{name}={value!r}")
            for name in ("nsigma_replace", "nsigma_ignore")
            for value, error in ((0.0, ValueError), ("5", TypeError))
        ),
        pytest.param(
            {"nsigma_ignore": 10.5}, ValueError, "nsigma_ignore", id="ignore>replace"
        ),
        pytest.param({"dilate": -1}, ValueError, "dilate", id="dilate=-1"),
        pytest.param({"dilate": 1.0}, TypeError, "dilate", id="dilate=1.0"),
        pytest.param({"blockwise": 1}, TypeError, "blockwise", id="blockwise=1"),
        # Design decision Q6: the default needs the thresholds of fit_blocks.
        pytest.param(
            {"model": MODEL}, ValueError, "blockwise", id="blockwise without blocks"
        ),
        pytest.param({"runlength": 0}, ValueError, "runlength", id="runlength=0"),
        pytest.param({"runlength": 5.0}, TypeError, "runlength", id="runlength=5.0"),
        pytest.param({"halfwidth": -1}, ValueError, "halfwidth", id="halfwidth=-1"),
        pytest.param({"minfrac": -0.1}, ValueError, "minfrac", id="minfrac=-0.1"),
        pytest.param({"minfrac": 1.5}, ValueError, "minfrac", id="minfrac=1.5"),
        pytest.param({"minfrac": np.nan}, ValueError, "minfrac", id="minfrac=nan"),
        pytest.param({"minfrac": "0.5"}, TypeError, "minfrac", id="minfrac='0.5'"),
        # A value that compares equal to "all" but is not a str.
        *(
            pytest.param(
                {"count_on": value}, ValueError, "count_on", id=f"count_on={i}"
            )
            for i, value in enumerate(["any", None, np.array("all")])
        ),
        pytest.param({"covariance": 1}, TypeError, "covariance", id="covariance=1"),
        pytest.param({"rng": 0}, TypeError, "rng", id="rng=0"),
    ],
)
def test_an_invalid_argument_of_repair_is_rejected(changes, error, name):
    with pytest.raises(error, match=f"^{name}: "):
        repair(**({"model": BLOCKED} | SPECTRA | changes))


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
        pytest.param({"rank": 0}, ValueError, "rank", id="rank=0"),
        # rank < npix, the 3 pixels of SPECTRA.
        pytest.param({"rank": 3}, ValueError, "rank", id="rank=3"),
        pytest.param({"rank": 1.0}, TypeError, "rank", id="rank=1.0"),
        pytest.param({"rank": True}, TypeError, "rank", id="rank=True"),
        *(
            pytest.param({"snr_bins": value}, error, name, id=f"snr_bins={value!r}")
            for value, error, name in (
                (0, ValueError, "snr_bins"),
                (True, TypeError, "snr_bins"),
                (2.0, TypeError, "snr_bins"),
                ((), ValueError, "snr_bins"),
                ((2.0, 1.0), ValueError, "snr_bins"),
                ((1.0, np.nan), ValueError, r"snr_bins\[1\]"),
                ("12", TypeError, r"snr_bins\[0\]"),
            )
        ),
        pytest.param({"noise": 1}, TypeError, "noise", id="noise=1"),
        *(
            pytest.param({"noise": True, key: value}, ValueError, "noise", id=key)
            for key, value in (("center", 0.9), ("snr_bins", 1))
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


@pytest.mark.parametrize(
    "function",
    [
        partial(pixelwise, nguard=1),
        infill,
        partial(fit_blocks, width=2, percentile=50),
        partial(repair, nguard=1, blockwise=False),
        partial(calibration, sizes=(1,), rng=np.random.default_rng(0)),
        partial(noise_audit, nguard=1),
    ],
    ids=["pixelwise", "infill", "fit_blocks", "repair", "calibration", "noise_audit"],
)
def test_a_call_validates_the_spectra_and_finds_their_bins_once(monkeypatch, function):
    # Chunks of one spectrum, so that a check or a binning per chunk shows.
    monkeypatch.setattr(_spectra, "CHUNK", 3)
    calls = []

    def counted(name, original):
        def wrapper(*args, **kwargs):
            calls.append(name)
            return original(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(_predict, "validate", counted("validate", _predict.validate))
    monkeypatch.setattr(Model, "_bins", counted("bins", Model._bins))
    function(MODEL, np.ones((4, 3)), np.ones((4, 3)))
    assert calls == ["validate", "bins"]


# A noise model whose C_s = 0.5 + diag(nu_s) is positive definite, with one block.
NOISE = Model(
    np.zeros((1, 3)),
    (np.full((3, 3), 0.5),),
    (10,),
    noise=True,
    blocks=Blocks([0], 1, 0, np.ones((1, 1))),
)


@pytest.mark.parametrize(
    ("function", "expected"),
    [
        (partial(pixelwise, nguard=1), [0, 1, 2, 3]),
        # The spectra with a pixel to replace, the bad pixel of row 1.
        (infill, [1]),
        (partial(fit_blocks, width=1, nguard=0, percentile=50), [0, 1, 2, 3]),
        (partial(repair, nguard=1), [0, 1, 2, 3]),
        (partial(calibration, sizes=(2, 3), rng=np.random.default_rng(0)), [0, 2, 3]),
        (partial(noise_audit, nguard=1), [0, 1, 2, 3]),
        # The prior needs no precision matrix.
        (partial(pixelwise, nguard=2), []),
    ],
    ids=[
        "pixelwise",
        "infill",
        "fit_blocks",
        "repair",
        "calibration",
        "noise_audit",
        "prior",
    ],
)
def test_a_noise_model_forms_the_precision_matrix_of_each_spectrum_once_per_call(
    monkeypatch, function, expected
):
    # Design section 4.2: P_s once per spectrum and call, for every operation of the
    # call, and never P of the noise-free covariance. Chunks of two spectra. Row 1
    # has a bad pixel, so that its two good pixels leave calibration none to
    # condition on at the sizes 2 and 3; C_s differs between spectra, as ivar does.
    monkeypatch.setattr(_spectra, "CHUNK", 6)
    calls = []

    def counted(S):
        calls.append(np.diagonal(S) - 0.5)
        return precision(S)

    monkeypatch.setattr(_model, "precision", counted)
    ivar = np.array([[1.0, 2.0, 4.0], [1.0, 0.0, 2.0], [4.0] * 3, [0.5] * 3])
    flux = np.random.default_rng(3).standard_normal(ivar.shape)
    function(NOISE, flux, ivar)
    # The diagonal of C_s is 0.5 + nu_s, with nu = 1/ivar at good pixels and 1/a_s^2
    # at the bad pixel, a_s^2 = 1.5 the mean ivar of its good pixels.
    nu = 1 / np.where(ivar == 0, 1.5, ivar)
    assert len(calls) == len(expected)
    for diagonal, s in zip(calls, expected, strict=True):
        np.testing.assert_allclose(diagonal, nu[s], rtol=2 * np.finfo(float).eps)
    assert NOISE._cache == {}


def test_infill_of_a_noise_model_forms_no_c_s_for_a_spectrum_without_a_replacement(
    monkeypatch,
):
    # It reads no covariance there, and C_s costs npix^2 (10.2 ms per spectrum
    # measured at npix 2400, against 0.059 ms for a model without noise).
    calls = []

    def counted(self, nu, label):
        calls.append(label)
        return original(self, nu, label)

    original = Model._with_noise
    monkeypatch.setattr(Model, "_with_noise", counted)
    ivar = np.ones((4, 3))
    ivar[[1, 3], 1] = 0.0
    infill(NOISE, np.zeros(ivar.shape), ivar)
    assert calls == [f"model: cov[0] + diag(nu) of row {s}" for s in (1, 3)]


@pytest.mark.parametrize(
    "function",
    [
        partial(pixelwise, nguard=1),
        infill,
        partial(fit_blocks, width=1, nguard=0, percentile=50),
        partial(calibration, sizes=(1,), rng=np.random.default_rng(0)),
        partial(noise_audit, nguard=1),
    ],
    ids=["pixelwise", "infill", "fit_blocks", "calibration", "noise_audit"],
)
def test_a_noise_model_reports_progress_after_each_spectrum(function):
    # A noise model works in chunks of one spectrum, whose P_s takes more than a
    # chunk of another model; repair is in its own tests.
    calls = []
    flux = np.random.default_rng(3).standard_normal((4, 3))
    function(
        NOISE, flux, np.ones(flux.shape), progress=lambda *args: calls.append(args)
    )
    assert [args[1:] for args in calls] == [(s, 4) for s in range(1, 5)]


@pytest.mark.parametrize(
    "function", [partial(pixelwise, nguard=5), infill], ids=["pixelwise", "infill"]
)
def test_a_noise_model_holds_the_matrices_of_one_spectrum_at_a_time(function):
    # C_s, P_s and the guard weights of a spectrum are freed before the next C_s is
    # formed: the peak measured 3.3 npix^2 x 8 bytes here in both, and 6.3 in
    # pixelwise and 5.3 in infill with the previous spectrum alive.
    npix = 200
    rng = np.random.default_rng(0)
    B = rng.standard_normal((npix, 20))
    model = Model(np.ones((1, npix)), (B @ B.T / 20,), (10,), noise=True)
    ivar = rng.uniform(1, 4, (3, npix))
    flux = 1 + rng.standard_normal((3, npix))
    mask = np.zeros(flux.shape, bool)
    mask[:, 10:20] = True
    tracemalloc.start()
    try:
        function(model, flux, ivar, mask)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 4.5 * npix**2 * 8


def test_importing_gspice_loads_neither_scipy_stats_nor_scipy_interpolate():
    code = (
        "import sys, gspice; "
        "assert not {'scipy.stats', 'scipy.interpolate'} & set(sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
