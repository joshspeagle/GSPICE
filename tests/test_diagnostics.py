from fractions import Fraction

import numpy as np
import pytest
from scipy import stats

from gspice import finite_sample_factor, fit, infill, pixelwise, simulate

EPS = np.finfo(float).eps
# Each held-out test averages over SETS training sets, each with HELD_OUT spectra of
# its own.
SETS, HELD_OUT = 2000, 100


@pytest.mark.parametrize(("n_train", "n_cond", "n_pred"), [(4, 0, 1), (30, 10, 2)])
def test_finite_sample_factor_is_its_formula_to_rounding(n_train, n_cond, n_pred):
    N = n_train
    exact = (
        Fraction(N + 1, N)
        * Fraction(N - 2, N - n_cond - 2)
        * Fraction(N - 1, N - n_cond - n_pred - 2)
    )
    # Six operations round, each by at most eps/2: 3 eps to first order.
    value = finite_sample_factor(n_train, n_cond, n_pred)
    assert abs(Fraction(value) - exact) <= 4 * EPS * exact


def test_finite_sample_factor_broadcasts_integers_of_any_dtype():
    n_train = np.array([[30], [200]], np.uint8)
    n_cond = [10, 20, 0]
    result = finite_sample_factor(n_train, n_cond, np.int16(2))
    expected = [[finite_sample_factor(N, k, 2) for k in n_cond] for N in (30, 200)]
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, expected)
    assert type(finite_sample_factor(np.array(30), np.uint64(10))) is float
    assert finite_sample_factor([30], 10).shape == (1,)


def assert_mean_1(n_train, npix, statistic):
    # The mean of statistic(model, flux, ivar) over the training sets is 1 within 3
    # standard errors across the sets (design section 11). Each simulation gives the
    # n_train spectra of a fit without clipping and the HELD_OUT spectra that the fit
    # predicts. The spectra of simulate(gaussian=True) without artifacts are
    # independent, have no bad pixels and are exactly Gaussian once scaled.
    rng = np.random.default_rng(12)
    values = np.empty(SETS)
    for j in range(SETS):
        spectra = simulate(n_train + HELD_OUT, npix, rng=rng, gaussian=True)
        model, _ = fit(spectra.flux[:n_train], spectra.ivar[:n_train], nsigma=())
        values[j] = statistic(model, spectra.flux[n_train:], spectra.ivar[n_train:])
    assert abs(np.mean(values) - 1) <= 3 * stats.sem(values)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("n_train", "npix", "replace", "ignore"),
    [
        # 8 ignored pixels around the replaced ones.
        (30, 20, np.r_[9:11], np.r_[5:9, 11:15]),
        (200, 50, np.r_[20:30], []),
    ],
    # (N_s, N_k, N_*), with N_k = npix - |M| - |I|.
    ids=["30-10-2", "200-40-10"],
)
def test_the_held_out_calibration_statistic_of_infill_has_the_mean_1(
    n_train, npix, replace, ignore
):
    replace, ignore = (np.isin(np.arange(npix), pixels) for pixels in (replace, ignore))

    def statistic(model, flux, ivar):
        result = infill(model, flux, ivar, replace=replace, ignore=ignore)
        return np.mean(result.chi2 / (result.dof * result.fs))

    assert_mean_1(n_train, npix, statistic)


@pytest.mark.slow
def test_the_held_out_z2_of_pixelwise_with_inflate_has_the_mean_1():
    # (N_s, N_k, N_*) = (200, 40, 1) where the guard window of 21 pixels is not
    # clipped, and N_k up to 50 at the ends of the spectrum.
    def statistic(model, flux, ivar):
        return np.mean(pixelwise(model, flux, ivar, nguard=10, inflate=True).z ** 2)

    assert_mean_1(200, 61, statistic)
