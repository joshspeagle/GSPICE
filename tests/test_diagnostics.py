from fractions import Fraction
from functools import partial

import numpy as np
import pytest
from scipy import stats

from _golden import checked
from gspice import (
    _spectra,
    finite_sample_factor,
    fit,
    infill,
    noise_audit,
    pixelwise,
    simulate,
)

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


NPIX, NSPEC = 12, 7


def audited():
    # A model of 40 simulated spectra, and 7 others in float32 with bad pixels: pixel 3
    # in every spectrum, by mask in rows 0 to 3 and by ivar 0 in rows 4 to 6, pixel 8
    # in every spectrum but row 2, and runs at both edges and inside. Flux is NaN at
    # the bad pixels and ivar inf at the masked ones, so that a read of either shows.
    rng = np.random.default_rng(13)
    train = simulate(40, NPIX, rng=rng)
    model, _ = fit(train.flux, train.ivar, nsigma=())
    test = simulate(NSPEC, NPIX, rng=rng)
    flux, ivar = test.flux.astype(np.float32), test.ivar.astype(np.float32)
    mask = np.zeros(flux.shape, bool)
    mask[:4, 3] = True
    ivar[4:, 3] = 0.0
    mask[[0, 1, 3, 4, 5, 6], 8] = True
    mask[1, 5:7] = True
    mask[6, :2] = True
    ivar[5, 10:] = 0.0
    flux[mask | (ivar == 0)] = np.nan
    ivar[mask] = np.inf
    return model, flux, ivar, mask


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("rows", [1, 2, None], ids=["1-row", "2-row", "one-chunk"])
def test_rho_is_the_rms_residual_in_reported_sigmas_over_the_good_spectra(
    monkeypatch, rows
):
    if rows is not None:
        monkeypatch.setattr(_spectra, "CHUNK", rows * NPIX)
    model, flux, ivar, mask = audited()
    rho = checked(partial(noise_audit, model, nguard=2), flux, ivar, mask)
    # With the same chunks, pixelwise computes the predictions of noise_audit from the
    # same arrays, bit for bit.
    predicted = pixelwise(model, flux, ivar, mask, nguard=2).flux
    good = ~(mask | (ivar == 0))
    np.testing.assert_array_equal(np.isnan(rho), ~good.any(axis=0))
    for i in np.flatnonzero(good.any(axis=0)):
        s = good[:, i]
        # In float64, from the float32 inputs, as noise_audit computes.
        terms = (flux[s, i] - predicted[s, i]) ** 2 * ivar[s, i]
        # Both sides compute the terms alike. Each sum of the n <= 7 positive terms,
        # in any order, lies within gamma_6 = 6u / (1 - 6u) of the exact sum,
        # relative, with u = eps / 2; the division and the root round once each, and
        # the root halves a relative error. So |drho| <= (gamma_6 + 3u) rho < 5 eps rho.
        assert abs(rho[i] - np.sqrt(np.mean(terms))) <= 5 * EPS * rho[i]


def test_progress_of_noise_audit_counts_the_spectra_done_after_each_chunk(
    monkeypatch,
):
    monkeypatch.setattr(_spectra, "CHUNK", 2 * NPIX)
    model, flux, ivar, mask = audited()
    calls = []
    noise_audit(model, flux, ivar, mask, progress=lambda *args: calls.append(args))
    assert calls == [("noise_audit", done, NSPEC) for done in (2, 4, 6, 7)]


@pytest.mark.slow
def test_rho_squared_minus_e_is_the_ratio_of_true_to_reported_noise_variance():
    # Design section 11. The reported sigma is 1/1.2 of the true sigma at pixels 100 to
    # 299 of the training and the held-out spectra, so k_i = 1.44 there and 1
    # elsewhere; the model absorbs it into its covariance.
    npix, n_test = 400, 4000
    k = np.ones(npix)
    k[100:300] = 1.2**2
    rng = np.random.default_rng(14)
    train, test = simulate(2000, npix, rng=rng), simulate(n_test, npix, rng=rng)
    model, _ = fit(train.flux, k * train.ivar, nsigma=())
    ivar = k * test.ivar
    rho = noise_audit(model, test.flux, ivar)
    predicted = pixelwise(model, test.flux, ivar).flux
    e = np.mean(ivar * (test.truth - predicted) ** 2, axis=0)
    # Without bad pixels, rho_i^2 - e_i is the mean over the spectra s of
    # d_si = ivar_si ((f_si - fhat_si)^2 - (t_si - fhat_si)^2), and its mean over a set
    # of pixels is the mean over s of D_s, the mean of d_si over the set. The noise is
    # independent between pixels and of ivar, and the model has one S/N bin, so fhat_si
    # does not depend on the noise of f_si, and E[d_si] = k_i. Given the model, the D_s of the independent spectra
    # are independent and identically distributed, so their mean has the standard
    # error std(D_s) / sqrt(n_test), and it is normal to good accuracy at n_test 4000.
    # The mean of each set then lies beyond 3 standard errors with the probability
    # 0.27%, and the test fails falsely with a probability of at most 0.54%
    # (Bonferroni).
    d = ivar * ((test.flux - predicted) ** 2 - (test.truth - predicted) ** 2)
    for pixels in (k > 1, k == 1):
        deviation = np.mean(rho[pixels] ** 2 - e[pixels] - k[pixels])
        assert abs(deviation) <= 3 * stats.sem(d[:, pixels].mean(axis=1))
