import warnings

import numpy as np
import pytest

from gspice import (standard_scale, covar, get_chimask, covar_iter_mask, gaussian_estimate,
                    gp_interp, gp_weights, cholesky_inv)
from gspice.synthetic import make_synthetic_spectra, inject_outliers
from helpers import direct_conditional


# ---------------------------------------------------------------- scaling
def test_standard_scale_definition(data):
    flux, ivar, _ = data
    mask = np.zeros(flux.shape, dtype = int)
    mask[3, 10:14] = 1
    mask[7, 0] = 1
    X, rs, rm = standard_scale(flux, ivar, mask)
    good = mask == 0
    expected_rs = np.sqrt((ivar * good).sum(1) / good.sum(1))
    np.testing.assert_allclose(rs, expected_rs, rtol = 1e-14)
    np.testing.assert_allclose(X.mean(axis = 0), 0., atol = 1e-12)
    # masked pixels are linearly interpolated before scaling
    expected = flux[3, 9] + (flux[3, 14] - flux[3, 9]) * (np.arange(10, 14) - 9) / 5.
    np.testing.assert_allclose(X[3, 10:14] + rm[10:14], rs[3] * expected, rtol = 1e-12)
    # constant extrapolation at the ends
    np.testing.assert_allclose(X[7, 0] + rm[0], rs[7] * flux[7, 1], rtol = 1e-12)


def test_standard_scale_default_mask_is_ivar_zero(data):
    flux, ivar, _ = data
    ivar = ivar.copy()
    ivar[5, 20] = 0.
    mask = np.zeros(flux.shape, dtype = int)
    mask[5, 20] = 1
    a = standard_scale(flux, ivar)
    b = standard_scale(flux, ivar, mask)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_standard_scale_fully_masked_spectrum_is_not_nan(data):
    flux, ivar, _ = data
    mask = np.zeros(flux.shape, dtype = int)
    mask[2] = 1
    with pytest.warns(RuntimeWarning, match = "no unmasked pixels"):
        X, rs, rm = standard_scale(flux, ivar, mask)
    assert np.all(np.isfinite(X)) and np.all(np.isfinite(rm))
    assert rs[2] == 0. and np.all(X[2] == 0.)
    cov, _ = covar(X)
    assert np.all(np.isfinite(cov))
    # the other spectra are scaled as without the empty one
    X2, rs2, rm2 = standard_scale(np.delete(flux, 2, 0), np.delete(ivar, 2, 0))
    np.testing.assert_allclose(np.delete(X, 2, 0), X2, rtol = 1e-13)


def test_no_warning_without_empty_spectra(data):
    flux, ivar, _ = data
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        standard_scale(flux, ivar)


def test_covar_matches_numpy(scaled):
    X, _, _, cov = scaled
    np.testing.assert_allclose(cov, np.cov(X.T), rtol = 1e-10, atol = 1e-12)


# ---------------------------------------------------------------- conditioning
@pytest.mark.parametrize("bruteforce", [False, True])
@pytest.mark.parametrize("ipred_idx", [[30], [20, 25], [0, 1, 59]])
def test_gaussian_estimate_matches_direct(spd, bruteforce, ipred_idx):
    rng = np.random.default_rng(4)
    n = spd.shape[0]
    X = rng.multivariate_normal(np.zeros(n), spd, size = 50)
    ipred = np.zeros(n)
    ipred[ipred_idx] = 1
    icond = np.ones(n)
    icond[10:41] = 0
    icond[ipred_idx] = 0
    out = gaussian_estimate(icond, ipred, spd, X, cholesky_inv(spd), bruteforce = bruteforce)
    assert len(out) == 4
    predoverD, predcovar, predkstar, kstar = out
    pred, ccov = direct_conditional(spd, X, ipred, icond)
    np.testing.assert_array_equal(kstar, ipred_idx if len(ipred_idx) == 1 else np.sort(ipred_idx))
    np.testing.assert_allclose(predkstar, pred, rtol = 1e-6, atol = 1e-7 * np.abs(pred).max())
    np.testing.assert_allclose(predcovar, ccov, rtol = 1e-6, atol = 1e-7 * np.abs(ccov).max())
    np.testing.assert_allclose(X @ predoverD, predkstar, rtol = 1e-12, atol = 1e-12)
    assert np.all(predoverD[icond == 0] == 0.)


def test_gaussian_estimate_compute_pred_false(spd):
    n = spd.shape[0]
    X = np.random.default_rng(1).normal(size = (4, n))
    ipred = np.zeros(n); ipred[3] = 1
    icond = np.ones(n); icond[0:8] = 0
    a = gaussian_estimate(icond, ipred, spd, X, cholesky_inv(spd), compute_pred = False)
    b = gaussian_estimate(icond, ipred, spd, X, cholesky_inv(spd))
    assert a[2] is None
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


@pytest.mark.parametrize("nguard", [0, 3, 20])
def test_gp_interp_matches_direct_per_pixel(scaled, nguard):
    X, _, _, cov = scaled
    Xs = X[:300]
    pred, var = gp_interp(Xs, cov, nguard = nguard)
    npix = cov.shape[0]
    assert pred.shape == Xs.shape and var.shape == Xs.shape
    for i in range(0, npix, 7):
        icond = np.ones(npix, dtype = bool)
        icond[max(0, i - nguard):i + nguard + 1] = False
        ip = np.zeros(npix, dtype = bool)
        ip[i] = True
        p, c = direct_conditional(cov, Xs, ip, icond)
        np.testing.assert_allclose(pred[:, i], p[:, 0], rtol = 1e-6, atol = 1e-8 * np.abs(p).max())
        np.testing.assert_allclose(var[:, i], c[0, 0], rtol = 1e-6)
    # the variance is the same for every spectrum
    assert np.all(var == var[0])


def test_gp_interp_bruteforce_agrees_with_default(scaled):
    X, _, _, cov = scaled
    c = cov[:50, :50] + 0.1 * np.eye(50)
    a = gp_interp(X[:20, :50], c, nguard = 4)
    b = gp_interp(X[:20, :50], c, nguard = 4, bruteforce = True)
    np.testing.assert_allclose(a[0], b[0], rtol = 1e-6, atol = 1e-8)
    np.testing.assert_allclose(a[1], b[1], rtol = 1e-6)


def test_gp_interp_irange(scaled):
    X, _, _, cov = scaled
    full, fullvar = gp_interp(X[:50], cov, nguard = 5)
    part, partvar = gp_interp(X[:50], cov, nguard = 5, irange = np.array([10, 30]))
    assert part.shape == (50, 21)
    np.testing.assert_allclose(part, full[:, 10:31], rtol = 1e-12, atol = 1e-12)
    np.testing.assert_allclose(partvar, fullvar[:, 10:31], rtol = 1e-12)
    for bad in ([1, 2, 3], [5], [30, 10], [-1, 4], [0, 120]):
        with pytest.raises(AssertionError):
            gp_interp(X[:5], cov, irange = np.array(bad))


def test_gp_weights_reproduce_gp_interp(scaled):
    X, _, _, cov = scaled
    A, v = gp_weights(cov, nguard = 7)
    pred, var = gp_interp(X[:40], cov, nguard = 7)
    np.testing.assert_array_equal(X[:40] @ A, pred)
    np.testing.assert_array_equal(np.tile(v, (40, 1)), var)
    assert np.all(A[np.abs(np.arange(120)[:, None] - np.arange(120)[None, :]) <= 7] == 0.)


# ---------------------------------------------------------------- outlier masks
def test_get_chimask_finds_spikes(data):
    flux, ivar, _ = data
    bad, truth = inject_outliers(flux, ivar, nspike = 100, amplitude = 40.)
    m = get_chimask(bad, ivar, np.zeros(flux.shape, dtype = int), 10.)
    assert m[truth].mean() > 0.9
    assert m.sum() <= 3 * truth.sum() + 20  # each spike plus one dilated pixel on either side; few false positives


def legacy_covar_iter_mask(flux, ivar, mask, nsigma, maxbadpix = 64):
    """Independent restatement of the masking of the 2022 code, built from get_chimask."""
    use = (mask != 0).sum(1) <= maxbadpix
    f, iv, m0 = flux[use], ivar[use], mask[use] != 0
    cm = np.zeros_like(m0)
    for ns in nsigma:
        this = m0 | cm
        cm = get_chimask(f, iv, this, ns).astype(bool)
    return this | cm


@pytest.fixture(scope = "module")
def contaminated():
    flux, ivar, _ = make_synthetic_spectra(nspec = 1500, npix = 100, seed = 8)
    flux, _ = inject_outliers(flux, ivar, nspike = 150, amplitude = 40., seed = 1)
    flux, _ = inject_outliers(flux, ivar, nspike = 150, amplitude = 10., seed = 2)
    mask = np.zeros(flux.shape, dtype = int)
    mask[::11, 50] = 1
    return flux, ivar, mask


def test_covar_iter_mask_default_is_noncumulative(contaminated):
    flux, ivar, mask = contaminated
    ns = np.array([20, 8, 6])
    cov, final = covar_iter_mask(flux, ivar, mask, nsigma = ns)
    np.testing.assert_array_equal(final != 0, legacy_covar_iter_mask(flux, ivar, mask, ns))
    cov2, final2 = covar_iter_mask(flux, ivar, mask, nsigma = ns, cumulative = False)
    np.testing.assert_array_equal(cov, cov2)
    # covariance of the final data
    X, _, _ = standard_scale(flux, ivar, final)
    np.testing.assert_allclose(cov, covar(X)[0], rtol = 1e-13)


def test_covar_iter_mask_cumulative_is_monotone(contaminated):
    flux, ivar, mask = contaminated
    finals = [covar_iter_mask(flux, ivar, mask, nsigma = ns, cumulative = True)[1] != 0
              for ns in ([20], [20, 8], [20, 8, 6])]
    assert np.all(finals[0] >= (mask != 0))
    assert np.all(finals[1] >= finals[0])
    assert np.all(finals[2] >= finals[1])
    assert finals[2].sum() > finals[0].sum()


def test_covar_iter_mask_cumulative_final_mask_contains_every_pass(contaminated):
    flux, ivar, mask = contaminated
    ns = np.array([20, 8, 6])
    _, final = covar_iter_mask(flux, ivar, mask, nsigma = ns, cumulative = True)
    # the first pass is the same in both modes, and its flags must survive in the cumulative mask
    first = get_chimask(flux, ivar, mask != 0, 20).astype(bool)
    assert np.all((final != 0)[first])


def test_covar_iter_mask_defaults_and_options(contaminated):
    flux, ivar, mask = contaminated
    a = covar_iter_mask(flux, ivar, np.zeros(flux.shape, dtype = int))
    b = covar_iter_mask(flux, ivar)  # mask=None means ivar == 0, i.e. nothing masked here
    np.testing.assert_array_equal(a[0], b[0])
    cov, final, refmean = covar_iter_mask(flux, ivar, mask, cumulative = True, return_scale = True)
    X, _, rm = standard_scale(flux, ivar, final)
    np.testing.assert_allclose(refmean, rm, rtol = 1e-13)


def test_covar_iter_mask_rejects_spectra_with_many_bad_pixels(contaminated):
    flux, ivar, mask = contaminated
    mask = mask.copy()
    mask[0, :70] = 1
    _, final = covar_iter_mask(flux, ivar, mask, maxbadpix = 64)
    assert np.all(final[0] == 1)
    assert not np.all(final[1] == 1)


def test_no_output_by_default(contaminated, capsys):
    flux, ivar, mask = contaminated
    X, _, _ = standard_scale(flux, ivar, mask)
    c, _ = covar(X)
    gp_interp(X[:10], c)
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""
