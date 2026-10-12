import numpy as np
import pytest
from scipy.ndimage import uniform_filter1d

from gspice import (standard_scale, covar, gp_weights, unscale, gp_infill, gp_blockwise, make_blocks,
                    block_operators, block_chisq, block_thresholds, exclude_extended, gp_repair)
from gspice.synthetic import make_synthetic_spectra, inject_outliers
from helpers import direct_conditional


@pytest.fixture(scope = "module")
def gauss(spd):
    """Gaussian draws from the covariance spd, so conditional estimates are meaningful."""
    return np.random.default_rng(7).multivariate_normal(np.zeros(spd.shape[0]), spd, size = 40)


def random_masks(shape, rng, nbad = 6):
    m = np.zeros(shape, dtype = bool)
    for s in range(shape[0]):
        m[s, rng.choice(shape[1], rng.integers(0, nbad), replace = False)] = True
    m[10, 5] = True
    m[10:20] = m[10]  # a group of spectra sharing one pattern
    return m


# ---------------------------------------------------------------- unscale
def test_unscale_inverts_standard_scale(data):
    flux, ivar, _ = data
    X, rs, rm = standard_scale(flux, ivar)
    f, v = unscale(X, np.full(flux.shape[1], 2.), rs, rm)
    np.testing.assert_allclose(f, flux, rtol = 1e-12, atol = 1e-13)
    np.testing.assert_allclose(v, 2. / rs[:, None]**2 * np.ones(flux.shape), rtol = 1e-14)
    f2, v2 = unscale(X[:, 3:9], np.ones((flux.shape[0], 6)), rs, rm[3:9])
    np.testing.assert_allclose(f2, flux[:, 3:9], rtol = 1e-12, atol = 1e-13)
    # without refmean only the scale is removed
    f3, _ = unscale(rs[:, None] * flux, 1., rs)
    np.testing.assert_allclose(f3, flux, rtol = 1e-13)


def test_unscale_empty_spectrum_gives_nan():
    f, v = unscale(np.zeros((2, 3)), np.ones(3), np.array([0., 2.]), np.zeros(3))
    assert np.all(np.isnan(f[0])) and np.all(np.isnan(v[0]))
    np.testing.assert_allclose(v[1], 0.25)


# ---------------------------------------------------------------- gp_infill
def test_gp_infill_matches_direct(spd, gauss):
    rng = np.random.default_rng(5)
    M = random_masks(gauss.shape, rng)
    filled, var, groups = gp_infill(gauss, spd, M, return_covar = True)
    for s in range(gauss.shape[0]):
        if not M[s].any():
            np.testing.assert_array_equal(filled[s], gauss[s])
            assert np.all(var[s] == 0.)
            continue
        p, c = direct_conditional(spd, gauss[s:s + 1], M[s], ~M[s])
        scale = np.abs(p).max() + 1e-300
        np.testing.assert_allclose(filled[s, M[s]], p[0], rtol = 1e-7, atol = 1e-8 * scale)
        np.testing.assert_allclose(var[s, M[s]], np.diag(c), rtol = 1e-7)
        np.testing.assert_array_equal(filled[s, ~M[s]], gauss[s, ~M[s]])
        assert np.all(var[s, ~M[s]] == 0.)
    # groups: every spectrum with a masked pixel appears once, with the conditional covariance
    seen = np.concatenate([g[0] for g in groups])
    assert sorted(seen) == sorted(np.flatnonzero(M.any(axis = 1)))
    for rows, cols, cc in groups:
        _, c = direct_conditional(spd, gauss[rows[:1]], M[rows[0]], ~M[rows[0]])
        np.testing.assert_allclose(cc, c, rtol = 1e-6, atol = 1e-9)
        assert np.all(M[rows][:, cols]) and np.all(M[rows].sum(1) == len(cols))
    assert len(groups) < M.any(axis = 1).sum()  # spectra sharing a pattern share a group


def test_gp_infill_with_ignored_pixels(spd, gauss):
    rng = np.random.default_rng(6)
    M = random_masks(gauss.shape, rng, nbad = 4)
    I = random_masks(gauss.shape, rng, nbad = 8)
    filled, var = gp_infill(gauss, spd, M, ignore = I)
    for s in range(gauss.shape[0]):
        Is = I[s] & ~M[s]
        np.testing.assert_array_equal(filled[s, Is], gauss[s, Is])  # ignored pixels are not replaced
        if not M[s].any():
            continue
        p, c = direct_conditional(spd, gauss[s:s + 1], M[s], ~(M[s] | Is))
        np.testing.assert_allclose(filled[s, M[s]], p[0], rtol = 1e-6, atol = 1e-8 * np.abs(p).max())
        np.testing.assert_allclose(var[s, M[s]], np.diag(c), rtol = 1e-7)


def test_gp_infill_never_reads_masked_or_ignored_values(spd, gauss):
    rng = np.random.default_rng(9)
    M = random_masks(gauss.shape, rng)
    I = random_masks(gauss.shape, rng)
    clean = gp_infill(gauss, spd, M, ignore = I)
    bad = gauss.copy()
    bad[M | I] = np.nan
    out = gp_infill(bad, spd, M, ignore = I)
    np.testing.assert_array_equal(out[0][M], clean[0][M])
    np.testing.assert_array_equal(out[1], clean[1])


def test_gp_infill_edge_cases(spd, gauss):
    filled, var = gp_infill(gauss, spd, np.zeros(gauss.shape, dtype = bool))
    np.testing.assert_array_equal(filled, gauss)
    assert np.all(var == 0.)
    full = np.zeros(gauss.shape, dtype = bool)
    full[0] = True
    filled, var = gp_infill(gauss, spd, full)  # nothing to condition on: prior mean and variance
    np.testing.assert_allclose(filled[0], 0., atol = 1e-8)
    np.testing.assert_allclose(var[0], np.diag(spd), rtol = 1e-8)
    np.testing.assert_array_equal(filled[1:], gauss[1:])


def test_gp_infill_conditional_variance_is_calibrated(scaled):
    """Held-out spectra: residuals of infilled pixels, divided by the predicted sigma, have unit scale."""
    X = scaled[0]
    train, test = X[:1500], X[1500:]
    c, _ = covar(train)
    M = np.zeros(test.shape, dtype = bool)
    rng = np.random.default_rng(1)
    for s in range(test.shape[0]):
        M[s, rng.choice(test.shape[1], 6, replace = False)] = True
    filled, var = gp_infill(test, c, M)
    z = (test - filled)[M] / np.sqrt(var[M])
    assert 0.9 < np.sqrt(np.mean(z**2)) < 1.1


# ---------------------------------------------------------------- blockwise
def test_gp_blockwise_matches_direct(spd, gauss):
    n = spd.shape[0]
    ip = np.arange(30, 45)
    ic = np.arange(0, 20)
    pred, ccov = gp_blockwise(gauss, spd, ip, ic)
    icm = np.zeros(n, bool); icm[ic] = True
    ipm = np.zeros(n, bool); ipm[ip] = True
    p, c = direct_conditional(spd, gauss, ipm, icm)
    np.testing.assert_allclose(pred, p, rtol = 1e-9, atol = 1e-10)
    np.testing.assert_allclose(ccov, c, rtol = 1e-9, atol = 1e-12)
    # boolean masks give the same answer
    pred2, ccov2 = gp_blockwise(gauss, spd, ipm, icm)
    np.testing.assert_array_equal(pred, pred2)
    # default conditioning set: everything outside the guard region
    pred3, ccov3 = gp_blockwise(gauss, spd, ip, nguard = 5)
    near = np.zeros(n, bool); near[25:50] = True
    p3, c3 = direct_conditional(spd, gauss, ipm, ~near)
    np.testing.assert_allclose(pred3, p3, rtol = 1e-8, atol = 1e-9)
    np.testing.assert_allclose(ccov3, c3, rtol = 1e-8, atol = 1e-12)
    # explicit conditioning set with a guard removes the nearby pixels from it
    ic2 = np.arange(0, 33)
    pred4, _ = gp_blockwise(gauss, spd, ip, ic2, nguard = 5)
    keep = np.zeros(n, bool); keep[0:25] = True
    p4, _ = direct_conditional(spd, gauss, ipm, keep)
    np.testing.assert_allclose(pred4, p4, rtol = 1e-8, atol = 1e-9)


def test_make_blocks():
    b = make_blocks(2415, 150, 75)
    assert len(b) == 32 and b[0][0] == 0 and b[-1][-1] == 2414 and all(len(x) == 150 for x in b)
    assert make_blocks(300, 100, 100)[-1][-1] == 299
    assert len(make_blocks(250, 100, 100)) == 3 and make_blocks(250, 100, 100)[-1][0] == 150


def test_block_chisq_is_whitened_residual(spd, gauss):
    blocks = [np.arange(5, 25), np.arange(30, 50)]
    ops = block_operators(spd, blocks, nguard = 3)
    chi = block_chisq(gauss, ops)
    assert chi.shape == (gauss.shape[0], 2)
    for k, a in enumerate(blocks):
        pred, ccov = gp_blockwise(gauss, spd, a, nguard = 3)
        r = gauss[:, a] - pred
        expected = np.einsum('ij,ij->i', r, np.linalg.solve(ccov, r.T).T) / len(a)
        np.testing.assert_allclose(chi[:, k], expected, rtol = 1e-8)
    thr = block_thresholds(gauss, ops, percentile = 90.)
    np.testing.assert_allclose(thr, np.percentile(chi, 90., axis = 0))


# ---------------------------------------------------------------- exclusion of extended artifacts
def reference_exclusion(M, I, run = 5, half = 150):
    """Restatement of the simulation script (centred 5-pixel window filter, then dilation)."""
    G = (M | I).astype(float)
    r = uniform_filter1d(G, run, axis = 1, mode = "constant") > 0.999
    out = I.copy()
    if r.any():
        wide = r.copy()
        for d in range(1, half + (run - 1) // 2 + 1):
            wide[:, d:] |= r[:, :-d]
            wide[:, :-d] |= r[:, d:]
        Iw = (I | wide) & ~M
        ok = (~(M | Iw)).sum(1) >= M.shape[1] // 2
        out = np.where(ok[:, None], Iw, I)
    return out


def test_exclude_extended_basic():
    npix = 1000
    M = np.zeros((4, npix), bool)
    I = np.zeros((4, npix), bool)
    M[0, 400:405] = True          # run of 5 in M
    M[1, 400:404] = True          # run of only 4
    M[2, 400:402] = True; I[2, 402:405] = True   # run of 5 split between M and I
    M[3, 400:402] = True; M[3, 410] = True       # scattered
    out = exclude_extended(M, I)
    assert out[0, 250:400].all() and out[0, 405:555].all()     # 150 pixels on either side
    assert not out[0, 249] and not out[0, 555]
    assert not out[0, 400:405].any()                            # replaced pixels are never ignored
    assert not out[1].any() and not out[3].any()                # runs shorter than 5 do nothing
    assert out[2, 250:400].all() and out[2, 405:555].all()
    np.testing.assert_array_equal(out[2, 402:405], True)       # pre-existing ignored pixels remain
    assert not (out & M).any()


def test_exclude_extended_half_spectrum_safeguard():
    npix = 400
    M = np.zeros((2, npix), bool)
    M[:, 200:205] = True
    out = exclude_extended(M, np.zeros_like(M), halfwidth = 150)
    assert not out.any()   # would leave fewer than half of the pixels, so skipped
    out = exclude_extended(M, np.zeros_like(M), halfwidth = 150, minfrac = 0.2)
    assert out[:, 50:200].all() and out[:, 205:355].all()
    out = exclude_extended(M, np.zeros_like(M), halfwidth = 150, minfrac = 0.2, runlength = 6)
    assert not out.any()  # no run of 6


def test_exclude_extended_edges_and_options():
    npix = 600
    M = np.zeros((1, npix), bool)
    M[0, 0:5] = True              # run at the first pixels
    out = exclude_extended(M, np.zeros_like(M), halfwidth = 20)
    assert out[0, 5:25].all() and not out[0, 25]
    out = exclude_extended(M, np.zeros_like(M), runlength = 3, halfwidth = 7)
    assert out[0, 5:12].all() and not out[0, 12]


def test_exclude_extended_matches_simulation_script():
    rng = np.random.default_rng(3)
    npix = 900
    M = rng.random((300, npix)) < 0.01
    I = (rng.random((300, npix)) < 0.04) & ~M
    for s in range(0, 300, 3):  # add runs
        a = rng.integers(0, npix - 8)
        (M if s % 2 else I)[s, a:a + rng.integers(2, 8)] = True
    I &= ~M
    np.testing.assert_array_equal(exclude_extended(M, I), reference_exclusion(M, I))


# ---------------------------------------------------------------- end to end
@pytest.fixture(scope = "module")
def sim():
    """Training (4000), held-out calibration (2000) and test (2200) spectra of 300 pixels; the
    covariance is estimated from the clean training spectra."""
    flux, ivar, truth = make_synthetic_spectra(nspec = 8200, npix = 300, nmodes = 10, seed = 21)
    X, rs, rm = standard_scale(flux[:4000], ivar[:4000])
    cov, _ = covar(X)
    weights = gp_weights(cov, nguard = 10)
    return dict(flux = flux[6000:], ivar = ivar[6000:], truth = truth[6000:], cov = cov, refmean = rm,
                weights = weights, cal_flux = flux[4000:6000], cal_ivar = ivar[4000:6000])


def test_gp_repair_replaces_spikes(sim):
    flux, ivar, truth = sim['flux'][:800], sim['ivar'][:800], sim['truth'][:800]
    bad, hit = inject_outliers(flux, ivar, nspike = 200, amplitude = 40., seed = 4, width = 2)
    new, var, info = gp_repair(bad, ivar, sim['cov'], sim['refmean'], nguard = 10, weights = sim['weights'])
    M = info['replace']
    assert M[hit].mean() > 0.95                      # spikes found
    assert (M & ~hit).sum() < 2 * 200 + 100  # other replaced pixels: dilation (2 per spike) and rare false positives
    np.testing.assert_array_equal(new[~M], bad[~M])  # untouched pixels are returned unchanged
    assert np.all(var[~M] == 0.) and np.all(var[M] > 0.)
    sig = 1. / np.sqrt(ivar)
    before = np.abs((bad - truth) / sig)[hit].mean()
    after = np.abs((new - truth) / sig)[hit & M].mean()
    assert before > 30 and after < 1.5               # repaired close to the noiseless truth
    # the conditional variance describes the noisy data, so compare with the spectrum before the spike
    z = ((new - flux)[hit & M]) / np.sqrt(var[hit & M])
    assert 0.8 < np.sqrt(np.mean(z**2)) < 1.25
    assert info['block_flag'] is None and info['refscale'].shape == (800,)


def test_gp_repair_equals_manual_pipeline(sim):
    flux, ivar = sim['flux'][:300], sim['ivar'][:300]
    bad, _ = inject_outliers(flux, ivar, nspike = 60, amplitude = 25., seed = 5)
    new, var, info = gp_repair(bad, ivar, sim['cov'], sim['refmean'], nguard = 10,
                               weights = sim['weights'], exclude_runs = False)
    M, I = info['replace'], info['ignore']
    X, rs, _ = standard_scale(bad, ivar, refmean = sim['refmean'])
    Xr, v = gp_infill(X, sim['cov'], M, ignore = I)
    f, fv = unscale(Xr, v, rs, sim['refmean'])
    np.testing.assert_allclose(new[M], f[M], rtol = 1e-12)
    np.testing.assert_allclose(var[M], fv[M], rtol = 1e-12)


def test_gp_repair_input_mask_and_zero_ivar(sim):
    flux, ivar = sim['flux'][:100].copy(), sim['ivar'][:100].copy()
    ivar[3, 50:53] = 0.
    flux[3, 50:53] = 1e6                     # nothing may depend on these values
    new, var, info = gp_repair(flux, ivar, sim['cov'], sim['refmean'], nguard = 10, weights = sim['weights'])
    assert info['replace'][3, 50:53].all()
    assert np.all(np.isfinite(new)) and np.all(var[3, 50:53] > 0)
    assert np.all(np.abs(new[3, 50:53] - sim['truth'][3, 50:53]) < 6 / np.sqrt(sim['ivar'][3, 50:53]))
    mask = np.zeros(flux.shape, int)
    mask[7, 100] = 1
    new2, _, info2 = gp_repair(sim['flux'][:100], sim['ivar'][:100], sim['cov'], sim['refmean'], mask = mask,
                               nguard = 10, weights = sim['weights'])
    assert info2['replace'][7, 100]
    assert abs(new2[7, 100] - sim['truth'][7, 100]) < 5 / np.sqrt(sim['ivar'][7, 100])


def test_gp_repair_clean_spectra_are_mostly_untouched(sim):
    flux, ivar = sim['flux'], sim['ivar']
    new, var, info = gp_repair(flux, ivar, sim['cov'], sim['refmean'], nguard = 10, weights = sim['weights'])
    assert info['replace'].sum() < 0.001 * flux.size


def test_gp_repair_fully_masked_spectrum(sim):
    ivar = sim['ivar'][:20].copy()
    ivar[4] = 0.
    with pytest.warns(RuntimeWarning):
        new, var, info = gp_repair(sim['flux'][:20], ivar, sim['cov'], sim['refmean'], nguard = 10,
                                   weights = sim['weights'])
    others = np.arange(20) != 4
    assert np.all(np.isfinite(new[others])) and np.all(np.isfinite(var[others]))


def _block_setup(sim):
    cov, rm = sim['cov'], sim['refmean']
    ops = block_operators(cov, make_blocks(300, 60, 30), nguard = 10)
    Xc, _, _ = standard_scale(sim['cal_flux'], sim['cal_ivar'], refmean = rm)  # held-out clean spectra
    return ops, block_thresholds(Xc, ops, percentile = 99.9)


def test_gp_repair_blockwise_detects_extended_offsets(sim):
    cov, rm = sim['cov'], sim['refmean']
    ops, thr = _block_setup(sim)
    ivar, truth = sim['ivar'][:600], sim['truth'][:600]
    sig = 1. / np.sqrt(ivar)
    bump = 0.5 * (1 - np.cos(np.linspace(0, 2 * np.pi, 80)))
    bad = sim['flux'][:600].copy()
    rows = np.arange(0, 600, 3)
    for s in rows:
        bad[s, 110:190] += 4.0 * bump * sig[s, 110:190]
    new, var, info = gp_repair(bad, ivar, cov, rm, nguard = 10, weights = sim['weights'],
                               block_ops = ops, block_thresh = thr)
    flagged = info['block_flag'].any(axis = 1)
    assert flagged[rows].mean() > 0.6                             # most corrupted spectra caught
    assert flagged[np.setdiff1d(np.arange(600), rows)].mean() < 0.05
    err_before = np.abs((bad - truth) / sig)[rows][:, 110:190].mean()
    err_after = np.abs((new - truth) / sig)[rows][:, 110:190].mean()
    assert err_after < err_before
    # a block threshold must be supplied with the operators
    with pytest.raises(ValueError):
        gp_repair(bad[:5], ivar[:5], cov, rm, nguard = 10, weights = sim['weights'], block_ops = ops)


def test_gp_repair_blocks_are_tested_after_the_pixelwise_pass(sim):
    """A strong narrow artifact must not make blocks fail (the blockwise test follows the pixelwise pass)."""
    cov, rm = sim['cov'], sim['refmean']
    ops, thr = _block_setup(sim)
    flux, ivar = sim['flux'][:200], sim['ivar'][:200]
    bad, hit = inject_outliers(flux, ivar, nspike = 100, amplitude = 200., seed = 6, width = 6)
    _, _, info = gp_repair(bad, ivar, cov, rm, nguard = 10, weights = sim['weights'],
                           block_ops = ops, block_thresh = thr)
    assert info['block_flag'].sum() < 0.1 * hit.any(axis = 1).sum()
    assert info['replace'][hit].mean() > 0.95


def test_gp_repair_exclusion_option_changes_only_the_ignore_set(sim):
    flux, ivar = sim['flux'][:400], sim['ivar'][:400]
    bad, hit = inject_outliers(flux, ivar, nspike = 150, amplitude = 60., seed = 8, width = 8)
    kw = dict(nguard = 10, weights = sim['weights'])
    _, _, a = gp_repair(bad, ivar, sim['cov'], sim['refmean'], exclude_runs = False, **kw)
    _, _, b = gp_repair(bad, ivar, sim['cov'], sim['refmean'], exclusion_halfwidth = 40, minfrac = 0.5, **kw)
    np.testing.assert_array_equal(a['replace'], b['replace'])
    assert b['ignore'].sum() > a['ignore'].sum()
    assert np.all(b['ignore'][a['ignore']])
    assert not (b['ignore'] & b['replace']).any()


def test_gp_repair_return_covar(sim):
    flux, ivar = sim['flux'][:100], sim['ivar'][:100]
    bad, _ = inject_outliers(flux, ivar, nspike = 20, amplitude = 50., seed = 9, width = 3)
    _, var, info = gp_repair(bad, ivar, sim['cov'], sim['refmean'], nguard = 10, weights = sim['weights'],
                             return_covar = True)
    for rows, cols, cc in info['groups']:
        for r in rows:
            np.testing.assert_allclose(np.diag(cc) / info['refscale'][r]**2, var[r, cols], rtol = 1e-10)
