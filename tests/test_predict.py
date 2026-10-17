import logging

import numpy as np
import pytest
from numpy.linalg import LinAlgError

from _golden import checked, same
from gspice import Model, _model, _spectra, pixelwise, scale, snr, unscale
from gspice._linalg import precision
from oracle import condition, guard_conditional, random_spd

EPS = np.finfo(float).eps
NPIX = 12
PIXELS = np.arange(NPIX)
# The mean of each of two S/N bins, near the S/N of the spectra of the bin.
MEANS = np.array([5 + 0.3 * np.cos(PIXELS), 20 + 0.3 * np.sin(PIXELS)])
# The bin of each spectrum: every chunk of two rows holds both bins.
BINS = np.array([0, 1, 1, 0, 1, 0, 0, 1])
# (spectrum, pixels, in mask): runs at both edges, interior runs and isolated pixels,
# in mask (NaN flux, inf ivar) or with ivar = 0 (NaN flux).
BAD = [
    (0, [0, 1], True),
    (1, [5, 6, 7], False),
    (2, [9], True),
    (3, [10, 11], False),
    (4, [3, 4], True),
    (6, [8], False),
]


def two_bins(kappa, center=None):
    # Spectra whose working vectors are drawn from N(0, cov[b]) at good pixels, and a
    # model whose edge is the S/N of a spectrum of bin 1.
    cov = (random_spd(NPIX, kappa, 0), random_spd(NPIX, kappa, 1))
    rng = np.random.default_rng(4)
    ivar = np.outer(rng.uniform(1, 16, BINS.size), 1 + 0.5 * np.sin(PIXELS))
    mask = np.zeros(ivar.shape, bool)
    for s, pixels, in_mask in BAD:
        if in_mask:
            mask[s, pixels] = True
        else:
            ivar[s, pixels] = 0.0
    a = scale(np.ones(ivar.shape), ivar, mask)[1]
    x = np.array([np.linalg.cholesky(cov[b]) @ rng.standard_normal(NPIX) for b in BINS])
    flux = (x + MEANS[BINS]) / a[:, np.newaxis] + (0.0 if center is None else center)
    flux[mask | (ivar == 0)] = np.nan
    ivar[mask] = np.inf
    ratio = snr(flux, ivar, mask)
    edges = (ratio[BINS == 1].min(),)
    model = Model(MEANS, cov, (100, 100), edges=edges, center=center)
    # side="right" puts the spectrum at the edge in bin 1.
    np.testing.assert_array_equal(np.searchsorted(edges, ratio, side="right"), BINS)
    return model, flux, ivar, mask


def arrays(prediction):
    return prediction.flux, prediction.var, prediction.z


def reference(model, b, nguard, flux, ivar, mask):
    # pixelwise of the spectra of bin b, from the 60-digit conditional of each window.
    center = 0.0 if model.center is None else model.center
    X, a, _ = scale(flux, ivar, mask, center=center, refmean=model.mean[b])
    xhat, var = guard_conditional(model.cov[b], nguard, X, range(NPIX))
    f, v = unscale(xhat, a, model.mean[b], center=center, var=var)
    return X, f, v, (X - xhat) / np.sqrt(var)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("nguard", [0, 2, NPIX - 2, NPIX - 1])
@pytest.mark.parametrize("kappa", [1e2, 1e8])
@pytest.mark.parametrize("center", [None, 0.9])
def test_pixelwise_matches_the_60_digit_reference(center, kappa, nguard):
    model, flux, ivar, mask = two_bins(kappa, center)
    result = pixelwise(model, flux, ivar, mask, nguard=nguard)
    bad = mask | (ivar == 0)
    c = 0.0 if center is None else center
    for b in (0, 1):
        rows = BINS == b
        X, f, var, z = reference(model, b, nguard, flux[rows], ivar[rows], mask[rows])
        # Design section 11: |dxhat| <= tol sigma D and |dvar| <= tol var, with
        # tol = 10 kappa eps, D = max(1, M/sqrt(npix)) and M = |L^-1 x| the Mahalanobis
        # norm of x, C = L L^T; the interpolated bad pixels make M reach 2500 here. The
        # two evaluations of (xhat + m)/a + c round three times each, by at most
        # eps (3|f| + 2|c|) in all to first order; those of var/a^2 round once each,
        # with the same a^2; and z = (x - xhat)/sqrt(var) moves by |dxhat|/sigma +
        # |z| |dvar|/(2 var) and three roundings of each evaluation, within
        # tol (D + |z|).
        tol = 10 * condition(model.cov[b]) * EPS
        L = np.linalg.cholesky(model.cov[b])
        M = np.linalg.norm(np.linalg.solve(L, X.T), axis=0)
        D = np.maximum(1, M / np.sqrt(NPIX))[:, np.newaxis]
        rounding = 4 * EPS * (np.abs(f) + abs(c))
        sigma = np.sqrt(var)
        assert np.all(np.abs(result.flux[rows] - f) <= tol * D * sigma + rounding)
        assert np.all(np.abs(result.var[rows] - var) <= (tol + EPS) * var)
        good = ~bad[rows]
        bound = tol * (D + np.abs(z))
        assert np.all(np.abs(result.z[rows][good] - z[good]) <= bound[good])
    np.testing.assert_array_equal(np.isnan(result.z), bad)


def test_slices_and_chunks_agree_with_one_call_to_rounding(monkeypatch):
    model, flux, ivar, mask = two_bins(1e8, center=0.9)
    whole = pixelwise(model, flux, ivar, mask, nguard=2)
    # The second slice is one row, for which numpy multiplies a vector by W.
    pieces = [
        arrays(pixelwise(model, flux[rows], ivar[rows], mask[rows], nguard=2))
        for rows in (slice(0, 3), slice(3, 4), slice(4, None))
    ]
    sliced = tuple(map(np.concatenate, zip(*pieces, strict=True)))
    monkeypatch.setattr(_spectra, "CHUNK", 2 * NPIX)
    chunked = arrays(pixelwise(model, flux, ivar, mask, nguard=2))
    # Only the summation order of xhat_i = sum_j x_j W_ji differs, and two orders
    # differ by at most 2 gamma_n sum_j |x_j| |W_ji|, gamma_n = n u/(1 - n u), with
    # u = eps/2 and n = npix. flux and z then round as in the reference test.
    gamma = NPIX * EPS / 2 / (1 - NPIX * EPS / 2)
    dflux, dz = np.empty(flux.shape), np.empty(flux.shape)
    for b in (0, 1):
        rows = BINS == b
        spectra = flux[rows], ivar[rows], mask[rows]
        X, a, _ = scale(*spectra, center=0.9, refmean=MEANS[b])
        W, var = model._guard_weights(b, 2)
        dxhat = 2 * gamma * (np.abs(X) @ np.abs(W))
        rounding = 4 * EPS * (np.abs(whole.flux[rows]) + 0.9)
        dflux[rows] = dxhat / a[:, np.newaxis] + rounding
        dz[rows] = dxhat / np.sqrt(var) + 4 * EPS * np.abs(whole.z[rows])
    good = ~np.isnan(whole.z)
    for result_flux, result_var, result_z in (sliced, chunked):
        assert same(result_var, whole.var)
        assert np.all(np.abs(result_flux - whole.flux) <= dflux)
        np.testing.assert_array_equal(np.isnan(result_z), ~good)
        assert np.all(np.abs(result_z[good] - whole.z[good]) <= dz[good])


def test_a_model_reuses_its_precision_matrix_when_nguard_changes(monkeypatch):
    def new_model():
        return Model(np.zeros((1, NPIX)), (random_spd(NPIX, 1e2, 0),), (100,))

    flux = np.random.default_rng(5).standard_normal((4, NPIX))
    ivar = np.ones(flux.shape)
    expected = {
        nguard: arrays(pixelwise(new_model(), flux, ivar, nguard=nguard))
        for nguard in (2, 5)
    }
    calls = []

    def counted(S):
        calls.append(S)
        return precision(S)

    monkeypatch.setattr(_model, "precision", counted)
    model = new_model()
    for nguard in (2, 5, 2):
        result = pixelwise(model, flux, ivar, nguard=nguard)
        assert same(arrays(result), expected[nguard])
    assert len(calls) == 1


def test_the_prior_needs_no_precision_matrix():
    model, flux, ivar, mask = two_bins(1e2, center=0.9)
    # Singular with a positive diagonal. Its entries are powers of 2, so dpotrf finds
    # the leading minor of order 2 exactly 0.
    v = 2.0 ** (PIXELS % 3 - 1)
    C = np.outer(v, v)
    prior = Model(MEANS[:1], (C,), (100,), center=0.9)
    result = pixelwise(prior, flux, ivar, mask, nguard=NPIX - 1)
    X, a, _ = scale(flux, ivar, mask, center=0.9, refmean=MEANS[0])
    assert same(result.flux, MEANS[0] / a[:, np.newaxis] + 0.9)
    assert same(result.var, np.diagonal(C) / np.square(a)[:, np.newaxis])
    expected_z = np.where(mask | (ivar == 0), np.nan, X / np.sqrt(np.diagonal(C)))
    assert same(result.z, expected_z)
    with pytest.raises(LinAlgError, match=r"^model: cov\[0\] is not positive definite"):
        pixelwise(prior, flux, ivar, mask, nguard=0)
    # Only bin 1 is singular: the message names it, and the spectra of bin 0 alone
    # never need its precision matrix.
    mixed = Model(MEANS, (model.cov[0], C), (100, 100), edges=model.edges)
    with pytest.raises(LinAlgError, match=r"^model: cov\[1\] is not positive definite"):
        pixelwise(mixed, flux, ivar, mask, nguard=0)
    rows = BINS == 0
    pixelwise(mixed, flux[rows], ivar[rows], mask[rows], nguard=0)


def test_a_covariance_that_fails_its_check_raises_before_any_chunk(monkeypatch):
    monkeypatch.setattr(_spectra, "CHUNK", 2 * NPIX)
    model, flux, ivar, mask = two_bins(1e2)
    cov = (model.cov[0], np.ones((NPIX, NPIX)))
    singular = Model(MEANS, cov, (100, 100), edges=model.edges)
    # The spectra of bin 0 first, so that the chunks of bin 1 come last.
    rows = np.argsort(BINS, kind="stable")
    calls = []
    with pytest.raises(LinAlgError, match=r"^model: cov\[1\] is "):
        pixelwise(
            singular,
            flux[rows],
            ivar[rows],
            mask[rows],
            nguard=2,
            progress=lambda *args: calls.append(args),
        )
    assert calls == []


def test_a_guard_window_that_fails_its_factorization_names_the_bin(monkeypatch):
    # A precision matrix that passed its check, but is not positive definite on the
    # window of pixel 7, as rounding can make it near the rcond limit.
    P = np.eye(NPIX)
    P[7, 7] = -1.0
    monkeypatch.setattr(_model, "precision", lambda S: P)
    model, flux, ivar, mask = two_bins(1e2)
    with pytest.raises(LinAlgError, match=r"^model: cov\[0\] is .* pixel 7$"):
        pixelwise(Model(MEANS[:1], model.cov[:1], (100,)), flux, ivar, mask, nguard=0)


def test_inputs_are_unchanged_and_outputs_are_read_only_and_not_shared():
    model, flux, ivar, mask = two_bins(1e2)
    outputs = checked(
        lambda *spectra: arrays(pixelwise(model, *spectra, nguard=2)), flux, ivar, mask
    )
    assert not any(output.flags.writeable for output in outputs)


def test_float32_and_f_ordered_inputs_give_the_result_of_a_float64_c_copy():
    model, flux, ivar, mask = two_bins(1e2, center=0.9)
    flux32, ivar32 = flux.astype(np.float32), ivar.astype(np.float32)
    cast = pixelwise(model, flux32.astype(float), ivar32.astype(float), mask, nguard=2)
    assert same(arrays(pixelwise(model, flux32, ivar32, mask, nguard=2)), arrays(cast))
    fortran = [np.asfortranarray(array) for array in (flux, ivar, mask)]
    expected = arrays(pixelwise(model, flux, ivar, mask, nguard=2))
    assert same(arrays(pixelwise(model, *fortran, nguard=2)), expected)


def test_progress_counts_the_spectra_done_after_each_chunk(monkeypatch):
    monkeypatch.setattr(_spectra, "CHUNK", 2 * NPIX)
    model, flux, ivar, mask = two_bins(1e2)
    calls = []
    pixelwise(
        model,
        flux[:5],
        ivar[:5],
        mask[:5],
        nguard=2,
        progress=lambda *args: calls.append(args),
    )
    assert calls == [("pixelwise", 2, 5), ("pixelwise", 4, 5), ("pixelwise", 5, 5)]


def test_a_call_on_a_new_model_logs_its_timings_at_debug(caplog):
    caplog.set_level(logging.DEBUG, logger="gspice")
    model, flux, ivar, mask = two_bins(1e2)
    one_bin = Model(MEANS[:1], model.cov[:1], (100,))
    pixelwise(one_bin, flux, ivar, mask, nguard=2)
    # P, the guard weights and the call.
    records = [(record.name, record.levelno) for record in caplog.records]
    assert records == [("gspice", logging.DEBUG)] * 3
    handlers = logging.getLogger("gspice").handlers
    assert any(isinstance(handler, logging.NullHandler) for handler in handlers)


def test_zero_spectra_give_empty_arrays_without_a_precision_matrix():
    singular = Model(np.zeros((1, 3)), (np.ones((3, 3)),), (5,))
    empty = np.zeros((0, 3))
    calls = []
    result = pixelwise(
        singular, empty, empty, nguard=0, progress=lambda *args: calls.append(args)
    )
    assert [(a.shape, a.dtype) for a in arrays(result)] == [((0, 3), np.float64)] * 3
    assert calls == []
