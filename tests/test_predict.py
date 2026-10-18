import dataclasses
import logging
from functools import partial

import numpy as np
import pytest
from numpy.linalg import LinAlgError

from _golden import arrays, checked, same
from gspice import (
    LowRank,
    Model,
    _model,
    _spectra,
    finite_sample_factor,
    fit,
    infill,
    pixelwise,
    scale,
    snr,
    unscale,
)
from gspice._linalg import precision
from oracle import (
    assert_gaussian,
    condition,
    conditional,
    dense,
    guard_conditional,
    kappa_w,
    low_rank,
    random_low_rank,
    random_spd,
    window,
)

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


def two_bins(kappa, center=None, rank=None):
    # Spectra whose working vectors are drawn from N(0, cov[b]) at good pixels, and a
    # model whose edge is the S/N of a spectrum of bin 1. With a rank, the covariances
    # are LowRank, with kappa_W 53 and 37 at kappa 1e2 and 9.5e7 and 5.9e7 at 1e8.
    if rank is None:
        cov = (random_spd(NPIX, kappa, 0), random_spd(NPIX, kappa, 1))
    else:
        cov = tuple(LowRank(*random_low_rank(NPIX, rank, kappa, b)) for b in (0, 1))
    rng = np.random.default_rng(4)
    ivar = np.outer(rng.uniform(1, 16, BINS.size), 1 + 0.5 * np.sin(PIXELS))
    mask = np.zeros(ivar.shape, bool)
    for s, pixels, in_mask in BAD:
        if in_mask:
            mask[s, pixels] = True
        else:
            ivar[s, pixels] = 0.0
    a = scale(np.ones(ivar.shape), ivar, mask)[1]
    roots = [np.linalg.cholesky(dense(C)) for C in cov]
    x = np.array([roots[b] @ rng.standard_normal(NPIX) for b in BINS])
    flux = (x + MEANS[BINS]) / a[:, np.newaxis] + (0.0 if center is None else center)
    flux[mask | (ivar == 0)] = np.nan
    ivar[mask] = np.inf
    ratio = snr(flux, ivar, mask)
    edges = (ratio[BINS == 1].min(),)
    model = Model(MEANS, cov, (100, 100), edges=edges, center=center)
    # side="right" puts the spectrum at the edge in bin 1.
    np.testing.assert_array_equal(np.searchsorted(edges, ratio, side="right"), BINS)
    return model, flux, ivar, mask


def exact(C):
    # The reference covariance: C, or the 60-digit rows of a LowRank, which a dense
    # float64 matrix would round by about kappa eps in the conditionals.
    return C if isinstance(C, np.ndarray) else low_rank(C.vectors, C.values, C.diag)


def tolerance(C, n_cond):
    # Design section 11: 10 kappa eps for the dense route, and 10 eps kappa_W for the
    # Woodbury route, but 1e-9 where n_cond >= 2 K and kappa_W <= 1e8.
    if isinstance(C, np.ndarray):
        return 10 * condition(C) * EPS
    kappa = kappa_w(C.vectors, C.values, C.diag)
    tight = (np.asarray(n_cond) >= 2 * C.values.size) & (kappa <= 1e8)
    return np.where(tight, np.minimum(10 * EPS * kappa, 1e-9), 10 * EPS * kappa)


def rows_of(pixels):
    selected = np.zeros((len(pixels), NPIX), bool)
    for s, row in enumerate(pixels):
        selected[s, row] = True
    return selected


# replace and ignore for the spectra of two_bins. With the bad pixels of BAD, they hold
# runs at both edges (rows 1 and 2), scattered pixels (3 and 4), pixels in both sets
# (1 and 4), bad pixels ignored (0, 1, 4 and 6) and replaced (1, 2 and 3), and a G
# that covers all pixels (3). Rows 0 and 6 of bin 0 share a pattern, rows 2 and 7 of
# bin 1 have equal G and M but not equal T, and rows 5 and 7 have equal patterns in
# different bins.
REPLACE = rows_of(
    [
        [4, 5],
        [0, 1, 2, 5, 6],
        [9, 10, 11],
        [1, 4, 7, 10],
        [2, 6, 9, 11],
        [9, 10, 11],
        [4, 5],
        [9, 10, 11],
    ]
)
IGNORE = rows_of(
    [[8], [2, 3], [7, 8], [0, 2, 3, 5, 6, 8, 9], [5, 6], [7, 8], [0, 1], [7, 8]]
)
# pixelwise and infill, with the options of the tests that run both. infill draws, from
# a new generator at each call, so that its results repeat and every output is an array.
FUNCTIONS = {
    "pixelwise": partial(pixelwise, nguard=2),
    "infill": lambda *args, **kwargs: infill(
        *args, **kwargs, rng=np.random.default_rng(6)
    ),
}
# Singular with a positive diagonal. Its entries are powers of 2, so the Cholesky factor
# at one pixel is exact, and dpotrf finds the leading minor of order 2 exactly 0.
ROOT = 2.0 ** (PIXELS % 3 - 1)
SINGULAR = np.outer(ROOT, ROOT)


def reference(model, b, nguard, flux, ivar, mask):
    # pixelwise of the spectra of bin b, from the 60-digit conditional of each window.
    center = 0.0 if model.center is None else model.center
    X, a, _ = scale(flux, ivar, mask, center=center, refmean=model.mean[b])
    xhat, var = guard_conditional(exact(model.cov[b]), nguard, X, range(NPIX))
    f, v = unscale(xhat, a, model.mean[b], center=center, var=var)
    return X, f, v, (X - xhat) / np.sqrt(var)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("nguard", [0, 2, NPIX - 2, NPIX - 1])
@pytest.mark.parametrize("kappa", [1e2, 1e8])
@pytest.mark.parametrize("center", [None, 0.9])
@pytest.mark.parametrize("rank", [None, 2], ids=["dense", "low-rank"])
def test_pixelwise_matches_the_60_digit_reference(rank, center, kappa, nguard):
    model, flux, ivar, mask = two_bins(kappa, center, rank)
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
        n_cond = [NPIX - len(window(i, nguard, NPIX)) for i in range(NPIX)]
        tol = tolerance(model.cov[b], n_cond)
        L = np.linalg.cholesky(dense(model.cov[b]))
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
    _, flux, ivar, mask = two_bins(1e2, center=0.9)
    prior = Model(MEANS[:1], (SINGULAR,), (100,), center=0.9)
    result = pixelwise(prior, flux, ivar, mask, nguard=NPIX - 1)
    X, a, _ = scale(flux, ivar, mask, center=0.9, refmean=MEANS[0])
    assert same(result.flux, MEANS[0] / a[:, np.newaxis] + 0.9)
    assert same(result.var, np.diagonal(SINGULAR) / np.square(a)[:, np.newaxis])
    expected_z = np.where(mask | (ivar == 0), np.nan, X / ROOT)
    assert same(result.z, expected_z)
    with pytest.raises(LinAlgError, match=r"^model: cov\[0\] is not positive definite"):
        pixelwise(prior, flux, ivar, mask, nguard=0)


# A LowRank whose kappa_W is 1 + 2 x 5e12, above the limit of decision A28.
ILL = LowRank(np.eye(NPIX, 1), [5e12], np.ones(NPIX))


@pytest.mark.parametrize("name", FUNCTIONS)
@pytest.mark.parametrize("failure", ["singular", "kappa_W", "inflate"])
def test_a_bin_that_fails_a_check_raises_before_any_chunk(monkeypatch, name, failure):
    monkeypatch.setattr(_spectra, "CHUNK", 2 * NPIX)
    # The bins of a model hold covariances of one kind.
    model, flux, ivar, mask = two_bins(1e2, rank=2 if failure == "kappa_W" else None)
    inflate = failure == "inflate"
    if inflate:
        # With N_s 5 in bin 1, N_s - N_k - 3 <= 0 for N_k >= 2: pixel 0 of pixelwise
        # has N_k 12 - 3, and row 1, row 4 below, replaces its 3 bad pixels.
        model = dataclasses.replace(model, n_train=(100, 5))
        where = {"pixelwise": "bin 1, pixel 0", "infill": "row 4"}[name]
        error, message = ValueError, rf"^inflate: .* N_s = 5 and N_k = 9 at {where}$"
    elif failure == "singular":
        model = dataclasses.replace(model, cov=(model.cov[0], SINGULAR))
        error, message = LinAlgError, r"^model: cov\[1\] is not positive definite"
    else:
        model = dataclasses.replace(model, cov=(model.cov[0], ILL))
        error = LinAlgError
        message = r"^model: cov\[1\] is numerically singular: kappa_W = 1e\+13 "
    # The spectra of bin 0 first, so that the chunks of bin 1 come last.
    rows = np.argsort(BINS, kind="stable")
    calls = []
    with pytest.raises(error, match=message):
        FUNCTIONS[name](
            model,
            flux[rows],
            ivar[rows],
            mask[rows],
            inflate=inflate,
            progress=lambda *args: calls.append(args),
        )
    assert calls == []
    # The spectra of bin 0 alone never need bin 1.
    rows = BINS == 0
    FUNCTIONS[name](model, flux[rows], ivar[rows], mask[rows], inflate=inflate)


def test_a_guard_window_that_fails_its_factorization_names_the_bin(monkeypatch):
    # A precision matrix that passed its check, but is not positive definite on the
    # window of pixel 7, as rounding can make it near the rcond limit.
    P = np.eye(NPIX)
    P[7, 7] = -1.0
    monkeypatch.setattr(_model, "precision", lambda S: P)
    model, flux, ivar, mask = two_bins(1e2)
    with pytest.raises(LinAlgError, match=r"^model: cov\[0\] is .* pixel 7$"):
        pixelwise(Model(MEANS[:1], model.cov[:1], (100,)), flux, ivar, mask, nguard=0)


# The options of the tests that run both functions with replace and ignore.
SETS = {"pixelwise": {}, "infill": {"replace": REPLACE, "ignore": IGNORE}}


@pytest.mark.parametrize("name", FUNCTIONS)
def test_inputs_are_unchanged_and_outputs_are_read_only_and_not_shared(name):
    model, flux, ivar, mask = two_bins(1e2)

    def outputs(*spectra, **sets):
        return arrays(FUNCTIONS[name](model, *spectra, **sets))

    result = checked(outputs, flux, ivar, mask, **SETS[name])
    assert not any(output.flags.writeable for output in result)


@pytest.mark.parametrize("name", FUNCTIONS)
def test_float32_and_f_ordered_inputs_give_the_result_of_a_float64_c_copy(name):
    model, flux, ivar, mask = two_bins(1e2, center=0.9)
    function = partial(FUNCTIONS[name], model, **SETS[name])
    flux32, ivar32 = flux.astype(np.float32), ivar.astype(np.float32)
    cast = function(flux32.astype(float), ivar32.astype(float), mask)
    assert same(arrays(function(flux32, ivar32, mask)), arrays(cast))
    fortran = [np.asfortranarray(array) for array in (flux, ivar, mask)]
    assert same(arrays(function(*fortran)), arrays(function(flux, ivar, mask)))


@pytest.mark.parametrize("name", FUNCTIONS)
def test_progress_counts_the_spectra_done_after_each_chunk(monkeypatch, name):
    monkeypatch.setattr(_spectra, "CHUNK", 2 * NPIX)
    model, flux, ivar, mask = two_bins(1e2)
    calls = []
    FUNCTIONS[name](
        model, flux[:5], ivar[:5], mask[:5], progress=lambda *args: calls.append(args)
    )
    assert calls == [(name, 2, 5), (name, 4, 5), (name, 5, 5)]


# P, the guard weights of pixelwise and the call.
@pytest.mark.parametrize(("name", "count"), [("pixelwise", 3), ("infill", 2)])
def test_a_call_on_a_new_model_logs_its_timings_at_debug(caplog, name, count):
    caplog.set_level(logging.DEBUG, logger="gspice")
    model, flux, ivar, mask = two_bins(1e2)
    one_bin = Model(MEANS[:1], model.cov[:1], (100,))
    FUNCTIONS[name](one_bin, flux, ivar, mask)
    records = [(record.name, record.levelno) for record in caplog.records]
    assert records == [("gspice", logging.DEBUG)] * count
    handlers = logging.getLogger("gspice").handlers
    assert any(isinstance(handler, logging.NullHandler) for handler in handlers)


@pytest.mark.parametrize(
    ("function", "expected"),
    [
        (partial(pixelwise, nguard=0), [((0, 3), np.float64)] * 3),
        (
            FUNCTIONS["infill"],
            [((0, 3), np.float64)] * 2
            + [((0, 3), np.bool_)] * 2
            + [((0,), np.float64), ((0,), np.int64), ((0,), np.float64)]
            + [((0, 3), np.float64)],
        ),
    ],
    ids=["pixelwise", "infill"],
)
def test_zero_spectra_give_empty_arrays_without_a_precision_matrix(function, expected):
    singular = Model(np.zeros((1, 3)), (np.ones((3, 3)),), (5,))
    empty = np.zeros((0, 3))
    calls = []
    result = function(singular, empty, empty, progress=lambda *args: calls.append(args))
    assert [(a.shape, a.dtype) for a in arrays(result)] == expected
    assert calls == []


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("given", [False, True], ids=["default", "given"])
@pytest.mark.parametrize("kappa", [1e2, 1e8])
@pytest.mark.parametrize("center", [None, 0.9])
@pytest.mark.parametrize("rank", [None, 2], ids=["dense", "low-rank"])
def test_infill_matches_the_60_digit_reference(rank, center, kappa, given):
    model, flux, ivar, mask = two_bins(kappa, center, rank)
    sets = {"replace": REPLACE, "ignore": IGNORE} if given else {}
    result = infill(model, flux, ivar, mask, **sets, covariance=True)
    bad = mask | (ivar == 0)
    M = REPLACE if given else bad
    G = M | bad | (IGNORE if given else False)
    T = M & ~bad
    np.testing.assert_array_equal(result.replaced, M)
    np.testing.assert_array_equal(result.ignored, G & ~M)
    np.testing.assert_array_equal(result.dof, np.count_nonzero(T, axis=1))
    # Outside M, the input flux, and 1/ivar at good pixels and inf at bad ones.
    var = np.divide(1.0, ivar, out=np.full(ivar.shape, np.inf), where=~bad)
    assert same(result.flux[~M], flux[~M])
    assert same(result.var[~M], var[~M])
    c = 0.0 if center is None else center
    for s, b in enumerate(BINS):
        pixels, cov = result.covariance(s)
        np.testing.assert_array_equal(pixels, np.flatnonzero(M[s]))
        assert same(np.diagonal(cov), result.var[s, pixels])
        if not pixels.size:
            assert result.chi2[s] == 0
            continue
        X, a, _ = scale(flux[[s]], ivar[[s]], mask[[s]], center=c, refmean=MEANS[b])
        g, t = np.flatnonzero(G[s]), np.flatnonzero(T[s])
        mean, reference, chi2 = conditional(exact(model.cov[b]), g, X, t)
        at = np.flatnonzero(M[s, g])
        reference = reference[np.ix_(at, at)]
        f, v = unscale(
            mean[:, at], a, MEANS[b, pixels], center=c, var=np.diagonal(reference)
        )
        # Design section 11: |dmu| <= tol sigma, |dvar| <= tol var and
        # |dchi2| <= tol (chi2 + |T|), with tol = 10 kappa eps. The mean reads only x_O,
        # a draw from N(0, cov[b]), so the Mahalanobis factor is 1. An entry of the
        # covariance errs by at most sigma_i sigma_j times the normwise error of the
        # correlation matrix, within tol as test_linalg checks for the pattern solve.
        # The two evaluations of (mu + m)/a + c round three times each, by at most
        # eps (3|f| + 2|c|) in all to first order; those of var/a^2 and cov/a^2 round
        # once each, with the same a^2, and |cov_ij| <= sigma_i sigma_j.
        tol = tolerance(model.cov[b], NPIX - g.size)
        sigma = np.sqrt(v[0])
        rounding = 4 * EPS * (np.abs(f[0]) + abs(c))
        assert np.all(np.abs(result.flux[s, pixels] - f[0]) <= tol * sigma + rounding)
        assert np.all(np.abs(result.var[s, pixels] - v[0]) <= (tol + EPS) * v[0])
        error = np.abs(cov - reference / a[0] ** 2)
        assert np.all(error <= (tol + EPS) * np.outer(sigma, sigma))
        assert abs(result.chi2[s] - chi2[0]) <= tol * (chi2[0] + t.size)


# The seed of the normals of the tests that compare draws with REPLACE and IGNORE, and
# the bound in sigma on the difference of two triangular solves of them. A draw adds
# R_MM^-T z_M to the mean. The solve errs by at most
# gamma_m sqrt(m) kappa(R_MM) sqrt(lambda_max(C_MM)) |z_M| in the 2-norm, with
# gamma_m = m u / (1 - m u), u = 2^-53 (Higham 2002, Theorem 8.5) and
# kappa(R_MM)^2 = kappa(C_MM). In the patterns of two_bins(1e2), m <= 5,
# kappa(C_MM) <= 18 and sqrt(lambda_max(C_MM)) <= 2.9 sigma, and with this seed
# |z_M| < 3.5. Two solves then differ by at most
# 2 gamma_5 sqrt(5) sqrt(18) 2.9 x 3.5 sigma = 1.1e-13 sigma.
DRAW_SEED, DRAW_TOL = 8, 1.1e-13


def test_slices_and_chunks_of_infill_agree_with_one_call_to_rounding(monkeypatch):
    model, flux, ivar, mask = two_bins(1e2, center=0.9)

    def repaired(rows, rng):
        spectra = (array[rows] for array in (flux, ivar, mask))
        sets = {"replace": REPLACE[rows], "ignore": IGNORE[rows]}
        return infill(model, *spectra, **sets, covariance=True, rng=rng)

    whole = repaired(slice(None), np.random.default_rng(DRAW_SEED))
    # The pattern of rows 0 and 6 spans slices and chunks, and each row alone is a
    # grouping of its own (design section 11). Consecutive slices share a generator,
    # so that they draw the normals of one call.
    singles = [slice(s, s + 1) for s in range(len(flux))]
    results = []
    for spans in ((slice(0, 3), slice(3, 4), slice(4, None)), singles):
        rng = np.random.default_rng(DRAW_SEED)
        results += [(rows, repaired(rows, rng)) for rows in spans]
    monkeypatch.setattr(_spectra, "CHUNK", 3 * NPIX)
    rng = np.random.default_rng(DRAW_SEED)
    results.append((slice(None), repaired(slice(None), rng)))
    # Every grouping factorizes the same P_GG, which gives the variances and
    # covariances, and then divides them by the same a^2. The means and chi2 also come
    # from products and solves with other numbers of spectra, whose rounding can
    # differ; each lies within the tolerance of the reference test, tol = 10 kappa eps,
    # so two lie within 2 tol, and flux rounds as there.
    tol = 10 * max(map(condition, model.cov)) * EPS
    sigma = np.sqrt(whole.var)
    dflux = 2 * tol * sigma + 4 * EPS * (np.abs(whole.flux) + 0.9)
    dchi2 = 2 * tol * (whole.chi2 + whole.dof)
    # Every grouping draws with the same R_MM and normals z_M, so two draws agree
    # within DRAW_TOL sigma and the 2 tol sigma of the means, and round as flux does
    # (design section 11).
    ddraws = (2 * tol + DRAW_TOL) * sigma + 4 * EPS * (np.abs(whole.draws) + 0.9)
    for rows, result in results:
        for name in ("var", "replaced", "ignored", "dof"):
            assert same(getattr(result, name), getattr(whole, name)[rows])
        M = whole.replaced[rows]
        assert same(result.flux[~M], whole.flux[rows][~M])
        assert np.all(np.abs(result.flux[M] - whole.flux[rows][M]) <= dflux[rows][M])
        error = np.abs(result.draws[M] - whole.draws[rows][M])
        assert np.all(error <= ddraws[rows][M])
        assert np.all(np.abs(result.chi2 - whole.chi2[rows]) <= dchi2[rows])
        for s, row in enumerate(range(len(flux))[rows]):
            assert same(result.covariance(s), whole.covariance(row))


@pytest.mark.parametrize("rows", [len(BINS), 2])
def test_each_spectrum_draws_from_its_own_row_of_normals_for_any_chunk_size(
    monkeypatch, rows
):
    monkeypatch.setattr(_spectra, "CHUNK", rows * NPIX)
    model, flux, ivar, mask = two_bins(1e2, center=0.9)
    # With P = I, the conditional mean is 0, and the draw of a pixel in working units
    # is its normal, exactly; row 3 has a G that covers all pixels, and rows 4 and 5,
    # one chunk of 2 rows, have no pixel to replace.
    identity = Model(
        MEANS, (np.eye(NPIX),) * 2, (100, 100), edges=model.edges, center=0.9
    )
    replace = REPLACE.copy()
    replace[[4, 5]] = False
    rng = np.random.default_rng(9)
    result = infill(identity, flux, ivar, mask, replace=replace, ignore=IGNORE, rng=rng)
    reference = np.random.default_rng(9)
    normals = reference.standard_normal(flux.shape)
    # Nothing else draws from rng.
    assert rng.bit_generator.state == reference.bit_generator.state
    _, a, _ = scale(flux, ivar, mask)
    drawn = (normals + MEANS[BINS]) / a[:, np.newaxis] + 0.9
    assert same(result.draws, np.where(replace, drawn, result.flux))


def test_rng_changes_no_output_but_the_draws():
    model, flux, ivar, mask = two_bins(1e2)
    sets = {"replace": REPLACE, "ignore": IGNORE}
    plain = infill(model, flux, ivar, mask, **sets)
    drawn = infill(model, flux, ivar, mask, **sets, rng=np.random.default_rng(0))
    assert same(arrays(dataclasses.replace(drawn, draws=None)), arrays(plain))


def test_a_draw_minus_its_mean_does_not_depend_on_the_flux():
    # draws - flux is F z_M / a at the replaced pixels: it depends on the pattern, ivar
    # and the normals, not on the data. One bin, so that the S/N chooses no other P.
    _, flux, ivar, mask = two_bins(1e2)
    model = Model(MEANS[:1], (random_spd(NPIX, 1e2, 0),), (100,), center=0.9)
    sets = {"replace": REPLACE, "ignore": IGNORE}
    results = [
        infill(model, f, ivar, mask, **sets, rng=np.random.default_rng(5))
        for f in (flux, 2 * flux)
    ]
    M = REPLACE
    spread = [(r.draws - r.flux)[M] for r in results]
    # Each draw and each mean rounds about 4 times, each time by at most eps of its
    # size plus the centre.
    size = sum(np.abs(r.draws[M]) + np.abs(r.flux[M]) + 0.9 for r in results)
    assert np.all(np.abs(spread[0] - spread[1]) <= 4 * EPS * size)


@pytest.mark.slow
@pytest.mark.parametrize("prior", [False, True], ids=["pattern", "prior"])
def test_the_draws_follow_the_conditional_distribution(prior):
    n = 20_000
    C = random_spd(NPIX, 1e2, 0)
    rng = np.random.default_rng(10)
    x = rng.standard_normal((n, NPIX)) @ np.linalg.cholesky(C).T
    # a = 2 in every spectrum, and pixel 6 is bad in all, so they share one pattern.
    ivar = np.full((n, NPIX), 4.0)
    mask = PIXELS == 6
    replace = np.isin(PIXELS, [2, 5, 6, 7])
    ignore = ~replace if prior else np.isin(PIXELS, [3, 4, 8])
    result = infill(
        Model(MEANS[:1], (C,), (100,), center=0.9),
        (x + MEANS[0]) / 2 + 0.9,
        ivar,
        np.tile(mask, (n, 1)),
        replace=replace,
        ignore=ignore,
        covariance=True,
        rng=rng,
    )
    pixels, cov = result.covariance(0)
    # The draws follow N(flux, cov) at the replaced pixels.
    assert_gaussian(result.draws[:, pixels], result.flux[:, pixels], cov)


@pytest.mark.parametrize("nguard", [0, 2])
def test_infill_of_a_pixel_with_its_guard_window_ignored_equals_pixelwise(nguard):
    # Design section 11, on rows 5 and 7, which have no bad pixels.
    model, flux, ivar, mask = two_bins(1e8, center=0.9)
    spectra = flux[[5, 7]], ivar[[5, 7]], mask[[5, 7]]
    predicted = pixelwise(model, *spectra, nguard=nguard)
    # Each result lies within the tolerances of its reference test, 10 kappa eps in
    # |dmu|/sigma, |dvar|/var and |dchi2|/(chi2 + 1), so the two lie within twice them
    # of each other, plus the rounding of flux and var there.
    tol = 2 * 10 * max(map(condition, model.cov)) * EPS
    sigma = np.sqrt(predicted.var)
    rounding = 8 * EPS * (np.abs(predicted.flux) + 0.9)
    for i in range(NPIX):
        replace = PIXELS == i
        ignore = (np.abs(PIXELS - i) <= nguard) & ~replace
        result = infill(model, *spectra, replace=replace, ignore=ignore)
        error = np.abs(result.flux[:, i] - predicted.flux[:, i])
        assert np.all(error <= tol * sigma[:, i] + rounding[:, i])
        var = predicted.var[:, i]
        assert np.all(np.abs(result.var[:, i] - var) <= (tol + 2 * EPS) * var)
        z2 = predicted.z[:, i] ** 2
        assert np.all(np.abs(result.chi2 - z2) <= tol * (z2 + 1))


def test_an_npix_mask_gives_the_result_of_that_mask_in_every_row():
    model, flux, ivar, mask = two_bins(1e2)
    replace, ignore = REPLACE[4], IGNORE[4]
    rows = (len(flux), 1)
    expected = infill(
        model,
        flux,
        ivar,
        mask,
        replace=np.tile(replace, rows),
        ignore=np.tile(ignore, rows),
    )
    result = infill(model, flux, ivar, mask, replace=replace, ignore=ignore)
    assert same(arrays(result), arrays(expected))


def test_infill_where_g_covers_all_pixels_needs_no_precision_matrix():
    _, flux, ivar, mask = two_bins(1e2, center=0.9)
    bad = mask | (ivar == 0)
    prior = Model(MEANS[:1], (SINGULAR,), (100,), center=0.9)
    # Each spectrum replaces its bad pixels and pixel 2, which is good in all, and
    # ignores the other pixels: G covers all pixels, and T is pixel 2.
    replace = bad | (PIXELS == 2)
    result = infill(
        prior, flux, ivar, mask, replace=replace, ignore=~replace, covariance=True
    )
    X, a, _ = scale(flux, ivar, mask, center=0.9, refmean=MEANS[0])
    a2 = np.square(a)[:, np.newaxis]
    assert same(result.flux[replace], (MEANS[0] / a[:, np.newaxis] + 0.9)[replace])
    assert same(result.var[replace], (np.diagonal(SINGULAR) / a2)[replace])
    assert same(result.chi2, (X[:, 2] / ROOT[2]) ** 2)
    for s in range(len(flux)):
        pixels, cov = result.covariance(s)
        assert same(cov, SINGULAR[np.ix_(pixels, pixels)] / a2[s])
    # Draws factor C at M, which is singular in the rows with a bad pixel.
    rng = np.random.default_rng(0)
    with pytest.raises(LinAlgError, match="on the pixels M, for the pattern of row"):
        infill(prior, flux, ivar, mask, replace=replace, ignore=~replace, rng=rng)
    # Pixels 2 and 4 of row 6: C at T is singular.
    replace[6, 4] = True
    error = r"^model: cov\[0\] is not positive definite on the pixels T, for the pattern of row 6$"
    with pytest.raises(LinAlgError, match=error):
        infill(prior, flux, ivar, mask, replace=replace, ignore=~replace)
    # Without a pixel to replace, no spectrum needs P.
    infill(prior, flux, ivar, mask, replace=np.zeros(flux.shape, bool))


def test_a_pattern_whose_factorization_fails_names_the_bin_and_its_first_row(
    monkeypatch,
):
    # A precision matrix that passed its check, but is not positive definite at pixel
    # 8, as rounding can make it near the rcond limit.
    P = np.eye(NPIX)
    P[8, 8] = -1.0
    monkeypatch.setattr(_model, "precision", lambda S: P)
    # Rows 4 to 7 form the second chunk, and rows 5 and 7 hold the only pattern with
    # pixel 8 in G and a pixel to replace.
    monkeypatch.setattr(_spectra, "CHUNK", 4 * NPIX)
    model, flux, ivar, mask = two_bins(1e2)
    replace = np.zeros(flux.shape, bool)
    replace[[5, 7], 8] = True
    error = (
        r"^model: cov\[0\] is not positive definite in its precision matrix on the "
        r"pixels G, for the pattern of row 5$"
    )
    with pytest.raises(LinAlgError, match=error):
        infill(
            Model(MEANS[:1], model.cov[:1], (100,)), flux, ivar, mask, replace=replace
        )


def test_covariance_needs_a_result_computed_with_covariance_true():
    model, flux, ivar, mask = two_bins(1e2)
    result = infill(model, flux, ivar, mask)
    with pytest.raises(ValueError, match="covariance=True"):
        result.covariance(0)


def heavy_tailed(nspec, rng):
    # Spectra without bad pixels whose signal and noise follow Student's t with 3
    # degrees of freedom, which has no fourth moment.
    basis = np.array([np.ones(NPIX), np.cos(PIXELS / 2), np.sin(PIXELS / 3)])
    ivar = np.outer(rng.uniform(100, 400, nspec), 1 + 0.5 * np.sin(PIXELS))
    flux = 1 + 0.05 * rng.standard_t(3, (nspec, 3)) @ basis
    flux += rng.standard_t(3, (nspec, NPIX)) / np.sqrt(ivar)
    return flux, ivar


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("center", [None, 0.9])
def test_the_in_sample_mean_of_z2_is_exact_for_any_data(center, weighted):
    # Over the N_s spectra of a fit without clipping, the w-weighted mean of Z^2 / N_*
    # is 1 - V2 / V1^2 for a mean-centred model, which is (N_s - 1) / N_s for unit
    # weights, and 1 for a constant centre, whatever the distribution of the data
    # (paper l.249, design section 11): at each pixel of pixelwise, and over the
    # spectra of one infill pattern.
    rng = np.random.default_rng(11)
    nspec = 40
    flux, ivar = heavy_tailed(nspec, rng)
    weights = rng.uniform(0.5, 2, nspec) if weighted else None
    model, _ = fit(flux, ivar, nsigma=(), center=center, weights=weights)
    w = np.ones(nspec) if weights is None else weights
    expected = 1 - w @ w / w.sum() ** 2 if center is None else 1.0
    # The identity is exact in exact arithmetic, and its rounding error grows with
    # kappa: 8.9e-9 at kappa 1.3e10 (design section 11), which scales to
    # 8.9e-9 x 1e4 / 1.3e10 = 6.8e-15 at kappa 1e4.
    assert condition(model.cov[0]) < 1e4
    z = pixelwise(model, flux, ivar, nguard=2).z
    assert np.all(np.abs(w @ z**2 / w.sum() - expected) <= 1e-12)
    result = infill(model, flux, ivar, replace=PIXELS % 4 == 1, ignore=PIXELS % 4 == 2)
    assert abs(w @ (result.chi2 / result.dof) / w.sum() - expected) <= 1e-12


@pytest.mark.parametrize("nbin", [1, 2])
@pytest.mark.parametrize("nguard", [2, NPIX - 1])
def test_inflate_multiplies_the_variance_of_each_pixel_by_its_f_s(nguard, nbin):
    model, flux, ivar, mask = two_bins(1e2)
    # The first nbin bins, with an N_s that differs between them.
    model = Model(
        MEANS[:nbin], model.cov[:nbin], (30, 60)[:nbin], edges=model.edges[: nbin - 1]
    )
    plain, inflated = (
        pixelwise(model, flux, ivar, mask, nguard=nguard, inflate=inflate)
        for inflate in (False, True)
    )
    # N_k = npix - |g|, with the window g of each pixel clipped at the ends.
    size = [len(window(i, nguard, NPIX)) for i in PIXELS]
    n_train = np.array(model.n_train)[BINS if nbin == 2 else 0]
    f = finite_sample_factor(np.reshape(n_train, (-1, 1)), NPIX - np.array(size))
    assert same(inflated.flux, plain.flux)
    # Each side of var rounds twice and each side of z at most four times, by at most
    # eps/2 each time: 2 eps and 3.25 eps to first order.
    np.testing.assert_allclose(inflated.var, plain.var * f, rtol=4 * EPS, atol=0)
    np.testing.assert_allclose(inflated.z, plain.z / np.sqrt(f), rtol=4 * EPS, atol=0)


def test_inflate_scales_the_infill_variances_covariances_draws_and_chi2_by_f_s():
    model, flux, ivar, mask = two_bins(1e2)
    model = dataclasses.replace(model, n_train=(30, 60))
    sets = {"replace": REPLACE, "ignore": IGNORE, "covariance": True}
    plain, inflated = (
        infill(
            model,
            flux,
            ivar,
            mask,
            **sets,
            inflate=inflate,
            rng=np.random.default_rng(DRAW_SEED),
        )
        for inflate in (False, True)
    )
    # N_k = npix - |G| of each spectrum.
    G = REPLACE | IGNORE | mask | (ivar == 0)
    f = finite_sample_factor(np.array(model.n_train)[BINS], NPIX - G.sum(axis=1))
    M = REPLACE
    assert same(inflated.flux, plain.flux)
    assert same(inflated.var[~M], plain.var[~M])
    # Each side of var and cov rounds twice, and each side of chi2 once.
    expected = plain.var * f[:, np.newaxis]
    np.testing.assert_allclose(inflated.var[M], expected[M], rtol=4 * EPS, atol=0)
    np.testing.assert_allclose(inflated.chi2, plain.chi2 / f, rtol=2 * EPS, atol=0)
    for s in range(len(flux)):
        pixels, cov = inflated.covariance(s)
        expected = plain.covariance(s)[1] * f[s]
        np.testing.assert_allclose(cov, expected, rtol=4 * EPS, atol=0)
        assert same(np.diagonal(cov), inflated.var[s, pixels])
    # The deviation of a draw from the mean grows by f^1/2. Both solves, R_MM^-T of
    # the normals times f^1/2 and f^1/2 times R_MM^-T of the normals, err by at most
    # DRAW_TOL / 2 sigma times f^1/2, and flux rounds as in the test of slices and
    # chunks.
    root = np.sqrt(f)[:, np.newaxis]
    deviation = inflated.draws - inflated.flux
    scaled = root * (plain.draws - plain.flux)
    size = np.abs(inflated.draws) + np.abs(plain.flux)
    size += root * (np.abs(plain.draws) + np.abs(plain.flux))
    tol = root * DRAW_TOL * np.sqrt(plain.var) + 4 * EPS * size
    assert np.all(np.abs(deviation - scaled)[M] <= tol[M])


def test_fs_is_f_s_of_each_spectrum_over_its_inflation_factor_with_inflate():
    model, flux, ivar, mask = two_bins(1e2)
    model = dataclasses.replace(model, n_train=(11, 60))
    # Row 2 replaces only its bad pixel, so its dof is 0. Row 5 replaces no pixel, so
    # its f_s(N_s, N_k, 1), undefined at N_s 11 and N_k 10, is not needed. With N_s 11,
    # rows 0 and 6 have N_s - N_k - dof - 2 = 0; row 3, whose G covers all pixels,
    # has N_k 0.
    replace = REPLACE.copy()
    replace[2] = PIXELS == 9
    replace[5] = False
    plain, inflated = (
        infill(model, flux, ivar, mask, replace=replace, ignore=IGNORE, inflate=inflate)
        for inflate in (False, True)
    )
    bad = mask | (ivar == 0)
    n_train = np.array(model.n_train)[BINS]
    n_cond = NPIX - np.count_nonzero(replace | IGNORE | bad, axis=1)
    dof = np.count_nonzero(replace & ~bad, axis=1)

    def factor(n_pred):
        return np.array(
            [
                finite_sample_factor(N, k, m) if m > 0 and N - k - m - 2 > 0 else np.nan
                for N, k, m in zip(n_train, n_cond, n_pred, strict=True)
            ]
        )

    np.testing.assert_array_equal(plain.fs, factor(dof))
    np.testing.assert_array_equal(inflated.fs, factor(dof) / factor(np.ones(8, int)))


@pytest.mark.parametrize(
    "kind",
    [{"center": 0.9}, {"weighted": True}, {"cov": two_bins(1e2, rank=2)[0].cov}],
    ids=["center", "weighted", "low-rank"],
)
def test_fs_is_1_for_a_model_without_a_derived_factor(kind):
    model, flux, ivar, mask = two_bins(1e2)
    # As in the test above, with rows of dof 0 and an undefined f_s(N_s, N_k, dof).
    replace = REPLACE.copy()
    replace[5] = False
    model = dataclasses.replace(model, n_train=(11, 60), **kind)
    result = infill(model, flux, ivar, mask, replace=replace)
    assert same(result.fs, np.ones(len(flux)))
