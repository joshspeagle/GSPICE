import copy
import pickle
from dataclasses import FrozenInstanceError
from fractions import Fraction
from functools import partial

import numpy as np
import pytest
from numpy.linalg import LinAlgError
from scipy import stats

from _golden import checked, same
from gspice import (
    Calibration,
    LowRank,
    Model,
    _model,
    _spectra,
    calibration,
    finite_sample_factor,
    fit,
    infill,
    noise_audit,
    pixelwise,
    simulate,
)
from oracle import random_low_rank, random_spd

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


# z2, n and fs of 5 spectra in 2 columns, and the entries that the methods use: not
# those with n = 0 (rows 2 and 4), a NaN z2 (row 1) or a NaN fs (rows 2 and 3). The
# constructor accepts z2 = 0, at row 2.
Z2 = np.array([[1.0, 4.0], [2.0, np.nan], [0.0, 9.0], [3.0, 1.0], [7.0, 2.0]])
N = np.array([[1, 2], [1, 2], [0, 2], [1, 2], [1, 0]])
FS = np.array([[1.0, 1.5], [1.25, 1.5], [1.0, np.nan], [np.nan, 1.5], [1.1, 1.5]])
USED = np.array([[1, 1], [1, 0], [0, 0], [0, 1], [1, 0]], bool)
# None, then 1 entry in column 0 and none in column 1.
WHERE = [None, np.array([False, True, False, False, False])]


def used(column, where):
    # z2 / fs and n at the entries of column that the methods use, in the rows of where.
    rows = USED[:, column] & (True if where is None else where)
    return Z2[rows, column] / FS[rows, column], N[rows, column]


# An error for the warning of a division by m - 1 = 0, which where avoids.
@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("where", WHERE, ids=["all", "row-1"])
def test_ratio_is_the_mean_of_z2_over_n_fs_and_its_standard_error(where):
    mean, sem = Calibration(Z2, N, FS).ratio(where)
    for j in (0, 1):
        x, n = used(j, where)
        m = x.size
        values = x / n
        # Each side rounds the mean of at most 3 values a few times, and the standard
        # error twice more.
        if m == 0:
            assert np.isnan(mean[j])
        else:
            assert mean[j] == pytest.approx(values.mean(), rel=4 * EPS)
        if m < 2:
            assert np.isnan(sem[j])
        else:
            assert sem[j] == pytest.approx(values.std(ddof=1) / np.sqrt(m), rel=4 * EPS)


@pytest.mark.parametrize("where", WHERE, ids=["all", "row-1"])
@pytest.mark.parametrize("q", [0.5, 0.9])
def test_tail_is_the_fraction_above_the_chi2_quantile_of_each_n(q, where):
    # At q = 0.5, the z2 / fs of column 1, 2.67 and 0.67, lie on either side of the
    # median of chi^2_2, 1.39, and both above that of chi^2_1, 0.45.
    expected = []
    for j in (0, 1):
        x, n = used(j, where)
        expected.append(np.mean(x > stats.chi2.ppf(q, n)) if x.size else np.nan)
    assert same(Calibration(Z2, N, FS).tail(q, where), np.array(expected))
    # An entry at the quantile is not above it.
    assert Calibration(z2=[[stats.chi2.ppf(q, 1)]], n=1, fs=1).tail(q) == 0


@pytest.mark.parametrize("where", WHERE, ids=["all", "row-1"])
def test_qq_gives_the_normal_scores_of_the_used_entries(where):
    for i in (0, 1):
        expected, observed = Calibration(Z2, N, FS).qq(i, where)
        x, n = used(i, where)
        m = x.size
        assert same(expected, stats.norm.ppf((np.arange(1, m + 1) - 0.5) / m))
        # Away from the tails, Phi^-1(F) and -Phi^-1(1 - F) of scipy agree to within
        # 1.8e-14 relative here.
        scores = np.sort(stats.norm.ppf(stats.chi2.cdf(x, n)))
        np.testing.assert_allclose(observed, scores, rtol=1e-12, atol=0)


def test_qq_scores_keep_their_precision_in_both_tails():
    # For n = 2, F(x) = 1 - exp(-x/2) and 1 - F(x) = exp(-x/2): F rounds to 1 at
    # x = 200, and 1 - F to 1 at x = 1e-20. z2 = 2 x with fs = 2.
    x = np.array([200.0, 3.0, 0.5, 1e-20])
    _, observed = Calibration(z2=2 * x[:, np.newaxis], n=2, fs=2).qq()
    cdf = -np.expm1(-x / 2)
    reference = np.where(cdf < 0.5, stats.norm.ppf(cdf), stats.norm.isf(np.exp(-x / 2)))
    np.testing.assert_allclose(observed, np.sort(reference), rtol=1e-12, atol=0)


NPIX = 12
PIXELS = np.arange(NPIX)


def spectra(rank=None):
    # Seven spectra and a model, dense, of the rank, or for rank "noise" a noise model,
    # with ivar from 0.5 to 2, so that C_s differs between spectra. Rows 1 to 5 have
    # bad pixels: an interior run, runs at both ends, isolated pixels, all but 3
    # pixels, and 2 pixels of ivar 0, so that their good pixels number 9, 8, 9, 3 and
    # 10, and 12 in rows 0 and 6.
    rng = np.random.default_rng(14)
    if rank is None or rank == "noise":
        C = random_spd(NPIX, 1e2, 0)
    else:
        C = LowRank(*random_low_rank(NPIX, rank, 1e2, 0))
    model = Model(np.zeros((1, NPIX)), (C,), (30,), noise=rank == "noise")
    flux = rng.standard_normal((7, NPIX))
    ivar = rng.uniform(0.5, 2, (7, NPIX)) if model.noise else np.ones((7, NPIX))
    mask = np.zeros((7, NPIX), bool)
    mask[1, 4:7] = True
    mask[2, [0, 1, 10, 11]] = True
    mask[3, [2, 7, 9]] = True
    mask[4] = ~np.isin(PIXELS, [3, 8, 11])
    ivar[5, [5, 6]] = 0.0
    return model, flux, ivar, mask


def drawn(bad, sizes, layout, rng):
    # The masks M (len(sizes), nspec, npix) that the docstring of calibration states.
    masks = np.zeros((len(sizes), *bad.shape), bool)
    for s, row in enumerate(bad):
        good = np.flatnonzero(~row)
        for j, n in enumerate(sizes):
            if good.size < n:
                continue
            if layout == "run":
                ranks = rng.integers(good.size - n + 1) + np.arange(n)
            else:
                ranks = rng.choice(good.size, n, replace=False)
            masks[j, s, good[ranks]] = True
    return masks


def fields(result):
    return result.z2, result.n, result.fs


@pytest.mark.parametrize("rows", [None, 2], ids=["one-chunk", "2-row-chunks"])
@pytest.mark.parametrize("layout", ["run", "scattered"])
@pytest.mark.parametrize("rank", [None, 3, "noise"], ids=["dense", "low-rank", "noise"])
def test_calibration_is_infill_of_the_documented_masks_for_any_chunk_size(
    monkeypatch, rank, layout, rows
):
    if rows is not None:
        monkeypatch.setattr(_spectra, "CHUNK", rows * NPIX)
    model, flux, ivar, mask = spectra(rank)
    # Size 9 covers every good pixel of rows 1 and 3, so that G covers all pixels and
    # the run crosses their bad pixels, and is above the 8 and 3 good pixels of rows 2
    # and 4.
    sizes = (1, 3, 9)

    def calibrated(*arrays):
        rng = np.random.default_rng(15)
        return fields(calibration(model, *arrays, sizes=sizes, layout=layout, rng=rng))

    z2, n, fs = checked(calibrated, flux, ivar, mask)
    masks = drawn(mask | (ivar == 0), sizes, layout, np.random.default_rng(15))
    for j, replace in enumerate(masks):
        # With the chunk size of calibration, infill groups the same spectra.
        expected = infill(model, flux, ivar, mask, replace=replace)
        assert same(z2[:, j], np.where(expected.dof > 0, expected.chi2, np.nan))
        assert same(n[:, j], expected.dof)
        assert same(fs[:, j], expected.fs)
    # Rows 2 and 4 have fewer good pixels than 9.
    assert np.array_equal(np.argwhere(n == 0), [[2, 2], [4, 2]])


@pytest.mark.parametrize(
    ("good", "sizes"),
    [
        ((12, 12, 12, 1, 7, 7, 7), (1,)),
        ((12, 12, 12, 1, 8, 8, 8), (1, 4)),
        ((12, 8, 7, 1), (1,)),
        ((12, 9, 7, 1), (1, 4)),
    ],
    ids=["odd-median-7", "odd-median-8", "even-median-7.5", "even-median-8"],
)
def test_the_default_sizes_are_at_most_half_the_median_good_pixel_count(good, sizes):
    # A row with one good pixel keeps the defaults; under a rule on the fewest good
    # pixels it would leave none.
    model, flux, *_ = spectra()
    flux, ivar = flux[: len(good)], np.ones((len(good), NPIX))
    mask = PIXELS >= np.array(good)[:, np.newaxis]
    results = [
        fields(
            calibration(
                model, flux, ivar, mask, sizes=given, rng=np.random.default_rng(16)
            )
        )
        for given in (None, sizes)
    ]
    assert same(*results)


def test_a_spectrum_with_few_good_pixels_leaves_the_default_sizes_of_the_others():
    # Rows 0 to 6 have 12, 9, 8, 9, 3, 10 and 12 good pixels: the median 9 keeps size
    # 4, which row 4 cannot fill.
    model, flux, ivar, mask = spectra()
    result = calibration(model, flux, ivar, mask, rng=np.random.default_rng(16))
    assert np.array_equal(result.n, [[1, 4]] * 4 + [[1, 0]] + [[1, 4]] * 2)
    assert np.array_equal(np.isnan(result.z2), result.n == 0)


def test_zero_spectra_give_all_six_default_sizes():
    model, *_ = spectra()
    empty = np.zeros((0, NPIX))
    result = calibration(model, empty, empty, rng=np.random.default_rng(0))
    assert [a.shape for a in fields(result)] == [(0, 6)] * 3


def test_progress_counts_the_spectra_done_after_each_chunk_of_calibration(
    monkeypatch,
):
    monkeypatch.setattr(_spectra, "CHUNK", 3 * NPIX)
    model, flux, ivar, mask = spectra()
    calls = []
    calibration(
        model,
        flux,
        ivar,
        mask,
        sizes=(1,),
        rng=np.random.default_rng(0),
        progress=lambda *args: calls.append(args),
    )
    assert calls == [
        ("calibration", 3, 7),
        ("calibration", 6, 7),
        ("calibration", 7, 7),
    ]


def test_a_covariance_that_fails_its_check_raises_before_calibration_starts(
    monkeypatch,
):
    # cov[1] has no inverse, and only row 6, in the last chunk, is in bin 1, at S/N
    # 100; the other rows have S/N 10.
    monkeypatch.setattr(_spectra, "CHUNK", 2 * NPIX)
    cov = (np.eye(NPIX), np.ones((NPIX, NPIX)))
    model = Model(np.zeros((2, NPIX)), cov, (30, 30), edges=(30.0,))
    flux = np.full((7, NPIX), 10.0)
    flux[6] = 100.0
    calls = []
    error = (
        r"^model: cov\[1\] is not positive definite: dpotrf failed at its leading "
        r"minor of order 2$"
    )
    with pytest.raises(LinAlgError, match=error):
        calibration(
            model,
            flux,
            np.ones(flux.shape),
            sizes=(1,),
            rng=np.random.default_rng(0),
            progress=lambda *args: calls.append(args),
        )
    assert calls == []
    # With one good pixel, row 6 has no conditioning pixel at size 1 and takes the
    # prior, without P.
    mask = np.zeros(flux.shape, bool)
    mask[6, 1:] = True
    result = calibration(
        model, flux, np.ones(flux.shape), mask, sizes=(1,), rng=np.random.default_rng(0)
    )
    assert result.n[6, 0] == 1 and np.isfinite(result.z2[6, 0])


def test_a_failed_factorization_names_the_row_of_the_caller_in_a_later_chunk(
    monkeypatch,
):
    # A precision matrix that passes its check, but is not positive definite on pixels
    # 4 and 5 together, which are bad only in row 5, in the third chunk of two rows. A
    # mask of size 1 adds one pixel to G, so no other row holds both.
    P = np.eye(NPIX)
    P[4, 5] = P[5, 4] = 2.0
    monkeypatch.setattr(_model, "precision", lambda S: P)
    monkeypatch.setattr(_spectra, "CHUNK", 2 * NPIX)
    model, flux, ivar, _ = spectra()
    mask = np.zeros(flux.shape, bool)
    mask[5, [4, 5]] = True
    calls = []
    error = (
        r"^model: cov\[0\] is not positive definite in its precision matrix on the "
        r"pixels G, for the pattern of row 5$"
    )
    with pytest.raises(LinAlgError, match=error):
        calibration(
            model,
            flux,
            ivar,
            mask,
            sizes=(1,),
            rng=np.random.default_rng(0),
            progress=lambda *args: calls.append(args),
        )
    assert calls == [("calibration", 2, 7), ("calibration", 4, 7)]


def test_a_calibration_holds_read_only_copies_of_the_shape_of_z2():
    # A float32 z2 in Fortran order needs a cast and a C-ordered copy.
    z2, fs = np.asfortranarray(Z2, dtype=np.float32), FS[:, :1]
    cal = Calibration(z2, np.uint8(2), fs)
    for name, dtype in (("z2", np.float64), ("n", np.int64), ("fs", np.float64)):
        array = getattr(cal, name)
        assert (array.dtype, array.shape) == (dtype, Z2.shape)
        assert array.flags.c_contiguous and not array.flags.writeable
    assert same(cal.z2, Z2.astype(np.float32).astype(np.float64))
    assert same(cal.n, np.full(Z2.shape, 2, np.int64))
    assert same(cal.fs, np.repeat(FS[:, :1], 2, axis=1))
    assert not np.shares_memory(cal.z2, z2) and not np.shares_memory(cal.fs, fs)
    with pytest.raises(FrozenInstanceError):
        cal.n = N
    # A C-ordered float64 z2 needs neither, and is copied all the same.
    z2 = Z2.copy()
    assert not np.shares_memory(Calibration(z2, N, FS).z2, z2) and z2.flags.writeable


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("z2", np.ones(5), ValueError),
        ("z2", np.ones((5, 2, 1)), ValueError),
        ("z2", np.full((5, 2), -1.0), ValueError),
        ("z2", np.full((5, 2), np.inf), ValueError),
        ("z2", np.ones((5, 2), complex), TypeError),
        ("z2", np.ones((5, 2), bool), TypeError),
        ("n", 1.0, TypeError),
        ("n", np.True_, TypeError),
        ("n", -1, ValueError),
        ("n", 2**53, ValueError),
        ("n", np.ones(5, np.int64), ValueError),
        ("fs", 0.0, ValueError),
        ("fs", np.inf, ValueError),
        ("fs", 1j, TypeError),
        ("fs", np.ones((2, 2)), ValueError),
    ],
)
def test_an_invalid_field_of_a_calibration_is_rejected_with_its_name(
    field, value, error
):
    with pytest.raises(error, match=f"^{field}: "):
        Calibration(**({"z2": Z2, "n": N, "fs": FS} | {field: value}))


DUPLICATES = {
    "pickle": lambda c: pickle.loads(pickle.dumps(c)),
    "copy": copy.copy,
    "deepcopy": copy.deepcopy,
}


@pytest.mark.parametrize("duplicate", DUPLICATES.values(), ids=list(DUPLICATES))
def test_pickle_and_copy_rebuild_a_calibration_through_its_checks(duplicate):
    cal = Calibration(Z2, N, FS)
    clone = duplicate(cal)
    assert same(fields(clone), fields(cal))
    assert not any(array.flags.writeable for array in fields(clone))
    # A calibration damaged behind its frozen fields fails the checks when it is
    # rebuilt.
    object.__setattr__(cal, "n", -N)
    with pytest.raises(ValueError, match="^n: "):
        duplicate(cal)


# The two Gaussian populations of the mixture test of design section 11, on 41 pixels,
# so that a prediction of 1 pixel has N_k = 40: squared-exponential covariances plus
# white noise of variance 0.1. Population 1, of probability 0.3, has another mean, 4
# times the amplitude and 2/5 of the correlation length of population 0.
LAGS = np.subtract.outer(np.arange(41), np.arange(41))
MEANS = (np.zeros(41), 2 * np.cos(np.arange(41) / 6))
COVARIANCES = tuple(
    amplitude**2 * np.exp(-0.5 * (LAGS / length) ** 2) + 0.1 * np.eye(41)
    for amplitude, length in ((1, 5), (4, 2))
)


def mixture(nspec, rng):
    # Spectra of the mixture and the population of each. ivar = 1, so a_s = 1 and the
    # working vector of a spectrum is its flux minus the model mean.
    labels = rng.random(nspec) < 0.3
    flux = np.empty((nspec, 41))
    for p in (0, 1):
        rows = labels == p
        flux[rows] = rng.multivariate_normal(MEANS[p], COVARIANCES[p], rows.sum())
    return flux, labels


@pytest.mark.slow
def test_each_population_of_a_mixture_has_the_mean_z2_of_its_own_moments():
    rng = np.random.default_rng(17)
    flux, _ = mixture(200, rng)
    model, _ = fit(flux, np.ones(flux.shape), nsigma=())
    flux, labels = mixture(20_000, rng)
    sizes = (1, 4, 16)
    result = calibration(model, flux, np.ones(flux.shape), sizes=sizes, rng=rng)
    # A spectrum x of population p has the mean mu_p and the covariance S_p. With m and
    # P^-1 the mean and the covariance of the model, the conditional mean of x_M given
    # the other pixels O is m_M - P_MM^-1 P_MO (x_O - m_O), so the residual is
    # r = P_MM^-1 P_{M,:} (x - m), and the conditional covariance is C = P_MM^-1. Then
    # E_p[r r^T] = C P_{M,:} A P_{:,M} C, with A = S_p + d d^T and d = mu_p - m, and
    # E_p[z2] = tr(C^-1 E_p[r r^T]) = tr(P_MM^-1 (P A P)_MM). The start of a run is
    # uniform and does not depend on x, so the expectation of the mean z2 of
    # population p is the mean of E_p[z2] over the starts.
    P = np.linalg.inv(model.cov[0])
    pulls = np.empty((2, len(sizes)))
    for p in (0, 1):
        d = MEANS[p] - model.mean[0]
        B = P @ (COVARIANCES[p] + np.outer(d, d)) @ P
        mean, sem = result.ratio(where=labels == p)
        for j, n in enumerate(sizes):
            runs = [slice(u, u + n) for u in range(41 - n + 1)]
            z2 = np.mean([np.trace(np.linalg.solve(P[M, M], B[M, M])) for M in runs])
            expected = z2 / (n * finite_sample_factor(200, 41 - n, n))
            pulls[p, j] = (mean[j] - expected) / sem[j]
    assert np.all(np.abs(pulls) <= 3)


NSPEC = 7


def audited(rank=None):
    # A model of 40 simulated spectra, of the rank, or for rank "noise" a noise model,
    # and 7 others in float32 with bad pixels: pixel 3 in every spectrum, by mask in
    # rows 0 to 3 and by ivar 0 in rows 4 to 6, and pixel 8 in every spectrum but row
    # 2. Flux is NaN at the bad pixels and ivar inf at the masked ones, so that a read
    # of either shows.
    rng = np.random.default_rng(13)
    train = simulate(40, NPIX, rng=rng)
    if rank == "noise":
        model, _ = fit(train.flux, train.ivar, nsigma=(), noise=True)
    else:
        model, _ = fit(train.flux, train.ivar, nsigma=(), rank=rank)
    test = simulate(NSPEC, NPIX, rng=rng)
    flux, ivar = test.flux.astype(np.float32), test.ivar.astype(np.float32)
    mask = np.zeros(flux.shape, bool)
    mask[:4, 3] = True
    ivar[4:, 3] = 0.0
    mask[[0, 1, 3, 4, 5, 6], 8] = True
    flux[mask | (ivar == 0)] = np.nan
    ivar[mask] = np.inf
    return model, flux, ivar, mask


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("rank", [None, 3, "noise"], ids=["dense", "low-rank", "noise"])
def test_rho_is_the_rms_residual_in_reported_sigmas_over_the_good_spectra(
    monkeypatch, rank
):
    # Chunks of 2 rows, the last one partial, so that the sums run over the chunks.
    monkeypatch.setattr(_spectra, "CHUNK", 2 * NPIX)
    model, flux, ivar, mask = audited(rank)
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


def test_rho_uses_the_prediction_of_the_bin_of_each_spectrum(monkeypatch):
    # Two bins with other means and covariances, whose spectra share chunks of 2 rows:
    # rows 0 to 3 at S/N 10 and rows 4 to 6 at S/N 100.
    monkeypatch.setattr(_spectra, "CHUNK", 2 * NPIX)
    cov = (random_spd(NPIX, 1e2, 0), random_spd(NPIX, 1e2, 1))
    means = np.array([np.zeros(NPIX), np.full(NPIX, 5.0)])
    model = Model(means, cov, (30, 30), edges=(30.0,))
    flux = np.random.default_rng(17).standard_normal((NSPEC, NPIX))
    flux += np.where(np.arange(NSPEC) < 4, 10.0, 100.0)[:, np.newaxis]
    ivar = np.ones(flux.shape)
    rho = noise_audit(model, flux, ivar, nguard=2)
    predicted = pixelwise(model, flux, ivar, nguard=2).flux
    # As in the test above, with every pixel good.
    expected = np.sqrt(np.mean((flux - predicted) ** 2, axis=0))
    assert np.all(np.abs(rho - expected) <= 5 * EPS * expected)


def test_progress_of_noise_audit_counts_the_spectra_done_after_each_chunk(
    monkeypatch,
):
    monkeypatch.setattr(_spectra, "CHUNK", 2 * NPIX)
    model, flux, ivar, mask = audited()
    calls = []
    noise_audit(model, flux, ivar, mask, progress=lambda *args: calls.append(args))
    assert calls == [("noise_audit", done, NSPEC) for done in (2, 4, 6, 7)]


def test_a_covariance_that_fails_its_check_raises_before_any_chunk(monkeypatch):
    # As in pixelwise: cov[1] has no inverse, and only row 6, in the last chunk, is in
    # bin 1, at S/N 100; the other rows have S/N 10.
    monkeypatch.setattr(_spectra, "CHUNK", 2 * NPIX)
    cov = (np.eye(NPIX), np.ones((NPIX, NPIX)))
    model = Model(np.zeros((2, NPIX)), cov, (10, 10), edges=(30.0,))
    flux = np.full((NSPEC, NPIX), 10.0)
    flux[6] = 100.0
    calls = []
    error = (
        r"^model: cov\[1\] is not positive definite: dpotrf failed at its leading "
        r"minor of order 2$"
    )
    with pytest.raises(LinAlgError, match=error):
        noise_audit(
            model,
            flux,
            np.ones(flux.shape),
            nguard=2,
            progress=lambda *args: calls.append(args),
        )
    assert calls == []


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
    # does not depend on the noise of f_si, and E[d_si] = k_i. Given the model, the D_s
    # of the independent spectra are independent and identically distributed, so their
    # mean has the standard error std(D_s) / sqrt(n_test), and it is normal to good
    # accuracy at n_test 4000. The mean of each set then lies beyond 3 standard errors
    # with the probability 0.27%, and the test fails falsely with a probability of at
    # most 0.54% (Bonferroni).
    d = ivar * ((test.flux - predicted) ** 2 - (test.truth - predicted) ** 2)
    for pixels in (k > 1, k == 1):
        deviation = np.mean(rho[pixels] ** 2 - e[pixels] - k[pixels])
        assert abs(deviation) <= 3 * stats.sem(d[:, pixels].mean(axis=1))
