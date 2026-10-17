from dataclasses import fields
from functools import partial

import numpy as np
import pytest
from numpy.linalg import LinAlgError

from _golden import checked, same
from gspice import Model, _model, _spectra, fit_blocks, infill, snr
from gspice._blocks import _chi2, _factors
from oracle import condition, conditional, random_spd, window

EPS = np.finfo(float).eps
NPIX = 40
PIXELS = np.arange(NPIX)
WIDTH, STRIDE = 10, 7
# The starts of fit_blocks at NPIX, WIDTH and STRIDE: blocks at both ends and inside,
# and a last one flush with the end.
STARTS = np.array([0, 7, 14, 21, 28, 30])
# The mean of each of two S/N bins, near the S/N of the spectra of the bin.
MEANS = np.array([np.full(NPIX, 10.0), np.full(NPIX, 100.0)])
# Singular with a positive diagonal of powers of 2 (as in test_predict).
ROOT = 2.0 ** (PIXELS % 3 - 1)
SINGULAR = np.outer(ROOT, ROOT)


def two_bins(bins, center=None):
    # A model of two bins and spectra without bad pixels whose working vectors are
    # drawn from N(0, cov[b]). ivar is a_s^2 at every pixel, a power of 2, so that a_s
    # is exact over any pixels, and the S/N, near 10 or 100, is far from the edge 30.
    cov = (random_spd(NPIX, 1e2, 0), random_spd(NPIX, 1e2, 1))
    model = Model(MEANS, cov, (100, 100), edges=(30.0,), center=center)
    rng = np.random.default_rng(2)
    a = 2.0 ** rng.integers(0, 3, len(bins))
    x = np.array([np.linalg.cholesky(cov[b]) @ rng.standard_normal(NPIX) for b in bins])
    flux = (x + MEANS[bins]) / a[:, np.newaxis] + (0.0 if center is None else center)
    ivar = np.repeat(np.square(a)[:, np.newaxis], NPIX, axis=1)
    ratio = snr(flux, ivar)
    np.testing.assert_array_equal(np.searchsorted(model.edges, ratio, "right"), bins)
    return model, flux, ivar


@pytest.mark.parametrize("nguard", [0, 3, 20])
@pytest.mark.parametrize("kappa", [1e2, 1e8])
def test_the_block_chi2_matches_the_60_digit_reference(kappa, nguard):
    # nguard 3 clips the guard of the first and the last blocks, and with nguard 20 the
    # guard of the block at 14 covers all pixels: the prior.
    C = random_spd(NPIX, kappa, 0)
    X = np.random.default_rng(1).standard_normal((3, NPIX)) @ np.linalg.cholesky(C).T
    chi2 = _chi2(
        _factors(Model(np.zeros((1, NPIX)), (C,), (100,)), 0, STARTS, WIDTH, nguard), X
    )
    # Design section 11: |dchi2| <= 10 kappa eps (chi2 + |T|), with T = A, which bounds
    # chi2_A = chi2 / |A| by the same tolerance relative to chi2_A + 1.
    tol = 10 * condition(C) * EPS
    for k, start in enumerate(STARTS):
        _, _, reference = conditional(
            C, window(start, nguard, NPIX, WIDTH), X, range(start, start + WIDTH)
        )
        reference /= WIDTH
        assert np.all(np.abs(chi2[:, k] - reference) <= tol * (reference + 1))


def test_the_block_chi2_never_reads_the_guard_pixels():
    # The guard pixels are marginalized, so an outlier there cannot cancel.
    start = 14
    C = random_spd(NPIX, 1e5, 0)
    X = np.random.default_rng(3).standard_normal((3, NPIX)) @ np.linalg.cholesky(C).T
    factors = _factors(
        Model(np.zeros((1, NPIX)), (C,), (100,)), 0, np.array([start]), WIDTH, 3
    )
    expected = _chi2(factors, X)
    X[
        :, [i for i in window(start, 3, NPIX, WIDTH) if not start <= i < start + WIDTH]
    ] = np.nan
    assert same(_chi2(factors, X), expected)


@pytest.mark.parametrize(
    ("npix", "width", "stride", "starts"),
    [
        # One block of all pixels.
        (10, 10, 3, [0]),
        # npix - width = 6 is a multiple of the stride, so no block is added.
        (10, 4, 3, [0, 3, 6]),
        # The last start 4 is below npix - width = 6, which is added.
        (10, 4, 4, [0, 4, 6]),
        # A stride far above npix.
        (10, 3, 2**64, [0, 7]),
    ],
)
def test_blocks_start_every_stride_and_the_last_ends_at_the_last_pixel(
    npix, width, stride, starts
):
    # With nguard npix, every G covers all pixels: the prior, without P.
    model = Model(np.zeros((1, npix)), (np.eye(npix),), (10,))
    flux = np.random.default_rng(4).standard_normal((2, npix))
    blocks = fit_blocks(
        model,
        flux,
        np.ones((2, npix)),
        width=width,
        stride=stride,
        nguard=npix,
        percentile=50,
    ).blocks
    assert same(blocks.starts, np.array(starts, dtype=np.int64))
    assert (blocks.width, blocks.nguard, blocks.thresholds.shape) == (
        width,
        npix,
        (1, len(starts)),
    )


@pytest.mark.parametrize("center", [None, 0.5])
def test_each_bin_gets_the_percentile_of_the_chi2_of_its_own_spectra(center):
    bins = np.array([0, 1] * 12 + [0] * 6)
    model, flux, ivar = two_bins(bins, center)
    result = fit_blocks(
        model, flux, ivar, width=WIDTH, stride=STRIDE, nguard=3, percentile=90
    )
    blocks = result.blocks
    assert same(blocks.starts, STARTS)
    # An independent chi2_A: infill of A with its guard ignored, whose chi2 is over
    # T = A, as no pixel is bad.
    chi2 = np.empty((len(bins), STARTS.size))
    for k, start in enumerate(STARTS):
        A = (PIXELS >= start) & (PIXELS < start + WIDTH)
        G = np.isin(PIXELS, window(start, 3, NPIX, WIDTH))
        chi2[:, k] = infill(model, flux, ivar, replace=A, ignore=G & ~A).chi2 / WIDTH
    expected = [np.percentile(chi2[bins == b], 90, axis=0) for b in (0, 1)]
    # Each chi2_A lies within 10 kappa eps (chi2_A + 1) of the exact value, as the
    # reference tests check, so two lie within twice that. A percentile interpolates
    # linearly between two order statistics, which move by at most the largest bound.
    tol = 2 * 10 * max(map(condition, model.cov)) * EPS * (chi2.max() + 1)
    assert np.all(np.abs(blocks.thresholds - expected) <= tol)
    # A new model, equal to the old one but for blocks, which shares its arrays and
    # its precision matrices.
    assert model.blocks is None
    for name in (f.name for f in fields(Model) if f.init and f.name != "blocks"):
        assert getattr(result, name) is getattr(model, name)
    assert all(result._precision(b) is model._precision(b) for b in (0, 1))


@pytest.mark.parametrize("nspec", [999, 1000])
def test_a_bin_needs_1000_spectra_at_the_percentile_99_9(nspec):
    # Design decision A7: m (100 - p) >= 100 with p the decimal 99.9. As a float,
    # 1000 (100 - 99.9) is below 100.
    model = Model(np.zeros((1, 2)), (np.eye(2),), (10,))
    flux = np.random.default_rng(5).standard_normal((nspec, 2))
    call = partial(
        fit_blocks, model, flux, np.ones(flux.shape), width=1, stride=1, nguard=0
    )
    if nspec == 1000:
        assert call().blocks.thresholds.shape == (1, 2)
    else:
        error = (
            r"^fit_blocks: percentile 99\.9 needs at least 1000 spectra in each S/N "
            r"bin, got 999 in bin 0$"
        )
        with pytest.raises(ValueError, match=error):
            call()


@pytest.mark.parametrize(
    ("rows", "message"), [(3, "got 0 in bin 1"), (0, "got 0 in bin 0")]
)
def test_a_bin_without_spectra_is_rejected(rows, message):
    model, flux, ivar = two_bins(np.zeros(3, dtype=int))
    with pytest.raises(ValueError, match=f"^fit_blocks: .*, {message}$"):
        fit_blocks(model, flux[:rows], ivar[:rows], width=WIDTH, percentile=50)


def test_bad_pixels_are_infilled_before_the_scan():
    # fit_blocks of spectra with bad pixels equals fit_blocks of their infilled flux
    # without bad pixels: the scan uses the conditional mean at bad pixels, and a_s of
    # the good pixels, which ivar makes exact.
    bins = np.array([0, 1] * 10)
    model, flux, ivar = two_bins(bins)
    mask = np.zeros(flux.shape, bool)
    mask[0, :3] = mask[1, 37:] = mask[2, 15:22] = True
    zero = np.zeros(flux.shape, bool)
    zero[3, [5, 20]] = zero[4, 33] = True
    flux_bad = np.where(mask | zero, np.nan, flux)
    ivar_bad = np.where(zero, 0.0, ivar)
    filled = infill(model, flux_bad, ivar_bad, mask).flux
    arguments = {"width": WIDTH, "stride": STRIDE, "nguard": 3, "percentile": 80}
    expected = fit_blocks(model, filled, ivar, **arguments).blocks
    result = fit_blocks(model, flux_bad, ivar_bad, mask, **arguments).blocks
    assert same(result.thresholds, expected.thresholds)


def test_progress_counts_the_spectra_of_each_chunk_and_chunks_change_no_threshold(
    monkeypatch,
):
    # Bin 0 holds rows 0, 2 and 4, two chunks of two rows, and bin 1 rows 1 and 3.
    bins = np.array([0, 1, 0, 1, 0])
    model, flux, ivar = two_bins(bins)
    arguments = {"width": WIDTH, "stride": STRIDE, "nguard": 3, "percentile": 50}
    whole = fit_blocks(model, flux, ivar, **arguments).blocks
    monkeypatch.setattr(_spectra, "CHUNK", 2 * NPIX)
    calls = []
    chunked = fit_blocks(
        model, flux, ivar, **arguments, progress=lambda *args: calls.append(args)
    ).blocks
    assert calls == [("fit_blocks", 2, 5), ("fit_blocks", 3, 5), ("fit_blocks", 5, 5)]
    # Only the products with fewer spectra differ, as in the test of the percentile.
    tol = 2 * 10 * max(map(condition, model.cov)) * EPS * (whole.thresholds.max() + 1)
    assert np.all(np.abs(chunked.thresholds - whole.thresholds) <= tol)


def test_fit_blocks_changes_no_input():
    bins = np.array([0, 1] * 3)
    model, flux, ivar = two_bins(bins)
    mask = np.zeros(flux.shape, bool)
    mask[0, 3] = True
    checked(
        lambda *spectra: (
            fit_blocks(model, *spectra, width=WIDTH, percentile=50).blocks.thresholds
        ),
        flux,
        ivar,
        mask,
    )


def test_a_scan_whose_blocks_cover_all_pixels_needs_no_precision_matrix():
    # chi2_A = x_i^2 / C_ii for the block of pixel i under the prior, exactly with
    # these powers of 2, although SINGULAR has no inverse.
    model = Model(np.zeros((1, NPIX)), (SINGULAR,), (10,))
    X = np.random.default_rng(6).standard_normal((3, NPIX))
    blocks = fit_blocks(
        model, X, np.ones(X.shape), width=1, stride=1, nguard=NPIX, percentile=50
    ).blocks
    assert same(blocks.thresholds[0], np.percentile((X / ROOT) ** 2, 50, axis=0))


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        # SINGULAR at T = A of block 0, whose G covers all pixels, is singular.
        (
            {"width": 2, "nguard": NPIX},
            "not positive definite on the pixels T, for block 0",
        ),
        # Block 3 is the first whose G holds pixel 25.
        (
            {"width": WIDTH, "stride": STRIDE, "nguard": 0},
            "not positive definite in its precision matrix on the pixels G, for block 3",
        ),
        # The blocks need no P, and infill names the row of the caller, also in a
        # later chunk.
        (
            {
                "width": 1,
                "stride": 1,
                "nguard": NPIX,
                "mask": (np.arange(3)[:, np.newaxis] == 2) & (PIXELS == 25),
            },
            (
                "not positive definite in its precision matrix on the pixels G, for "
                "the pattern of row 2"
            ),
        ),
    ],
    ids=["prior", "pattern", "infill"],
)
def test_a_block_whose_factorization_fails_names_the_bin_and_the_block_at_once(
    monkeypatch, arguments, message
):
    # A precision matrix that passed its check, but is not positive definite at pixel
    # 25, as rounding can make it near the rcond limit.
    P = np.eye(NPIX)
    P[25, 25] = -1.0
    monkeypatch.setattr(_model, "precision", lambda S: P)
    monkeypatch.setattr(_spectra, "CHUNK", 2 * NPIX)
    flux = np.random.default_rng(7).standard_normal((3, NPIX))
    calls = []
    with pytest.raises(LinAlgError, match=rf"^model: cov\[0\] is {message}$"):
        fit_blocks(
            Model(np.zeros((1, NPIX)), (SINGULAR,), (10,)),
            flux,
            np.ones(flux.shape),
            **arguments,
            percentile=50,
            progress=lambda *args: calls.append(args),
        )
    # Before any spectrum.
    assert calls == []


def test_a_covariance_that_fails_the_check_of_its_precision_matrix_raises_at_once():
    # As in pixelwise: cov[1] has no inverse, and no G covers all pixels. The factors
    # of every bin come before any spectrum, so the spectra of bin 0, at S/N 10, are
    # not scanned first; those of bin 1 have S/N 100.
    model = Model(MEANS, (np.eye(NPIX), SINGULAR), (10, 10), edges=(30.0,))
    flux = np.repeat(MEANS, 2, axis=0)
    calls = []
    error = (
        r"^model: cov\[1\] is not positive definite: dpotrf failed at its leading "
        r"minor of order 2$"
    )
    with pytest.raises(LinAlgError, match=error):
        fit_blocks(
            model,
            flux,
            np.ones(flux.shape),
            width=WIDTH,
            nguard=0,
            percentile=50,
            progress=lambda *args: calls.append(args),
        )
    assert calls == []
