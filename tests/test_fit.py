import logging
from dataclasses import astuple

import numpy as np
import pytest
from numpy.linalg import LinAlgError

from _golden import checked, same
from gspice import _model, _spectra, fit, pixelwise, scale, snr
from gspice._linalg import precision

EPS = np.finfo(float).eps
NSPEC, NPIX, NGUARD = 200, 12, 2


def spectra(nspec=NSPEC):
    # Continuum-normalized spectra: a smooth signal of rank 3 plus white noise. Rows 10
    # and above have 5% bad pixels, in mask (NaN flux, inf ivar) or with ivar 0 (NaN
    # flux); rows 0 to 9 have none, for the spikes of the tests. At N_s 200, in-sample
    # |z| of the noise stays below 3.6, and the ceiling of design section 4.1 is 13.8.
    rng = np.random.default_rng(0)
    pixels = np.arange(NPIX)
    basis = np.array([np.ones(NPIX), np.cos(pixels / 2), np.sin(pixels / 3)])
    ivar = np.outer(rng.uniform(100, 400, nspec), 1 + 0.5 * np.sin(pixels))
    flux = 1 + 0.05 * rng.standard_normal((nspec, 3)) @ basis
    flux += rng.standard_normal((nspec, NPIX)) / np.sqrt(ivar)
    bad = (rng.random((nspec, NPIX)) < 0.05) & (np.arange(nspec) >= 10)[:, np.newaxis]
    mask = bad & (rng.random((nspec, NPIX)) < 0.5)
    ivar[bad & ~mask] = 0.0
    ivar[mask] = np.inf
    flux[bad] = np.nan
    return flux, ivar, mask


def spike(flux, ivar, s, i, sigma):
    flux[s, i] += sigma / np.sqrt(ivar[s, i])


def recorder():
    calls = []
    return calls, lambda *args: calls.append(args)


@pytest.mark.parametrize("center", [None, 0.9])
@pytest.mark.parametrize("weighted", [False, True])
def test_the_final_fit_is_the_estimator_on_the_final_mask(weighted, center):
    flux, ivar, mask = spectra()
    weights = np.random.default_rng(1).uniform(0.5, 2, NSPEC) if weighted else None
    model, training = fit(
        flux, ivar, mask, nguard=NGUARD, center=center, weights=weights
    )
    rows = ~training.all(axis=1)
    c = 0.0 if center is None else center
    # x_s = a_s (ftilde_s - c) on the final mask, before any mean is subtracted.
    spectra_used = flux[rows], ivar[rows], training[rows]
    X, _, _ = scale(*spectra_used, center=c, refmean=np.zeros(NPIX))
    w = np.ones(len(X)) if weights is None else weights[rows]
    if center is None:
        # np.cov divides by V1 - V2/V1, which is N_s - 1 for unit weights.
        mean = np.average(X, axis=0, weights=w)
        cov = np.cov(X, rowvar=False, aweights=w)
        divisor = w.sum() - w @ w / w.sum()
    else:
        mean = np.zeros(NPIX)
        cov = (X.T * w) @ X / w.sum()
        divisor = w.sum()
    # Each entry sums N_s terms w_s (x_si - m_i)(x_sj - m_j) / divisor, and both
    # computations round each term by about (N_s + 5) eps of w_s A_si A_sj, with
    # A = |x| + |m|; to first order, an error of m does not move S.
    A = np.abs(X) + np.abs(mean)
    tol = 2 * (len(X) + 5) * EPS * (A.T * w) @ A / divisor
    assert np.all(np.abs(model.cov[0] - cov) <= tol)
    # The mean rounds by at most (N_s + 2) eps of sum_s w_s |x_s| / V1 in each.
    assert np.all(
        np.abs(model.mean[0] - mean) <= 2 * (len(X) + 2) * EPS * w @ np.abs(X) / w.sum()
    )
    assert model.n_train == (np.count_nonzero(rows),)
    assert (model.center, model.weighted) == (center, weighted)


def test_equal_weights_give_the_unweighted_fit_and_weight_0_excludes_a_spectrum():
    flux, ivar, mask = spectra()
    model, training = fit(flux, ivar, mask, nguard=NGUARD)
    equal, equal_mask = fit(
        flux, ivar, mask, nguard=NGUARD, weights=np.full(NSPEC, 2.5)
    )
    assert same(
        (equal.mean, equal.cov[0], equal_mask), (model.mean, model.cov[0], training)
    )
    assert equal.weighted is False
    weights = np.ones(NSPEC)
    weights[3] = 0.0
    calls, progress = recorder()
    zero, zero_mask = fit(
        flux, ivar, mask, nguard=NGUARD, weights=weights, progress=progress
    )
    assert zero_mask[3].all()
    assert zero.n_train == (model.n_train[0] - 1,)
    # It is also left out of the three clipping passes, and the fit equals the fit of
    # the other spectra bit for bit.
    assert [total for _, done, total in calls if done == total] == [NSPEC - 1] * 4
    others = [np.delete(array, 3, axis=0) for array in (flux, ivar, mask)]
    rest, rest_mask = fit(*others, nguard=NGUARD)
    assert same(
        (zero.mean, zero.cov[0], np.delete(zero_mask, 3, axis=0)),
        (rest.mean, rest.cov[0], rest_mask),
    )


def test_a_spectrum_that_a_mask_covers_completely_is_dropped_from_later_fits():
    flux, ivar, mask = spectra()
    # Row 0 is good only at pixels 4 to 6, and pass 1 flags its spike at pixel 5 with
    # both neighbours (|z| 12.7 > 10). Row 1 has no good pixel.
    mask[0] = True
    mask[0, 4:7] = False
    spike(flux, ivar, 0, 5, 40.0)
    mask[1] = True
    calls, progress = recorder()
    model, training = fit(
        flux, ivar, mask, nguard=NGUARD, nsigma=(10, 1000), progress=progress
    )
    # The spectra of passes 1 and 2 and of the final fit.
    totals = [total for _, done, total in calls if done == total]
    assert totals == [NSPEC - 1, NSPEC - 2, NSPEC - 2]
    assert model.n_train == (NSPEC - 2,)
    assert training[:2].all()


@pytest.mark.parametrize(("center", "needed"), [(None, NPIX + 1), (0.9, NPIX)])
def test_a_fit_needs_npix_spectra_and_one_more_for_a_mean(center, needed):
    # Row 0 has more than maxbadpix bad pixels.
    flux, ivar, mask = spectra(needed + 1)
    mask[0, :4] = True
    options = {"nguard": NGUARD, "nsigma": (), "center": center, "maxbadpix": 3}
    model, _ = fit(flux, ivar, mask, **options)
    assert model.n_train == (needed,)
    # Row 0 counts once, as removed by maxbadpix, also with weight 0.
    weights = np.ones(needed + 1)
    weights[:2] = 0.0
    message = (
        rf"^fit: the final fit needs at least {needed} spectra for a \S+ model of "
        rf"{NPIX} pixels, got {needed - 1}: of {needed + 1} spectra, maxbadpix "
        r"removed 1, weight 0 removed 1 and full masks removed 0$"
    )
    with pytest.raises(ValueError, match=message):
        fit(flux, ivar, mask, **options, weights=weights)


def test_empty_nsigma_fits_on_the_bad_pixels_without_clipping():
    flux, ivar, mask = spectra()
    # A spike that a clipping pass would flag.
    spike(flux, ivar, 5, 6, 40.0)
    calls, progress = recorder()
    model, training = fit(flux, ivar, mask, nguard=NGUARD, nsigma=(), progress=progress)
    np.testing.assert_array_equal(training, mask | (ivar == 0))
    assert calls == [("fit", NSPEC, NSPEC)]
    assert model.n_train == (NSPEC,)


def test_pixelwise_uses_the_precision_matrix_that_fit_computed(monkeypatch):
    calls = []

    def counted(S):
        calls.append(S)
        return precision(S)

    monkeypatch.setattr(_model, "precision", counted)
    flux, ivar, mask = spectra()
    model, _ = fit(flux, ivar, mask, nguard=NGUARD, nsigma=(10, 8))
    # One per pass, and the last for the returned model.
    assert len(calls) == 3
    pixelwise(model, flux, ivar, mask, nguard=NGUARD)
    assert len(calls) == 3


def test_each_pass_logs_and_a_pass_above_the_in_sample_ceiling_warns(caplog):
    caplog.set_level(logging.DEBUG, logger="gspice")
    # N_s 20, npix 12 and nguard 2: the ceiling is sqrt(19 (20 - 7 - 1) / 20) = 3.38.
    # Row 0 is good only at pixels 4 to 6, and pass 2 flags its spike at pixel 5
    # (|z| 4.0 > 3, the noise below 2.3), so 3 of 240 pixels, and drops it.
    flux, ivar, mask = spectra(20)
    mask[0] = True
    mask[0, 4:7] = False
    spike(flux, ivar, 0, 5, 40.0)
    weights = np.repeat([1.0, 2.0], 10)
    fit(flux, ivar, mask, nguard=NGUARD, nsigma=(20, 3), weights=weights)
    records = [
        (record.levelno, record.getMessage())
        for record in caplog.records
        if record.levelno > logging.DEBUG
    ]
    assert records == [
        (
            logging.WARNING,
            (
                "fit: pass 1 has nsigma 20 above 3.38, the in-sample ceiling of |z| "
                "at N_s = 20"
            ),
        ),
        (
            logging.INFO,
            (
                "fit: pass 1 with nsigma 20 flagged 0% of the pixels of its 20 "
                "spectra; spectra dropped: 0"
            ),
        ),
        (
            logging.INFO,
            (
                "fit: pass 2 with nsigma 3 flagged 1.25% of the pixels of its 20 "
                "spectra; spectra dropped: 1"
            ),
        ),
        # V1^2/V2 = (9 + 2 * 10)^2 / (9 + 4 * 10) = 17.16 for the 19 spectra left.
        (
            logging.INFO,
            (
                "fit: the final fit of 19 spectra is weighted, with V1^2/V2 = 17.16 "
                "effective spectra"
            ),
        ),
    ]
    assert caplog.records[-1].getMessage().startswith("fit: 20 spectra in ")


@pytest.mark.parametrize(
    ("center", "nguard", "ceiling"),
    [
        # 2 nguard + 1 >= npix, so N_k = 0, and sqrt(19 (20 - 0 - 1) / 20) = 4.25.
        (None, NPIX, "4.25"),
        # A constant centre adds no mean to the leverage: sqrt(20 - 7) = 3.61.
        (0.9, NGUARD, "3.61"),
    ],
)
def test_the_ceiling_depends_on_the_centre_and_the_conditioning_pixels(
    center, nguard, ceiling, caplog
):
    caplog.set_level(logging.WARNING, logger="gspice")
    flux, ivar, mask = spectra(20)
    fit(flux, ivar, mask, nguard=nguard, nsigma=(20,), center=center)
    expected = (
        f"fit: pass 1 has nsigma 20 above {ceiling}, the in-sample ceiling of |z| at "
        "N_s = 20"
    )
    assert [record.getMessage() for record in caplog.records] == [expected]


@pytest.mark.parametrize(
    ("nsigma", "covariance"),
    [((10,), "the clipping covariance of pass 1"), ((), "the final covariance")],
)
def test_a_singular_covariance_raises_naming_its_fit_and_n_s(nsigma, covariance):
    flux, ivar, mask = spectra()
    # Pixel 1 repeats pixel 0 in every spectrum, so the covariance is singular.
    for array in (flux, ivar, mask):
        array[:, 1] = array[:, 0]
    with pytest.raises(LinAlgError, match=rf"^fit: {covariance} \(N_s = {NSPEC}\) is "):
        fit(flux, ivar, mask, nguard=NGUARD, nsigma=nsigma)


def test_a_singular_covariance_of_a_bin_raises_naming_the_bin_and_its_n_s():
    # Rows 0 to 99, with 16 times the ivar, form bin 1, where pixel 1 repeats pixel 0.
    flux, ivar, mask = spectra()
    ivar[:100] *= 16
    for array in (flux, ivar, mask):
        array[:100, 1] = array[:100, 0]
    ratio = snr(flux, ivar, mask)
    edge = float(ratio[100:].max() + ratio[:100].min()) / 2
    assert ratio[100:].max() < edge < ratio[:100].min()
    message = (
        rf"^fit: the final covariance of S/N bin 1 \(snr >= {edge}\) \(N_s = 100\) is "
    )
    with pytest.raises(LinAlgError, match=message):
        fit(flux, ivar, mask, nguard=NGUARD, nsigma=(), snr_bins=(edge,))


def test_a_pixel_of_variance_0_raises_naming_its_fit():
    flux, ivar, mask = spectra()
    # Pixel 4 is good and 0 in every spectrum. 2022 also raised LinAlgError.
    flux[:, 4], ivar[:, 4], mask[:, 4] = 0.0, 100.0, False
    message = (
        rf"^fit: the clipping covariance of pass 1 \(N_s = {NSPEC}\) is not positive "
        r"definite: its variance at pixel 4 is 0$"
    )
    with pytest.raises(LinAlgError, match=message):
        fit(flux, ivar, mask, nguard=NGUARD, nsigma=(10,))


def test_progress_counts_the_spectra_of_each_fit_after_each_chunk(monkeypatch):
    monkeypatch.setattr(_spectra, "CHUNK", 64 * NPIX)
    flux, ivar, mask = spectra()
    calls, progress = recorder()
    fit(flux, ivar, mask, nguard=NGUARD, nsigma=(10, 8), progress=progress)
    # Two passes, then the final fit.
    assert calls == [("fit", done, NSPEC) for done in (64, 128, 192, 200)] * 3


def test_inputs_are_unchanged_and_outputs_are_not_shared():
    flux, ivar, mask = spectra()

    def outputs(flux, ivar, mask, weights):
        model, training = fit(flux, ivar, mask, nguard=NGUARD, weights=weights)
        return model.mean, model.cov[0], training

    checked(outputs, flux, ivar, mask, np.linspace(1, 2, NSPEC))


def test_a_low_rank_fit_holds_the_leading_eigenpairs_of_s_and_its_variances():
    flux, ivar, mask = spectra()
    dense, training = fit(flux, ivar, mask, nguard=NGUARD, nsigma=())
    model, low_mask = fit(flux, ivar, mask, nguard=NGUARD, nsigma=(), rank=4)
    S, low = dense.cov[0], model.cov[0]
    assert same(
        (model.mean, low_mask, model.n_train), (dense.mean, training, dense.n_train)
    )
    V, values = low.vectors, low.values
    assert V.shape == (NPIX, 4)
    # eigh is backward stable: each eigenpair and the orthogonality err by a small
    # multiple of npix eps |S|_2.
    top = np.linalg.eigvalsh(S)[::-1]
    tol = 10 * NPIX * EPS * top[0]
    assert np.all(np.abs(values - top[:4]) <= tol)
    assert np.all(np.linalg.norm(S @ V - V * values, axis=0) <= tol)
    assert np.all(np.abs(V.T @ V - np.eye(4)) <= 10 * NPIX * EPS)
    # d = diag(S) - t and c = d + t, with t = sum_k lambda_k V_ik^2 <= S_ii to
    # rounding, round twice each.
    c = low.diag + np.square(V) @ values
    assert np.all(np.abs(c - np.diagonal(S)) <= 4 * EPS * np.diagonal(S))


def test_the_clipping_passes_of_a_low_rank_fit_use_its_rank_below_npix_spectra():
    # 10 spectra without bad pixels, too few for a dense model of 12 pixels. A pass of
    # the fit flags what pixelwise z of the low-rank model fitted on bad flags.
    flux, ivar, mask = spectra(10)
    options = {"nguard": NGUARD, "rank": 3}
    first, _ = fit(flux, ivar, mask, **options, nsigma=())
    z = pixelwise(first, flux, ivar, mask, nguard=NGUARD).z
    flagged = _spectra.dilate(np.abs(z) > 2, 1)
    assert 0 < np.count_nonzero(flagged) < flagged.size
    model, training = fit(flux, ivar, mask, **options, nsigma=(2,))
    np.testing.assert_array_equal(training, flagged)
    final, _ = fit(flux, ivar, training, **options, nsigma=())
    assert same(
        (model.mean, astuple(model.cov[0])), (final.mean, astuple(final.cov[0]))
    )


@pytest.mark.parametrize(("center", "needed"), [(None, 5), (0.9, 4)])
def test_a_low_rank_fit_needs_k_plus_1_spectra_and_one_more_for_a_mean(
    center, needed, caplog
):
    # Rank 3 needs S of rank 4, so that lambda_3 > 0 and d > 0 can hold.
    caplog.set_level(logging.WARNING, logger="gspice")
    flux, ivar, mask = spectra(needed)
    options = {"nguard": NGUARD, "nsigma": (), "center": center, "rank": 3}
    # A clipping pass too: N_s - N_k - 1 < 0 with N_k = 12 - 5, so the ceiling of a
    # dense model is not defined, and no ceiling is derived for a low-rank one.
    model, _ = fit(flux, ivar, mask, **(options | {"nsigma": (20,)}))
    assert model.n_train == (needed,)
    assert caplog.records == []
    message = (
        rf"^fit: the final fit needs at least {needed} spectra for a \S+ model of "
        rf"rank 3, got {needed - 1}: of {needed - 1} spectra, maxbadpix removed 0, "
        r"weight 0 removed 0 and full masks removed 0$"
    )
    with pytest.raises(ValueError, match=message):
        fit(flux[1:], ivar[1:], mask[1:], **options)


@pytest.mark.parametrize(
    ("S", "rank", "message"),
    [
        # Indefinite, as rounding can make a sample covariance of rank below K.
        (
            np.diag([3.0, 2.0, -1.0]),
            3,
            r"needs lambda_K > 0 for rank 3, got lambda_3 = -1$",
        ),
        # The first eigenpair is exactly (3, e_0), so d_0 = 3 - 3 = 0.
        (
            np.diag([3.0, 2.0, 1.0]),
            1,
            r"needs d_i > 0 at every pixel for rank 1, got d_i <= 0 at pixels 0$",
        ),
    ],
)
def test_a_low_rank_covariance_needs_lambda_k_and_every_d_i_positive(S, rank, message):
    with pytest.raises(ValueError, match=f"^fit: the final covariance {message}"):
        _model._low_rank(S, rank, "fit: the final covariance")


def test_a_low_rank_fit_needs_no_good_pixel_at_every_pixel():
    # Design section 4.1 sets no such requirement for rank=K: pixel 6 is interpolated in
    # every spectrum.
    flux, ivar, mask = spectra()
    mask[:, 6] = True
    with pytest.raises(ValueError, match="needs a good pixel in one of its"):
        fit(flux, ivar, mask, nguard=NGUARD, nsigma=())
    model, _ = fit(flux, ivar, mask, nguard=NGUARD, nsigma=(), rank=3)
    assert np.all(model.cov[0].diag > 0)


def test_snr_bins_puts_the_edges_at_quantiles_of_the_used_spectra_and_assigns_them():
    # Row 10 has more than maxbadpix bad pixels and row 11 weight 0. Both are left
    # out of the quantiles, and an S/N far above the others would move the top edge.
    flux, ivar, mask = spectra()
    mask[10, :5] = True
    ivar[10:12, ~mask[10]] *= 100
    weights = np.ones(NSPEC)
    weights[11] = 0.0
    options = {"nguard": NGUARD, "nsigma": (), "maxbadpix": 4, "weights": weights}
    model, training = fit(flux, ivar, mask, **options, snr_bins=3)
    ratio = snr(flux, ivar, mask)
    used = np.ones(NSPEC, dtype=bool)
    used[[10, 11]] = False
    edges = tuple(np.quantile(ratio[used], [1 / 3, 2 / 3]))
    assert model.edges == edges
    bins = np.searchsorted(edges, ratio, side="right")
    assert model.n_train == tuple(np.bincount(bins[used]))
    np.testing.assert_array_equal(training.all(axis=1), ~used)
    # A sequence gives the interior edges, and an edge equal to an S/N puts that
    # spectrum in the bin above it.
    given = (float(np.sort(ratio)[60]), 16.0)
    model, _ = fit(flux, ivar, mask, **options, snr_bins=given)
    bins = np.searchsorted(given, ratio, side="right")
    assert model.edges == given
    assert model.n_train == tuple(np.bincount(bins[used]))
    assert np.count_nonzero(bins[used] == 0) == 60
    # A spectrum without a good pixel, which only maxbadpix >= npix lets through to the
    # full-mask check, is left out of the quantiles and the fit like one of weight 0.
    options["maxbadpix"] = NPIX
    weights[12] = 0.0
    model, _ = fit(flux, ivar, mask, **options, snr_bins=3)
    weights[12] = 1.0
    mask[12] = True
    full, training = fit(flux, ivar, mask, **options, snr_bins=3)
    assert same(
        (full.edges, full.n_train, full.mean), (model.edges, model.n_train, model.mean)
    )
    assert training[12].all()


@pytest.mark.parametrize(
    "options",
    [{}, {"center": 0.9, "weights": np.linspace(1, 2, NSPEC)}, {"rank": 3}],
    ids=["dense", "center-weights", "rank"],
)
def test_each_bin_is_the_fit_of_its_spectra_alone(options):
    # Design decision A6: each bin clips and fits its own spectra. Spikes in rows 0
    # to 9 give flags.
    flux, ivar, mask = spectra()
    for s in range(10):
        spike(flux, ivar, s, s, 40.0)
    options = {"nguard": NGUARD, "nsigma": (10, 6)} | options
    weights = options.pop("weights", None)
    calls, progress = recorder()
    model, training = fit(
        flux, ivar, mask, **options, weights=weights, snr_bins=3, progress=progress
    )
    bad = (mask != 0) | (ivar == 0)
    assert np.any(training & ~bad & ~training.all(axis=1)[:, np.newaxis])
    bins = np.searchsorted(model.edges, snr(flux, ivar, mask), side="right")
    single_calls, single_progress = recorder()
    for b in range(3):
        rows = bins == b
        single, single_mask = fit(
            flux[rows],
            ivar[rows],
            mask[rows],
            **options,
            weights=None if weights is None else weights[rows],
            progress=single_progress,
        )
        pair = (model.cov[b], single.cov[0])
        if "rank" in options:
            pair = tuple(map(astuple, pair))
        assert same(*pair)
        assert same(
            (model.mean[b], model.n_train[b], training[rows]),
            (single.mean[0], single.n_train[0], single_mask),
        )
        # pixelwise uses the precision matrix that fit computed for the bin.
        assert same(model._precision(b), single._precision(0))
    # Each fit of a bin counts its own spectra.
    assert calls == single_calls
    assert model.weighted is (weights is not None)


def test_snr_bins_1_gives_the_fit_of_one_bin_bit_for_bit():
    flux, ivar, mask = spectra()
    spike(flux, ivar, 0, 5, 40.0)
    weights = np.linspace(1, 2, NSPEC)
    default, default_mask = fit(flux, ivar, mask, nguard=NGUARD, weights=weights)
    model, training = fit(flux, ivar, mask, nguard=NGUARD, weights=weights, snr_bins=1)
    assert same(
        (model.mean, model.cov, model.n_train, model.edges, model.weighted, training),
        (
            default.mean,
            default.cov,
            default.n_train,
            default.edges,
            default.weighted,
            default_mask,
        ),
    )


def test_a_bin_below_its_requirement_raises_naming_it():
    # Bin 2 holds the 12 spectra of highest S/N, too few for 12 pixels: 11 and one
    # without a good pixel, whose S/N is inf and which maxbadpix removes. One of
    # them has weight 0. The counts leave out the spectra of bin 1 that maxbadpix
    # and weight 0 remove.
    flux, ivar, mask = spectra()
    mask[0, :5] = True
    ratio = snr(flux, ivar, mask)
    mask[1] = True
    ratio[1] = np.inf
    order = np.argsort(ratio)
    edges = (float(ratio[order[20]]), float(ratio[order[188]]))
    assert edges[0] < ratio[0] < edges[1]
    weights = np.ones(NSPEC)
    weights[order[[195, 102]]] = 0.0
    message = (
        rf"^fit: pass 1 of S/N bin 2 \(snr >= {edges[1]}\) needs at least 13 spectra "
        r"for a mean-centred model of 12 pixels, got 10: of 12 spectra, maxbadpix "
        r"removed 1, weight 0 removed 1 and full masks removed 0$"
    )
    calls, progress = recorder()
    with pytest.raises(ValueError, match=message):
        fit(
            flux,
            ivar,
            mask,
            maxbadpix=4,
            weights=weights,
            snr_bins=edges,
            progress=progress,
        )
    # Before bins 0 and 1 clip.
    assert calls == []
    # A bin between two edges names both.
    edges = (float(ratio[order[20]]), float(ratio[order[32]]))
    message = rf"^fit: pass 1 of S/N bin 1 \({edges[0]} <= snr < {edges[1]}\) needs "
    with pytest.raises(ValueError, match=message):
        fit(flux, ivar, mask, snr_bins=edges)


def test_tied_quantiles_leave_a_bin_empty_and_raise_naming_it():
    # 150 copies of row 0 tie the quantiles at 1/3 and 2/3, so bin 1 between them is
    # empty.
    flux, ivar, mask = spectra()
    for array in (flux, ivar, mask):
        array[50:] = array[0]
    tie = snr(flux[:1], ivar[:1], mask[:1])[0]
    message = rf"^fit: pass 1 of S/N bin 1 \({tie} <= snr < {tie}\) needs .* got 0"
    with pytest.raises(ValueError, match=message):
        fit(flux, ivar, mask, nguard=NGUARD, snr_bins=3)


def test_weights_equal_within_each_bin_give_an_unweighted_model():
    # Only the ratios of the weights in a bin matter, so f_s holds (design A13).
    flux, ivar, mask = spectra()
    edges = (float(np.median(snr(flux, ivar, mask))),)
    bins = np.searchsorted(edges, snr(flux, ivar, mask), side="right")
    options = {"nguard": NGUARD, "snr_bins": edges}
    model, training = fit(flux, ivar, mask, **options)
    scaled, scaled_mask = fit(flux, ivar, mask, **options, weights=1.0 + bins)
    assert scaled.weighted is False
    # Unequal weights in one bin make the model weighted.
    weights = np.where(bins == 1, np.linspace(1, 2, NSPEC), 1.0)
    assert fit(flux, ivar, mask, **options, weights=weights)[0].weighted is True
    assert same(
        (scaled.mean, scaled.cov, scaled_mask), (model.mean, model.cov, training)
    )


def test_snr_bins_without_a_used_spectrum_raises_the_requirement_error():
    # No quantiles exist, so the fit has one bin and raises as without snr_bins.
    flux, ivar, mask = spectra()
    message = (
        r"^fit: pass 1 needs at least 13 spectra .* got 0: .* weight 0 removed 200"
    )
    with pytest.raises(ValueError, match=message):
        fit(flux, ivar, mask, weights=np.zeros(NSPEC), snr_bins=3)
