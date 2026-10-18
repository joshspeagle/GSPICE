import dataclasses
import itertools
import math
from typing import NamedTuple

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from numpy.linalg import LinAlgError
from scipy.optimize import minimize_scalar

from _golden import arrays, checked, same
from gspice import (
    Blocks,
    LowRank,
    Model,
    _model,
    _predict,
    _repair,
    _spectra,
    finite_sample_factor,
    fit,
    fit_blocks,
    infill,
    pixelwise,
    repair,
    simulate,
)
from gspice._blocks import block_chi2
from gspice._predict import filled
from gspice._repair import _exclude_extended
from oracle import condition


def exclusion(M, I, K, *numbers):
    # _exclude_extended(M, I, K, runlength, halfwidth, minfrac), after a check that it
    # changes no input array and returns a new one.
    return checked(lambda M, I, K: _exclude_extended(M, I, K, *numbers), M, I, K)


def grid(*rows):
    # One string per spectrum and one character per pixel: "." is in neither set, "M"
    # is replaced and "I" is ignored.
    return np.array([list(row) for row in rows])


# (runlength, halfwidth, rows, expected rows), with K = M | I and no fallback.
WIDENING = [
    pytest.param(5, 2, ["......IIIII....."], ["....IIIIIIIII..."], id="runlength"),
    pytest.param(5, 2, ["......IIII......"], ["......IIII......"], id="runlength-1"),
    pytest.param(
        5,
        3,
        ["IIIII...........", "...........IIIII"],
        ["IIIIIIII........", "........IIIIIIII"],
        id="runs at the ends",
    ),
    pytest.param(
        5,
        3,
        ["IIII............", "............IIII"],
        ["IIII............", "............IIII"],
        id="runs of runlength-1 at the ends",
    ),
    pytest.param(
        5,
        1,
        ["...IIIII........", "........IIIII..."],
        ["..IIIIIII.......", ".......IIIIIII.."],
        id="runs near the ends",
    ),
    # A window of an even length has no centre pixel.
    pytest.param(
        4,
        1,
        [
            "IIII............",
            "............IIII",
            "......IIII......",
            "...III..........",
        ],
        [
            "IIIII...........",
            "...........IIIII",
            ".....IIIIII.....",
            "...III..........",
        ],
        id="even runlength",
    ),
    pytest.param(
        1, 1, ["..I.......M....."], [".III.....IMI...."], id="runlength 1 counts M"
    ),
    pytest.param(
        5,
        2,
        ["..IIIII..II...IIIII....."],
        ["IIIIIIIIIII.IIIIIIIII..."],
        id="several runs",
    ),
    pytest.param(
        5, 16, ["M.....IIIII....."], ["MIIIIIIIIIIIIIII"], id="halfwidth npix"
    ),
    # M counts toward a run and never joins the ignored set.
    pytest.param(
        5,
        3,
        [".....MMIIM......"],
        ["..IIIMMIIMIII..."],
        id="M in the run and its reach",
    ),
    # Neither a run nor its widening wraps to the next or the previous row.
    pytest.param(
        5,
        3,
        [
            "..............II",
            "III.............",
            "...........IIIII",
            "................",
            "IIIII...........",
        ],
        [
            "..............II",
            "III.............",
            "........IIIIIIII",
            "................",
            "IIIIIIII........",
        ],
        id="no wrapping between rows",
    ),
    # A run of npix - 1 pixels. The window, cut to npix + 1 pixels, lets it through only
    # at runlength npix - 1, and allocates no buffer of runlength pixels.
    *(
        pytest.param(
            runlength,
            1,
            ["IIIIIIIIIIIIIII."],
            [expected],
            id=f"run of npix-1, runlength {runlength}",
        )
        for runlength, expected in [
            (15, "IIIIIIIIIIIIIIII"),
            (16, "IIIIIIIIIIIIIII."),
            (17, "IIIIIIIIIIIIIII."),
            (10**12, "IIIIIIIIIIIIIII."),
        ]
    ),
]


@pytest.mark.parametrize(("runlength", "halfwidth", "rows", "expected"), WIDENING)
def test_runs_of_the_counting_set_widen_the_ignored_set_within_each_row(
    runlength, halfwidth, rows, expected
):
    state = grid(*rows)
    M, I = state == "M", state == "I"
    result = exclusion(M, I, M | I, runlength, halfwidth, 0.0)
    assert same(result, grid(*expected) == "I")


# With runlength 5 and halfwidth 2, widening leaves 7, 5 and 6 of the 16 pixels of these
# spectra to condition on.
ROWS = ["......IIIII.....", "MM....IIIII.....", "M.....IIIII....."]
WIDENED = ["....IIIIIIIII...", "MM..IIIIIIIII...", "M...IIIIIIIII..."]


@pytest.mark.parametrize(
    ("minfrac", "widened"),
    [
        # floor(minfrac * 16) is 5, 6, 7 and 8: each spectrum meets its count exactly
        # once, and misses it by one pixel at the next.
        (0.3125, [0, 1, 2]),
        (0.375, [0, 2]),
        (0.4375, [0]),
        (0.5, []),
        # 7.2 is floored.
        (0.45, [0]),
    ],
)
def test_a_spectrum_keeps_its_ignored_set_when_widening_leaves_too_few_pixels(
    minfrac, widened
):
    state = grid(*ROWS)
    M, I = state == "M", state == "I"
    expected = [WIDENED[s] if s in widened else ROWS[s] for s in range(3)]
    result = exclusion(M, I, M | I, 5, 2, minfrac)
    assert same(result, grid(*expected) == "I")


def test_the_floor_of_minfrac_times_npix_is_taken_in_float_arithmetic():
    # 0.29 * 100 is 28.999999999999996, so the floor is 28 and not 29. Widening leaves
    # 28 of 100 pixels to condition on.
    state = grid("." * 5 + "I" * 62 + "." * 33)
    M, I = state == "M", state == "I"
    result = exclusion(M, I, M | I, 5, 5, 0.29)
    assert same(result, grid("I" * 72 + "." * 28) == "I")


def test_a_counting_set_inside_m_and_i_can_widen_a_spectrum_that_m_and_i_cannot():
    # Design decision Q4: the fallback is per spectrum. Both runs widen the ignored set
    # to 22 of 40 pixels, which leaves 18 to condition on, below floor(0.5 * 40). The
    # first run alone widens it to 16 pixels, which leaves 24.
    state = grid("." * 5 + "I" * 5 + "." * 15 + "I" * 5 + "." * 10)
    M, I = state == "M", state == "I"
    first_run = I.copy()
    first_run[:, 25:] = False
    expected = grid("." * 2 + "I" * 11 + "." * 12 + "I" * 5 + "." * 10) == "I"
    assert same(exclusion(M, I, M | I, 5, 3, 0.5), I)
    assert same(exclusion(M, I, first_run, 5, 3, 0.5), expected)


def reference(M, I, K, runlength, halfwidth, minfrac):
    # Step 4 of design section 5.4 with plain loops, spectrum by spectrum.
    nspec, npix = M.shape
    result = I.copy()
    for s in range(nspec):
        widened = I[s].copy()
        stop = 0
        for flagged, run in itertools.groupby(K[s]):
            start, stop = stop, stop + len(list(run))
            if flagged and stop - start >= runlength:
                widened[max(start - halfwidth, 0) : stop + halfwidth] = True
        widened[M[s]] = False
        n_cond = sum(not (m or w) for m, w in zip(M[s], widened, strict=True))
        if n_cond >= math.floor(minfrac * npix):
            result[s] = widened
    return result


@st.composite
def cases(draw):
    nspec = draw(st.integers(1, 8))
    npix = draw(st.integers(4, 20))
    shape = (nspec, npix)
    runlength = draw(st.integers(1, 6))
    halfwidth = draw(st.integers(0, 6))
    # Per pixel: in neither set (0), replaced (1) or ignored (2). Each spectrum has
    # scattered pixels and a run of runlength - 1 to runlength + 1 pixels, so that most
    # spectra have a run to widen around.
    state = draw(hnp.arrays(np.int8, shape, elements=st.sampled_from([0, 0, 0, 1, 2])))
    for row in state:
        start = draw(st.integers(0, npix - 1))
        run = row[start : start + draw(st.integers(runlength - 1, runlength + 1))]
        run[:] = draw(hnp.arrays(np.int8, run.shape, elements=st.integers(1, 2)))
    M, I = state == 1, state == 2
    # K is M | I in some spectra. In the others it leaves out a quarter of those pixels.
    leave_out = draw(
        hnp.arrays(bool, shape, elements=st.sampled_from([False, False, False, True]))
    )
    subset = draw(hnp.arrays(bool, (nspec, 1)))
    K = (M | I) & ~(leave_out & subset)
    # floor(minfrac * npix) is the number of pixels that widening leaves in a spectrum
    # that it changes, or one more or fewer, so that the fallback is decided at its
    # boundary. If widening changes no spectrum, any count will do.
    widened = reference(M, I, K, runlength, halfwidth, 0.0)
    left = np.count_nonzero(~(M | widened), axis=1)
    counts = left[(widened != I).any(axis=1)].tolist() or left.tolist()
    count = draw(st.sampled_from(counts)) + draw(st.sampled_from([0, 1, -1]))
    # The half pixel keeps floor(minfrac * npix) from falling short of count by rounding.
    minfrac = draw(st.sampled_from([count, count + 0.5])) / npix
    return M, I, K, runlength, halfwidth, float(np.clip(minfrac, 0.0, 1.0))


@given(cases())
def test_exclusion_matches_a_direct_implementation_of_step_4(case):
    assert same(exclusion(*case), reference(*case))


# The spectra of artifact_case: NPIX pixels with a_s = A, the noise 1 in working units.
NPIX, NSPEC, A = 120, 10, 16.0
PIXELS = np.arange(NPIX)
# Blocks of WIDTH pixels every 15 pixels, 7 in all, with the guard BLOCK_GUARD and the
# threshold 2.5 of chi2_A, which a clean block exceeds with probability 1.0e-5, as
# chi2_A |A| ~ chi2_30.
WIDTH, BLOCK_GUARD = 30, 5
STARTS = np.arange(0, NPIX - WIDTH + 1, 15)
# The options of repair for artifact_case besides the defaults: z on windows of 11
# pixels, and an exclusion that leaves 60 pixels or more in the spectra it widens.
OPTIONS = {"nguard": 5, "runlength": 5, "halfwidth": 10}
NSIGMA_REPLACE, NSIGMA_IGNORE, MINFRAC = 10.0, 5.0, 0.5
# ivar at every pixel: a_s stays A, and no pixel is bad.
FULL = np.full((NSPEC, NPIX), A**2)


def artifact_case():
    # A model and NSPEC spectra drawn from it, with artifacts. The working vectors
    # x = A flux - m are drawn from N(0, C), C a squared-exponential kernel of length 6
    # and variance 9 plus the noise 1, around the mean m = A truth of narrow
    # absorption lines. Rows:
    # 0, 9: clean.
    # 1: a cosmic ray of 60 sigma on pixels 30 and 31, and masked pixel 32, whose
    #    interpolated value is so far from its prediction that it replaces pixel 33
    #    (design decision A12); the replaced pixels widen the ignored set.
    # 2: a bad column of flux 4 on pixels 60 to 69, whose replaced and ignored pixels
    #    form a run that widens the ignored set.
    # 3: pixels 45 to 74 shifted by 1 pixel: only the block pass flags them, and the
    #    block widens the ignored set.
    # 4: masked pixels 10 to 15, 80 and 100, with ivar 0 and flux 0. The masked run
    #    widens the ignored set for count_on="all" only.
    # 5: two bad columns, whose widening would leave too few pixels: no exclusion.
    # 6: cosmic rays of 40 sigma at both ends.
    # 7: masked pixels 0 to 2.
    # 8: a cosmic ray of 14 sigma on pixel 70, which is ignored but not replaced.
    # artifact_statistics checks that every |z|, block ratio and exclusion count is
    # away from its threshold.
    C = 9 * np.exp(-0.5 * ((PIXELS[:, np.newaxis] - PIXELS) / 6.0) ** 2) + np.eye(NPIX)
    centres = np.arange(4, NPIX, 8)
    lines = np.exp(-0.5 * ((PIXELS[:, np.newaxis] - centres) / 1.2) ** 2)
    truth = 1 - 0.4 * lines.sum(axis=1)
    mean = A * truth
    blocks = Blocks(STARTS, WIDTH, BLOCK_GUARD, np.full((1, STARTS.size), 2.5))
    model = Model(mean[np.newaxis], (C,), (1000,), blocks=blocks)
    rng = np.random.default_rng(11)
    x = rng.standard_normal((NSPEC, NPIX)) @ np.linalg.cholesky(C).T
    x[1, 30:32] += 60.0
    for s, columns in ((2, [slice(60, 70)]), (5, [slice(20, 28), slice(85, 93)])):
        for column in columns:
            x[s, column] = 4 * A - mean[column]
    block = slice(45, 75)
    x[3, block] += A * (np.interp(PIXELS[block] - 1.0, PIXELS, truth) - truth[block])
    x[6, [0, NPIX - 1]] += 40.0
    x[8, 70] += 14.0
    flux = (x + mean) / A
    ivar = FULL.copy()
    mask = np.zeros(flux.shape, dtype=bool)
    mask[1, 32] = mask[4, 10:16] = mask[4, [80, 100]] = mask[7, :3] = True
    ivar[mask] = 0.0
    flux[mask] = 0.0
    return model, flux, ivar, mask


class Steps(NamedTuple):
    z1: np.ndarray
    z2: np.ndarray
    ratios: list
    flagged: np.ndarray
    Mb: np.ndarray
    M: np.ndarray
    I: np.ndarray
    K: np.ndarray
    left: np.ndarray
    ignored: np.ndarray


def steps(model, flux, ivar, mask, count_on="all", **options):
    # Design section 5.4 with the public pixelwise and infill, spectrum by spectrum in
    # the block pass, for spectra with ivar A^2 at good pixels, so that a_s = A at
    # every step. ratios holds the ratios of each scan of each spectrum.
    options = OPTIONS | options
    nguard, d = options["nguard"], options.get("dilate", 1)
    bad = mask | (ivar == 0)
    n = len(flux)
    ftilde = np.array(
        [np.interp(PIXELS, PIXELS[~b], f[~b]) for f, b in zip(flux, bad, strict=True)]
    )
    # z at every pixel, bad pixels included (design decision A12).
    z1 = pixelwise(model, ftilde, FULL[:n], nguard=nguard).z
    M1 = _spectra.dilate(np.abs(z1) > NSIGMA_REPLACE, d) | bad
    I1 = _spectra.dilate(np.abs(z1) > NSIGMA_IGNORE, d) & ~M1
    blocks = model.blocks
    cover = (PIXELS >= blocks.starts[:, np.newaxis]) & (
        PIXELS < blocks.starts[:, np.newaxis] + blocks.width
    )
    flagged = np.zeros((n, blocks.starts.size), dtype=bool)
    Mb = np.zeros(flux.shape, dtype=bool)
    ratios = []
    for s in range(n if options.get("blockwise", True) else 0):
        while True:
            row = slice(s, s + 1)
            replace = M1[row] | I1[row] | Mb[row]
            filled = infill(model, flux[row], ivar[row], mask[row], replace=replace)
            ratio = np.empty(blocks.starts.size)
            for k, start in enumerate(blocks.starts):
                G = (PIXELS >= start - blocks.nguard) & (
                    PIXELS < start + blocks.width + blocks.nguard
                )
                chi2 = infill(
                    model, filled.flux, FULL[:1], replace=cover[k], ignore=G & ~cover[k]
                ).chi2
                ratio[k] = chi2[0] / blocks.width / blocks.thresholds[0, k]
            ratios.append(ratio.copy())
            ratio[flagged[s]] = 0.0
            k = np.argmax(ratio)
            if ratio[k] <= 1:
                break
            flagged[s, k] = True
            Mb[s] |= cover[k]
    X1 = infill(model, flux, ivar, mask, replace=M1 | Mb, ignore=I1).flux
    z2 = pixelwise(model, X1, FULL[:n], nguard=nguard).z
    M = M1 | Mb | _spectra.dilate(np.abs(z2) > NSIGMA_REPLACE, d)
    I = _spectra.dilate(np.abs(z2) > NSIGMA_IGNORE, d) & ~M
    K = M | I
    if count_on == "flagged":
        core = (np.abs(z1) > NSIGMA_REPLACE) | (np.abs(z2) > NSIGMA_IGNORE)
        K = (core & ~bad) | Mb
    exclusion = options["runlength"], options["halfwidth"]
    if exclusion[0] is None:
        return Steps(z1, z2, ratios, flagged, Mb, M, I, K, np.full(n, NPIX), I)
    left = np.count_nonzero(~(M | reference(M, I, K, *exclusion, 0.0)), axis=1)
    ignored = reference(M, I, K, *exclusion, MINFRAC)
    return Steps(z1, z2, ratios, flagged, Mb, M, I, K, left, ignored)


def assert_margins(case):
    # Every |z| at least 1e-6 relative from both thresholds, every block ratio 1e-6
    # from 1, and every count of pixels left by the exclusion 2 from its floor, so that
    # rounding, and the Schur route of r26, cannot flip a decision.
    for z in (case.z1, case.z2):
        for nsigma in (NSIGMA_REPLACE, NSIGMA_IGNORE):
            assert np.all(np.abs(np.abs(z) - nsigma) >= 1e-6 * nsigma)
    assert np.all(np.abs(np.array(case.ratios) - 1) >= 1e-6)
    assert np.all(np.abs(case.left - math.floor(MINFRAC * NPIX)) >= 2)


def captured(monkeypatch):
    # The arguments of each call of _exclude_extended in repair.
    calls = []

    def exclude(*args):
        calls.append(args)
        return _exclude_extended(*args)

    monkeypatch.setattr(_repair, "_exclude_extended", exclude)
    return calls


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"count_on": "flagged"},
        {"dilate": 0},
        {"dilate": 2},
        {"blockwise": False},
        {"runlength": None},
    ],
    ids=["all", "flagged", "dilate=0", "dilate=2", "blockwise=False", "runlength=None"],
)
def test_repair_follows_the_steps_of_section_5_4(monkeypatch, options):
    # Done when 2 of issue #59: for "flagged", K is the direct construction of step 4,
    # (((|z1| > nsigma_replace) | (|z2| > nsigma_ignore)) minus bad) | M_b.
    model, flux, ivar, mask = artifact_case()
    case = steps(model, flux, ivar, mask, **options)
    assert_margins(case)
    calls = captured(monkeypatch)
    result = repair(model, flux, ivar, mask, **(OPTIONS | options))
    assert same(result.replaced, case.M) and same(result.ignored, case.ignored)
    expected = case.flagged if options.get("blockwise", True) else None
    assert same(result.blocks, expected)
    if options.get("runlength", 5) is None:
        assert calls == []
        return
    ((M, I, K, *numbers),) = calls
    assert numbers == [OPTIONS["runlength"], OPTIONS["halfwidth"], MINFRAC]
    assert same(M, case.M) and same(I, case.I) and same(K, case.K)
    if options.get("count_on") == "flagged":
        # The masked run of row 4 is outside K, and K is inside M | I.
        assert not K[4, 10:16].any() and not (K & ~(M | I)).any()
        assert not (case.ignored != case.I)[4].any()


def test_the_result_is_infill_with_the_replaced_and_ignored_pixels():
    # Step 5, with inflate, covariance and draws: unreplaced pixels are the input.
    model, flux, ivar, mask = artifact_case()
    options = {"inflate": True, "covariance": True}
    result = repair(model, flux, ivar, mask, **OPTIONS, **options, rng=rng())
    expected = infill(
        model,
        flux,
        ivar,
        mask,
        replace=result.replaced,
        ignore=result.ignored,
        **options,
        rng=rng(),
    )
    assert same(arrays(result), arrays(expected))
    assert same(result.covariance(2), expected.covariance(2))
    assert same(result.flux[~result.replaced], flux[~result.replaced])


def rng():
    return np.random.default_rng(5)


def test_a_noise_model_follows_the_steps_with_one_precision_matrix_per_spectrum(
    monkeypatch,
):
    # The noise model of the distribution of artifact_case in flux units: a_s = A at
    # every spectrum, so nu_s = 1/A^2 at every pixel and C_s = C/A^2. The steps of the
    # public pixelwise and infill of the noise model give its masks. Step 5 uses the
    # P_s of the guard weights, where infill uses the factor of C_s (design section
    # 12, (e)): it is that infill bit for bit when infill forms P_s first, and agrees
    # with the public infill to rounding.
    model, flux, ivar, mask = artifact_case()
    signal = (model.cov[0] - np.eye(NPIX)) / A**2
    noise = Model(model.mean / A, (signal,), (1000,), noise=True, blocks=model.blocks)
    case = steps(noise, flux, ivar, mask)
    assert_margins(case)
    monkeypatch.setattr(_spectra, "CHUNK", 4 * NPIX)
    calls = []
    options = {"covariance": True, "progress": lambda *args: calls.append(args)}
    result = repair(noise, flux, ivar, mask, **OPTIONS, **options, rng=rng())
    assert same(result.replaced, case.M) and same(result.ignored, case.ignored)
    assert same(result.blocks, case.flagged)
    inputs = (noise, flux, ivar, mask)
    masks = {"replace": result.replaced, "ignore": result.ignored}
    public = infill(*inputs, **masks, covariance=True)
    # Each route lies within 10 kappa eps of the exact conditional (design section 11),
    # with C_s = C / A^2 of the condition number of C.
    tol = 2 * 10 * condition(model.cov[0]) * np.finfo(float).eps
    replaced = result.replaced
    sigma = np.sqrt(public.var[replaced])
    assert np.all(np.abs(result.flux - public.flux)[replaced] <= tol * sigma)
    assert np.all(np.abs(result.var - public.var)[replaced] <= tol * sigma**2)
    assert np.all(np.abs(result.chi2 - public.chi2) <= tol * (public.chi2 + public.dof))
    for s in range(NSPEC):
        cov = public.covariance(s)[1]
        scale = np.sqrt(np.outer(np.diagonal(cov), np.diagonal(cov)))
        assert np.all(np.abs(result.covariance(s)[1] - cov) <= tol * scale)

    def with_precision(*args):
        cover = noisy(*args)
        cover._precision(0)
        return cover

    noisy = _predict.noisy
    monkeypatch.setattr(_predict, "noisy", with_precision)
    expected = infill(
        noise,
        flux,
        ivar,
        mask,
        replace=result.replaced,
        ignore=result.ignored,
        covariance=True,
        rng=rng(),
    )
    assert same(arrays(result), arrays(expected))
    for s in range(NSPEC):
        assert same(result.covariance(s), expected.covariance(s))
    # A noise model works in chunks of one spectrum, whatever CHUNK: after steps 1
    # to 4 and after step 5 of each.
    assert calls == [
        ("repair", done, NSPEC) for done in range(1, NSPEC + 1) for _ in "15"
    ]
    # The dense model of the same distribution in scaled units finds the same pixels.
    scaled = repair(model, flux, ivar, mask, **OPTIONS)
    assert same(result.replaced, scaled.replaced)
    assert same(result.ignored, scaled.ignored)
    assert same(result.blocks, scaled.blocks)
    empty = repair(noise, flux[:0], ivar[:0], mask[:0], **OPTIONS, rng=rng())
    shapes = [array.shape for array in arrays(empty)[:8]]
    assert shapes == [(0, NPIX)] * 4 + [(0,)] * 3 + [(0, NPIX)]
    assert empty.blocks.shape == (0, STARTS.size)


def test_repair_matches_r26_gp_repair():
    # Done when 1 of issue #59 (design section 14, item 4.3): r26 with max_block_iter
    # = nblock, count_on "all", dilate 1 and ivar 0 only at masked pixels, on the same
    # blocks and thresholds. r26 computes its pixelwise weights by the Schur route, so
    # the margins of artifact_case keep every decision away from rounding.
    from r26.repairUtils import block_operators, gp_repair, make_blocks

    model, flux, ivar, mask = artifact_case()
    case = steps(model, flux, ivar, mask)
    assert_margins(case)
    # The case holds each kind of artifact that the brief of #59 names.
    widened = (case.ignored != case.I).any(axis=1)
    # The cosmic rays, the bad column and the masked pixels are replaced.
    assert case.M[1, 30:32].all() and case.M[6, [0, NPIX - 1]].all()
    assert case.M[2, 60:70].all() and case.M[mask].all()
    # Only the z of masked pixel 32 of row 1 replaces pixel 33.
    assert np.abs(case.z1[1, 32]) > NSIGMA_REPLACE > np.abs(case.z1[1, [33, 34]]).max()
    assert case.M[1, 33] and np.abs(case.z2[1, 32:35]).max() < NSIGMA_REPLACE
    # The block pass flags the shifted block, whose pixels no |z| flags, and nothing
    # else; one more scan of row 3 finds no other block.
    assert np.flatnonzero(case.flagged.any(axis=1)).tolist() == [3]
    assert np.flatnonzero(case.flagged[3]).tolist() == [3]
    assert np.abs(case.z1[3]).max() < NSIGMA_IGNORE
    # The exclusion widens rows 1 to 4, and falls back in row 5.
    assert np.flatnonzero(widened).tolist() == [1, 2, 3, 4]
    assert case.left[5] < math.floor(MINFRAC * NPIX)
    # The weak cosmic ray and its neighbours are ignored.
    assert case.I[8, 69:72].all() and not case.M[8].any()
    blocks = model.blocks
    stride = int(blocks.starts[1] - blocks.starts[0])
    assert same(np.array([b[0] for b in make_blocks(NPIX, WIDTH, stride)]), STARTS)
    C = model.cov[0]
    newflux, _, info = gp_repair(
        flux,
        ivar,
        C,
        model.mean[0],
        mask=mask,
        nguard=OPTIONS["nguard"],
        block_ops=block_operators(C, make_blocks(NPIX, WIDTH, stride), BLOCK_GUARD),
        block_thresh=blocks.thresholds[0],
        max_block_iter=STARTS.size,
        runlength=OPTIONS["runlength"],
        exclusion_halfwidth=OPTIONS["halfwidth"],
        minfrac=MINFRAC,
    )
    result = repair(model, flux, ivar, mask, **OPTIONS)
    assert same(result.replaced, info["replace"])
    assert same(result.ignored, info["ignore"])
    assert same(result.blocks, info["block_flag"])
    # Q14: only var differs, so it is not compared.
    assert np.all(np.abs(result.flux - newflux) <= 1e-12 * np.abs(newflux))


def test_each_spectrum_is_scanned_with_the_thresholds_of_its_bin():
    # Two bins with the model of artifact_case. Row 3, with the shifted block, and the
    # rows of higher S/N are in bin 1. The block pass flags only in the bin whose
    # thresholds are those of artifact_case, 1000 times lower than the other.
    model, flux, ivar, mask = artifact_case()
    ratio = np.median(flux * np.sqrt(ivar), axis=1)
    bins = (ratio >= ratio[3]).astype(int)
    assert 0 < bins.sum() < NSPEC - 1
    single = repair(model, flux, ivar, mask, **OPTIONS)
    for low in (0, 1):
        thresholds = np.full((2, STARTS.size), 2500.0)
        thresholds[low] = model.blocks.thresholds[0]
        binned = Model(
            np.repeat(model.mean, 2, axis=0),
            model.cov * 2,
            (1000, 1000),
            edges=(ratio[3],),
            blocks=dataclasses.replace(model.blocks, thresholds=thresholds),
        )
        result = repair(binned, flux, ivar, mask, **OPTIONS)
        assert same(result.blocks, single.blocks & (bins == low)[:, np.newaxis])


def test_the_block_pass_flags_blocks_until_no_ratio_exceeds_1():
    # Design decision Q5, with thresholds below every positive chi2_A: a spectrum
    # flags blocks until its replaced pixels cover all pixels, whose conditional mean
    # is then the prior mean 0, with chi2_A = 0.
    model, flux, ivar, mask = artifact_case()
    low = dataclasses.replace(model.blocks, thresholds=np.full((1, STARTS.size), 1e-9))
    model = dataclasses.replace(model, blocks=low)
    case = steps(model, flux, ivar, mask)
    assert_margins(case)
    result = repair(model, flux, ivar, mask, **OPTIONS)
    assert same(result.blocks, case.flagged)
    assert 3 * NSPEC < np.count_nonzero(result.blocks) < STARTS.size * NSPEC


def test_a_block_is_flagged_at_most_once(monkeypatch):
    # Design decision Q5. Block 3 has a threshold below its chi2_A after its own
    # infill, and the others one above every chi2_A: each spectrum flags block 3 once,
    # and a second flag would repeat the same round forever, so the scans are counted.
    model, flux, ivar, mask = artifact_case()
    scans = []

    def counted(factors, X):
        scans.append(len(X))
        assert len(scans) <= 2, "a block was flagged again"
        return block_chi2(factors, X)

    monkeypatch.setattr(_repair, "block_chi2", counted)
    thresholds = np.full((1, STARTS.size), 1e9)
    thresholds[0, 3] = 1e-9
    one = dataclasses.replace(model.blocks, thresholds=thresholds)
    result = repair(dataclasses.replace(model, blocks=one), flux, ivar, mask, **OPTIONS)
    assert same(result.blocks, np.repeat(STARTS[np.newaxis] == STARTS[3], NSPEC, 0))


def test_inputs_are_unchanged_and_outputs_are_read_only_and_not_shared():
    model, flux, ivar, mask = artifact_case()

    def outputs(*spectra):
        result = repair(model, *spectra, **OPTIONS, rng=rng())
        return (*arrays(result), result.blocks)

    assert not any(o.flags.writeable for o in checked(outputs, flux, ivar, mask))


def test_the_block_pass_infills_again_with_the_rows_of_the_caller(monkeypatch):
    # With chunks of two rows, row 3 is the second row of its chunk and the only one
    # that flags a block, so its infill again gets row 3, which messages then name.
    model, flux, ivar, mask = artifact_case()
    monkeypatch.setattr(_spectra, "CHUNK", 2 * NPIX)
    calls = []

    def spy(model, X, bins, rows, G, M):
        calls.append((len(X), rows.tolist()))
        return filled(model, X, bins, rows, G, M)

    monkeypatch.setattr(_repair, "filled", spy)
    repair(model, flux, ivar, mask, **OPTIONS)
    assert (1, [3]) in calls and all(n == len(rows) for n, rows in calls)


def test_chunks_change_no_mask_and_the_flux_only_by_rounding(monkeypatch):
    model, flux, ivar, mask = artifact_case()
    whole = repair(model, flux, ivar, mask, **OPTIONS)
    monkeypatch.setattr(_spectra, "CHUNK", 3 * NPIX)
    chunked = repair(model, flux, ivar, mask, **OPTIONS)
    for name in ("replaced", "ignored", "blocks", "dof"):
        assert same(getattr(chunked, name), getattr(whole, name))
    # Products with fewer rows can round differently.
    for name in ("flux", "var", "chi2"):
        np.testing.assert_allclose(getattr(chunked, name), getattr(whole, name), 1e-12)


def test_progress_counts_the_spectra_of_each_chunk_in_detection_and_in_infill(
    monkeypatch,
):
    model, flux, ivar, mask = artifact_case()
    monkeypatch.setattr(_spectra, "CHUNK", 2 * NPIX)
    calls = []
    repair(
        model,
        flux[:5],
        ivar[:5],
        mask[:5],
        **OPTIONS,
        progress=lambda *args: calls.append(args),
    )
    assert calls == [("repair", 2, 5), ("repair", 4, 5), ("repair", 5, 5)] * 2


# A precision matrix that passes its check but is not positive definite at pixel 25,
# as rounding can make it near the rcond limit (as in test_blocks).
BROKEN = np.eye(NPIX)
BROKEN[25, 25] = -1.0


@pytest.mark.parametrize(
    ("options", "message", "chunks"),
    [
        # Block 0 is the first whose G holds pixel 25.
        ({}, "on the pixels G, for block 0", []),
        # With the prior for z and no block pass, the bad columns of row 5, in the
        # third chunk of two rows, put pixel 25 into the G of an infill.
        (
            {"blockwise": False},
            "on the pixels G, for the pattern of row 5",
            [("repair", 2, NSPEC), ("repair", 4, NSPEC)],
        ),
    ],
    ids=["blocks", "infill"],
)
def test_a_failed_factorization_names_the_block_at_once_or_the_row_of_the_caller(
    monkeypatch, options, message, chunks
):
    monkeypatch.setattr(_model, "precision", lambda S: BROKEN)
    monkeypatch.setattr(_spectra, "CHUNK", 2 * NPIX)
    model, flux, ivar, mask = artifact_case()
    calls = []
    error = (
        rf"^model: cov\[0\] is not positive definite in its precision matrix {message}$"
    )
    with pytest.raises(LinAlgError, match=error):
        repair(
            model,
            flux,
            ivar,
            mask,
            **(OPTIONS | {"nguard": NPIX} | options),
            progress=lambda *args: calls.append(args),
        )
    assert calls == chunks


def test_a_covariance_that_fails_the_check_of_its_precision_matrix_raises_at_once():
    # cov[0] has no inverse, also with the prior for z and no block pass.
    model, flux, ivar, mask = artifact_case()
    singular = dataclasses.replace(model, cov=(np.ones((NPIX, NPIX)),))
    calls = []
    with pytest.raises(LinAlgError, match=r"^model: cov\[0\] is not positive definite"):
        repair(
            singular,
            flux,
            ivar,
            mask,
            nguard=NPIX,
            blockwise=False,
            progress=lambda *args: calls.append(args),
        )
    assert calls == []


def test_inflate_raises_before_the_final_infill_where_f_s_is_not_defined():
    # Design section 8. Rows 1 to 5 condition on at most 96 pixels, and row 6, with
    # its two cosmic rays, on 113, so that N_s - N_k - 3 <= 0 at N_s = 100.
    model, flux, ivar, mask = artifact_case()
    model = dataclasses.replace(model, n_train=(100,))
    calls = []
    error = r"^inflate: .*, got N_s = 100 and N_k = 113 at row 6$"
    with pytest.raises(ValueError, match=error):
        repair(
            model,
            flux,
            ivar,
            mask,
            **OPTIONS,
            inflate=True,
            progress=lambda *args: calls.append(args),
        )
    assert calls == [("repair", NSPEC, NSPEC)]


def test_zero_spectra_give_empty_arrays():
    model, *_ = artifact_case()
    empty = np.zeros((0, NPIX))
    result = repair(model, empty, empty)
    assert [a.shape for a in arrays(result)[:7]] == [(0, NPIX)] * 4 + [(0,)] * 3
    assert result.blocks.shape == (0, STARTS.size)


def test_a_strong_cosmic_ray_and_a_bad_column_are_replaced():
    # The fast end-to-end test of design section 11: simulate, fit, fit_blocks and
    # repair at npix 100.
    rng = np.random.default_rng(2026)
    train = simulate(600, 100, rng=rng, artifacts=0.2)
    clean = simulate(1000, 100, rng=rng)
    test = simulate(5, 100, rng=rng)
    model, _ = fit(train.flux, train.ivar)
    model = fit_blocks(model, clean.flux, clean.ivar, width=40, stride=20)
    flux = test.flux.copy()
    flux[0, 50] += 30 / np.sqrt(test.ivar[0, 50])
    flux[1, 20:30] = 2.0
    result = repair(model, flux, test.ivar)
    assert result.replaced[0, 50] and result.replaced[1, 20:30].all()


def mean_near(values, at, centre, width):
    # The mean of values where at is within width of centre.
    return values[np.abs(at - centre) <= width].mean()


def ratio_rms(z2, count, rows):
    # sqrt(sum z^2 / sum count) over rows, the per-spectrum sums z2 and counts.
    return np.sqrt(z2[rows].sum() / count[rows].sum())


@pytest.mark.slow
def test_repair_of_simulated_spectra_replaces_the_artifacts():
    # The end-to-end test of design section 11, at npix 1000, where the exclusion of
    # 150 pixels on either side of a flagged block of 150 pixels can leave half the
    # pixels. The test spectra have a twin without artifacts: simulate draws the noise
    # before the artifacts, so their flux differs only at the artifacts.
    npix, ntest = 1000, 3000
    rng = np.random.default_rng(2026)
    train = simulate(3000, npix, rng=rng, artifacts=0.4)
    clean = simulate(1000, npix, rng=rng)
    test = simulate(ntest, npix, rng=np.random.default_rng(7), artifacts=0.4)
    twin = simulate(ntest, npix, rng=np.random.default_rng(7))
    assert same(test.flux[~test.artifacts], twin.flux[~test.artifacts])
    model, _ = fit(train.flux, train.ivar)
    model = fit_blocks(model, clean.flux, clean.ivar)
    result = repair(model, test.flux, test.ivar)
    unexcluded = repair(model, test.flux, test.ivar, runlength=None)
    sigma = 1 / np.sqrt(test.ivar)
    size = np.count_nonzero(test.artifacts, axis=1)
    # simulate gives cosmic rays 1 to 3 pixels, bad columns 5 to 20 and shifted
    # blocks 200 to 600.
    rays, columns, shifts = size <= 3, (size >= 5) & (size <= 20), size >= 200
    clean_rows = size == 0
    rays &= ~clean_rows
    peak = np.where(test.artifacts, (test.flux - twin.flux) / sigma, 0).max(axis=1)
    found = (result.replaced | ~test.artifacts).all(axis=1)

    # Completeness above 25 sigma of at least 98%, less 3 binomial standard errors.
    strong = rays & (peak > 25)
    n = np.count_nonzero(strong)
    assert found[strong].mean() >= 0.98 - 3 * math.sqrt(0.98 * 0.02 / n)
    assert found[columns].all()
    # False replacements per clean spectrum, not above the paper's 2.8 by more than 3
    # standard errors.
    false = np.count_nonzero(result.replaced[clean_rows], axis=1)
    assert false.mean() <= 2.8 + 3 * false.std(ddof=1) / math.sqrt(false.size)
    # The Z of the replaced pixels against the twin, and their dispersion, larger
    # without the exclusion by more than 3 standard errors of the paired difference,
    # from 1000 bootstrap samples of the spectra.
    sums = []
    for r in (result, unexcluded):
        z = (r.flux - twin.flux) / np.sqrt(r.var)
        sums.append((np.where(r.replaced, z**2, 0).sum(axis=1), r.replaced.sum(axis=1)))
    every = np.arange(ntest)
    samples = np.random.default_rng(0).integers(ntest, size=(1000, ntest))
    rms, rms_unexcluded = (ratio_rms(*s, every) for s in sums)
    differences = [ratio_rms(*sums[1], k) - ratio_rms(*sums[0], k) for k in samples]
    assert rms_unexcluded - rms > 3 * np.std(differences)

    # The report of design section 11, as a check and not an assertion.
    pixels = np.arange(npix)
    shift = np.empty(ntest)
    for s in np.flatnonzero(shifts):
        j = pixels[test.artifacts[s]]
        difference = test.flux[s, j] - twin.flux[s, j]

        def residual(d, s=s, j=j, difference=difference):
            moved = np.interp(j - d, pixels, test.truth[s]) - test.truth[s, j]
            return np.sum((moved - difference) ** 2)

        shift[s] = abs(
            min(
                (
                    minimize_scalar(residual, bounds=b, method="bounded")
                    for b in ((0.5, 3), (-3, -0.5))
                ),
                key=lambda o: o.fun,
            ).x
        )
    local = rays | columns | clean_rows
    n_cond = npix - np.count_nonzero(result.replaced | result.ignored, axis=1)
    fs = finite_sample_factor(model.n_train[0], n_cond, 1)
    _, count = sums[0]
    replaced_any = (result.replaced & test.artifacts).any(axis=1)
    print(
        f"\nreplaced pixels per clean spectrum {false.mean():.2f} (paper 2.8); "
        f"clean spectra affected {100 * (false > 0).mean():.2f}% (1.7%)\n"
        f"Z dispersion {rms:.3f}, without the exclusion {rms_unexcluded:.3f} (paper "
        f"1.02 and 1.29); f_s^1/2 {np.sqrt((fs * count).sum() / count.sum()):.3f} "
        f"(1.04); on spectra without a shifted block "
        f"{ratio_rms(*sums[0], local):.3f} and {ratio_rms(*sums[1], local):.3f}\n"
        f"rms (repaired - truth)/sigma "
        f"{np.sqrt(np.mean(((result.flux - test.truth) / sigma)[result.replaced] ** 2)):.3f}"
        f" (0.65)\ncompleteness above 25 sigma {found[strong].mean():.3f} of {n}; at "
        f"12 +- 1 sigma {mean_near(found[rays], peak[rays], 12, 1):.2f} (0.5); shifted "
        f"blocks with a replaced pixel at 1.9 +- 0.25 pixels "
        f"{mean_near(replaced_any[shifts], shift[shifts], 1.9, 0.25):.2f} (0.5)"
    )


def test_a_low_rank_model_repairs_as_the_dense_model_of_its_covariance():
    # C - I of artifact_case has rank 60 to rounding: the eigenvalues after the 60th
    # sum to 7e-14 in magnitude. The masks then agree, as assert_margins checks for
    # the dense model in the test of the steps, and the flux to rounding.
    model, flux, ivar, mask = artifact_case()
    values, V = np.linalg.eigh(model.cov[0] - np.eye(NPIX))
    cov = (LowRank(V[:, -60:], values[-60:], np.ones(NPIX)),)
    low = dataclasses.replace(model, cov=cov)
    expected = repair(model, flux, ivar, mask, **OPTIONS)
    result = repair(low, flux, ivar, mask, **OPTIONS)
    for name in ("replaced", "ignored", "blocks"):
        assert same(getattr(result, name), getattr(expected, name))
    # C has kappa 135, so a change of 1e-13 in C moves the conditional means by about
    # 1e-11 sigma at most; measured 1.5e-13 sigma.
    sigma = np.sqrt(expected.var)
    assert np.all(np.abs(result.flux - expected.flux) <= 1e-9 * sigma)
