import numpy as np
import pytest
from numpy.linalg import LinAlgError

import gspice
from _golden import same
from gspice import _model, legacy
from oracle import conditional, guard_conditional
from test_golden import (
    GOLDEN,
    MODE,
    RTOL,
    assert_close,
    check_golden_case,
    check_output,
)


def pixelwise_as_gp_interp(spec, cov, nguard, irange=None):
    # With ivar = 1, a zero mean and no centre, x is spec, and the outputs are in the
    # units of gp_interp.
    nspec, npix = spec.shape
    model = gspice.Model(np.zeros((1, npix)), (cov,), (nspec,))
    result = gspice.pixelwise(model, spec, np.ones_like(spec), nguard=nguard)
    pixels = slice(None) if irange is None else slice(irange[0], irange[1] + 1)
    return result.flux[:, pixels], result.var[:, pixels]


def infill_as_gaussian_estimate(icond, ipred, cov, Dvec, covinv, bruteforce=False):
    # With ivar = 1, a zero mean and no centre, x is Dvec, and the outputs are in the
    # units of gaussian_estimate. G holds the pixels outside icond and M those of
    # ipred; covinv is not used. The infill of the unit spectra gives the rows of
    # predoverD.
    nspec, npix = Dvec.shape
    model = gspice.Model(np.zeros((1, npix)), (cov,), (nspec,))
    sets = {"replace": ipred != 0, "ignore": icond == 0}
    result = gspice.infill(model, Dvec, np.ones_like(Dvec), **sets, covariance=True)
    kstar, predcovar = result.covariance(0)
    predkstar = result.flux[:, kstar]
    if bruteforce:
        # The 2022 layout: predkstar is (nspec,) for one predicted pixel.
        return predcovar, predkstar[:, 0] if kstar.size == 1 else predkstar, kstar
    unit = np.eye(npix)
    predoverD = gspice.infill(model, unit, np.ones_like(unit), **sets).flux[:, kstar]
    return predoverD, predcovar, predkstar, kstar


def fit_as_covar_iter_mask(flux, ivar, mask, nsigma=(20, 8, 6), maxbadpix=64):
    # The defaults of covar_iter_mask, and its passes, which mask only the flags of
    # the pass before.
    model, training = gspice.fit(
        flux, ivar, mask, nsigma=nsigma, maxbadpix=maxbadpix, cumulative=False
    )
    return model.cov[0], training.astype(np.float64)


def new_api(module, function):
    return NEW_API[function]


NEW_API = {
    "standard_scale": lambda spec, ivar, mask=None: gspice.scale(spec, ivar, mask),
    "gp_interp": pixelwise_as_gp_interp,
    "gaussian_estimate": infill_as_gaussian_estimate,
    "covar_iter_mask": fit_as_covar_iter_mask,
}

BC_NEW = {
    "standard_scale.E.with_fully_masked": "Q1",
    "standard_scale.E.ivar0.mask": "Q11",
    "standard_scale.D1.ivar0.mask": "Q11",
    "gp_interp.D3.ng5": "S11.4",
    "gp_interp.D3.ng5.weights": "S11.4",
    "gp_interp.D3.ng20": "S11.4",
    "gp_interp.D3.ng20.weights": "S11.4",
    "gaussian_estimate.D3.many21.guard5": "S11.4",
}


@pytest.mark.parametrize(
    "case_id",
    [
        case["id"]
        for case in GOLDEN.manifest["cases"]
        if case["kind"] == "golden"
        and case["function"] in NEW_API
        and case["id"] not in BC_NEW
    ],
)
def test_new_api_matches_2022_outputs(case_id):
    check_golden_case(case_id, new_api)


@pytest.mark.parametrize(
    "case_id", ["standard_scale.E.ivar0.mask", "standard_scale.D1.ivar0.mask"]
)
def test_scale_counts_ivar_0_as_bad_also_outside_the_mask(case_id):
    case = GOLDEN.cases[case_id]
    kwargs = GOLDEN.arguments(case)
    spec, ivar, mask = kwargs["spec"], kwargs["ivar"], kwargs["mask"]
    new = gspice.scale(spec, ivar, mask)
    live = legacy.standard_scale(spec, ivar, (mask != 0) | (ivar == 0))
    for output in case["outputs"]:
        mode = MODE[(case["function"], output["name"])]
        index = output["index"]
        check_output(case, output, new[index], live[index], kwargs, mode)
    # 2022 used the flux of such a pixel and counted its 0 in the mean ivar.
    old = GOLDEN.expected(case)["refscale"]
    differs = np.abs(new[1] - old) > RTOL * np.abs(old)
    np.testing.assert_array_equal(differs, ((mask == 0) & (ivar == 0)).any(axis=1))


@pytest.mark.parametrize(
    "convert",
    [
        lambda spec: np.rint(1000 * spec).astype(np.int64),
        lambda spec: spec.astype("f4"),
    ],
    ids=["integer", "float32"],
)
def test_scale_casts_flux_to_float64_before_it_interpolates(convert):
    kwargs = GOLDEN.arguments(GOLDEN.cases["standard_scale.D1.mask"])
    ivar, mask = kwargs["ivar"], kwargs["mask"]
    flux = convert(kwargs["spec"])
    cast = flux.astype(np.float64)
    assert same(gspice.scale(flux, ivar, mask), gspice.scale(cast, ivar, mask))
    # Legacy keeps the 2022 arithmetic: it interpolates in the dtype of the flux, so it
    # truncates integers (D-20) and rounds to float32. A change at a bad pixel moves
    # refmean, and with it Dvec in the whole column, but Dvec + refmean, refscale times
    # the interpolated flux, changes only there. At good pixels it agrees to the
    # rounding of the sum, a few eps max|Dvec + refmean|.
    old, ref = (legacy.standard_scale(f, ivar, mask) for f in (flux, cast))
    scaled, scaled_ref = old[0] + old[2], ref[0] + ref[2]
    differs = np.abs(scaled - scaled_ref) > 1e-12 * np.abs(scaled_ref).max()
    assert differs.any()
    assert not differs[mask == 0].any()


def test_scale_rejects_the_spectrum_without_a_good_pixel():
    case = GOLDEN.cases["standard_scale.E.with_fully_masked"]
    # 2022 returned NaN in every refmean entry.
    assert np.isnan(GOLDEN.expected(case)["refmean"]).all()
    with pytest.raises(ValueError, match="^mask, ivar: .* rows 1$"):
        NEW_API[case["function"]](**GOLDEN.arguments(case))


# The edges, their neighbours, and the pixels where the 2022 predvar errs most: 34,
# 39, 40, 32, 38 and 21 at nguard 5, and 34, 38, 21, 39, 49 and 40 at nguard 20.
D3_PIXELS = [0, 1, 21, 32, 34, 38, 39, 40, 48, 49]


@pytest.mark.parametrize(
    "case_id",
    [
        case_id
        for case_id, item in BC_NEW.items()
        if item == "S11.4" and case_id.startswith("gp_interp.")
    ],
)
def test_pixelwise_moves_the_d3_predvar_to_the_60_digit_reference(case_id):
    # pred at its golden tolerance, at most 1e-10.
    _, predvar = check_golden_case(case_id, new_api, skip={"predvar"})
    # The 60-digit conditional variances.
    case = GOLDEN.cases[case_id]
    kwargs = GOLDEN.arguments(case)
    cov, nguard = kwargs["cov"], kwargs["nguard"]
    _, reference = guard_conditional(cov, nguard, np.zeros((0, len(cov))), D3_PIXELS)
    old = GOLDEN.expected(case)["predvar"][:, D3_PIXELS]
    new = predvar[:, D3_PIXELS]
    assert np.all(np.abs(new - reference) <= 1e-12 * reference)
    # The 2022 result is more than 1e-10 from the reference at some of the pixels.
    assert np.any(np.abs(old - reference) > 1e-10 * reference)


def test_infill_moves_the_d3_predcovar_to_the_60_digit_reference():
    case_id = "gaussian_estimate.D3.many21.guard5"
    # The other outputs at their golden tolerances, which are at most 1e-10.
    new = check_golden_case(case_id, new_api, skip={"predcovar"})[1]
    # The 60-digit conditional covariance of the pixels outside icond, at ipred.
    case = GOLDEN.cases[case_id]
    kwargs = GOLDEN.arguments(case)
    cov, icond, ipred = kwargs["cov"], kwargs["icond"], kwargs["ipred"]
    G = np.flatnonzero(icond == 0)
    _, reference, _ = conditional(cov, G, np.zeros((0, len(cov))))
    at = np.flatnonzero(ipred[G] != 0)
    reference = reference[np.ix_(at, at)]
    old = GOLDEN.expected(case)["predcovar"]
    assert np.max(np.abs(new - reference)) <= 1e-12 * np.max(np.abs(reference))
    variance = np.diagonal(reference)
    assert np.all(np.abs(np.diagonal(new) - variance) <= 1e-12 * variance)
    # The 2022 result is more than 1e-10 from the reference at some of the pixels.
    assert np.any(np.abs(np.diagonal(old) - variance) > 1e-10 * variance)


# The entries in which the final mask of the default fit differs from the stored 2022
# mask (design section 11).
CUMULATIVE_CHANGES = {"D1": 157, "D2": 152, "D3": 106, "D4": 8}


@pytest.mark.parametrize(("dataset", "changed"), CUMULATIVE_CHANGES.items())
def test_fit_masks_the_flags_of_every_pass_as_paper_algorithm_4(dataset, changed):
    case = GOLDEN.cases[f"covar_iter_mask.{dataset}.default"]
    kwargs = GOLDEN.arguments(case)
    flux, ivar, mask = kwargs["flux"], kwargs["ivar"], kwargs["mask"]
    _, training = gspice.fit(flux, ivar, mask)
    # Alg. 4 on legacy get_chimask, with its >= and Schur updates, on the spectra with
    # at most 64 bad pixels.
    expected = GOLDEN.expected(case)
    kept = expected["kept_rows"]
    flux, ivar = flux[kept], ivar[kept]
    M = (mask[kept] != 0) | (ivar == 0)
    for nsigma in (20, 8, 6):
        M |= legacy.get_chimask(flux, ivar, M, nsigma)
    replay = np.ones(mask.shape, dtype=bool)
    replay[kept] = M
    np.testing.assert_array_equal(training, replay)
    # 2022 masked the flags of the last two passes only.
    assert np.count_nonzero(training != (expected["finalmask"] != 0)) == changed


def test_fit_flags_a_pixel_whose_z_exceeds_nsigma_but_not_one_that_equals_it():
    # The record behaviour.get_chimask.ge_tie holds the 2022 >=: get_chimask flags the
    # pixel at nsigma equal to its own |z|.
    case = GOLDEN.cases["behaviour.get_chimask.ge_tie"]
    kwargs = GOLDEN.arguments(case)
    flux, ivar, mask = kwargs["flux"], kwargs["ivar"], kwargs["mask"]
    index = tuple(case["expect"]["tie"]["index"])
    # A tie exists only bit for bit, so |z| comes from the function that computes z in
    # fit, here for pass 1: fit uses every spectrum of D1, and M_1 = bad.
    bad = (mask != 0) | (ivar == 0)
    z = _model._clipping_z(flux, ivar, np.arange(len(flux)), bad, None, 20, "", None)
    tie = abs(float(z[index]))
    for nsigma, flagged in ((tie, False), (np.nextafter(tie, 0), True)):
        _, training = gspice.fit(flux, ivar, mask, nsigma=(nsigma,))
        assert training[index] == flagged


def test_fit_counts_ivar_0_as_bad_also_against_maxbadpix():
    inputs = GOLDEN.inputs("D1")
    flux, ivar, mask = inputs["flux"], inputs["ivar_z"], inputs["mask"]
    model, training = gspice.fit(flux, ivar, mask, maxbadpix=8, cumulative=False)
    assert model.n_train == (1990,)
    live = legacy.covar_iter_mask(flux, ivar, (mask != 0) | (ivar == 0), maxbadpix=8)
    mode = MODE[("covar_iter_mask", "cov")]
    assert_close(model.cov[0], live[0], mode, RTOL, 0.0, "cov")
    np.testing.assert_array_equal(training, live[1] != 0)
    # 2022 counted only the pixels in mask against maxbadpix.
    _, old = legacy.covar_iter_mask(flux, ivar, mask, maxbadpix=8)
    assert np.count_nonzero(~old.all(axis=1)) == 1992


def test_fit_drops_the_spectrum_that_clipping_masks_completely():
    # The inherited-Q1 input of design section 11: row 11 of D1 is good only at pixels
    # 24 to 26, and pass 1 flags its spike at pixel 25 with both neighbours.
    kwargs = GOLDEN.arguments(GOLDEN.cases["covar_iter_mask.D1.default"])
    flux, ivar, mask = kwargs["flux"], kwargs["ivar"], kwargs["mask"]
    mask[11] = 1
    mask[11, 24:27] = 0
    ivar[mask != 0] = 0
    flux[11, 25] += 20 / np.sqrt(ivar[11, 25])
    # 2022 returned a covariance of NaN, and legacy raises.
    with pytest.raises(ValueError, match="^nsigma: no good pixel in rows 11$"):
        legacy.covar_iter_mask(flux, ivar, mask, nsigma=[6])
    model, training = gspice.fit(flux, ivar, mask, nsigma=[6])
    assert model.n_train == (1999,)
    assert training[11].all()


def test_fit_needs_a_good_pixel_at_every_pixel():
    kwargs = GOLDEN.arguments(GOLDEN.cases["covar_iter_mask.D1.default"])
    flux, ivar, mask = kwargs["flux"], kwargs["ivar"], kwargs["mask"]
    mask[:, [25, 31]] = 1
    # 2022 interpolated the two pixels from their neighbours in every spectrum, which
    # makes the covariance singular.
    with pytest.raises(LinAlgError):
        legacy.covar_iter_mask(flux, ivar, mask)
    message = r"^fit: pass 1 needs a good pixel .* got none at pixels 25, 31$"
    with pytest.raises(ValueError, match=message):
        gspice.fit(flux, ivar, mask)


def test_fit_needs_one_spectrum_more_than_pixels():
    inputs = GOLDEN.inputs("D1")
    flux, ivar, mask = (inputs[key][:50] for key in ("flux", "ivar", "mask"))
    # 2022 accepted 50 spectra for 50 pixels, whose covariance is singular.
    with pytest.raises(LinAlgError):
        legacy.covar_iter_mask(flux, ivar, mask)
    message = r"^fit: pass 1 needs at least 51 spectra .*, got 50: of 50 spectra, "
    with pytest.raises(ValueError, match=message):
        gspice.fit(flux, ivar, mask)
