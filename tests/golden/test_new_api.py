import numpy as np
import pytest

import gspice
from _golden import same
from gspice import legacy
from oracle import guard_conditional
from test_golden import GOLDEN, MODE, RTOL, check_golden_case, check_output


def pixelwise_as_gp_interp(spec, cov, nguard, irange=None):
    # With ivar = 1, a zero mean and no centre, x is spec, and the outputs are in the
    # units of gp_interp.
    nspec, npix = spec.shape
    model = gspice.Model(np.zeros((1, npix)), (cov,), (nspec,))
    result = gspice.pixelwise(model, spec, np.ones_like(spec), nguard=nguard)
    pixels = slice(None) if irange is None else slice(irange[0], irange[1] + 1)
    return result.flux[:, pixels], result.var[:, pixels]


NEW_API = {
    "standard_scale": lambda spec, ivar, mask=None: gspice.scale(spec, ivar, mask),
    "gp_interp": pixelwise_as_gp_interp,
}

BC_NEW = {
    "standard_scale.E.with_fully_masked": "Q1",
    "standard_scale.E.ivar0.mask": "Q11",
    "standard_scale.D1.ivar0.mask": "Q11",
    "gp_interp.D3.ng5": "S11.4",
    "gp_interp.D3.ng5.weights": "S11.4",
    "gp_interp.D3.ng20": "S11.4",
    "gp_interp.D3.ng20.weights": "S11.4",
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
    check_golden_case(case_id, lambda module, function: NEW_API[function])


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
    "case_id", [case_id for case_id, item in BC_NEW.items() if item == "S11.4"]
)
def test_pixelwise_moves_the_d3_predvar_to_the_60_digit_reference(case_id):
    case = GOLDEN.cases[case_id]
    kwargs = GOLDEN.arguments(case)
    pred, predvar = pixelwise_as_gp_interp(**kwargs)
    expected = GOLDEN.expected(case)
    output = next(item for item in case["outputs"] if item["name"] == "pred")
    mode = MODE[("gp_interp", "pred")]
    check_output(case, output, pred, expected["pred"], kwargs, mode)
    # The 60-digit conditional variances.
    cov, nguard = kwargs["cov"], kwargs["nguard"]
    _, reference = guard_conditional(cov, nguard, np.zeros((0, len(cov))), D3_PIXELS)
    new, old = predvar[:, D3_PIXELS], expected["predvar"][:, D3_PIXELS]
    assert np.all(np.abs(new - reference) <= 1e-12 * reference)
    # The 2022 result is more than 1e-10 from the reference at some of the pixels.
    assert np.any(np.abs(old - reference) > 1e-10 * reference)
