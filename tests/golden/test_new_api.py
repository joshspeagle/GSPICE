import numpy as np
import pytest

import gspice
from _golden import same
from gspice import legacy
from test_golden import GOLDEN, MODE, RTOL, check_golden_case, check_output

NEW_API = {
    "standard_scale": lambda spec, ivar, mask=None: gspice.scale(spec, ivar, mask)
}

BC_NEW = {
    "standard_scale.E.with_fully_masked": "Q1",
    "standard_scale.E.ivar0.mask": "Q11",
    "standard_scale.D1.ivar0.mask": "Q11",
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
