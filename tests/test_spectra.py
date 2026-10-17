import operator

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from _golden import same
from gspice import _spectra, scale, snr, unscale
from gspice._spectra import interpolate

EPS = np.finfo(float).eps


def magnitudes(high):
    # Above 1e-3, no product in these tests underflows, where a relative error bound
    # would fail.
    positive = st.floats(1e-3, high)
    return st.sampled_from([0.0, -0.0]) | positive | positive.map(operator.neg)


@st.composite
def spectra(draw):
    nspec = draw(st.integers(1, 4))
    npix = draw(st.integers(1, 40))
    shape = (nspec, npix)
    j = np.arange(npix)
    # Per row: random, no bad pixel, alternating, or bad but for the kept pixel.
    patterns = st.one_of(
        hnp.arrays(bool, npix),
        st.just(np.zeros(npix, bool)),
        st.integers(0, 1).map(lambda phase: j % 2 == phase),
        st.just(np.ones(npix, bool)),
    )
    bad = np.array([draw(patterns) for _ in range(nspec)])
    keep = draw(st.lists(st.integers(0, npix - 1), min_size=nspec, max_size=nspec))
    bad[np.arange(nspec), keep] = False
    flux = draw(hnp.arrays(np.float64, shape, elements=magnitudes(1e3)))
    ivar = draw(hnp.arrays(np.float64, shape, elements=st.floats(1e-3, 1e3)))
    mask = None
    in_mask = np.zeros(shape, bool)
    if draw(st.booleans()):
        in_mask = bad & draw(hnp.arrays(bool, shape))
        # Nonzero means in mask, whatever the dtype.
        value = draw(st.sampled_from([True, 3, -1, 0.5, np.nan]))
        mask = np.where(in_mask, value, False)
    ivar[bad & ~in_mask] = 0.0
    return flux, ivar, mask, bad


@given(spectra())
def test_interpolation_matches_np_interp_and_keeps_good_pixels(case):
    flux, _, _, bad = case
    x = interpolate(flux, bad)
    assert same(x[~bad], flux[~bad])
    # np.interp computes the same slope, but some builds fuse slope * (i - lo) + f_lo
    # into one multiply-add; the results then differ by up to 2 eps max|f_good|.
    for row, values, b in zip(x, flux, bad, strict=True):
        good = np.flatnonzero(~b)
        expected = np.interp(np.flatnonzero(b), good, values[good])
        assert np.all(np.abs(row[b] - expected) <= 8 * EPS * np.abs(values[good]).max())


@given(spectra(), st.data())
def test_unscale_inverts_scale_at_good_pixels(case, data):
    flux, ivar, mask, bad = case
    center = data.draw(magnitudes(10))
    given_mean = hnp.arrays(np.float64, flux.shape[1], elements=magnitudes(1e3))
    refmean = data.draw(st.none() | given_mean)
    X, a, m = scale(flux, ivar, mask, center=center, refmean=refmean)
    f, var = unscale(X, a, m, center=center)
    assert var is None
    # To first order, the six roundings of scale and unscale give
    # |f' - f| <= (eps/2)(6|f| + 5|center| + |refmean/a_s|), at most 6 eps times the
    # largest term.
    largest = np.maximum(np.abs(flux), np.abs(m / a[:, np.newaxis]))
    bound = 8 * EPS * np.maximum(largest, abs(center))
    assert np.all(np.abs(f - flux)[~bad] <= bound[~bad])


def test_scale_and_unscale_follow_their_formulas_on_exact_values():
    # a = (2, 1). With center 1, a (ftilde - 1) is [[0, 2, 4], [2, 2, 2]], with pixel 1
    # of row 1 interpolated, and its column mean is [1, 2, 3]. Every step is exact.
    flux = np.array([[1.0, 2.0, 3.0], [3.0, np.nan, 3.0]])
    ivar = np.array([[4.0, 4.0, 4.0], [1.0, 0.0, 1.0]])
    X, a, m = scale(flux, ivar, center=1.0)
    np.testing.assert_array_equal(a, [2.0, 1.0])
    np.testing.assert_array_equal(m, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(X, [[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]])
    given = scale(flux, ivar, center=1.0, refmean=np.zeros(3))[0]
    np.testing.assert_array_equal(given, [[0.0, 2.0, 4.0], [2.0, 2.0, 2.0]])
    f, var = unscale(X, a, m, center=1.0, var=np.array([4.0, 8.0, 12.0]))
    np.testing.assert_array_equal(f, [[1.0, 2.0, 3.0], [3.0, 3.0, 3.0]])
    np.testing.assert_array_equal(var, [[1.0, 2.0, 3.0], [4.0, 8.0, 12.0]])


def test_every_nonzero_mask_value_marks_a_pixel_in_mask():
    # 2022 maskinterp counts 0.5 as good (D-19).
    flux = np.array([[1.0, 4.0, 2.0, 8.0, 5.0, 7.0], [3.0, 1.0, 4.0, 1.0, 5.0, 9.0]])
    ivar = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
    mask = np.array([[0.5, 0, np.nan, 0, -1, 0], [0, np.inf, 0, 1e-300, 0, 0]])
    for function in (scale, snr):
        assert same(function(flux, ivar, mask), function(flux, ivar, mask != 0))


@given(spectra())
def test_snr_is_the_median_of_flux_times_root_ivar_over_good_pixels(case):
    flux, ivar, mask, bad = case
    expected = [
        np.median(f[~b] * np.sqrt(iv[~b]))
        for f, iv, b in zip(flux, ivar, bad, strict=True)
    ]
    np.testing.assert_array_equal(snr(flux, ivar, mask), expected)


def test_a_zero_snr_is_positive_zero():
    # A sort may order -0.0 and +0.0 either way, so only one sign of a zero median is
    # reproducible on every platform.
    flux = np.array([[-0.0, -0.0, -0.0], [0.0, -0.0, -0.0]])
    assert not np.signbit(snr(flux, np.ones(flux.shape))).any()


def fortran(a):
    return None if a is None else np.asfortranarray(a)


@given(spectra())
def test_other_float_dtypes_and_f_order_give_the_result_of_a_float64_c_copy(case):
    flux, ivar, mask, _ = case
    X, a, m = scale(flux, ivar, mask)
    for dtype in (np.float32, np.longdouble):
        # Where long double is wider than float64, a third of a value is often not a
        # float64.
        f, iv, x = (np.asarray(v, dtype) / 3 for v in (flux, ivar, X))
        f64, iv64, x64 = (v.astype(np.float64) for v in (f, iv, x))
        for function in (scale, snr):
            assert same(function(f, iv, mask), function(f64, iv64, mask))
        assert same(unscale(x, a, m, var=iv), unscale(x64, a, m, var=iv64))
    for function in (scale, snr):
        expected = function(flux, ivar, mask)
        assert same(function(*map(fortran, (flux, ivar, mask))), expected)


@given(spectra())
def test_results_do_not_depend_on_the_chunk_size(case):
    flux, ivar, mask, _ = case
    expected = scale(flux, ivar, mask), snr(flux, ivar, mask)
    # A chunk of 1 element holds 1 row, the minimum; one of 2 npix elements, 2 rows.
    for chunk in (1, 2 * flux.shape[1]):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(_spectra, "CHUNK", chunk)
            assert same(scale(flux, ivar, mask), expected[0])
            assert same(snr(flux, ivar, mask), expected[1])


@pytest.mark.parametrize("name", ["flux", "ivar"])
def test_a_value_error_names_its_row_in_the_whole_input(monkeypatch, name):
    # Two rows per chunk, so row 3 is row 1 of the second chunk.
    monkeypatch.setattr(_spectra, "CHUNK", 2 * 4)
    arrays = {"flux": np.ones((5, 4)), "ivar": np.ones((5, 4))}
    arrays[name][3, 2] = np.inf
    with pytest.raises(ValueError, match=rf"^{name}: .* got inf at row 3, pixel 2$"):
        scale(**arrays)


@pytest.mark.filterwarnings("error")
@given(spectra(), st.data())
def test_values_at_bad_pixels_are_never_read(case, data):
    flux, ivar, mask, bad = case
    poison = hnp.arrays(
        np.float64,
        flux.shape,
        elements=st.sampled_from([np.nan, np.inf, -np.inf, -1.0]),
    )
    in_mask = np.zeros(bad.shape, bool) if mask is None else mask != 0
    dirty_flux = np.where(bad, data.draw(poison), flux)
    dirty_ivar = np.where(in_mask, data.draw(poison), ivar)
    for function in (scale, snr):
        expected = function(flux, ivar, mask)
        assert same(function(dirty_flux, dirty_ivar, mask), expected)


def checked(function, *args, **kwargs):
    inputs = [value for value in (*args, *kwargs.values()) if value is not None]
    pristine = [value.copy() for value in inputs]
    result = function(*args, **kwargs)
    for value, copy in zip(inputs, pristine, strict=True):
        assert same(value, copy)
    for output in result if isinstance(result, tuple) else (result,):
        assert not any(np.shares_memory(output, value) for value in inputs)
    return result


@given(spectra(), st.data())
def test_no_input_is_modified_or_shared_with_an_output(case, data):
    flux, ivar, mask, _ = case
    refmean = data.draw(hnp.arrays(np.float64, flux.shape[1], elements=magnitudes(1e3)))
    var = data.draw(hnp.arrays(np.float64, flux.shape, elements=st.floats(0, 1e3)))
    X, a, m = checked(scale, flux, ivar, mask)
    checked(scale, flux, ivar, mask, refmean=refmean)
    checked(snr, flux, ivar, mask)
    checked(unscale, X, a, m, var=var)


def test_zero_spectra_give_empty_results():
    empty = np.zeros((0, 3))
    results = [
        *scale(empty, empty, empty, refmean=np.zeros(3)),
        snr(empty, empty),
        *unscale(empty, np.zeros(0), np.zeros(3), var=np.zeros(3)),
    ]
    assert [(r.shape, r.dtype) for r in results] == [
        ((0, 3), np.float64),
        ((0,), np.float64),
        ((3,), np.float64),
        ((0,), np.float64),
        ((0, 3), np.float64),
        ((0, 3), np.float64),
    ]
