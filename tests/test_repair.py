import itertools
import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from _golden import checked, same
from gspice._repair import _exclude_extended


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
