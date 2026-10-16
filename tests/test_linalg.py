import functools

import numpy as np
import pytest
from numpy.linalg import LinAlgError

from gspice import _linalg
from gspice._linalg import group_rows, guard_weights, precision, solve_pattern
from oracle import conditional

EPS = np.finfo(float).eps
NPIX = 48
NSPEC = 3


def random_spd(kappa):
    # Q diag(lambda) Q^T with log-spaced eigenvalues; the mean of C and C^T is exactly
    # symmetric.
    Q, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((NPIX, NPIX)))
    C = (Q * np.logspace(0, -np.log10(kappa), NPIX)) @ Q.T
    return (C + C.T) / 2


def kernel(length):
    j = np.arange(NPIX)
    return np.exp(-0.5 * ((j[:, np.newaxis] - j) / length) ** 2) + 1e-6 * np.eye(NPIX)


# Condition numbers of the matrices scaled to unit diagonal: 89, 8.0e4, 7.5e7, 1.2e7
# and 2.9e7.
MATRICES = {
    "spd1e2": random_spd(1e2),
    "spd1e5": random_spd(1e5),
    "spd1e8": random_spd(1e8),
    "kernel5": kernel(5),
    "kernel15": kernel(15),
}


@functools.cache
def case(name):
    C = MATRICES[name]
    d = 1 / np.sqrt(np.diagonal(C))
    kappa = np.linalg.cond(d[:, np.newaxis] * C * d)
    X = np.random.default_rng(1).standard_normal((NSPEC, NPIX))
    X = X @ np.linalg.cholesky(C).T
    # Design section 11: 10 kappa eps against the 60-digit oracle.
    return C, precision(C), X, 10 * kappa * EPS


def test_precision_is_exactly_symmetric_and_c_contiguous():
    P = precision(random_spd(1e2))
    assert P.flags.c_contiguous
    np.testing.assert_array_equal(P, P.T)


def test_precision_rejects_a_covariance_with_condition_number_1e14():
    # Design decision A26: rcond is 5.3e-15 for S scaled to unit diagonal.
    with pytest.raises(LinAlgError, match="numerically singular: rcond = "):
        precision(random_spd(1e14))


def test_precision_rejects_a_nan_rcond(monkeypatch):
    monkeypatch.setattr(_linalg.lapack, "dpocon", lambda *args, **kwargs: (np.nan, 0))
    with pytest.raises(LinAlgError, match="rcond = nan"):
        precision(random_spd(1e2))


def test_precision_checks_the_condition_of_s_scaled_to_unit_diagonal():
    # rcond is 4.5e-9 for S scaled to unit diagonal and 5.9e-31 for S itself.
    d = np.logspace(-6, 6, NPIX)
    precision(d[:, np.newaxis] * random_spd(1e8) * d)


def test_precision_rejects_a_matrix_that_is_not_positive_definite():
    S = random_spd(1e2)
    S[5, 5] = -1.0
    with pytest.raises(LinAlgError, match="not positive definite"):
        precision(S)


GUARD_CASES = [(name, nguard) for name in MATRICES for nguard in (0, 1, 5)]
GUARD_CASES += [("kernel5", NPIX // 3), ("spd1e5", NPIX - 2)]


@pytest.mark.parametrize(("name", "nguard"), GUARD_CASES)
def test_guard_weights_match_the_reference(name, nguard):
    C, P, X, tol = case(name)
    W, var = guard_weights(P, np.diagonal(C), nguard)
    # The edges, pixels next to them and interior pixels.
    for i in (0, 1, 7, 24, 40, 47):
        g = range(max(0, i - nguard), min(NPIX, i + nguard + 1))
        assert np.all(W[g, i] == 0)
        mean, cov, _ = conditional(C, g, X)
        k = i - g.start
        assert np.max(np.abs(X @ W[:, i] - mean[:, k])) <= tol * np.sqrt(cov[k, k])
        assert abs(var[i] - cov[k, k]) <= tol * cov[k, k]


@pytest.mark.parametrize("nguard", [NPIX - 2, NPIX - 1])
def test_guard_windows_that_cover_all_pixels_give_the_exact_prior(nguard):
    C, P, *_ = case("kernel5")
    W, var = guard_weights(P, np.diagonal(C), nguard)
    prior = range(max(0, NPIX - 1 - nguard), nguard + 1)
    assert np.all(W[:, prior] == 0)
    np.testing.assert_array_equal(var[prior], np.diagonal(C)[prior])


def test_guard_weights_reject_a_window_that_is_not_positive_definite():
    P = np.eye(NPIX)
    P[7, 7] = -1.0
    with pytest.raises(LinAlgError, match="window of pixel 7"):
        guard_weights(P, np.ones(NPIX), 0)


def mask(pixels):
    selected = np.zeros(NPIX, bool)
    selected[list(pixels)] = True
    return selected


# (G, M, T) as pixel lists: T is a subset of M, and M of G.
PATTERNS = {
    # A run M with a guard of 4 pixels on each side; pixels 5 and 23 are bad.
    "run": ([5, *range(16, 32)], [5, *range(20, 28)], [20, 21, 22, *range(24, 28)]),
    # The order (M \ T, T) of M is not its pixel order.
    "scattered": (
        [2, 3, 9, 14, 15, 22, 30, 31, 33, 38, 41, 44, 46, 47],
        [3, 9, 15, 22, 31, 38, 44, 46, 47],
        [9, 22, 38, 46, 47],
    ),
    # G covers all pixels.
    "prior": (range(NPIX), [0, 4, 11, 17, 23, 29, 35, 40, 47], [4, 17, 29, 40, 47]),
    # G \ M and M \ T are empty.
    "replace": (range(10, 20), range(10, 20), range(10, 20)),
}


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("name", MATRICES)
def test_pattern_solve_matches_the_reference(name, pattern):
    C, P, X, tol = case(name)
    G, M, T = (mask(pixels) for pixels in PATTERNS[pattern])
    solution = solve_pattern(P, G, M, T, X, covariance=True)
    mean, cov, chi2 = conditional(C, np.flatnonzero(G), X, np.flatnonzero(T))
    at = np.flatnonzero(M[G])
    mean, cov = mean[:, at], cov[np.ix_(at, at)]
    # Covariance errors in units of sigma_i sigma_j, which is |dvar|/var on the
    # diagonal.
    scale = np.sqrt(np.outer(np.diagonal(cov), np.diagonal(cov)))
    assert np.max(np.abs(solution.mean - mean) / np.sqrt(np.diagonal(cov))) <= tol
    assert np.max(np.abs(solution.cov - cov) / scale) <= tol
    np.testing.assert_array_equal(solution.cov, solution.cov.T)
    np.testing.assert_array_equal(np.diagonal(solution.cov), solution.var)
    assert np.max(np.abs(solution.chi2 - chi2) / (chi2 + np.count_nonzero(T))) <= tol
    # With a unit normal at pixel j of M in row j, the rows of D = draws - mean are
    # the columns of the map from normals to draws, so D^T D is the covariance of the
    # draws.
    m = np.count_nonzero(M)
    unit = solve_pattern(P, G, M, T, X[[0] * m], normals=np.eye(NPIX)[M])
    D = unit.draws - unit.mean
    assert np.max(np.abs(D.T @ D - cov) / scale) <= tol


def test_pattern_solve_never_reads_the_data_in_G_outside_T():
    _, P, X, *_ = case("spd1e5")
    G, M, T = (mask(pixels) for pixels in PATTERNS["run"])
    normals = np.random.default_rng(2).standard_normal(X.shape)
    expected = solve_pattern(P, G, M, T, X, normals, covariance=True)
    X = X.copy()
    X[:, G & ~T] = np.nan
    for actual, value in zip(
        solve_pattern(P, G, M, T, X, normals, covariance=True), expected, strict=True
    ):
        np.testing.assert_array_equal(actual, value)


@pytest.mark.parametrize("G", [[], [3, 4, 5]])
def test_a_pattern_without_pixels_to_predict_skips_lapack(monkeypatch, G):
    _, P, X, *_ = case("spd1e5")
    monkeypatch.setattr(_linalg, "lapack", None)
    empty = mask([])
    solution = solve_pattern(P, mask(G), empty, empty, X, normals=X, covariance=True)
    assert solution.mean.shape == solution.draws.shape == (NSPEC, 0)
    assert (solution.var.shape, solution.cov.shape) == ((0,), (0, 0))
    np.testing.assert_array_equal(solution.chi2, np.zeros(NSPEC))


def test_an_empty_T_gives_chi2_0():
    _, P, X, *_ = case("spd1e5")
    solution = solve_pattern(P, mask(range(10, 20)), mask(range(12, 18)), mask([]), X)
    np.testing.assert_array_equal(solution.chi2, np.zeros(NSPEC))


def test_pattern_solve_rejects_a_pattern_on_which_P_is_not_positive_definite():
    P = np.eye(NPIX)
    P[7, 7] = -1.0
    with pytest.raises(LinAlgError, match="not positive definite on G"):
        solve_pattern(P, mask([6, 7]), mask([6]), mask([]), np.zeros((1, NPIX)))


def test_rows_with_equal_keys_share_a_group_in_increasing_row_order():
    # Keys 1 and 0 differ only in column 12, which packbits puts in the second byte.
    # With 2000 rows, an unstable sort would reorder the rows of a group.
    keys = np.zeros((3, 13), bool)
    keys[1, 12] = True
    keys[2, [0, 5]] = True
    labels = np.random.default_rng(3).integers(0, 3, 2000)
    groups = group_rows(keys[labels])
    assert sorted(labels[rows[0]] for rows in groups) == [0, 1, 2]
    for rows in groups:
        np.testing.assert_array_equal(rows, np.flatnonzero(labels == labels[rows[0]]))


def test_f_ordered_keys_give_the_same_groups():
    keys = np.eye(3, 9, dtype=bool)[[0, 1, 0]]
    groups = [rows.tolist() for rows in group_rows(keys)]
    assert [rows.tolist() for rows in group_rows(np.asfortranarray(keys))] == groups


def test_no_rows_give_no_groups():
    assert group_rows(np.zeros((0, 5), bool)) == []
