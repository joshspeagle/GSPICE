import functools

import numpy as np
import pytest
from numpy.linalg import LinAlgError

from gspice import _linalg
from gspice._linalg import (
    cholesky,
    group_rows,
    guard_means,
    guard_solutions,
    guard_weights,
    precision,
    prior_chi2,
    prior_draws,
    solve_factor,
    solve_low_rank,
    solve_mean,
    solve_pattern,
    woodbury,
)
from oracle import (
    condition,
    conditional,
    guard_conditional,
    inverse,
    kappa_w,
    low_rank,
    random_low_rank,
    random_spd,
    window,
)

EPS = np.finfo(float).eps
NPIX = 48
NSPEC = 3


def kernel(length):
    j = np.arange(NPIX)
    return np.exp(-0.5 * ((j[:, np.newaxis] - j) / length) ** 2) + 1e-6 * np.eye(NPIX)


# Condition numbers of the matrices scaled to unit diagonal: 89, 8.0e4, 7.5e7, 1.2e7
# and 2.9e7.
MATRICES = {
    "spd1e2": random_spd(NPIX, 1e2, 0),
    "spd1e5": random_spd(NPIX, 1e5, 0),
    "spd1e8": random_spd(NPIX, 1e8, 0),
    "kernel5": kernel(5),
    "kernel15": kernel(15),
}


@functools.cache
def case(name):
    C = MATRICES[name]
    X = np.random.default_rng(1).standard_normal((NSPEC, NPIX))
    X = X @ np.linalg.cholesky(C).T
    # Design section 11: 10 kappa eps against the 60-digit oracle.
    return C, precision(C), X, 10 * condition(C) * EPS


def test_precision_is_exactly_symmetric_and_c_contiguous():
    P = precision(random_spd(NPIX, 1e2, 0))
    assert P.flags.c_contiguous
    np.testing.assert_array_equal(P, P.T)


def test_precision_rejects_a_covariance_with_condition_number_1e14():
    # Design decision A26: rcond is 5.3e-15 for S scaled to unit diagonal.
    with pytest.raises(LinAlgError, match="numerically singular: rcond = "):
        precision(random_spd(NPIX, 1e14, 0))


def test_precision_rejects_a_nan_rcond(monkeypatch):
    monkeypatch.setattr(_linalg.lapack, "dpocon", lambda *args, **kwargs: (np.nan, 0))
    with pytest.raises(LinAlgError, match="rcond = nan"):
        precision(random_spd(NPIX, 1e2, 0))


def test_precision_checks_the_condition_of_s_scaled_to_unit_diagonal():
    # rcond is 4.5e-9 for S scaled to unit diagonal and 5.9e-31 for S itself.
    d = np.logspace(-6, 6, NPIX)
    precision(d[:, np.newaxis] * random_spd(NPIX, 1e8, 0) * d)


def test_precision_rejects_a_matrix_that_is_not_positive_definite():
    S = random_spd(NPIX, 1e2, 0)
    S[5, 5] = -1.0
    with pytest.raises(LinAlgError, match="not positive definite"):
        precision(S)


GUARD_CASES = [(name, nguard) for name in MATRICES for nguard in (0, 1, 5)]
GUARD_CASES += [("kernel5", NPIX // 3), ("spd1e5", NPIX - 2)]


@pytest.mark.parametrize(("name", "nguard"), GUARD_CASES)
def test_guard_weights_and_means_match_the_reference(name, nguard):
    C, P, X, tol = case(name)
    V, var = guard_solutions(P, np.diagonal(C), nguard)
    W = guard_weights(P, V)
    # The edges, pixels next to them and interior pixels.
    pixels = [0, 1, 7, 24, 40, 47]
    for i in pixels:
        assert np.all(W[max(0, i - nguard) : i + nguard + 1, i] == 0)
    mean, reference = guard_conditional(C, nguard, X, pixels)
    sigma = np.sqrt(reference)
    assert np.all(np.abs(X @ W[:, pixels] - mean) <= tol * sigma)
    assert np.all(np.abs(guard_means(P, V, X)[:, pixels] - mean) <= tol * sigma)
    assert np.all(np.abs(var[pixels] - reference) <= tol * reference)


@pytest.mark.parametrize("nguard", [NPIX - 2, NPIX - 1])
def test_guard_windows_that_cover_all_pixels_give_the_exact_prior(nguard):
    C, P, X, _ = case("kernel5")
    V, var = guard_solutions(P, np.diagonal(C), nguard)
    prior = range(max(0, NPIX - 1 - nguard), nguard + 1)
    assert np.all(guard_weights(P, V)[:, prior] == 0)
    assert np.all(guard_means(P, V, X)[:, prior] == 0)
    np.testing.assert_array_equal(var[prior], np.diagonal(C)[prior])


@pytest.mark.parametrize("nguard", [1, 5])
@pytest.mark.parametrize("name", ["spd1e2", "spd1e8", "kernel5"])
def test_outliers_of_1e4_sigma_in_the_guard_window_cancel_to_a_rounding_bound(
    name, nguard
):
    # guard_means computes mu_i = x_i - v^T p, with v = P_gg^-1 e_i from the Cholesky
    # factor R of P_gg and p = (P x)_g. x_g enters v^T p as v^T P_gg x_g = x_i, which
    # cancels x_i only to rounding. v solves the computed P_gg, so the error of P does
    # not break the cancellation. The backward errors of the product (Higham 2002,
    # eq. 3.13), of the Cholesky solve (Thm 10.4) and of the dot product, with
    # gamma_n = n u / (1 - n u) <= n eps for u = eps / 2, give to first order
    #   |d mu_i| <= eps (npix |v|^T |P_g:| |x| + (3 |g| + 1) |v|^T |R| |R^T| |y|
    #               + |g| |v|^T |p| + |mu_i|)
    # with y = P_gg^-1 p = x_g - mu_g. It grows linearly with the outliers, and the
    # reference does not depend on x_g. The bound adds to the 10 kappa eps sigma of
    # design section 11 for the rest of x, drawn from N(0, C).
    C, P, X, tol = case(name)
    V, _ = guard_solutions(P, np.diagonal(C), nguard)
    for i in [0, 1, 7, 24, 47]:
        g = window(i, nguard, NPIX)
        x = X.copy()
        x[:, g] += 1e4 * np.sqrt(np.diagonal(C)[g]) * (-1) ** np.arange(len(g))
        mu = guard_means(P, V, x)[:, i]
        mean, reference = guard_conditional(C, nguard, x, [i])
        n = len(g)
        PGG = P[g.start : g.stop, g.start : g.stop]
        v = np.linalg.solve(PGG, np.eye(n)[i - g.start])
        R = np.linalg.cholesky(PGG)
        p = x @ P[:, g.start : g.stop]
        y = np.linalg.solve(PGG, p.T)
        bound = EPS * (
            NPIX * np.abs(v) @ np.abs(P[g.start : g.stop]) @ np.abs(x.T)
            + (3 * n + 1) * np.abs(v) @ np.abs(R) @ np.abs(R.T) @ np.abs(y)
            + n * np.abs(p) @ np.abs(v)
            + np.abs(mu)
        )
        error = np.abs(mu - mean[:, 0])
        assert np.all(error <= tol * np.sqrt(reference) + bound)


def test_guard_solutions_reject_a_window_that_is_not_positive_definite():
    P = np.eye(NPIX)
    P[7, 7] = -1.0
    with pytest.raises(LinAlgError, match="window of pixel 7"):
        guard_solutions(P, np.ones(NPIX), 0)


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


def draw_covariance(solve, M, X):
    # With a unit normal at the j-th pixel of M in row j, the rows of D = draws - mean
    # are the columns of the map from normals to draws, so D^T D is the covariance of
    # the draws.
    m = np.count_nonzero(M)
    unit = solve(X[[0] * m], np.eye(m))
    D = unit.draws - unit.mean
    return D.T @ D


def solver(route, C):
    # solve_pattern from P, or the same conditional from the Cholesky factor of C.
    if route == "factor":
        return functools.partial(solve_factor, cholesky(C))
    return functools.partial(solve_pattern, precision(C))


@pytest.mark.parametrize("route", ["precision", "factor"])
@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("name", MATRICES)
def test_pattern_solve_matches_the_reference(name, pattern, route):
    C, _, X, tol = case(name)
    G, M, T = (mask(pixels) for pixels in PATTERNS[pattern])
    solve = solver(route, C)
    solution = solve(G, M, T, X, covariance=True)
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
    draws = draw_covariance(functools.partial(solve, G, M, T), M, X)
    assert np.max(np.abs(draws - cov) / scale) <= tol


@pytest.mark.parametrize("route", ["precision", "factor"])
def test_pattern_solve_never_reads_the_data_in_G_outside_T(route):
    C, _, X, *_ = case("spd1e5")
    solve = solver(route, C)
    G, M, T = (mask(pixels) for pixels in PATTERNS["run"])
    normals = np.random.default_rng(2).standard_normal((len(X), np.sum(M)))
    expected = solve(G, M, T, X, normals, covariance=True)
    X = X.copy()
    X[:, G & ~T] = np.nan
    for actual, value in zip(
        solve(G, M, T, X, normals, covariance=True), expected, strict=True
    ):
        np.testing.assert_array_equal(actual, value)


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("name", MATRICES)
def test_pattern_solve_without_variances_keeps_the_mean_chi2_and_draws(name, pattern):
    # Exact: the skipped inverse of R_MM enters neither the mean, chi2 nor the draws.
    _, P, X, _ = case(name)
    G, M, T = (mask(pixels) for pixels in PATTERNS[pattern])
    normals = np.random.default_rng(2).standard_normal((len(X), np.sum(M)))
    expected = solve_pattern(P, G, M, T, X, normals)
    solution = solve_pattern(P, G, M, T, X, normals, variance=False)
    for field in ("mean", "chi2", "draws"):
        np.testing.assert_array_equal(
            getattr(solution, field), getattr(expected, field)
        )
    assert np.isnan(solution.var).all() and solution.var.shape == expected.var.shape


@pytest.mark.parametrize("G", [[], [3, 4, 5]])
def test_a_pattern_without_pixels_to_predict_skips_lapack(monkeypatch, G):
    _, P, X, *_ = case("spd1e5")
    monkeypatch.setattr(_linalg, "lapack", None)
    empty = mask([])
    solution = solve_pattern(
        P, mask(G), empty, empty, X, normals=X[:, :0], covariance=True
    )
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
    with pytest.raises(LinAlgError, match="^not positive definite in its precision"):
        solve_pattern(P, mask([6, 7]), mask([6]), mask([]), np.zeros((1, NPIX)))


# (G, M) as pixel lists with |G| > NPIX / 2, where filled takes the mean from C_OO.
MEAN_PATTERNS = {
    "run": (range(10, 40), range(14, 36)),
    # O scattered, with both edges, and M not a run.
    "scattered": (
        [i for i in range(NPIX) if i % 7],
        [i for i in range(NPIX) if i % 7 and i % 2],
    ),
    # C_OO is 1 x 1.
    "one in O": ([i for i in range(NPIX) if i != 17], range(18, 43)),
}


@pytest.mark.parametrize("pattern", MEAN_PATTERNS)
@pytest.mark.parametrize("name", MATRICES)
def test_the_mean_from_c_oo_matches_the_reference_without_reading_g(name, pattern):
    C, _, X, tol = case(name)
    G, M = (mask(pixels) for pixels in MEAN_PATTERNS[pattern])
    mean, cov, _ = conditional(C, np.flatnonzero(G), X)
    at = np.flatnonzero(M[G])
    X = X.copy()
    X[:, G] = np.nan
    # The tolerance of design section 11 for the P route.
    error = (solve_mean(C, G, M, X) - mean[:, at]) / np.sqrt(np.diagonal(cov)[at])
    assert np.max(np.abs(error)) <= tol


def test_the_mean_from_c_oo_rejects_a_c_oo_that_is_not_positive_definite():
    C = np.eye(NPIX)
    C[7, 7] = -1.0
    G = ~mask([6, 7])
    with pytest.raises(LinAlgError, match="^not positive definite on the pixels O$"):
        solve_mean(C, G, G, np.zeros((1, NPIX)))


@pytest.mark.parametrize("name", MATRICES)
def test_the_prior_is_exact_and_its_chi2_matches_the_reference(name):
    C, _, X, tol = case(name)
    _, M, T = (mask(pixels) for pixels in PATTERNS["prior"])
    CMM, var = C[np.ix_(M, M)], np.diagonal(C)[M]
    _, _, chi2 = conditional(C, range(NPIX), X, np.flatnonzero(T))
    result = prior_chi2(C[np.ix_(T, T)], X[:, T])
    assert np.max(np.abs(result - chi2) / (chi2 + np.sum(T))) <= tol
    # Unit normals at the pixels of M give the rows of L^T, so D^T D = C_MM.
    D = prior_draws(CMM, np.eye(len(CMM)))
    scale = np.sqrt(np.outer(var, var))
    assert np.max(np.abs(D.T @ D - CMM) / scale) <= tol


def test_the_prior_rejects_a_covariance_that_is_numerically_singular_at_t():
    # Rank 1 plus 1e-14 I: dpotrf succeeds, but C at T scaled to unit diagonal has an
    # rcond near 1e-14, below the limit of decision A26.
    v = 2.0 ** (np.arange(NPIX) % 3 - 1)
    C = np.outer(v, v) + 1e-14 * np.eye(NPIX)
    with pytest.raises(LinAlgError, match="^numerically singular on the pixels T: "):
        prior_chi2(C[:3, :3], np.zeros((1, 3)))


def test_the_prior_without_t_skips_lapack_and_gives_chi2_0(monkeypatch):
    _, _, X, _ = case("spd1e5")
    # LAPACK rejects an empty triangular solve, with a message on stderr.
    monkeypatch.setattr(_linalg, "lapack", None)
    np.testing.assert_array_equal(
        prior_chi2(np.zeros((0, 0)), X[:, :0]), np.zeros(NSPEC)
    )


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


def low_rank_case(kappa, spread=False):
    V, values, d = random_low_rank(NPIX, 6, kappa, 0)
    if spread:
        # Pixels with d_i much smaller than c_ii, where kappa_W exceeds kappa.
        d[::7] *= 1e-6
    return V, values, d


# kappa of C scaled to unit diagonal and kappa_W: 43 and 54, 1.0e5 and 1.3e5, 1.2e8
# and 1.6e8, and 4.0e6 and 1.7e7 for the spread d, where max c_ii/d_i is 1.5e6.
LOW_RANK = {
    "low1e2": low_rank_case(1e2),
    "low1e5": low_rank_case(1e5),
    "low1e8": low_rank_case(1e8),
    "spread": low_rank_case(1e2, spread=True),
}


@pytest.mark.parametrize("name", LOW_RANK)
def test_the_woodbury_precision_matches_the_reference(name):
    V, values, d = LOW_RANK[name]
    P = woodbury(V, values, d)
    reference = inverse(low_rank(V, values, d))
    # P_ii = 1/d_i - (L L^T)_ii cancels by c_ii/d_i at most, and an entry P_ij by
    # (c_ii c_jj / (d_i d_j))^1/2 in units of (P_ii P_jj)^1/2 >= 1 / (c_ii c_jj)^1/2;
    # kappa_W bounds both (decision A28), so the error is within 10 eps kappa_W.
    scale = np.sqrt(np.outer(np.diagonal(reference), np.diagonal(reference)))
    assert np.max(np.abs(P - reference) / scale) <= 10 * EPS * kappa_w(V, values, d)
    assert P.flags.c_contiguous
    np.testing.assert_array_equal(P, P.T)


@pytest.mark.parametrize(("value", "rejected"), [(5e12 - 1, False), (5e12, True)])
def test_woodbury_rejects_kappa_w_above_1e13(value, rejected):
    # V = e_0 and d = 1: c_0 = 1 + lambda, dtilde_0 = 1/c_0 and the largest eigenvalue
    # is lambda/c_0, so kappa_W = 1 + 2 lambda: 1e13 - 1 and 1e13 + 1.
    V = np.eye(NPIX, 1)
    if not rejected:
        woodbury(V, np.array([value]), np.ones(NPIX))
        return
    message = r"^numerically singular: kappa_W = 1e\+13 for the matrix scaled to "
    with pytest.raises(LinAlgError, match=message):
        woodbury(V, np.array([value]), np.ones(NPIX))


def test_woodbury_rejects_a_factor_that_is_not_positive_definite(monkeypatch):
    # kappa_W <= 1e13 keeps Lambda^-1 + V^T D^-1 V positive definite in practice, so
    # dpotrf is made to fail.
    monkeypatch.setattr(_linalg.lapack, "dpotrf", lambda A, lower: (A, 2))
    V, values, d = LOW_RANK["low1e2"]
    with pytest.raises(LinAlgError, match="^not positive definite: .* order 2 of "):
        woodbury(V, values, d)


# The patterns of solve_low_rank, which needs pixels outside G.
LOW_RANK_PATTERNS = {name: PATTERNS[name] for name in ("run", "scattered", "replace")}
# |O| = 3 conditioning pixels, fewer than K = 6.
LOW_RANK_PATTERNS["few"] = (range(3, NPIX), range(3, NPIX, 2), range(3, NPIX, 4))


@pytest.mark.parametrize("pattern", LOW_RANK_PATTERNS)
@pytest.mark.parametrize("name", LOW_RANK)
def test_the_low_rank_solve_matches_the_reference(name, pattern):
    V, values, d = LOW_RANK[name]
    G, M, T = (mask(pixels) for pixels in LOW_RANK_PATTERNS[pattern])
    C = (V * values) @ V.T + np.diag(d)
    X = np.random.default_rng(1).standard_normal((NSPEC, NPIX))
    X = X @ np.linalg.cholesky(C).T
    solve = functools.partial(solve_low_rank, V, values, d, G, M, T)
    solution = solve(X, covariance=True)
    mean, cov, chi2 = conditional(
        low_rank(V, values, d), np.flatnonzero(G), X, np.flatnonzero(T)
    )
    at = np.flatnonzero(M[G])
    mean, cov = mean[:, at], cov[np.ix_(at, at)]
    # Design section 11: 10 eps kappa_W, and 1e-9 where |O| >= 2K and kappa_W <= 1e8.
    kappa = kappa_w(V, values, d)
    tol = 10 * EPS * kappa
    if NPIX - np.count_nonzero(G) >= 2 * values.size and kappa <= 1e8:
        tol = min(tol, 1e-9)
    scale = np.sqrt(np.outer(np.diagonal(cov), np.diagonal(cov)))
    assert np.max(np.abs(solution.mean - mean) / np.sqrt(np.diagonal(cov))) <= tol
    assert np.max(np.abs(solution.cov - cov) / scale) <= tol
    np.testing.assert_array_equal(solution.cov, solution.cov.T)
    np.testing.assert_array_equal(np.diagonal(solution.cov), solution.var)
    # chi2 comes from the factor of C_TT|O, or with draws from that of C_MM|O.
    drawn = solve(X, np.zeros((NSPEC, np.count_nonzero(M))))
    for result in (solution, drawn):
        error = np.abs(result.chi2 - chi2) / (chi2 + np.count_nonzero(T))
        assert np.max(error) <= tol
    # Unit normals give the map U from normals to draws, as in draw_covariance. In the
    # order (M \ T, T), U is upper triangular with U U^T = C_MM|O: the unique such
    # factor, R_MM^-T of solve_pattern, so that both routes give the same draws.
    m = np.count_nonzero(M)
    unit = solve(X[[0] * m], np.eye(m))
    order = np.concatenate([np.flatnonzero(~T[M]), np.flatnonzero(T[M])])
    U = (unit.draws - unit.mean)[np.ix_(order, order)].T
    np.testing.assert_array_equal(np.tril(U, -1), 0.0)
    assert (
        np.max(
            np.abs(U @ U.T - cov[np.ix_(order, order)]) / scale[np.ix_(order, order)]
        )
        <= tol
    )


@pytest.mark.parametrize("G", [[0], [5]], ids=["in G", "in O"])
@pytest.mark.parametrize(
    ("value", "rejected"), [(1e13 - 1.5, False), (1e13 - 0.5, True)]
)
def test_the_low_rank_solve_rejects_kappa_k_above_1e13(value, rejected, G):
    # V = e_0 and d = 1: E^T E = lambda, so kappa_K = 1 + lambda, 1e13 - 0.5 and
    # 1e13 + 0.5 to the rounding of sqrt(lambda)^2, below 0.01, with pixel 0 in G or O.
    arguments = (np.eye(NPIX, 1), np.array([value]), np.ones(NPIX))
    pattern = mask(G), mask(G), mask([]), np.zeros((1, NPIX))
    if not rejected:
        solve_low_rank(*arguments, *pattern)
        return
    message = r"^numerically singular: kappa_K = 1e\+13 for I \+ Lambda"
    with pytest.raises(LinAlgError, match=message):
        solve_low_rank(*arguments, *pattern)


@pytest.mark.parametrize(
    ("call", "message"),
    [(1, "^not positive definite: .* order 2 of I"), (2, "^not positive definite in")],
    ids=["system", "conditional"],
)
def test_the_low_rank_solve_rejects_a_factor_that_is_not_positive_definite(
    monkeypatch, call, message
):
    # kappa_K <= 1e13 keeps I + E_O^T E_O and C_TT|O positive definite in practice,
    # so the call-th dpotrf is made to fail.
    calls = []
    factor = _linalg.lapack.dpotrf

    def failing(A, lower):
        calls.append(A)
        return (A, 2) if len(calls) == call else factor(A, lower=lower)

    monkeypatch.setattr(_linalg.lapack, "dpotrf", failing)
    V, values, d = LOW_RANK["low1e2"]
    G, M, T = (mask(pixels) for pixels in PATTERNS["run"])
    with pytest.raises(LinAlgError, match=message):
        solve_low_rank(V, values, d, G, M, T, np.zeros((1, NPIX)))
