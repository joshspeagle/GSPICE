import functools

import numpy as np
import pytest
from numpy.linalg import LinAlgError

from gspice import _linalg
from gspice._linalg import guard_weights, precision
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
