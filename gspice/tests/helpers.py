"""Helpers shared by the tests (direct, GSPICE-free implementations used as references)."""

import numpy as np


def random_spd(n, rng, cond = 1e3):
    """Random symmetric positive-definite matrix with the given condition number."""
    q, _ = np.linalg.qr(rng.normal(size = (n, n)))
    w = np.logspace(0, -np.log10(cond), n)
    return (q * w) @ q.T


def direct_conditional(cov, X, ipred, icond):
    """Direct Gaussian conditional mean and covariance of pixels ipred given icond (no GSPICE code)."""
    k = np.flatnonzero(np.asarray(icond))
    ks = np.flatnonzero(np.asarray(ipred))
    W = np.linalg.solve(cov[np.ix_(k, k)], cov[np.ix_(k, ks)])
    return X[:, k] @ W, cov[np.ix_(ks, ks)] - cov[np.ix_(ks, k)] @ W
