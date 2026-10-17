from typing import NamedTuple

import numpy as np
from numpy.linalg import LinAlgError
from numpy.typing import NDArray
from scipy.linalg import lapack

# Design decision A26.
MIN_RCOND = 1e-13


class Conditional(NamedTuple):
    mean: NDArray[np.float64]
    var: NDArray[np.float64]
    chi2: NDArray[np.float64]
    cov: NDArray[np.float64] | None
    draws: NDArray[np.float64] | None


def precision(S: NDArray[np.float64]) -> NDArray[np.float64]:
    # S is symmetric, so S.T is S, and it is F-contiguous when S is C-contiguous: f2py
    # then copies it without a transpose.
    L, info = lapack.dpotrf(S.T, lower=1)
    if info:
        raise LinAlgError(
            f"not positive definite: dpotrf failed at its leading minor of order {info}"
        )
    rcond = _rcond(S, L)
    # not >= also rejects a NaN rcond.
    if not rcond >= MIN_RCOND:
        raise LinAlgError(
            f"numerically singular: rcond = {rcond:.2g} for the matrix scaled to unit "
            f"diagonal, below {MIN_RCOND:.0e}"
        )
    P: NDArray[np.float64]
    P, _ = lapack.dpotri(L, lower=1, overwrite_c=1)
    _mirror(P)
    # P is exactly symmetric, so its transpose is P as a C-contiguous view.
    return P.T


def guard_weights(
    P: NDArray[np.float64], c: NDArray[np.float64], nguard: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    npix = P.shape[0]
    # Row i holds the weights of pixel i, so each one is a contiguous write; a window
    # that covers all pixels keeps its zeros, the prior mean.
    Wt = np.zeros((npix, npix))
    var = np.empty(npix)
    for i in range(npix):
        g = slice(max(0, i - nguard), min(npix, i + nguard + 1))
        if g.stop - g.start == npix:
            var[i] = c[i]
            continue
        R, info = lapack.dpotrf(P[g, g], lower=1)
        if info:
            raise LinAlgError(
                f"not positive definite in its precision matrix on the window of "
                f"pixel {i}"
            )
        e = np.zeros(g.stop - g.start)
        e[i - g.start] = 1.0
        v, _ = lapack.dpotrs(R, e, lower=1)
        # Rows of a C-contiguous P are contiguous, so P[g] is read in one pass.
        np.matmul(-v, P[g], out=Wt[i])
        Wt[i, g] = 0.0
        var[i] = v[i - g.start]
    return Wt.T, var


def solve_pattern(
    P: NDArray[np.float64],
    G: NDArray[np.bool_],
    M: NDArray[np.bool_],
    T: NDArray[np.bool_],
    X: NDArray[np.float64],
    normals: NDArray[np.float64] | None = None,
    covariance: bool = False,
) -> Conditional:
    nspec = X.shape[0]
    m = np.count_nonzero(M)
    if m == 0:
        return Conditional(
            np.zeros((nspec, 0)),
            np.zeros(0),
            np.zeros(nspec),
            np.zeros((0, 0)) if covariance else None,
            None if normals is None else np.zeros((nspec, 0)),
        )
    # G ordered as (G \ M, M \ T, T), so that M and T are trailing blocks of R.
    index = np.concatenate(
        [np.flatnonzero(G & ~M), np.flatnonzero(M & ~T), np.flatnonzero(T)]
    )
    a = index.size - m
    t = index.size - np.count_nonzero(T)
    R, info = lapack.dpotrf(P[np.ix_(index, index)], lower=1)
    if info:
        raise LinAlgError(
            "not positive definite in its precision matrix on the pixels G"
        )
    # x0 is x with its entries in G set to 0, so that no value in G enters the mean
    # and an outlier there cannot cancel.
    X0 = X.copy()
    X0[:, G] = 0.0
    y, _ = lapack.dtrtrs(R, P[index] @ X0.T, lower=1)
    RMM = R[a:, a:]
    # mu_M = -(R^-T y)_M = -R_MM^-T y_M, because R^-T is upper triangular.
    mean, _ = lapack.dtrtrs(RMM, -y[a:], lower=1, trans=1)
    # C_MM = (R_MM R_MM^T)^-1 = R_MM^-T R_MM^-1, whose diagonal is the column sums of
    # (R_MM^-1)^2.
    inverse, _ = lapack.dtrtri(RMM, lower=1)
    var = np.einsum("ij,ij->j", inverse, inverse)
    # chi2 = ||R_TT^T r_T||^2, because C_TT^-1 is the Schur complement R_TT R_TT^T.
    r = X[:, T].T - mean[t - a :]
    chi2 = np.sum((R[t:, t:].T @ r) ** 2, axis=0)
    order = np.argsort(index[a:])
    cov: NDArray[np.float64] | None = None
    if covariance:
        cov, _ = lapack.dlauum(inverse, lower=1)
        _mirror(cov)
        # dlauum sums in another order than einsum, so var replaces its diagonal.
        np.fill_diagonal(cov, var)
        cov = cov[np.ix_(order, order)]
    draws: NDArray[np.float64] | None = None
    if normals is not None:
        # (mu_G + R^-T z)_M = mu_M + R_MM^-T z_M.
        noise, _ = lapack.dtrtrs(RMM, normals[:, index[a:]].T, lower=1, trans=1)
        draws = (mean + noise).T[:, order]
    return Conditional(mean.T[:, order], var[order], chi2, cov, draws)


def solve_prior(
    C: NDArray[np.float64],
    M: NDArray[np.bool_],
    T: NDArray[np.bool_],
    X: NDArray[np.float64],
    covariance: bool = False,
) -> Conditional:
    # Where G covers all pixels, the conditional is the prior N(0, C): exact from C, and
    # without P, which a singular C does not have (design section 4.2).
    nspec = X.shape[0]
    pixels = np.flatnonzero(M)
    chi2 = np.zeros(nspec)
    if T.any():
        CTT = C[np.ix_(T, T)]
        L, info = lapack.dpotrf(CTT, lower=1)
        if info:
            raise LinAlgError("not positive definite on the pixels T")
        # The check of precision (decision A26), on C at T.
        rcond = _rcond(CTT, L)
        if not rcond >= MIN_RCOND:
            raise LinAlgError(
                f"numerically singular on the pixels T: rcond = {rcond:.2g}"
            )
        # chi2 = ||L^-1 x_T||^2, with C_TT = L L^T and the prior mean 0.
        r, _ = lapack.dtrtrs(L, X[:, T].T, lower=1)
        chi2 = np.sum(r**2, axis=0)
    cov = C[np.ix_(pixels, pixels)] if covariance else None
    mean = np.zeros((nspec, pixels.size))
    return Conditional(mean, np.diagonal(C)[pixels], chi2, cov, None)


def group_rows(keys: NDArray[np.bool_]) -> list[NDArray[np.intp]]:
    # Each packed row is one void scalar, which np.unique sorts faster than rows with
    # axis=0 (design decision A17). The view needs C order, which packbits copies from
    # keys. The groups follow the order of the packed keys.
    packed = np.ascontiguousarray(np.packbits(keys, axis=1))
    rows = packed.view(np.dtype((np.void, packed.shape[1])))[:, 0]
    _, inverse, counts = np.unique(rows, return_inverse=True, return_counts=True)
    # A stable sort keeps the rows of each group in increasing order.
    order = np.argsort(inverse, kind="stable")
    starts = np.cumsum(counts) - counts
    return [order[s : s + n] for s, n in zip(starts.tolist(), counts.tolist())]


def _rcond(S: NDArray[np.float64], L: NDArray[np.float64]) -> float:
    # The reciprocal condition number of S scaled to unit diagonal, from its Cholesky
    # factor L. With D = diag(S)^-1/2, D S D = (D L)(D L)^T, and D L is lower triangular
    # with a positive diagonal: it is the Cholesky factor of S scaled to unit diagonal.
    d = 1 / np.sqrt(np.diagonal(S))
    anorm = np.max(d * (np.abs(S) @ d))
    rcond, _ = lapack.dpocon(L * d[:, np.newaxis], anorm, uplo="L")
    return float(rcond)


def _mirror(A: NDArray[np.float64]) -> None:
    # A is F-ordered, so each column of its upper triangle is one contiguous write.
    for j in range(1, A.shape[0]):
        A[:j, j] = A[j, :j]
