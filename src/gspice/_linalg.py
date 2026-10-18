from typing import NamedTuple

import numpy as np
from numpy.linalg import LinAlgError
from numpy.typing import NDArray
from scipy.linalg import lapack

# Design decision A26.
MIN_RCOND = 1e-13
# Design decision A28.
MAX_KAPPA_W = 1e13


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


def low_rank_variances(
    V: NDArray[np.float64], values: NDArray[np.float64]
) -> NDArray[np.float64]:
    # The diagonal of V diag(values) V^T.
    variances: NDArray[np.float64] = np.square(V) @ values
    return variances


def woodbury(
    V: NDArray[np.float64], values: NDArray[np.float64], d: NDArray[np.float64]
) -> NDArray[np.float64]:
    # P = D^-1 - L L^T, L = D^-1 V R^-T, R R^T = Lambda^-1 + V^T D^-1 V: the inverse of
    # C = V Lambda V^T + D without forming C (design section 4.2), rejected where
    # kappa_W > 1e13 (decision A28).
    c = d + low_rank_variances(V, values)
    scaled = d / c
    B = V * np.sqrt(values) / np.sqrt(c)[:, np.newaxis]
    kappa = (np.max(scaled) + np.linalg.eigvalsh(B.T @ B)[-1]) / np.min(scaled)
    # not <= also rejects a NaN kappa_W.
    if not kappa <= MAX_KAPPA_W:
        raise LinAlgError(
            f"numerically singular: kappa_W = {kappa:.2g} for the matrix scaled to "
            f"unit diagonal, above {MAX_KAPPA_W:.0e}"
        )
    E = V / np.sqrt(d)[:, np.newaxis]
    # numpy computes E^T E and L L^T by syrk, so A and P are exactly symmetric.
    A = E.T @ E
    A[np.diag_indices_from(A)] += 1 / values
    R, info = lapack.dpotrf(A, lower=1)
    if info:
        raise LinAlgError(
            f"not positive definite: dpotrf failed at the leading minor of order "
            f"{info} of Lambda^-1 + V^T D^-1 V"
        )
    Lt, _ = lapack.dtrtrs(R, (V / d[:, np.newaxis]).T, lower=1)
    P: NDArray[np.float64] = Lt.T @ Lt
    np.negative(P, out=P)
    P[np.diag_indices_from(P)] += 1 / d
    return P


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
    variance: bool = True,
) -> Conditional:
    # variance=False skips the inverse of R_MM, |M|^3/3 flops, and var is then NaN;
    # covariance=True needs variance=True.
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
    R = factor_pattern(P[np.ix_(index, index)])
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
    if variance:
        inverse, _ = lapack.dtrtri(RMM, lower=1)
        var = np.einsum("ij,ij->j", inverse, inverse)
    else:
        var = np.full(m, np.nan)
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
        # (mu_G + R^-T z)_M = mu_M + R_MM^-T z_M, with z_M in the order of index[a:].
        z = normals[:, np.argsort(order)].T
        noise, _ = lapack.dtrtrs(RMM, z, lower=1, trans=1)
        draws = (mean + noise).T[:, order]
    return Conditional(mean.T[:, order], var[order], chi2, cov, draws)


def prior_chi2(
    CTT: NDArray[np.float64], XT: NDArray[np.float64]
) -> NDArray[np.float64]:
    # chi2 = ||L^-1 x_T||^2 of the prior N(0, C), with C_TT = L L^T, and XT the data at
    # T (design section 4.2).
    if not CTT.size:
        return np.zeros(XT.shape[0])
    L = factor_prior(CTT)
    r, _ = lapack.dtrtrs(L, XT.T, lower=1)
    chi2: NDArray[np.float64] = np.sum(r**2, axis=0)
    return chi2


def prior_draws(
    CMM: NDArray[np.float64], normals: NDArray[np.float64]
) -> NDArray[np.float64]:
    # L z_M, a draw from N(0, C_MM) with C_MM = L L^T. Unlike the solve at T, the
    # product needs no check of the condition of L.
    L, info = lapack.dpotrf(CMM, lower=1)
    if info:
        raise LinAlgError("not positive definite on the pixels M")
    draws: NDArray[np.float64] = normals @ L.T
    return draws


def factor_pattern(PGG: NDArray[np.float64]) -> NDArray[np.float64]:
    # The lower Cholesky factor of P at the pixels G.
    R: NDArray[np.float64]
    R, info = lapack.dpotrf(PGG, lower=1)
    if info:
        raise LinAlgError(
            "not positive definite in its precision matrix on the pixels G"
        )
    return R


def factor_prior(CTT: NDArray[np.float64]) -> NDArray[np.float64]:
    # The lower Cholesky factor of C at the pixels T, with the check of precision
    # (decision A26) on C at T.
    L: NDArray[np.float64]
    L, info = lapack.dpotrf(CTT, lower=1)
    if info:
        raise LinAlgError("not positive definite on the pixels T")
    rcond = _rcond(CTT, L)
    if not rcond >= MIN_RCOND:
        raise LinAlgError(f"numerically singular on the pixels T: rcond = {rcond:.2g}")
    return L


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
