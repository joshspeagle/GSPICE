import numpy as np
from numpy.linalg import LinAlgError
from numpy.typing import NDArray
from scipy.linalg import lapack

# Design decision A26.
MIN_RCOND = 1e-13


def precision(S: NDArray[np.float64]) -> NDArray[np.float64]:
    # S is symmetric, so S.T is S, and it is F-contiguous when S is C-contiguous: f2py
    # then copies it without a transpose.
    L, info = lapack.dpotrf(S.T, lower=1)
    if info:
        raise LinAlgError(
            f"S is not positive definite: dpotrf failed at its leading minor of order "
            f"{info}"
        )
    # With D = diag(S)^-1/2, D S D = (D L)(D L)^T, and D L is lower triangular with a
    # positive diagonal: it is the Cholesky factor of S scaled to unit diagonal.
    d = 1 / np.sqrt(np.diagonal(S))
    anorm = np.max(d * (np.abs(S) @ d))
    rcond, _ = lapack.dpocon(L * d[:, np.newaxis], anorm, uplo="L")
    # not >= also rejects a NaN rcond.
    if not rcond >= MIN_RCOND:
        raise LinAlgError(
            f"S is numerically singular: rcond = {rcond:.2g} for S scaled to unit "
            f"diagonal, below {MIN_RCOND:.0e}"
        )
    P: NDArray[np.float64]
    P, _ = lapack.dpotri(L, lower=1, overwrite_c=1)
    _mirror(P)
    # P is exactly symmetric, so its transpose is P as a C-contiguous view.
    return P.T


def _mirror(A: NDArray[np.float64]) -> None:
    # A is F-ordered, so each column of its upper triangle is one contiguous write.
    for j in range(1, A.shape[0]):
        A[:j, j] = A[j, :j]
