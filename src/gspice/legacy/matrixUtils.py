"""The 2022 matrix functions: the inverse of a matrix and of its submatrices."""

import numpy as np
from scipy import linalg
from scipy.linalg import lapack

from ._errors import _LegacyError


def cholesky_inv(M):
    """Inverse of a positive definite matrix, from its Cholesky factor.

    M = L L^T with L lower triangular, and LAPACK dpotri forms M^-1 from L.

    Parameters
    ----------
    M : array_like, shape (N, N)
        Positive definite matrix with finite entries. Only its lower triangle
        enters the result.

    Returns
    -------
    ndarray of float64, shape (N, N)
        The inverse of M, symmetric. A float32 M is factorized in float32.

    Raises
    ------
    ValueError
        If M is not square or has a non-finite entry.
    numpy.linalg.LinAlgError
        If M is not positive definite.

    Notes
    -----
    Cost: about N^3 flops.
    """

    L = linalg.cholesky(M, lower=True)

    # dpotri rejects a 0 x 0 matrix and prints a message.
    if L.shape == (0, 0):
        return np.zeros((0, 0))

    # Call LAPACK dpotri to get inverse (only lower triangle populated)
    Minv0 = lapack.dpotri(L, lower=True)[0]

    Minv = Minv0 + Minv0.T - np.diag(Minv0.diagonal())
    return Minv


def submatrix_inv(M, Minv, imask, bruteforce=False):
    """Inverse of the submatrix of M that imask keeps, from the inverse of M.

    With the kept indices k and the removed indices r, A = M[k, k] and
    A^-1 = P - Q U^-1 Q^T, where P = Minv[k, k], Q = Minv[k, r] and U = Minv[r, r].

    Parameters
    ----------
    M : ndarray, shape (N, N)
        Symmetric positive definite matrix.
    Minv : ndarray, shape (N, N)
        Inverse of M. bruteforce does not read it.
    imask : ndarray, shape (N,)
        Nonzero for the rows and columns to keep.
    bruteforce : bool, optional
        If True, invert A with cholesky_inv.

    Returns
    -------
    Ainv : ndarray, shape (nk, nk)
        The inverse of A. Without bruteforce, if imask keeps every row, Minv
        itself, not a copy.

    Raises
    ------
    ValueError
        If M is not square or imask.shape[0] is not N.
    numpy.linalg.LinAlgError
        If U is singular, or with bruteforce, if A is not positive definite.

    Notes
    -----
    Cost: about 2 nr^3 + 2 nk nr (nk + nr) flops, for nr removed and nk kept
    indices; with bruteforce, about nk^3 flops.
    """
    if M.shape[0] != M.shape[1]:
        raise _LegacyError("M must be a square matrix.")
    if imask.shape[0] != M.shape[0]:
        raise _LegacyError("M and imask have incompatible dimensions.")

    # ensure imask is boolean type because will use logical operators
    imask = imask.astype(bool)

    k = np.where(imask)[0]

    r = np.where(~imask)[0]
    nr = len(r)

    if bruteforce:
        A = M[np.ix_(k, k)]
        Ainv = cholesky_inv(A)
        return Ainv

    if nr == 0:
        return Minv

    Uinv = linalg.inv(Minv[np.ix_(r, r)])
    Qt = Minv[np.ix_(r, k)]  # faster than Q assuming nr < nk
    Ainv = Minv[np.ix_(k, k)] - Qt.T @ Uinv @ Qt

    return Ainv


def submatrix_inv_mult(M, Minv, imask, Y, MinvY, pad=True, bruteforce=False):
    """Product of the inverse of a submatrix of M with Y, from the inverse of M.

    With the kept indices k, the removed indices r, A = M[k, k], P = Minv[k, k],
    Q = Minv[k, r] and U = Minv[r, r], A^-1 Y[k] = P Y[k] - Q U^-1 Q^T Y[k]. As
    MinvY[k] = P Y[k] + Q Y[r] and MinvY[r] = Q^T Y[k] + U Y[r], this is
    MinvY[k] - Q (U^-1 (MinvY[r] - U Y[r]) + Y[r]), which the function evaluates,
    so Y[r] can hold any values.

    Parameters
    ----------
    M : ndarray, shape (N, N)
        Symmetric positive definite matrix.
    Minv : ndarray, shape (N, N)
        Inverse of M. bruteforce does not read it.
    imask : ndarray, shape (N,)
        Nonzero for the rows and columns to keep.
    Y : ndarray, shape (N, m)
        The matrix to multiply.
    MinvY : ndarray, shape (N, m)
        Minv @ Y. bruteforce checks only that it is 2-D.
    pad : bool, optional
        If True, return N rows, with 0 in the rows r.
    bruteforce : bool, optional
        If True, invert A with cholesky_inv, and return nk rows whatever pad is.

    Returns
    -------
    ndarray, shape (N, m) if pad, else (nk, m)
        A^-1 Y[k]. Without bruteforce, if imask keeps every row, MinvY itself,
        not a copy.

    Raises
    ------
    ValueError
        If M is not square, imask.shape[0] is not N, or Y or MinvY is not 2-D.
    numpy.linalg.LinAlgError
        If U has more than one row and is not positive definite, or with
        bruteforce, if A is not positive definite.

    Notes
    -----
    Cost: about nr^3 + 2 nr (2 nr + nk) m flops, for nr removed and nk kept
    indices; with bruteforce, about nk^3 + 2 nk^2 m flops.
    """
    if M.shape[0] != M.shape[1]:
        raise _LegacyError("M must be a square matrix.")
    if imask.shape[0] != M.shape[0]:
        raise _LegacyError("M and imask have incompatible dimensions.")
    if Y.ndim != 2:
        raise _LegacyError("Y must be column vector.")
    if MinvY.ndim != 2:
        raise _LegacyError("MinvY must be column vector.")

    # ensure imask is boolean type because will use logical operators
    imask = imask.astype(bool)

    k = np.where(imask)[0]

    r = np.where(~imask)[0]
    nr = len(r)

    if bruteforce:
        A = M[np.ix_(k, k)]
        Ainv = cholesky_inv(A)
        Ainvy = Ainv.T @ Y[k, :]
        return Ainvy

    if nr == 0:
        return MinvY

    U = Minv[np.ix_(r, r)]
    Yr = Y[r, :]
    Qty = MinvY[r, :] - U @ Yr
    Qt = Minv[np.ix_(r, k)]

    if U.shape[0] == 1:
        UinvQtY = Qty / U[0]
    else:
        UinvQtY = cholesky_inv(U) @ Qty

    if pad:
        AinvY0 = MinvY.copy(order="K")
        AinvY0[k, :] -= Qt.T @ (UinvQtY + Yr)
        AinvY0[r, :] = 0
        return AinvY0
    else:
        AinvY = MinvY[k, :] - Qt.T @ (UinvQtY + Yr)
        return AinvY
