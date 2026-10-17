from typing import Any, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._model import sample_factor
from ._spectra import Integer, as_array, located, require


@overload
def finite_sample_factor(
    n_train: Integer, n_cond: Integer, n_pred: Integer = 1
) -> float: ...


@overload
def finite_sample_factor(
    n_train: ArrayLike, n_cond: ArrayLike, n_pred: ArrayLike = 1
) -> float | NDArray[np.float64]: ...


def finite_sample_factor(
    n_train: ArrayLike, n_cond: ArrayLike, n_pred: ArrayLike = 1
) -> float | NDArray[np.float64]:
    """Held-out calibration factor E[Z^2] / N_* for Gaussian data (paper l.247).

    A spectrum that the training spectra do not include is predicted at N_* pixels from
    N_k conditioning pixels, with the mean and the covariance (divisor N_s - 1) of N_s
    unweighted training spectra. With r the residual of the N_* pixels from their
    conditional mean and C their full conditional covariance, Z^2 = r^T C^-1 r has the
    mean

        E[Z^2] / N_* = f_s = (1 + 1/N_s) (N_s - 2)/(N_s - N_k - 2)
                             (N_s - 1)/(N_s - N_k - N_* - 2).

    It is exact for independent Gaussian working vectors and pixel sets that do not
    depend on the data. Pixels that are neither conditioned on nor predicted, such as
    a guard window or ignored pixels, do not count in N_k. f_s does not hold for data
    that are not Gaussian, such as scaled spectra with a spread of S/N. The arguments
    broadcast, and the arithmetic is float64, which holds each count exactly.

    Parameters
    ----------
    n_train : array_like of int
        N_s, the number of training spectra; from 1 to 2^53 - 1.
    n_cond : array_like of int
        N_k, the number of conditioning pixels; from 0 to 2^53 - 1.
    n_pred : array_like of int, optional
        N_*, the number of predicted pixels; from 1 to 2^53 - 1.

    Returns
    -------
    float or ndarray
        f_s (unit-free): a float if the arguments are scalars, and otherwise a float64
        array of their broadcast shape.

    Raises
    ------
    TypeError
        If an argument does not have an integer dtype; bool is not one.
    ValueError
        If an argument is outside its range, the arguments do not broadcast, or
        N_s - N_k - N_* - 2 <= 0. The message names the arguments and the first
        such entry.
    """
    counts = (
        _counts(n_train, "n_train", 1),
        _counts(n_cond, "n_cond", 0),
        _counts(n_pred, "n_pred", 1),
    )
    try:
        broadcast = np.broadcast_arrays(*counts)
    except ValueError:
        shapes = [count.shape for count in counts]
        raise ValueError(
            f"n_train, n_cond, n_pred: expected shapes that broadcast together, got "
            f"{shapes[0]}, {shapes[1]} and {shapes[2]}"
        ) from None
    factor = sample_factor(*broadcast)
    undefined = np.isnan(factor)
    if undefined.any():
        index = np.unravel_index(np.argmax(undefined), factor.shape)
        # Python integers, so that the difference cannot wrap around.
        n_s, n_k, n_star = (int(count[index]) for count in broadcast)
        raise ValueError(
            f"n_train, n_cond, n_pred: expected n_train - n_cond - n_pred - 2 > 0, got "
            f"{n_s} - {n_k} - {n_star} - 2 = {n_s - n_k - n_star - 2}{located(index)}"
        )
    return float(factor) if factor.ndim == 0 else factor


def _counts(value: ArrayLike, name: str, minimum: int) -> NDArray[np.integer[Any]]:
    array = as_array(value, name)
    if array.dtype.kind not in "iu":
        raise TypeError(f"{name}: expected an integer dtype, got {array.dtype}")
    # Below 2^53, float64 holds every integer, so sample_factor tests the sign of
    # n_train - n_cond - n_pred - 2 exactly.
    ok = (array >= minimum) & (array < 2**53)
    require(ok, array, name, f"integers from {minimum} to 2^53 - 1", None)
    return array
