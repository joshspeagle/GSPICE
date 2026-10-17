import logging
import time
from typing import Any, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._model import Model, sample_factor
from ._predict import pixelwise, validated
from ._spectra import (
    Integer,
    Progress,
    check_progress,
    chunks,
    integer,
    integers,
    located,
    require,
)

_logger = logging.getLogger("gspice")


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


def noise_audit(
    model: Model,
    flux: ArrayLike,
    ivar: ArrayLike,
    mask: ArrayLike | None = None,
    *,
    nguard: Integer = 20,
    progress: Progress | None = None,
) -> NDArray[np.float64]:
    """Ratio of the residual scatter to the reported noise at each pixel.

    With f the flux and fhat its prediction by `pixelwise` with nguard, in flux units,

        rho_i^2 = mean_s (f_si - fhat_si)^2 ivar_si,

    where the mean runs over the spectra s whose pixel i is good (see `scale`). With t
    the noise-free flux, let k_i be the mean of ivar_si var(f_si - t_si), which is 1
    where ivar is right, and e_i the mean of ivar_si (t_si - fhat_si)^2. Then

        E[rho_i^2] = k_i + E[e_i]

    if fhat_si does not depend on the noise of f_si. That holds for spectra not used to
    fit model, with mask and ivar independent of the noise and noise uncorrelated beyond
    nguard, except where the interpolation of a bad pixel carries f_si or a pixel within
    nguard of i into fhat_si (see `pixelwise`). With several S/N bins, f_si also enters
    the median S/N that picks the bin of s. e_i holds the astrophysical error, the noise
    propagated from the conditioning pixels and the estimation error of model, so
    E[rho_i^2] >= k_i where the identity holds, and e_i grows with S/N. Next to a bad
    run longer than nguard, rho_i^2 can fall below k_i. A model absorbs into its
    covariance a k_i that its training spectra share, so the z of `pixelwise` is blind
    to it; rho is not.

    Parameters
    ----------
    model : Model
        The model.
    flux, ivar, mask : array_like, shape (nspec, npix)
        As in `scale`, with the npix of model.
    nguard : int, optional
        The half-width of the guard window of `pixelwise` (pixels); >= 0.
    progress : callable, optional
        progress(stage, done, total), called after each chunk of spectra with the
        stage "noise_audit", the number of spectra done and their total.

    Returns
    -------
    ndarray, shape (npix,)
        rho (unit-free); NaN at the pixels where no spectrum is good.

    Raises
    ------
    TypeError
        If model is not a Model, nguard is not an integer, progress is not callable
        or None, or flux, ivar or mask has a dtype that `scale` does not allow.
    ValueError
        If flux, ivar or mask has a shape or a value that `scale` does not allow, flux
        has another npix than model, nguard < 0, or a spectrum has no good pixel.
    numpy.linalg.LinAlgError
        As in `pixelwise`.

    Notes
    -----
    Cost: that of `pixelwise`, which runs on one chunk of max(1, 2**24 // npix)
    spectra at a time and checks and bins each chunk again. Beyond the inputs, their
    float64 copies if they are long double, and the arrays of `pixelwise` for one
    chunk, noise_audit holds a bool array of the input shape and a few float64 arrays
    of the chunk size.
    """
    start = time.perf_counter()
    nguard = integer(nguard, "nguard", 0)
    check_progress(progress)
    flux, ivar, bad = validated(model, flux, ivar, mask, inflate=False)
    nspec, npix = flux.shape
    # The weights of the bins with spectra, before any spectrum, so that a covariance
    # that fails its check raises at once, as in pixelwise.
    for b in np.unique(model._bins(flux, ivar, bad)).tolist():
        model._guard_weights(b, nguard)
    total, count = np.zeros(npix), np.zeros(npix, dtype=np.int64)
    for rows in chunks(nspec, npix):
        good = ~bad[rows]
        count += np.count_nonzero(good, axis=0)
        # One call per chunk, so that the predictions of all spectra are never held
        # at once.
        predicted = pixelwise(
            model, flux[rows], ivar[rows], bad[rows], nguard=nguard
        ).flux
        # 0 at the bad pixels, whose flux and ivar are never read.
        terms = np.zeros(good.shape)
        np.subtract(flux[rows], predicted, out=terms, where=good)
        np.square(terms, out=terms)
        np.multiply(terms, ivar[rows], out=terms, where=good)
        total += terms.sum(axis=0)
        if progress is not None:
            progress("noise_audit", rows.stop, nspec)
    rho = np.full(npix, np.nan)
    np.divide(total, count, out=rho, where=count > 0)
    np.sqrt(rho, out=rho)
    _logger.debug(
        "noise_audit: %d spectra in %.3f s", nspec, time.perf_counter() - start
    )
    return rho


def _counts(value: ArrayLike, name: str, minimum: int) -> NDArray[np.integer[Any]]:
    array = integers(value, name)
    # Below 2^53, float64 holds every integer, so sample_factor tests the sign of
    # n_train - n_cond - n_pred - 2 exactly.
    ok = (array >= minimum) & (array < 2**53)
    require(ok, array, name, f"integers from {minimum} to 2^53 - 1", None)
    return array
