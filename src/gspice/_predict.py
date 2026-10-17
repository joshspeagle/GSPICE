import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._model import Model
from ._spectra import (
    Progress,
    check_progress,
    chunks,
    integer,
    interpolate,
    require_good_pixel,
    root_mean_ivar,
    to_flux,
    to_working,
    validate,
)

_logger = logging.getLogger("gspice")


@dataclass(frozen=True, eq=False)
class Prediction:
    """The prediction of each pixel of a set of spectra, from `pixelwise`.

    The arrays that `pixelwise` returns are read-only.

    Attributes
    ----------
    flux : ndarray, shape (nspec, npix)
        The prediction of each pixel (flux).
    var : ndarray, shape (nspec, npix)
        The conditional variance of each pixel (flux^2), which includes the noise.
    z : ndarray, shape (nspec, npix)
        (f - flux) / sqrt(var) at good pixels, with f the input flux, and NaN at bad
        pixels (unit-free).
    """

    flux: NDArray[np.float64]
    var: NDArray[np.float64]
    z: NDArray[np.float64]


def pixelwise(
    model: Model,
    flux: ArrayLike,
    ivar: ArrayLike,
    mask: ArrayLike | None = None,
    *,
    nguard: int = 20,
    progress: Progress | None = None,
) -> Prediction:
    """Predict each pixel from all pixels farther than nguard from it (paper Alg. 3).

    For pixel i of a spectrum in S/N bin b, with x its working vector (see `Model`) and
    P the inverse of cov[b],

        x_i | x_O ~ N(-(P_gg^-1 P_gO x_O)_i, (P_gg^-1)_ii),
        g = {j : |j - i| <= nguard},

    where O holds the pixels outside g. Prediction.flux and Prediction.var are this
    mean and variance in flux units, as `unscale` returns them. Bad pixels enter as
    interpolated values; their z is NaN, and var omits the error of those values (next
    to a 60-pixel bad run at correlation length 20, <z^2> measured 0.65 to 2.45,
    depending on the noise level). nguard >= npix - 1 gives the prior.

    Parameters
    ----------
    model : Model
        The model.
    flux, ivar, mask : array_like, shape (nspec, npix)
        As in `scale`, with the npix of model.
    nguard : int, optional
        The half-width of the guard window g (pixels); >= 0.
    progress : callable, optional
        progress(stage, done, total), called after each chunk of spectra with the
        stage "pixelwise", the number of spectra done and their total.

    Returns
    -------
    Prediction
        The prediction, in flux units.

    Raises
    ------
    TypeError
        If model is not a Model, nguard is not an integer, progress is not callable or
        None, or flux, ivar or mask has a dtype that `scale` does not allow.
    ValueError
        If flux, ivar or mask has a shape or a value that `scale` does not allow, flux
        has another npix than model, nguard < 0, or a spectrum has no good pixel.
    numpy.linalg.LinAlgError
        If nguard < npix - 1 and the covariance of a bin with spectra fails the check
        of its precision matrix: it is not positive definite, or the reciprocal
        condition number of the covariance scaled to unit diagonal is below 1e-13.
        The message names the bin.

    Notes
    -----
    Cost: npix^3 flops for P at the first use of a bin, and about
    npix (w^3/3 + 2 npix w) flops for the guard weights at the first use of a bin and
    after each change of nguard, w = min(2 nguard + 1, npix); then 2 npix^2 flops per
    spectrum. The model keeps P and the guard weights (see `Model`).
    """
    start = time.perf_counter()
    nguard = integer(nguard, "nguard", 0)
    check_progress(progress)
    flux, ivar, bad = _validated(model, flux, ivar, mask)
    nspec, npix = flux.shape
    bins = model._bins(flux, ivar, bad)
    # The weights of the bins with spectra, before any spectrum, so that a covariance
    # that fails its check raises at once. A bin without spectra computes no P.
    for b in np.unique(bins).tolist():
        model._guard_weights(b, nguard)
    predicted, variance, z = (np.empty((nspec, npix)) for _ in range(3))
    for rows in chunks(nspec, npix):
        ftilde = interpolate(flux[rows], bad[rows])
        a = root_mean_ivar(ivar[rows], bad[rows])
        out = predicted[rows], variance[rows], z[rows]
        if len(model.cov) == 1:
            _predict(model, 0, nguard, ftilde, a, *out)
        else:
            _predict_bins(model, nguard, ftilde, a, bins[rows], out)
        z[rows][bad[rows]] = np.nan
        if progress is not None:
            progress("pixelwise", rows.stop, nspec)
    for array in (predicted, variance, z):
        array.flags.writeable = False
    _logger.debug("pixelwise: %d spectra in %.3f s", nspec, time.perf_counter() - start)
    return Prediction(predicted, variance, z)


def _validated(
    model: Model, flux: ArrayLike, ivar: ArrayLike, mask: ArrayLike | None
) -> tuple[NDArray[Any], NDArray[Any], NDArray[np.bool_]]:
    if not isinstance(model, Model):
        raise TypeError(f"model: expected a Model, got {type(model).__name__}")
    flux, ivar, bad = validate(flux, ivar, mask)
    if flux.shape[1] != model.mean.shape[1]:
        raise ValueError(
            f"flux: expected {model.mean.shape[1]} pixels like model, got shape "
            f"{flux.shape}"
        )
    require_good_pixel(bad)
    return flux, ivar, bad


def _predict_bins(
    model: Model,
    nguard: int,
    ftilde: NDArray[np.float64],
    a: NDArray[np.float64],
    bins: NDArray[np.intp],
    out: tuple[NDArray[np.float64], ...],
) -> None:
    # The temporaries of a bin end with the call.
    for b in np.unique(bins).tolist():
        at = np.flatnonzero(bins == b)
        parts = np.empty((3, at.size, ftilde.shape[1]))
        _predict(model, b, nguard, ftilde[at], a[at], *parts)
        for array, part in zip(out, parts, strict=True):
            array[at] = part


def _predict(
    model: Model,
    b: int,
    nguard: int,
    x: NDArray[np.float64],
    a: NDArray[np.float64],
    predicted: NDArray[np.float64],
    variance: NDArray[np.float64],
    z: NDArray[np.float64],
) -> None:
    W, var = model._guard_weights(b, nguard)
    center = 0.0 if model.center is None else model.center
    mean = model.mean[b]
    # x holds the interpolated flux of spectra of bin b and becomes their working
    # vector, X of scale bit for bit.
    to_working(x, a, center, mean)
    np.matmul(x, W, out=predicted)
    np.subtract(x, predicted, out=z)
    z /= np.sqrt(var)
    to_flux(predicted, a, mean, center, out=predicted)
    np.divide(var, np.square(a)[:, np.newaxis], out=variance)
