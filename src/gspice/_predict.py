import logging
import time
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
from numpy.linalg import LinAlgError
from numpy.typing import ArrayLike, NDArray

from ._linalg import Conditional, group_rows, solve_pattern, solve_prior
from ._model import Model
from ._spectra import (
    Progress,
    as_array,
    boolean,
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


class _Covariances(NamedTuple):
    patterns: tuple[NDArray[np.float64], ...]
    pattern: NDArray[np.intp]
    refscale: NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class Repaired:
    """Spectra whose replaced pixels hold their conditional mean, from `infill`.

    The arrays that `infill` returns are read-only. With covariance=True,
    `covariance(s)` gives the conditional covariance of the replaced pixels of
    spectrum s.

    Attributes
    ----------
    flux : ndarray, shape (nspec, npix)
        The conditional mean at the replaced pixels, and the input flux elsewhere
        (flux).
    var : ndarray, shape (nspec, npix)
        The conditional variance at the replaced pixels, which includes the noise;
        1/ivar at the other good pixels; inf at the other bad pixels (flux^2).
    replaced : ndarray of bool, shape (nspec, npix)
        The replaced pixels M.
    ignored : ndarray of bool, shape (nspec, npix)
        The pixels that are neither conditioned on nor replaced, G minus M.
    chi2 : ndarray, shape (nspec,)
        r_T^T C_TT^-1 r_T over the replaced pixels T that are good in the input, with
        r_T their input values minus the conditional mean and C_TT their conditional
        covariance; 0 where dof is 0 (unit-free).
    dof : ndarray of int64, shape (nspec,)
        The number |T| of replaced pixels that are good in the input.
    """

    flux: NDArray[np.float64]
    var: NDArray[np.float64]
    replaced: NDArray[np.bool_]
    ignored: NDArray[np.bool_]
    chi2: NDArray[np.float64]
    dof: NDArray[np.int64]
    _covariances: _Covariances | None = field(default=None, repr=False)

    def covariance(self, s: int) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Return the conditional covariance of the replaced pixels of spectrum s.

        Parameters
        ----------
        s : int
            The row of the spectrum; 0 <= s < nspec.

        Returns
        -------
        pixels : ndarray of int64, shape (m,)
            The replaced pixels of spectrum s, in increasing order.
        cov : ndarray, shape (m, m)
            Their conditional covariance (flux^2). Its diagonal is var[s, pixels].

        Raises
        ------
        TypeError
            If s is not an integer.
        ValueError
            If infill ran without covariance=True, or s < 0 or s >= nspec.
        """
        if self._covariances is None:
            raise ValueError(
                "covariance: expected a result computed with covariance=True, got one "
                "computed without it"
            )
        s = integer(s, "s", 0)
        nspec = len(self.chi2)
        if s >= nspec:
            raise ValueError(
                f"s: expected an integer < {nspec}, the number of spectra, got {s}"
            )
        patterns, pattern, refscale = self._covariances
        # As infill divides var, so that the diagonal is var[s, pixels] bit for bit.
        cov = patterns[pattern[s]] / np.square(refscale[s])
        return np.flatnonzero(self.replaced[s]), cov


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


def infill(
    model: Model,
    flux: ArrayLike,
    ivar: ArrayLike,
    mask: ArrayLike | None = None,
    *,
    replace: ArrayLike | None = None,
    ignore: ArrayLike | None = None,
    covariance: bool = False,
    progress: Progress | None = None,
) -> Repaired:
    """Replace pixels by their Gaussian conditional mean given the trusted pixels.

    For a spectrum in S/N bin b, with x its working vector (see `Model`) and P the
    inverse of cov[b], let G = replace | ignore | bad, with the bad pixels
    (mask != 0) | (ivar == 0), and let O hold the other pixels. Then

        x_G | x_O ~ N(-P_GG^-1 P_GO x_O, P_GG^-1).

    Repaired.flux is this mean at the replaced pixels M = replace, in flux units as
    `unscale` returns it, and the input flux elsewhere. The pixels of G outside M keep
    their input value and are reported as ignored; a pixel in both replace and ignore
    is replaced. Values in G never enter the mean, so an outlier there cannot change
    it. chi2 compares the input values at T = M & ~bad, the replaced pixels that are
    good in the input, with their conditional distribution. Where G covers all pixels,
    the result is the prior N(0, cov[b]), which infill computes from cov[b] without P.
    Spectra of one S/N bin with equal G, M and T form a pattern, whose spectra in one
    chunk of max(1, 2**24 // npix) rows share one factorization of P_GG. To infill
    with another model the pixels that a result r replaced and ignored, pass
    replace=r.replaced and ignore=r.ignored.

    Parameters
    ----------
    model : Model
        The model.
    flux, ivar, mask : array_like, shape (nspec, npix)
        As in `scale`, with the npix of model.
    replace : array_like of bool, shape (nspec, npix) or (npix,), optional
        The pixels to replace, M. An (npix,) mask applies to every spectrum, for
        example to predict a block. None: the bad pixels.
    ignore : array_like of bool, shape (nspec, npix) or (npix,), optional
        Pixels that are neither conditioned on nor replaced, unless they are in
        replace, such as a guard around a block. None: no pixels.
    covariance : bool, optional
        If True, keep the conditional covariance of each pattern for
        `Repaired.covariance`.
    progress : callable, optional
        progress(stage, done, total), called after each chunk of spectra with the
        stage "infill", the number of spectra done and their total.

    Returns
    -------
    Repaired
        The infilled spectra, in flux units.

    Raises
    ------
    TypeError
        If model is not a Model, covariance is not a bool, progress is not callable or
        None, flux, ivar or mask has a dtype that `scale` does not allow, or replace
        or ignore is not of a bool dtype.
    ValueError
        If flux, ivar or mask has a shape or a value that `scale` does not allow, flux
        has another npix than model, a spectrum has no good pixel, or replace or
        ignore has another shape than (nspec, npix) or (npix,).
    numpy.linalg.LinAlgError
        If the covariance of a bin fails the check of its precision matrix, as in
        `pixelwise`, where a spectrum of the bin has a pixel to replace and G does not
        cover all its pixels; or if P_GG of a pattern, or cov[b] at T for a pattern
        whose G covers all pixels, is not positive definite. The message names the
        bin, and for a pattern its first spectrum.

    Notes
    -----
    Cost: npix^3 flops for P at the first use of a bin; then per pattern and chunk
    |G|^3/3 + |M|^3/3 flops, and |M|^3/3 more with covariance=True, and per spectrum
    2 |G| (npix + |G|) flops. An (npix,) mask on spectra with their own bad pixels
    makes one pattern per spectrum, with G the mask and those bad pixels, so a large
    mask then costs |G|^3/3 flops per spectrum. The model keeps P (see `Model`). With
    covariance=True, the result keeps |M|^2 x 8 bytes per pattern and chunk.
    """
    start = time.perf_counter()
    covariance = boolean(covariance, "covariance")
    check_progress(progress)
    flux, ivar, bad = _validated(model, flux, ivar, mask)
    nspec, npix = flux.shape
    replace = _pixels(replace, "replace", bad)
    ignore = _pixels(ignore, "ignore", np.broadcast_to(False, bad.shape))
    replaced, ignored = (np.empty((nspec, npix), dtype=bool) for _ in range(2))
    needs = np.empty(nspec, dtype=bool)
    for rows in chunks(nspec, npix):
        M = replace[rows]
        G = M | ignore[rows] | bad[rows]
        replaced[rows] = M
        ignored[rows] = G & ~M
        needs[rows] = M.any(axis=1) & ~G.all(axis=1)
    bins = model._bins(flux, ivar, bad)
    # P of the bins that need it, before any spectrum, so that a covariance that fails
    # its check raises at once.
    for b in np.unique(bins[needs]).tolist():
        model._precision(b)
    center = 0.0 if model.center is None else model.center
    result = np.empty((nspec, npix))
    var = np.full((nspec, npix), np.inf)
    chi2 = np.zeros(nspec)
    dof = np.empty(nspec, dtype=np.int64)
    refscale = np.empty(nspec)
    patterns: list[NDArray[np.float64]] = []
    pattern = np.empty(nspec, dtype=np.intp)
    count = 0
    # The bin of each spectrum, one-hot, is part of the key of its pattern.
    one_hot = np.eye(len(model.cov), dtype=bool)
    for rows in chunks(nspec, npix):
        good = ~bad[rows]
        result[rows] = flux[rows]
        np.divide(1.0, ivar[rows], out=var[rows], where=good, dtype=np.float64)
        # x holds the interpolated flux, which becomes the working vector of each
        # pattern, X of scale bit for bit; the input flux at bad pixels is never read.
        x = interpolate(flux[rows], bad[rows])
        a = root_mean_ivar(ivar[rows], bad[rows])
        refscale[rows] = a
        M = replaced[rows]
        G = M | ignored[rows]
        T = M & good
        dof[rows] = np.count_nonzero(T, axis=1)
        keys = np.concatenate([G, M, T, one_hot[bins[rows]]], axis=1)
        for group in group_rows(keys):
            count += 1
            first, spectra = group[0], rows.start + group
            b = int(bins[spectra[0]])
            pixels = np.flatnonzero(M[first])
            cov = np.zeros((0, 0)) if covariance else None
            if pixels.size:
                X = x[group]
                to_working(X, a[group], center, model.mean[b])
                key = G[first], M[first], T[first]
                solution = _solve(model, b, *key, X, covariance, spectra[0])
                at = np.ix_(spectra, pixels)
                mean = model.mean[b, pixels]
                result[at] = to_flux(solution.mean, a[group], mean, center)
                var[at] = solution.var / np.square(a[group])[:, np.newaxis]
                chi2[spectra] = solution.chi2
                cov = solution.cov
            if cov is not None:
                pattern[spectra] = len(patterns)
                patterns.append(cov)
        if progress is not None:
            progress("infill", rows.stop, nspec)
    for array in (result, var, replaced, ignored, chi2, dof):
        array.flags.writeable = False
    _logger.debug(
        "infill: %d spectra, %d patterns in %.3f s",
        nspec,
        count,
        time.perf_counter() - start,
    )
    covariances = (
        _Covariances(tuple(patterns), pattern, refscale) if covariance else None
    )
    return Repaired(result, var, replaced, ignored, chi2, dof, _covariances=covariances)


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


def _pixels(
    value: ArrayLike | None, name: str, default: NDArray[np.bool_]
) -> NDArray[np.bool_]:
    if value is None:
        return default
    array = as_array(value, name)
    # A bool dtype, so that no integer array is read as pixel indices.
    if array.dtype != bool:
        raise TypeError(f"{name}: expected a bool dtype, got {array.dtype}")
    if array.shape not in (default.shape, default.shape[1:]):
        raise ValueError(
            f"{name}: expected shape {default.shape} like flux or "
            f"{default.shape[1:]}, got {array.shape}"
        )
    return np.broadcast_to(array, default.shape)


def _solve(
    model: Model,
    b: int,
    G: NDArray[np.bool_],
    M: NDArray[np.bool_],
    T: NDArray[np.bool_],
    X: NDArray[np.float64],
    covariance: bool,
    row: int,
) -> Conditional:
    # infill computed P before the first chunk, so only a solve can raise here.
    try:
        if G.all():
            return solve_prior(model.cov[b], M, T, X, covariance=covariance)
        return solve_pattern(model._precision(b), G, M, T, X, covariance=covariance)
    except LinAlgError as error:
        raise LinAlgError(
            f"model: cov[{b}] is {error}, for the pattern of row {row}"
        ) from None


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
