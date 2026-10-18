import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
from numpy.linalg import LinAlgError
from numpy.typing import ArrayLike, NDArray

from ._linalg import Conditional, group_rows, prior_chi2, prior_draws, solve_pattern
from ._model import Model, sample_factor, submatrix_of, variances_of
from ._spectra import (
    Boolean,
    Integer,
    Progress,
    as_array,
    boolean,
    check_progress,
    check_rng,
    chunks,
    integer,
    interpolate,
    located,
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
        The conditional variance of each pixel (flux^2), which includes the noise;
        multiplied by f_s(N_s, N_k, 1) with inflate=True.
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

    `repair` returns one too. The arrays that `infill` and `repair` return are
    read-only. With covariance=True, `covariance(s)` gives the conditional covariance
    of the replaced pixels of spectrum s.

    Attributes
    ----------
    flux : ndarray, shape (nspec, npix)
        The conditional mean at the replaced pixels, and the input flux elsewhere
        (flux).
    var : ndarray, shape (nspec, npix)
        The conditional variance at the replaced pixels, which includes the noise,
        multiplied by f_s(N_s, N_k, 1) with inflate=True; 1/ivar at the other good
        pixels; inf at the other bad pixels (flux^2).
    replaced : ndarray of bool, shape (nspec, npix)
        The replaced pixels M.
    ignored : ndarray of bool, shape (nspec, npix)
        The pixels that are neither conditioned on nor replaced, G minus M.
    chi2 : ndarray, shape (nspec,)
        r_T^T C_TT^-1 r_T over the replaced pixels T that are good in the input, with
        r_T their input values minus the conditional mean and C_TT their conditional
        covariance, which inflate=True multiplies by f_s(N_s, N_k, 1); 0 where dof is
        0 (unit-free).
    dof : ndarray of int64, shape (nspec,)
        The number |T| of replaced pixels that are good in the input.
    fs : ndarray, shape (nspec,)
        f_s(N_s, N_k, dof) of `finite_sample_factor`, with N_s the n_train of the bin
        of the spectrum and N_k = npix - |G| its conditioning pixels, divided by
        f_s(N_s, N_k, 1) with inflate=True. NaN where dof is 0 or
        N_s - N_k - dof - 2 <= 0. 1 for a model with a constant centre, weights or a
        `LowRank` covariance, for which no factor is derived (unit-free). For the
        other models and replace and ignore masks that do not depend on the data,
        chi2 / (dof fs) has the mean 1 for held-out Gaussian spectra, with or without
        inflate.
    draws : ndarray, shape (nspec, npix), or None
        A draw from the conditional distribution at the replaced pixels, with the
        covariance of `covariance`, and the input flux elsewhere (flux); None if
        the call ran without rng.
    blocks : ndarray of bool, shape (nspec, nblock), or None
        The blocks of model.blocks that the block pass of `repair` flagged in each
        spectrum; None for a result of `infill` or of `repair` with blockwise=False.
    """

    flux: NDArray[np.float64]
    var: NDArray[np.float64]
    replaced: NDArray[np.bool_]
    ignored: NDArray[np.bool_]
    chi2: NDArray[np.float64]
    dof: NDArray[np.int64]
    fs: NDArray[np.float64]
    draws: NDArray[np.float64] | None
    blocks: NDArray[np.bool_] | None = None
    _covariances: _Covariances | None = field(default=None, repr=False)

    def covariance(self, s: Integer) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
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
            Their conditional covariance (flux^2), multiplied by f_s(N_s, N_k, 1) with
            inflate=True. Its diagonal is var[s, pixels].

        Raises
        ------
        TypeError
            If s is not an integer.
        ValueError
            If the call ran without covariance=True, or s < 0 or s >= nspec.
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
    nguard: Integer = 20,
    inflate: Boolean = False,
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
    depending on the noise level). A bad pixel farther than nguard from i takes its
    value from the nearest good pixels, so that a good pixel within nguard of i, or i
    itself next to a bad run longer than nguard, can enter the prediction of i.
    nguard >= npix - 1 gives the prior.

    Parameters
    ----------
    model : Model
        The model.
    flux, ivar, mask : array_like, shape (nspec, npix)
        As in `scale`, with the npix of model.
    nguard : int, optional
        The half-width of the guard window g (pixels); >= 0.
    inflate : bool, optional
        If True, multiply var by f_s(N_s, N_k, 1) of `finite_sample_factor`, with N_s
        the n_train of the bin of the spectrum and N_k = npix - |g| the conditioning
        pixels of each pixel, more near the ends, where g is clipped. z^2 then has the
        mean 1 for held-out Gaussian spectra. Needs a dense, mean-centred, unweighted
        model.
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
        If model is not a Model, nguard is not an integer, inflate is not a bool,
        progress is not callable or None, or flux, ivar or mask has a dtype that
        `scale` does not allow.
    ValueError
        If flux, ivar or mask has a shape or a value that `scale` does not allow, flux
        has another npix than model, nguard < 0, or a spectrum has no good pixel. If
        inflate is True and model has a constant centre, weights or a `LowRank`
        covariance, or N_s - N_k - 3 <= 0 at a pixel of a bin with spectra, where
        f_s(N_s, N_k, 1) is not defined; the message names N_s and N_k.
    numpy.linalg.LinAlgError
        If nguard < npix - 1 and the covariance of a bin with spectra fails the check
        of its precision matrix: it is not positive definite, or the reciprocal
        condition number of the covariance scaled to unit diagonal is below 1e-13, or
        kappa_W of a `LowRank` covariance is above 1e13. The message names the bin.

    Notes
    -----
    Cost: npix^3 flops for P at the first use of a bin, npix^2 K for a `LowRank` of
    rank K, and about npix (w^3/3 + 2 npix w) flops for the guard weights at the first use of a bin and
    after each change of nguard, w = min(2 nguard + 1, npix); then 2 npix^2 flops per
    spectrum. The model keeps P and the guard weights (see `Model`).
    """
    start = time.perf_counter()
    nguard = integer(nguard, "nguard", 0)
    inflate = boolean(inflate, "inflate")
    check_progress(progress)
    flux, ivar, bad = validated(model, flux, ivar, mask, inflate)
    nspec, npix = flux.shape
    bins = model._bins(flux, ivar, bad)
    # f_s(N_s, N_k, 1) of each bin and pixel with inflate, and None, for 1, without.
    inflation = None
    if inflate:
        # |g| of each pixel. A half-width above npix changes no window, and could
        # overflow int64.
        pixels, half = np.arange(npix), min(nguard, npix)
        window = np.minimum(pixels + half + 1, npix) - np.maximum(pixels - half, 0)
        n_train = np.array(model.n_train)[:, np.newaxis]
        used = np.isin(range(len(model.cov)), bins)[:, np.newaxis]
        inflation = _inflation(n_train, npix - window, used, ("bin", "pixel"))
    weigh(model, nguard, bins)
    center = 0.0 if model.center is None else model.center
    predicted, variance, z = (np.empty((nspec, npix)) for _ in range(3))
    for rows in chunks(nspec, npix):
        X, a = working(model, flux[rows], ivar[rows], bad[rows], bins[rows])
        f, v, zs = predicted[rows], variance[rows], z[rows]
        predict(model, nguard, X, bins[rows], inflation, f, v, zs)
        to_flux(f, a, means(model, bins[rows]), center, out=f)
        v /= np.square(a)[:, np.newaxis]
        zs[bad[rows]] = np.nan
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
    inflate: Boolean = False,
    covariance: Boolean = False,
    rng: np.random.Generator | None = None,
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
    With rng, Repaired.draws holds the draw mu_M + F z_M of x_M, in flux units like the
    mean, with mu_M the mean at M, F F^T = (P_GG^-1)_MM and z_M the normals of the
    spectrum at M. Spectra of one S/N bin with equal G, M and T form a pattern, whose
    spectra in one chunk of max(1, 2**24 // npix) rows share one factorization of
    P_GG. To infill with another model the pixels that a result r replaced and
    ignored, pass replace=r.replaced and ignore=r.ignored.

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
    inflate : bool, optional
        If True, multiply var and the covariance at the replaced pixels by
        f_s(N_s, N_k, 1) of `finite_sample_factor`, with N_s the n_train of the bin of
        the spectrum and N_k = npix - |G| its conditioning pixels, divide chi2 by it,
        and multiply the deviation F z_M of each draw from the mean by its square
        root. For held-out Gaussian spectra, the squared residual of every fixed
        linear combination of the replaced pixels, divided by its variance from the
        covariance, then has the mean 1. Needs a dense, mean-centred, unweighted
        model.
    covariance : bool, optional
        If True, keep the conditional covariance of each pattern for
        `Repaired.covariance`.
    rng : numpy.random.Generator, optional
        If given, the source of Repaired.draws: infill draws
        rng.standard_normal((n, npix)) for each chunk of n rows, in row order, and
        nothing else, and each spectrum uses its row of these normals. Calls on
        consecutive slices with one generator give the draws of one call, to
        rounding. Draws widen the errors of measurements made on them (paper l.550);
        var and covariance give the uncertainty of the replaced pixels. None: no
        draws.
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
        If model is not a Model, inflate or covariance is not a bool, rng is neither
        None nor a numpy.random.Generator, progress is not callable or None, flux, ivar
        or mask has a dtype that `scale` does not allow, or replace or ignore is not of
        a bool dtype.
    ValueError
        If flux, ivar or mask has a shape or a value that `scale` does not allow, flux
        has another npix than model, a spectrum has no good pixel, or replace or
        ignore has another shape than (nspec, npix) or (npix,). If inflate is True and
        model has a constant centre, weights or a `LowRank` covariance, or
        N_s - N_k - 3 <= 0 for a spectrum with a pixel to replace, where
        f_s(N_s, N_k, 1) is not defined; the message names N_s and N_k.
    numpy.linalg.LinAlgError
        If the covariance of a bin fails the check of its precision matrix, as in
        `pixelwise`, where a spectrum of the bin has a pixel to replace and G does not
        cover all its pixels; if P_GG of a pattern is not positive definite; or, for a
        pattern whose G covers all pixels, if cov[b] at T fails that check, or if rng
        is given and cov[b] at M is not positive definite. The message names the bin,
        and for a pattern its first spectrum.

    Notes
    -----
    Cost: npix^3 flops for P at the first use of a bin, npix^2 K for a `LowRank` of
    rank K; then per pattern and chunk
    |G|^3/3 + |M|^3/3 flops, and |M|^3/3 more with covariance=True, and per spectrum
    2 |G| (npix + |G|) flops. An (npix,) mask on spectra with their own bad pixels
    makes one pattern per spectrum, with G the mask and those bad pixels, so a large
    mask then costs |G|^3/3 flops per spectrum. The model keeps P (see `Model`). With
    covariance=True, the result keeps |M|^2 x 8 bytes per pattern and chunk. With rng,
    each spectrum costs npix normals and |M|^2 flops more. Where G covers all pixels,
    it costs 2 |M|^2 flops more instead, and each pattern and chunk |M|^3/3 flops for
    the factor of cov[b] at M, which a `LowRank` forms at |M|^2 K flops.
    """
    start = time.perf_counter()
    inflate = boolean(inflate, "inflate")
    covariance = boolean(covariance, "covariance")
    if rng is not None:
        check_rng(rng)
    check_progress(progress)
    flux, ivar, bad = validated(model, flux, ivar, mask, inflate)
    replace = _pixels(replace, "replace", bad)
    ignore = _pixels(ignore, "ignore", np.broadcast_to(False, bad.shape))
    return infilled(
        model,
        flux,
        ivar,
        bad,
        model._bins(flux, ivar, bad),
        replace,
        ignore,
        inflate=inflate,
        covariance=covariance,
        rng=rng,
        progress=progress,
        stage="infill",
        start=start,
    )


# The layer under the public functions that predict. They validate the spectra with
# validated and find their S/N bins once per call. infilled is infill after its
# checks, on all spectra in flux units, and weigh computes the guard weights of the
# bins of all spectra. The other functions work on one chunk of spectra at a time in
# working units, and name rows in messages by the rows that their caller passes.


def validated(
    model: Model,
    flux: ArrayLike,
    ivar: ArrayLike,
    mask: ArrayLike | None,
    inflate: bool,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[np.bool_]]:
    if not isinstance(model, Model):
        raise TypeError(f"model: expected a Model, got {type(model).__name__}")
    if inflate and not model._derived:
        kind = "dense" if model._dense else "low-rank"
        raise ValueError(
            f"inflate: expected a dense, mean-centred, unweighted model, for which f_s "
            f"is derived, got a {kind} model with center {model.center} and weighted "
            f"{model.weighted}"
        )
    flux, ivar, bad = validate(flux, ivar, mask)
    if flux.shape[1] != model.mean.shape[1]:
        raise ValueError(
            f"flux: expected {model.mean.shape[1]} pixels like model, got shape "
            f"{flux.shape}"
        )
    require_good_pixel(bad)
    return flux, ivar, bad


def infilled(
    model: Model,
    flux: NDArray[Any],
    ivar: NDArray[Any],
    bad: NDArray[np.bool_],
    bins: NDArray[np.intp],
    replace: NDArray[np.bool_],
    ignore: NDArray[np.bool_],
    *,
    inflate: bool,
    covariance: bool,
    rng: np.random.Generator | None,
    progress: Progress | None,
    stage: str,
    start: float,
) -> Repaired:
    # infill after its checks, with replace and ignore of shape (nspec, npix); stage
    # names the caller in progress and in the log, and start is the time of its call.
    nspec, npix = flux.shape
    replaced, ignored = (np.empty((nspec, npix), dtype=bool) for _ in range(2))
    needs = np.empty(nspec, dtype=bool)
    # N_k = npix - |G|, the conditioning pixels of each spectrum.
    n_cond = np.empty(nspec, dtype=np.int64)
    for rows in chunks(nspec, npix):
        M = replace[rows]
        G = M | ignore[rows] | bad[rows]
        replaced[rows] = M
        ignored[rows] = G & ~M
        n_cond[rows] = npix - np.count_nonzero(G, axis=1)
        needs[rows] = M.any(axis=1) & (n_cond[rows] > 0)
    n_train = np.array(model.n_train)[bins]
    # f_s(N_s, N_k, 1) of each spectrum with inflate, defined where a pixel is
    # replaced, and 1 without.
    inflation = np.ones(nspec)
    if inflate:
        inflation = _inflation(n_train, n_cond, replaced.any(axis=1), ("row",))
    # P of the bins that need it, before any spectrum, so that a covariance that fails
    # its check raises at once.
    for b in np.unique(bins[needs]).tolist():
        model._precision(b)
    center = 0.0 if model.center is None else model.center
    result = np.empty((nspec, npix))
    draws = None if rng is None else np.array(flux, dtype=np.float64, order="C")
    var = np.full((nspec, npix), np.inf)
    chi2 = np.zeros(nspec)
    dof = np.empty(nspec, dtype=np.int64)
    refscale = np.empty(nspec)
    patterns: list[NDArray[np.float64]] = []
    pattern = np.empty(nspec, dtype=np.intp)
    count = 0
    for rows in chunks(nspec, npix):
        good = ~bad[rows]
        result[rows] = flux[rows]
        np.divide(1.0, ivar[rows], out=var[rows], where=good, dtype=np.float64)
        # X of scale bit for bit; the input flux at bad pixels is never read.
        X, a = working(model, flux[rows], ivar[rows], bad[rows], bins[rows])
        # Design decision A16: a row of normals for every spectrum, also one without a
        # pixel to replace.
        normals = None if rng is None else rng.standard_normal(X.shape)
        # The deviation of a draw is f^1/2 R_MM^-T z_M, so z carries f^1/2.
        if normals is not None and inflate:
            normals *= np.sqrt(inflation[rows])[:, np.newaxis]
        refscale[rows] = a
        M = replaced[rows]
        T = M & good
        dof[rows] = np.count_nonzero(T, axis=1)
        index = np.arange(rows.start, rows.stop)
        for group, pixels, solution in conditionals(
            model, X, bins[rows], index, M | ignored[rows], M, T, normals, covariance
        ):
            count += 1
            spectra = index[group]
            cov = np.zeros((0, 0)) if covariance else None
            if solution is not None:
                at = np.ix_(spectra, pixels)
                mean = model.mean[bins[spectra[0]], pixels]
                result[at] = to_flux(solution.mean, a[group], mean, center)
                if draws is not None and solution.draws is not None:
                    draws[at] = to_flux(solution.draws, a[group], mean, center)
                # The key of a pattern holds the bin and G, so its spectra share N_s,
                # N_k and their factor.
                factor = inflation[spectra[0]]
                var[at] = solution.var * factor / np.square(a[group])[:, np.newaxis]
                chi2[spectra] = solution.chi2 / factor
                # f C_MM, which covariance(s) divides by a^2 as var.
                cov = None if solution.cov is None else solution.cov * factor
            if cov is not None:
                pattern[spectra] = len(patterns)
                patterns.append(cov)
        if progress is not None:
            progress(stage, rows.stop, nspec)
    # Design section 4.3: the models without a derived factor report 1.
    fs = np.ones(nspec)
    if model._derived:
        fs = sample_factor(n_train, n_cond, dof) / inflation
    for array in (result, var, replaced, ignored, chi2, dof, fs):
        array.flags.writeable = False
    if draws is not None:
        draws.flags.writeable = False
    _logger.debug(
        "%s: %d spectra, %d patterns in %.3f s",
        stage,
        nspec,
        count,
        time.perf_counter() - start,
    )
    covariances = (
        _Covariances(tuple(patterns), pattern, refscale) if covariance else None
    )
    return Repaired(
        result, var, replaced, ignored, chi2, dof, fs, draws, _covariances=covariances
    )


def weigh(model: Model, nguard: int, bins: NDArray[np.intp]) -> None:
    # The guard weights of the bins with spectra, before any spectrum, so that a
    # covariance that fails its check raises at once. A bin without spectra computes
    # no P.
    for b in np.unique(bins).tolist():
        model._guard_weights(b, nguard)


def working(
    model: Model,
    flux: NDArray[Any],
    ivar: NDArray[Any],
    bad: NDArray[np.bool_],
    bins: NDArray[np.intp],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    # The working vectors X of a chunk of spectra, a new C-contiguous array, and their
    # a_s (see Model).
    X = interpolate(flux, bad)
    a = root_mean_ivar(ivar, bad)
    to_working(X, a, 0.0 if model.center is None else model.center, means(model, bins))
    return X, a


def predict(
    model: Model,
    nguard: int,
    X: NDArray[np.float64],
    bins: NDArray[np.intp],
    inflation: NDArray[np.float64] | None,
    predicted: NDArray[np.float64],
    var: NDArray[np.float64] | None,
    z: NDArray[np.float64] | None,
) -> None:
    # The pixelwise prediction of the working vectors X of a chunk of spectra, with
    # inflation (nbin, npix) the factor of the variance, or None for 1: its mean into
    # predicted, its variance into var (working units) and z into z, at every pixel,
    # bad pixels included (design decision A12). var or z None is not computed, and z
    # may be predicted, which then ends as z. The caller computed the guard weights.
    if len(model.cov) == 1:
        factor = None if inflation is None else inflation[0]
        _predict(model, 0, nguard, X, factor, predicted, var, z)
        return
    # The temporaries of a bin end with its iteration.
    for b in np.unique(bins).tolist():
        at = np.flatnonzero(bins == b)
        mean = np.empty((at.size, X.shape[1]))
        part_var, part_z = (
            None if out is None else np.empty_like(mean) for out in (var, z)
        )
        factor = None if inflation is None else inflation[b]
        _predict(model, b, nguard, X[at], factor, mean, part_var, part_z)
        predicted[at] = mean
        if var is not None:
            var[at] = part_var
        if z is not None:
            z[at] = part_z


def zscores(
    model: Model, nguard: int, X: NDArray[np.float64], bins: NDArray[np.intp]
) -> NDArray[np.float64]:
    # z of pixelwise at every pixel of the working vectors X of a chunk of spectra,
    # without inflation. The caller computed the guard weights.
    z = np.empty(X.shape)
    predict(model, nguard, X, bins, None, z, None, z)
    return z


def conditionals(
    model: Model,
    X: NDArray[np.float64],
    bins: NDArray[np.intp],
    rows: NDArray[np.intp],
    G: NDArray[np.bool_],
    M: NDArray[np.bool_],
    T: NDArray[np.bool_],
    normals: NDArray[np.float64] | None,
    covariance: bool,
) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp], Conditional | None]]:
    # The patterns of a chunk of spectra with the working vectors X, their S/N bins
    # and their rows in the caller's arrays, with T in M in G: for each pattern, its
    # spectra in X, the pixels of M, and the solution of solve_pattern, or None where
    # M is empty. normals (nspec, npix) give the draws. The caller computed P of the
    # bins that need it, so only a solve can raise.
    # The bin of each spectrum, one-hot, is part of the key of its pattern.
    one_hot = np.eye(len(model.cov), dtype=bool)
    keys = np.concatenate([G, M, T, one_hot[bins]], axis=1)
    for group in group_rows(keys):
        first = group[0]
        pixels = np.flatnonzero(M[first])
        solution = None
        if pixels.size:
            z = None if normals is None else normals[np.ix_(group, pixels)]
            key = G[first], M[first], T[first]
            b, row = int(bins[first]), int(rows[first])
            solution = _solve(model, b, *key, X[group], z, covariance, row)
        yield group, pixels, solution


def filled(
    model: Model,
    X: NDArray[np.float64],
    bins: NDArray[np.intp],
    rows: NDArray[np.intp],
    G: NDArray[np.bool_],
    M: NDArray[np.bool_],
) -> NDArray[np.float64]:
    # The working vectors X of a chunk of spectra with x_M replaced by its
    # conditional mean given the pixels outside G, as infill computes it.
    result = X.copy()
    T = np.zeros(M.shape, dtype=bool)
    for group, pixels, solution in conditionals(
        model, X, bins, rows, G, M, T, None, False
    ):
        if solution is not None:
            result[np.ix_(group, pixels)] = solution.mean
    return result


def _inflation(
    n_train: NDArray[np.int64],
    n_cond: NDArray[np.int64],
    used: NDArray[np.bool_],
    axes: tuple[str, ...],
) -> NDArray[np.float64]:
    # f_s(N_s, N_k, 1) at each entry of the broadcast arguments, after a check that it
    # is defined where used is True; axes name the dimensions in the message.
    inflation = sample_factor(n_train, n_cond, 1)
    undefined = np.isnan(inflation) & used
    if undefined.any():
        index = np.unravel_index(np.argmax(undefined), undefined.shape)
        n_s = np.broadcast_to(n_train, undefined.shape)[index]
        n_k = np.broadcast_to(n_cond, undefined.shape)[index]
        raise ValueError(
            f"inflate: expected N_s - N_k - 3 > 0, for which f_s(N_s, N_k, 1) is "
            f"defined, got N_s = {n_s} and N_k = {n_k}{located(index, axes)}"
        )
    return inflation


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
    normals: NDArray[np.float64] | None,
    covariance: bool,
    row: int,
) -> Conditional:
    try:
        if G.all():
            # The prior N(0, C): exact from C, and without P, which a singular C does
            # not have. It forms C at M only for the covariance or the draws, and
            # otherwise at T alone (design section 4.2).
            C, pixels = model.cov[b], np.flatnonzero(M)
            cov: NDArray[np.float64] | None = None
            draws: NDArray[np.float64] | None = None
            if covariance or normals is not None:
                CMM = submatrix_of(C, pixels)
                chi2 = prior_chi2(CMM[np.ix_(T[M], T[M])], X[:, T])
                cov = CMM if covariance else None
                draws = None if normals is None else prior_draws(CMM, normals)
            else:
                chi2 = prior_chi2(submatrix_of(C, np.flatnonzero(T)), X[:, T])
            mean = np.zeros((X.shape[0], pixels.size))
            return Conditional(mean, variances_of(C)[pixels], chi2, cov, draws)
        return solve_pattern(model._precision(b), G, M, T, X, normals, covariance)
    except LinAlgError as error:
        raise LinAlgError(
            f"model: cov[{b}] is {error}, for the pattern of row {row}"
        ) from None


def means(model: Model, bins: NDArray[np.intp]) -> NDArray[np.float64]:
    # The mean of the bin of each spectrum, as one row for a model of one bin.
    means: NDArray[np.float64] = (
        model.mean[0] if len(model.cov) == 1 else model.mean[bins]
    )
    return means


def _predict(
    model: Model,
    b: int,
    nguard: int,
    X: NDArray[np.float64],
    inflation: NDArray[np.float64] | None,
    predicted: NDArray[np.float64],
    var: NDArray[np.float64] | None,
    z: NDArray[np.float64] | None,
) -> None:
    W, v = model._guard_weights(b, nguard)
    if inflation is not None:
        v = v * inflation
    np.matmul(X, W, out=predicted)
    if z is not None:
        np.subtract(X, predicted, out=z)
        z /= np.sqrt(v)
    if var is not None:
        var[...] = v
