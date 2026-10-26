import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
from numpy.linalg import LinAlgError
from numpy.typing import ArrayLike, NDArray

from ._linalg import (
    Conditional,
    group_rows,
    guard_means,
    prior_chi2,
    prior_draws,
    solve_factor,
    solve_low_rank,
    solve_mean,
    solve_pattern,
    solve_shared,
)
from ._model import LowRank, Model, sample_factor, submatrix_of, variances_of
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
# Design section 12, (f): the patterns of one M share the factor of P_BB where
# |E| + |D| <= |B| / SHARED and, for n patterns, |B| >= SMALLEST (1 + 2 / n).
SHARED, SMALLEST = 16, 64


@dataclass(frozen=True, eq=False)
class Prediction:
    r"""The prediction of each pixel of a set of spectra, from `pixelwise`.

    The arrays that `pixelwise` returns are read-only.

    Attributes
    ----------
    flux : ndarray, shape (nspec, npix)
        The prediction of each pixel (flux).
    var : ndarray, shape (nspec, npix)
        The conditional variance of each pixel (flux²), which includes the noise.
        With ``inflate=True``, it is multiplied by :math:`f_s(N_s, N_k, 1)`.
    z : ndarray, shape (nspec, npix)
        :math:`(f - \mathrm{flux}) / \sqrt{\mathrm{var}}` at good pixels, with :math:`f`
        the input flux, and NaN at bad pixels (unit-free).
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
    r"""Spectra whose replaced pixels hold their conditional mean, from `infill`.

    `repair` also returns one. The arrays that `infill` and `repair` return are
    read-only. With ``covariance=True``, ``covariance(s)`` gives the conditional
    covariance of the replaced pixels of spectrum ``s``.

    Attributes
    ----------
    flux : ndarray, shape (nspec, npix)
        The conditional mean at the replaced pixels, and the input flux elsewhere
        (flux).
    var : ndarray, shape (nspec, npix)
        The conditional variance at the replaced pixels, which includes the noise.
        With ``inflate=True``, it is multiplied by :math:`f_s(N_s, N_k, 1)`. 1/ivar at
        the other good pixels, and inf at the other bad pixels (flux²).
    replaced : ndarray of bool, shape (nspec, npix)
        The replaced pixels :math:`M`.
    ignored : ndarray of bool, shape (nspec, npix)
        The pixels that are neither conditioned on nor replaced, :math:`G \setminus M`.
    chi2 : ndarray, shape (nspec,)
        :math:`r_T^\mathsf{T} C_{TT}^{-1} r_T` over the replaced pixels :math:`T` that
        are good in the input (unit-free). :math:`r_T` is their input values minus the
        conditional mean. :math:`C_{TT}` is their conditional covariance, which
        ``inflate=True`` multiplies by :math:`f_s(N_s, N_k, 1)`. 0 where ``dof`` is 0.
    dof : ndarray of int64, shape (nspec,)
        The number :math:`|T|` of replaced pixels that are good in the input.
    fs : ndarray, shape (nspec,)
        :math:`f_s(N_s, N_k, \mathrm{dof})` of `finite_sample_factor` (unit-free),
        divided by :math:`f_s(N_s, N_k, 1)` with ``inflate=True``. :math:`N_s` is the
        ``n_train`` of the bin of the spectrum, and :math:`N_k = \mathrm{npix} - |G|`
        the number of its conditioning pixels. NaN where ``dof`` is 0 or
        :math:`N_s - N_k - \mathrm{dof} - 2 \le 0`. 1 for a model with a constant
        centre, weights, a `LowRank` covariance or noise, for which no factor is
        derived. Take the other models, and replace and ignore masks that do not depend
        on the data. Then ``chi2 / (dof * fs)`` has the mean 1 for held-out Gaussian
        spectra, with or without ``inflate``.
    draws : ndarray, shape (nspec, npix), or None
        A draw from the conditional distribution at the replaced pixels, with the
        covariance of `covariance`, and the input flux elsewhere (flux). None if the
        call ran without ``rng``.
    blocks : ndarray of bool, shape (nspec, nblock), or None
        The blocks of ``model.blocks`` that the block pass of `repair` flagged in each
        spectrum. None for a result of `infill`, or of `repair` with
        ``blockwise=False``.
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
        """Return the conditional covariance of the replaced pixels of spectrum ``s``.

        Parameters
        ----------
        s : int
            The row of the spectrum; from 0 to nspec − 1.

        Returns
        -------
        pixels : ndarray of int64, shape (m,)
            The replaced pixels of spectrum ``s``, in increasing order.
        cov : ndarray, shape (m, m)
            Their conditional covariance (flux²), multiplied by
            :math:`f_s(N_s, N_k, 1)` with ``inflate=True``. Its diagonal is
            ``var[s, pixels]``.

        Raises
        ------
        TypeError
            If ``s`` is not an integer.
        ValueError
            If the call ran without ``covariance=True``, or ``s`` is outside 0 to nspec
            − 1.
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
    r"""Predict each pixel from all pixels farther than nguard from it (paper Alg. 3).

    Take pixel :math:`i` of a spectrum in S/N bin :math:`b`, with :math:`x` its working
    vector (see `Model`) and :math:`P` the inverse of ``cov[b]``.
    Let :math:`g = \{j : |j - i| \le \mathrm{nguard}\}`, and let :math:`O` hold the
    pixels outside :math:`g`. Then

    .. math::

        x_i \mid x_O \sim N\bigl(-(P_{gg}^{-1} P_{gO} x_O)_i,
        \ (P_{gg}^{-1})_{ii}\bigr).

    ``Prediction.flux`` and ``Prediction.var`` are this mean and variance in flux units,
    as `unscale` returns them. Bad pixels enter as interpolated values. Their z is NaN,
    and ``var`` omits the error of those values. Next to a run of 60 bad pixels at the
    correlation length 20, the measured :math:`\langle z^2 \rangle` was 0.65 to 2.45,
    depending on the noise level.

    A bad pixel farther than ``nguard`` from :math:`i` takes its value from the nearest
    good pixels. So a good pixel within ``nguard`` of :math:`i` can enter the prediction
    of :math:`i`. Next to a run of bad pixels longer than ``nguard``, :math:`i` itself
    can also enter it. ``nguard >= npix - 1`` gives the prior.

    For a noise-aware model, :math:`x` is in flux units, and :math:`P` is the inverse of
    the covariance :math:`C_s` of the spectrum (see `Model`). pixelwise forms :math:`P`
    for each spectrum. Where :math:`2 \, \mathrm{nguard} + 1 \le \mathrm{npix}`, it
    computes the mean as :math:`x_i - (P_{gg}^{-1} (P x)_g)_i`. This is equal in exact
    arithmetic, but the values in :math:`g` cancel only to rounding.

    Parameters
    ----------
    model : Model
        The model.
    flux, ivar, mask : array_like, shape (nspec, npix)
        As in `scale`, with the npix of ``model``.
    nguard : int, optional
        The half-width of the guard window :math:`g` (pixels); 0 or more.
    inflate : bool, optional
        If True, multiply ``var`` by :math:`f_s(N_s, N_k, 1)` of `finite_sample_factor`.
        :math:`N_s` is the ``n_train`` of the bin of the spectrum.
        :math:`N_k = \mathrm{npix} - |g|` is the number of conditioning pixels of each
        pixel, which is larger near the ends, where :math:`g` is clipped.
        :math:`z^2` then has the mean 1 for held-out Gaussian spectra. It needs a dense,
        mean-centred, unweighted scaled model.
    progress : callable, optional
        ``progress(stage, done, total)``, called after each chunk of spectra, with the
        stage ``"pixelwise"``, the number of spectra done and their total. For a
        noise-aware model, a chunk is one spectrum.

    Returns
    -------
    Prediction
        The prediction, in flux units.

    Raises
    ------
    TypeError
        If ``model`` is not a Model, ``nguard`` is not an integer, ``inflate`` is not a
        bool, or ``progress`` is not callable or None. If ``flux``, ``ivar`` or ``mask``
        has a dtype that `scale` does not allow.
    ValueError
        If ``flux``, ``ivar`` or ``mask`` has a shape or a value that `scale` does not
        allow. If ``flux`` has another npix than ``model``, ``nguard`` is negative, or a
        spectrum has no good pixel. If ``inflate`` is True and ``model`` has a constant
        centre, weights, a `LowRank` covariance or noise. If ``inflate`` is True and
        :math:`N_s - N_k - 3 \le 0` at a pixel of a bin with spectra, where
        :math:`f_s(N_s, N_k, 1)` is not defined; the message names :math:`N_s` and
        :math:`N_k`.
    numpy.linalg.LinAlgError
        If ``nguard < npix - 1`` and the covariance of a bin with spectra fails the
        check of its precision matrix. The check fails if the matrix is not positive
        definite, or if the reciprocal condition number of the covariance, scaled to
        unit diagonal, is below :math:`10^{-13}`. It also fails if :math:`\kappa_W` of a
        `LowRank` covariance is above :math:`10^{13}`. The message names the bin, or for
        a noise-aware model the row whose :math:`C_s` fails.

    Notes
    -----
    Cost:

    - :math:`\mathrm{npix}^3` flops for :math:`P` at the first use of a bin, or
      :math:`\mathrm{npix}^2 K` for a `LowRank` of rank :math:`K`.
    - About :math:`\mathrm{npix} (w^3/3 + 2 \, \mathrm{npix} \, w)` flops for the guard
      weights, at the first use of a bin and after each change of ``nguard``, with
      :math:`w = \min(2 \, \mathrm{nguard} + 1, \mathrm{npix})`.
    - Then :math:`2 \, \mathrm{npix}^2` flops per spectrum.
    - The model keeps :math:`P` and the guard weights (see `Model`).
    - A noise-aware model costs :math:`P` per spectrum, and keeps none.
      Where :math:`2 \, \mathrm{nguard} + 1 \le \mathrm{npix}`, it costs about
      :math:`\mathrm{npix} (w^3/3 + 2 w^2)` flops for its guard windows and
      :math:`2 \, \mathrm{npix} (\mathrm{npix} + w)` for the mean, instead of the guard
      weights.
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
    for rows in spans(model, nspec, npix):
        X, a = working(model, flux[rows], ivar[rows], bad[rows], bins[rows])
        f, v, zs = predicted[rows], variance[rows], z[rows]
        cover = noisy(model, ivar[rows], bad[rows], rows.start)
        predict(cover, nguard, X, bins[rows], inflation, f, v, zs)
        # Free C_s, P_s and the guard solutions before the next spectrum forms its own.
        del cover
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
    r"""Replace pixels by their Gaussian conditional mean given the conditioning pixels.

    Take a spectrum in S/N bin :math:`b`, with :math:`x` its working vector (see
    `Model`) and :math:`P` the inverse of ``cov[b]``. Let :math:`G` be the pixels of
    ``replace | ignore | bad``, with the bad pixels ``(mask != 0) | (ivar == 0)``.
    The other pixels :math:`O` are the conditioning pixels. Then

    .. math::

        x_G \mid x_O \sim N(-P_{GG}^{-1} P_{GO} x_O, \ P_{GG}^{-1}).

    ``Repaired.flux`` is this mean at the replaced pixels :math:`M` = ``replace``, in
    flux units as `unscale` returns it, and the input flux elsewhere. The pixels of
    :math:`G` outside :math:`M` keep their input value, and the result reports them as
    ignored. A pixel in both ``replace`` and ``ignore`` is replaced. Values in :math:`G`
    never enter the mean, so an outlier there cannot change it. ``chi2`` compares the
    input values at :math:`T = M \setminus \mathrm{bad}`, the replaced pixels that are
    good in the input, with their conditional distribution.

    Where :math:`G` covers all pixels, the result is the prior
    :math:`N(0, \mathrm{cov}[b])`, which infill computes from ``cov[b]`` without
    :math:`P`. For a noise-aware model, :math:`x` is in flux units, and the :math:`C_s`
    of the spectrum (see `Model`) replaces ``cov[b]``. Each spectrum then has its own
    :math:`P` and pattern.

    With ``rng``, ``Repaired.draws`` holds the draw :math:`\mu_M + F z_M` of
    :math:`x_M`, in flux units like the mean. Here :math:`\mu_M` is the mean at
    :math:`M`, :math:`F F^\mathsf{T} = (P_{GG}^{-1})_{MM}`, and :math:`z_M` holds the
    normals of the spectrum at :math:`M`.

    Spectra of one S/N bin with equal :math:`G`, :math:`M` and :math:`T` form a pattern.
    The spectra of a pattern in one chunk of ``max(1, 2**24 // npix)`` rows share one
    factorization of :math:`P_{GG}`. To infill with another model the pixels that a
    result ``r`` replaced and ignored, pass ``replace=r.replaced`` and
    ``ignore=r.ignored``.

    Parameters
    ----------
    model : Model
        The model.
    flux, ivar, mask : array_like, shape (nspec, npix)
        As in `scale`, with the npix of ``model``.
    replace : array_like of bool, shape (nspec, npix) or (npix,), optional
        The pixels to replace, :math:`M`. An (npix,) mask applies to every spectrum, for
        example to predict a range of pixels. None: the bad pixels.
    ignore : array_like of bool, shape (nspec, npix) or (npix,), optional
        Pixels that are neither conditioned on nor replaced, unless they are in
        ``replace``, such as a guard around a range of pixels. None: no pixels.
    inflate : bool, optional
        If True, multiply ``var`` and the covariance at the replaced pixels by
        :math:`f_s(N_s, N_k, 1)` of `finite_sample_factor`. :math:`N_s` is the
        ``n_train`` of the bin of the spectrum, and :math:`N_k = \mathrm{npix} - |G|`
        the number of its conditioning pixels. Also divide ``chi2`` by the factor, and
        multiply the deviation :math:`F z_M` of each draw from the mean by its square
        root. Then, for held-out Gaussian spectra, take any fixed linear combination of
        the replaced pixels. Its squared residual, divided by its variance from the
        covariance, has the mean 1. It needs a dense, mean-centred, unweighted scaled
        model.
    covariance : bool, optional
        If True, keep the conditional covariance of each pattern for
        `Repaired.covariance`.
    rng : numpy.random.Generator, optional
        If given, the source of ``Repaired.draws``. infill draws
        ``rng.standard_normal((n, npix))`` for each chunk of :math:`n` rows, in row
        order, and nothing else. Each spectrum uses its row of these normals. Calls on
        consecutive slices with one generator give the draws of one call, to rounding.
        Draws widen the errors of measurements made on them; ``var`` and ``covariance``
        give the uncertainty of the replaced pixels. None: no draws.
    progress : callable, optional
        ``progress(stage, done, total)``, called after each chunk of spectra, with the
        stage ``"infill"``, the number of spectra done and their total. For a
        noise-aware model, a chunk is one spectrum.

    Returns
    -------
    Repaired
        The infilled spectra, in flux units.

    Raises
    ------
    TypeError
        If ``model`` is not a Model, or ``inflate`` or ``covariance`` is not a bool.
        If ``rng`` is neither None nor a numpy.random.Generator, or ``progress`` is not
        callable or None. If ``flux``, ``ivar`` or ``mask`` has a dtype that `scale`
        does not allow, or ``replace`` or ``ignore`` is not of a bool dtype.
    ValueError
        If ``flux``, ``ivar`` or ``mask`` has a shape or a value that `scale` does not
        allow. If ``flux`` has another npix than ``model``, or a spectrum has no good
        pixel. If ``replace`` or ``ignore`` has another shape than (nspec, npix) or
        (npix,). If ``inflate`` is True and ``model`` has a constant centre, weights, a
        `LowRank` covariance or noise. If ``inflate`` is True and
        :math:`N_s - N_k - 3 \le 0` for a spectrum with a pixel to replace, where
        :math:`f_s(N_s, N_k, 1)` is not defined; the message names :math:`N_s` and
        :math:`N_k`.
    numpy.linalg.LinAlgError
        If the covariance of a bin fails the check of its precision matrix, as in
        `pixelwise`, where a spectrum of the bin has a pixel to replace and :math:`G`
        does not cover all its pixels. If :math:`P_{GG}` of a pattern is not positive
        definite. For a pattern whose :math:`G` covers all pixels, if ``cov[b]`` at
        :math:`T` fails that check, or if ``rng`` is given and ``cov[b]`` at :math:`M`
        is not positive definite. The message names the bin, and for a pattern its first
        spectrum.

        For a noise-aware model, :math:`C_s` replaces ``cov[b]``. A :math:`C_s` that
        fails the check of its precision matrix raises with its row in the message.
        A `LowRank` ``cov[0]`` forms no precision matrix. Its :math:`C_s` raises where
        :math:`\kappa_K > 10^{13}`, with

        .. math::

            \kappa_K = 1 + \lambda_{\max}(\Lambda^{1/2} V^\mathsf{T} D_s^{-1} V
            \Lambda^{1/2}),

        :math:`\Lambda = \mathrm{diag}(\mathrm{values})`, and
        :math:`D_s = \mathrm{diag}(\mathrm{diag} + \nu_s)` of :math:`C_s`. It also
        raises where the factor of
        :math:`I + \Lambda^{1/2} V_O^\mathsf{T} D_{s,O}^{-1} V_O \Lambda^{1/2}` fails,
        or the factor of its conditional covariance at :math:`M` (at :math:`T` without
        ``rng``). The message names its row. :math:`\kappa_K` accepts every :math:`C_s`
        that the :math:`\kappa_W` check of `LowRank` accepts. It also accepts some with
        :math:`\kappa_W` up to :math:`\mathrm{npix} \, \kappa_K`, which `pixelwise` and
        `repair` still reject.

    Notes
    -----
    Cost:

    - :math:`\mathrm{npix}^3` flops for :math:`P` at the first use of a bin, or
      :math:`\mathrm{npix}^2 K` for a `LowRank` of rank :math:`K`.
    - Per pattern and chunk, :math:`|G|^3/3 + |M|^3/3` flops, and :math:`|M|^3/3` more
      with ``covariance=True``.
    - Per spectrum, :math:`2 |G| (\mathrm{npix} + |G|)` flops, or
      :math:`2 |G| \, \mathrm{npix}` where :math:`G` is one run.
    - Without ``rng``, some patterns share one factor: :math:`n \ge 2` patterns of a
      chunk with one bin and one :math:`M`, and :math:`G` short of all pixels.
      Let :math:`B` be the pixels in the :math:`G` of all of them. For each pattern, let
      :math:`E` be the pixels of its :math:`G` outside :math:`B`. Let :math:`D` be the
      pixels of :math:`M` in the :math:`T` of another of them but not in its own.
      The patterns share the factor under three conditions. Each has
      :math:`|E| + |D| \le |B|/16`, and :math:`|B| \ge 64 (1 + 2/n)`. The shared factor
      costs fewer flops than their own factors, or than the mean from :math:`C_{OO}`.
    - These patterns together cost :math:`|B|^3/3 + |M|^3/3` flops, and :math:`|M|^3/3`
      more with ``covariance=True``. Let :math:`U` be the pixels in the :math:`T` of any
      of them. Each of these patterns costs
      :math:`|B|^2 |E| + |M|^2 |E| + |U| (|D| + |E|)^2` flops, and :math:`|M|^2 |E|`
      more with ``covariance=True``. Each of their spectra costs
      :math:`2|G|\,\mathrm{npix} + |B|^2 + |M|^2 + 2|U|^2 + 4(|U| + |E|)(|D| + |E|)`
      flops. An (npix,) mask on spectra with a few bad pixels of their own is such a
      case.
    - The model keeps :math:`P` (see `Model`).
    - With ``covariance=True``, the result keeps :math:`|M|^2 \times 8` bytes per
      pattern and chunk.
    - With ``rng``, each spectrum costs npix normals and :math:`|M|^2` flops more.
      Where :math:`G` covers all pixels, it costs :math:`2 |M|^2` flops more instead.
      Each pattern and chunk then also costs :math:`|M|^3/3` flops for the factor of
      ``cov[b]`` at :math:`M`, which a `LowRank` forms at :math:`|M|^2 K` flops.
    - A noise-aware model forms :math:`C_s` for each spectrum with a pixel to replace,
      and every spectrum is its own pattern. Where :math:`G` is also short of all
      pixels, a dense ``cov[0]`` forms no :math:`P_s`. Per spectrum, it costs
      :math:`\mathrm{npix}^3/3` flops for the Cholesky factor of :math:`C_s`, before the
      pattern costs above. From this factor, :math:`P_{GG}` and :math:`P_{G,:} x_0` cost
      about :math:`(\mathrm{npix} - g_0)^2 |G| + \mathrm{npix} |G|^2` flops, with
      :math:`g_0` the first pixel of :math:`G`.
    - A `LowRank` ``cov[0]`` of rank :math:`K` forms no :math:`P_s`. Per spectrum, it
      costs :math:`\mathrm{npix} K^2` flops for its :math:`K \times K` system, and
      :math:`|M| K^2` for the mean and ``var``. ``chi2`` costs
      :math:`|T|^2 K + |T|^3/3`. With ``rng``, ``chi2`` and the draws cost
      :math:`|M|^2 K + |M|^3/3` instead.
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
        kind = "noise" if model.noise else "dense" if model._dense else "low-rank"
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
    # its check raises at once. A noise model factors C_s per spectrum.
    if not model.noise:
        for b in np.unique(bins[needs]).tolist():
            model._precision(b)
    filling = Filling(
        model,
        flux,
        ivar,
        bad,
        bins,
        inflation=inflation,
        inflate=inflate,
        covariance=covariance,
        rng=rng,
    )
    for rows in spans(model, nspec, npix):
        X, a = working(model, flux[rows], ivar[rows], bad[rows], bins[rows])
        M = replaced[rows]
        # A spectrum without a pixel to replace reads no covariance, so a noise model
        # forms no C_s for it.
        cover = noisy(model, ivar[rows], bad[rows], rows.start) if M.any() else model
        filling.fill(cover, rows, X, a, M, ignored[rows])
        del cover
        if progress is not None:
            progress(stage, rows.stop, nspec)
    # Design section 4.3: the models without a derived factor report 1.
    fs = np.ones(nspec)
    if model._derived:
        fs = sample_factor(n_train, n_cond, filling.dof) / inflation
    return filling.done(replaced, ignored, fs, stage, start)


class Filling:
    # The outputs of infill of nspec spectra in flux units, which fill writes chunk
    # by chunk and done returns as a Repaired, so that infilled and the noise path of
    # repair share them. inflation holds f_s(N_s, N_k, 1) of each spectrum, or 1,
    # and inflate says whether the draws carry it.

    def __init__(
        self,
        model: Model,
        flux: NDArray[Any],
        ivar: NDArray[Any],
        bad: NDArray[np.bool_],
        bins: NDArray[np.intp],
        *,
        inflation: NDArray[np.float64],
        inflate: bool,
        covariance: bool,
        rng: np.random.Generator | None,
    ) -> None:
        nspec, npix = flux.shape
        self.model, self.flux, self.ivar, self.bad, self.bins = (
            model,
            flux,
            ivar,
            bad,
            bins,
        )
        self.inflation, self.inflate = inflation, inflate
        self.covariance, self.rng = covariance, rng
        self.result = np.empty((nspec, npix))
        self.draws = None if rng is None else np.array(flux, np.float64, order="C")
        self.var = np.full((nspec, npix), np.inf)
        self.chi2 = np.zeros(nspec)
        self.dof = np.empty(nspec, dtype=np.int64)
        self.refscale = np.empty(nspec)
        self.patterns: list[NDArray[np.float64]] = []
        self.pattern = np.empty(nspec, dtype=np.intp)
        self.count = 0

    def fill(
        self,
        cover: Model,
        rows: slice,
        X: NDArray[np.float64],
        a: NDArray[np.float64],
        M: NDArray[np.bool_],
        ignored: NDArray[np.bool_],
    ) -> None:
        # The spectra rows, with their working vectors X, their a_s and their
        # replaced and ignored pixels, under the covariance of cover from noisy.
        model = self.model
        good = ~self.bad[rows]
        self.result[rows] = self.flux[rows]
        np.divide(
            1.0, self.ivar[rows], out=self.var[rows], where=good, dtype=np.float64
        )
        # Design decision A16: a row of normals for every spectrum, also one without a
        # pixel to replace.
        normals = None if self.rng is None else self.rng.standard_normal(X.shape)
        # The deviation of a draw is f^1/2 R_MM^-T z_M, so z carries f^1/2.
        if normals is not None and self.inflate:
            normals *= np.sqrt(self.inflation[rows])[:, np.newaxis]
        self.refscale[rows] = a
        T = M & good
        self.dof[rows] = np.count_nonzero(T, axis=1)
        index = np.arange(rows.start, rows.stop)
        center = 0.0 if model.center is None else model.center
        for group, pixels, solution in conditionals(
            cover,
            X,
            self.bins[rows],
            index,
            M | ignored,
            M,
            T,
            normals,
            covariance=self.covariance,
            variance=True,
        ):
            self.count += 1
            spectra = index[group]
            cov = np.zeros((0, 0)) if self.covariance else None
            if solution is not None:
                at = np.ix_(spectra, pixels)
                mean = model.mean[self.bins[spectra[0]], pixels]
                self.result[at] = to_flux(solution.mean, a[group], mean, center)
                if self.draws is not None and solution.draws is not None:
                    self.draws[at] = to_flux(solution.draws, a[group], mean, center)
                # The key of a pattern holds the bin and G, so its spectra share N_s,
                # N_k and their factor.
                factor = self.inflation[spectra[0]]
                scale = np.square(a[group])[:, np.newaxis]
                self.var[at] = solution.var * factor / scale
                self.chi2[spectra] = solution.chi2 / factor
                # f C_MM, which covariance(s) divides by a^2 as var.
                cov = None if solution.cov is None else solution.cov * factor
            if cov is not None:
                self.pattern[spectra] = len(self.patterns)
                self.patterns.append(cov)

    def done(
        self,
        replaced: NDArray[np.bool_],
        ignored: NDArray[np.bool_],
        fs: NDArray[np.float64],
        stage: str,
        start: float,
    ) -> Repaired:
        # The result, read-only; stage names the caller in the log, and start is the
        # time of its call.
        arrays = (self.result, self.var, replaced, ignored, self.chi2, self.dof, fs)
        for array in arrays:
            array.flags.writeable = False
        if self.draws is not None:
            self.draws.flags.writeable = False
        _logger.debug(
            "%s: %d spectra, %d patterns in %.3f s",
            stage,
            len(fs),
            self.count,
            time.perf_counter() - start,
        )
        covariances = None
        if self.covariance:
            covariances = _Covariances(
                tuple(self.patterns), self.pattern, self.refscale
            )
        return Repaired(*arrays, self.draws, _covariances=covariances)


def weigh(model: Model, nguard: int, bins: NDArray[np.intp]) -> None:
    # The guard weights of the bins with spectra, before any spectrum, so that a
    # covariance that fails its check raises at once. A bin without spectra computes
    # no P, and a noise model forms its guard solutions per spectrum.
    if model.noise:
        return
    for b in np.unique(bins).tolist():
        model._guard_weights(b, nguard)


def spans(model: Model, nspec: int, npix: int) -> Iterator[slice]:
    # The chunks of spectra of the functions that predict: max(1, 2**24 // npix)
    # spectra, or one for a noise model, whose factor of C_s per spectrum costs more
    # than a chunk of another model, so that progress comes after each spectrum.
    if model.noise:
        return (slice(s, s + 1) for s in range(nspec))
    return chunks(nspec, npix)


def noisy(model: Model, ivar: NDArray[Any], bad: NDArray[np.bool_], row: int) -> Model:
    # The model whose covariance conditions a chunk of spectra, of first row row:
    # model, or for a noise model, whose chunks hold one spectrum, the model of
    # C_s = cov[0] + diag(nu_s), with nu_si = 1/ivar_si at good pixels and 1/a_s^2 at
    # bad ones (design section 4.1). It forms P_s, or the Cholesky factor of C_s, at
    # its first use and keeps it for every operation of the call on the spectrum
    # (decision A22).
    if not model.noise:
        return model
    a = root_mean_ivar(ivar, bad)[0]
    nu = np.full(bad.shape[1], 1 / a**2)
    np.divide(1.0, ivar[0], out=nu, where=~bad[0], dtype=np.float64)
    return model._with_noise(nu, f"model: cov[0] + diag(nu) of row {row}")


def working(
    model: Model,
    flux: NDArray[Any],
    ivar: NDArray[Any],
    bad: NDArray[np.bool_],
    bins: NDArray[np.intp],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    # The working vectors X of a chunk of spectra, a new C-contiguous array, and their
    # scale of the working units: a_s, or 1 for a noise model (see Model).
    X = interpolate(flux, bad)
    a = np.ones(len(flux)) if model.noise else root_mean_ivar(ivar, bad)
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
    Y: NDArray[np.float64] | None = None,
) -> None:
    # The pixelwise prediction of the working vectors X of a chunk of spectra, with
    # inflation (nbin, npix) the factor of the variance, or None for 1: its mean into
    # predicted, its variance into var (working units) and the z of Y, X if None,
    # into z, at every pixel, bad pixels included (design decision A12). var or z None
    # is not computed, and z may be predicted, which then ends as z. The caller
    # computed the guard weights of a model without noise.
    Y = X if Y is None else Y
    if len(model.cov) == 1:
        factor = None if inflation is None else inflation[0]
        _predict(model, 0, nguard, X, factor, predicted, var, z, Y)
        return
    # The temporaries of a bin end with its iteration.
    for b in np.unique(bins).tolist():
        at = np.flatnonzero(bins == b)
        mean = np.empty((at.size, X.shape[1]))
        part_var, part_z = (
            None if out is None else np.empty_like(mean) for out in (var, z)
        )
        factor = None if inflation is None else inflation[b]
        _predict(model, b, nguard, X[at], factor, mean, part_var, part_z, Y[at])
        predicted[at] = mean
        if var is not None:
            var[at] = part_var
        if z is not None:
            z[at] = part_z


def zscores(
    model: Model,
    nguard: int,
    X: NDArray[np.float64],
    bins: NDArray[np.intp],
    Y: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    # z of the working vectors Y, X if None, at every pixel against the pixelwise
    # prediction from the working vectors X of the same chunk of spectra, without
    # inflation. The caller computed the guard weights of a model
    # without noise.
    z = np.empty(X.shape)
    predict(model, nguard, X, bins, None, z, None, z, Y)
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
    variance: bool,
) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp], Conditional | None]]:
    # The patterns of a chunk of spectra with the working vectors X, their S/N bins
    # and their rows in the caller's arrays, with T in M in G: for each pattern, its
    # spectra in X, the pixels of M, and their conditional (see _solve), or None where
    # M is empty. normals (nspec, npix) give the draws. variance=False skips the var
    # of solve_pattern, which is then NaN; the prior and the low-rank route keep var.
    # The caller computed P of the bins that need it, so only a solve can raise, or
    # for a noise model P_s or the factor of C_s.
    # The bin of each spectrum, one-hot, is part of the key of its pattern. The bin and
    # M lead the key, so that the patterns of one bin and one M are adjacent in the
    # order of group_rows, as _shared needs.
    one_hot = np.eye(len(model.cov), dtype=bool)
    keys = np.concatenate([one_hot[bins], M, G, T], axis=1)
    patterns = group_rows(keys)
    first = np.array([group[0] for group in patterns], dtype=np.intp)
    g, npix = np.count_nonzero(G, axis=1)[first], G.shape[1]
    # The mean alone, of a dense model without noise, from C_OO at |O|^3/3 flops
    # instead of |G|^3/3 where npix/2 < |G| < npix (design section 12, (j)).
    from_cov = (2 * g > npix) & (g < npix)
    if variance or normals is not None or model._label is not None:
        from_cov[:] = False
    elif from_cov.any():
        dense = np.array([not isinstance(C, LowRank) for C in model.cov])
        from_cov &= dense[bins[first]] & ~T.any(axis=1)[first]
    # The draws keep the factor of solve_pattern, and a noise model its own route.
    shared = {}
    if normals is None and model._label is None and len(patterns) > 1:
        shared = _shared(
            model, X, bins, G, M, T, patterns, first, g, from_cov, covariance, variance
        )
    for i, group in enumerate(patterns):
        first_row = group[0]
        pixels = np.flatnonzero(M[first_row])
        solution = shared.get(i)
        if pixels.size and solution is None:
            z = None if normals is None else normals[np.ix_(group, pixels)]
            key = G[first_row], M[first_row], T[first_row]
            b, row = int(bins[first_row]), int(rows[first_row])
            solution = _solve(
                model, b, *key, X[group], z, covariance, variance, row, from_cov[i]
            )
        yield group, pixels, solution


def _shared(
    model: Model,
    X: NDArray[np.float64],
    bins: NDArray[np.intp],
    G: NDArray[np.bool_],
    M: NDArray[np.bool_],
    T: NDArray[np.bool_],
    patterns: list[NDArray[np.intp]],
    first: NDArray[np.intp],
    g: NDArray[np.intp],
    from_cov: NDArray[np.bool_],
    covariance: bool,
    variance: bool,
) -> dict[int, Conditional]:
    # The solutions of solve_shared, by position in patterns, for the runs of adjacent
    # patterns with one bin and one M, with first their first rows, g their |G| and
    # from_cov where they take the mean from C_OO (design section 12, (f)). A pattern
    # joins where its E = G \ B and D = U \ T hold together at most |B| / SHARED pixels,
    # with B the pixels in the G of all patterns of the run and U the pixels in the T of
    # any of them. The n patterns that join share the factor where |B| >= SMALLEST (1 +
    # 2 / n), below which the overhead of the calls outweighs the flops saved at npix
    # 1000 to 2400 (#109), and where it costs fewer flops than their own routes.
    solutions: dict[int, Conditional] = {}
    npix = G.shape[1]
    # B is in G, so a pattern with |G| < SMALLEST cannot join, and the rows of the
    # others are gathered alone.
    positions = np.flatnonzero((g >= SMALLEST) & (g < npix) & M.any(axis=1)[first])
    if positions.size < 2:
        return solutions
    first = first[positions]
    Mf, bf = M[first], bins[first]
    starts = np.flatnonzero(
        np.r_[True, (bf[1:] != bf[:-1]) | (Mf[1:] != Mf[:-1]).any(axis=1)]
    )
    for start, stop in zip(starts, np.r_[starts[1:], first.size]):
        # |B| is at most the least |G| of the run, which bounds the test below.
        run = np.arange(start, stop)
        least = int(np.min(g[positions[run]]))
        if run.size < 2 or least * run.size < SMALLEST * (run.size + 2):
            continue
        Gc, Tc = G[first[run]], T[first[run]]
        B, U = Gc.all(axis=0), Tc.any(axis=0)
        size = int(np.count_nonzero(B))
        E, D = np.count_nonzero(Gc & ~B, axis=1), np.count_nonzero(U & ~Tc, axis=1)
        keep = SHARED * (E + D) <= size
        run, Gc, Tc, E, D = run[keep], Gc[keep], Tc[keep], E[keep], D[keep]
        n = run.size
        if n < 2 or size * n < SMALLEST * (n + 2):
            continue
        cluster = positions[run]
        # Three times the flops of the factors: with the variances, those of R_MM
        # and, for each pattern, the solve of W_M by R_MM. For each spectrum, the
        # shared route also projects its chi2 residual and multiplies by the E rows of
        # P, and its own route solves by the factor of P_GG instead of P_BB.
        m, u = int(np.count_nonzero(Mf[start])), int(np.count_nonzero(U))
        inverse, update = (m**3, size**2 + m**2) if variance else (0, size**2)
        own = np.where(
            from_cov[cluster], (npix - g[cluster]) ** 3, g[cluster] ** 3 + inverse
        )
        spectra = np.array([patterns[i].size for i in cluster])
        shared = 12 * (u + E) * (D + E) + 3 * E * (2 * npix + 4 * size)
        own += 3 * spectra * np.where(from_cov[cluster], 0, g[cluster] ** 2 - size**2)
        cost = size**3 + inverse + 3 * update * E.sum() + np.sum(spectra * shared)
        if cost >= own.sum():
            continue
        groups = [patterns[i] for i in cluster]
        stops = np.cumsum([group.size for group in groups])
        local = [slice(stop - group.size, stop) for group, stop in zip(groups, stops)]
        P = model._precision(int(bf[start]))
        Xc = X[np.concatenate(groups)]
        try:
            results = solve_shared(
                P, Gc, Mf[start], Tc, Xc, local, covariance, variance
            )
        except LinAlgError:
            # solve_pattern then raises for the first pattern that fails, as without
            # the shared factor.
            continue
        solutions.update(zip(cluster.tolist(), results))
    return solutions


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
        model, X, bins, rows, G, M, T, None, covariance=False, variance=False
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
    variance: bool,
    row: int,
    from_cov: bool,
) -> Conditional:
    C, prior = model.cov[b], G.all()
    # The model of one spectrum of a noise model, which has a label, conditions C_s
    # without P_s: a LowRank from V and d + nu_s, and a dense one from its Cholesky
    # factor unless the call formed P_s (design section 12, (b) and (e)).
    noise = model._label is not None and not prior
    # Outside the try, as the messages of P and of the factor name the covariance. P
    # also checks C for the mean from C_OO.
    P = L = None
    if noise and not isinstance(C, LowRank):
        P, L = model._conditioner(b, C)
    elif not (prior or noise):
        P = model._precision(b)
    try:
        # The mean alone from C_OO (see conditionals); isinstance narrows C.
        if from_cov and isinstance(C, np.ndarray):
            mean = solve_mean(C, G, M, X)
            return Conditional(
                mean, np.full(mean.shape[1], np.nan), np.zeros(len(mean)), None, None
            )
        if isinstance(C, LowRank) and noise:
            V, values, d = C.vectors, C.values, C.diag
            return solve_low_rank(V, values, d, G, M, T, X, normals, covariance)
        if L is not None:
            return solve_factor(
                L, G, M, T, X, normals, covariance=covariance, variance=variance
            )
        if P is None:
            # The prior N(0, C): exact from C, and without P, which a singular C does
            # not have. It forms C at M only for the covariance or the draws, and
            # otherwise at T alone (design section 4.2).
            pixels = np.flatnonzero(M)
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
        return solve_pattern(
            P, G, M, T, X, normals, covariance=covariance, variance=variance
        )
    except LinAlgError as error:
        raise LinAlgError(
            f"{model._name(b)} is {error}, for the pattern of row {row}"
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
    Y: NDArray[np.float64],
) -> None:
    if model._label is None or 2 * nguard + 1 > X.shape[1]:
        W, v = model._guard_weights(b, nguard)
        np.matmul(X, W, out=predicted)
    else:
        # The model of one spectrum of a noise model, whose _label is set: its guard
        # solutions serve this spectrum alone, so it skips W (design section 12,
        # item (a)). V has 2 nguard + 1 columns, so the route needs no more memory
        # than W where they fit in npix.
        V, v = model._guard_solutions(b, nguard)
        predicted[...] = guard_means(model._precision(b), V, X)
    if inflation is not None:
        v = v * inflation
    if z is not None:
        np.subtract(Y, predicted, out=z)
        z /= np.sqrt(v)
    if var is not None:
        var[...] = v
