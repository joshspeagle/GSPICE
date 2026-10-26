import logging
import math
import time
from fractions import Fraction
from typing import Any, NamedTuple

import numpy as np
from numpy.linalg import LinAlgError
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import lapack

from ._linalg import factor_pattern, factor_prior
from ._model import NGUARD_MAX, Blocks, Model, submatrix_of
from ._predict import filled, noisy, spans, validated, working
from ._spectra import (
    Integer,
    Progress,
    Real,
    check_progress,
    integer,
    real,
    to_flux,
    to_working,
)

_logger = logging.getLogger("gspice")


def fit_blocks(
    model: Model,
    flux: ArrayLike,
    ivar: ArrayLike,
    mask: ArrayLike | None = None,
    *,
    width: Integer = 150,
    stride: Integer = 75,
    nguard: Integer = 20,
    percentile: Real = 99.9,
    progress: Progress | None = None,
) -> Model:
    r"""Return the model with block thresholds from clean spectra.

    Block :math:`j` holds the ``width`` pixels from ``starts[j]``. The starts are 0,
    ``stride``, 2 ``stride`` and so on, up to npix − ``width``. If the last of them is
    smaller than npix − ``width``, the start npix − ``width`` follows. The defaults
    give 31 blocks at npix 2400. fit_blocks first replaces the bad pixels by their
    conditional mean, as `infill` does. It does not detect outliers by z.

    Take a spectrum in S/N bin :math:`b`, with :math:`x` the working vector of its
    infilled flux (see `Model`) and :math:`P` the inverse of ``cov[b]``. Let :math:`A`
    be a block, :math:`G` the pixels within ``nguard`` of :math:`A`, clipped at the
    ends, and :math:`O` the pixels outside :math:`G`. Then

    .. math::

        \chi^2_A = \frac{r_A^\mathsf{T} C_{AA}^{-1} r_A}{|A|}, \qquad r_A = x_A - \mu_A,

    where :math:`\mu_A` and :math:`C_{AA} = (P_{GG}^{-1})_{AA}` are the conditional mean
    and covariance of :math:`x_A` given :math:`x_O`. The guard pixels
    :math:`G \setminus A` are thus marginalized. Where :math:`G` covers all pixels,
    :math:`\mu_A` and :math:`C_{AA}` are those of the prior, 0 and ``cov[b]`` at
    :math:`A`, computed without :math:`P`.

    ``thresholds[b, j]`` is ``np.percentile`` (linear) of :math:`\chi^2_A` of block
    :math:`j`, at :math:`p` = ``percentile``, over the :math:`m` spectra of bin
    :math:`b`. A block without an artifact then exceeds its threshold with the
    probability

    .. math::

        \frac{m (1 - p/100) + p/100}{m + 1}

    if :math:`(m - 1) p/100` is an integer. Otherwise the probability is at most this
    value, for a density that falls beyond the threshold. For :math:`p = 99.9`, it is
    0.20% at :math:`m = 1000` and 0.12% at :math:`m = 5000`. `repair` also infills the
    pixels that z flags, which lowers this rate. Use clean held-out spectra.

    For a noise-aware model, :math:`x` is in flux units. The covariance :math:`C_s` of
    each spectrum (see `Model`) then replaces ``cov[b]`` in its infill and its blocks.

    Parameters
    ----------
    model : Model
        The model.
    flux, ivar, mask : array_like, shape (nspec, npix)
        As in `scale`, with the npix of ``model``. Clean spectra, with at least
        :math:`m` spectra in each bin such that :math:`m (100 - p) \ge 100`. The check
        reads :math:`p` as the decimal ``Fraction(str(p))``: 1000 spectra at
        :math:`p = 99.9`.
    width : int, optional
        The number of pixels per block, :math:`|A|`; from 1 to npix.
    stride : int, optional
        The distance between the starts of consecutive blocks (pixels); 1 or more.
    nguard : int, optional
        The half-width of the guard window around each block (pixels); from 0 to
        :math:`2^{63} - 1`, as for `Blocks`.
    percentile : float, optional
        The percentile :math:`p` of the thresholds; :math:`0 < p < 100`.
    progress : callable, optional
        ``progress(stage, done, total)``, called after each chunk of spectra, with the
        stage ``"fit_blocks"``, the number of spectra done and their total. For a
        noise-aware model, a chunk is one spectrum.

    Returns
    -------
    Model
        A new model, equal to ``model`` except for ``blocks``: a `Blocks` that holds the
        starts, width, nguard and thresholds.

    Raises
    ------
    TypeError
        If ``model`` is not a Model, or ``width``, ``stride`` or ``nguard`` is not an
        integer. If ``percentile`` is not a real number, or ``progress`` is not callable
        or None. If ``flux``, ``ivar`` or ``mask`` has a dtype that `scale` does not
        allow.
    ValueError
        If ``flux``, ``ivar`` or ``mask`` has a shape or a value that `scale` does not
        allow. If ``flux`` has another npix than ``model``, or a spectrum has no good
        pixel. If ``width``, ``stride``, ``nguard`` or ``percentile`` is outside its
        range. If a bin has fewer spectra than :math:`p` needs; the message names the
        bin and its count. If a threshold is not finite or is 0, as for spectra equal to
        their conditional mean.
    numpy.linalg.LinAlgError
        If the covariance of a bin fails the check of its precision matrix, as in
        `pixelwise`, and the :math:`G` of a block does not cover all pixels.
        If :math:`P_{GG}` of a block is not positive definite. If ``cov[b]`` at
        :math:`T = A` fails that check, for a block whose :math:`G` covers all pixels.
        Also as in `infill`, for the bad pixels. The message names the bin, and the
        block or the row of a pattern. For a noise-aware model, as in `infill`, with
        :math:`C_s` in place of ``cov[b]``.

    Notes
    -----
    Cost:

    - :math:`\mathrm{npix}^3` flops for :math:`P` at the first use of a bin, then
      :math:`|G|^3/3` flops per bin and block.
    - About :math:`2\,\mathrm{npix}^2 + \sum_A |G|^2` flops per spectrum, plus the
      `infill` of the bad pixels, of the mean only.
    - The factors of all bins take :math:`\mathrm{nbin} \sum_A |G|^2 \times 8` bytes
      during the scan: 8.8 MB per bin at npix 2400 with the defaults, and 649 MB at
      ``stride=1``.
    - Beyond the inputs, and their float64 copies if they are long double, fit_blocks
      holds a bool array of the input shape. It also holds the :math:`\chi^2_A` of the
      spectra of one bin (nblock float64 per spectrum), and a few float64 arrays of one
      chunk of ``max(1, 2**24 // npix)`` spectra.
    - The new model shares the arrays of ``model``, its precision matrices and its guard
      weights (see `Model`).
    - Unless every block covers all pixels with its guard, a noise-aware model forms
      :math:`P_s` and the factors of the blocks for each spectrum. :math:`P_s` takes
      :math:`\mathrm{npix}^3` flops for a dense ``cov[0]``, or :math:`\mathrm{npix}^2 K`
      for a `LowRank` of rank :math:`K`. Otherwise the infill of the bad pixels forms no
      :math:`P_s`, and costs as in `infill`.
    """
    start = time.perf_counter()
    width = integer(width, "width", 1)
    stride = integer(stride, "stride", 1)
    nguard = integer(nguard, "nguard", 0, NGUARD_MAX)
    percentile = real(percentile, "percentile")
    if not 0 < percentile < 100:
        raise ValueError(
            f"percentile: expected a number > 0 and < 100, got {percentile}"
        )
    check_progress(progress)
    flux, ivar, bad = validated(model, flux, ivar, mask, inflate=False)
    nspec, npix = flux.shape
    if width > npix:
        raise ValueError(
            f"width: expected an integer <= {npix}, the pixels of model, got {width}"
        )
    nbin = len(model.cov)
    bins = model._bins(flux, ivar, bad)
    # Decision A7, in exact arithmetic with the decimal that prints as percentile: in
    # float arithmetic, 99.9 would need 1001 spectra.
    p = Fraction(str(percentile))
    for b, m in enumerate(np.bincount(bins, minlength=nbin).tolist()):
        if m * (100 - p) < 100:
            raise ValueError(
                f"fit_blocks: percentile {percentile} needs at least "
                f"{math.ceil(100 / (100 - p))} spectra in each S/N bin, got {m} in "
                f"bin {b}"
            )
    # A stride above npix gives the starts of npix, and keeps arange in int64.
    starts = np.append(np.arange(0, npix - width, min(stride, npix)), npix - width)
    # The factors of every bin, and P of the bins with bad pixels for their infill,
    # before any spectrum, so that a covariance that fails a check raises at once. A
    # noise model forms them for each spectrum.
    factors: list[list[BlockFactor]] = []
    if not model.noise:
        factors = [block_factors(model, b, starts, width, nguard) for b in range(nbin)]
        for b in np.unique(bins[bad.any(axis=1)]).tolist():
            model._precision(b)
    thresholds = np.empty((nbin, starts.size))
    done = 0
    for b in range(nbin):
        rows = np.flatnonzero(bins == b)
        chi2 = np.empty((rows.size, starts.size))
        for c in spans(model, rows.size, npix):
            spectra = rows[c]
            cover = (
                noisy(model, ivar[spectra], bad[spectra], int(spectra[0]))
                if model.noise
                else model
            )
            F = (
                block_factors(cover, 0, starts, width, nguard)
                if model.noise
                else factors[b]
            )
            X = _scan_vectors(model, cover, b, flux, ivar, bad, spectra)
            chi2[c] = block_chi2(F, X)
            del cover, F
            done += spectra.size
            if progress is not None:
                progress("fit_blocks", done, nspec)
        thresholds[b] = np.percentile(chi2, percentile, axis=0)
    _logger.debug(
        "fit_blocks: %d spectra, %d blocks in %.3f s",
        nspec,
        starts.size,
        time.perf_counter() - start,
    )
    return model._with_blocks(Blocks(starts, width, nguard, thresholds))


def _scan_vectors(
    model: Model,
    cover: Model,
    b: int,
    flux: NDArray[Any],
    ivar: NDArray[Any],
    bad: NDArray[np.bool_],
    spectra: NDArray[np.intp],
) -> NDArray[np.float64]:
    # The working vectors x of the infilled flux of the spectra of bin b, under the
    # covariance of cover.
    bins = np.full(spectra.size, b)
    G = bad[spectra]
    X, a = working(model, flux[spectra], ivar[spectra], G, bins)
    X = filled(cover, X, bins, spectra, G, G)
    # Through flux units, as the docstring of fit_blocks defines x.
    center = 0.0 if model.center is None else model.center
    x = np.array(flux[spectra], dtype=np.float64)
    to_flux(X, a, model.mean[b], center, out=X)
    x[G] = X[G]
    to_working(x, a, center, model.mean[b])
    return x


class BlockFactor(NamedTuple):
    # Block A and G, the pixels within nguard of A clipped at the ends; order lists the
    # pixels of G, counted from G.start, as (G \ A, A). R is the lower Cholesky factor
    # of P_GG in that order, with P the precision matrix of the bin. Where G covers all
    # pixels, P is None and R is the factor of C_AA, the prior (design section 4.2).
    A: slice
    G: slice
    order: NDArray[np.intp]
    R: NDArray[np.float64]
    P: NDArray[np.float64] | None


def block_factors(
    model: Model, b: int, starts: NDArray[np.int64], width: int, nguard: int
) -> list[BlockFactor]:
    npix = model.mean.shape[1]
    factors = []
    # Python integers, so that a large nguard cannot overflow.
    for k, start in enumerate(starts.tolist()):
        A = slice(start, start + width)
        G = slice(max(start - nguard, 0), min(start + width + nguard, npix))
        order = np.r_[G.start : A.start, A.stop : G.stop, A] - G.start
        P = None if G.stop - G.start == npix else model._precision(b)
        try:
            if P is None:
                R = factor_prior(submatrix_of(model.cov[b], np.arange(A.start, A.stop)))
            else:
                R = factor_pattern(P[G, G][np.ix_(order, order)])
        except LinAlgError as error:
            raise LinAlgError(f"{model._name(b)} is {error}, for block {k}") from None
        factors.append(BlockFactor(A, G, order, R, P))
    return factors


def block_chi2(
    factors: list[BlockFactor], X: NDArray[np.float64]
) -> NDArray[np.float64]:
    # chi2_A (nspec, nblock) of the working vectors X of spectra of the bin of factors.
    chi2 = np.empty((len(X), len(factors)))
    Y = None
    for k, (A, G, order, R, P) in enumerate(factors):
        width = A.stop - A.start
        if P is None:
            # ||L^-1 x_A||^2 with L L^T = C_AA, the prior covariance of x_A.
            w, _ = lapack.dtrtrs(R, X[:, A].T, lower=1)
        else:
            if Y is None:
                # (P x)^T of every spectrum, once for all blocks: P is symmetric.
                Y = X @ P
            # y = R^-1 (P x)_G in the order of R. As R R^T = P_GG and r_G = x_G - mu_G
            # = P_GG^-1 (P x)_G, R^T r_G = y, so R_AA^T r_A = y_A (R^T is upper
            # triangular) and chi2_A |A| = ||y_A||^2. The guard pixels enter y as R^T
            # x_(G \ A), which is 0 in the rows of A, so an outlier there cancels to
            # rounding (test_blocks bounds it).
            w, _ = lapack.dtrtrs(R, np.take(Y, G.start + order, axis=1).T, lower=1)
            w = w[-width:]
        chi2[:, k] = np.sum(w**2, axis=0) / width
    return chi2
