import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._model import Model, rebuild, sample_factor
from ._predict import (
    conditionals,
    means,
    noisy,
    predict,
    spans,
    validated,
    weigh,
    working,
)
from ._spectra import (
    Integer,
    Progress,
    Real,
    as_array,
    check_choice,
    check_progress,
    check_rng,
    integer,
    integers,
    items,
    located,
    numeric,
    real,
    require,
    to_flux,
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
    r"""Held-out calibration factor :math:`E[Z^2] / N_*` for Gaussian data.

    Take a held-out spectrum, predicted at :math:`N_*` pixels from :math:`N_k`
    conditioning pixels. The prediction uses the mean and the covariance (divisor
    :math:`N_s - 1`) of :math:`N_s` unweighted training spectra. Let :math:`r` be the
    residual of the :math:`N_*` pixels from their conditional mean, and :math:`C` their
    full conditional covariance. Then :math:`Z^2 = r^\mathsf{T} C^{-1} r` has the mean

    .. math::

        \frac{E[Z^2]}{N_*} = f_s = \left(1 + \frac{1}{N_s}\right)
        \frac{N_s - 2}{N_s - N_k - 2} \, \frac{N_s - 1}{N_s - N_k - N_* - 2}.

    The factor is exact for independent Gaussian working vectors and pixel sets that do
    not depend on the data. Pixels that are neither conditioned on nor predicted, such
    as a guard window or ignored pixels, do not count in :math:`N_k`. :math:`f_s` does
    not hold for data that are not Gaussian, such as scaled spectra with a spread of
    S/N. The arguments broadcast. The arithmetic is float64, which holds each count
    exactly.

    Parameters
    ----------
    n_train : array_like of int
        :math:`N_s`, the number of training spectra; from 1 to :math:`2^{53} - 1`.
    n_cond : array_like of int
        :math:`N_k`, the number of conditioning pixels; from 0 to :math:`2^{53} - 1`.
    n_pred : array_like of int, optional
        :math:`N_*`, the number of predicted pixels; from 1 to :math:`2^{53} - 1`.

    Returns
    -------
    float or ndarray
        :math:`f_s` (unit-free): a float if the arguments are scalars, and otherwise a
        float64 array of their broadcast shape.

    Raises
    ------
    TypeError
        If an argument does not have an integer dtype; bool is not one.
    ValueError
        If an argument is outside its range, the arguments do not broadcast, or
        :math:`N_s - N_k - N_* - 2 \le 0`. The message names the arguments and the first
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


@dataclass(frozen=True, eq=False)
class Calibration:
    r"""Held-out whitened residuals, from `calibration` or another held-out result.

    Row :math:`s` is a spectrum, and column :math:`j` is one prediction of ``n[s, j]``
    of its pixels. ``z2[s, j]`` is :math:`r^\mathsf{T} C^{-1} r`, with :math:`r` the
    residuals of these pixels from their conditional mean and :math:`C` their full
    conditional covariance. ``fs[s, j]`` is its finite-sample factor, such as
    :math:`f_s` of `finite_sample_factor`. For held-out Gaussian spectra and pixel sets
    that do not depend on the data, ``z2 / (n * fs)`` then has the mean 1. The methods
    use only the entries with ``n > 0`` and finite ``z2`` and ``fs``.

    ``Calibration(z2=r.chi2[:, None], n=r.dof[:, None], fs=r.fs[:, None])`` holds the
    result ``r`` of `infill` on held-out spectra, with replace and ignore masks that do
    not depend on the data. ``Calibration(z2=p.z**2, n=1, fs=1)`` holds the result ``p``
    of `pixelwise` with ``inflate=True``.

    The constructor takes array_like values, and broadcasts ``n`` and ``fs`` to the
    shape of ``z2``. It stores read-only copies: ``z2`` and ``fs`` as float64, and ``n``
    as int64. Two calibrations are equal only if they are the same object. Pickle, copy
    and deepcopy check the fields again.

    Attributes
    ----------
    z2 : ndarray, shape (nspec, ncol)
        :math:`Z^2` of each prediction (unit-free): finite and 0 or more, or NaN.
    n : ndarray of int64, shape (nspec, ncol)
        The number of predicted pixels: from 0 to :math:`2^{53} - 1`.
    fs : ndarray, shape (nspec, ncol)
        The finite-sample factor of each prediction (unit-free): finite and positive, or
        NaN.

    Raises
    ------
    TypeError
        If ``z2`` or ``fs`` does not have an integer or floating dtype, or ``n`` does
        not have an integer dtype; bool is neither. The message starts with the name of
        the field.
    ValueError
        If ``z2`` is not 2-D, or ``n`` or ``fs`` does not broadcast to the shape of
        ``z2``. If a field has a value that is not allowed above. The message starts
        with the name of the field.
    """

    z2: NDArray[np.float64]
    n: NDArray[np.int64]
    fs: NDArray[np.float64]

    # Written by hand, as that of Model, so that the fields are checked and n and fs
    # broadcast to the shape of z2.
    def __init__(self, z2: ArrayLike, n: ArrayLike, fs: ArrayLike) -> None:
        z2 = numeric(z2, "z2")
        if z2.ndim != 2:
            raise ValueError(
                f"z2: expected a 2-D array of shape (nspec, k), got shape {z2.shape}"
            )
        ok = np.isnan(z2) | (np.isfinite(z2) & (z2 >= 0))
        require(ok, z2, "z2", "finite values >= 0 or NaN", None)
        n = _broadcast(_counts(n, "n", 0), "n", z2.shape)
        fs = numeric(fs, "fs")
        ok = np.isnan(fs) | (np.isfinite(fs) & (fs > 0))
        require(ok, fs, "fs", "finite values > 0 or NaN", None)
        fs = _broadcast(fs, "fs", z2.shape)
        for name, value, dtype in (
            ("z2", z2, np.float64),
            ("n", n, np.int64),
            ("fs", fs, np.float64),
        ):
            array = np.array(value, dtype=dtype, order="C")
            array.flags.writeable = False
            object.__setattr__(self, name, array)

    __reduce__ = rebuild

    def ratio(
        self, where: ArrayLike | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""Mean of ``z2 / (n * fs)`` in each column, and its standard error.

        Take the :math:`m` entries of column :math:`j` that the methods use, in the
        spectra of ``where``. ``mean[j]`` is the mean of their ``z2 / (n * fs)``, and
        :math:`\mathrm{sem}[j] = \mathrm{std} / \sqrt{m}`, with std their standard
        deviation with the divisor :math:`m - 1`. For held-out spectra, sem is
        conditional on the fitted model. It falls with the number of held-out spectra,
        but the spread across training sets does not.

        Parameters
        ----------
        where : array_like of bool, shape (nspec,), optional
            The spectra to use. None: all.

        Returns
        -------
        mean : ndarray, shape (ncol,)
            The mean (unit-free); NaN where :math:`m = 0`.
        sem : ndarray, shape (ncol,)
            Its standard error (unit-free); NaN where :math:`m < 2`.

        Raises
        ------
        TypeError
            If ``where`` does not have a bool dtype.
        ValueError
            If ``where`` has another shape than (nspec,).
        """
        used = self._used(where)
        m = np.count_nonzero(used, axis=0)
        shape = self.z2.shape
        values = np.divide(self.z2, self.n * self.fs, out=np.zeros(shape), where=used)
        total = values.sum(axis=0)
        mean = np.divide(total, m, out=np.full(m.shape, np.nan), where=m > 0)
        squares = np.square(values - mean, out=np.zeros(shape), where=used).sum(axis=0)
        # std / sqrt(m) = sqrt(sum of squares / ((m - 1) m)).
        variance = np.divide(
            squares, m * (m - 1), out=np.full(m.shape, np.nan), where=m > 1
        )
        return mean, np.sqrt(variance)

    def tail(
        self, q: Real = 0.99, where: ArrayLike | None = None
    ) -> NDArray[np.float64]:
        r"""Fraction of each column with ``z2 / fs`` above a :math:`\chi^2_n` quantile.

        Take the :math:`m` entries of column :math:`j` that the methods use, in the
        spectra of ``where``. The result is the fraction of them with ``z2 / fs`` above
        :math:`F_n^{-1}(q)`. :math:`F_n` is the :math:`\chi^2` distribution function
        with the ``n`` of the entry as its degrees of freedom. For held-out Gaussian
        spectra, the fraction approaches :math:`1 - q` only for :math:`N_s - N_k \gg n`.
        It is 1.37% at :math:`(N_s, N_k, n) = (200, 40, 10)` for :math:`q = 0.99`.

        Parameters
        ----------
        q : float, optional
            The quantile; :math:`0 < q < 1`.
        where : array_like of bool, shape (nspec,), optional
            The spectra to use. None: all.

        Returns
        -------
        ndarray, shape (ncol,)
            The fraction (unit-free); NaN where :math:`m = 0`.

        Raises
        ------
        TypeError
            If ``q`` is not a real number, or ``where`` does not have a bool dtype.
        ValueError
            If ``q`` is not strictly between 0 and 1, or ``where`` has another shape
            than (nspec,).
        """
        q = real(q, "q")
        if not 0 < q < 1:
            raise ValueError(f"q: expected a number strictly between 0 and 1, got {q}")
        used = self._used(where)
        # Imported here, so that `import gspice` does not load scipy.stats.
        from scipy import stats

        # The quantile of each distinct n, which is usually one per column.
        df, inverse = np.unique(self.n[used], return_inverse=True)
        above = np.zeros(used.shape, dtype=bool)
        limit = np.take(stats.chi2.ppf(q, df), inverse)
        above[used] = self.z2[used] / self.fs[used] > limit
        m = np.count_nonzero(used, axis=0)
        fraction: NDArray[np.float64] = np.divide(
            np.count_nonzero(above, axis=0),
            m,
            out=np.full(m.shape, np.nan),
            where=m > 0,
        )
        return fraction

    def qq(
        self, i: Integer = 0, where: ArrayLike | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""Normal scores of column ``i`` against :math:`\chi^2_n`, for a QQ plot.

        Take the :math:`m` entries of column ``i`` that the methods use, in the spectra
        of ``where``. ``expected`` holds :math:`\Phi^{-1}((j - 1/2) / m)` for
        :math:`j = 1, \dots, m`, with :math:`\Phi` the standard normal distribution
        function. ``observed`` holds the sorted scores
        :math:`\Phi^{-1}(F_n(\mathrm{z2} / \mathrm{fs}))`. :math:`F_n` is the
        :math:`\chi^2` distribution function with the ``n`` of the entry as its degrees
        of freedom. A score is :math:`\Phi^{-1}(F_n)` where :math:`F_n < 1/2`. It is
        :math:`-\Phi^{-1}(1 - F_n)` elsewhere, from the survival function, so that both
        tails keep their precision. A score is :math:`-\infty` where ``z2`` is 0, and
        :math:`+\infty` where the survival function underflows to 0. For held-out
        Gaussian spectra, ``observed`` approaches ``expected`` under the condition of
        `tail`.

        Parameters
        ----------
        i : int, optional
            The column; from 0 to ncol − 1.
        where : array_like of bool, shape (nspec,), optional
            The spectra to use. None: all.

        Returns
        -------
        expected : ndarray, shape (m,)
            The normal scores of the plotting positions, in increasing order.
        observed : ndarray, shape (m,)
            The scores of the entries, in increasing order.

        Raises
        ------
        TypeError
            If ``i`` is not an integer, or ``where`` does not have a bool dtype.
        ValueError
            If ``i`` is outside 0 to ncol − 1, or ``where`` has another shape than
            (nspec,).
        """
        i = integer(i, "i", 0)
        k = self.z2.shape[1]
        if i >= k:
            raise ValueError(
                f"i: expected an integer < {k}, the number of columns, got {i}"
            )
        # Imported here, so that `import gspice` does not load scipy.stats.
        from scipy import stats

        used = self._used(where, i)
        x, n = self.z2[used, i] / self.fs[used, i], self.n[used, i]
        cdf = stats.chi2.cdf(x, n)
        scores = np.where(
            cdf < 0.5, stats.norm.ppf(cdf), stats.norm.isf(stats.chi2.sf(x, n))
        )
        m = scores.size
        expected = stats.norm.ppf((np.arange(1, m + 1) - 0.5) / m)
        return expected, np.sort(scores)

    def _used(
        self, where: ArrayLike | None, column: int | slice = slice(None)
    ) -> NDArray[np.bool_]:
        # The entries of the columns that the methods use (design section 6), in the
        # spectra of where.
        n, z2, fs = self.n[:, column], self.z2[:, column], self.fs[:, column]
        used = (n > 0) & np.isfinite(z2) & np.isfinite(fs)
        if where is None:
            return used
        nspec = self.z2.shape[0]
        where = as_array(where, "where")
        if where.dtype != bool:
            raise TypeError(f"where: expected a bool dtype, got {where.dtype}")
        if where.shape != (nspec,):
            raise ValueError(
                f"where: expected shape ({nspec},) for the spectra of z2, got "
                f"{where.shape}"
            )
        used[~where] = False
        return used


def noise_audit(
    model: Model,
    flux: ArrayLike,
    ivar: ArrayLike,
    mask: ArrayLike | None = None,
    *,
    nguard: Integer = 20,
    progress: Progress | None = None,
) -> NDArray[np.float64]:
    r"""Ratio of the residual scatter to the reported noise at each pixel.

    Let :math:`f` be the flux and :math:`\hat f` its prediction by `pixelwise` with
    ``nguard``, in flux units. Then

    .. math::

        \rho_i^2 = \operatorname{mean}_s \, (f_{si} - \hat f_{si})^2 \,
        \mathrm{ivar}_{si},

    where the mean runs over the spectra :math:`s` whose pixel :math:`i` is good (see
    `scale`). Let :math:`t` be the noise-free flux. Let :math:`\eta_i` be the mean of
    :math:`\mathrm{ivar}_{si} \operatorname{var}(f_{si} - t_{si})`, which is 1 where
    ivar is right, and :math:`e_i` the mean of
    :math:`\mathrm{ivar}_{si} (t_{si} - \hat f_{si})^2`. Then

    .. math::

        E[\rho_i^2] = \eta_i + E[e_i]

    if :math:`\hat f_{si}` does not depend on the noise of :math:`f_{si}`. That holds
    for held-out spectra, with ``mask`` and ``ivar`` independent of the noise, and noise
    that is uncorrelated beyond ``nguard``. It fails where the interpolation of a bad
    pixel carries :math:`f_{si}`, or a pixel within ``nguard`` of :math:`i`, into
    :math:`\hat f_{si}` (see `pixelwise`). With several S/N bins, :math:`f_{si}` also
    enters the median S/N that picks the bin of :math:`s`.

    :math:`e_i` holds the astrophysical error, the noise propagated from the
    conditioning pixels, and the estimation error of ``model``.
    So :math:`E[\rho_i^2] \ge \eta_i` where the identity holds, and :math:`e_i` grows
    with S/N. Next to a run of bad pixels longer than ``nguard``, :math:`\rho_i^2` can
    fall below :math:`\eta_i`. A scaled model absorbs into its covariance an
    :math:`\eta_i` that its training spectra share, so the z of `pixelwise` is blind to
    it. :math:`\rho` is not.

    Parameters
    ----------
    model : Model
        The model.
    flux, ivar, mask : array_like, shape (nspec, npix)
        As in `scale`, with the npix of ``model``.
    nguard : int, optional
        The half-width of the guard window of `pixelwise` (pixels); 0 or more.
    progress : callable, optional
        ``progress(stage, done, total)``, called after each chunk of spectra, with the
        stage ``"noise_audit"``, the number of spectra done and their total. For a
        noise-aware model, a chunk is one spectrum.

    Returns
    -------
    ndarray, shape (npix,)
        :math:`\rho` (unit-free); NaN at the pixels where no spectrum is good.

    Raises
    ------
    TypeError
        If ``model`` is not a Model, ``nguard`` is not an integer, or ``progress`` is
        not callable or None. If ``flux``, ``ivar`` or ``mask`` has a dtype that `scale`
        does not allow.
    ValueError
        If ``flux``, ``ivar`` or ``mask`` has a shape or a value that `scale` does not
        allow. If ``flux`` has another npix than ``model``, ``nguard`` is negative, or a
        spectrum has no good pixel.
    numpy.linalg.LinAlgError
        As in `pixelwise`.

    Notes
    -----
    Cost: that of `pixelwise`. Beyond the inputs, and their float64 copies if they are
    long double, noise_audit holds a bool array of the input shape. It also holds a few
    float64 arrays of one chunk of ``max(1, 2**24 // npix)`` spectra.
    """
    start = time.perf_counter()
    nguard = integer(nguard, "nguard", 0)
    check_progress(progress)
    flux, ivar, bad = validated(model, flux, ivar, mask, inflate=False)
    nspec, npix = flux.shape
    bins = model._bins(flux, ivar, bad)
    weigh(model, nguard, bins)
    center = 0.0 if model.center is None else model.center
    total, count = np.zeros(npix), np.zeros(npix, dtype=np.int64)
    for rows in spans(model, nspec, npix):
        good = ~bad[rows]
        count += np.count_nonzero(good, axis=0)
        # The flux of pixelwise, without its var and z.
        X, a = working(model, flux[rows], ivar[rows], bad[rows], bins[rows])
        predicted = np.empty(X.shape)
        cover = noisy(model, ivar[rows], bad[rows], rows.start)
        predict(cover, nguard, X, bins[rows], None, predicted, None, None)
        to_flux(predicted, a, means(model, bins[rows]), center, out=predicted)
        # 0 at the bad pixels, whose flux and ivar are never read.
        terms = np.zeros(good.shape)
        np.subtract(flux[rows], predicted, out=terms, where=good)
        np.square(terms, out=terms)
        np.multiply(terms, ivar[rows], out=terms, where=good)
        total += terms.sum(axis=0)
        # Free the arrays of this chunk before the next one is built.
        del X, predicted, terms, cover
        if progress is not None:
            progress("noise_audit", rows.stop, nspec)
    rho = np.full(npix, np.nan)
    np.divide(total, count, out=rho, where=count > 0)
    np.sqrt(rho, out=rho)
    _logger.debug(
        "noise_audit: %d spectra in %.3f s", nspec, time.perf_counter() - start
    )
    return rho


def calibration(
    model: Model,
    flux: ArrayLike,
    ivar: ArrayLike,
    mask: ArrayLike | None = None,
    *,
    sizes: Iterable[Integer] | None = None,
    layout: Literal["run", "scattered"] = "run",
    rng: np.random.Generator,
    progress: Progress | None = None,
) -> Calibration:
    r"""Held-out whitened residuals for masks drawn independently of the data.

    For each spectrum and each size :math:`n` of ``sizes``, calibration draws a set
    :math:`M` of :math:`n` good pixels. It computes, as `infill` with ``replace=M``
    does,

    .. math::

        z^2 = r_M^\mathsf{T} C_{MM}^{-1} r_M,

    with :math:`r_M` the input values at :math:`M` minus their conditional mean given
    the pixels outside :math:`G = M \cup \mathrm{bad}`, and :math:`C_{MM}` their full
    conditional covariance. It also computes :math:`f_s(N_s, \mathrm{npix} - |G|, n)` of
    `finite_sample_factor`, NaN where it is not defined. :math:`N_s` is the ``n_train``
    of the bin of the spectrum. The factor is 1 where no factor is derived (see
    `Repaired.fs`). For held-out Gaussian spectra of a model with a derived factor,
    :math:`z^2 / (n f_s)` then has the mean 1. For a noise-aware model, the covariance
    is the :math:`C_s` of the spectrum (see `Model`).

    Rank the :math:`m` good pixels of a spectrum from 0 to :math:`m - 1` in increasing
    order. With ``layout="run"``, :math:`M` holds those of ranks ``start`` to
    ``start + n - 1``, with ``start = rng.integers(m - n + 1)``.
    With ``layout="scattered"``, :math:`M` holds those of ranks
    ``rng.choice(m, n, replace=False)``. ``rng`` draws these spectrum by spectrum in row
    order, and size by size within a spectrum. It draws nothing else, so the masks do
    not depend on the chunking. A spectrum with :math:`m < n` draws nothing for that
    size. Its entry there has ``n`` 0 and ``z2`` NaN.

    Parameters
    ----------
    model : Model
        The model.
    flux, ivar, mask : array_like, shape (nspec, npix)
        As in `scale`, with the npix of ``model``.
    sizes : iterable of int, optional
        The sizes :math:`n`, one column of the result each, in this order; 1 or more.
        None: the sizes 1, 4, 16, 64, 256 and 1024 that are not above half the median
        number of good pixels per spectrum (``np.median``). Without a spectrum, all six.
        A spectrum with fewer good pixels than a size has ``n`` 0 there, so the spectra
        of a column can differ between sizes. `Calibration` methods with
        ``where=(result.n > 0).all(axis=1)`` compare all sizes on the same spectra.
    layout : {"run", "scattered"}, optional
        Whether :math:`M` is a run of good pixels, or good pixels scattered at random.
    rng : numpy.random.Generator
        The source of the masks, as above.
    progress : callable, optional
        ``progress(stage, done, total)``, called after each chunk of spectra, with the
        stage ``"calibration"``, the number of spectra done and their total. For a
        noise-aware model, a chunk is one spectrum.

    Returns
    -------
    Calibration
        ``z2``, ``n`` and ``fs`` of each spectrum (row) and size (column).

    Raises
    ------
    TypeError
        If ``model`` is not a Model, or ``sizes`` is neither None nor an iterable of
        integers. If ``rng`` is not a numpy.random.Generator, or ``progress`` is not
        callable or None. If ``flux``, ``ivar`` or ``mask`` has a dtype that `scale`
        does not allow.
    ValueError
        If ``flux``, ``ivar`` or ``mask`` has a shape or a value that `scale` does not
        allow. If ``flux`` has another npix than ``model``, or a spectrum has no good
        pixel. If ``sizes`` is empty or holds an integer below 1, or ``layout`` is
        neither ``"run"`` nor ``"scattered"``. If ``sizes`` is None and the median
        number of good pixels per spectrum is below 2.
    numpy.linalg.LinAlgError
        Before the first chunk, if the covariance of a bin fails its check, for a bin
        with a spectrum that has more good pixels than the smallest size. For a
        noise-aware model, the check is at the spectrum. Otherwise as in `infill` with
        ``replace=M``.

    Notes
    -----
    Cost:

    - For each size, :math:`\mathrm{npix}^3` flops for :math:`P` at the first use of a
      bin.
    - Then :math:`|G|^3/3` flops per pattern of a chunk of ``max(1, 2**24 // npix)``
      spectra, or :math:`n^3/3` where :math:`G` covers all pixels. A spectrum has at
      most one pattern per size.
    - Per spectrum, :math:`2 |G| (\mathrm{npix} + |G|)` flops, or
      :math:`2 |G| \, \mathrm{npix}` where :math:`G` is one run.
    - The masks of a chunk take up to :math:`\mathrm{len(sizes)} \times 2^{24}` bytes.
    - A noise-aware model costs per spectrum and size as in `infill`, with
      :math:`M = T`. A dense ``cov[0]`` factors :math:`C_s` once per spectrum for all
      sizes.
    """
    start = time.perf_counter()
    if sizes is not None:
        sizes = items(sizes, "sizes", None, partial(integer, minimum=1))
    check_choice(layout, "layout", ("run", "scattered"))
    check_rng(rng)
    check_progress(progress)
    flux, ivar, bad = validated(model, flux, ivar, mask, inflate=False)
    nspec, npix = flux.shape
    count = npix - np.count_nonzero(bad, axis=1)
    if sizes is None:
        # Design section 5.3: the median count, so that a few spectra with many bad
        # pixels do not remove a size for the others.
        median = np.median(count) if nspec else np.inf
        sizes = tuple(size for size in (1, 4, 16, 64, 256, 1024) if 2 * size <= median)
        if not sizes:
            raise ValueError(
                "sizes: expected a median of at least 2 good pixels per spectrum for "
                f"the default sizes, got {median}"
            )
    bins = model._bins(flux, ivar, bad)
    # P of each bin with a spectrum that keeps a conditioning pixel for some size
    # (m > n), before any spectrum, so that a covariance that fails its check raises
    # at once, as in infill. A noise model factors C_s per spectrum.
    if not model.noise:
        for b in np.unique(bins[count > min(sizes)]).tolist():
            model._precision(b)
    shape = (nspec, len(sizes))
    z2, n = np.full(shape, np.nan), np.empty(shape, dtype=np.int64)
    for rows in spans(model, nspec, npix):
        X, _ = working(model, flux[rows], ivar[rows], bad[rows], bins[rows])
        index = np.arange(rows.start, rows.stop)
        cover = noisy(model, ivar[rows], bad[rows], rows.start)
        for j, M in enumerate(_masks(bad[rows], sizes, layout, rng)):
            # chi2, dof and fs of infill with replace=M. M holds only good pixels, so
            # T = M, and dof = |M| is 0, with z2 NaN, only where no mask was drawn.
            # chi2 needs no variances (design section 12, optimization (h)).
            for group, _, solution in conditionals(
                cover,
                X,
                bins[rows],
                index,
                M | bad[rows],
                M,
                M,
                None,
                covariance=False,
                variance=False,
            ):
                if solution is not None:
                    z2[index[group], j] = solution.chi2
            n[rows, j] = np.count_nonzero(M, axis=1)
        # Free the arrays of this chunk before the next one is built.
        del X, M, cover
        if progress is not None:
            progress("calibration", rows.stop, nspec)
    # Design section 4.3: the models without a derived factor report 1. M holds only
    # good pixels, so N_k = count - n.
    fs = np.ones(shape)
    if model._derived:
        n_train = np.array(model.n_train)[bins, np.newaxis]
        fs = sample_factor(n_train, count[:, np.newaxis] - n, n)
    _logger.debug(
        "calibration: %d spectra, %d sizes in %.3f s",
        nspec,
        len(sizes),
        time.perf_counter() - start,
    )
    return Calibration(z2, n, fs)


def _masks(
    bad: NDArray[np.bool_],
    sizes: Sequence[Integer],
    layout: str,
    rng: np.random.Generator,
) -> NDArray[np.bool_]:
    # The masks M of calibration, (len(sizes), nspec, npix), drawn spectrum by spectrum
    # and size by size within a spectrum.
    masks = np.zeros((len(sizes), *bad.shape), dtype=bool)
    for s, row in enumerate(bad):
        good = np.flatnonzero(~row)
        for j, n in enumerate(sizes):
            if good.size < n:
                continue
            if layout == "run":
                start = rng.integers(good.size - n + 1)
                masks[j, s, good[start : start + n]] = True
            else:
                masks[j, s, good[rng.choice(good.size, n, replace=False)]] = True
    return masks


def _broadcast(value: NDArray[Any], name: str, shape: tuple[int, ...]) -> NDArray[Any]:
    try:
        return np.broadcast_to(value, shape)
    except ValueError:
        raise ValueError(
            f"{name}: expected a shape that broadcasts to {shape} like z2, got "
            f"{value.shape}"
        ) from None


def _counts(value: ArrayLike, name: str, minimum: int) -> NDArray[np.integer[Any]]:
    array = integers(value, name)
    # Below 2^53, float64 holds every integer, so sample_factor tests the sign of
    # n_train - n_cond - n_pred - 2 exactly, and the methods of Calibration use the
    # exact n.
    ok = (array >= minimum) & (array < 2**53)
    require(ok, array, name, f"integers from {minimum} to 2^53 - 1", None)
    return array
