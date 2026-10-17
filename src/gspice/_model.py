import logging
import math
import os
import time
import zipfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import KW_ONLY, dataclass, field, fields
from functools import partial
from itertools import pairwise
from typing import Any

import numpy as np
from numpy.linalg import LinAlgError
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import issymmetric

from ._linalg import guard_weights, precision
from ._spectra import (
    Boolean,
    Integer,
    Progress,
    Real,
    boolean,
    check_progress,
    chunks,
    dilate,
    finite_array,
    integer,
    interpolate,
    listing,
    median_snr,
    numeric,
    real,
    require,
    root_mean_ivar,
    validate,
    vector,
)

# The arrays that load reads. save also writes gspice_version and cov0, cov1, ...
_REQUIRED = ("schema", "mean", "n_train", "edges", "center", "weighted")

_logger = logging.getLogger("gspice")

# The path-like objects of the Python glossary, which os.fsdecode accepts.
FilePath = str | bytes | os.PathLike[str] | os.PathLike[bytes]


@dataclass(frozen=True, eq=False)
class Model:
    """Fitted Gaussian model of spectra on a common pixel grid.

    A spectrum in S/N bin b has the working vector x ~ N(0, cov[b]) in scaled units.
    Let ftilde be the flux of the spectrum with its bad pixels interpolated, and a the
    square root of the mean inverse variance of its good pixels (see `scale`). Then
    x = a (ftilde - center) - mean[b] for a constant centre, and
    x = a ftilde - mean[b] otherwise.

    The constructor takes the attributes below as array_like values, iterables and
    numpy scalars, and stores them as the types given here, with read-only float64
    copies of the arrays. The attributes after n_train are keyword-only. The
    constructor does not check that a covariance is positive definite, which costs
    npix^3. Two models are equal only if they are the same object.

    A model computes the precision matrix of a bin at its first use and keeps it, with
    the guard weights of the latest nguard of each bin (see `pixelwise`). Each matrix
    takes npix^2 x 8 bytes, 46 MB at npix 2400. Pickle, copy and deepcopy start
    without them.

    Attributes
    ----------
    mean : ndarray, shape (nbin, npix)
        The vector mean[b] of each of the nbin >= 1 S/N bins (scaled), subtracted to
        give x. npix >= 1 is the number of pixels. Finite.
    cov : tuple of ndarray, each shape (npix, npix)
        The covariance of x in each bin (scaled^2): nbin finite, exactly symmetric
        matrices with a positive diagonal.
    n_train : tuple of int
        The number N_s >= 1 of training spectra in each bin (nbin integers).
    edges : tuple of float
        The nbin - 1 interior S/N edges between the bins: finite and strictly
        increasing. Spectrum s with signal-to-noise ratio snr_s (see `snr`) is in the
        bin np.searchsorted(edges, snr_s, side="right").
    center : float or None
        None for a mean-centred model, or the constant centre c (flux).
    weighted : bool
        True if the model was fitted with positive weights that are not all equal.

    Raises
    ------
    TypeError
        If a field has a type or dtype that is not allowed above. The message starts
        with the name of the field.
    ValueError
        If a field has a shape or a value that is not allowed above. The message
        starts with the name of the field.
    """

    mean: NDArray[np.float64]
    cov: tuple[NDArray[np.float64], ...]
    n_train: tuple[int, ...]
    _: KW_ONLY
    edges: tuple[float, ...]
    center: float | None
    weighted: bool
    _cache: dict[tuple[str, int], Any] = field(init=False, repr=False)

    # Written by hand because the parameters take more than the fields store: any
    # iterable, array_like values and numpy scalars. The fields have no defaults, so
    # a field that __init__ does not set raises AttributeError instead of reading a
    # class default. _: KW_ONLY keeps the fields after n_train out of __match_args__.
    # A dataclass subclass would generate an __init__ without these checks.
    def __init__(
        self,
        mean: ArrayLike,
        cov: Iterable[ArrayLike],
        n_train: Iterable[Integer],
        *,
        edges: Iterable[Real] = (),
        center: Real | None = None,
        weighted: Boolean = False,
    ) -> None:
        mean = numeric(mean, "mean")
        if mean.ndim != 2 or 0 in mean.shape:
            raise ValueError(
                f"mean: expected a 2-D array of shape (nbin, npix) with nbin >= 1 and "
                f"npix >= 1, got shape {mean.shape}"
            )
        mean = finite_array(mean, "mean", ("bin", "pixel"))
        nbin, npix = mean.shape
        cov = _items(cov, "cov", nbin, partial(_covariance, npix=npix))
        n_train = _items(n_train, "n_train", nbin, partial(integer, minimum=1))
        edges = _items(edges, "edges", nbin - 1, real)
        for i, (low, high) in enumerate(pairwise(edges)):
            if low >= high:
                raise ValueError(
                    f"edges: expected strictly increasing values, "
                    f"got edges[{i}] = {low} and edges[{i + 1}] = {high}"
                )
        center = None if center is None else real(center, "center")
        weighted = boolean(weighted, "weighted")
        # After the checks, because issymmetric of scipy 1.11 rejects read-only arrays.
        for array in (mean, *cov):
            array.flags.writeable = False
        for name, value in (
            ("mean", mean),
            ("cov", cov),
            ("n_train", n_train),
            ("edges", edges),
            ("center", center),
            ("weighted", weighted),
            ("_cache", {}),
        ):
            object.__setattr__(self, name, value)

    @property
    def _derived(self) -> bool:
        # The kinds of model for which f_s is derived (design section 4.3, A13).
        return self.center is None and not self.weighted

    def __reduce__(self) -> tuple[partial["Model"], tuple[()]]:
        # Pickle, copy and deepcopy rebuild through __init__, so they check the fields
        # again. Keywords keep a pickle valid when a later version adds a field.
        init = {f.name: getattr(self, f.name) for f in fields(self) if f.init}
        return partial(type(self), **init), ()

    def _precision(self, b: int, *, subject: str | None = None) -> NDArray[np.float64]:
        P: NDArray[np.float64] | None = self._cache.get(("precision", b))
        if P is None:
            start = time.perf_counter()
            try:
                P = precision(self.cov[b])
            except LinAlgError as error:
                where = f"model: cov[{b}]" if subject is None else subject
                raise LinAlgError(f"{where} is {error}") from None
            _logger.debug(
                "cov[%d]: precision matrix in %.3f s", b, time.perf_counter() - start
            )
            # Read-only, like the fields: a write would change every later result.
            P.flags.writeable = False
            self._cache["precision", b] = P
        return P

    def _bins(
        self, flux: NDArray[Any], ivar: NDArray[Any], bad: NDArray[np.bool_]
    ) -> NDArray[np.intp]:
        bins = np.zeros(len(flux), dtype=np.intp)
        # A model with one bin skips the S/N, which sorts every row.
        if len(self.cov) > 1:
            for rows in chunks(*flux.shape):
                snr = median_snr(flux[rows], ivar[rows], bad[rows])
                bins[rows] = np.searchsorted(self.edges, snr, side="right")
        return bins

    def _guard_weights(
        self, b: int, nguard: int, *, subject: str | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        cached = self._cache.get(("guard", b))
        if cached is not None and cached[0] == nguard:
            return cached[1], cached[2]
        C = self.cov[b]
        npix = len(C)
        if nguard < npix - 1:
            P = self._precision(b, subject=subject)
            start = time.perf_counter()
            try:
                W, var = guard_weights(P, np.diagonal(C), nguard)
            except LinAlgError as error:
                where = f"model: cov[{b}]" if subject is None else subject
                raise LinAlgError(f"{where} is {error}") from None
        else:
            # Every window covers all pixels: the exact prior, without P (design
            # section 4.2).
            start = time.perf_counter()
            W, var = np.zeros((npix, npix)), np.diagonal(C).copy()
        _logger.debug(
            "cov[%d]: guard weights for nguard %d in %.3f s",
            b,
            nguard,
            time.perf_counter() - start,
        )
        for array in (W, var):
            array.flags.writeable = False
        self._cache["guard", b] = nguard, W, var
        return W, var


def sample_factor(
    n_train: ArrayLike, n_cond: ArrayLike, n_pred: ArrayLike
) -> NDArray[np.float64]:
    # f_s of design section 4.3, NaN where n_pred < 1 or n_train - n_cond - n_pred - 2
    # <= 0 (then n_train > 3 for n_cond >= 0, so every denominator is positive). The
    # counts become float64, which holds every integer below 2^53, so an unsigned dtype
    # cannot wrap around.
    N, k, m = np.broadcast_arrays(
        *(np.asarray(count, dtype=np.float64) for count in (n_train, n_cond, n_pred))
    )
    defined = (m >= 1) & (N - k - m - 2 > 0)
    factor = np.full(N.shape, np.nan)
    N, k, m = N[defined], k[defined], m[defined]
    factor[defined] = (1 + 1 / N) * (N - 2) / (N - k - 2) * (N - 1) / (N - k - m - 2)
    return factor


def fit(
    flux: ArrayLike,
    ivar: ArrayLike,
    mask: ArrayLike | None = None,
    *,
    nsigma: Sequence[Real] | NDArray[np.integer[Any] | np.floating[Any]] = (
        20.0,
        8.0,
        6.0,
    ),
    maxbadpix: Integer = 64,
    nguard: Integer = 20,
    cumulative: Boolean = True,
    center: Real | None = None,
    weights: ArrayLike | None = None,
    progress: Progress | None = None,
) -> tuple[Model, NDArray[np.bool_]]:
    """Fit a model to training spectra with iterative outlier clipping (paper Alg. 4).

    The bad pixels are (mask != 0) | (ivar == 0), and fit never reads flux or ivar
    there. A spectrum is used unless it has more than maxbadpix bad pixels, has weight
    0, or a pass mask or the final mask covers it completely; N_s counts the spectra
    of a fit. A fit on a mask interpolates the pixels in it, scales each spectrum by
    a_s over the pixels outside it (see `scale`), and estimates

        m = sum_s w_s x_s / V1,  S = sum_s w_s (x_s - m)(x_s - m)^T / (V1 - V2 / V1),

    where x_s = a_s ftilde_s, V1 = sum_s w_s and V2 = sum_s w_s^2; with unit weights,
    m is the column mean and the divisor is N_s - 1. With a constant centre c, m = 0,
    x_s = a_s (ftilde_s - c) and S = sum_s w_s x_s x_s^T / V1.

    Pass n of N = len(nsigma) fits on its mask M_n with unit weights, computes the
    in-sample z of `pixelwise` with nguard at every pixel, interpolated ones included,
    and flags F_n: each pixel with |z| > nsigma[n - 1] and its two neighbours.
    M_1 = bad. With cumulative=True (paper Alg. 4), M_{n+1} = M_n | F_n and the final
    mask is M_{N+1}; with cumulative=False (2022), M_{n+1} = bad | F_n and the final
    mask is bad | F_{N-1} | F_N. nsigma=() fits on the mask bad, without clipping.
    The final fit uses the final mask and the weights w_s; reweighting the clipping
    raised false flags in the paper (l.548).

    With cumulative=False, fit equals legacy covar_iter_mask, the covariance to
    rounding and the mask exactly, if nguard is 20, center and weights are None, flux
    is float64, ivar == 0 only at pixels in mask, nsigma is not empty, no |z| lies
    within the error of the 2022 Schur updates of a threshold, no pass mask or final
    mask covers completely a spectrum with at most maxbadpix bad pixels, and fit
    raises no error.

    Parameters
    ----------
    flux, ivar, mask : array_like, shape (nspec, npix)
        As in `scale`, except that a spectrum without a good pixel is dropped.
    nsigma : sequence of float or ndarray, shape (N,), optional
        The threshold on |z| of each clipping pass, in order; finite and > 0.
    maxbadpix : int, optional
        The largest number of bad pixels of a used spectrum; >= 0.
    nguard : int, optional
        The half-width of the guard window of z (pixels); >= 0.
    cumulative : bool, optional
        Whether each pass keeps the flags of every earlier pass (paper Alg. 4), or
        only those of the pass before it (2022).
    center : float, optional
        The constant centre c (flux); finite. None: a mean-centred model.
    weights : array_like, shape (nspec,), optional
        The weights w_s of the final fit, of an integer or floating dtype; finite and
        >= 0. None: equal weights.
    progress : callable, optional
        progress(stage, done, total), called after each chunk of spectra of each pass
        and of the final fit, with the stage "fit", the number of spectra of that fit
        done, and their total N_s.

    Returns
    -------
    model : Model
        The model, of one bin, with n_train (N_s,), the centre, and weighted True if
        the weights of the spectra of the final fit are not all equal. It holds its
        precision matrix (see `Model`).
    mask : ndarray of bool, shape (nspec, npix)
        The final mask, and True at every pixel of a spectrum that is not used.

    Raises
    ------
    TypeError
        If an argument has a type or a dtype that is not allowed above.
    ValueError
        If an argument has a shape or a value that is not allowed above, or if a fit
        lacks the spectra it needs: N_s - 1 >= npix for a mean-centred model, N_s >=
        npix for a constant centre, and a good pixel in one of them at every pixel.
        The message names the counts or the first such pixels.
    numpy.linalg.LinAlgError
        If a covariance has a variance of 0, or fails the check of its precision
        matrix: it is not positive definite, or the reciprocal condition number of the
        covariance scaled to unit diagonal is below 1e-13, which can happen for N_s - 1
        close to npix. Clipping checks P only if nguard < npix - 1. The message names
        the pass or the final fit and N_s, and the pixel of variance 0 or the
        reciprocal condition number below 1e-13.

    Notes
    -----
    Cost per pass: about N_s npix^2 flops for the covariance, npix^3 for its precision
    matrix, npix (w^3/3 + 2 npix w) for the guard weights, w = min(2 nguard + 1, npix),
    and 2 N_s npix^2 for z. The final fit costs about N_s npix^2 + npix^3 flops. Each
    fit holds the working vectors of its spectra, N_s npix x 8 bytes.
    """
    start = time.perf_counter()
    thresholds = _thresholds(nsigma)
    maxbadpix = integer(maxbadpix, "maxbadpix", 0)
    nguard = integer(nguard, "nguard", 0)
    cumulative = boolean(cumulative, "cumulative")
    if center is not None:
        center = real(center, "center")
    check_progress(progress)
    flux, ivar, bad = validate(flux, ivar, mask)
    nspec, npix = flux.shape
    if weights is None:
        weights = np.ones(nspec)
    else:
        weights = vector(weights, "weights", nspec, "row")
        require(weights >= 0, weights, "weights", "values >= 0", ("row",))
    over = np.count_nonzero(bad, axis=1) > maxbadpix
    zero = ~over & (weights == 0)
    removed = (nspec, int(np.count_nonzero(over)), int(np.count_nonzero(zero)))
    rows = np.flatnonzero(~over & ~zero & ~bad.all(axis=1))
    M = bad[rows]
    n_cond = max(npix - 2 * nguard - 1, 0)
    for n, threshold in enumerate(thresholds, 1):
        _check(M, center, f"pass {n}", removed)
        n_train = rows.size
        # In a spectrum of typical leverage, the in-sample |z| of an outlier of any
        # amplitude approaches this ceiling (design section 4.1).
        if center is None:
            ceiling = math.sqrt((n_train - 1) * (n_train - n_cond - 1) / n_train)
        else:
            ceiling = math.sqrt(n_train - n_cond)
        if threshold > ceiling:
            _logger.warning(
                "fit: pass %d has nsigma %g above %.3g, the in-sample ceiling of |z| "
                "at N_s = %d",
                n,
                threshold,
                ceiling,
                n_train,
            )
        subject = f"fit: the clipping covariance of pass {n} (N_s = {n_train})"
        z = _clipping_z(flux, ivar, rows, M, center, nguard, subject, progress)
        flagged = dilate(np.abs(z, out=z) > threshold, 1)
        # Free z before the next fit allocates its working vectors.
        del z
        # The final mask is M_N | F_N in both modes.
        if cumulative or n == len(thresholds):
            M = M | flagged
        else:
            M = bad[rows] | flagged
        keep = ~M.all(axis=1)
        _logger.info(
            "fit: pass %d with nsigma %g flagged %.3g%% of the pixels of its %d "
            "spectra; spectra dropped: %d",
            n,
            threshold,
            100 * np.count_nonzero(flagged) / flagged.size,
            n_train,
            n_train - np.count_nonzero(keep),
        )
        rows, M = rows[keep], M[keep]
    _check(M, center, "the final fit", removed)
    n_train = rows.size
    # Only the ratios of the weights matter. In units of the largest, V2 neither
    # overflows nor underflows, and equal weights are exactly 1.
    w = weights[rows] / np.max(weights[rows])
    subject = f"fit: the final covariance (N_s = {n_train})"
    _, model = _fitted(flux, ivar, rows, M, center, w, subject, progress)
    if model.weighted:
        _logger.info(
            "fit: the final fit of %d spectra is weighted, with V1^2/V2 = %.4g "
            "effective spectra",
            n_train,
            w.sum() ** 2 / np.vdot(w, w),
        )
    model._precision(0, subject=subject)
    training = np.ones((nspec, npix), dtype=bool)
    training[rows] = M
    _logger.debug("fit: %d spectra in %.3f s", nspec, time.perf_counter() - start)
    return model, training


def _thresholds(value: object) -> tuple[float, ...]:
    nsigma = numeric(value, "nsigma")
    if nsigma.ndim == 0:
        raise TypeError(f"nsigma: expected a 1-D sequence or array, got {nsigma}")
    if nsigma.ndim != 1:
        raise ValueError(
            f"nsigma: expected a 1-D sequence or array, got shape {nsigma.shape}"
        )
    ok = np.isfinite(nsigma) & (nsigma > 0)
    require(ok, nsigma, "nsigma", "finite values > 0", ("element",))
    return tuple(nsigma.astype(np.float64).tolist())


def _check(
    M: NDArray[np.bool_],
    center: float | None,
    fitted: str,
    removed: tuple[int, int, int],
) -> None:
    # The requirements of design section 4.1 for the fit whose mask is M.
    n_train, npix = M.shape
    needed = npix + 1 if center is None else npix
    if n_train < needed:
        nspec, over, zero = removed
        full = nspec - over - zero - n_train
        kind = "a mean-centred" if center is None else "a constant-centre"
        raise ValueError(
            f"fit: {fitted} needs at least {needed} spectra for {kind} model of "
            f"{npix} pixels, got {n_train}: of {nspec} spectra, maxbadpix removed "
            f"{over}, weight 0 removed {zero} and full masks removed {full}"
        )
    empty = np.flatnonzero(M.all(axis=0))
    if empty.size:
        raise ValueError(
            f"fit: {fitted} needs a good pixel in one of its {n_train} spectra at "
            f"every pixel, got none at pixels {listing(empty, 'pixels')}"
        )


def _clipping_z(
    flux: NDArray[Any],
    ivar: NDArray[Any],
    rows: NDArray[np.intp],
    M: NDArray[np.bool_],
    center: float | None,
    nguard: int,
    subject: str,
    progress: Progress | None,
) -> NDArray[np.float64]:
    # The in-sample z of the spectra rows, whose mask is M, at every pixel under the
    # clipping model with unit weights (design decisions A12 and A23). With unit
    # weights, X holds the working vectors.
    X, model = _fitted(flux, ivar, rows, M, center, np.ones(rows.size), subject, None)
    W, var = model._guard_weights(0, nguard, subject=subject)
    sigma = np.sqrt(var)
    nspec, npix = X.shape
    # z = (x - x W) / sqrt(var), as in pixelwise, in place of the working vectors.
    for c in chunks(nspec, npix):
        X[c] -= X[c] @ W
        X[c] /= sigma
        if progress is not None:
            progress("fit", c.stop, nspec)
    return X


def _fitted(
    flux: NDArray[Any],
    ivar: NDArray[Any],
    rows: NDArray[np.intp],
    M: NDArray[np.bool_],
    center: float | None,
    w: NDArray[np.float64],
    subject: str,
    progress: Progress | None,
) -> tuple[NDArray[np.float64], Model]:
    # The model of design section 4.1 on the spectra rows, whose mask is M, with the
    # weights w, and the working vectors x_s times sqrt(w_s).
    X = np.empty(M.shape)
    a = np.empty(rows.size)
    for c in chunks(*M.shape):
        X[c] = interpolate(flux[rows[c]], M[c])
        a[c] = root_mean_ivar(ivar[rows[c]], M[c])
    nspec, npix = X.shape
    if center is not None:
        X -= center
    X *= a[:, np.newaxis]
    V1 = w.sum()
    if center is None:
        mean = w @ X / V1
        X -= mean
        divisor = V1 - np.vdot(w, w) / V1
    else:
        mean = np.zeros(npix)
        divisor = V1
    X *= np.sqrt(w)[:, np.newaxis]
    # numpy computes B^T B by syrk and copies one triangle to the other, so each
    # product, and so their sum, is exactly symmetric.
    S = np.zeros((npix, npix))
    for c in chunks(nspec, npix):
        B = X[c]
        S += B.T @ B
        if progress is not None:
            progress("fit", c.stop, nspec)
    S /= divisor
    # Model rejects a diagonal entry of 0 with a ValueError about cov[0]. Such a pixel
    # makes S singular, so fit raises as for any S that is not positive definite.
    constant = np.flatnonzero(np.diagonal(S) == 0)
    if constant.size:
        raise LinAlgError(
            f"{subject} is not positive definite: its variance at pixel {constant[0]} "
            "is 0"
        )
    weighted = np.any(w != 1)
    model = Model(mean[np.newaxis], (S,), (nspec,), center=center, weighted=weighted)
    return X, model


def save(model: Model, path: FilePath) -> None:
    """Write model to the .npz file path.

    The archive is not compressed. It holds the arrays schema (the integer 1),
    gspice_version, mean, n_train, edges, center (NaN if None) and weighted, and cov0,
    cov1, ... for the covariance of each bin. `load` reads it.

    Parameters
    ----------
    model : Model
        The model to write.
    path : str or path-like
        The file to write, with any suffix: save adds none. An existing file is
        replaced.

    Raises
    ------
    TypeError
        If model is not a Model, or path is not a str or path-like.
    OSError
        If the file cannot be written.
    """
    if not isinstance(model, Model):
        raise TypeError(f"model: expected a Model, got {type(model).__name__}")
    arrays = _arrays(model)
    # fsdecode rejects an integer, which open would take as a file descriptor.
    with open(os.fsdecode(path), "wb") as file:
        np.savez(file, **arrays)


def load(path: FilePath) -> Model:
    """Read a model written by `save`.

    Parameters
    ----------
    path : str or path-like
        A file written by `save`.

    Returns
    -------
    Model
        The model.

    Raises
    ------
    ValueError
        If the file is not a zip archive, is a damaged one, has no schema or a schema
        other than the 0-d integer 1, lacks one of the arrays schema, mean, n_train,
        edges, center and weighted, or holds an array that `save` does not write; the
        message names path. Also if `Model` rejects a field, for example a count of
        covariances other than the number of bins; the message names the field.
    TypeError
        If path is not a str or path-like, or if `Model` rejects a field.
    OSError
        If the file cannot be read, and for some damaged archives.
    """
    # fsdecode rejects an integer, as in save, and makes a bytes path text for the
    # messages.
    path = os.fsdecode(path)
    with open(path, "rb") as file:
        # The prefixes by which np.load recognizes a zip archive. It returns an array
        # for an .npy file and tries pickle for other files.
        if file.read(4) not in (b"PK\x03\x04", b"PK\x05\x06"):
            raise ValueError(
                f"path: expected an .npz archive written by save, got a file that is "
                f"not a zip archive in {path}"
            )
        file.seek(0)
        # The zip directory is read on opening, and each array on access with its
        # CRC-32 check.
        try:
            with np.load(file, allow_pickle=False) as archive:
                return _from_archive(archive, path)
        except zipfile.BadZipFile as error:
            raise ValueError(
                f"path: expected an .npz archive written by save, got a damaged zip "
                f"archive ({error}) in {path}"
            ) from None


def _from_archive(archive: Any, path: str) -> Model:
    schema = archive.get("schema")
    if (
        not isinstance(schema, np.ndarray)
        or schema.ndim
        or schema.dtype.kind not in "iu"
        or schema != 1
    ):
        # The slices bound the message for a raw zip member.
        found = "no schema" if schema is None else f"schema {str(schema)[:40]}"
        raise ValueError(
            f"path: expected a file written by save, with schema 1, got {found} in "
            f"{path}"
        )
    missing = [key for key in _REQUIRED if key not in archive]
    if missing:
        raise ValueError(
            f"path: expected the arrays that save writes, got no {', '.join(missing)} "
            f"in {path}"
        )
    cov: list[NDArray[Any]] = []
    while f"cov{len(cov)}" in archive:
        cov.append(archive[f"cov{len(cov)}"])
    known = {*_REQUIRED, "gspice_version", *(f"cov{b}" for b in range(len(cov)))}
    unknown = sorted(set(archive.files) - known)
    if unknown:
        writer = archive.get("gspice_version")
        by = "" if writer is None else f", written by gspice {str(writer)[:40]},"
        raise ValueError(
            f"path: expected only the arrays that save writes, got "
            f"{', '.join(unknown)}{by} in {path}"
        )
    center = archive["center"][()]
    return Model(
        archive["mean"],
        cov,
        archive["n_train"],
        edges=archive["edges"],
        center=None if isinstance(center, float) and math.isnan(center) else center,
        weighted=archive["weighted"][()],
    )


def _arrays(model: Model) -> dict[str, Any]:
    # __init__ defines __version__ after it imports this module.
    from . import __version__

    return {
        "schema": np.int64(1),
        "gspice_version": __version__,
        "mean": model.mean,
        "n_train": np.array(model.n_train, dtype=np.int64),
        "edges": np.array(model.edges),
        "center": np.nan if model.center is None else model.center,
        "weighted": model.weighted,
    } | {f"cov{b}": C for b, C in enumerate(model.cov)}


def _covariance(value: ArrayLike, name: str, npix: int) -> NDArray[np.float64]:
    C = numeric(value, name)
    if C.shape != (npix, npix):
        raise ValueError(
            f"{name}: expected shape ({npix}, {npix}) for the {npix} pixels of mean, "
            f"got {C.shape}"
        )
    C = finite_array(C, name, ("row", "column"))
    if not issymmetric(C):
        gap = np.abs(C - C.T).max()
        raise ValueError(
            f"{name}: expected an exactly symmetric matrix, "
            f"got max |C - C.T| = {gap:.3g}"
        )
    # The prior of a window that covers every pixel takes its variances from the
    # diagonal, without the factorization that would reject it (design section 4.2).
    diagonal = np.diagonal(C)
    require(diagonal > 0, diagonal, name, "a positive diagonal", ("pixel",))
    return C


def _items(
    value: Iterable[Any], name: str, count: int, check: Callable[[Any, str], Any]
) -> tuple[Any, ...]:
    # Outside the try, a TypeError raised by a generator keeps its own message.
    try:
        iterator = iter(value)
    except TypeError:
        raise TypeError(
            f"{name}: expected an iterable, got {type(value).__name__}"
        ) from None
    items = tuple(iterator)
    if len(items) != count:
        raise ValueError(f"{name}: expected {count} values, got {len(items)}")
    return tuple(check(item, f"{name}[{i}]") for i, item in enumerate(items))
