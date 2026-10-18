import logging
import math
import os
import time
import zipfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import KW_ONLY, dataclass, field, fields
from functools import partial
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.linalg import LinAlgError
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import eigh, issymmetric

from ._linalg import (
    cholesky,
    guard_solutions,
    guard_weights,
    low_rank_variances,
    precision,
    woodbury,
)
from ._spectra import (
    Boolean,
    Integer,
    Progress,
    Real,
    boolean,
    check_progress,
    chunked_snr,
    chunks,
    dilate,
    finite_array,
    integer,
    integers,
    interpolate,
    items,
    listing,
    numeric,
    real,
    require,
    root_mean_ivar,
    validate,
    vector,
)

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

# The arrays that load reads. save also writes gspice_version, cov{b} or the arrays
# of _LOW_RANK with the suffix b for each bin b, and the arrays of _BLOCKS for a model
# with blocks.
_REQUIRED = ("schema", "mean", "n_train", "edges", "center", "weighted")
_LOW_RANK = ("vectors", "values", "diag")
_BLOCKS = ("block_starts", "block_width", "block_nguard", "block_thresholds")

_logger = logging.getLogger("gspice")

# The largest nguard of a Blocks, which save writes as int64.
NGUARD_MAX = 2**63 - 1

# The path-like objects of the Python glossary, which os.fsdecode accepts.
FilePath = str | bytes | os.PathLike[str] | os.PathLike[bytes]


def rebuild(instance: "DataclassInstance") -> tuple[Callable[[], object], tuple[()]]:
    # The __reduce__ of the frozen classes: pickle, copy and deepcopy rebuild through
    # __init__, so they check the fields again. Keywords keep a pickle valid when a
    # later version adds a field.
    init = {f.name: getattr(instance, f.name) for f in fields(instance) if f.init}
    return partial(type(instance), **init), ()


@dataclass(frozen=True, eq=False)
class Blocks:
    """Block-scan geometry and per-block thresholds (paper 3.2), from `fit_blocks`.

    Block k holds the pixels starts[k] to starts[k] + width - 1. Its statistic chi2_A
    of `fit_blocks` compares the block with its prediction from the pixels farther
    than nguard from it, and thresholds[b, k] is the threshold of chi2_A for the
    spectra of S/N bin b. The constructor takes array_like values, of an integer dtype
    for starts, and numpy scalars, and stores them as the types given here, with
    read-only copies of the arrays. Two Blocks are equal only if they are the same
    object.

    Attributes
    ----------
    starts : ndarray of int64, shape (nblock,)
        The first pixel of each of the nblock >= 1 blocks: >= 0 and strictly
        increasing.
    width : int
        The pixels per block, |A|; >= 1.
    nguard : int
        The half-width of the guard window around each block (pixels); from 0 to
        2^63 - 1, so that `save` can write it.
    thresholds : ndarray, shape (nbin, nblock)
        The threshold of chi2_A of each bin and block (unit-free): finite and > 0.

    Raises
    ------
    TypeError
        If a field has a type or dtype that is not allowed above. The message starts
        with the name of the field.
    ValueError
        If a field has a shape or a value that is not allowed above. The message
        starts with the name of the field.
    """

    starts: NDArray[np.int64]
    width: int
    nguard: int
    thresholds: NDArray[np.float64]

    # Written by hand, as in Model.
    def __init__(
        self,
        starts: ArrayLike,
        width: Integer,
        nguard: Integer,
        thresholds: ArrayLike,
    ) -> None:
        starts = integers(starts, "starts")
        if starts.ndim != 1 or not starts.size:
            raise ValueError(
                f"starts: expected a 1-D array of at least one start, got shape "
                f"{starts.shape}"
            )
        # The range of int64, checked before the cast.
        ok = (starts >= 0) & (starts <= np.iinfo(np.int64).max)
        require(ok, starts, "starts", "values from 0 to 2^63 - 1", ("block",))
        starts = starts.astype(np.int64)
        _increasing(starts.tolist(), "starts")
        width = integer(width, "width", 1)
        nguard = integer(nguard, "nguard", 0, NGUARD_MAX)
        thresholds = numeric(thresholds, "thresholds")
        if thresholds.ndim != 2 or thresholds.shape[1] != starts.size:
            raise ValueError(
                f"thresholds: expected shape (nbin, {starts.size}), a column for each "
                f"start, got {thresholds.shape}"
            )
        thresholds = finite_array(thresholds, "thresholds", ("bin", "block"))
        require(
            thresholds > 0, thresholds, "thresholds", "values > 0", ("bin", "block")
        )
        for array in (starts, thresholds):
            array.flags.writeable = False
        for name, value in (
            ("starts", starts),
            ("width", width),
            ("nguard", nguard),
            ("thresholds", thresholds),
        ):
            object.__setattr__(self, name, value)

    __reduce__ = rebuild


@dataclass(frozen=True, eq=False)
class LowRank:
    """Low-rank plus diagonal covariance C = V diag(values) V^T + diag(diag).

    `fit` with rank=K returns a `Model` with a LowRank covariance. A model computes its
    precision matrix by the Woodbury identity, without forming C, and
    rejects it where kappa_W > 1e13, with c_i = diag_i + sum_k values_k V_ik^2 the
    variances of C, dtilde = diag / c, Vtilde = diag(c)^-1/2 V and

        kappa_W = (max dtilde + lambda_max(diag(values)^1/2 Vtilde^T Vtilde
                   diag(values)^1/2)) / min dtilde,

    a bound on the condition number of C scaled to unit diagonal and on max c_i/diag_i.
    The constructor takes array_like values and stores read-only float64 copies. Two
    LowRank are equal only if they are the same object.

    Attributes
    ----------
    vectors : ndarray, shape (npix, K)
        The K >= 1 columns of V, for npix >= 1 pixels: finite. `fit` returns
        orthonormal columns (unit-free), in decreasing order of values.
    values : ndarray, shape (K,)
        The variance of each column (scaled^2; flux^2 for a noise model): finite and
        > 0.
    diag : ndarray, shape (npix,)
        The diagonal term (scaled^2; flux^2 for a noise model): finite and >= 0;
        `Model` requires > 0, except for a noise model.

    Raises
    ------
    TypeError
        If a field has a dtype that is not integer or floating. The message starts
        with the name of the field.
    ValueError
        If a field has a shape or a value that is not allowed above. The message
        starts with the name of the field.
    """

    vectors: NDArray[np.float64]
    values: NDArray[np.float64]
    diag: NDArray[np.float64]

    # Written by hand, as in Model.
    def __init__(self, vectors: ArrayLike, values: ArrayLike, diag: ArrayLike) -> None:
        vectors = numeric(vectors, "vectors")
        if vectors.ndim != 2 or 0 in vectors.shape:
            raise ValueError(
                f"vectors: expected a 2-D array of shape (npix, K) with npix >= 1 and "
                f"K >= 1, got shape {vectors.shape}"
            )
        vectors = finite_array(vectors, "vectors", ("pixel", "vector"))
        npix, rank = vectors.shape
        values = vector(values, "values", rank, "vector")
        require(values > 0, values, "values", "values > 0", ("vector",))
        diag = vector(diag, "diag", npix, "pixel")
        require(diag >= 0, diag, "diag", "diag >= 0", ("pixel",))
        for name, array in (("vectors", vectors), ("values", values), ("diag", diag)):
            array.flags.writeable = False
            object.__setattr__(self, name, array)

    __reduce__ = rebuild


@dataclass(frozen=True, eq=False)
class Model:
    """Fitted Gaussian model of spectra on a common pixel grid.

    A spectrum in S/N bin b has the working vector x ~ N(0, cov[b]) in scaled units.
    Let ftilde be the flux of the spectrum with its bad pixels interpolated, and a the
    square root of the mean inverse variance of its good pixels (see `scale`). Then
    x = a (ftilde - center) - mean[b] for a constant centre, and
    x = a ftilde - mean[b] otherwise.

    A noise model (noise=True) has one bin and works in flux units: spectrum s has

        x = ftilde - mean[0] ~ N(0, C_s),  C_s = cov[0] + diag(nu_s),

    where cov[0] is the covariance of the noise-free flux, and nu_si = 1/ivar_si at the
    good pixels and 1/a^2 at the bad pixels of the spectrum. The constructor checks only
    the diagonal of cov[0]; prediction rejects a C_s that is not positive definite
    where it factors C_s, and the prior factors only C_s at the target pixels.

    The constructor takes the attributes below as array_like values, iterables and
    numpy scalars, and stores them as the types given here, with read-only float64
    copies of the arrays and new `LowRank` covariances. The attributes after n_train
    are keyword-only. The constructor does not check that a covariance is positive
    definite, which costs npix^3. Two models are equal only if they are the same
    object.

    A model computes the precision matrix of a bin at its first use and keeps it, with
    the guard weights of the latest nguard of each bin (see `pixelwise`). Each matrix
    takes npix^2 x 8 bytes, 46 MB at npix 2400, also for a `LowRank` covariance.
    Pickle, copy and deepcopy start without them.

    Attributes
    ----------
    mean : ndarray, shape (nbin, npix)
        The vector mean[b] of each of the nbin >= 1 S/N bins (scaled; flux for a noise
        model), subtracted to give x. npix >= 1 is the number of pixels. Finite.
    cov : tuple of ndarray, each shape (npix, npix), or LowRank
        The covariance of x in each bin (scaled^2), or of the noise-free flux of a
        noise model (flux^2): nbin finite, exactly symmetric matrices with a positive
        diagonal, or nbin `LowRank` covariances of npix pixels with diag > 0. All bins
        hold one kind. For a noise model, a diagonal >= 0, or a `LowRank` with
        diag >= 0; a positive semi-definite cov[0] makes every C_s positive definite.
    n_train : tuple of int
        The number N_s >= 1 of training spectra in each bin (nbin integers).
    edges : tuple of float
        The nbin - 1 interior S/N edges between the bins: finite and strictly
        increasing. Spectrum s with signal-to-noise ratio snr_s (see `snr`) is in the
        bin np.searchsorted(edges, snr_s, side="right").
    center : float or None
        None for a mean-centred model, or the constant centre c (flux); None for a
        noise model.
    noise : bool
        True for a noise model, which has one bin and center None.
    weighted : bool
        True if the model was fitted with positive weights that are not all equal in
        a bin.
    blocks : Blocks or None
        The blocks and thresholds of `fit_blocks`, with a row of thresholds for each
        bin and every block within the npix pixels; None for a model without them.

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
    cov: tuple[NDArray[np.float64] | LowRank, ...]
    n_train: tuple[int, ...]
    _: KW_ONLY
    edges: tuple[float, ...]
    center: float | None
    noise: bool
    weighted: bool
    blocks: Blocks | None
    _cache: dict[tuple[str, int], Any] = field(init=False, repr=False)
    # The name of cov[0] in messages. It is set only on the model of one spectrum of a
    # noise model, which it marks for the low-rank route of the infills.
    _label: str | None = field(init=False, repr=False)

    # Written by hand because the parameters take more than the fields store: any
    # iterable, array_like values and numpy scalars. The fields have no defaults, so
    # a field that __init__ does not set raises AttributeError instead of reading a
    # class default. _: KW_ONLY keeps the fields after n_train out of __match_args__.
    # A dataclass subclass would generate an __init__ without these checks.
    def __init__(
        self,
        mean: ArrayLike,
        cov: Iterable[ArrayLike | LowRank],
        n_train: Iterable[Integer],
        *,
        edges: Iterable[Real] = (),
        center: Real | None = None,
        noise: Boolean = False,
        weighted: Boolean = False,
        blocks: Blocks | None = None,
    ) -> None:
        mean = numeric(mean, "mean")
        if mean.ndim != 2 or 0 in mean.shape:
            raise ValueError(
                f"mean: expected a 2-D array of shape (nbin, npix) with nbin >= 1 and "
                f"npix >= 1, got shape {mean.shape}"
            )
        mean = finite_array(mean, "mean", ("bin", "pixel"))
        nbin, npix = mean.shape
        noise = boolean(noise, "noise")
        cov = items(cov, "cov", nbin, partial(_covariance, npix=npix, noise=noise))
        low = [isinstance(C, LowRank) for C in cov]
        if not all(low) and any(low):
            other = low.index(not low[0])
            kind = ["dense", "LowRank"]
            raise ValueError(
                f"cov: expected covariances of one kind, all dense or all LowRank, got "
                f"a {kind[low[0]]} cov[0] and a {kind[low[other]]} cov[{other}]"
            )
        n_train = items(n_train, "n_train", nbin, partial(integer, minimum=1))
        edges = items(edges, "edges", nbin - 1, real)
        _increasing(edges, "edges")
        center = None if center is None else real(center, "center")
        if noise and (nbin > 1 or center is not None):
            raise ValueError(
                f"noise: expected a model of one bin and center None for True, got nbin "
                f"{nbin} and center {center}"
            )
        weighted = boolean(weighted, "weighted")
        if blocks is not None:
            blocks = _check_blocks(blocks, nbin, npix)
        mean.flags.writeable = False
        for name, value in (
            ("mean", mean),
            ("cov", cov),
            ("n_train", n_train),
            ("edges", edges),
            ("center", center),
            ("noise", noise),
            ("weighted", weighted),
            ("blocks", blocks),
            ("_cache", {}),
            ("_label", None),
        ):
            object.__setattr__(self, name, value)

    @property
    def _dense(self) -> bool:
        # All covariances are of one kind.
        return not isinstance(self.cov[0], LowRank)

    @property
    def _derived(self) -> bool:
        # The kinds of model for which f_s is derived (design section 4.3, A13).
        return (
            self._dense and self.center is None and not self.weighted and not self.noise
        )

    __reduce__ = rebuild

    def _copy(self, **changes: Any) -> "Model":
        # This model with the fields in changes, without the checks of __init__; it
        # shares the other fields, read-only, and starts with an empty cache unless
        # changes gives one.
        model = object.__new__(Model)
        for f in fields(self):
            object.__setattr__(model, f.name, getattr(self, f.name))
        for name, value in ({"_cache": {}} | changes).items():
            object.__setattr__(model, name, value)
        return model

    def _with_blocks(self, blocks: Blocks) -> "Model":
        # This model with the blocks of fit_blocks, which fit it, and a copy of the
        # cache, because P and the guard weights depend only on cov.
        return self._copy(blocks=blocks, _cache=dict(self._cache))

    def _with_noise(self, nu: NDArray[np.float64], label: str) -> "Model":
        # The model of C_s = cov[0] + diag(nu) for one spectrum of a noise model
        # (design section 4.1), which label names in messages. It goes only to the
        # functions that condition, never to working, validated or the factor f_s:
        # with noise False it would pass for a mean-centred model in scaled units.
        # nu > 0 keeps the diagonal positive, as __init__ requires.
        C = self.cov[0]
        noisy: NDArray[np.float64] | LowRank
        if isinstance(C, LowRank):
            noisy = LowRank(C.vectors, C.values, C.diag + nu)
        else:
            noisy = C.copy()
            noisy.flat[:: len(nu) + 1] += nu
            noisy.flags.writeable = False
        return self._copy(cov=(noisy,), noise=False, _label=label)

    def _name(self, b: int) -> str:
        # cov[b] in messages.
        return f"model: cov[{b}]" if self._label is None else self._label

    def _precision(self, b: int, *, subject: str | None = None) -> NDArray[np.float64]:
        P: NDArray[np.float64] | None = self._cache.get(("precision", b))
        if P is None:
            start = time.perf_counter()
            try:
                C = self.cov[b]
                if isinstance(C, LowRank):
                    P = woodbury(C.vectors, C.values, C.diag)
                else:
                    P = precision(C)
            except LinAlgError as error:
                where = self._name(b) if subject is None else subject
                raise LinAlgError(f"{where} is {error}") from None
            _logger.debug(
                "cov[%d]: precision matrix in %.3f s", b, time.perf_counter() - start
            )
            # Read-only, like the fields: a write would change every later result.
            P.flags.writeable = False
            self._cache["precision", b] = P
        return P

    def _conditioner(
        self, b: int, C: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
        # (P, None) where a step of the call formed P of C, the dense cov[b], else
        # (None, L) with L its lower Cholesky factor and the check of P: the infills
        # of the model of one spectrum of a noise model (design section 12, (e)).
        P: NDArray[np.float64] | None = self._cache.get(("precision", b))
        if P is not None:
            return P, None
        L: NDArray[np.float64] | None = self._cache.get(("factor", b))
        if L is None:
            start = time.perf_counter()
            try:
                L = cholesky(C)
            except LinAlgError as error:
                raise LinAlgError(f"{self._name(b)} is {error}") from None
            _logger.debug(
                "cov[%d]: Cholesky factor in %.3f s", b, time.perf_counter() - start
            )
            L.flags.writeable = False
            self._cache["factor", b] = L
        return None, L

    def _bins(
        self, flux: NDArray[Any], ivar: NDArray[Any], bad: NDArray[np.bool_]
    ) -> NDArray[np.intp]:
        # A model with one bin skips the S/N, which sorts every row.
        if len(self.cov) == 1:
            return np.zeros(len(flux), dtype=np.intp)
        return np.searchsorted(self.edges, chunked_snr(flux, ivar, bad), side="right")

    def _guard_solutions(
        self, b: int, nguard: int, *, subject: str | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        # V and var of guard_solutions, kept for the model of one spectrum of a noise
        # model, whose pixelwise passes reuse them.
        cached = self._cache.get(("solutions", b))
        if cached is not None and cached[0] == nguard:
            return cached[1], cached[2]
        V, var = self._solve_guards(b, nguard, subject)
        self._cache["solutions", b] = nguard, V, var
        return V, var

    def _solve_guards(
        self, b: int, nguard: int, subject: str | None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        # V and var of guard_solutions, for nguard < npix - 1, read-only.
        P = self._precision(b, subject=subject)
        start = time.perf_counter()
        try:
            V, var = guard_solutions(P, variances_of(self.cov[b]), nguard)
        except LinAlgError as error:
            where = self._name(b) if subject is None else subject
            raise LinAlgError(f"{where} is {error}") from None
        _logger.debug(
            "cov[%d]: guard solutions for nguard %d in %.3f s",
            b,
            nguard,
            time.perf_counter() - start,
        )
        for array in (V, var):
            array.flags.writeable = False
        return V, var

    def _guard_weights(
        self, b: int, nguard: int, *, subject: str | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        cached = self._cache.get(("guard", b))
        if cached is not None and cached[0] == nguard:
            return cached[1], cached[2]
        npix = self.mean.shape[1]
        if nguard < npix - 1:
            V, var = self._solve_guards(b, nguard, subject)
            start = time.perf_counter()
            W = guard_weights(self._precision(b), V)
        else:
            # Every window covers all pixels: the exact prior, without P (design
            # section 4.2).
            start = time.perf_counter()
            W, var = np.zeros((npix, npix)), variances_of(self.cov[b])
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


# The operations on a covariance C of a model, dense or LowRank.


def variances_of(C: NDArray[np.float64] | LowRank) -> NDArray[np.float64]:
    # The diagonal of C, a new array.
    if isinstance(C, LowRank):
        return C.diag + low_rank_variances(C.vectors, C.values)
    return np.diagonal(C).copy()


def submatrix_of(
    C: NDArray[np.float64] | LowRank, pixels: NDArray[np.intp]
) -> NDArray[np.float64]:
    # C at the pixels, a new array, without forming a LowRank C elsewhere.
    if not isinstance(C, LowRank):
        return C[np.ix_(pixels, pixels)]
    B = C.vectors[pixels] * np.sqrt(C.values)
    # numpy computes B B^T by syrk, so the block is exactly symmetric. Its diagonal
    # is that of variances_of, which the guard weights use.
    S: NDArray[np.float64] = B @ B.T
    np.fill_diagonal(S, variances_of(C)[pixels])
    return S


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
    rank: Integer | None = None,
    noise: Boolean = False,
    snr_bins: Integer | Iterable[Real] | None = None,
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
    x_s = a_s (ftilde_s - c) and S = sum_s w_s x_s x_s^T / V1. With rank=K, the
    covariance is the `LowRank` C = V diag(lambda) V^T + diag(d), with (V, lambda) the
    K leading eigenpairs of S and d = diag(S) - sum_k lambda_k V_k^2, so that C and S
    have the same diagonal.

    Pass n of N = len(nsigma) fits on its mask M_n with unit weights and the rank,
    computes the in-sample z of `pixelwise` with nguard at every pixel, interpolated
    ones included, and flags F_n: each pixel with |z| > nsigma[n - 1] and its two
    neighbours. M_1 = bad. With cumulative=True (paper Alg. 4), M_{n+1} = M_n | F_n
    and the final mask is M_{N+1}; with cumulative=False (2022), M_{n+1} = bad | F_n
    and the final mask is bad | F_{N-1} | F_N. nsigma=() fits on the mask bad, without
    clipping.
    The final fit uses the final mask and the weights w_s; reweighting the clipping
    raised false flags in the paper (l.548).

    With noise=True, the passes clip as above, and the final fit estimates the
    covariance of the noise-free flux in flux units (design section 4.1). Let u_si be
    1 at the pixels outside the final mask and 0 elsewhere, and v_s = a_s^2 w_s. Then

        W_i = sum_s v_s u_si,  mu_i = sum_s v_s u_si f_si / W_i,
        A_si = v_s^1/2 u_si (f_si - mu_i),  D1_ij = sum_s v_s u_si u_sj,
        D2_ij = sum_s v_s^2 u_si u_sj,  N_i = sum_s v_s u_si (1 - v_s / W_i) / ivar_si,
        Sigma_ij = ((A^T A)_ij - delta_ij N_i)
                   / (D1_ij - D2_ij (1/W_i + 1/W_j - D1_ij / (W_i W_j))).

    Sigma is unbiased element by element if the spectra are independent with a common
    mean and signal covariance, the noise has zero mean and variance 1/ivar and is
    uncorrelated between pixels and with the signal, and the masks, weights and ivar
    do not depend on the flux; the outlier flags of the final mask break the last
    condition. The model holds mean[0] = mu and cov[0] = Q max(Lambda, 0) Q^T, with
    Sigma = Q Lambda Q^T its eigendecomposition, which is biased upward. With rank=K,
    cov[0] is the `LowRank` C = V diag(lambda) V^T + diag(d), with (V, lambda) the K
    leading eigenpairs of Sigma and d_i = sum_{k>K} max(lambda_k, 0) Q_ik^2, so that C
    and Q max(Lambda, 0) Q^T have the same diagonal.

    With snr_bins, spectrum s goes to the S/N bin np.searchsorted(edges, snr_s,
    side="right"), with snr_s from `snr` on flux, ivar and mask. Each bin runs the
    passes and the final fit above on its own spectra, so bin b of the model equals
    the model of fit on the spectra of bin b alone, with their own clipping, m, S and
    N_s. Each bin clips with its own N_s, so the in-sample ceiling of |z| falls in
    small bins, and they flag fewer outliers (design decision A6).

    With cumulative=False, fit equals legacy covar_iter_mask, the covariance to
    rounding and the mask exactly, if nguard is 20, center, weights, rank and snr_bins
    are None, noise is False, flux is float64, ivar == 0 only at pixels in mask,
    nsigma is not empty, no |z| lies within the error of the 2022 Schur updates of a
    threshold, no pass mask or final mask covers completely a spectrum with at most
    maxbadpix bad pixels, and fit raises no error.

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
    rank : int, optional
        The rank K of a low-rank model, used by every pass; from 1 to npix - 1. None:
        a dense model.
    noise : bool, optional
        If True, fit a noise model (see `Model`), with center and snr_bins None.
    snr_bins : int or sequence of float, optional
        The S/N bins: their number n >= 1, with the interior edges at the 1/n, ...,
        (n - 1)/n quantiles (np.quantile, linear) of snr_s of the spectra that fit uses
        (at most maxbadpix bad pixels, a weight > 0 and a good pixel); or the interior
        edges, finite and strictly increasing. None: one bin.
    progress : callable, optional
        progress(stage, done, total), called after each chunk of spectra of each pass
        and of the final fit, with the stage "fit", the number of spectra of that fit
        done, and their total N_s; with S/N bins, each fit of a bin counts the
        spectra of that bin.

    Returns
    -------
    model : Model
        The model, with the edges, N_s of each bin in n_train, the centre, weighted
        True if the weights of the spectra of a bin in the final fit are not all equal,
        and `LowRank` covariances with rank=K. It holds its precision matrices (see
        `Model`), except a noise model, whose prediction forms one per spectrum. For a
        noise model, weighted is True if the weights w_s are not all equal, whatever
        a_s.
    mask : ndarray of bool, shape (nspec, npix)
        The final mask, and True at every pixel of a spectrum that is not used.

    Raises
    ------
    TypeError
        If an argument has a type or a dtype that is not allowed above.
    ValueError
        If an argument has a shape or a value that is not allowed above, or if a fit
        (of a bin, with snr_bins) lacks the spectra it needs: for a dense model,
        N_s - 1 >= npix if it is mean-centred, N_s >= npix for a constant centre, and
        a good pixel in one of them at every pixel; with rank=K, N_s - 1 >= K + 1 if
        it is mean-centred and N_s >= K + 1 for a constant centre, for S of rank
        K + 1 or more. With rank=K, also if lambda_K <= 0 or a d_i <= 0. If noise is
        True with a centre or snr_bins. With noise=True, the passes need the spectra
        above, and the final fit raises instead if fewer than two spectra are good at
        a pixel, if a denominator of Sigma is <= 0, or with rank=K if lambda_K <= 0
        for Sigma. With two spectra at each pixel, a denominator is <= 0 where no
        spectrum is good at both pixels of a pair, or by rounding near 0. The message
        names the pass or the final fit, the bin and its edges, the counts, the first
        such pixels or pairs of pixels, or lambda_K. fit checks the counts of every
        bin before any bin clips.
    numpy.linalg.LinAlgError
        If a covariance has a variance of 0, or fails the check of its precision
        matrix: it is not positive definite, or the reciprocal condition number of the
        covariance scaled to unit diagonal is below 1e-13, which can happen for N_s - 1
        close to npix; with rank=K, kappa_W of `LowRank` is above 1e13. Clipping checks
        P only if nguard < npix - 1. The message names the pass or the final fit, the
        bin and N_s, and the pixel of variance 0, the reciprocal condition number below
        1e-13 or kappa_W. With noise=True, only for the clipping covariances.

    Notes
    -----
    Cost per pass and bin: about N_s npix^2 flops for the covariance, npix^3 for its
    precision matrix, npix (w^3/3 + 2 npix w) for the guard weights,
    w = min(2 nguard + 1, npix), and 2 N_s npix^2 for z, with N_s of the bin. The
    final fit costs about N_s npix^2 + npix^3 flops per bin. With nbin > 1, the S/N
    sorts the npix values of each spectrum once. With rank=K, each fit adds about
    (4/3) npix^3 flops for the K leading eigenpairs of S, more than the npix^3 of a
    dense P, and P then costs about npix^2 K flops. Each fit holds the working vectors
    of its spectra, N_s npix x 8 bytes. With noise=True, the final fit costs about
    3 N_s npix^2 flops for A^T A, D1 and D2 and a full eigendecomposition of an
    npix x npix matrix, also with rank=K, and holds a few arrays of one chunk of
    max(1, 2**24 // npix) spectra instead of the working vectors.
    """
    start = time.perf_counter()
    thresholds = _thresholds(nsigma)
    maxbadpix = integer(maxbadpix, "maxbadpix", 0)
    nguard = integer(nguard, "nguard", 0)
    cumulative = boolean(cumulative, "cumulative")
    if center is not None:
        center = real(center, "center")
    if rank is not None:
        rank = integer(rank, "rank", 1)
    noise = boolean(noise, "noise")
    if snr_bins is not None:
        snr_bins = _snr_bins(snr_bins)
    if noise and (center is not None or snr_bins is not None):
        raise ValueError(
            f"noise: expected center and snr_bins None for True, got center {center} "
            f"and snr_bins {snr_bins}"
        )
    check_progress(progress)
    flux, ivar, bad = validate(flux, ivar, mask)
    nspec, npix = flux.shape
    if rank is not None and rank >= npix:
        raise ValueError(
            f"rank: expected an integer < {npix}, the pixels of flux, got {rank}"
        )
    if weights is None:
        weights = np.ones(nspec)
    else:
        weights = vector(weights, "weights", nspec, "row")
        require(weights >= 0, weights, "weights", "values >= 0", ("row",))
    over = np.count_nonzero(bad, axis=1) > maxbadpix
    zero = ~over & (weights == 0)
    rows = np.flatnonzero(~over & ~zero & ~bad.all(axis=1))
    edges, bins = _assign(flux, ivar, bad, rows, snr_bins)
    nbin = len(edges) + 1
    # Each bin runs the whole procedure on its own spectra (design decision A6).
    spectra = [rows[bins[rows] == b] for b in range(nbin)]
    names = [_of(b, edges) for b in range(nbin)]
    removed = [_removed(bins == b, over, zero) for b in range(nbin)]
    # The check of pass 1 of every bin before any bin clips, as validation.
    first = "pass 1" if thresholds else "the final fit"
    # The noise-free estimator checks its own requirements.
    if thresholds or not noise:
        for group, of, counts in zip(spectra, names, removed, strict=True):
            _check(bad[group], center, rank, first + of, counts)
    clip = partial(
        _clip,
        flux,
        ivar,
        bad,
        thresholds=thresholds,
        cumulative=cumulative,
        center=center,
        rank=rank,
        nguard=nguard,
        progress=progress,
    )
    masks, models = [], []
    for group, of, counts in zip(spectra, names, removed, strict=True):
        used, M = clip(rows=group, removed=counts, of=of)
        masks.append((used, M))
        models.append(
            _final(
                flux,
                ivar,
                used,
                M,
                weights=weights,
                center=center,
                rank=rank,
                noise=noise,
                removed=counts,
                of=of,
                progress=progress,
            )
        )
    if edges:
        model = Model(
            np.concatenate([single.mean for single in models]),
            [single.cov[0] for single in models],
            [single.n_train[0] for single in models],
            edges=edges,
            center=center,
            weighted=any(single.weighted for single in models),
        )
        # The precision matrices that _final checked, shared as in _with_blocks.
        for b, single in enumerate(models):
            model._cache["precision", b] = single._precision(0)
    else:
        model = models[0]
    # After the fits, which hold their working vectors.
    training = np.ones((nspec, npix), dtype=bool)
    for used, M in masks:
        training[used] = M
    _logger.debug("fit: %d spectra in %.3f s", nspec, time.perf_counter() - start)
    return model, training


def _clip(
    flux: NDArray[Any],
    ivar: NDArray[Any],
    bad: NDArray[np.bool_],
    *,
    thresholds: tuple[float, ...],
    cumulative: bool,
    center: float | None,
    rank: int | None,
    nguard: int,
    progress: Progress | None,
    rows: NDArray[np.intp],
    removed: tuple[int, int, int],
    of: str,
) -> tuple[NDArray[np.intp], NDArray[np.bool_]]:
    # The clipping passes of the spectra rows, with unit weights (decision A23): the
    # rows that the final fit uses and their final mask. of names the bin.
    M = bad[rows]
    n_cond = max(bad.shape[1] - 2 * nguard - 1, 0)
    for n, threshold in enumerate(thresholds, 1):
        _check(M, center, rank, f"pass {n}{of}", removed)
        n_train = rows.size
        # In a spectrum of typical leverage, the in-sample |z| of an outlier of any
        # amplitude approaches this ceiling of a dense model (design section 4.1). No
        # ceiling is derived for a low-rank model.
        if rank is None:
            if center is None:
                ceiling = math.sqrt((n_train - 1) * (n_train - n_cond - 1) / n_train)
            else:
                ceiling = math.sqrt(n_train - n_cond)
            if threshold > ceiling:
                _logger.warning(
                    "fit: pass %d%s has nsigma %g above %.3g, the in-sample ceiling "
                    "of |z| at N_s = %d",
                    n,
                    of,
                    threshold,
                    ceiling,
                    n_train,
                )
        subject = f"fit: the clipping covariance of pass {n}{of} (N_s = {n_train})"
        z = _clipping_z(flux, ivar, rows, M, center, rank, nguard, subject, progress)
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
            "fit: pass %d%s with nsigma %g flagged %.3g%% of the pixels of its %d "
            "spectra; spectra dropped: %d",
            n,
            of,
            threshold,
            100 * np.count_nonzero(flagged) / flagged.size,
            n_train,
            n_train - np.count_nonzero(keep),
        )
        rows, M = rows[keep], M[keep]
    return rows, M


def _final(
    flux: NDArray[Any],
    ivar: NDArray[Any],
    rows: NDArray[np.intp],
    M: NDArray[np.bool_],
    *,
    weights: NDArray[np.float64],
    center: float | None,
    rank: int | None,
    noise: bool,
    removed: tuple[int, int, int],
    of: str,
    progress: Progress | None,
) -> Model:
    # The final fit of the spectra rows on their final mask M, with the weights: a
    # model of one bin that holds its checked P, except a noise model, which never
    # uses the P of its noise-free covariance. of names the bin.
    fitted = f"the final fit{of}"
    if noise:
        # The requirement of a denominator > 0 on the diagonal of _noise_free, before
        # np.max of the weights, which needs a spectrum.
        few = np.flatnonzero(np.count_nonzero(~M, axis=0) < 2)
        if few.size:
            raise ValueError(
                f"fit: {fitted} needs two spectra good at every pixel for the "
                f"noise-free covariance, got fewer than two of its {rows.size} "
                f"spectra, on the final mask, at pixels {listing(few, 'pixels')}: "
                f"{_counts(removed, rows.size)}"
            )
    else:
        _check(M, center, rank, fitted, removed)
    n_train = rows.size
    # Only the ratios of the weights matter. In units of the largest, V2 neither
    # overflows nor underflows, and equal weights are exactly 1.
    w = weights[rows] / np.max(weights[rows])
    subject = f"fit: the final covariance{of} (N_s = {n_train})"
    estimator = ""
    if noise:
        mean, S, v = _noise_free(flux, ivar, rows, M, w, fitted, progress)
        model = Model(
            mean[np.newaxis],
            (_noise_covariance(S, rank, subject),),
            (n_train,),
            noise=True,
            weighted=np.any(w != 1),
        )
        # The weights of the estimator, a_s^2 w_s, set the effective count, also
        # where the weights of the user are equal.
        w, estimator = v, " for the weights a_s^2 w_s of the noise-free estimator"
    else:
        _, model = _fitted(flux, ivar, rows, M, center, rank, w, subject, progress)
        model._precision(0, subject=subject)
    if np.any(w != w[0]):
        _logger.info(
            "fit: %s of %d spectra is weighted, with V1^2/V2 = %.4g effective spectra%s",
            fitted,
            n_train,
            w.sum() ** 2 / np.vdot(w, w),
            estimator,
        )
    return model


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


def _snr_bins(value: Any) -> int | tuple[float, ...]:
    # The interior edges, or the number of bins (design section 8).
    if isinstance(value, Iterable):
        edges = items(value, "snr_bins", None, real)
        _increasing(edges, "snr_bins")
        return edges
    return integer(value, "snr_bins", 1)


def _assign(
    flux: NDArray[Any],
    ivar: NDArray[Any],
    bad: NDArray[np.bool_],
    rows: NDArray[np.intp],
    snr_bins: int | tuple[float, ...] | None,
) -> tuple[tuple[float, ...], NDArray[np.intp]]:
    # The interior edges and the bin of every spectrum (design section 4.1, S/N
    # statistic). A spectrum without a good pixel has the S/N inf of chunked_snr, so it
    # goes to the last bin, where it counts among the spectra that maxbadpix, weight 0
    # or full masks removed.
    if snr_bins is None or snr_bins == 1:
        return (), np.zeros(len(flux), dtype=np.intp)
    ratio = chunked_snr(flux, ivar, bad)
    if isinstance(snr_bins, int):
        # Equal-count bins of the spectra of the first pass (decision A5).
        q = np.arange(1, snr_bins) / snr_bins
        edges = tuple(np.quantile(ratio[rows], q).tolist()) if rows.size else ()
    else:
        edges = snr_bins
    return edges, np.searchsorted(edges, ratio, side="right")


def _of(b: int, edges: tuple[float, ...]) -> str:
    # The name of bin b in messages; none for a model of one bin.
    if not edges:
        return ""
    if b == 0:
        where = f"snr < {edges[0]}"
    elif b == len(edges):
        where = f"snr >= {edges[-1]}"
    else:
        where = f"{edges[b - 1]} <= snr < {edges[b]}"
    return f" of S/N bin {b} ({where})"


def _removed(
    in_bin: NDArray[np.bool_], over: NDArray[np.bool_], zero: NDArray[np.bool_]
) -> tuple[int, int, int]:
    # The spectra of a bin, and those of them that maxbadpix and weight 0 removed.
    return (
        int(np.count_nonzero(in_bin)),
        int(np.count_nonzero(in_bin & over)),
        int(np.count_nonzero(in_bin & zero)),
    )


def _check(
    M: NDArray[np.bool_],
    center: float | None,
    rank: int | None,
    fitted: str,
    removed: tuple[int, int, int],
) -> None:
    # The requirements of design section 4.1 for the fit whose mask is M, before its
    # covariance. S has rank N_s - 1 or less (N_s for a constant centre). With rank K,
    # lambda_K > 0 and d > 0 need S of rank K + 1 or more, because S = V Lambda V^T and
    # d = 0 where its rank is K; _low_rank checks them.
    n_train, npix = M.shape
    size = npix if rank is None else rank + 1
    needed = size + 1 if center is None else size
    if n_train < needed:
        kind = "a mean-centred" if center is None else "a constant-centre"
        of = f"{npix} pixels" if rank is None else f"rank {rank}"
        raise ValueError(
            f"fit: {fitted} needs at least {needed} spectra for {kind} model of "
            f"{of}, got {n_train}: {_counts(removed, n_train)}"
        )
    # A pixel bad in every spectrum makes a dense S singular. The low-rank estimator
    # has no such requirement (design section 4.1).
    if rank is not None:
        return
    empty = np.flatnonzero(M.all(axis=0))
    if empty.size:
        raise ValueError(
            f"fit: {fitted} needs a good pixel in one of its {n_train} spectra at "
            f"every pixel, got none at pixels {listing(empty, 'pixels')}"
        )


def _counts(removed: tuple[int, int, int], n_train: int) -> str:
    # The spectra that a fit of n_train spectra lost, for messages.
    nspec, over, zero = removed
    full = nspec - over - zero - n_train
    return (
        f"of {nspec} spectra, maxbadpix removed {over}, weight 0 removed {zero} and "
        f"full masks removed {full}"
    )


def _clipping_z(
    flux: NDArray[Any],
    ivar: NDArray[Any],
    rows: NDArray[np.intp],
    M: NDArray[np.bool_],
    center: float | None,
    rank: int | None,
    nguard: int,
    subject: str,
    progress: Progress | None,
) -> NDArray[np.float64]:
    # The in-sample z of the spectra rows, whose mask is M, at every pixel under the
    # clipping model of the rank with unit weights (design section 4.1, decisions A12
    # and A23). With unit weights, X holds the working vectors.
    w = np.ones(rows.size)
    X, model = _fitted(flux, ivar, rows, M, center, rank, w, subject, None)
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
    rank: int | None,
    w: NDArray[np.float64],
    subject: str,
    progress: Progress | None,
) -> tuple[NDArray[np.float64], Model]:
    # The model of design section 4.1 on the spectra rows, whose mask is M, with the
    # rank and the weights w, and the working vectors x_s times sqrt(w_s).
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
    C = S if rank is None else _low_rank(S, rank, subject)
    weighted = np.any(w != 1)
    model = Model(mean[np.newaxis], (C,), (nspec,), center=center, weighted=weighted)
    return X, model


def _noise_free(
    flux: NDArray[Any],
    ivar: NDArray[Any],
    rows: NDArray[np.intp],
    M: NDArray[np.bool_],
    weights: NDArray[np.float64],
    fitted: str,
    progress: Progress | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    # The noise-free estimator of design section 4.1 (D-63) on the spectra rows, whose
    # mask is M, with the weights of the user: mu and Sigma before the eigenvalue
    # floor, in flux units, and the weights w_s = a_s^2 weights_s in units of the
    # largest. u_si = 1 where M_si is False.
    nspec, npix = M.shape
    w = np.empty(nspec)
    for c in chunks(nspec, npix):
        w[c] = np.square(root_mean_ivar(ivar[rows[c]], M[c]))
    w *= weights
    # Sigma does not change when w is scaled; in units of the largest, w^2 cannot
    # overflow.
    w /= np.max(w)
    W, total = np.zeros(npix), np.zeros(npix)
    for c in chunks(nspec, npix):
        u = ~M[c]
        W += w[c] @ u
        # The flux at bad pixels is never read.
        total += w[c] @ np.where(u, flux[rows[c]].astype(np.float64, copy=False), 0)
    mean = total / W
    AtA, D1, D2 = (np.zeros((npix, npix)) for _ in range(3))
    # sum_s w_s u_si / ivar_si and sum_s w_s^2 u_si / ivar_si, for N_i.
    first, second = np.zeros(npix), np.zeros(npix)
    for c in chunks(nspec, npix):
        u = ~M[c]
        root = np.sqrt(w[c])[:, np.newaxis]
        # One buffer per chunk holds A, B, E and 1/ivar in turn. numpy computes
        # Y^T Y by syrk, so each product is exactly symmetric.
        Y = np.subtract(flux[rows[c]], mean, dtype=np.float64)
        Y[~u] = 0.0
        Y *= root
        AtA += Y.T @ Y
        np.multiply(u, root, out=Y)
        D1 += Y.T @ Y
        np.multiply(u, w[c][:, np.newaxis], out=Y)
        D2 += Y.T @ Y
        # E is 0 at the bad pixels, where 1/ivar is never read.
        np.divide(1.0, ivar[rows[c]], out=Y, where=u, dtype=np.float64)
        first += w[c] @ Y
        second += np.square(w[c]) @ Y
        if progress is not None:
            progress("fit", c.stop, nspec)
    # Float addition and multiplication commute, so the denominators are exactly
    # symmetric. In exact arithmetic, a denominator is > 0 if a spectrum is good at
    # both pixels and two spectra are good at each, and 0 otherwise; the caller
    # checked the second kind of 0, which rounding moves off 0, because D1_ii is not
    # exactly W_i. Rounding can also take a denominator near 0, at weights that
    # differ by a factor near 1/eps, to <= 0 or to a small value that passes.
    r = 1 / W
    denominator = D1 - D2 * (r[:, np.newaxis] + r - D1 * np.outer(r, r))
    pairs = np.argwhere(np.triu(~(denominator > 0)))
    if pairs.size:
        named = ", ".join(f"({i}, {j})" for i, j in pairs[:10].tolist())
        more = f", ... ({len(pairs)} pairs)" if len(pairs) > 10 else ""
        raise ValueError(
            f"fit: {fitted} needs a denominator of the noise-free covariance > 0 at "
            f"every pair of pixels, got <= 0 at the pairs {named}{more}, where no "
            f"spectrum is good at both pixels, or where weights differ by a factor "
            f"near 1/eps"
        )
    AtA[np.diag_indices(npix)] -= first - second * r
    return mean, AtA / denominator, w


def _noise_covariance(
    S: NDArray[np.float64], rank: int | None, subject: str
) -> NDArray[np.float64] | LowRank:
    # The covariance of a noise model from Sigma (design section 4.1): Sigma_+ as
    # B B^T, which numpy computes by syrk, so that it is exactly symmetric, or with
    # rank K a LowRank. d needs the floored tail, so the eigendecomposition is full.
    values, Q = eigh(S)
    if rank is None:
        B = Q * np.sqrt(np.maximum(values, 0))
        floored: NDArray[np.float64] = B @ B.T
        return floored
    tail = len(S) - rank
    leading, V = values[tail:][::-1], Q[:, tail:][:, ::-1]
    _require_lambda_k(leading, rank, subject)
    d = low_rank_variances(Q[:, :tail], np.maximum(values[:tail], 0))
    return LowRank(V, leading, d)


def _low_rank(S: NDArray[np.float64], rank: int, subject: str) -> LowRank:
    # The estimator of design section 4.1 and its requirements lambda_K > 0 and d > 0.
    npix = len(S)
    values, V = eigh(S, subset_by_index=(npix - rank, npix - 1))
    # In decreasing order of the eigenvalues.
    values, V = values[::-1], V[:, ::-1]
    _require_lambda_k(values, rank, subject)
    d = np.diagonal(S) - low_rank_variances(V, values)
    nonpositive = np.flatnonzero(d <= 0)
    if nonpositive.size:
        raise ValueError(
            f"{subject} needs d_i > 0 at every pixel for rank {rank}, got d_i <= 0 at "
            f"pixels {listing(nonpositive, 'pixels')}"
        )
    return LowRank(V, values, d)


def _require_lambda_k(values: NDArray[np.float64], rank: int, subject: str) -> None:
    # The requirement lambda_K > 0 of the K leading eigenvalues, in decreasing order.
    if not values[-1] > 0:
        raise ValueError(
            f"{subject} needs lambda_K > 0 for rank {rank}, got lambda_{rank} = "
            f"{values[-1]:.3g}"
        )


def save(model: Model, path: FilePath) -> None:
    """Write model to the .npz file path.

    The archive is not compressed. It holds the arrays schema (the integer 1),
    gspice_version, mean, n_train, edges, center (NaN if None), weighted and, for a
    noise model, noise (True); for the
    covariance of each bin b, cov{b}, or vectors{b}, values{b} and diag{b} for the
    fields of a `LowRank`; and, if model.blocks is not None, block_starts,
    block_width, block_nguard and block_thresholds for its fields. `load` reads it.

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
        edges, center and weighted, holds some but not all of the four block arrays,
        holds vectors{b} without values{b} or diag{b}, or holds an array that `save`
        does not write; the message names path. Also if `Model`, `LowRank` or
        `Blocks` rejects a field, for example a count of covariances other than the
        number of bins; the message names the field.
    TypeError
        If path is not a str or path-like, or if `Model`, `LowRank` or `Blocks`
        rejects a field.
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
    with_blocks = any(key in archive for key in _BLOCKS)
    required = [*_REQUIRED, *_BLOCKS] if with_blocks else list(_REQUIRED)
    # The covariance of bin b: cov{b}, or the fields of a LowRank.
    bins: list[list[str]] = []
    while True:
        b = len(bins)
        if f"cov{b}" in archive:
            bins.append([f"cov{b}"])
        elif f"vectors{b}" in archive:
            bins.append([f"{name}{b}" for name in _LOW_RANK])
        else:
            break
        required += bins[-1]
    missing = [key for key in required if key not in archive]
    if missing:
        raise ValueError(
            f"path: expected the arrays that save writes, got no {', '.join(missing)} "
            f"in {path}"
        )
    known = {*required, *_BLOCKS, "gspice_version", "noise"}
    unknown = sorted(set(archive.files) - known)
    if unknown:
        writer = archive.get("gspice_version")
        by = "" if writer is None else f", written by gspice {str(writer)[:40]},"
        raise ValueError(
            f"path: expected only the arrays that save writes, got "
            f"{', '.join(unknown)}{by} in {path}"
        )
    cov = []
    for b, keys in enumerate(bins):
        if len(keys) == 1:
            cov.append(archive[keys[0]])
            continue
        arrays = [archive[key] for key in keys]
        try:
            cov.append(LowRank(*arrays))
        except (TypeError, ValueError) as error:
            raise type(error)(f"cov[{b}]: {error}") from None
    # [()] returns a 0-d array as a scalar and other arrays as they are.
    blocks = Blocks(*(archive[key][()] for key in _BLOCKS)) if with_blocks else None
    center = archive["center"][()]
    # save writes noise only if it is True (design section 9).
    noise = archive["noise"][()] if "noise" in archive else False
    return Model(
        archive["mean"],
        cov,
        archive["n_train"],
        edges=archive["edges"],
        center=None if isinstance(center, float) and math.isnan(center) else center,
        noise=noise,
        weighted=archive["weighted"][()],
        blocks=blocks,
    )


def _arrays(model: Model) -> dict[str, Any]:
    # __init__ defines __version__ after it imports this module.
    from . import __version__

    arrays = {
        "schema": np.int64(1),
        "gspice_version": __version__,
        "mean": model.mean,
        "n_train": np.array(model.n_train, dtype=np.int64),
        "edges": np.array(model.edges),
        "center": np.nan if model.center is None else model.center,
        "weighted": model.weighted,
    }
    # Only if True, so that a version without noise models reads the other files
    # (design section 9).
    if model.noise:
        arrays["noise"] = True
    for b, C in enumerate(model.cov):
        if isinstance(C, LowRank):
            arrays |= {f"{name}{b}": getattr(C, name) for name in _LOW_RANK}
        else:
            arrays[f"cov{b}"] = C
    # Without blocks, none of their arrays, so that a version without blocks reads the
    # file (design section 9).
    if model.blocks is not None:
        arrays |= {
            "block_starts": model.blocks.starts,
            "block_width": np.int64(model.blocks.width),
            "block_nguard": np.int64(model.blocks.nguard),
            "block_thresholds": model.blocks.thresholds,
        }
    return arrays


def _covariance(
    value: ArrayLike | LowRank, name: str, npix: int, noise: bool
) -> NDArray[np.float64] | LowRank:
    if isinstance(value, LowRank):
        # A new LowRank, so that copy and deepcopy of a model check its fields again.
        low = LowRank(value.vectors, value.values, value.diag)
        if low.diag.size != npix:
            raise ValueError(
                f"{name}: expected a LowRank of {npix} pixels like mean, got vectors "
                f"of shape {low.vectors.shape}"
            )
        # P = D^-1 - L L^T needs d > 0 (design section 3.3), which nu > 0 gives a
        # noise model.
        if not noise:
            require(low.diag > 0, low.diag, name, "diag > 0", ("pixel",))
        # Finite fields can give infinite variances, as a dense C cannot.
        with np.errstate(over="ignore"):
            variances = variances_of(low)
        require(np.isfinite(variances), variances, name, "finite variances", ("pixel",))
        return low
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
    # A noise model adds nu > 0 to a diagonal that is >= 0 if cov[0] is positive
    # semi-definite.
    diagonal = np.diagonal(C)
    if noise:
        require(diagonal >= 0, diagonal, name, "a diagonal >= 0", ("pixel",))
    else:
        require(diagonal > 0, diagonal, name, "a positive diagonal", ("pixel",))
    # After the checks, because issymmetric of scipy 1.11 rejects read-only arrays.
    C.flags.writeable = False
    return C


def _check_blocks(blocks: object, nbin: int, npix: int) -> Blocks:
    if not isinstance(blocks, Blocks):
        raise TypeError(
            f"blocks: expected a Blocks or None, got {type(blocks).__name__}"
        )
    # A new Blocks, so that copy and deepcopy of a model check its fields again.
    blocks = Blocks(blocks.starts, blocks.width, blocks.nguard, blocks.thresholds)
    rows = len(blocks.thresholds)
    if rows != nbin:
        raise ValueError(
            f"blocks: expected a row of thresholds for each of the {nbin} bins of "
            f"mean, got {rows}"
        )
    # The starts increase, so the last block ends last. Python integers cannot
    # overflow.
    end = int(blocks.starts[-1]) + blocks.width
    if end > npix:
        raise ValueError(
            f"blocks: expected blocks within the {npix} pixels of mean, got a last "
            f"block that ends at starts[-1] + width = {end}"
        )
    return blocks


def _increasing(values: Sequence[Any], name: str) -> None:
    for i, (low, high) in enumerate(pairwise(values)):
        if low >= high:
            raise ValueError(
                f"{name}: expected strictly increasing values, "
                f"got {name}[{i}] = {low} and {name}[{i + 1}] = {high}"
            )
