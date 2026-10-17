import logging
import math
import os
import time
import zipfile
from collections.abc import Callable, Iterable
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
    boolean,
    chunks,
    finite_array,
    integer,
    median_snr,
    numeric,
    real,
    require,
)

# The arrays that load reads. save also writes gspice_version and cov0, cov1, ...
_REQUIRED = ("schema", "mean", "n_train", "edges", "center", "weighted")

_logger = logging.getLogger("gspice")


@dataclass(frozen=True, eq=False)
class Model:
    """Fitted Gaussian model of spectra on a common pixel grid.

    A spectrum in S/N bin b has the working vector x ~ N(0, cov[b]) in scaled units.
    Let ftilde be the flux of the spectrum with its bad pixels interpolated, and a the
    square root of the mean inverse variance of its good pixels (see `scale`). Then
    x = a (ftilde - center) - mean[b] for a constant centre, and
    x = a ftilde - mean[b] otherwise.

    The constructor takes the attributes below as array_like values or sequences, and
    stores them as the types given here, with read-only float64 copies of the arrays.
    The attributes after n_train are keyword-only. The constructor does not check that
    a covariance is positive definite, which costs npix^3. Two models are equal only if
    they are the same object.

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
    edges: tuple[float, ...] = ()
    center: float | None = None
    weighted: bool = False
    _cache: dict[tuple[str, int], Any] = field(
        init=False, repr=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        mean = numeric(self.mean, "mean")
        if mean.ndim != 2 or 0 in mean.shape:
            raise ValueError(
                f"mean: expected a 2-D array of shape (nbin, npix) with nbin >= 1 and "
                f"npix >= 1, got shape {mean.shape}"
            )
        mean = finite_array(mean, "mean", ("bin", "pixel"))
        nbin, npix = mean.shape
        cov = _items(self.cov, "cov", nbin, partial(_covariance, npix=npix))
        n_train = _items(self.n_train, "n_train", nbin, partial(integer, minimum=1))
        edges = _items(self.edges, "edges", nbin - 1, real)
        for i, (low, high) in enumerate(pairwise(edges)):
            if low >= high:
                raise ValueError(
                    f"edges: expected strictly increasing values, "
                    f"got edges[{i}] = {low} and edges[{i + 1}] = {high}"
                )
        center = None if self.center is None else real(self.center, "center")
        weighted = boolean(self.weighted, "weighted")
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
        ):
            object.__setattr__(self, name, value)

    def __reduce__(self) -> tuple[partial["Model"], tuple[()]]:
        # Pickle, copy and deepcopy rebuild through __init__, so they check the fields
        # again. Keywords keep a pickle valid when a later version adds a field.
        init = {f.name: getattr(self, f.name) for f in fields(self) if f.init}
        return partial(type(self), **init), ()

    def _precision(self, b: int) -> NDArray[np.float64]:
        P: NDArray[np.float64] | None = self._cache.get(("precision", b))
        if P is None:
            start = time.perf_counter()
            try:
                P = precision(self.cov[b])
            except LinAlgError as error:
                raise LinAlgError(f"model: cov[{b}] is {error}") from None
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
        self, b: int, nguard: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        cached = self._cache.get(("guard", b))
        if cached is not None and cached[0] == nguard:
            return cached[1], cached[2]
        C = self.cov[b]
        npix = len(C)
        if nguard < npix - 1:
            P = self._precision(b)
            start = time.perf_counter()
            try:
                W, var = guard_weights(P, np.diagonal(C), nguard)
            except LinAlgError as error:
                raise LinAlgError(f"model: cov[{b}] is {error}") from None
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


def save(model: Model, path: str | os.PathLike[str]) -> None:
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
    # fspath rejects an integer, which open would take as a file descriptor.
    with open(os.fspath(path), "wb") as file:
        np.savez(file, **arrays)


def load(path: str | os.PathLike[str]) -> Model:
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
    with open(os.fspath(path), "rb") as file:
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


def _from_archive(archive: Any, path: str | os.PathLike[str]) -> Model:
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
        tuple(cov),
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
    try:
        items = tuple(value)
    except TypeError:
        raise TypeError(
            f"{name}: expected a sequence, got {type(value).__name__}"
        ) from None
    if len(items) != count:
        raise ValueError(
            f"{name}: expected a sequence of length {count}, got length {len(items)}"
        )
    return tuple(check(item, f"{name}[{i}]") for i, item in enumerate(items))
