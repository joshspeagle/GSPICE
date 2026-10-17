from collections.abc import Callable, Iterable
from dataclasses import KW_ONLY, dataclass, fields
from functools import partial
from itertools import pairwise
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import issymmetric

from ._spectra import finite_array, integer, numeric, real


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

    Attributes
    ----------
    mean : ndarray, shape (nbin, npix)
        The vector mean[b] of each of the nbin >= 1 S/N bins (scaled), subtracted to
        give x. npix >= 1 is the number of pixels. Finite.
    cov : tuple of ndarray, each shape (npix, npix)
        The covariance of x in each bin (scaled^2): nbin finite, exactly symmetric
        matrices.
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
        if not isinstance(self.weighted, bool | np.bool_):
            raise TypeError(
                f"weighted: expected a bool, got {type(self.weighted).__name__}"
            )
        # After the checks, because issymmetric of scipy 1.11 rejects read-only arrays.
        for array in (mean, *cov):
            array.flags.writeable = False
        for name, value in (
            ("mean", mean),
            ("cov", cov),
            ("n_train", n_train),
            ("edges", edges),
            ("center", center),
            ("weighted", bool(self.weighted)),
        ):
            object.__setattr__(self, name, value)

    def __reduce__(self) -> tuple[partial["Model"], tuple[()]]:
        # Pickle, copy and deepcopy rebuild through __init__, so they check the fields
        # again. Keywords keep a pickle valid when a later version adds a field.
        init = {f.name: getattr(self, f.name) for f in fields(self) if f.init}
        return partial(type(self), **init), ()


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
