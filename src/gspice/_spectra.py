import math
import numbers
from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Elements per chunk of rows (design section 8): 6990 rows at npix 2400.
CHUNK = 2**24


def chunks(nspec: int, npix: int) -> Iterator[slice]:
    step = max(1, CHUNK // npix)
    for start in range(0, nspec, step):
        yield slice(start, start + step)


def validate(
    flux: ArrayLike, ivar: ArrayLike, mask: ArrayLike | None
) -> tuple[NDArray[Any], NDArray[Any], NDArray[np.bool_]]:
    flux = numeric(flux, "flux")
    ivar = numeric(ivar, "ivar")
    if mask is not None:
        mask = np.asarray(mask)
        if mask.dtype.kind not in "biuf":
            raise TypeError(
                f"mask: expected a bool, integer or floating dtype, got {mask.dtype}"
            )
    if flux.ndim != 2:
        raise ValueError(f"flux: expected a 2-D array, got shape {flux.shape}")
    if flux.shape[1] == 0:
        raise ValueError(f"flux: expected at least one pixel, got shape {flux.shape}")
    for name, array in (("ivar", ivar), ("mask", mask)):
        if array is not None and array.shape != flux.shape:
            raise ValueError(
                f"{name}: expected shape {flux.shape} like flux, got {array.shape}"
            )
    bad = np.empty(flux.shape, dtype=bool)
    for rows in chunks(*flux.shape):
        iv = ivar[rows]
        in_mask = False if mask is None else mask[rows] != 0
        require(
            in_mask | (np.isfinite(iv) & (iv >= 0)),
            iv,
            "ivar",
            "finite values >= 0 at pixels not in mask",
            start=rows.start,
        )
        bad[rows] = in_mask | (iv == 0)
        f = flux[rows]
        require(
            bad[rows] | np.isfinite(f),
            f,
            "flux",
            "finite values at good pixels",
            start=rows.start,
        )
    return flux, ivar, bad


def require_good_pixel(bad: NDArray[np.bool_]) -> None:
    empty = np.flatnonzero(bad.all(axis=1))
    if empty.size:
        listed = ", ".join(map(str, empty[:10]))
        more = f", ... ({empty.size} rows)" if empty.size > 10 else ""
        raise ValueError(
            f"mask, ivar: expected a good pixel in every spectrum, got none in rows "
            f"{listed}{more}"
        )


def interpolate(flux: NDArray[Any], bad: NDArray[np.bool_]) -> NDArray[np.float64]:
    # Every row needs a good pixel. x is a new C-contiguous float64 array, so the flat
    # indices of bad are its pixels, and flux stays as it is. The work runs over the
    # maximal runs of bad pixels, so the temporaries scale with the bad pixels.
    x = np.empty(flux.shape)
    x[...] = flux
    npix = x.shape[1]
    flat = x.reshape(-1)
    holes = np.flatnonzero(bad)
    starts = np.ones(holes.size, dtype=bool)
    starts[1:] = (np.diff(holes) != 1) | (holes[1:] % npix == 0)
    ends = np.ones(holes.size, dtype=bool)
    ends[:-1] = starts[1:]
    run = np.cumsum(starts) - 1
    first, last = holes[starts], holes[ends]
    # lo and hi are the good pixels on either side of a run; a run at an end of its row
    # uses the one good pixel next to it for both.
    lo = np.where(first % npix == 0, last + 1, first - 1)[run]
    hi = np.where(last % npix == npix - 1, first - 1, last + 1)[run]
    f_lo = flat[lo]
    span = hi - lo
    # slope * (i - lo) + f_lo is the formula of np.interp and of the scipy 1.11
    # interp1d that 2022 used, and it is exact on a constant segment.
    slope = (flat[hi] - f_lo) / np.maximum(span, 1)
    flat[holes] = np.where(span > 0, slope * (holes - lo) + f_lo, f_lo)
    return x


def root_mean_ivar(ivar: NDArray[Any], bad: NDArray[np.bool_]) -> NDArray[np.float64]:
    # numpy sums a contiguous row pairwise and a strided axis sequentially. As a
    # C-contiguous float64 array, ivar of every dtype and layout sums pairwise in
    # float64, which gives the 2022 refscale of a C-contiguous float64 ivar bit for bit.
    iv = np.ascontiguousarray(ivar, dtype=np.float64)
    a: NDArray[np.float64] = np.sqrt(
        np.where(bad, 0.0, iv).sum(axis=1) / np.count_nonzero(~bad, axis=1)
    )
    return a


def scale(
    flux: ArrayLike,
    ivar: ArrayLike,
    mask: ArrayLike | None = None,
    *,
    center: float = 0.0,
    refmean: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Scale spectra to unit mean noise:  X_s = a_s (ftilde_s - center) - refmean.

    The bad pixels are (mask != 0) | (ivar == 0), and scale never reads flux or ivar
    there. ftilde_s is flux_s cast to float64, then linearly interpolated over its bad
    pixels and held constant beyond its first and last good pixels. a_s is the square
    root of the mean of ivar_s over its good pixels. With float64 flux,
    scale(flux, ivar, mask) equals the 2022 standard_scale with the mask
    (mask != 0) | (ivar == 0), to rounding.

    Parameters
    ----------
    flux : array_like, shape (nspec, npix)
        Flux, of an integer or floating dtype; finite at good pixels.
    ivar : array_like, shape (nspec, npix)
        Inverse variance (flux^-2), of an integer or floating dtype; finite and >= 0
        at pixels not in mask.
    mask : array_like, shape (nspec, npix), optional
        Nonzero marks a pixel in mask, of a bool, integer or floating dtype. None: no
        pixel is in mask.
    center : float, optional
        Constant subtracted before scaling (flux).
    refmean : array_like, shape (npix,), optional
        Mean subtracted after scaling (scaled); finite. None: the column mean of
        a_s (ftilde_s - center), which needs at least one spectrum. For held-out
        spectra, pass the model mean of their S/N bin: their own mean breaks the
        finite-sample factor f_s.

    Returns
    -------
    X : ndarray, shape (nspec, npix)
        Scaled spectra (scaled).
    refscale : ndarray, shape (nspec,)
        a_s (flux^-1).
    refmean : ndarray, shape (npix,)
        The subtracted mean (scaled); a copy if refmean is given.

    Raises
    ------
    TypeError
        If an argument has a dtype or a type that is not allowed above.
    ValueError
        If an argument has a shape or a value that is not allowed above, or a
        spectrum has no good pixel.
    """
    flux, ivar, bad = validate(flux, ivar, mask)
    require_good_pixel(bad)
    center = real(center, "center")
    nspec, npix = flux.shape
    if refmean is not None:
        refmean = _vector(refmean, "refmean", npix, "pixel")
    elif nspec == 0:
        raise ValueError(
            f"flux: expected at least one spectrum when refmean is None, got shape "
            f"{flux.shape}"
        )
    X = np.empty((nspec, npix))
    a = np.empty(nspec)
    for rows in chunks(nspec, npix):
        X[rows] = interpolate(flux[rows], bad[rows])
        a[rows] = root_mean_ivar(ivar[rows], bad[rows])
    # The order of the 2022 standard_scale, which gives its bits: interpolate, scale,
    # then subtract the column mean of the whole C-contiguous array.
    X -= center
    X *= a[:, np.newaxis]
    if refmean is None:
        refmean = X.mean(axis=0)
    X -= refmean
    return X, a, refmean


def unscale(
    x: ArrayLike,
    refscale: ArrayLike,
    refmean: ArrayLike,
    *,
    center: float = 0.0,
    var: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Invert scale:  f = (x + refmean)/refscale + center,  var_f = var/refscale^2.

    Parameters
    ----------
    x : array_like, shape (nspec, n)
        Values at n pixels (scaled), of an integer or floating dtype.
    refscale : array_like, shape (nspec,)
        a_s of scale (flux^-1); finite and > 0.
    refmean : array_like, shape (n,)
        The mean that scale subtracted at these pixels (scaled); finite.
    center : float, optional
        The center of scale (flux).
    var : array_like, shape (n,) or (nspec, n), optional
        Variances of x (scaled^2), of an integer or floating dtype; >= 0.

    Returns
    -------
    f : ndarray, shape (nspec, n)
        Flux.
    var_f : ndarray, shape (nspec, n), or None
        Variances of f (flux^2); None if var is None.

    Raises
    ------
    TypeError
        If an argument has a dtype or a type that is not allowed above.
    ValueError
        If an argument has a shape or a value that is not allowed above.
    """
    x = numeric(x, "x")
    if x.ndim != 2:
        raise ValueError(f"x: expected a 2-D array, got shape {x.shape}")
    nspec, n = x.shape
    refscale = _vector(refscale, "refscale", nspec, "row")
    require(refscale > 0, refscale, "refscale", "values > 0", ("row",))
    refmean = _vector(refmean, "refmean", n, "pixel")
    center = real(center, "center")
    if var is not None:
        var = numeric(var, "var")
        if var.shape not in ((n,), x.shape):
            raise ValueError(
                f"var: expected shape ({n},) or {x.shape} like x, got {var.shape}"
            )
        require(var >= 0, var, "var", "values >= 0", ("row", "pixel")[-var.ndim :])
    # dtype=np.float64 keeps a long double x or var from making long double results.
    f = np.add(x, refmean, dtype=np.float64)
    f /= refscale[:, np.newaxis]
    f += center
    if var is None:
        return f, None
    return f, np.divide(var, np.square(refscale)[:, np.newaxis], dtype=np.float64)


def snr(
    flux: ArrayLike, ivar: ArrayLike, mask: ArrayLike | None = None
) -> NDArray[np.float64]:
    """Median of flux sqrt(ivar) over the good pixels of each spectrum.

    The bad pixels are (mask != 0) | (ivar == 0), and snr never reads flux or ivar
    there. The products are computed in float64.

    Parameters
    ----------
    flux, ivar, mask : array_like, shape (nspec, npix)
        As in scale.

    Returns
    -------
    ndarray, shape (nspec,)
        Signal-to-noise ratio (unit-free).

    Raises
    ------
    TypeError
        If an argument has a dtype that scale does not allow.
    ValueError
        If an argument has a shape or a value that scale does not allow, or a
        spectrum has no good pixel.
    """
    flux, ivar, bad = validate(flux, ivar, mask)
    require_good_pixel(bad)
    out = np.empty(flux.shape[0])
    for rows in chunks(*flux.shape):
        good = ~bad[rows]
        # Bad pixels sort last as inf, so the median indexes the first n values, with
        # the (a + b) / 2 of np.median; np.nanmedian loops over rows in Python for
        # npix >= 600.
        values = np.full(good.shape, np.inf)
        np.sqrt(ivar[rows], out=values, where=good, dtype=np.float64)
        np.multiply(flux[rows], values, out=values, where=good, dtype=np.float64)
        values.sort(axis=1)
        n = np.count_nonzero(good, axis=1)
        r = np.arange(n.size)
        # The sort may order -0.0 and +0.0, which are equal, either way, and on some
        # platforms differently on each call; + 0.0 turns a median of -0.0 into +0.0.
        out[rows] = (values[r, (n - 1) // 2] + values[r, n // 2]) / 2 + 0.0
    return out


def numeric(value: ArrayLike, name: str) -> NDArray[Any]:
    try:
        array = np.asarray(value)
    except ValueError as error:
        # A ragged nested sequence.
        raise ValueError(f"{name}: {error}") from None
    if array.dtype.kind not in "iuf":
        raise TypeError(
            f"{name}: expected an integer or floating dtype, got {array.dtype}"
        )
    return array


def finite_array(
    value: NDArray[Any], name: str, axes: tuple[str, ...]
) -> NDArray[np.float64]:
    # A new C-contiguous float64 array (the default order keeps the input layout). The
    # check runs after the cast, so a long double beyond the float64 range fails it.
    array = np.array(value, dtype=np.float64, order="C")
    require(np.isfinite(array), array, name, "finite values", axes)
    return array


def _vector(value: ArrayLike, name: str, size: int, axis: str) -> NDArray[np.float64]:
    vector = numeric(value, name)
    if vector.shape != (size,):
        raise ValueError(f"{name}: expected shape ({size},), got {vector.shape}")
    return finite_array(vector, name, (axis,))


def require(
    ok: NDArray[np.bool_],
    values: NDArray[Any],
    name: str,
    expected: str,
    axes: tuple[str, ...] = ("row", "pixel"),
    start: int = 0,
) -> None:
    if not ok.all():
        index = np.unravel_index(np.argmin(ok), ok.shape)
        where = (start + index[0], *index[1:])
        located = ", ".join(f"{axis} {i}" for axis, i in zip(axes, where, strict=True))
        raise ValueError(
            f"{name}: expected {expected}, got {values[index]} at {located}"
        )


def real(value: object, name: str) -> float:
    if not isinstance(value, numbers.Real) or isinstance(value, bool):
        raise TypeError(f"{name}: expected a real number, got {type(value).__name__}")
    try:
        number = float(value)
    except OverflowError:
        number = math.inf
    if not math.isfinite(number):
        raise ValueError(f"{name}: expected a finite number, got {number}")
    return number


def integer(value: object, name: str, minimum: int) -> int:
    if not isinstance(value, numbers.Integral) or isinstance(value, bool):
        raise TypeError(f"{name}: expected an integer, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name}: expected an integer >= {minimum}, got {value}")
    return int(value)
