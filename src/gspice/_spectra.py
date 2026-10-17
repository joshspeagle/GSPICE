import math
import numbers
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import maximum_filter1d

# Elements per chunk of rows (design section 8): 6990 rows at npix 2400.
CHUNK = 2**24

# progress(stage, done, total), with done and total counting spectra (design section 5).
Progress = Callable[[str, int, int], object]
# The scalars that integer, real and boolean accept, for the public annotations.
Integer = int | np.integer[Any]
Real = float | np.floating[Any] | np.integer[Any]
Boolean = bool | np.bool_


def chunks(nspec: int, npix: int) -> Iterator[slice]:
    step = max(1, CHUNK // npix)
    for start in range(0, nspec, step):
        yield slice(start, min(start + step, nspec))


def validate(
    flux: ArrayLike, ivar: ArrayLike, mask: ArrayLike | None
) -> tuple[NDArray[Any], NDArray[Any], NDArray[np.bool_]]:
    flux = numeric(flux, "flux")
    ivar = numeric(ivar, "ivar")
    if mask is not None:
        mask = as_array(mask, "mask")
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
        raise ValueError(
            f"mask, ivar: expected a good pixel in every spectrum, got none in rows "
            f"{listing(empty, 'rows')}"
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


def dilate(flags: NDArray[np.bool_], d: int) -> NDArray[np.bool_]:
    # The maximum over the 2 d + 1 pixels around each pixel of a row, with the pixels
    # beyond its ends unflagged, at a cost that does not grow with d. From any pixel,
    # d = npix - 1 already reaches the whole row.
    size = 2 * min(d, flags.shape[1] - 1) + 1
    dilated: NDArray[np.bool_] = maximum_filter1d(
        flags, size, axis=1, mode="constant", cval=0
    )
    return dilated


def root_mean_ivar(ivar: NDArray[Any], bad: NDArray[np.bool_]) -> NDArray[np.float64]:
    # numpy sums a contiguous row pairwise and a strided axis sequentially. As a
    # C-contiguous float64 array, ivar of every dtype and layout sums pairwise in
    # float64, which gives the 2022 refscale of a C-contiguous float64 ivar bit for bit.
    iv = np.ascontiguousarray(ivar, dtype=np.float64)
    a: NDArray[np.float64] = np.sqrt(
        np.where(bad, 0.0, iv).sum(axis=1) / np.count_nonzero(~bad, axis=1)
    )
    return a


def to_working(
    x: NDArray[np.float64],
    a: NDArray[np.float64],
    center: float,
    refmean: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    # x = a (x - center) - refmean in place, in the order of the 2022 standard_scale,
    # which gives its bits. refmean None: the column mean of a (x - center).
    x -= center
    x *= a[:, np.newaxis]
    if refmean is None:
        refmean = x.mean(axis=0)
    x -= refmean
    return refmean


def to_flux(
    x: NDArray[Any],
    a: NDArray[np.float64],
    refmean: NDArray[np.float64],
    center: float,
    out: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    # f = (x + refmean)/a + center, the inverse of to_working.
    f: NDArray[np.float64] = np.add(x, refmean, out=out)
    f /= a[:, np.newaxis]
    f += center
    return f


def median_snr(
    flux: NDArray[Any], ivar: NDArray[Any], bad: NDArray[np.bool_]
) -> NDArray[np.float64]:
    good = ~bad
    # Bad pixels sort last as inf, so the median indexes the first n values, with the
    # (a + b) / 2 of np.median; np.nanmedian loops over rows in Python for npix >= 600.
    values = np.full(good.shape, np.inf)
    np.sqrt(ivar, out=values, where=good, dtype=np.float64)
    np.multiply(flux, values, out=values, where=good, dtype=np.float64)
    values.sort(axis=1)
    n = np.count_nonzero(good, axis=1)
    r = np.arange(n.size)
    # The sort may order -0.0 and +0.0, which are equal, either way, and on some
    # platforms differently on each call; + 0.0 turns a median of -0.0 into +0.0.
    low, high = values[r, (n - 1) // 2], values[r, n // 2]
    median: NDArray[np.float64] = (low + high) / 2 + 0.0
    return median


def scale(
    flux: ArrayLike,
    ivar: ArrayLike,
    mask: ArrayLike | None = None,
    *,
    center: Real = 0.0,
    refmean: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Scale spectra to unit mean noise:  X_s = a_s (ftilde_s - center) - refmean.

    The bad pixels are (mask != 0) | (ivar == 0), and scale never reads flux or ivar
    there. It checks and computes the float64 values of the inputs: a long double
    beyond the float64 range is not finite, and one that rounds to 0 in float64 is 0.
    ftilde_s is flux_s cast to float64, then linearly interpolated over its bad pixels
    and held constant beyond its first and last good pixels. a_s is the square root of
    the mean of ivar_s over its good pixels. With float64 flux, scale(flux, ivar, mask)
    equals the 2022 standard_scale with the mask (mask != 0) | (ivar == 0), to
    rounding.

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
        refmean = vector(refmean, "refmean", npix, "pixel")
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
    # After the chunks, as in 2022: refmean is the mean of the whole array.
    refmean = to_working(X, a, center, refmean)
    return X, a, refmean


def unscale(
    x: ArrayLike,
    refscale: ArrayLike,
    refmean: ArrayLike,
    *,
    center: Real = 0.0,
    var: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Invert scale:  f = (x + refmean)/refscale + center,  var_f = var/refscale^2.

    It computes with the float64 values of the inputs, as scale does.

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
    refscale = vector(refscale, "refscale", nspec, "row")
    require(refscale > 0, refscale, "refscale", "values > 0", ("row",))
    refmean = vector(refmean, "refmean", n, "pixel")
    center = real(center, "center")
    if var is not None:
        var = numeric(var, "var")
        if var.shape not in ((n,), x.shape):
            raise ValueError(
                f"var: expected shape ({n},) or {x.shape} like x, got {var.shape}"
            )
        require(var >= 0, var, "var", "values >= 0", ("row", "pixel")[-var.ndim :])
    f = to_flux(x, refscale, refmean, center)
    if var is None:
        return f, None
    return f, var / np.square(refscale)[:, np.newaxis]


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
        out[rows] = median_snr(flux[rows], ivar[rows], bad[rows])
    return out


def as_array(value: object, name: str) -> NDArray[Any]:
    try:
        return np.asarray(value)
    except ValueError as error:
        # A ragged nested sequence.
        raise ValueError(f"{name}: {error}") from None


def integers(value: object, name: str) -> NDArray[np.integer[Any]]:
    array = as_array(value, name)
    if array.dtype.kind not in "iu":
        raise TypeError(f"{name}: expected an integer dtype, got {array.dtype}")
    return array


def numeric(value: object, name: str) -> NDArray[Any]:
    array = as_array(value, name)
    if array.dtype.kind not in "iuf":
        raise TypeError(
            f"{name}: expected an integer or floating dtype, got {array.dtype}"
        )
    # A long double can hold values beyond the float64 range. It is cast here, so the
    # checks and the computation see float64 values: inf beyond that range, and 0
    # where float64 rounds it to 0, also under a caller's errstate. The test is on the
    # type, because numpy treats a long double of the size of float64 as float64. The
    # other dtypes are cast per chunk.
    if array.dtype.type is np.longdouble:
        with np.errstate(over="ignore", under="ignore"):
            array = array.astype(np.float64)
    return array


def finite_array(
    value: NDArray[Any], name: str, axes: tuple[str, ...]
) -> NDArray[np.float64]:
    # A new C-contiguous float64 array (the default order keeps the input layout).
    array = np.array(value, dtype=np.float64, order="C")
    require(np.isfinite(array), array, name, "finite values", axes)
    return array


def vector(value: ArrayLike, name: str, size: int, axis: str) -> NDArray[np.float64]:
    array = numeric(value, name)
    if array.shape != (size,):
        raise ValueError(f"{name}: expected shape ({size},), got {array.shape}")
    return finite_array(array, name, (axis,))


def require(
    ok: NDArray[np.bool_],
    values: NDArray[Any],
    name: str,
    expected: str,
    axes: tuple[str, ...] | None = ("row", "pixel"),
    start: int = 0,
) -> None:
    if not ok.all():
        index = np.unravel_index(np.argmin(ok), ok.shape)
        where = (start + index[0], *index[1:]) if index else index
        raise ValueError(
            f"{name}: expected {expected}, got {values[index]}{located(where, axes)}"
        )


def located(index: tuple[Any, ...], axes: tuple[str, ...] | None = None) -> str:
    # Where the first failing entry is (design section 8): " at row 1, pixel 2" with
    # named axes, " at index (1, 2)" without, and nothing for a 0-d array.
    if not index:
        return ""
    if axes is None:
        return f" at index {tuple(map(int, index))}"
    return " at " + ", ".join(
        f"{axis} {i}" for axis, i in zip(axes, index, strict=True)
    )


def listing(indices: NDArray[np.intp], noun: str) -> str:
    # The first 10 indices, and their count if there are more (design section 8).
    more = f", ... ({indices.size} {noun})" if indices.size > 10 else ""
    return ", ".join(map(str, indices[:10])) + more


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


def integer(value: object, name: str, minimum: int, maximum: int | None = None) -> int:
    if not isinstance(value, numbers.Integral) or isinstance(value, bool):
        raise TypeError(f"{name}: expected an integer, got {type(value).__name__}")
    number = int(value)
    if number < minimum:
        raise ValueError(f"{name}: expected an integer >= {minimum}, got {number}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name}: expected an integer <= {maximum}, got {number}")
    return number


def boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool | np.bool_):
        raise TypeError(f"{name}: expected a bool, got {type(value).__name__}")
    return bool(value)


def check_progress(progress: object) -> None:
    if progress is not None and not callable(progress):
        raise TypeError(
            f"progress: expected a callable or None, got {type(progress).__name__}"
        )


def check_rng(rng: object) -> None:
    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            f"rng: expected a numpy.random.Generator, got {type(rng).__name__}"
        )
