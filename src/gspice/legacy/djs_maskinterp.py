"""Linear interpolation over masked entries, after the IDL routine djs_maskinterp."""

# Written by Aaron Meisner.
# From https://github.com/desihub/gfa_reduce/blob/master/py/gfa_reduce/analysis/djs_maskinterp.py

import numpy as np
from scipy.interpolate import interp1d

from ._errors import _LegacyError


def maskinterp1(yval, mask):
    """Interpolate linearly over the bad entries of a 1-D array.

    The grid is regular. Entries before the first or after the last good entry get
    the value of that entry, as with the /const option of the IDL routine. With one
    good entry, of value v, a bad entry with a non-finite value gets v, and every
    other entry gets yval * 0 + v.

    Parameters
    ----------
    yval : array_like, shape (n,)
        Values. An integer dtype truncates the interpolated values.
    mask : ndarray, shape (n,)
        Nonzero after conversion to int marks a bad entry, so 0.5 is good.

    Returns
    -------
    ndarray, shape (n,)
        A copy of yval, interpolated. Without bad or without good entries, the copy
        is unchanged.
    """

    yval = np.array(yval)

    mask = mask.astype(int)

    bad = mask != 0
    if np.sum(bad) == 0:
        return yval

    good = mask == 0
    ngood = np.sum(good)
    if ngood == 0:
        return yval

    if ngood == 1:
        # yval * 0 is NaN exactly at the non-finite entries. Elsewhere, yval * 0 + value
        # keeps the 2022 result, with its signs of zero and its NaN at an infinite good
        # entry; and object arrays still work.
        value = yval[good][0]
        zero = yval * 0
        return np.where(bad & (zero != 0), value, zero + value)

    ynew = yval
    ny = len(yval)

    igood = (np.where(good))[0]
    ibad = (np.where(bad))[0]
    f = interp1d(igood, yval[igood], fill_value="extrapolate")

    yval[bad] = f(ibad)

    if igood[0] != 0:
        ynew[0 : igood[0]] = ynew[igood[0]]
    if igood[ngood - 1] != (ny - 1):
        ynew[(igood[ngood - 1] + 1) : ny] = ynew[igood[ngood - 1]]

    return ynew


def maskinterp(yval, mask, axis):
    """Apply maskinterp1 to each row or to each column of a 2-D array.

    Parameters
    ----------
    yval : ndarray, shape (nrow, ncol)
        Values.
    mask : ndarray, shape (nrow, ncol)
        Bad entries, as in maskinterp1.
    axis : {0, 1}
        0 interpolates along each row, and 1 along each column, as in the IDL
        routine.

    Returns
    -------
    ndarray, shape (nrow, ncol)
        A copy of yval, interpolated.

    Raises
    ------
    ValueError
        If yval or mask is not 2-D, their shapes differ, or axis is not 0 or 1.
    """

    mask = mask.astype(int)

    sh_yval = yval.shape
    sh_mask = mask.shape

    if len(sh_yval) != 2:
        raise _LegacyError(f"yval: expected a 2-D array, got {len(sh_yval)} dimensions")
    if len(sh_mask) != 2:
        raise _LegacyError(f"mask: expected a 2-D array, got {len(sh_mask)} dimensions")

    if sh_yval != sh_mask:
        raise _LegacyError(
            f"yval, mask: expected equal shapes, got {sh_yval} and {sh_mask}"
        )

    if axis not in (0, 1):
        raise _LegacyError(f"axis: expected 0 or 1, got {axis!r}")

    wbad = np.where(mask != 0)
    ynew = np.copy(yval)

    if axis == 0:
        bad_stripe_indices = np.unique(wbad[0])
    else:
        bad_stripe_indices = np.unique(wbad[1])

    for ind in bad_stripe_indices:
        if axis == 0:
            ynew[ind, :] = maskinterp1(ynew[ind, :], mask[ind, :])
        else:
            ynew[:, ind] = maskinterp1(ynew[:, ind], mask[:, ind])

    return ynew


def average_bilinear(yval, mask):
    """Mean of the interpolations of maskinterp along the rows and the columns.

    Parameters
    ----------
    yval : ndarray, shape (nrow, ncol)
        Values.
    mask : ndarray, shape (nrow, ncol)
        Bad entries, as in maskinterp1.

    Returns
    -------
    ndarray, shape (nrow, ncol)
        (maskinterp(yval, mask, 0) + maskinterp(yval, mask, 1)) / 2.
    """
    int0 = maskinterp(yval, mask, 0)
    int1 = maskinterp(yval, mask, 1)
    interp = (int0 + int1) / 2.0

    return interp
