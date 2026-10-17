import math

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import grey_opening

from ._spectra import dilate


def _exclude_extended(
    M: NDArray[np.bool_],
    I: NDArray[np.bool_],
    K: NDArray[np.bool_],
    runlength: int,
    halfwidth: int,
    minfrac: float,
) -> NDArray[np.bool_]:
    # Step 4 of design section 5.4, with the replaced, ignored and counting sets M, I
    # and K of shape (nspec, npix), and M and I disjoint. It returns a new ignored set.
    npix = M.shape[1]
    # The opening of each row by a window of runlength pixels, with the pixels beyond
    # its ends unflagged, keeps the runs of K of at least runlength pixels. No run fills
    # npix + 1 pixels, so a longer window is cut to that size, which bounds the buffers
    # of the filters.
    runs = grey_opening(K, size=(1, min(runlength, npix + 1)), mode="constant")
    widened = dilate(runs, halfwidth)
    widened |= I
    widened &= ~M
    # The floor is in float arithmetic (design decision A10).
    n_cond = npix - np.count_nonzero(M | widened, axis=1)
    short = n_cond < math.floor(minfrac * npix)
    widened[short] = I[short]
    return widened
