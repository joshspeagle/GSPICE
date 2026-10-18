import dataclasses
import math
import time
from functools import partial
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import grey_opening

from ._blocks import BlockFactor, block_chi2, block_factors
from ._model import Blocks, Model
from ._predict import (
    Filling,
    Repaired,
    filled,
    infilled,
    noisy,
    spans,
    validated,
    working,
    zscores,
)
from ._spectra import (
    Boolean,
    Integer,
    Progress,
    Real,
    boolean,
    check_choice,
    check_progress,
    check_rng,
    integer,
    real,
)
from ._spectra import dilate as widen


def repair(
    model: Model,
    flux: ArrayLike,
    ivar: ArrayLike,
    mask: ArrayLike | None = None,
    *,
    nguard: Integer = 20,
    nsigma_replace: Real = 10.0,
    nsigma_ignore: Real = 5.0,
    dilate: Integer = 1,
    blockwise: Boolean = True,
    runlength: Integer | None = 5,
    halfwidth: Integer = 150,
    minfrac: Real = 0.5,
    count_on: Literal["all", "flagged"] = "all",
    inflate: Boolean = False,
    covariance: Boolean = False,
    rng: np.random.Generator | None = None,
    progress: Progress | None = None,
) -> Repaired:
    """Detect and replace bad pixels with the procedure the paper validates (l.510).

    For a spectrum with the working vector x (see `Model`) and the bad pixels
    bad = (mask != 0) | (ivar == 0), let z(y) be the z of `pixelwise` with nguard at
    every pixel of a vector y, its interpolated bad pixels included, and dil(F, d) the
    pixels within d of a pixel of F along the spectrum. A "conditional mean" is that of
    `infill`. The steps, in this order, are:

    1. M1 = dil(|z(x)| > nsigma_replace, dilate) | bad, and
       I1 = dil(|z(x)| > nsigma_ignore, dilate) minus M1.
    2. If blockwise: x_b is x with x at M1 | I1 replaced by its conditional mean given
       the other pixels. While the largest ratio chi2_A(x_b) / thresholds[b, A] over
       the blocks A of model.blocks that are not flagged exceeds 1, with chi2_A of
       `fit_blocks` and b the S/N bin of the spectrum, the block of that ratio is
       flagged and its pixels join M_b, and x_b becomes x with x at M1 | I1 | M_b
       replaced by its conditional mean given the other pixels. A block is flagged at
       most once, so there are at most nblock rounds. Without blockwise, M_b is empty.
    3. x1 is x with x at M1 | M_b replaced by its conditional mean given the pixels
       outside M1 | M_b | I1. M = M1 | M_b | dil(|z(x1)| > nsigma_replace, dilate),
       and I = dil(|z(x1)| > nsigma_ignore, dilate) minus M.
    4. If runlength is not None: every run of at least runlength consecutive pixels of
       the counting set K adds every pixel within halfwidth of the run to I, minus M,
       unless fewer than floor(minfrac npix) pixels outside M and I would remain (D-57).
       K is M | I for count_on="all". For "flagged", K holds the pixels outside bad
       with |z(x)| > nsigma_replace or |z(x1)| > nsigma_ignore, and M_b.
    5. The result is that of `infill` with replace=M, ignore=I, and inflate,
       covariance and rng, bit for bit, plus the flagged blocks. For a dense noise
       model whose steps 1 to 4 form P_s, for guard windows or blocks short of all
       pixels, step 5 conditions on that P_s, where `infill` uses the Cholesky factor
       of C_s, so the two agree to rounding.

    Nothing limits the replaced set; the paper measured calibration up to 1024 of 2415
    replaced pixels (l.530). Since M and I depend on the data, f_s does not hold for
    Repaired.chi2 (see `finite_sample_factor`). For a noise model, x is in flux units,
    and C_s of the spectrum (see `Model`) takes the place of cov[b] in every step:
    repair runs the five steps one spectrum at a time, with one P_s for all of them.

    Parameters
    ----------
    model : Model
        The model; with blockwise=True, one with blocks from `fit_blocks`.
    flux, ivar, mask : array_like, shape (nspec, npix)
        As in `scale`, with the npix of model.
    nguard : int, optional
        The half-width of the guard window of z (pixels); >= 0.
    nsigma_replace : float, optional
        The threshold of |z| for replacing a pixel; finite and > 0.
    nsigma_ignore : float, optional
        The threshold of |z| for ignoring a pixel; finite, > 0 and <= nsigma_replace.
    dilate : int, optional
        The pixels d by which dil widens the flags of z on each side; >= 0.
    blockwise : bool, optional
        If True, run the block pass of step 2.
    runlength : int or None, optional
        The shortest run of K that widens I (pixels); >= 1. None skips step 4.
    halfwidth : int, optional
        The pixels on each side of a run that join I; >= 0.
    minfrac : float, optional
        The fraction of npix that step 4 leaves to condition on; from 0 to 1.
    count_on : {"all", "flagged"}, optional
        The counting set K of step 4.
    inflate, covariance : bool, optional
        As in `infill`, for step 5. inflate needs a dense, mean-centred, unweighted
        model without noise.
    rng : numpy.random.Generator, optional
        As in `infill`, for step 5: the source of Repaired.draws. None: no draws.
    progress : callable, optional
        progress(stage, done, total), called with the stage "repair", the number of
        spectra done and their total, after each chunk of spectra of steps 1 to 4, and
        again after each chunk of step 5; a noise model runs the five steps on chunks
        of one spectrum, and reports after its steps 1 to 4 and after its step 5.

    Returns
    -------
    Repaired
        The repaired spectra, in flux units, with replaced = M and ignored = I, and
        the flagged blocks of each spectrum in blocks, or None if blockwise is False.

    Raises
    ------
    TypeError
        If model is not a Model, nguard, dilate, runlength or halfwidth is not an
        integer (or None for runlength), nsigma_replace, nsigma_ignore or minfrac is
        not a real number, blockwise, inflate or covariance is not a bool, rng is
        neither None nor a numpy.random.Generator, progress is not callable or None,
        or flux, ivar or mask has a dtype that `scale` does not allow.
    ValueError
        If flux, ivar or mask has a shape or a value that `scale` does not allow, flux
        has another npix than model, a spectrum has no good pixel, an argument is
        outside its range above, count_on is neither "all" nor "flagged", or blockwise
        is True and model has no blocks. If inflate is True and model has a constant
        centre, weights, a `LowRank` covariance or noise, or, before step 5,
        N_s - N_k - 3 <= 0 for a spectrum with a replaced pixel, with
        N_k = npix - |M | I|; the message names N_s and N_k.
    numpy.linalg.LinAlgError
        If the covariance of a bin with spectra fails the check of its precision
        matrix, as in `pixelwise`; repair checks P of every bin with spectra before
        any spectrum, also for nguard >= npix - 1, where `pixelwise` computes no P.
        As in `fit_blocks` for the blocks of a bin with spectra, with blockwise=True,
        or as in `infill` for a conditional mean. The message names the bin, and the
        block or the row. For a noise model, as in `infill`, with C_s in place of
        cov[b], and checked for each spectrum where a step factors C_s.

    Notes
    -----
    Cost: npix^3 flops for P at the first use of a bin, the guard weights as in
    `pixelwise`, and with blockwise=True |G|^3/3 flops per block and bin with spectra,
    with G the block and its guard. Then per spectrum 4 npix^2 flops for the two
    pixelwise passes, about 2 npix^2 + sum_A |G|^2 flops per scan of its blocks,
    one scan and one infill per flagged block, and two infills as in `infill`: that
    of step 5, and that of step 2, whose last x_b gives x1, or without blockwise that
    of step 3. All but the infill of step 5 compute only the mean.
    repair holds replaced and ignored of all spectra, and a few arrays of one chunk.
    A noise model costs P_s, npix^3 flops for a dense cov[0] or npix^2 K for a
    `LowRank` of rank K, its guard windows as in `pixelwise` and its block factors
    for each spectrum; the infills of a `LowRank` cov[0] do not use P_s, and cost as
    in `infill`.
    """
    start = time.perf_counter()
    nguard = integer(nguard, "nguard", 0)
    nsigma_replace = _positive(nsigma_replace, "nsigma_replace")
    nsigma_ignore = _positive(nsigma_ignore, "nsigma_ignore")
    if nsigma_ignore > nsigma_replace:
        raise ValueError(
            f"nsigma_ignore: expected a number <= nsigma_replace, {nsigma_replace}, "
            f"got {nsigma_ignore}"
        )
    d = integer(dilate, "dilate", 0)
    blockwise = boolean(blockwise, "blockwise")
    if runlength is not None:
        runlength = integer(runlength, "runlength", 1)
    halfwidth = integer(halfwidth, "halfwidth", 0)
    minfrac = real(minfrac, "minfrac")
    if not 0 <= minfrac <= 1:
        raise ValueError(f"minfrac: expected a number from 0 to 1, got {minfrac}")
    check_choice(count_on, "count_on", ("all", "flagged"))
    inflate = boolean(inflate, "inflate")
    covariance = boolean(covariance, "covariance")
    if rng is not None:
        check_rng(rng)
    check_progress(progress)
    flux, ivar, bad = validated(model, flux, ivar, mask, inflate)
    if blockwise and model.blocks is None:
        raise ValueError(
            "blockwise: expected False for a model without blocks, got True; "
            "fit_blocks gives a model blocks"
        )
    blocks = model.blocks if blockwise else None
    nspec, npix = flux.shape
    bins = model._bins(flux, ivar, bad)
    detect = partial(
        _detect,
        blocks=blocks,
        nguard=nguard,
        nsigma_replace=nsigma_replace,
        nsigma_ignore=nsigma_ignore,
        d=d,
        runlength=runlength,
        halfwidth=halfwidth,
        minfrac=minfrac,
        count_on=count_on,
    )
    nblock = 0 if blocks is None else blocks.starts.size
    flagged = np.empty((nspec, nblock), dtype=bool)
    replaced, ignored = (np.empty((nspec, npix), dtype=bool) for _ in range(2))
    factors: dict[int, list[BlockFactor]] = {}
    filling = None
    if model.noise:
        # Step 5 runs on each spectrum after its steps 1 to 4, so that every step
        # reuses its P_s (decision A22). No factor f_s is derived (design section 4.3).
        filling = Filling(
            model,
            flux,
            ivar,
            bad,
            bins,
            inflation=np.ones(nspec),
            inflate=False,
            covariance=covariance,
            rng=rng,
        )
    else:
        # P, the guard weights and the block factors of the bins with spectra, before
        # any spectrum, so that a covariance that fails a check raises at once. Every
        # step but the block scan can need P.
        for b in np.unique(bins).tolist():
            model._precision(b)
            model._guard_weights(b, nguard)
            if blocks is not None:
                factors[b] = block_factors(
                    model, b, blocks.starts, blocks.width, blocks.nguard
                )
    for rows in spans(model, nspec, npix):
        X, a = working(model, flux[rows], ivar[rows], bad[rows], bins[rows])
        index = np.arange(rows.start, rows.stop)
        cover = noisy(model, ivar[rows], bad[rows], rows.start)
        if model.noise and blocks is not None:
            factors = {
                0: block_factors(cover, 0, blocks.starts, blocks.width, blocks.nguard)
            }
        flagged[rows], replaced[rows], ignored[rows] = detect(
            cover, factors, X, bins[rows], index, bad[rows]
        )
        if progress is not None:
            progress("repair", rows.stop, nspec)
        if filling is not None:
            filling.fill(cover, rows, X, a, replaced[rows], ignored[rows])
            if progress is not None:
                progress("repair", rows.stop, nspec)
        del cover
        if model.noise:
            # Free the block factors of the spectrum, which hold its P_s.
            factors = {}
    # The block factors can take 649 MB per bin (see fit_blocks); step 5 needs none.
    del factors
    if filling is None:
        # Step 5, which checks inflate where G is known.
        result = infilled(
            model,
            flux,
            ivar,
            bad,
            bins,
            replaced,
            ignored,
            inflate=inflate,
            covariance=covariance,
            rng=rng,
            progress=progress,
            stage="repair",
            start=start,
        )
    else:
        result = filling.done(replaced, ignored, np.ones(nspec), "repair", start)
    flagged.flags.writeable = False
    return dataclasses.replace(result, blocks=None if blocks is None else flagged)


def _detect(
    model: Model,
    factors: dict[int, list[BlockFactor]],
    X: NDArray[np.float64],
    bins: NDArray[np.intp],
    rows: NDArray[np.intp],
    bad: NDArray[np.bool_],
    *,
    blocks: Blocks | None,
    nguard: int,
    nsigma_replace: float,
    nsigma_ignore: float,
    d: int,
    runlength: int | None,
    halfwidth: int,
    minfrac: float,
    count_on: str,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    # Steps 1 to 4 of design section 5.4 for the working vectors X of a chunk of
    # spectra, with the block factors of their bins: the flagged blocks, M and I.
    # Step 1, with G1 = M1 | I1.
    z1 = zscores(model, nguard, X, bins)
    M1 = _dilated(z1, nsigma_replace, d) | bad
    G1 = M1 | _dilated(z1, nsigma_ignore, d)
    # Steps 2 and 3. The last x_b of the block pass conditions on the pixels outside
    # G1 | M_b, as x1 does, so its values at M1 | M_b are those of x1.
    if blocks is None:
        flagged = np.zeros((len(X), 0), dtype=bool)
        Mb = np.zeros(X.shape, dtype=bool)
        X1 = filled(model, X, bins, rows, G1, M1)
    else:
        flagged, Mb, X_b = _block_pass(model, blocks, factors, X, bins, rows, G1)
        X1 = np.where(M1 | Mb, X_b, X)
    z2 = zscores(model, nguard, X1, bins)
    M = M1 | Mb | _dilated(z2, nsigma_replace, d)
    I = _dilated(z2, nsigma_ignore, d) & ~M
    # Step 4.
    if runlength is not None:
        if count_on == "all":
            K = M | I
        else:
            core = (np.abs(z1) > nsigma_replace) | (np.abs(z2) > nsigma_ignore)
            K = (core & ~bad) | Mb
        I = _exclude_extended(M, I, K, runlength, halfwidth, minfrac)
    return flagged, M, I


def _positive(value: object, name: str) -> float:
    number = real(value, name)
    if number <= 0:
        raise ValueError(f"{name}: expected a number > 0, got {number}")
    return number


def _dilated(z: NDArray[np.float64], nsigma: float, d: int) -> NDArray[np.bool_]:
    # dil(|z| > nsigma, d) of design section 5.4.
    return widen(np.abs(z) > nsigma, d)


def _block_pass(
    model: Model,
    blocks: Blocks,
    factors: dict[int, list[BlockFactor]],
    X: NDArray[np.float64],
    bins: NDArray[np.intp],
    rows: NDArray[np.intp],
    G: NDArray[np.bool_],
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.float64]]:
    # Step 2 of design section 5.4 for the working vectors X of a chunk of spectra, with
    # G = M1 | I1: the flagged blocks (nspec, nblock), their pixels M_b and the last
    # x_b.
    starts = blocks.starts[:, np.newaxis]
    pixels = np.arange(X.shape[1])
    cover = (pixels >= starts) & (pixels < starts + blocks.width)
    flagged = np.zeros((len(X), starts.size), dtype=bool)
    Mb = np.zeros(X.shape, dtype=bool)
    x_b = filled(model, X, bins, rows, G, G)
    # The spectra to scan: all at first, then those whose x_b changed.
    scan = np.arange(len(X))
    while scan.size:
        ratio = np.empty((scan.size, starts.size))
        # One scan of all these spectra of a bin (Q5: the worst block of a spectrum
        # has the largest ratio).
        for b in np.unique(bins[scan]).tolist():
            at = bins[scan] == b
            ratio[at] = block_chi2(factors[b], x_b[scan[at]]) / blocks.thresholds[b]
        ratio[flagged[scan]] = 0.0
        worst = np.argmax(ratio, axis=1)
        exceeds = ratio[np.arange(scan.size), worst] > 1
        scan, worst = scan[exceeds], worst[exceeds]
        flagged[scan, worst] = True
        Mb[scan] |= cover[worst]
        G_b = G[scan] | Mb[scan]
        x_b[scan] = filled(model, X[scan], bins[scan], rows[scan], G_b, G_b)
    return flagged, Mb, x_b


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
    widened = widen(runs, halfwidth)
    widened |= I
    widened &= ~M
    # The floor is in float arithmetic (design decision A10).
    n_cond = npix - np.count_nonzero(M | widened, axis=1)
    short = n_cond < math.floor(minfrac * npix)
    widened[short] = I[short]
    return widened
