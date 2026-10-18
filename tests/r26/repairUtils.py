# Vendored from gspice/repairUtils.py of the branch revision-2026 (r26) at
# ebbb0653f9b24848572fde4da08f0494488224a0, for tests/test_repair.py only. The
# functions are copied unchanged. The module docstring, the functions that gp_repair
# does not call and the imports that only they use are left out.

import numpy as np
from scipy import linalg
from scipy.ndimage import binary_dilation

from .matrixUtils import cholesky_inv
from .specUtils import standard_scale, gp_weights


def _dilate(mask, width = 1):
    """Flag every pixel within width pixels (along the wavelength axis) of a flagged pixel."""
    if width <= 0:
        return np.asarray(mask, dtype = bool).copy()
    return binary_dilation(np.asarray(mask, dtype = bool), np.ones((1, 2 * width + 1), dtype = bool))


def unscale(pred, predvar, refscale, refmean = None):
    """
    Transform predictions and variances from scaled units back to flux units

    ``standard_scale`` computes X = refscale * flux - refmean. This inverts it
    for a prediction of X: flux = (pred + refmean) / refscale, and the variance
    of the flux is predvar / refscale**2.

        Parameters:
            pred (np.ndarray): nspec X n predictions in scaled units (for
                               example from gp_interp or gp_infill)
            predvar (np.ndarray): variances in scaled units, either n (the same
                               for every spectrum, as from gp_weights) or
                               nspec X n
            refscale (np.ndarray): (nspec,) scale factors from standard_scale
            refmean (np.ndarray): (n,) mean subtracted by standard_scale, or
                               None if pred has not been mean-subtracted

        Returns:
            flux (np.ndarray): nspec X n predictions in the units of the input flux
            fluxvar (np.ndarray): nspec X n variances in the squared units of the input flux

        Spectra with refscale == 0 (no unmasked pixels) return NaN.
    """
    pred = np.asarray(pred, dtype = np.float64)
    refscale = np.asarray(refscale, dtype = np.float64)
    predvar = np.broadcast_to(np.asarray(predvar, dtype = np.float64), pred.shape)
    if refmean is not None:
        pred = pred + np.asarray(refmean).reshape(1, -1)
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        rs = np.where(refscale > 0, refscale, np.nan)[:, np.newaxis]
        return pred / rs, predvar / rs**2


def gp_infill(spec, cov, mask, ignore = None, covinv = None, return_covar = False):
    """
    Replace masked pixels of each spectrum by their Gaussian conditional mean

    For spectrum s let M be its masked pixels, I its ignored pixels (not in M)
    and O all the others. The replaced values are the conditional mean of x_M
    given x_O, and the returned variances are the diagonal of the conditional
    covariance. Pixels in I are neither used as conditioning data nor replaced
    (this is the two-threshold scheme of Section 2.6: replace pixels with
    |Z| > 10, condition only on pixels with |Z| < 5, ignore the rest). Spectra
    with identical (mask, ignore) patterns share one factorization, so the cost
    is one small Cholesky factorization per distinct pattern. This is the
    "variant of Algorithm 2" that predicts all masked pixels of one spectrum at
    once.

        Parameters:
            spec (np.ndarray) nspec X npix : scaled, mean-subtracted spectra
                         (output of standard_scale). Values at masked and
                         ignored pixels are never used and may be anything.
            cov (np.ndarray) npix X npix : pixel covariance matrix
            mask (np.ndarray) nspec X npix : nonzero == replace
            ignore (np.ndarray) nspec X npix : nonzero == do not condition on,
                         do not replace (default None); pixels also in mask are
                         treated as masked
            covinv (np.ndarray) npix X npix : inverse of cov (computed if not given)
            return_covar (bool) : also return the full conditional covariance of the
                         masked pixels (see below)

        Returns:
            filled (np.ndarray) nspec X npix : copy of spec with masked pixels replaced
            predvar (np.ndarray) nspec X npix : conditional variance (scaled units) at
                         masked pixels, 0 elsewhere
            groups (list) : only if return_covar. One tuple (rows, cols, condcov) per
                         distinct mask pattern: the spectra ``rows`` that share it,
                         the masked pixels ``cols``, and their len(cols) X len(cols)
                         conditional covariance in scaled units (divide by
                         refscale[row]**2 for flux units). The covariance is the
                         same for every spectrum of the group. Off-diagonal terms are
                         needed for quantities that combine several replaced pixels,
                         such as equivalent widths.

        Comments:
            Uses the precision matrix P = cov^-1: for the index set G = M | I,
            x_G | x_O ~ N(-P_GG^-1 P_GO x_O, P_GG^-1), so only a |G| X |G|
            factorization is required.
    """
    spec = np.asarray(spec, dtype = np.float64)
    nspec, npix = spec.shape
    assert cov.shape == (npix, npix), "spectra and covariance have incompatible dimensions."
    M = np.asarray(mask) != 0
    assert M.shape == spec.shape, "mask must have the shape of spec."
    if ignore is None:
        Ign = np.zeros_like(M)
    else:
        Ign = (np.asarray(ignore) != 0) & ~M
        assert Ign.shape == spec.shape, "ignore must have the shape of spec."
    if covinv is None:
        covinv = cholesky_inv(cov)
    P = 0.5 * (covinv + covinv.T)

    filled = spec.copy()
    predvar = np.zeros_like(spec)
    groups = []
    G = M | Ign
    spec0 = np.where(G, 0., spec) #masked and ignored entries do not enter P_GO x_O

    #group spectra with identical patterns
    key = np.packbits(np.concatenate([M, Ign], axis = 1), axis = 1)
    _, inv = np.unique(key, axis = 0, return_inverse = True)
    inv = inv.ravel()
    order = np.argsort(inv, kind = 'stable')
    bounds = np.flatnonzero(np.diff(inv[order])) + 1
    for rows in np.split(order, bounds):
        m = M[rows[0]]
        if not m.any():
            continue
        g = np.flatnonzero(G[rows[0]])
        mpos = np.flatnonzero(m[g]) #positions of the masked pixels within g
        cf = linalg.cho_factor(P[np.ix_(g, g)], lower = True, check_finite = False)
        rhs = P[g, :] @ spec0[rows].T #P_GO x_O, with x_G zeroed
        mu = -linalg.cho_solve(cf, rhs, check_finite = False)
        cg = linalg.cho_solve(cf, np.eye(len(g)), check_finite = False) #P_GG^-1
        cm = cg[np.ix_(mpos, mpos)]
        cols = g[mpos]
        filled[np.ix_(rows, cols)] = mu[mpos].T
        predvar[np.ix_(rows, cols)] = np.diag(cm)
        if return_covar:
            groups.append((rows, cols, 0.5 * (cm + cm.T)))

    if return_covar:
        return filled, predvar, groups
    return filled, predvar


def make_blocks(npix, width = 150, stride = 75):
    """
    Overlapping blocks of consecutive pixels covering the spectrum

        Parameters:
            npix (int)   : number of pixels
            width (int)  : pixels per block (default 150)
            stride (int) : separation of block starts (default 75); a final block
                           ending at the last pixel is added if needed

        Returns:
            blocks (list) : list of integer index arrays, one per block
    """
    starts = list(range(0, npix - width + 1, stride))
    if starts[-1] != npix - width:
        starts.append(npix - width)
    return [np.arange(s0, s0 + width) for s0 in starts]


def _block_operator(cov, ipred, icond):
    """Weights W, conditional covariance and its Cholesky factor for predicting ipred from icond."""
    cf = linalg.cho_factor(cov[np.ix_(icond, icond)], lower = True, check_finite = False)
    W = linalg.cho_solve(cf, cov[np.ix_(icond, ipred)], check_finite = False)
    ccov = cov[np.ix_(ipred, ipred)] - cov[np.ix_(ipred, icond)] @ W
    return W, 0.5 * (ccov + ccov.T)


def _block_conditioning_set(npix, ipred, nguard):
    """Pixels farther than nguard from every predicted pixel."""
    excl = np.zeros(npix, dtype = bool)
    excl[ipred] = True
    if nguard > 0:
        excl = _dilate(excl[np.newaxis, :], nguard)[0]
    return np.flatnonzero(~excl)


def _as_index(sel, npix):
    sel = np.asarray(sel)
    if sel.dtype == bool or (sel.dtype.kind == 'f' and sel.size == npix):
        return np.flatnonzero(sel != 0)
    return np.unique(sel.astype(np.int64))


def block_operators(cov, blocks, nguard = 20):
    """
    Precompute the operators needed for blockwise tests

    Each block is predicted from all pixels farther than nguard from it.

        Parameters:
            cov (np.ndarray) npix X npix : pixel covariance matrix
            blocks (list) : index arrays, for example from make_blocks
            nguard (int)  : guard half-width around each block (default 20)

        Returns:
            ops (list) : one tuple (ipred, icond, W, L) per block, where W
                         are the prediction weights and L is the lower Cholesky
                         factor of the conditional covariance
    """
    npix = cov.shape[0]
    ops = []
    for a in blocks:
        a = _as_index(a, npix)
        b = _block_conditioning_set(npix, a, nguard)
        W, ccov = _block_operator(cov, a, b)
        ops.append((a, b, W, np.linalg.cholesky(ccov)))
    return ops


def block_chisq(spec, ops):
    """
    Whitened chi^2 per pixel of each block's prediction residual

    For each block, (x - pred)^T C^-1 (x - pred) / len(block) with C the
    conditional covariance of the block. This is the statistic of the blockwise
    test, sensitive to extended artifacts such as wavelength-calibration errors.

        Parameters:
            spec (np.ndarray) nspec X npix : scaled, mean-subtracted spectra
            ops (list) : output of block_operators

        Returns:
            chisq (np.ndarray) nspec X nblocks
    """
    out = np.empty((spec.shape[0], len(ops)))
    for k, (a, b, W, L) in enumerate(ops):
        r = spec[:, a] - spec[:, b] @ W
        y = linalg.solve_triangular(L, r.T, lower = True, check_finite = False)
        out[:, k] = np.sum(y**2, axis = 0) / len(a)
    return out


def exclude_extended(mask, ignore, runlength = 5, halfwidth = 150, minfrac = 0.5):
    """
    Exclusion of the neighbourhood of extended artifacts from the conditioning set

    A wavelength-calibration error spanning hundreds of pixels produces large
    |Z| only on the flanks of strong lines, while its other pixels are biased
    coherently by less than the exclusion threshold. If they stay in the
    conditioning set the repaired pixels are biased and their variances too
    small. Wherever mask | ignore contains a run of at least ``runlength``
    consecutive pixels, every pixel within ``halfwidth`` pixels of the run is
    added to ``ignore``. The replaced set is not changed. In a spectrum where
    this would leave fewer than ``minfrac`` of the pixels (rounded down) for
    conditioning, the step is skipped.

        Parameters:
            mask (np.ndarray) nspec X npix : pixels to replace
            ignore (np.ndarray) nspec X npix : pixels not used for conditioning
            runlength (int)  : consecutive flagged pixels that define an extended artifact (5)
            halfwidth (int)  : pixels excluded on either side of a run (150)
            minfrac (float)  : minimum fraction of pixels left to condition on (0.5)

        Returns:
            ignore (np.ndarray) nspec X npix : the enlarged ignore mask (bool),
                         never overlapping mask
    """
    M = np.asarray(mask) != 0
    I = (np.asarray(ignore) != 0) & ~M
    n, npix = M.shape
    if runlength > npix:
        return I
    G = (M | I).astype(np.int64)
    cs = np.concatenate([np.zeros((n, 1), dtype = np.int64), np.cumsum(G, axis = 1)], axis = 1)
    run = (cs[:, runlength:] - cs[:, :-runlength]) == runlength #run[s, a]: pixels a..a+runlength-1 all flagged
    if not run.any():
        return I
    a0 = np.arange(npix - runlength + 1)
    lo = np.clip(a0 - halfwidth, 0, npix)
    hi = np.clip(a0 + runlength + halfwidth, 0, npix)
    rr, aa = np.nonzero(run)
    d = np.zeros((n, npix + 1), dtype = np.int64)
    np.add.at(d, (rr, lo[aa]), 1)
    np.add.at(d, (rr, hi[aa]), -1)
    wide = np.cumsum(d, axis = 1)[:, :npix] > 0
    Iw = (I | wide) & ~M
    ok = np.sum(~(M | Iw), axis = 1) >= int(np.floor(minfrac * npix))
    return np.where(ok[:, np.newaxis], Iw, I)


def gp_repair(flux, ivar, cov, refmean, mask = None, nguard = 20, nsigma_replace = 10.,
              nsigma_ignore = 5., block_ops = None, block_thresh = None, max_block_iter = 6,
              exclude_runs = True, runlength = 5, exclusion_halfwidth = 150, minfrac = 0.5,
              weights = None, covinv = None, return_covar = False):
    """
    Detect and replace outlying pixels in spectra (the validated repair procedure)

    Given a clean covariance (for example from covar_iter_mask with
    return_scale = True), this applies the procedure evaluated on simulated
    spectra in Section 4 of the paper, to every spectrum:

    1. Pixelwise pass. Z-scores from the pixelwise estimate with guard window
       nguard. Pixels with |Z| > nsigma_replace are replaced (set M1) and pixels
       with nsigma_ignore < |Z| <= nsigma_replace are ignored (set I1); both
       are dilated by one pixel.
    2. Blockwise pass (only if block_ops and block_thresh are given). On a copy
       with M1 and I1 infilled, the block whose chi^2 exceeds its threshold by the
       largest factor is flagged and infilled, repeatedly, until no block exceeds
       its threshold (at most max_block_iter blocks). Flagged blocks are replaced.
       The blockwise test must follow the pixelwise pass: otherwise a strong
       localized artifact contaminates the prediction of every block.
    3. Second pixelwise pass on the spectrum repaired with M1 and the flagged
       blocks (ignoring I1), giving the final replaced set M (|Z| > nsigma_replace,
       plus M1 and the flagged blocks) and ignored set I
       (|Z| > nsigma_ignore, dilated, not in M).
    4. If exclude_runs, exclude_extended widens I around extended artifacts.
    5. Pixels in M are replaced by their conditional mean given the original
       pixels outside M and I (gp_infill), with conditional variance.

    Pixels flagged in the input mask (or with ivar == 0 if mask is None) are
    interpolated for scaling, as in standard_scale, and are also replaced.

        Parameters:
            flux (np.ndarray) nspec X npix : flux
            ivar (np.ndarray) nspec X npix : inverse variance
            cov (np.ndarray) npix X npix : covariance of scaled spectra
            refmean (np.ndarray) npix : mean of the scaled training spectra, as
                         returned by standard_scale or covar_iter_mask(return_scale = True)
            mask (np.ndarray) nspec X npix : input mask, 0 == good (default ivar == 0)
            nguard (int) : guard half-width of the pixelwise estimate (20)
            nsigma_replace (float) : |Z| above which pixels are replaced (10)
            nsigma_ignore (float) : |Z| above which pixels are not used to condition (5)
            block_ops (list) : output of block_operators (default None: no blockwise pass)
            block_thresh (np.ndarray) : output of block_thresholds, one value per block
            max_block_iter (int) : most blocks flagged per spectrum (6)
            exclude_runs (bool) : apply exclude_extended (True)
            runlength, exclusion_halfwidth, minfrac : parameters of exclude_extended
                         (5, 150, 0.5)
            weights (tuple) : (predoverD, predvar) from gp_weights(cov, nguard), to avoid
                         recomputing them when calling on successive chunks of spectra
            covinv (np.ndarray) : inverse of cov (computed if not given)
            return_covar (bool) : include the conditional covariance groups of gp_infill in info

        Returns:
            newflux (np.ndarray) nspec X npix : flux with replaced pixels filled in
                         (unreplaced pixels are returned unchanged)
            newvar (np.ndarray) nspec X npix : conditional variance in squared flux
                         units at replaced pixels, 0 elsewhere. It replaces the
                         original 1/ivar of those pixels and is correlated between
                         pixels of one spectrum (see info['groups'])
            info (dict) : 'replace' (bool nspec X npix), 'ignore' (bool nspec X npix),
                         'block_mask' (bool nspec X npix, pixels in flagged blocks),
                         'block_flag' (bool nspec X nblocks or None), 'refscale' (nspec,),
                         and if return_covar 'groups' (covariances in scaled units)
    """
    flux = np.asarray(flux, dtype = np.float64)
    ivar = np.asarray(ivar, dtype = np.float64)
    nspec, npix = flux.shape
    inmask = (ivar == 0) if mask is None else (np.asarray(mask) != 0)

    X, refscale, _ = standard_scale(flux, ivar, inmask, refmean = refmean)
    if covinv is None:
        covinv = cholesky_inv(cov)
    if weights is None:
        weights = gp_weights(cov, nguard = nguard, covinv = covinv)
    A, pvar = weights
    sd = np.sqrt(pvar)[np.newaxis, :]

    #1. pixelwise pass on the data
    Z0 = (X - X @ A) / sd
    M1 = _dilate(np.abs(Z0) > nsigma_replace, 1) | inmask
    I1 = _dilate(np.abs(Z0) > nsigma_ignore, 1) & ~M1
    del Z0

    #2. blockwise pass on a copy with M1 and I1 infilled
    use_blocks = block_ops is not None
    if use_blocks:
        if block_thresh is None or len(block_thresh) != len(block_ops):
            raise ValueError("block_thresh must give one threshold per block in block_ops.")
        block_thresh = np.asarray(block_thresh, dtype = np.float64)
        Mb = np.zeros_like(M1)
        bflag = np.zeros((nspec, len(block_ops)), dtype = bool)
        Xb = gp_infill(X, cov, M1 | I1, covinv = covinv)[0]
        todo = np.arange(nspec)
        for it in range(max_block_iter):
            if todo.size == 0:
                break
            ratio = block_chisq(Xb[todo], block_ops) / block_thresh[np.newaxis, :]
            ratio[bflag[todo]] = 0.
            k = ratio.argmax(axis = 1)
            hit = ratio[np.arange(todo.size), k] > 1
            todo, k = todo[hit], k[hit]
            if todo.size == 0:
                break
            for s, kk in zip(todo, k):
                bflag[s, kk] = True
                Mb[s, block_ops[kk][0]] = True
            Xb[todo] = gp_infill(X[todo], cov, (M1 | I1 | Mb)[todo], covinv = covinv)[0]
        del Xb
    else:
        Mb = np.zeros_like(M1)
        bflag = None

    #3. second pixelwise pass on the spectrum repaired with M1 and the blocks
    Mrep = M1 | Mb
    X1 = gp_infill(X, cov, Mrep, ignore = I1, covinv = covinv)[0] if Mrep.any() else X
    Z2 = (X1 - X1 @ A) / sd
    M = Mrep | _dilate(np.abs(Z2) > nsigma_replace, 1)
    I = _dilate(np.abs(Z2) > nsigma_ignore, 1) & ~M
    del Z2, X1

    #4. exclusion of the neighbourhoods of extended artifacts
    if exclude_runs:
        I = exclude_extended(M, I, runlength = runlength, halfwidth = exclusion_halfwidth,
                             minfrac = minfrac)

    #5. final replacement
    info = dict(replace = M, ignore = I, block_mask = Mb, block_flag = bflag, refscale = refscale)
    if M.any():
        out = gp_infill(X, cov, M, ignore = I, covinv = covinv, return_covar = return_covar)
        Xr, var = out[0], out[1]
        if return_covar:
            info['groups'] = out[2]
    else:
        Xr, var = X, np.zeros_like(X)
        if return_covar:
            info['groups'] = []
    rep, fluxvar = unscale(Xr, var, refscale, refmean)
    newflux = np.where(M, rep, flux)
    newvar = np.where(M, fluxvar, 0.)
    return newflux, newvar, info
