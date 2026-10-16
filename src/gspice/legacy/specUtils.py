"""The 2022 spectral functions: scaling, covariance, prediction and masking."""

import operator

import numpy as np
from scipy.ndimage import binary_dilation as dilate

from ._errors import _LegacyError
from .djs_maskinterp import maskinterp
from .matrixUtils import cholesky_inv, submatrix_inv_mult


def standard_scale(spec, ivar, mask=None):
    """Interpolate over bad pixels, scale spectra to about unit noise, and center.

    Dvec = refscale * maskinterp(spec) - refmean, where maskinterp interpolates
    along each spectrum, refscale is the square root of the mean ivar of the good
    pixels of each spectrum, and refmean is the mean of the scaled spectra.

    Parameters
    ----------
    spec : ndarray, shape (nspec, npix)
        Flux. An integer dtype truncates the interpolated values.
    ivar : ndarray, shape (nspec, npix)
        Inverse variance of the flux.
    mask : ndarray, shape (nspec, npix), optional
        Nonzero, NaN included, marks a bad pixel. None marks the pixels where
        ivar == 0.

    Returns
    -------
    Dvec : ndarray, shape (nspec, npix)
        The scaled and centered spectra, dimensionless.
    refscale : ndarray, shape (nspec,)
        The scale of each spectrum, in units of 1/flux. A spectrum without good
        pixels gets NaN, and one with a non-finite ivar at a good pixel a
        non-finite value; either makes every value of Dvec and refmean non-finite.
    refmean : ndarray, shape (npix,)
        The mean of the scaled spectra.
    """

    if mask is None:
        pixmask = ivar == 0
    else:
        pixmask = np.where(mask == 0, 0, 1)

    Dvec = maskinterp(yval=spec, mask=pixmask, axis=0)

    wt = 1 - pixmask
    # ivar * wt would be NaN at a bad pixel with a non-finite ivar. The product with
    # wt, an int64 array, keeps the float64 sum of a float32 ivar, as in 2022.
    meanivar = np.sum(np.where(pixmask, 0, ivar) * wt, axis=1) / np.sum(wt, axis=1)
    refscale = np.sqrt(meanivar)
    Dvec = Dvec * refscale[:, np.newaxis]
    refmean = Dvec.mean(axis=0)
    Dvec = Dvec - refmean.reshape(1, -1)

    return Dvec, refscale, refmean


def covar(spec, checkmean=False):
    """Sample covariance of the pixels over the spectra.

    cov = (spec - refmean)^T (spec - refmean) / (nspec - 1), where refmean is the
    mean spectrum.

    Parameters
    ----------
    spec : ndarray, shape (nspec, npix)
        Spectra, float64.
    checkmean : bool, optional
        If ``checkmean is True``, print the minimum, maximum and standard
        deviation of the column means after centering.

    Returns
    -------
    cov : ndarray, shape (npix, npix)
        The covariance, in the units of spec squared.
    refmean : ndarray, shape (npix,)
        The mean spectrum, in the units of spec.

    Raises
    ------
    ValueError
        If spec is not float64.
    """

    nspec, _ = spec.shape

    if spec.dtype != "float64":
        raise _LegacyError("spec.dtype must be float64")

    refmean = spec.mean(axis=0)
    spec = spec - refmean.reshape(1, -1)

    if checkmean is True:
        mnd = spec.mean(axis=0)
        print(f"min = {mnd.min()}, max = {mnd.max()}, std = {mnd.std(ddof=1)}")

    cov = (spec.T @ spec) / (nspec - 1)
    return cov, refmean


def get_chimask(flux, ivar, mask, nsigma):
    """Flag the pixels that deviate from their GSPICE prediction by nsigma or more.

    Z = (Dvec - pred) / sqrt(predvar), where Dvec comes from standard_scale, and
    pred and predvar from gp_interp at nguard 20 with the covariance of Dvec. A pixel
    with |Z| >= nsigma is flagged, together with its neighbours in its spectrum.

    Parameters
    ----------
    flux : ndarray, shape (nspec, npix)
        Flux.
    ivar : ndarray, shape (nspec, npix)
        Inverse variance of the flux.
    mask : ndarray, shape (nspec, npix), or None
        Bad pixels, as in standard_scale.
    nsigma : float
        Threshold on |Z|.

    Returns
    -------
    chimask : ndarray of bool, shape (nspec, npix)
        True for a flagged pixel.

    Notes
    -----
    Cost: that of covar and of gp_interp at nguard 20.
    """

    Dvec, _, _ = standard_scale(flux, ivar, mask)
    covmat, _ = covar(Dvec)
    pred, predvar = gp_interp(Dvec, covmat, nguard=20)
    chi = (Dvec - pred) / np.sqrt(predvar)
    flag = np.abs(chi) >= nsigma
    chimask = dilate(flag, np.array([[1, 1, 1]])).astype(flag.dtype)

    return chimask


def gaussian_estimate(icond, ipred, cov, Dvec, covinv, bruteforce=False):
    """Gaussian conditional mean and covariance of some pixels, given others.

    With S = cov, the conditioning pixels k and the predicted pixels k*, the mean is
    Dvec[:, k] S_kk^-1 S_kk* and the covariance is S_k*k* - S_k*k S_kk^-1 S_kk*.
    The default path updates covinv with submatrix_inv_mult; bruteforce inverts S_kk
    with cholesky_inv. If icond and ipred overlap, each shared pixel gets its own
    value as the mean and a variance of 0 to rounding, which can be negative.

    Parameters
    ----------
    icond : ndarray, shape (npix,)
        Nonzero for the conditioning pixels k.
    ipred : ndarray, shape (npix,)
        Nonzero for the predicted pixels k*.
    cov : ndarray, shape (npix, npix)
        Covariance of the pixels.
    Dvec : ndarray, shape (nspec, npix)
        Spectra, such as the output of standard_scale.
    covinv : ndarray, shape (npix, npix)
        Inverse of cov. bruteforce does not read it.
    bruteforce : bool, optional
        If True, return (predcovar, predkstar, kstar) only.

    Returns
    -------
    predoverD : ndarray, shape (npix, nk*)
        S_kk^-1 S_kk* in the rows k and 0 in the other rows, so that the mean is
        Dvec @ predoverD.
    predcovar : ndarray, shape (nk*, nk*)
        The conditional covariance, in the units of cov.
    predkstar : ndarray, shape (nspec, nk*)
        The conditional mean, in the units of Dvec. With bruteforce and nk* = 1,
        its shape is (nspec,).
    kstar : ndarray of int, shape (nk*,)
        The indices of the predicted pixels.

    Notes
    -----
    Cost: about 2 npix (npix + nk*) nk* + nr^3 + 2 nr (nr + npix) nk* +
    2 nspec npix nk* flops, where nr is the number of pixels outside k; with
    bruteforce, about nk^3 + 2 nk nk* (2 nk + nk*) + 2 nspec nk nk* flops.
    """

    _, npix = Dvec.shape

    k = np.where(icond)[0]
    kstar = np.where(ipred)[0]
    nkstar = len(kstar)
    cov_kkstar = cov[np.ix_(kstar, k)]
    cov_kstark = cov_kkstar.T
    cov_kstarkstar = cov[np.ix_(kstar, kstar)]

    if bruteforce:
        cov_kk = cov[np.ix_(k, k)]

        # Cholesky inversion is faster than SVD here
        icov_kk = cholesky_inv(cov_kk)
        del cov_kk  # save memory

        # compute prediction covariance (See RW, Ch. 2)
        predcovar = cov_kstarkstar - cov_kkstar @ icov_kk @ cov_kstark

        tmp = cov_kkstar @ icov_kk
        del icov_kk

        if nkstar == 1:
            tmp2 = np.zeros(npix)
            tmp2[k] = tmp
            predkstar = Dvec @ tmp2
        else:
            predkstar = Dvec[:, k] @ tmp.T  # this is memory intensive

        return predcovar, predkstar, kstar
    else:
        Y = cov[:, kstar]
        Minvy = covinv @ Y
        Ainvy0 = submatrix_inv_mult(M=cov, Minv=covinv, imask=icond, Y=Y, MinvY=Minvy)
        predkstar = Dvec @ Ainvy0
        predoverD = Ainvy0

        # compute prediction covariance (See RW, Ch. 2)
        predcovar = cov_kstarkstar - Y.T @ Ainvy0
        return predoverD, predcovar, predkstar, kstar


def _index(value, name):
    # numpy 1.26 still reads a numpy bool as 0 or 1, with a DeprecationWarning.
    if isinstance(value, np.bool_):
        raise TypeError(f"{name}: expected an integer, got {value!r}")
    try:
        return operator.index(value)
    except TypeError as err:
        raise TypeError(f"{name}: expected an integer, got {value!r}") from err


def gp_interp(spec, cov, nguard=20, irange=None, bruteforce=False):
    """Predict each pixel from the pixels outside a guard window around it (GSPICE).

    For pixel i, gaussian_estimate conditions on every pixel outside i - nguard to
    i + nguard, with the inverse of cov from cholesky_inv.

    Parameters
    ----------
    spec : ndarray, shape (nspec, npix)
        Spectra, such as the output of standard_scale.
    cov : ndarray, shape (npix, npix)
        Positive definite covariance of the pixels.
    nguard : int, optional
        Half width of the guard window.
    irange : sequence of int, optional
        Predict the pixels irange[0] to irange[1], both included, and ignore later
        elements. The default is every pixel.
    bruteforce : bool, optional
        True raises ValueError, as in 2022.

    Returns
    -------
    pred : ndarray, shape (nspec, n)
        The predictions of the n = irange[1] - irange[0] + 1 predicted pixels, in
        the units of spec.
    predvar : ndarray, shape (nspec, n)
        The conditional variances, in the units of spec squared. All rows are equal.

    Raises
    ------
    ValueError
        If cov is not square, does not match spec or has a non-finite entry, spec
        has no pixels, nguard is negative, or irange is not 1-D, has fewer than 2
        elements or breaks 0 <= irange[0] <= irange[1] <= npix - 1.
    TypeError
        If nguard, irange[0] or irange[1] is not an integer, or is a numpy bool.
    numpy.linalg.LinAlgError
        If cov is not positive definite.

    Notes
    -----
    Cost: about npix^3 flops for the inverse of cov, and per predicted pixel about
    2 npix^2 + w^3 + 2 w (w + npix) + 4 nspec npix flops, where w is the number of
    pixels in the guard window, at most 2 nguard + 1.
    """

    nspec, npix = spec.shape
    if cov.shape[0] != cov.shape[1]:
        raise _LegacyError("cov: covariance must be square matrix.")
    if npix != cov.shape[0]:
        raise _LegacyError(
            "spec, cov: spectra and covariance have incompatible dimensions."
        )
    nguard = _index(nguard, "nguard")
    if nguard < 0:
        raise _LegacyError(f"nguard: expected an integer >= 0, got {nguard}")
    if npix == 0:
        raise _LegacyError("spec: expected at least one pixel, got 0")

    if irange is not None:
        # The elements come from irange itself, so that np.asarray cannot promote them.
        if np.ndim(irange) != 1 or np.size(irange) < 2:
            raise _LegacyError(
                f"irange: expected a 1-D sequence of at least 2 integers, got shape {np.shape(irange)}"
            )
        i0 = _index(irange[0], "irange")
        i1 = _index(irange[1], "irange")
        if not 0 <= i0 <= i1 <= npix - 1:
            raise _LegacyError(
                f"irange: expected 0 <= irange[0] <= irange[1] <= {npix - 1}, got {i0} and {i1}"
            )
    else:
        i0 = 0
        i1 = npix - 1

    szpred = i1 - i0 + 1
    predvar = np.zeros((nspec, szpred))

    if bruteforce:
        pred = np.zeros((nspec, szpred))
    else:
        predoverD = np.zeros((npix, szpred))

    covinv = cholesky_inv(cov)

    for i in range(i0, i1 + 1):
        ipred = np.zeros(npix)
        ipred[i] = 1

        icond = np.ones(npix)
        j0 = np.max([0, i - nguard])
        j1 = np.min([i + nguard + 1, npix])
        icond[j0:j1] = 0

        predoverD0, predcovar, predkstar, kstar = gaussian_estimate(
            icond=icond,
            ipred=ipred,
            cov=cov,
            Dvec=spec,
            covinv=covinv,
            bruteforce=bruteforce,
        )

        if bruteforce:
            pred[:, kstar - i0] = predkstar
        else:
            predoverD[:, kstar - i0] = predoverD0
        predvar[:, kstar - i0] = np.diag(predcovar)

    pred = spec.dot(predoverD)

    return pred, predvar


def covar_iter_mask(flux, ivar, mask, nsigma=np.array([20, 8, 6]), maxbadpix=64):  # noqa: B008
    """Covariance of the spectra after passes of get_chimask, one per threshold.

    Spectra with more than maxbadpix bad pixels in mask are dropped. Each pass runs
    get_chimask with the bad pixels of mask and the flags of the previous pass
    masked, not those of earlier passes. finalmask combines the mask of the last
    pass with its flags, and the covariance comes from standard_scale and covar
    with finalmask.

    Parameters
    ----------
    flux : ndarray, shape (nspec, npix)
        Flux.
    ivar : ndarray, shape (nspec, npix)
        Inverse variance of the flux.
    mask : ndarray, shape (nspec, npix)
        Nonzero, NaN included, marks a bad pixel. None is not supported.
    nsigma : sequence of float, optional
        The thresholds of the passes, in order; at least one.
    maxbadpix : int, optional
        The largest number of bad pixels in mask that a kept spectrum has.

    Returns
    -------
    cov : ndarray, shape (npix, npix)
        The covariance of the kept spectra, scaled as by standard_scale.
    finalmask : ndarray of float64, shape (nspec, npix)
        1.0 for a masked pixel and 0.0 for another. Every pixel of a dropped
        spectrum is 1.0.

    Raises
    ------
    ValueError
        If fewer than npix spectra are kept.
    numpy.linalg.LinAlgError
        If the covariance of a pass is not positive definite. With exactly npix
        kept spectra it is singular, so a pass usually raises this error, and
        otherwise the returned covariance is singular.

    Notes
    -----
    Cost: that of get_chimask per pass, plus standard_scale and covar.
    """

    nspec, npix = flux.shape

    nbadpix = np.sum(mask != 0, axis=1)
    objmask = nbadpix <= maxbadpix
    wmask = np.where(objmask)[0]
    nmask = len(wmask)

    if nmask < npix:
        raise _LegacyError(
            f"Not enough good spectra: {nmask} spectra have at most maxbadpix = {maxbadpix} bad pixels in mask, and at least npix = {npix} are needed"
        )

    chimask = np.zeros(mask[wmask].shape)

    for sigma in nsigma:
        thismask = np.logical_or(chimask, (mask[wmask] != 0))

        chimask = get_chimask(
            flux=flux[wmask], ivar=ivar[wmask], mask=thismask, nsigma=sigma
        )

    finalmask = np.ones((nspec, npix))
    finalmask[wmask] = np.logical_or(thismask, chimask)

    spec, _, _ = standard_scale(
        spec=flux[wmask], ivar=ivar[wmask], mask=finalmask[wmask]
    )
    cov, _ = covar(spec)

    return cov, finalmask
