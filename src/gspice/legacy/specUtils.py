import operator

import numpy as np
from scipy.ndimage import binary_dilation as dilate

from ._errors import _LegacyError
from .djs_maskinterp import maskinterp
from .matrixUtils import cholesky_inv, submatrix_inv_mult


def standard_scale(spec, ivar, mask=None):
    """
    Scale input data to have uniform variance per spectrum

        Parameters:
            spec (np.ndarray): spec (nspec, npix)
            ivar (np.ndarray): ivar (nspec, npix)
            mask (np.ndarray): mask (nspec, npix);
                               0 == good, default is ivar == 0

        Returns:
            spec (np.ndarray): scaled flux array (npsec, npix)
    """

    # if no mask is passed, use ivar == 0 as mask (setting infinity error as bad pixel)
    if mask is None:
        pixmask = ivar == 0
    else:
        pixmask = np.where(mask == 0, 0, 1)  # scale mask to 0 or 1

    # interpolate over masked pixels in the spectral direction
    Dvec = maskinterp(yval=spec, mask=pixmask, axis=0)

    # renormalize each spectra by sqrt(mean(ivar))
    wt = 1 - pixmask  # set weight such that only good pixels contribute
    meanivar = np.sum(ivar * wt, axis=1) / np.sum(wt, axis=1)
    refscale = np.sqrt(meanivar)
    Dvec = Dvec * refscale[:, np.newaxis]  # rescale data as roughly data/sigma
    refmean = Dvec.mean(axis=0)
    Dvec = Dvec - refmean.reshape(1, -1)

    return Dvec, refscale, refmean


def covar(spec, checkmean=False):
    """
    Compute covariance of cleaned data vector using BLAS

        Parameters:
            spec (np.ndarray) nspec X npix: spectral data (nspec, npix); must be float64

        Returns:
            cov (np.ndarray) npix X npix: covariance matrix of the pixels
            refmean (np.array) 1 X nspec: reference mean of each spectra
    """

    nspec, _ = spec.shape

    # check data type
    if spec.dtype != "float64":
        raise _LegacyError("spec.dtype must be float64")

    # make columns of Dmask mean 0 for covariance computation
    refmean = spec.mean(axis=0)
    spec = spec - refmean.reshape(1, -1)

    # verify that mean subtraction worked
    if checkmean is True:
        mnd = spec.mean(axis=0)
        print(f"min = {mnd.min()}, max = {mnd.max()}, std = {mnd.std(ddof=1)}")

    # compute covariance
    cov = (spec.T @ spec) / (nspec - 1)
    return cov, refmean


def get_chimask(flux, ivar, mask, nsigma):
    """
    Compute mask of outliers with respect to GSPICE posterior covariance.

        Parameters:
            flux (np.ndarray): flux (nspec, npix)
            ivar (np.ndarray): ivar (nspec, npix)
            mask (np.ndarray): mask (nspec, npix) (0 == good)
            nsigma (int)     : threshold to clip data at
        Returns:
            chimask (np.ndarray): mask of nonconformat pixels (0 == good)
    """

    # interpolate over masked pixels in the spectral direction
    Dvec, _, _ = standard_scale(flux, ivar, mask)

    # obtain empirical covariance for this data array
    covmat, _ = covar(Dvec)

    # compute GSPICE predicted mean and covariance
    pred, predvar = gp_interp(Dvec, covmat, nguard=20)

    # compute z-score
    chi = (Dvec - pred) / np.sqrt(predvar)

    # clip at nsigma, dilate mask by 1 pixel
    flag = np.abs(chi) >= nsigma
    chimask = dilate(flag, np.array([[1, 1, 1]])).astype(flag.dtype)

    return chimask


def gaussian_estimate(icond, ipred, cov, Dvec, covinv, bruteforce=False):
    """
    Computes Gaussian conditional estimate

        Parameters:
            icond (np.array) : index list of pixels to condition on ??is it nspec x npix or flattened?
            ipred (np.array) : index list of pixels to predict ??
            cov (np.ndarray) npix X npix    : covariance matrix
            spec (np.ndarray) nspec X npix  : data vector
            covinv (np.ndarray) npix X npix : inverse of cov,
                                              to speed up computation
            bruteforce (bool) : flag to use bruteforce code

        Returns:
            predcovar (np.ndarray) npix X npix  : predicted covariance matrix
            predoverD (np.ndarray) npix X kstar : matrix to multiply by spec to
                                                  get predicted spectra;
                                                  kstar refers to row(s) to be
                                                  predicted
    """

    _, npix = Dvec.shape

    k = np.where(icond)[0]  # where is good data
    kstar = np.where(ipred)[0]
    nkstar = len(kstar)  # number of pixels to predict on
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

        # compute icov_kk (dot) cov_kstark
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
        # compute icov_kk (dot) cov_kstark using GSPICE
        Y = cov[:, kstar]

        Minvy = covinv @ Y

        # Ainvy is icov_kk (dot) cov_kstarK
        # Ainvy0 is Ainvy zero-padded
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
    """
    Computes pixelwise conditional prediction for each pixel (GSPICE routine)

        Parameters:
            spec (np.ndarray) nspec X npix : spectra matrix
            cov (np.ndarray) npix X npix : pixel covariance matrix
            nguard (int) : number of guard pixels around GCE pixel(s)
            irange (np.array) 2 X 1 : 2 element array specifying range for prediction pixels

        Returns:
            pred (np.ndarray) nspec X npix : GSPICE predicted spectra mean
            predcovar (np.ndarray) npix X npix : GSPICE predicted posterior spectra variance
    """

    # shape of input spectra array
    nspec, npix = spec.shape
    # check proper dimensionality
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

    # range of spectral bins to operate on
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

    # allocate output arrays
    szpred = i1 - i0 + 1
    predvar = np.zeros((nspec, szpred))

    if bruteforce:
        pred = np.zeros((nspec, szpred))
    else:
        predoverD = np.zeros((npix, szpred))

    # pre-compute inverse covariance
    covinv = cholesky_inv(cov)

    # loop over pixels
    for i in range(i0, i1 + 1):
        # ipred == 1 for pixels to be predicted
        ipred = np.zeros(npix)
        ipred[i] = 1

        # condition on reference pixels specified by icond
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
    """
    Compute spectral covariance with iterative masking using GSPICE

        Parameters:
            flux (np.ndarray) nspec X npix : de-redshifted flux array
            ivar (np.ndarray) nspec X npix : de-redshifted ivar array
            mask (np.ndarray) nspec X npix : de-redshifted mask array, 0 == good;
                                             if not set, set to ivar == 0
            nsigma (np.array)              : array of nsigma clipping values to be passed to chimask
            maxbadpix (int)                : reject spectra with more than maxbadpix
                                             pixels masked in input mask

        Returns:
            cov (np.ndarray) npix X npix : covariance of pixels
            finalmask (np.ndarray) nspec X npix : final mask after iteration
    """

    nspec, npix = flux.shape

    # reject spectra with too many bad pixels
    nbadpix = np.sum(mask != 0, axis=1)  # number of bad pixels for each spectra
    objmask = nbadpix <= maxbadpix
    wmask = np.where(objmask)[0]  # index of good spectra
    nmask = len(wmask)  # number of good spectra

    if nmask < npix:
        raise _LegacyError(
            f"Not enough good spectra: {nmask} spectra have at most maxbadpix = {maxbadpix} bad pixels in mask, and at least npix = {npix} are needed"
        )

    chimask = np.zeros(mask[wmask].shape)

    for sigma in nsigma:  # loop over iteratively masking
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
