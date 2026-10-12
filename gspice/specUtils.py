import warnings
import numpy as np
from .djs_maskinterp import maskinterp
#from .gspiceMain import gp_interp
from .matrixUtils import cholesky_inv, submatrix_inv_mult

from time import time 

def standard_scale(spec, ivar, mask = None):
    """
    Scale input data to have uniform variance per spectrum 

        Parameters:
            spec (np.ndarray): spec (nspec, npix)
            ivar (np.ndarray): ivar (nspec, npix)
            mask (np.ndarray): mask (nspec, npix); 
                               0 == good, default is ivar == 0

        Returns:
            spec (np.ndarray): scaled, mean-subtracted flux array (nspec, npix)
            refscale (np.ndarray): (nspec,) scale factor applied to each
                               spectrum, sqrt(mean(ivar)) over unmasked pixels.
                               It is 0 for a spectrum with no unmasked pixels
                               (such a spectrum is set to the mean spectrum and
                               a warning is issued).
            refmean (np.ndarray): (npix,) mean of the scaled spectra that was
                               subtracted
    """

    #if no mask is passed, use ivar == 0 as mask (setting infinity error as bad pixel)
    if mask is None:
        pixmask = (ivar == 0)
    else:
        pixmask = np.where(mask == 0, 0, 1) #scale mask to 0 or 1
    
    #interpolate over masked pixels in the spectral direction
    Dvec = maskinterp(yval = spec, mask = pixmask, axis = 0)#, const = ).astype(np.float64) #dependent on pydl ##/const??
    
    #renormalize each spectra by sqrt(mean(ivar))
    wt = 1 - pixmask # set weight such that only good pixels contribute 
    ngood = np.sum(wt, axis = 1)
    empty = ngood == 0 #spectra with no good pixels would give 0/0 = NaN
    if empty.any():
        warnings.warn(f"{empty.sum()} spectra have no unmasked pixels; they are "
                      "set to the mean spectrum (refscale = 0). Remove them "
                      "before estimating a covariance.", RuntimeWarning, stacklevel = 2)
    meanivar = np.sum(ivar * wt, axis = 1)/np.where(empty, 1, ngood)
    refscale = np.sqrt(meanivar)
    Dvec = np.where(empty[:, np.newaxis], 0., Dvec) #drop (possibly NaN) data of empty spectra
    Dvec = Dvec * refscale[:, np.newaxis] #rescale data as roughly data/sigma
    refmean = Dvec[~empty].mean(axis = 0) if (~empty).any() else np.zeros(Dvec.shape[1])
    Dvec = Dvec - refmean.reshape(1, -1) 
    Dvec[empty] = 0. #empty spectra become the mean spectrum
    
    return Dvec, refscale, refmean

def covar(spec, checkmean = False): ##DONE
    """
    Compute covariance of cleaned data vector using BLAS 

        Parameters:
            spec (np.ndarray) nspec X npix: spectral data (nspec, npix); must be float64

        Returns:
            cov (np.ndarray) npix X npix: covariance matrix of the pixels
            refmean (np.array) 1 X nspec: reference mean of each spectra
    """

    nspec, npix = spec.shape

    #check data type
    assert spec.dtype == 'float64', "spec.dtype must be float64"

    #make columns of Dmask mean 0 for covariance computation
    refmean = spec.mean(axis = 0)
    spec = spec - refmean.reshape(1, -1) 

    #verify that mean subtraction worked
    if checkmean is True:
        mnd = spec.mean(axis = 0)
        print(f"min = {mnd.min()}, max = {mnd.max()}, std = {mnd.std(ddof = 1)}") #only on request
    
    #compute covariance
    cov = (spec.T @ spec)/(nspec - 1)
    return cov, refmean 

from scipy.ndimage import binary_dilation as dilate
def get_chimask (flux, ivar, mask, nsigma, nguard = 20): ##DONE
    """
    Compute mask of outliers with respect to GSPICE posterior covariance.
    
        Parameters:
            flux (np.ndarray): flux (nspec, npix)
            ivar (np.ndarray): ivar (nspec, npix)
            mask (np.ndarray): mask (nspec, npix) (0 == good)
            nsigma (int)     : threshold to clip data at 
            nguard (int)     : half-width of the guard window (default 20)
        Returns:
            chimask (np.ndarray): mask of nonconformat pixels (0 == good)

        Comments:
            The covariance is estimated from the (interpolated, scaled) input
            spectra themselves and the Z-score is evaluated on those same
            mask-interpolated data, so pixels that are masked on input are
            tested at their interpolated values.
    """
    
    #interporlate over masked pixels in the spectral direction
    Dvec, refscale, refmean = standard_scale(flux, ivar, mask)

    #obtain empirical covariance for this data array
    covmat, _ = covar(Dvec) 

    #compute GSPICE predicted mean and covariance
    pred, predvar = gp_interp(Dvec, covmat, nguard=nguard)

    #compute z-score 
    chi = (Dvec - pred)/np.sqrt(predvar)

    #clip at nsigma, dilate mask by 1 pixel
    flag = np.abs(chi) >= nsigma
    chimask = dilate(flag, np.array([[1, 1, 1]])).astype(flag.dtype)
    
    return chimask 

def gaussian_estimate(icond, ipred, cov, Dvec, covinv, bruteforce = False,
                      compute_pred = True):
    """
    Computes Gaussian conditional estimate

        Parameters:
            icond (np.array) : (npix,) flags of pixels to condition on (nonzero == use)
            ipred (np.array) : (npix,) flags of pixels to predict (nonzero == predict)
            cov (np.ndarray) npix X npix    : covariance matrix
            Dvec (np.ndarray) nspec X npix  : data vector 
            covinv (np.ndarray) npix X npix : inverse of cov, 
                                              to speed up computation
            bruteforce (bool) : flag to use bruteforce code
            compute_pred (bool) : if False, skip the nspec X npix product that
                                  gives predkstar and return None in its place.
                                  gp_interp does this because it forms all
                                  predictions with one matrix product at the end.

        Returns:
            predoverD (np.ndarray) npix X kstar : matrix to multiply by spec to 
                                                  get predicted spectra; 
                                                  kstar refers to row(s) to be
                                                  predicted
            predcovar (np.ndarray) kstar X kstar : predicted covariance matrix
            predkstar (np.ndarray) nspec X kstar : predicted values (None if
                                                  compute_pred is False)
            kstar (np.ndarray) : indices of the predicted pixels

        Both the default and the bruteforce code paths return the same four
        items. (Before version 1.0.0 the bruteforce path returned
        (predcovar, predkstar, kstar).)
    """

    nspec, npix = Dvec.shape

    k = np.where(icond)[0] #where is good data
    nk = len(k) #number of good pixels
    kstar = np.where(ipred)[0]
    nkstar = len(kstar) #number of pixels to predict on
    cov_kkstar = cov[np.ix_(kstar, k)]
    cov_kstark     = cov_kkstar.T
    cov_kstarkstar = cov[np.ix_(kstar, kstar)]

    if bruteforce:
        cov_kk = cov[np.ix_(k, k)]

        #Cholesky inversion is faster than SVD here
        icov_kk = cholesky_inv(cov_kk)
        del cov_kk #save memory

        #compute prediction covariance (See RW, Ch. 2)
        predcovar = cov_kstarkstar - cov_kkstar @ icov_kk @ cov_kstark
        
        #compute icov_kk (dot) cov_kstark, zero-padded to npix rows
        predoverD = np.zeros((npix, nkstar))
        predoverD[k, :] = (cov_kkstar @ icov_kk).T
        del icov_kk 
    else:
        #compute icov_kk (dot) cov_kstark using GSPICE
        Y = cov[:, kstar]
        Minvy = covinv @ Y
        
        #Ainvy is icov_kk (dot) cov_kstarK
        #Ainvy0 is Ainvy zero-padded
        Ainvy0 = submatrix_inv_mult(M=cov, Minv=covinv, imask=icond, Y = Y,
                                    MinvY=Minvy, pad = True)
        predoverD = Ainvy0

        #compute prediction covariance (See RW, Ch. 2)
        predcovar = cov_kstarkstar - Y.T @ Ainvy0 #precision up to 1e-8 with Julia

    predkstar = Dvec @ predoverD if compute_pred else None #takes time; optional
    return predoverD, predcovar, predkstar, kstar

def gp_interp(spec, cov, nguard = 20, irange = None, bruteforce = False, verbose = False):
    """
    Computes pixelwise conditional prediction for each pixel (GSPICE routine)

        Parameters:
            spec (np.ndarray) nspec X npix : spectra matrix 
            cov (np.ndarray) npix X npix : pixel covariance matrix
            nguard (int) : half-width of the guard window around the predicted pixel;
                           pixels j with |i - j| <= nguard are not used to predict pixel i
            irange (np.array) 2 X 1 : 2 element array [i0, i1] specifying the (inclusive) range
                           of prediction pixels; default is all pixels
            bruteforce (bool) : use direct inversion of each conditioning
                           submatrix instead of the Schur-complement update
                           (slow; for testing)
            verbose (bool) : print progress every 100 pixels
        
        Returns:
            pred (np.ndarray) nspec X npix : GSPICE predicted spectra mean
            predcovar (np.ndarray) nspec X npix : GSPICE predicted posterior spectra variance
                           (the same for every spectrum, repeated down the rows)
    """
    
    #shape of input spectra array
    nspec, npix = spec.shape
    #check proper dimensionality
    assert cov.shape[0] == cov.shape[1], "covariance must be square matrix."
    assert npix == cov.shape[0], "spectra and covariance have incompatible dimensions."
    
    #range of spectral bins to operate on
    if irange is not None: 
        assert np.size(irange) == 2, "Range must have 2 elements."
        i0 = int(irange[0])
        i1 = int(irange[1])
        assert 0 <= i0 <= i1 < npix, "Range must satisfy 0 <= irange[0] <= irange[1] < npix."
    else:
        i0 = 0
        i1 = npix - 1
    
    #allocate output arrays
    szpred = i1 - i0 + 1
    predvar = np.zeros((nspec, szpred))
    predoverD = np.zeros((npix, szpred))

    #pre-compute inverse covariance
    covinv = cholesky_inv(cov)

    #loop over pixels
    t0 = time() #start timing
    for i in range(i0, i1 + 1):
        #ipred == 1 for pixels to be predicted
        ipred = np.zeros(npix)
        ipred[i] = 1

        #condition on reference pixels specified by icond
        icond = np.ones(npix)
        j0 = np.max([0, i - nguard])
        j1 = np.min([i + nguard + 1, npix])
        icond[j0:j1] = 0

        predoverD0, predcovar, _, kstar = gaussian_estimate(icond=icond, ipred=ipred, cov=cov,
                                                 Dvec=spec, covinv=covinv, bruteforce = bruteforce,
                                                 compute_pred = False)

        predoverD[:, kstar - i0] = predoverD0
        predvar[:, kstar - i0] = np.diag(predcovar)

        if verbose and ((i % 100) == 0):
            print(f"Iteration # {i} finished at {time() - t0} seconds.")

    #all predictions with one matrix product
    pred = spec.dot(predoverD)
    if verbose:
        print(f"Matrix multiplication time: {time() - t0}")

    return pred, predvar 

def covar_iter_mask(flux, ivar, mask = None, nsigma = np.array([20, 8, 6]), maxbadpix = 64,
                    nguard = 20, verbose = False):
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
            nguard (int)                   : half-width of the guard window
            verbose (bool)                 : print progress

        Returns:
            cov (np.ndarray) npix X npix : covariance of pixels
            finalmask (np.ndarray) nspec X npix : final mask after iteration 
                                             (all ones for rejected spectra)

        Comments:
            Pass n uses input_mask | chimask_{n-1} only, so pixels flagged in
            pass n-2 or earlier but not again in pass n-1 are unmasked again,
            and the final mask is input | chimask_{N-1} | chimask_N. This
            differs from Algorithm 4 of the paper (M <- M or M_Z).
    """

    t0 = time()

    nspec, npix = flux.shape
    if mask is None:
        mask = (ivar == 0)

    #reject spectra with too many bad pixels
    nbadpix = np.sum(mask != 0, axis = 1) #number of bad pixels for each spectra
    objmask = nbadpix <= maxbadpix #flat for objects to be discarded (too many bad pix)
    wmask = np.where(objmask)[0] #index of good spectra
    nmask = len(wmask) #number of good spectra

    assert nmask >= npix, "Not enough good spectra"

    chimask = np.zeros(mask[wmask].shape)
    
    for sigma in nsigma: #loop over iteratively masking
        if verbose:
            print(f"Pass nsigma = {sigma}")
        thismask = np.logical_or(chimask, (mask[wmask] != 0))
        
        chimask = get_chimask(flux = flux[wmask], ivar = ivar[wmask], mask = thismask, 
                            nsigma = sigma, nguard = nguard)
        if verbose:
            print(f"Mean chimask = {np.mean(chimask)}")
            print(f"Time: {time()- t0} seconds.")
        t0 = time()

    finalmask = np.ones((nspec, npix))
    finalmask[wmask] = np.logical_or(thismask, chimask)

    spec, _, _ = standard_scale(spec=flux[wmask], ivar= ivar[wmask], mask=finalmask[wmask])
    cov, _ = covar(spec)

    return cov, finalmask
