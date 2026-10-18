# Vendored from gspice/specUtils.py of the branch revision-2026 (r26) at
# ebbb0653f9b24848572fde4da08f0494488224a0, for tests/test_repair.py only. The
# functions are copied unchanged. The module docstring, the functions that gp_repair
# does not call and the imports that only they use are left out.

import warnings
import numpy as np
from .djs_maskinterp import maskinterp
from .matrixUtils import cholesky_inv, submatrix_inv_mult

from time import time 


def standard_scale(spec, ivar, mask = None, refmean = None):
    """
    Scale input data to have uniform variance per spectrum 

        Parameters:
            spec (np.ndarray): spec (nspec, npix)
            ivar (np.ndarray): ivar (nspec, npix)
            mask (np.ndarray): mask (nspec, npix); 
                               0 == good, default is ivar == 0
            refmean (np.ndarray): optional (npix,) mean of the scaled spectra to
                               subtract. If None (default) the mean over the
                               input spectra is used. Pass the value returned
                               for the training set to scale new spectra
                               consistently.

        Returns:
            spec (np.ndarray): scaled, mean-subtracted flux array (nspec, npix)
            refscale (np.ndarray): (nspec,) scale factor applied to each
                               spectrum, sqrt(mean(ivar)) over unmasked pixels.
                               It is 0 for a spectrum with no unmasked pixels
                               (such a spectrum is set to the mean spectrum and
                               a warning is issued).
            refmean (np.ndarray): (npix,) mean of the scaled spectra that was
                               subtracted

        Use ``unscale`` to transform predictions and variances back to the
        units of the input flux.
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
    if refmean is None:
        refmean = Dvec[~empty].mean(axis = 0) if (~empty).any() else np.zeros(Dvec.shape[1])
    Dvec = Dvec - refmean.reshape(1, -1) 
    Dvec[empty] = 0. #empty spectra become the mean spectrum
    
    return Dvec, refscale, refmean


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


def gp_weights(cov, nguard = 20, irange = None, covinv = None, bruteforce = False, verbose = False):
    """
    Linear weights of the pixelwise conditional estimate (GSPICE routine)

    For every pixel i the Gaussian conditional mean given all pixels outside the
    guard window around i is a fixed linear combination of the (scaled,
    mean-subtracted) spectrum, so predictions for any number of spectra are one
    matrix product, ``pred = spec @ predoverD``. This function computes the
    weights once so that they can be reused (``gp_interp`` calls it).

        Parameters:
            cov (np.ndarray) npix X npix : pixel covariance matrix
            nguard (int) : half-width of the guard window around the predicted pixel;
                           pixels j with |i - j| <= nguard are not used to predict pixel i
            irange (np.array) 2 X 1 : 2 element array [i0, i1] specifying the (inclusive)
                           range of prediction pixels; default is all pixels
            covinv (np.ndarray) npix X npix : inverse of cov (computed if not given)
            bruteforce (bool) : direct inversion of each conditioning submatrix (slow; for testing)
            verbose (bool) : print progress every 100 pixels

        Returns:
            predoverD (np.ndarray) npix X szpred : weights, zero inside each guard window
            predvar (np.ndarray) szpred : conditional variance of each predicted pixel
                           (identical for every spectrum)
    """
    npix = cov.shape[0]
    assert cov.shape[0] == cov.shape[1], "covariance must be square matrix."

    #range of spectral bins to operate on
    if irange is not None: 
        assert np.size(irange) == 2, "Range must have 2 elements."
        i0 = int(irange[0])
        i1 = int(irange[1])
        assert 0 <= i0 <= i1 < npix, "Range must satisfy 0 <= irange[0] <= irange[1] < npix."
    else:
        i0 = 0
        i1 = npix - 1
    
    szpred = i1 - i0 + 1
    predvar = np.zeros(szpred)
    predoverD = np.zeros((npix, szpred))

    #pre-compute inverse covariance
    if covinv is None:
        covinv = cholesky_inv(cov)
    dummy = np.zeros((1, npix))

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
                                                 Dvec=dummy, covinv=covinv, bruteforce = bruteforce,
                                                 compute_pred = False)

        predoverD[:, kstar - i0] = predoverD0
        predvar[kstar - i0] = np.diag(predcovar)

        if verbose and ((i % 100) == 0):
            print(f"Iteration # {i} finished at {time() - t0} seconds.")

    return predoverD, predvar
