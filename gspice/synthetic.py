"""
Small synthetic spectra for tests and examples.

The spectra are smooth, correlated and noisy, with a population-wide scatter in
signal-to-noise ratio, so that they have the structure GSPICE assumes (a
dominant low-dimensional covariance plus heteroscedastic noise) without needing
any data files. They are not meant to resemble real stellar spectra.
"""

import numpy as np


def make_synthetic_spectra(nspec = 2000, npix = 120, nmodes = 8, snr = (20., 100.), seed = 0):
    """
    Generate continuum-normalized synthetic spectra with noise

        Parameters:
            nspec (int)  : number of spectra
            npix (int)   : number of pixels
            nmodes (int) : number of independent smooth modes of variation
            snr (tuple)  : (min, max) of the log-uniform distribution of the
                           per-spectrum signal-to-noise ratio
            seed (int)   : seed of the random number generator

        Returns:
            flux (np.ndarray) nspec X npix : noisy spectra
            ivar (np.ndarray) nspec X npix : inverse variance (flat in wavelength
                              for each spectrum, with an extra noisy blue end)
            truth (np.ndarray) nspec X npix : noiseless spectra
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0., 1., npix)
    basis = np.array([np.sin(2 * np.pi * (k + 1) * t + k) * np.exp(-1.5 * k * t / max(nmodes, 1))
                      for k in range(nmodes)])
    amp = 0.04 * 0.7**np.arange(nmodes)
    lines = 0.15 * np.exp(-0.5 * ((t - 0.3) / 0.01)**2) + 0.1 * np.exp(-0.5 * ((t - 0.7) / 0.01)**2)
    truth = 1. - lines + (rng.normal(size = (nspec, nmodes)) * amp) @ basis
    sigma = 1. / np.exp(rng.uniform(np.log(snr[0]), np.log(snr[1]), size = (nspec, 1)))
    shape = 1. + 0.5 * np.exp(-t / 0.1) #noisier blue end
    sigma = sigma * shape[np.newaxis, :]
    flux = truth + rng.normal(size = (nspec, npix)) * sigma
    return flux, 1. / sigma**2, truth


def inject_outliers(flux, ivar, nspike = 150, amplitude = 30., seed = 1, width = 1):
    """
    Add narrow positive spikes (cosmic-ray-like) to randomly chosen spectra

        Parameters:
            flux, ivar : arrays from make_synthetic_spectra (flux is not modified)
            nspike (int)       : number of spectra that receive one spike
            amplitude (float)  : peak height in units of the local noise
            seed (int)         : seed of the random number generator
            width (int)        : number of adjacent pixels affected

        Returns:
            newflux (np.ndarray) : flux with spikes
            truthmask (np.ndarray) nspec X npix : bool, True at the corrupted pixels
    """
    rng = np.random.default_rng(seed)
    nspec, npix = flux.shape
    newflux = flux.copy()
    truthmask = np.zeros(flux.shape, dtype = bool)
    for s in rng.choice(nspec, size = nspike, replace = False):
        i = rng.integers(5, npix - 5 - width)
        sl = slice(i, i + width)
        newflux[s, sl] += amplitude / np.sqrt(ivar[s, sl])
        truthmask[s, sl] = True
    return newflux, truthmask
