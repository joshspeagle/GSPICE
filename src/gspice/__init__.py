"""Gaussian pixelwise conditional estimation for spectra."""

from ._model import Model, load, save
from ._spectra import scale, snr, unscale
from .legacy import (
    cholesky_inv,
    covar,
    covar_iter_mask,
    gaussian_estimate,
    get_chimask,
    gp_interp,
    maskinterp,
    maskinterp1,
    standard_scale,
    submatrix_inv,
    submatrix_inv_mult,
)

__version__ = "2.0.0.dev0"

__all__ = [
    "Model",
    "cholesky_inv",
    "covar",
    "covar_iter_mask",
    "gaussian_estimate",
    "get_chimask",
    "gp_interp",
    "legacy",
    "load",
    "maskinterp",
    "maskinterp1",
    "save",
    "scale",
    "snr",
    "standard_scale",
    "submatrix_inv",
    "submatrix_inv_mult",
    "unscale",
]
