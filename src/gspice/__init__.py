"""Gaussian pixelwise conditional estimation for spectra."""

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
    "cholesky_inv",
    "covar",
    "covar_iter_mask",
    "gaussian_estimate",
    "get_chimask",
    "gp_interp",
    "legacy",
    "maskinterp",
    "maskinterp1",
    "standard_scale",
    "submatrix_inv",
    "submatrix_inv_mult",
]
