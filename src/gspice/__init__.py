"""Gaussian pixelwise conditional estimation for spectra."""

import logging

from ._model import Model, load, save
from ._predict import Prediction, pixelwise
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

# Without a handler, Python prints the records of level WARNING and above to stderr.
# With one, the application decides where they go.
logging.getLogger("gspice").addHandler(logging.NullHandler())

__version__ = "2.0.0.dev0"

__all__ = [
    "Model",
    "Prediction",
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
    "pixelwise",
    "save",
    "scale",
    "snr",
    "standard_scale",
    "submatrix_inv",
    "submatrix_inv_mult",
    "unscale",
]
