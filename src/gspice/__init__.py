"""Gaussian pixelwise conditional estimation for spectra."""

import logging
import warnings
from typing import TYPE_CHECKING, Any

from . import legacy
from ._blocks import fit_blocks
from ._datasets import Simulation, simulate
from ._diagnostics import finite_sample_factor, noise_audit
from ._model import Blocks, Model, fit, load, save
from ._predict import Prediction, Repaired, infill, pixelwise
from ._spectra import scale, snr, unscale

# Without a handler, Python prints the records of level WARNING and above to stderr.
# With one, the application decides where they go.
logging.getLogger("gspice").addHandler(logging.NullHandler())

__version__ = "2.0.0.dev0"

__all__ = [
    "Blocks",
    "Model",
    "Prediction",
    "Repaired",
    "Simulation",
    "finite_sample_factor",
    "fit",
    "fit_blocks",
    "infill",
    "legacy",
    "load",
    "noise_audit",
    "pixelwise",
    "save",
    "scale",
    "simulate",
    "snr",
    "unscale",
]

# The 2022 names that gspice no longer defines, with their replacements in the new API
# (design section 7.2). None: the function has no replacement outside gspice.legacy.
_REPLACEMENT = {
    "standard_scale": "scale",
    "covar": "fit",
    "covar_iter_mask": "fit",
    "get_chimask": "pixelwise",
    "gp_interp": "pixelwise",
    "gaussian_estimate": "infill",
    "cholesky_inv": None,
    "submatrix_inv": None,
    "submatrix_inv_mult": None,
    "maskinterp1": None,
    "maskinterp": None,
}


# PEP 562: gspice.<name> of a 2022 name returns gspice.legacy.<name> with a
# DeprecationWarning. Type checkers do not see the hook, so they report a misspelled or
# deprecated name instead of typing every attribute of gspice as Any.
if not TYPE_CHECKING:

    def __getattr__(name: str) -> Any:
        if name not in _REPLACEMENT:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        use = f"gspice.legacy.{name}"
        if replacement := _REPLACEMENT[name]:
            use = f"gspice.{replacement}, or {use} to keep the 2022 results"
        warnings.warn(
            f"gspice.{name} is deprecated and will be removed in 3.0.0. Use {use}.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(legacy, name)
