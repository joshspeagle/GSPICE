"""Shared fixtures: small synthetic data generated on the fly (no data files needed)."""

import numpy as np
import pytest

from gspice import standard_scale, covar
from gspice.synthetic import make_synthetic_spectra
from helpers import random_spd


@pytest.fixture(scope = "session")
def spd():
    return random_spd(60, np.random.default_rng(11))


@pytest.fixture(scope = "session")
def data():
    """flux, ivar, truth: 2500 spectra of 120 pixels."""
    return make_synthetic_spectra(nspec = 2500, npix = 120, seed = 3)


@pytest.fixture(scope = "session")
def scaled(data):
    """Scaled spectra X, refscale, refmean, and their covariance."""
    flux, ivar, _ = data
    X, rs, rm = standard_scale(flux, ivar)
    cov, _ = covar(X)
    return X, rs, rm, cov
