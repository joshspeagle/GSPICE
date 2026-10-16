import numpy as np
import pytest
from numpy.linalg import LinAlgError

from gspice import _linalg
from gspice._linalg import precision

NPIX = 48


def random_spd(kappa):
    # Q diag(lambda) Q^T with log-spaced eigenvalues; the mean of C and C^T is exactly
    # symmetric.
    Q, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((NPIX, NPIX)))
    C = (Q * np.logspace(0, -np.log10(kappa), NPIX)) @ Q.T
    return (C + C.T) / 2


def test_precision_is_exactly_symmetric_and_c_contiguous():
    P = precision(random_spd(1e2))
    assert P.flags.c_contiguous
    np.testing.assert_array_equal(P, P.T)


def test_precision_rejects_a_covariance_with_condition_number_1e14():
    # Design decision A26: rcond is 5.3e-15 for S scaled to unit diagonal.
    with pytest.raises(LinAlgError, match="numerically singular: rcond = "):
        precision(random_spd(1e14))


def test_precision_rejects_a_nan_rcond(monkeypatch):
    monkeypatch.setattr(_linalg.lapack, "dpocon", lambda *args, **kwargs: (np.nan, 0))
    with pytest.raises(LinAlgError, match="rcond = nan"):
        precision(random_spd(1e2))


def test_precision_checks_the_condition_of_s_scaled_to_unit_diagonal():
    # rcond is 4.5e-9 for S scaled to unit diagonal and 5.9e-31 for S itself.
    d = np.logspace(-6, 6, NPIX)
    precision(d[:, np.newaxis] * random_spd(1e8) * d)


def test_precision_rejects_a_matrix_that_is_not_positive_definite():
    S = random_spd(1e2)
    S[5, 5] = -1.0
    with pytest.raises(LinAlgError, match="not positive definite"):
        precision(S)
