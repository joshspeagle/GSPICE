from fractions import Fraction

import numpy as np
import pytest

from gspice import finite_sample_factor

EPS = np.finfo(float).eps


@pytest.mark.parametrize(("n_train", "n_cond", "n_pred"), [(4, 0, 1), (30, 10, 2)])
def test_finite_sample_factor_is_its_formula_to_rounding(n_train, n_cond, n_pred):
    N = n_train
    exact = (
        Fraction(N + 1, N)
        * Fraction(N - 2, N - n_cond - 2)
        * Fraction(N - 1, N - n_cond - n_pred - 2)
    )
    # Six operations round, each by at most eps/2: 3 eps to first order.
    value = finite_sample_factor(n_train, n_cond, n_pred)
    assert abs(Fraction(value) - exact) <= 4 * EPS * exact


def test_finite_sample_factor_broadcasts_integers_of_any_dtype():
    n_train = np.array([[30], [200]], np.uint8)
    n_cond = [10, 20, 0]
    result = finite_sample_factor(n_train, n_cond, np.int16(2))
    expected = [[finite_sample_factor(N, k, 2) for k in n_cond] for N in (30, 200)]
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, expected)
    assert type(finite_sample_factor(np.array(30), np.uint64(10))) is float
    assert finite_sample_factor([30], 10).shape == (1,)
