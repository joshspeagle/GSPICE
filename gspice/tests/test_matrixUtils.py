import numpy as np
import pytest

from gspice import cholesky_inv, submatrix_inv, submatrix_inv_mult


def test_cholesky_inv_small():
    m = np.array([[2., 1., 0.], [1., 2., 1.], [0., 1., 2.]])
    expected = 0.25 * np.array([[3., -2., 1.], [-2., 4., -2.], [1., -2., 3.]])
    np.testing.assert_allclose(cholesky_inv(m), expected, atol = 1e-14)


def test_cholesky_inv_random(spd):
    np.testing.assert_allclose(spd @ cholesky_inv(spd), np.eye(spd.shape[0]), atol = 1e-10)


def test_submatrix_inv_small():
    m = np.array([[2., 1., 0.], [1., 2., 1.], [0., 1., 2.]])
    out = submatrix_inv(m, np.linalg.inv(m), np.array([1, 1, 0]))
    np.testing.assert_allclose(out, np.array([[2., -1.], [-1., 2.]]) / 3., atol = 1e-14)


def test_submatrix_inv_no_removal(spd):
    inv = np.linalg.inv(spd)
    np.testing.assert_allclose(submatrix_inv(spd, inv, np.ones(spd.shape[0])), inv)


@pytest.mark.parametrize("removed", [slice(11, 51), slice(0, 1), slice(40, 60), slice(5, 6)])
def test_submatrix_inv_matches_direct(spd, removed):
    imask = np.ones(spd.shape[0])
    imask[removed] = 0
    k = np.flatnonzero(imask)
    expected = np.linalg.inv(spd[np.ix_(k, k)])
    out = submatrix_inv(spd, np.linalg.inv(spd), imask)
    np.testing.assert_allclose(out, expected, rtol = 1e-7, atol = 1e-8 * np.abs(expected).max())
    brute = submatrix_inv(spd, np.linalg.inv(spd), imask, bruteforce = True)
    np.testing.assert_allclose(brute, expected, rtol = 1e-8, atol = 1e-10 * np.abs(expected).max())


@pytest.mark.parametrize("removed", [slice(11, 51), slice(20, 21), slice(0, 5)])
def test_submatrix_inv_mult_matches_direct(spd, removed):
    rng = np.random.default_rng(2)
    n = spd.shape[0]
    imask = np.ones(n)
    imask[removed] = 0
    k = np.flatnonzero(imask)
    Y = rng.normal(size = (n, 3))
    inv = np.linalg.inv(spd)
    expected = np.linalg.solve(spd[np.ix_(k, k)], Y[k])
    scale = np.abs(expected).max()
    unpadded = submatrix_inv_mult(spd, inv, imask, Y, inv @ Y, pad = False)
    np.testing.assert_allclose(unpadded, expected, rtol = 1e-6, atol = 1e-8 * scale)
    padded = submatrix_inv_mult(spd, inv, imask, Y, inv @ Y, pad = True)
    np.testing.assert_allclose(padded[k], expected, rtol = 1e-6, atol = 1e-8 * scale)
    assert np.all(padded[imask == 0] == 0.)
    brute = submatrix_inv_mult(spd, inv, imask, Y, inv @ Y, bruteforce = True)
    np.testing.assert_allclose(brute, expected, rtol = 1e-7, atol = 1e-8 * scale)
