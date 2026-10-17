import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, logit

from ._spectra import boolean, integer, real

MAX_DEPTH = 0.95


@dataclass(frozen=True, eq=False)
class Simulation:
    """Synthetic spectra from `simulate`, with their noise-free flux and artifacts.

    The arrays that `simulate` returns are read-only.

    Attributes
    ----------
    flux : ndarray, shape (nspec, npix)
        The flux, with noise and artifacts (flux).
    ivar : ndarray, shape (nspec, npix)
        The inverse variance of the noise, 1 / sigma^2 (flux^-2).
    truth : ndarray, shape (nspec, npix)
        The noise-free flux without artifacts (flux).
    artifacts : ndarray of bool, shape (nspec, npix)
        The pixels that an injected artifact covers.
    snr : ndarray, shape (nspec,)
        The continuum signal-to-noise ratio that each spectrum was drawn with
        (unit-free).
    labels : ndarray of int64, shape (nspec,)
        The population of each spectrum: 0, or 1 for the giants.
    """

    flux: NDArray[np.float64]
    ivar: NDArray[np.float64]
    truth: NDArray[np.float64]
    artifacts: NDArray[np.bool_]
    snr: NDArray[np.float64]
    labels: NDArray[np.int64]


def simulate(
    nspec: int,
    npix: int = 500,
    *,
    rng: np.random.Generator,
    artifacts: float = 0.0,
    gaussian: bool = False,
) -> Simulation:
    """Synthetic continuum-normalized spectra for tests and examples (no network).

    Each spectrum has the continuum 1 and nline = ceil(npix / 25) absorption lines
    k = 1, ..., nline at evenly spaced centres, with Gaussian profiles p_k of standard
    deviations from 1.5 to 4 pixels; the lines depend only on npix. Two latent
    parameters theta set the depths. A spectrum is in population 1, the giants, with
    probability 0.3, and has theta ~ N((0, 2), I); otherwise theta ~ N(0, I). Line k
    has the depth 0.95 expit(logit(d_k / 0.95) + r_k . theta), with d_k from 0.1 to
    0.6 its depth at theta = 0 and r_k = 0.5 (cos k, sin k), so that the depths are
    correlated, in (0, 0.95), and not Gaussian. Then truth = 1 - sum_k depth_k p_k,
    in (0, 1]. The continuum S/N has the density p(S/N) ~ (S/N)^-2.5 on [30, 400]
    (paper l.494), and the noise is Gaussian with sigma = sqrt(truth) / snr.

    With gaussian=True, truth = mu + J theta, theta ~ N(0, I), where mu is the truth
    of population 0 at theta = 0 and the columns of J are its derivatives there. Every
    spectrum has label 0 and the S/N 50, and sigma = sqrt(mu) / 50. So ivar is equal
    for all spectra, and without artifacts flux ~ N(mu, J J^T + diag(sigma^2)) and
    the scaled spectra (see `scale`) are exactly Gaussian. truth can then leave
    (0, 1].

    round(artifacts nspec) spectra (half to even), chosen at random, get one artifact
    each, of a type drawn uniformly from three. A cosmic ray adds h sigma to each of
    1 to 3 adjacent pixels, with h log-uniform on [5, 50]. A bad column sets the flux
    of 5 to 20 adjacent pixels to 2. A shifted block of 200 to 600 adjacent pixels j
    has the flux truth(j - d) plus the noise, with d from 0.5 to 3 pixels of either
    sign, and truth interpolated linearly and held constant beyond its ends. An
    artifact covers at most npix - 1 pixels, or 1 if npix is 1, from a uniformly
    drawn start where it fits. For one state of rng, every value of artifacts gives
    the same ivar, truth, snr and labels, and the same flux outside the pixels in
    Simulation.artifacts.

    Parameters
    ----------
    nspec : int
        The number of spectra; >= 0.
    npix : int, optional
        The number of pixels; >= 1.
    rng : numpy.random.Generator
        The source of all randomness. With the same libraries and thread settings,
        the same state gives the same result bit for bit.
    artifacts : float, optional
        The fraction of spectra with an artifact; from 0 to 1.
    gaussian : bool, optional
        Draw exactly Gaussian spectra, as above.

    Returns
    -------
    Simulation
        The spectra.

    Raises
    ------
    TypeError
        If an argument has a type that is not allowed above.
    ValueError
        If an argument has a value that is not allowed above.

    Notes
    -----
    Cost: O(nspec npix) time and memory, plus the nline profiles of npix values each
    and, with gaussian=False, 2 nspec nline npix flops for the line sum.
    """
    nspec = integer(nspec, "nspec", 0)
    npix = integer(npix, "npix", 1)
    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            f"rng: expected a numpy.random.Generator, got {type(rng).__name__}"
        )
    artifacts = real(artifacts, "artifacts")
    if not 0 <= artifacts <= 1:
        raise ValueError(f"artifacts: expected a number from 0 to 1, got {artifacts}")
    gaussian = boolean(gaussian, "gaussian")
    profiles, depth, response = _lines(npix)
    if gaussian:
        mu, J = _linearization(profiles, depth, response)
        truth = mu + rng.standard_normal((nspec, 2)) @ J.T
        labels = np.zeros(nspec, dtype=np.int64)
        # Near 47.0, the median of the S/N law below.
        snr = np.full(nspec, 50.0)
        ivar = np.square(snr)[:, np.newaxis] / mu
    else:
        labels = (rng.random(nspec) < 0.3).astype(np.int64)
        theta = rng.standard_normal((nspec, 2)) + np.outer(labels, (0.0, 2.0))
        depths = MAX_DEPTH * expit(logit(depth / MAX_DEPTH) + theta @ response.T)
        truth = 1 - depths @ profiles
        # The inverse of the distribution function of the S/N, in a form where u = 0
        # gives 30 exactly and u < 1 stays below 400 by about 3.5e-15 relative.
        u = rng.random(nspec)
        snr = 30 * (1 - u * (1 - (30 / 400) ** 1.5)) ** (-1 / 1.5)
        ivar = np.square(snr)[:, np.newaxis] / truth
    # In place, so that the noise holds one temporary of the size of flux.
    flux = rng.standard_normal((nspec, npix))
    flux /= np.sqrt(ivar)
    flux += truth
    flagged = _inject(flux, ivar, truth, rng, round(artifacts * nspec))
    for array in (flux, ivar, truth, flagged, snr, labels):
        array.flags.writeable = False
    return Simulation(flux, ivar, truth, flagged, snr, labels)


def _lines(
    npix: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    # The profiles (nline, npix), the depths at theta = 0 (nline,) and the responses
    # (nline, 2). The fractional parts of k sqrt(2) and k sqrt(3) spread the widths and
    # the depths evenly over their ranges, and the response turns by 1 radian per line.
    # From k = 1, every line, also a single one, depends on both latent parameters.
    nline = math.ceil(npix / 25)
    k = np.arange(1, nline + 1, dtype=np.float64)
    width = 1.5 + 2.5 * (k * math.sqrt(2) % 1)
    depth = 0.1 + 0.5 * (k * math.sqrt(3) % 1)
    # The pixels span [-0.5, npix - 0.5], and the centres divide it evenly.
    centre = (k - 0.5) * npix / nline - 0.5
    offset = (np.arange(npix) - centre[:, np.newaxis]) / width[:, np.newaxis]
    # 0 beyond 30 widths, where a profile is below 1e-195. No sum changes, because every
    # pixel lies within 8.4 widths of a line, and nothing underflows: subnormal tails
    # would raise under a caller's errstate and slow the line sum twofold.
    profiles = np.exp(
        -0.5 * np.square(offset), out=np.zeros_like(offset), where=np.abs(offset) < 30
    )
    response = 0.5 * np.column_stack((np.cos(k), np.sin(k)))
    return profiles, depth, response


def _linearization(
    profiles: NDArray[np.float64],
    depth: NDArray[np.float64],
    response: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    # mu (npix,) and J (npix, 2). At theta = 0, the depth of line k is d_k, and its
    # derivative is d_k (1 - d_k / MAX_DEPTH) r_k.
    slope = depth * (1 - depth / MAX_DEPTH)
    return 1 - depth @ profiles, -profiles.T @ (slope[:, np.newaxis] * response)


def _inject(
    flux: NDArray[np.float64],
    ivar: NDArray[np.float64],
    truth: NDArray[np.float64],
    rng: np.random.Generator,
    count: int,
) -> NDArray[np.bool_]:
    # Adds the artifacts to flux in place, and returns the pixels that they cover.
    nspec, npix = flux.shape
    rows = rng.choice(nspec, count, replace=False)
    # 0, 1 and 2: a cosmic ray, a bad column and a shifted block.
    kinds = rng.integers(3, size=count)
    shortest, longest = np.array([(1, 3), (5, 20), (200, 600)])[kinds].T
    widths = np.minimum(
        rng.integers(shortest, longest, endpoint=True), max(npix - 1, 1)
    )
    starts = rng.integers(npix - widths, endpoint=True)
    # u sets the height of a cosmic ray and the size of the shift of a block.
    u = rng.random(count)
    heights = 5 * 10**u
    shifts = rng.choice((-1.0, 1.0), count) * (0.5 + 2.5 * u)
    pixels = np.arange(npix)
    flagged = np.zeros((nspec, npix), dtype=bool)
    for s, kind, start, width, height, shift in zip(
        rows, kinds, starts, widths, heights, shifts, strict=True
    ):
        run = slice(start, start + width)
        flagged[s, run] = True
        if kind == 0:
            flux[s, run] += height / np.sqrt(ivar[s, run])
        elif kind == 1:
            flux[s, run] = 2.0
        else:
            # The noise stays; the truth under it moves.
            shifted = np.interp(pixels[run] - shift, pixels, truth[s])
            flux[s, run] += shifted - truth[s, run]
    return flagged
