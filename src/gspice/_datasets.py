import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, logit

from ._spectra import Boolean, Integer, Real, boolean, check_rng, integer, real

MAX_DEPTH = 0.95


@dataclass(frozen=True, eq=False)
class Simulation:
    r"""Synthetic spectra from `simulate`, with their noise-free flux and artifacts.

    The arrays that `simulate` returns are read-only.

    Attributes
    ----------
    flux : ndarray, shape (nspec, npix)
        The flux, with noise and artifacts (flux).
    ivar : ndarray, shape (nspec, npix)
        The inverse variance of the noise, :math:`1/\sigma^2` (flux⁻²).
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
    nspec: Integer,
    npix: Integer = 500,
    *,
    rng: np.random.Generator,
    artifacts: Real = 0.0,
    gaussian: Boolean = False,
) -> Simulation:
    r"""Synthetic continuum-normalized spectra for tests and examples (no network).

    Each spectrum has the continuum 1 and
    :math:`\mathrm{nline} = \lceil \mathrm{npix}/25 \rceil` absorption lines
    :math:`j = 1, \dots, \mathrm{nline}` at evenly spaced centres. Line :math:`j` has a
    Gaussian profile :math:`p_j`, with a standard deviation from 1.5 to 4 pixels.
    The lines depend only on npix. Two latent parameters :math:`\theta` set the depths.
    A spectrum is in population 1, the giants, with the probability 0.3, and then has
    :math:`\theta \sim N((0, 2), I)`. Otherwise :math:`\theta \sim N(0, I)`.
    Line :math:`j` has the depth

    .. math::

        \mathrm{depth}_j = 0.95\,\mathrm{expit}\bigl(\mathrm{logit}(d_j / 0.95)
        + r_j \cdot \theta\bigr),

    with :math:`d_j` from 0.1 to 0.6 its depth at :math:`\theta = 0`, and
    :math:`r_j = 0.5 (\cos j, \sin j)`. The depths are thus correlated, in (0, 0.95),
    and not Gaussian. Then :math:`\mathrm{truth} = 1 - \sum_j \mathrm{depth}_j\, p_j`,
    in (0, 1]. The continuum S/N has the density
    :math:`p(\mathrm{S/N}) \propto (\mathrm{S/N})^{-2.5}` on [30, 400]. The noise is
    Gaussian with :math:`\sigma = \sqrt{\mathrm{truth}} / \mathrm{snr}`.

    With ``gaussian=True``, :math:`\mathrm{truth} = \mu + J \theta` with
    :math:`\theta \sim N(0, I)`. Here :math:`\mu` is the truth of population 0 at
    :math:`\theta = 0`, and the columns of :math:`J` are its derivatives there.
    Every spectrum has the label 0 and the S/N 50, and :math:`\sigma = \sqrt{\mu} / 50`.
    So ``ivar`` is equal for all spectra. Without artifacts,
    :math:`\mathrm{flux} \sim N(\mu, J J^\mathsf{T} + \mathrm{diag}(\sigma^2))`, and the
    scaled spectra (see `scale`) are exactly Gaussian. truth can then leave (0, 1].

    ``round(artifacts * nspec)`` spectra (half to even), chosen at random, get one
    artifact each. Its type is drawn uniformly from three:

    - A cosmic ray adds :math:`h \sigma` to each of 1 to 3 adjacent pixels, with
      :math:`h` log-uniform on [5, 50].
    - A bad column sets the flux of 5 to 20 adjacent pixels to 2.
    - A shifted segment of 200 to 600 adjacent pixels :math:`i` has the flux
      :math:`\mathrm{truth}(i - d)` plus the noise. The shift :math:`d` is from 0.5 to 3
      pixels, of either sign. truth is interpolated linearly, and held constant beyond
      its ends.

    An artifact covers at most npix − 1 pixels, or 1 if npix is 1. Its start is drawn
    uniformly from the starts where it fits. For one state of ``rng``, every value of
    ``artifacts`` gives the same ``ivar``, ``truth``, ``snr`` and ``labels``. It also
    gives the same ``flux`` outside the pixels in ``Simulation.artifacts``.

    Parameters
    ----------
    nspec : int
        The number of spectra; 0 or more.
    npix : int, optional
        The number of pixels; 1 or more.
    rng : numpy.random.Generator
        The source of all randomness. With the same libraries and thread settings, the
        same state gives the same result bit for bit.
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
    Cost: :math:`O(\mathrm{nspec}\,\mathrm{npix})` time and memory, plus the nline
    profiles of npix values each. With ``gaussian=False``, the line sum adds
    :math:`2\,\mathrm{nspec}\,\mathrm{nline}\,\mathrm{npix}` flops.
    """
    nspec = integer(nspec, "nspec", 0)
    npix = integer(npix, "npix", 1)
    check_rng(rng)
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
