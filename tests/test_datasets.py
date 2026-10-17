import numpy as np
import pytest
from scipy import stats
from scipy.special import expit, logit

from _golden import same
from gspice import _datasets, simulate

EPS = np.finfo(float).eps


def arrays(sim):
    return sim.flux, sim.ivar, sim.truth, sim.artifacts, sim.snr, sim.labels


def runs(flagged):
    # The row, first pixel and width of the run of flagged pixels of each spectrum that
    # has one, after a check that no spectrum has two.
    starts = flagged & ~np.pad(flagged, ((0, 0), (1, 0)))[:, :-1]
    assert np.all(starts.sum(axis=1) <= 1)
    rows = np.flatnonzero(starts.any(axis=1))
    return rows, np.argmax(starts[rows], axis=1), flagged[rows].sum(axis=1)


def shift_of(truth, observed, pixels):
    # A block shifted by d has observed_j = (1 - f) truth[j + q] + f truth[j + q + 1]
    # at each of its pixels j, with q = floor(-d) and f = -d - q. For each q, f is
    # fitted by least squares over the pixels whose j + q and j + q + 1 are in the
    # spectrum; returns d and the largest residual for the q that fits best.
    fits = []
    for q in range(-4, 4):
        j = pixels[(pixels + q >= 0) & (pixels + q + 1 < truth.size)]
        a, b = truth[j + q], truth[j + q + 1]
        y = observed[j] - a
        f = (y @ (b - a)) / ((b - a) @ (b - a))
        fits.append((np.abs(y - f * (b - a)).max(), -(q + f)))
    residual, d = min(fits)
    return d, residual


@pytest.mark.parametrize("gaussian", [False, True])
@pytest.mark.parametrize("nspec", [0, 4])
def test_the_arrays_have_their_shapes_and_dtypes_and_are_read_only(nspec, gaussian):
    sim = simulate(
        nspec, 30, rng=np.random.default_rng(0), artifacts=0.5, gaussian=gaussian
    )
    assert [(a.shape, a.dtype) for a in arrays(sim)] == [
        ((nspec, 30), np.float64),
        ((nspec, 30), np.float64),
        ((nspec, 30), np.float64),
        ((nspec, 30), np.bool_),
        ((nspec,), np.float64),
        ((nspec,), np.int64),
    ]
    assert not any(a.flags.writeable for a in arrays(sim))


@pytest.mark.parametrize("gaussian", [False, True])
def test_a_seed_gives_its_spectra_bit_for_bit_and_another_seed_others(gaussian):
    def spectra(seed):
        rng = np.random.default_rng(seed)
        return arrays(simulate(20, 300, rng=rng, artifacts=0.5, gaussian=gaussian))

    assert same(spectra(1), spectra(1))
    assert not np.array_equal(spectra(1)[0], spectra(2)[0])


def test_the_values_are_in_their_ranges():
    sim = simulate(400, 100, rng=np.random.default_rng(2), artifacts=0.5)
    assert np.all((sim.snr >= 30) & (sim.snr <= 400))
    assert set(sim.labels.tolist()) == {0, 1}
    assert np.all((sim.truth > 0) & (sim.truth <= 1))
    # sigma = sqrt(truth) / snr, each side with at most two roundings.
    expected = np.square(sim.snr)[:, np.newaxis] / sim.truth
    np.testing.assert_allclose(sim.ivar, expected, rtol=2 * EPS)
    assert np.isfinite(sim.flux).all()


def test_gaussian_spectra_have_label_0_and_one_ivar_from_the_mean():
    npix = 60
    sim = simulate(50, npix, rng=np.random.default_rng(3), artifacts=0.5, gaussian=True)
    mu, _ = _datasets._linearization(*_datasets._lines(npix))
    assert np.all(sim.labels == 0)
    assert np.all(sim.snr == 50)
    assert np.all(sim.ivar == sim.ivar[0])
    np.testing.assert_allclose(sim.ivar[0], 50**2 / mu, rtol=2 * EPS)
    assert np.isfinite(sim.flux).all()


@pytest.mark.parametrize("gaussian", [False, True])
def test_no_floating_point_error_arises_under_a_strict_errstate(gaussian):
    # The tails of the line profiles underflowed at the default npix.
    with np.errstate(all="raise"):
        simulate(20, rng=np.random.default_rng(10), artifacts=0.5, gaussian=gaussian)


def test_the_gaussian_mode_linearizes_the_line_model_of_population_0():
    profiles, depth, response = _datasets._lines(120)

    def truth(theta):
        return 1 - 0.95 * expit(logit(depth / 0.95) + response @ theta) @ profiles

    mu, J = _datasets._linearization(profiles, depth, response)
    np.testing.assert_allclose(mu, truth(np.zeros(2)), rtol=0, atol=4 * EPS)
    # The central difference errs by about h^2 |truth'''| / 6 < 1e-12 and by the
    # rounding of truth over 2 h, about eps / h = 2e-10.
    h = 1e-6
    for i, step in enumerate(h * np.eye(2)):
        difference = (truth(step) - truth(-step)) / (2 * h)
        np.testing.assert_allclose(J[:, i], difference, rtol=0, atol=1e-9)


def test_each_artifact_changes_one_run_of_flagged_pixels_as_its_type_states():
    nspec, npix = 299, 400
    clean = simulate(nspec, npix, rng=np.random.default_rng(4))
    sim = simulate(nspec, npix, rng=np.random.default_rng(4), artifacts=0.3)
    assert not clean.artifacts.any()
    for name in ("ivar", "truth", "snr", "labels"):
        assert same(getattr(sim, name), getattr(clean, name))
    good = ~sim.artifacts
    assert same(sim.flux[good], clean.flux[good])
    rows, starts, widths = runs(sim.artifacts)
    assert rows.size == round(0.3 * nspec) == 90
    found = set()
    shifts = []
    for s, start, width in zip(rows, starts, widths, strict=True):
        run = slice(start, start + width)
        change = sim.flux[s] - clean.flux[s]
        if width <= 3:
            found.add("cosmic ray")
            # h sigma at each pixel, 5 <= h <= 50. The sum flux + h sigma rounds by
            # at most eps |flux| / 2, which moves h by that times sqrt(ivar), below
            # 1e-12 here.
            h = change[run] * np.sqrt(sim.ivar[s, run])
            assert np.ptp(h) <= 1e-9
            assert 5 - 1e-9 <= h[0] <= 50 + 1e-9
        elif width <= 20:
            found.add("bad column")
            assert np.all(sim.flux[s, run] == 2.0)
        else:
            found.add("shifted block")
            assert 200 <= width <= npix - 1
            # np.interp rounds j - d by up to npix eps / 2, which moves its value by
            # that times |b - a| <= 1, and the change rounds by about eps.
            observed = sim.truth[s] + change
            d, residual = shift_of(
                sim.truth[s], observed, np.arange(start, start + width)
            )
            assert residual <= 1e-12
            assert 0.5 - 1e-9 <= abs(d) <= 3 + 1e-9
            shifts.append(d)
    assert found == {"cosmic ray", "bad column", "shifted block"}
    shifts = np.array(shifts)
    assert np.any(shifts < 0) and np.any(shifts > 0)
    # Interpolated, not rounded to whole pixels.
    assert np.any(np.abs(shifts - np.round(shifts)) > 0.1)


@pytest.mark.parametrize("npix", [1, 5, 50])
def test_every_artifact_type_fits_a_spectrum_of_few_pixels(npix):
    sim = simulate(60, npix, rng=np.random.default_rng(5), artifacts=1.0)
    rows, _, widths = runs(sim.artifacts)
    np.testing.assert_array_equal(rows, np.arange(60))
    # An artifact leaves a pixel clean unless npix is 1.
    cap = max(npix - 1, 1)
    allowed = {
        *range(1, min(3, cap) + 1),
        *range(min(5, cap), min(20, cap) + 1),
        *range(min(200, cap), min(600, cap) + 1),
    }
    assert set(widths.tolist()) <= allowed
    assert np.isfinite(sim.flux).all()


@pytest.mark.slow
def test_the_snr_follows_the_power_law_and_30_percent_are_giants():
    n = 4000
    sim = simulate(n, 1, rng=np.random.default_rng(6))
    # Each exact test rejects a correct implementation with probability 0.001.
    law = stats.truncpareto(1.5, 400 / 30, scale=30)
    assert stats.kstest(sim.snr, law.cdf).pvalue > 0.001
    assert stats.binomtest(np.count_nonzero(sim.labels), n, 0.3).pvalue > 0.001


@pytest.mark.slow
def test_the_line_depths_follow_the_latent_parameters_of_each_population():
    # At npix 1, the one line k = 1 has the profile 1, so truth = 1 - depth and
    # logit(depth / 0.95) - logit(d_1 / 0.95) = r_1 . theta, with r_1 = 0.5 (cos 1,
    # sin 1): N(0, 0.5^2) in population 0 and N(sin 1, 0.5^2) in population 1.
    sim = simulate(4000, 1, rng=np.random.default_rng(9))
    d1 = 0.1 + 0.5 * (np.sqrt(3) % 1)
    x = logit((1 - sim.truth[:, 0]) / 0.95) - logit(d1 / 0.95)
    # Each exact test rejects a correct implementation with probability 0.001.
    for label, mean in ((0, 0.0), (1, np.sin(1))):
        law = stats.norm(mean, 0.5)
        assert stats.kstest(x[sim.labels == label], law.cdf).pvalue > 0.001


@pytest.mark.slow
def test_gaussian_flux_has_the_stated_mean_and_covariance():
    n, npix = 20_000, 40
    sim = simulate(n, npix, rng=np.random.default_rng(7), gaussian=True)
    mu, J = _datasets._linearization(*_datasets._lines(npix))
    L = np.linalg.cholesky(J @ J.T + np.diag(mu / 50**2))
    # With Sigma = L L^T, y = L^-1 (flux - mu) is iid N(0, I) if flux ~ N(mu, Sigma).
    # Then each entry of sqrt(n) mean(y) is N(0, 1); and (n - 1) cov(y) is Wishart, so
    # the diagonal of cov(y) has the standard deviation sqrt(2 / (n - 1)) about 1 and
    # the other entries sqrt(1 / (n - 1)) about 0, close to normal at this n. Over the
    # m entries tested, z = Phi^-1(1 - 0.001 / (2 m)) standard deviations each bound
    # the probability of a failure by 0.001 (Bonferroni).
    y = np.linalg.solve(L, (sim.flux - mu).T).T
    upper = np.triu_indices(npix)
    z = stats.norm.isf(0.001 / (2 * (npix + upper[0].size)))
    assert np.all(np.abs(y.mean(axis=0)) <= z / np.sqrt(n))
    deviation = (np.cov(y, rowvar=False) - np.eye(npix))[upper]
    sd = np.where(upper[0] == upper[1], np.sqrt(2 / (n - 1)), np.sqrt(1 / (n - 1)))
    assert np.all(np.abs(deviation) <= z * sd)


@pytest.mark.slow
def test_the_noise_has_zero_mean_and_unit_variance():
    sim = simulate(1000, 400, rng=np.random.default_rng(8))
    r = (sim.flux - sim.truth) * np.sqrt(sim.ivar)
    # For n iid N(0, 1) values, the mean has the standard error 1 / sqrt(n) and the
    # variance sqrt(2 / (n - 1)); 4 standard errors fail with probability 6.3e-5 each.
    n = r.size
    assert abs(r.mean()) <= 4 / np.sqrt(n)
    assert abs(r.var(ddof=1) - 1) <= 4 * np.sqrt(2 / (n - 1))
