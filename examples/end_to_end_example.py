"""
End-to-end example: clean a covariance, then detect and repair artifacts.

This follows the procedure validated in Section 4 of the paper:

1. Estimate the covariance of the scaled spectra from a contaminated training
   set with iterative clipping (Algorithm 4, cumulative masks).
2. Calibrate the blockwise thresholds on held-out clean spectra.
3. Repair test spectra that contain injected cosmic rays, bad columns and
   wavelength-calibration errors (pixelwise test, blockwise test, replacement of
   |Z| > 10 pixels conditioned on |Z| < 5, exclusion window around extended
   artifacts), with and without the exclusion window.
4. Compare with the known truth.

By default it runs on small synthetic spectra in about a minute with only
numpy and scipy. With --mock it uses a subset of the PHOENIX-based mock spectra
produced by run_mocks.py (see examples/README.md), which needs h5py.

    python end_to_end_example.py
    python end_to_end_example.py --mock data/base.h5 --ntrain 12000 --ntest 3000
"""
import argparse
import time

import numpy as np

import gspice
from gspice.synthetic import make_synthetic_spectra


def inject_synthetic(flux, ivar, rng, p_cosmic = 0.3, p_badcol = 0.15, p_shift = 0.1):
    """Cosmic rays, bad columns and local wavelength shifts for the synthetic spectra."""
    n, m = flux.shape
    x = flux.copy()
    bad = np.zeros(flux.shape, dtype = bool)
    sig = 1. / np.sqrt(ivar)
    pix = np.arange(m)
    for s in range(n):
        if rng.random() < p_shift:
            w = rng.integers(m // 6, m // 3)
            a = rng.integers(5, m - w - 5)
            sl = slice(a, a + w)
            x[s, sl] = np.interp(pix[sl] - rng.choice([-1, 1]) * rng.uniform(0.5, 3.), pix, flux[s])
            bad[s, sl] = True
        if rng.random() < p_badcol:
            w = rng.integers(5, 20)
            a = rng.integers(5, m - w - 5)
            x[s, a:a + w] *= 1 + rng.choice([-1, 1]) * rng.uniform(0.3, 0.8)
            bad[s, a:a + w] = True
        if rng.random() < p_cosmic:
            w = rng.integers(1, 4)
            a = rng.integers(5, m - w - 5)
            x[s, a:a + w] += rng.uniform(5., 50.) * sig[s, a:a + w]
            bad[s, a:a + w] = True
    return x, bad


def load(args):
    """Returns clean flux, ivar, truth, and a function that injects artifacts."""
    n = args.ntrain + args.ncal + args.ntest
    if args.mock is None:
        flux, ivar, truth = make_synthetic_spectra(nspec = n, npix = args.npix, nmodes = 10, seed = 21)
        return flux, ivar, truth, inject_synthetic, dict(nguard = 10, width = 60, stride = 30, halfwidth = 30)
    import h5py
    import make_mocks as mm
    with h5py.File(args.mock, "r") as h:
        flux, ivar, truth = (np.asarray(h[k][:n], dtype = np.float64) for k in ("flux", "ivar", "s_true"))

    def inject(f, iv, rng):
        x, am, _ = mm.inject_artifacts(f, iv, rng)
        return np.asarray(x, dtype = np.float64), am > 0
    return flux, ivar, truth, inject, dict(nguard = 20, width = 150, stride = 75, halfwidth = 150)


def main():
    p = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mock", help = "HDF5 file of mock spectra from run_mocks.py (default: synthetic spectra)")
    p.add_argument("--npix", type = int, default = 300, help = "pixels of the synthetic spectra")
    p.add_argument("--ntrain", type = int, default = 4000)
    p.add_argument("--ncal", type = int, default = 2000, help = "clean held-out spectra for block thresholds")
    p.add_argument("--ntest", type = int, default = 2000)
    p.add_argument("--seed", type = int, default = 1)
    args = p.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    flux, ivar, truth, inject, cfg = load(args)
    nt, nc = args.ntrain, args.ncal
    npix = flux.shape[1]
    ng = cfg["nguard"]
    print(f"{flux.shape[0]} spectra of {npix} pixels")

    # 1. covariance from contaminated training spectra, with iterative clipping
    train, train_bad = inject(flux[:nt], ivar[:nt], rng)
    cov, mask, refmean = gspice.covar_iter_mask(train, ivar[:nt], cumulative = True, nguard = ng,
                                                return_scale = True)
    print(f"training: {train_bad.mean():.2%} of pixels corrupted; iterative clipping masked "
          f"{mask[train_bad].mean():.1%} of them and {mask[~train_bad].mean():.2%} of clean pixels "
          f"[{time.time() - t0:.0f} s]")

    # 2. pixelwise weights and blockwise thresholds, from held-out clean spectra
    weights = gspice.gp_weights(cov, nguard = ng)
    ops = gspice.block_operators(cov, gspice.make_blocks(npix, cfg["width"], cfg["stride"]), nguard = ng)
    cal, _, _ = gspice.standard_scale(flux[nt:nt + nc], ivar[nt:nt + nc], refmean = refmean)
    thresh = gspice.block_thresholds(cal, ops, percentile = 99.9)

    # 3. repair test spectra
    sl = slice(nt + nc, None)
    clean, iv, tr = flux[sl], ivar[sl], truth[sl]
    test, bad = inject(clean, iv, rng)
    sig = 1. / np.sqrt(iv)
    kw = dict(nguard = ng, weights = weights, block_ops = ops, block_thresh = thresh,
              exclusion_halfwidth = cfg["halfwidth"])
    print(f"test: {len(clean)} spectra, {bad.any(axis = 1).mean():.0%} with artifacts")
    for exclude in (False, True):
        new, var, info = gspice.gp_repair(test, iv, cov, refmean, exclude_runs = exclude, **kw)
        M = info['replace']
        used = M | info['ignore']
        chi_before = np.mean(((test - tr) / sig)**2)
        chi_after = np.mean(((new - tr) / sig)**2)
        hit = bad & M
        z = (new - clean)[hit] / np.sqrt(var[hit])
        _, _, cinfo = gspice.gp_repair(clean, iv, cov, refmean, exclude_runs = exclude, **kw)
        print(f"\nexclusion window {'on ' if exclude else 'off'}")
        print(f"  artifact pixels replaced: {hit.sum() / bad.sum():.1%}; replaced or excluded from conditioning: "
              f"{(bad & used).sum() / bad.sum():.1%}")
        print(f"  mean chi^2 per pixel against the truth: {chi_before:.2f} before, {chi_after:.2f} after repair")
        print(f"  replaced artifact pixels: rms (repaired - truth)/sigma = "
              f"{np.sqrt(np.mean(((new - tr) / sig)[hit]**2)):.2f}; "
              f"std of (repaired - clean data)/predicted sigma = {z.std():.2f}")
        print(f"  clean spectra: {cinfo['replace'].sum() / len(clean):.2f} pixels replaced per spectrum, "
              f"{cinfo['replace'].any(axis = 1).mean():.1%} of spectra affected")
    print(f"\ndone in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
