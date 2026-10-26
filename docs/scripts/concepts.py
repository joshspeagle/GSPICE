"""Write the data of the figures and tables of the Concepts pages to docs/src/data/.

    python docs/scripts/concepts.py

Each function draws spectra from gspice.simulate with its own fixed seed, runs gspice
on them, and writes concepts-<name>.json for one figure or table. The figure components
state each setup.
"""

from typing import Any

import numpy as np
from explainers import write
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import binary_dilation

import gspice

DECIMALS = 5
# The defaults of fit, pixelwise and repair that the figures show.
NSIGMA = (20.0, 8.0, 6.0)
NGUARD = 20
NSIGMA_REPLACE, NSIGMA_IGNORE = 10.0, 5.0

Mask = NDArray[np.bool_]


def rounded(values: ArrayLike) -> list[Any]:
    listed: list[Any] = np.round(
        np.asarray(values, dtype=np.float64), DECIMALS
    ).tolist()
    return listed


def run(mask: Mask) -> tuple[int, int]:
    # The first pixel and the length of the one run of True in mask.
    pixels = np.flatnonzero(mask)
    return int(pixels[0]), int(pixels.size)


def dilated(flags: Mask, d: int) -> Mask:
    # The dilation of repair and fit along each spectrum, with nothing beyond the ends.
    structure = np.ones((1, 2 * d + 1), dtype=bool)
    return binary_dilation(np.atleast_2d(flags), structure=structure).reshape(
        flags.shape
    )


def units() -> None:
    rng = np.random.default_rng(14)
    npix = 100
    train = gspice.simulate(4000, npix, rng=rng)
    test = gspice.simulate(1, npix, rng=rng)
    model, _ = gspice.fit(train.flux, train.ivar)
    # A run of bad pixels over the line at pixel 62.
    bad = np.zeros((1, npix), dtype=bool)
    bad[0, 56:69] = True
    x, a, mean = gspice.scale(test.flux, test.ivar, bad, refmean=model.mean[0])
    scaled = x + mean
    start, width = run(bad[0])
    data = {
        "npix": npix,
        "ntrain": len(train.flux),
        "snr": round(float(test.snr[0])),
        "start": start,
        "width": width,
        "interpolated": rounded(scaled[0] / a[0]),
        "a": round(float(a[0]), 2),
        "scaled": rounded(scaled[0]),
        "mean": rounded(mean),
        "x": rounded(x[0]),
    }
    write("concepts-units.json", data)


def clipping() -> None:
    # The passes of fit, one at a time: pass n fits on the mask B_n of the passes
    # before it and computes the in-sample z at every pixel, the interpolated pixels
    # of B_n included. fit with the mask B_n and nsigma=() gives that model, and the
    # prediction of pixelwise with the mask B_n gives that z.
    rng = np.random.default_rng(18)
    nspec, npix, artifacts = 4000, 200, 0.3
    train = gspice.simulate(nspec, npix, rng=rng, artifacts=artifacts)
    B = np.zeros((nspec, npix), dtype=bool)
    passes = []
    for threshold in NSIGMA:
        model, _ = gspice.fit(train.flux, train.ivar, B, nsigma=(), maxbadpix=npix)
        prediction = gspice.pixelwise(model, train.flux, train.ivar, B, nguard=NGUARD)
        x, a, mean = gspice.scale(train.flux, train.ivar, B)
        interpolated = (x + mean) / a[:, np.newaxis]
        z = (interpolated - prediction.flux) / np.sqrt(prediction.var)
        flags = dilated(np.abs(z) > threshold, 1)
        passes.append((threshold, B, z, flags))
        B = B | flags
    _, final = gspice.fit(train.flux, train.ivar, nsigma=NSIGMA, nguard=NGUARD)
    if not np.array_equal(final, B):
        raise SystemExit("the passes do not reproduce the mask of fit")
    # The first spectrum with a bad column, 5 to 20 pixels, in which each pass flags
    # new pixels, and whose flags reach at most 2 pixels beyond the column.
    length = train.artifacts.sum(axis=1)
    new = np.array(
        [np.count_nonzero(flags & ~before, axis=1) for _, before, _, flags in passes]
    )
    outside = np.count_nonzero(B & ~train.artifacts, axis=1)
    shown = np.flatnonzero(
        (length >= 5) & (length <= 20) & np.all(new > 0, axis=0) & (outside <= 2)
    )[0]
    start, width = run(train.artifacts[shown])
    data = {
        "npix": npix,
        "ntrain": nspec,
        "artifacts": artifacts,
        "nguard": NGUARD,
        "spectrum": int(shown),
        "start": start,
        "width": width,
        "flux": rounded(train.flux[shown]),
        "passes": [
            {
                "threshold": threshold,
                # None (JSON null) where an earlier pass masked the pixel.
                "z": [
                    None if masked else round(abs(float(v)), 2)
                    for v, masked in zip(z[shown], before[shown], strict=True)
                ],
                "before": np.flatnonzero(before[shown]).tolist(),
                "flagged": np.flatnonzero(flags[shown] & ~before[shown]).tolist(),
            }
            for threshold, before, z, flags in passes
        ],
    }
    write("concepts-clipping.json", data)


def snr() -> None:
    rng = np.random.default_rng(16)
    nspec, npix, nbin = 20000, 100, 10
    sim = gspice.simulate(nspec, npix, rng=rng)
    _, a, _ = gspice.scale(sim.flux, sim.ivar)
    ratio = gspice.snr(sim.flux, sim.ivar)
    # The pixel whose noise-free flux varies most: the core of a line.
    pixel = int(np.argmax(np.var(sim.truth, axis=0)))
    edges = np.quantile(ratio, np.linspace(0, 1, nbin + 1))
    which = np.clip(np.searchsorted(edges, ratio, side="right") - 1, 0, nbin - 1)
    bins = []
    for b in range(nbin):
        s = which == b
        bins.append(
            {
                "low": round(float(edges[b]), 1),
                "high": round(float(edges[b + 1]), 1),
                # The variance between spectra of a_s t_si, and the mean noise variance
                # a_s^2 / ivar_si, in scaled units squared.
                "signal": round(float(np.var(a[s] * sim.truth[s, pixel])), 2),
                "noise": round(float(np.mean(a[s] ** 2 / sim.ivar[s, pixel])), 3),
            }
        )
    model, _ = gspice.fit(sim.flux, sim.ivar, snr_bins=3)
    data = {
        "nspec": nspec,
        "npix": npix,
        "pixel": pixel,
        "bins": bins,
        "edges": [round(e, 1) for e in model.edges],
    }
    write("concepts-snr.json", data)


def heldout() -> None:
    rng = np.random.default_rng(17)
    npix, nheld, repeats = 100, 2000, 10
    held = gspice.simulate(nheld, npix, rng=rng, gaussian=True)
    # N_k of the pixelwise prediction of each pixel, larger near the ends.
    pixels = np.arange(npix)
    window = np.minimum(pixels + NGUARD + 1, npix) - np.maximum(pixels - NGUARD, 0)
    n_cond = npix - window
    points = []
    for n_train in (110, 120, 140, 170, 200, 300, 500, 1000, 2000, 5000):
        held_out, in_sample = [], []
        for _ in range(repeats):
            train = gspice.simulate(n_train, npix, rng=rng, gaussian=True)
            model, _ = gspice.fit(train.flux, train.ivar, nsigma=())
            for spectra, values in ((held, held_out), (train, in_sample)):
                z = gspice.pixelwise(model, spectra.flux, spectra.ivar, nguard=NGUARD).z
                values.append(np.mean(z**2))
        points.append(
            {
                "ntrain": n_train,
                "held_out": round(float(np.sqrt(np.mean(held_out))), 3),
                "in_sample": round(float(np.sqrt(np.mean(in_sample))), 3),
            }
        )
    # sqrt of the mean f_s over the pixels, the expected mean of z^2.
    curve = np.geomspace(105, 6000, 120).round().astype(np.int64)
    expected = [
        np.sqrt(np.mean(gspice.finite_sample_factor(int(n), n_cond))) for n in curve
    ]
    data = {
        "npix": npix,
        "nguard": NGUARD,
        "nheld": nheld,
        "repeats": repeats,
        "points": points,
        "curve": {"ntrain": curve.tolist(), "rms": rounded(expected)},
    }
    write("concepts-heldout.json", data)


def blockscan() -> None:
    rng = np.random.default_rng(15)
    npix = 400
    train = gspice.simulate(3000, npix, rng=rng)
    clean = gspice.simulate(1000, npix, rng=rng)
    test = gspice.simulate(1, npix, rng=rng)
    model, _ = gspice.fit(train.flux, train.ivar)
    model = gspice.fit_blocks(model, clean.flux, clean.ivar)
    blocks = model.blocks
    assert blocks is not None
    # A broad weak artifact: 3 sigma added to each of 200 pixels.
    offset = 3.0
    artifact = np.zeros(npix, dtype=bool)
    artifact[100:300] = True
    flux = np.array(test.flux)
    ivar = test.ivar
    flux[0, artifact] += offset / np.sqrt(ivar[0, artifact])
    pixelwise = gspice.pixelwise(model, flux, ivar, nguard=NGUARD)
    repaired = gspice.repair(model, flux, ivar)
    plain = gspice.repair(model, flux, ivar, blockwise=False)
    assert repaired.blocks is not None
    if not repaired.blocks[0].any():
        raise SystemExit("repair flags no block of the spectrum")
    # Step 1 of repair, as src/gspice/_repair.py runs it: M1 and I1 from the pixelwise
    # z with dilate=1, and x_blk, with x at M1 | I1 replaced by its conditional mean
    # given the other pixels. The spectrum has no bad pixel.
    z1 = np.abs(pixelwise.z)
    G1 = dilated(z1 > NSIGMA_REPLACE, 1) | dilated(z1 > NSIGMA_IGNORE, 1)
    x_blk = gspice.infill(model, flux, ivar, replace=G1).flux
    # The first round of the block pass: chi^2_A of each block on x_blk, from infill with
    # the guard pixels of the block ignored, over its threshold. The round flags the
    # block of the largest ratio, which the figure shows.
    results, ratios = [], []
    for k, first in enumerate(blocks.starts.tolist()):
        block = np.zeros(npix, dtype=bool)
        block[first : first + blocks.width] = True
        guard = dilated(block, blocks.nguard) & ~block
        result = gspice.infill(model, x_blk, ivar, replace=block, ignore=guard)
        results.append((block, result))
        ratios.append(result.chi2[0] / result.dof[0] / blocks.thresholds[0, k])
    shown = int(np.argmax(ratios))
    ratio = ratios[shown]
    if not (ratio > 1 and repaired.blocks[0, shown]):
        raise SystemExit("the first round of the block pass disagrees with repair")
    block, result = results[shown]
    start, width = run(block)
    data = {
        "npix": npix,
        "ntrain": len(train.flux),
        "nclean": len(clean.flux),
        "snr": round(float(test.snr[0])),
        "offset": offset,
        "artifact": list(run(artifact)),
        "flux": rounded(flux[0]),
        "truth": rounded(test.truth[0]),
        "pixelwise": rounded(pixelwise.flux[0]),
        "above": int(np.count_nonzero(z1[0, artifact] > NSIGMA_IGNORE)),
        "step1": int(np.count_nonzero(G1)),
        "block": {
            "start": start,
            "width": width,
            "nguard": blocks.nguard,
            "mu": rounded(result.flux[0, block]),
            "sd": rounded(np.sqrt(result.var[0, block])),
            "ratio": round(float(ratio), 2),
        },
        "starts": blocks.starts.tolist(),
        "flagged": np.flatnonzero(repaired.blocks[0]).tolist(),
        "found": round(
            float(np.mean((repaired.replaced | repaired.ignored)[0, artifact])), 2
        ),
        "found_plain": round(
            float(np.mean((plain.replaced | plain.ignored)[0, artifact])), 2
        ),
    }
    write("concepts-blockscan.json", data)


def thresholds() -> None:
    rng = np.random.default_rng(19)
    npix, half = 400, 10
    train = gspice.simulate(3000, npix, rng=rng)
    test = gspice.simulate(300, npix, rng=rng, artifacts=1.0)
    model, _ = gspice.fit(train.flux, train.ivar)
    z = np.abs(gspice.pixelwise(model, test.flux, test.ivar, nguard=NGUARD).z)
    replace = z > NSIGMA_REPLACE
    ignore = z > NSIGMA_IGNORE
    # The pixels above nsigma_ignore only.
    between = ignore & ~replace
    # Step 1 of repair, with dilate=1.
    M1 = dilated(replace, 1)
    I1 = dilated(ignore, 1) & ~M1
    # The first cosmic ray, away from the ends, with a pixel above each threshold, so
    # that the figure shows all three kinds. Its caption needs the first pixel above
    # nsigma_ignore only to lie next to the first pixel above nsigma_replace, so that
    # dilation makes it a replaced pixel.
    length = test.artifacts.sum(axis=1)
    for s in np.flatnonzero(length <= 3):
        first, _ = run(test.artifacts[s])
        window = slice(first - half, first + half + 1)
        if first < half or first + half >= npix:
            continue
        above = np.flatnonzero(replace[s, window])
        only = np.flatnonzero(between[s, window])
        if above.size and only.size and abs(int(only[0]) - int(above[0])) == 1:
            break
    else:
        raise SystemExit("no cosmic ray shows both thresholds")
    pixels = np.arange(window.start, window.stop)
    data = {
        "npix": npix,
        "ntrain": len(train.flux),
        "pixels": pixels.tolist(),
        "artifact": np.flatnonzero(test.artifacts[s, window]).tolist(),
        "z": rounded(z[s, window]),
        "replace": np.flatnonzero(replace[s, window]).tolist(),
        "ignore": np.flatnonzero(between[s, window]).tolist(),
        "M1": np.flatnonzero(M1[s, window]).tolist(),
        "I1": np.flatnonzero(I1[s, window]).tolist(),
    }
    write("concepts-thresholds.json", data)


def shifted() -> None:
    rng = np.random.default_rng(20)
    npix, percentile = 1000, 99.9
    train = gspice.simulate(3000, npix, rng=rng)
    clean = gspice.simulate(1000, npix, rng=rng)
    test = gspice.simulate(900, npix, rng=rng, artifacts=1.0)
    model, _ = gspice.fit(train.flux, train.ivar)
    model = gspice.fit_blocks(model, clean.flux, clean.ivar, percentile=percentile)
    # Each spectrum has one artifact, and a shifted segment is the only one longer
    # than 20 pixels.
    rows = np.flatnonzero(test.artifacts.sum(axis=1) > 20)
    flux, ivar = test.flux[rows], test.ivar[rows]
    segments = test.artifacts[rows]
    result = gspice.repair(model, flux, ivar)
    assert result.blocks is not None
    amplitude = np.abs(test.flux - test.truth)[rows] * np.sqrt(ivar)
    strong = segments & (amplitude > 5)
    # The chi^2 test of the known segment: chi^2 per pixel of the segment, given the
    # pixels beyond a guard of NGUARD, against the percentile of the same statistic on
    # the clean spectra.
    found = 0
    for s, segment in enumerate(segments):
        guard = dilated(segment, NGUARD) & ~segment
        statistic = gspice.infill(
            model, flux[[s]], ivar[[s]], replace=segment, ignore=guard
        )
        reference = gspice.infill(
            model, clean.flux, clean.ivar, replace=segment, ignore=guard
        )
        limit = np.percentile(reference.chi2 / reference.dof, percentile)
        found += int(statistic.chi2[0] / statistic.dof[0] > limit)
    lengths = segments.sum(axis=1)
    data = {
        "npix": npix,
        "ntrain": len(train.flux),
        "nclean": len(clean.flux),
        "nshifted": int(rows.size),
        "shortest": int(lengths.min()),
        "longest": int(lengths.max()),
        "flagged": round(float(np.mean(result.blocks.any(axis=1))), 2),
        "known": round(found / rows.size, 2),
        "strong": round(float(np.mean((result.replaced | result.ignored)[strong])), 2),
        "all": round(float(np.mean((result.replaced | result.ignored)[segments])), 2),
    }
    write("concepts-shifted.json", data)


def main() -> None:
    units()
    clipping()
    snr()
    heldout()
    blockscan()
    thresholds()
    shifted()


if __name__ == "__main__":
    main()
