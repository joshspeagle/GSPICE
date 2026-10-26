"""Write the data of the explainers and of the figures of the quickstart and guides.

    python docs/scripts/explainers.py

It fits a model with gspice.fit to spectra of gspice.simulate(gaussian=True), which
all have the same ivar, and runs the code of the quickstart page and of three guides.
It writes to docs/src/data/:

- explainer.json: the mean and the covariance of the model in flux units, held-out
  spectra with their noise-free flux, rounded, and the range of the ±2σ bands over
  every block of 1 to WIDEST pixels;
- explainer-check.json: for docs/tests/gauss.test.mjs, the conditional mean and
  standard deviation of a few masks on the rounded data, from numpy and from
  gspice.infill;
- quickstart.json: one spectrum of the quickstart, with its prediction;
- guide-infill.json: one spectrum of Fill known bad pixels, with the conditional mean
  and three draws of its gap;
- guide-outliers.json: the histogram of the z-scores of the clean spectra of Find
  outliers, and the count that a Gaussian distribution gives in each bin;
- covariance.json: for the covariance explorer, NSCATTER training spectra of the
  model, as integers of FLUX_UNIT from the stored mean, and the pair it opens on;
- two-pixel.json: the pixel pairs of the two-pixel explainer;
- guard.json: the defect that the guard window explainer adds to a held-out spectrum,
  and the range of its flux and predictions;
- repair-steps.json: two spectra of Find and repair artifacts, with their masks after
  each step of gspice.repair, from gspice.pixelwise and gspice.repair, its result, and
  whether minfrac stopped step 4;
- covariance-check.json, two-pixel-check.json and guard-check.json: for
  docs/tests/explainers.test.mjs, the heat-map step of each cell, the conditional
  mean and standard deviation of gspice.infill, and the result of gspice.pixelwise,
  on the rounded data.

It prints the sizes of the files and the largest change that the rounding makes to the
explainer's prediction and standard deviation, and to the training spectra. It fails if
the first reaches 1e-3, or if the masks of the repair steps do not relate as repair
states.
"""

import contextlib
import inspect
import io
import itertools
import json
import re
import runpy
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.linalg import cho_factor, cho_solve
from scipy.ndimage import maximum_filter1d

import gspice

DOCS = Path(__file__).parents[1]
OUT = DOCS / "src" / "data"
PAGES = DOCS / "src" / "content" / "docs"
NPIX, NTRAIN, NEXAMPLE = 120, 4000, 8
# Decimals kept in explainer.json: flux and mean, then the covariance (flux^2).
DECIMALS, COV_DECIMALS = 6, 7
# The widest masked block of the explainer, which leaves 20 pixels to condition on.
WIDEST = 100
# The spectrum of the quickstart that its figure shows.
SHOWN = 3
# The draws that the figure of Fill known bad pixels shows.
DRAWS = 3
# The bins of the z-scores of Find outliers, and the threshold that its figure marks;
# the first and last bins also count the z-scores beyond them.
Z_EDGES = np.linspace(-10, 10, 41)
# Copies the `> 5` of `flagged = (np.abs(z) > 5)` in guides/outliers.mdx.
THRESHOLD = 5
# The training spectra that the covariance explorer shows, stored as integers of
# FLUX_UNIT from the rounded mean.
NSCATTER, FLUX_UNIT = 100, 1e-3
# The heat map of docs/brand.md has five steps, at correlations -1, -0.5, 0, 0.5 and
# 1; each cell takes the nearest one.
HEAT_EDGES = (-0.75, -0.25, 0.25, 0.75)
# The two-pixel explainer pairs the pixel of largest variance with the pixels whose
# correlation with it comes nearest to each of these.
PAIR_CORRELATIONS = (0.95, 0.5, 0.0, -0.95)
# The guard window explainer adds DEFECT_HEIGHT to the flux of 1 to DEFECT_WIDEST
# pixels from DEFECT_START of held-out spectrum GUARD_SPECTRUM, and allows nguard from
# 0 to GUARD_WIDEST. GUARD_CHECKS are the nguard and defect widths of its test. A bad
# column of gspice.simulate sets the flux to 2, about 1 above the continuum; a smaller
# defect hardly moves the prediction of its neighbours in this model.
GUARD_SPECTRUM, DEFECT_START, DEFECT_HEIGHT = 0, 50, 1.0
DEFECT_WIDEST, GUARD_WIDEST = 20, 30
GUARD_CHECKS = ((0, 1), (3, 8), (8, 8), (20, 20), (30, 5))
# The test spectra of Find and repair artifacts that the stepper shows: a bad column,
# whose steps 1, 3 and 4 change the masks, and a shifted segment, whose step 2 flags a
# block. The script checks both.
REPAIR_SHOWN = (19, 63)
# Decimals of the data that the explainers only draw.
PLOT_DECIMALS = 4
# The defaults of gspice.repair, which the guide uses.
REPAIR_OPTIONS = {
    name: parameter.default
    for name, parameter in inspect.signature(gspice.repair).parameters.items()
}

Floats = NDArray[np.float64]


def conditional(
    mean: Floats, cov: Floats, x: Floats, masked: NDArray[np.bool_]
) -> tuple[Floats, Floats]:
    O = ~masked
    # cho_factor also fails if the rounded covariance is not positive definite, as the
    # Cholesky factor of docs/src/lib/gauss.ts would in the browser.
    solve = cho_solve(cho_factor(cov[np.ix_(O, O)]), cov[np.ix_(O, masked)])
    mu = mean[masked] + solve.T @ (x[O] - mean[O])
    var = np.diag(cov[np.ix_(masked, masked)]) - np.sum(
        cov[np.ix_(O, masked)] * solve, axis=0
    )
    return mu, np.sqrt(var)


def block(start: int, width: int) -> NDArray[np.bool_]:
    masked = np.zeros(NPIX, dtype=bool)
    masked[start : start + width] = True
    return masked


def explainer(
    rng: np.random.Generator,
) -> tuple[dict[str, Floats], Floats, Floats, float]:
    train = gspice.simulate(NTRAIN, NPIX, rng=rng, gaussian=True)
    held_out = gspice.simulate(NEXAMPLE, NPIX, rng=rng, gaussian=True)
    # Every spectrum has the same ivar, so the same scale a: the flux-unit model is
    # mean / a and cov / a^2, and infill with replace equals conditioning it.
    ivar = np.vstack([train.ivar, held_out.ivar])
    if not np.all(ivar == ivar[0]):
        raise SystemExit("the spectra do not all have the same ivar")
    model, _ = gspice.fit(train.flux, train.ivar)
    a = np.sqrt(ivar[0].mean())
    mean, cov = model.mean[0] / a, model.cov[0] / a**2
    stored = {
        "mean": np.round(mean, DECIMALS),
        "cov": np.round(cov, COV_DECIMALS),
        "flux": np.round(held_out.flux, DECIMALS),
        "truth": np.round(held_out.truth, DECIMALS),
    }
    masks = [
        block(start, width)
        for width in (1, 5, 20, 50, WIDEST)
        for start in range(0, NPIX - width + 1, 7)
    ]
    rounding = max(
        np.abs(
            np.subtract(
                conditional(mean, cov, held_out.flux[s], m),
                conditional(stored["mean"], stored["cov"], stored["flux"][s], m),
            )
        ).max()
        for s in range(NEXAMPLE)
        for m in masks
    )
    print(f"largest change of the prediction or sd by rounding: {rounding:.2e}")
    if rounding >= 1e-3:
        raise SystemExit("the rounding changes the prediction by 1e-3 or more")
    data = {
        "npix": NPIX,
        "ntrain": NTRAIN,
        "widest": WIDEST,
        "band": band(stored),
        "mean": stored["mean"].tolist(),
        "cov": stored["cov"].tolist(),
        "examples": [
            {"flux": f.tolist(), "truth": t.tolist()}
            for f, t in zip(stored["flux"], stored["truth"], strict=True)
        ],
    }
    write("explainer.json", data)
    write("explainer-check.json", check(stored, held_out.ivar, a))
    return stored, train.flux, held_out.ivar, a


# The lowest and highest flux of the ±2σ bands, over every example and every block that
# the explainer allows, so that its axes hold them.
def band(stored: dict[str, Floats]) -> list[float]:
    low, high = np.inf, -np.inf
    for width in range(1, WIDEST + 1):
        for start in range(NPIX - width + 1):
            for flux in stored["flux"]:
                mu, sd = conditional(
                    stored["mean"], stored["cov"], flux, block(start, width)
                )
                low = min(low, (mu - 2 * sd).min())
                high = max(high, (mu + 2 * sd).max())
    return [round(float(low), DECIMALS), round(float(high), DECIMALS)]


def check(
    stored: dict[str, Floats], ivar: Floats, a: float
) -> dict[str, list[dict[str, Any]]]:
    model = scaled_model(stored, a)
    cases = []
    for spectrum, start, width in ((0, 50, 20), (3, 0, 1), (5, 10, WIDEST)):
        masked = block(start, width)
        mu, sd = conditional(
            stored["mean"], stored["cov"], stored["flux"][spectrum], masked
        )
        infilled = gspice.infill(
            model, stored["flux"][[spectrum]], ivar[[spectrum]], replace=masked
        )
        cases.append(
            {
                "spectrum": spectrum,
                "start": start,
                "width": width,
                "numpy": {"mu": mu.tolist(), "sd": sd.tolist()},
                "infill": {
                    "mu": infilled.flux[0, masked].tolist(),
                    "sd": np.sqrt(infilled.var[0, masked]).tolist(),
                },
            }
        )
    return {"cases": cases}


def scaled_model(stored: dict[str, Floats], a: float) -> gspice.Model:
    # The model of the rounded data, in scaled units, for gspice.infill and pixelwise.
    return gspice.Model([stored["mean"] * a], [stored["cov"] * a**2], [NTRAIN])


def correlation(cov: Floats) -> Floats:
    sd = np.sqrt(np.diag(cov))
    return cov / np.outer(sd, sd)


def covariance_explorer(stored: dict[str, Floats], train: Floats) -> None:
    spectra = np.round((train[:NSCATTER] - stored["mean"]) / FLUX_UNIT).astype(int)
    shown = stored["mean"] + spectra * FLUX_UNIT
    print(
        "largest change of a training spectrum by rounding: "
        f"{np.abs(shown - train[:NSCATTER]).max():.2e}"
    )
    R = correlation(stored["cov"])
    # The pixel of largest variance and the pixel most anti-correlated with it.
    x = int(np.argmax(np.diag(stored["cov"])))
    data = {
        "unit": FLUX_UNIT,
        "pair": [x, int(np.argmin(R[x]))],
        "spectra": spectra.tolist(),
    }
    write("covariance.json", data)
    steps = np.digitize(R, HEAT_EDGES)
    write("covariance-check.json", {"steps": ["".join(map(str, r)) for r in steps]})


def two_pixel(stored: dict[str, Floats], ivar: Floats, a: float) -> None:
    cov = stored["cov"]
    R = correlation(cov)
    i = int(np.argmax(np.diag(cov)))
    others = np.arange(NPIX) != i
    pairs = [
        [i, int(np.flatnonzero(others)[np.argmin(np.abs(R[i, others] - r))])]
        for r in PAIR_CORRELATIONS
    ]
    write("two-pixel.json", {"pairs": pairs})
    # gspice.infill of pixel i from pixel j alone, the other pixels ignored, with
    # pixel j at k standard deviations from its mean.
    model = scaled_model(stored, a)
    cases = []
    for i, j in pairs:
        replace = np.arange(NPIX) == i
        ignore = ~replace & (np.arange(NPIX) != j)
        for k in (-2.0, 0.0, 1.5):
            flux = stored["mean"].copy()
            flux[j] += k * np.sqrt(cov[j, j])
            result = gspice.infill(
                model, flux[None], ivar[:1], replace=replace[None], ignore=ignore[None]
            )
            cases.append(
                {
                    "pair": [i, j],
                    "value": flux[j],
                    "mu": result.flux[0, i],
                    "sd": np.sqrt(result.var[0, i]),
                }
            )
    write("two-pixel-check.json", {"cases": cases})


def guard_window(stored: dict[str, Floats], ivar: Floats, a: float) -> None:
    model = scaled_model(stored, a)
    widths = np.arange(1, DEFECT_WIDEST + 1)
    defects = np.repeat(stored["flux"][[GUARD_SPECTRUM]], widths.size, axis=0)
    for row, width in enumerate(widths):
        defects[row, DEFECT_START : DEFECT_START + width] += DEFECT_HEIGHT
    rows = np.repeat(ivar[[GUARD_SPECTRUM]], widths.size, axis=0)
    # The range of the flux, the truth and every prediction, with the ±2σ of the pixel
    # next to the defect, over all nguard and widths, for the axes of the figure.
    values = [defects.ravel(), stored["truth"][GUARD_SPECTRUM]]
    next_to = (np.arange(widths.size), DEFECT_START + widths)
    for nguard in range(GUARD_WIDEST + 1):
        result = gspice.pixelwise(model, defects, rows, nguard=nguard)
        mu, sd = result.flux[next_to], np.sqrt(result.var[next_to])
        values += [result.flux.ravel(), mu - 2 * sd, mu + 2 * sd]
    flux = np.concatenate(values)
    data = {
        "spectrum": GUARD_SPECTRUM,
        "start": DEFECT_START,
        "height": DEFECT_HEIGHT,
        "widest_defect": DEFECT_WIDEST,
        "widest_nguard": GUARD_WIDEST,
        "threshold": THRESHOLD,
        "range": [
            round(float(flux.min()), DECIMALS),
            round(float(flux.max()), DECIMALS),
        ],
    }
    write("guard.json", data)
    cases = []
    for nguard, width in GUARD_CHECKS:
        result = gspice.pixelwise(
            model, defects[[width - 1]], ivar[[GUARD_SPECTRUM]], nguard=nguard
        )
        cases.append(
            {
                "nguard": nguard,
                "width": width,
                "mu": result.flux[0].tolist(),
                "sd": np.sqrt(result.var[0]).tolist(),
                "z": result.z[0].tolist(),
            }
        )
    write("guard-check.json", {"cases": cases})


Masks = tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]


def block_pixels(model: gspice.Model, flagged: NDArray[np.bool_]) -> NDArray[np.bool_]:
    # The pixels of the flagged blocks (nspec, nblock) of each spectrum, M_blk of repair.
    assert model.blocks is not None
    starts, width = model.blocks.starts[:, None], model.blocks.width
    pixels = np.arange(model.cov[0].shape[0])
    cover = (pixels >= starts) & (pixels < starts + width)
    in_blocks: NDArray[np.bool_] = np.any(flagged[:, :, None] & cover, axis=1)
    return in_blocks


def replay(model: gspice.Model, test: Any, result: gspice.Repaired) -> list[Masks]:
    # The replaced and ignored pixels and the flagged blocks after each of steps 1 to 4
    # of gspice.repair with its defaults, as its docstring states them, for spectra
    # without bad pixels: step 1 from the z of pixelwise, with the dilation of repair;
    # step 2 adds the pixels of the blocks of repair, M_blk, to those to replace; step 3
    # from repair without step 4; step 4 from repair.
    if np.any(test.ivar == 0):
        raise SystemExit("the test spectra have bad pixels, which step 1 omits")
    z = np.abs(
        gspice.pixelwise(model, test.flux, test.ivar, nguard=REPAIR_OPTIONS["nguard"]).z
    )
    size = 2 * REPAIR_OPTIONS["dilate"] + 1

    def widen(flags: NDArray[np.bool_]) -> NDArray[np.bool_]:
        # dil(F, d) of repair: no flag beyond the ends of a spectrum.
        dilated: NDArray[np.bool_] = maximum_filter1d(
            flags, size, axis=1, mode="constant", cval=0
        )
        return dilated

    M1 = widen(z > REPAIR_OPTIONS["nsigma_replace"])
    I1 = widen(z > REPAIR_OPTIONS["nsigma_ignore"]) & ~M1
    third = gspice.repair(model, test.flux, test.ivar, runlength=None)
    assert result.blocks is not None and third.blocks is not None
    none = np.zeros_like(result.blocks)
    Mb = block_pixels(model, result.blocks)
    return [
        (M1, I1, none),
        (M1 | Mb, I1 & ~Mb, result.blocks),
        (third.replaced, third.ignored, third.blocks),
        (result.replaced, result.ignored, result.blocks),
    ]


def runs(mask: NDArray[np.bool_]) -> list[list[int]]:
    # The runs of True of a mask, as [first, last + 1].
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(int), [0]))))
    return edges.reshape(-1, 2).tolist()


def repair_guide() -> None:
    page = run("guides/repair.mdx")
    model, test, result = page["model"], page["test"], page["result"]
    steps = replay(model, test, result)
    # Step 4 only adds ignored pixels, and the pixels of a flagged block are replaced.
    (M3, I3, blocks3), (M4, I4, blocks4) = steps[2:]
    if not (
        np.array_equal(blocks3, blocks4)
        and np.array_equal(M3, M4)
        and np.all(I4 >= I3)
        and np.all(M4 >= block_pixels(model, blocks4))
    ):
        raise SystemExit("the masks of steps 3 and 4 do not relate as repair states")
    column, segment = REPAIR_SHOWN
    # Whether steps 2, 3 and 4 change the masks of the bad column.
    changed = [
        any(
            not np.array_equal(a[column], b[column]) for a, b in zip(*pair, strict=True)
        )
        for pair in itertools.pairwise(steps)
    ]
    length = test.artifacts.sum(axis=1)
    if not (
        5 <= length[column] <= 20
        and changed == [False, True, True]
        and length[segment] > 20
        and steps[1][2][segment].any()
    ):
        raise SystemExit("REPAIR_SHOWN no longer names the spectra it describes")
    # Where step 4 keeps the ignored pixels of step 3 although minfrac=0 changes them,
    # it would leave fewer than floor(minfrac npix) pixels to predict from.
    shown = list(REPAIR_SHOWN)
    unbounded = gspice.repair(model, test.flux[shown], test.ivar[shown], minfrac=0.0)
    examples = []
    for row, s in enumerate(REPAIR_SHOWN):
        replaced = result.replaced[s]
        examples.append(
            {
                "spectrum": s,
                "flux": np.round(test.flux[s], PLOT_DECIMALS).tolist(),
                "truth": np.round(test.truth[s], PLOT_DECIMALS).tolist(),
                "steps": [
                    {
                        "replaced": runs(Ms[s]),
                        "ignored": runs(Is[s]),
                        "blocks": np.flatnonzero(Fs[s]).tolist(),
                    }
                    for Ms, Is, Fs in steps
                ],
                "mu": np.round(result.flux[s, replaced], PLOT_DECIMALS).tolist(),
                "sd": np.round(
                    np.sqrt(result.var[s, replaced]), PLOT_DECIMALS
                ).tolist(),
                "too_few": bool(
                    np.array_equal(I4[s], I3[s])
                    and not np.array_equal(unbounded.ignored[row], I3[s])
                ),
            }
        )
    names = (
        "nsigma_replace",
        "nsigma_ignore",
        "dilate",
        "runlength",
        "halfwidth",
        "minfrac",
    )
    data = {
        "npix": test.flux.shape[1],
        "options": {name: REPAIR_OPTIONS[name] for name in names},
        "starts": model.blocks.starts.tolist(),
        "width": model.blocks.width,
        "examples": examples,
    }
    write("repair-steps.json", data)


def run(page: str) -> dict[str, Any]:
    # The Python blocks of the page, run as one script, as tests/test_docs.py does.
    code = "".join(
        re.findall(
            r"```python\n(.*?)```",
            (PAGES / page).read_text(encoding="utf-8"),
            re.DOTALL,
        )
    )
    # ruff forbids exec (S102), so the code runs from a file.
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "page.py"
        script.write_text(code, encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            return runpy.run_path(str(script))


def quickstart() -> None:
    page = run("start/quickstart.mdx")
    test, result, mask = page["test"], page["result"], page["mask"][SHOWN]
    data = {
        "spectrum": SHOWN,
        "snr": round(float(test.snr[SHOWN])),
        "start": int(np.argmax(mask)),
        "width": int(mask.sum()),
        "flux": np.round(test.flux[SHOWN], DECIMALS).tolist(),
        "truth": np.round(test.truth[SHOWN], DECIMALS).tolist(),
        "mu": np.round(result.flux[SHOWN, mask], DECIMALS).tolist(),
        "sd": np.round(np.sqrt(result.var[SHOWN, mask]), DECIMALS).tolist(),
    }
    write("quickstart.json", data)


def infill_guide() -> None:
    # `result` is the last infill of the page, of the gap; the draws are those of its
    # multiple imputation, of spectrum 0.
    page = run("guides/infill.mdx")
    test, gap = page["test"], page["gap"]
    data = {
        "snr": round(float(test.snr[0])),
        "start": int(np.argmax(gap)),
        "width": int(gap.sum()),
        "flux": np.round(test.flux[0], DECIMALS).tolist(),
        "truth": np.round(test.truth[0], DECIMALS).tolist(),
        "mu": np.round(page["result"].flux[0, gap], DECIMALS).tolist(),
        "draws": [np.round(d[gap], DECIMALS).tolist() for d in page["draws"][:DRAWS]],
    }
    write("guide-infill.json", data)


def outliers_guide() -> None:
    page = run("guides/outliers.mdx")
    z = page["z"][page["clean"]].ravel()
    counts, _ = np.histogram(np.clip(z, Z_EDGES[0], Z_EDGES[-1]), Z_EDGES)
    # The expected count of each bin for z ~ N(0, 1), with the tails in the end bins.
    cdf = stats.norm.cdf(Z_EDGES)
    cdf[0], cdf[-1] = 0.0, 1.0
    data = {
        "edges": Z_EDGES.tolist(),
        "counts": counts.tolist(),
        "gaussian": np.round(z.size * np.diff(cdf), 3).tolist(),
        "threshold": THRESHOLD,
    }
    write("guide-outliers.json", data)


def write(name: str, data: dict[str, Any]) -> None:
    path = OUT / name
    path.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    print(f"{path.relative_to(DOCS)}: {path.stat().st_size / 1000:.0f} kB")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stored, train, ivar, a = explainer(np.random.default_rng(11))
    covariance_explorer(stored, train)
    two_pixel(stored, ivar, a)
    guard_window(stored, ivar, a)
    quickstart()
    infill_guide()
    outliers_guide()
    repair_guide()


if __name__ == "__main__":
    main()
