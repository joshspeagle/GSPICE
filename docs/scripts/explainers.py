"""Write the data of the mask explainer and of the figures of the quickstart and guides.

    python docs/scripts/explainers.py

It fits a model with gspice.fit to spectra of gspice.simulate(gaussian=True), which
all have the same ivar, and runs the code of the quickstart page and of two guides. It
writes to docs/src/data/:

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
  outliers, and the count that a Gaussian distribution gives in each bin.

It prints the sizes of the files and the largest change of the explainer's prediction
and standard deviation that the rounding makes, and fails if that change reaches 1e-3.
"""

import contextlib
import io
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


def explainer(rng: np.random.Generator) -> None:
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
    # The model of the rounded data, in scaled units, for gspice.infill.
    model = gspice.Model([stored["mean"] * a], [stored["cov"] * a**2], [NTRAIN])
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
    explainer(np.random.default_rng(11))
    quickstart()
    infill_guide()
    outliers_guide()


if __name__ == "__main__":
    main()
