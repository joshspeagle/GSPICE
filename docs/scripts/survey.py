"""Write the data of the figures of the worked example to docs/src/data/.

    python docs/scripts/survey.py --data DIR/spectra.npz

It runs the code of example/survey.mdx on the survey of spectra.py, and fails unless the
printed output equals the output blocks of the page. It reads the variables of the
page by their names, such as strong, kinds, result and checks, and writes:

- survey-clipped.json: at each pixel, the number of training spectra with an artifact
  and the number that fit clipped;
- survey-repair.json: a window of one test spectrum with a cosmic ray, its spectrum
  without artifacts, and the prediction and standard deviation of its replaced pixels;
- survey-calibration.json: the mean of z^2 / (n f_s) and its standard error at one size
  n, in each S/N quartile of the test spectra, for the two models of the page.
"""

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
from concepts import rounded
from explainers import PAGES, run, runs, write
from numpy.typing import NDArray

PAGE = "example/survey.mdx"
# The pixels of the repair figure, the pixels it shows before the cosmic ray, and the
# size n of the calibration figure.
WINDOW, BEFORE, SIZE = 100, 40, 64


def checked(data: Path) -> dict[str, Any]:
    page, out = run(PAGE, data)
    text = (PAGES / PAGE).read_text(encoding="utf-8")
    if out != "".join(re.findall(r"```text output\n(.*?)```", text, re.DOTALL)):
        raise SystemExit(f"the output blocks of {PAGE} differ:\n{out}")
    return page


def single_run(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    return np.array([len(runs(row)) == 1 for row in mask])


def clipped(page: dict[str, Any]) -> None:
    train = page["train"]
    wave = page["data"]["wave"][page["pixels"]]
    data = {
        "ntrain": int(page["model"].n_train[0]),
        "wave": [round(float(wave[0]), 2), round(float(wave[-1]), 2)],
        "artifacts": page["artifacts"][train].sum(axis=0).tolist(),
        "clipped": page["mask"].sum(axis=0).tolist(),
        "column_peak": int(page["kinds"]["column"][train].sum(axis=0).max()),
    }
    write("survey-clipped.json", data)


def repair(page: dict[str, Any]) -> None:
    # The test spectrum with one cosmic ray and no other artifact in the pixels. Repair
    # replaced its strong pixels in one run. Its S/N is nearest the median of the test
    # spectra.
    test, kinds, result = page["test"], page["kinds"], page["result"]
    cosmic = kinds["cosmic"][test]
    other = (kinds["column"] | kinds["shift"])[test]
    strong = page["strong"][test]
    snr = page["data"]["snr"][test]
    good = (
        single_run(cosmic)
        & ~other.any(axis=1)
        & single_run(result.replaced)
        & strong.any(axis=1)
        & (strong <= result.replaced).all(axis=1)
    )
    rows = np.flatnonzero(good)
    s = int(rows[np.argmin(np.abs(snr[rows] - np.median(snr)))])
    npix = cosmic.shape[1]
    lo = int(np.clip(np.argmax(cosmic[s]) - BEFORE, 0, npix - WINDOW))
    shown = slice(lo, lo + WINDOW)
    replaced = result.replaced[s, shown]
    wave = page["data"]["wave"][page["pixels"]][shown]
    data = {
        "spectrum": s,
        "wave": [round(float(wave[0]), 2), round(float(wave[-1]), 2)],
        "snr": round(float(snr[s])),
        "flux": rounded(page["flux"][test][s, shown]),
        "clean": rounded(page["clean"][test][s, shown]),
        "cosmic": np.flatnonzero(cosmic[s, shown]).tolist(),
        "replaced": np.flatnonzero(replaced).tolist(),
        "mu": rounded(result.flux[s, shown][replaced]),
        "sd": rounded(np.sqrt(result.var[s, shown][replaced])),
    }
    write("survey-repair.json", data)


def calibration(page: dict[str, Any]) -> None:
    quartile, checks = page["quartile"], page["checks"]
    column = list(next(iter(checks.values())).n[0]).index(SIZE)
    models = {}
    for name, check in checks.items():
        # ratio gives (mean, sem) per quartile; keep the column of SIZE.
        ratios = np.array([check.ratio(where=quartile == q) for q in range(4)])
        mean, sem = ratios[:, :, column].T
        models[name] = {"mean": rounded(mean), "sem": rounded(sem)}
    data = {
        "n": SIZE,
        "edges": [round(float(e)) for e in page["edges"]],
        "count": int(np.count_nonzero(quartile == 0)),
        "models": models,
    }
    write("survey-calibration.json", data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    page = checked(args.data)
    clipped(page)
    repair(page)
    calibration(page)


if __name__ == "__main__":
    main()
