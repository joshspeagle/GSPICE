"""Make the figure of the fitting tutorial from the simulated survey of spectra.py.

    python docs/scripts/figures.py --data DIR/spectra.npz

It runs the computations of tutorials/fitting on the survey, prints the numbers that
the tutorial quotes as JSON, and writes its figure to docs/src/assets/figures/tutorials/
as NAME-light.svg and NAME-dark.svg. It needs matplotlib, from the dependency group
figures.
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import gspice

OUT = Path(__file__).parents[1] / "src" / "assets" / "figures" / "tutorials"
ACCENT = "#F0592A"
THEMES = {
    "light": {
        "ink": "#222222",
        "muted": "#6b6b6b",
        "grid": "#e3e3e3",
        "blue": "#2a78d6",
        "aqua": "#1baf7a",
    },
    "dark": {
        "ink": "#d6d6d6",
        "muted": "#9a9a9a",
        "grid": "#3a3a3a",
        "blue": "#3987e5",
        "aqua": "#199e70",
    },
}
SIGMA = 5  # the amplitude above which an artifact pixel counts as strong
# The pixels outside a guard window of the default nguard = 20 on each side.
GUARD = 2 * 20 + 1


def compute(path):
    data = np.load(path)
    pixels = (data["wave"] > 3800) & (data["wave"] < 5370)
    wave = data["wave"][pixels]
    s = {k: data[k][:, pixels] for k in ("flux", "ivar", "clean", "clean_ivar")}
    kinds = {k: data[k][:, pixels] for k in ("cosmic", "column", "shift")}
    artifacts = kinds["cosmic"] | kinds["column"] | kinds["shift"]
    flux, ivar, clean, clean_ivar = s["flux"], s["ivar"], s["clean"], s["clean_ivar"]
    n = len(flux)
    train = slice(0, int(0.6 * n))
    amplitude = np.abs(flux - clean) * np.sqrt(clean_ivar)
    strong = artifacts & (amplitude > SIGMA)
    every_kind = data["cosmic"] | data["column"] | data["shift"]
    num = {
        "npix": int(wave.size),
        "nspec": n,
        "spectra_with_artifact": every_kind.any(axis=1).mean(),
    }
    r = {"wave": wave}

    # Fitting.
    model, mask = gspice.fit(flux[train], ivar[train])
    a = artifacts[train]
    num["fit"] = {
        "n_train": model.n_train,
        "clipped": mask.mean(),
        "strong_clipped": mask[strong[train]].mean(),
        "other_clipped": mask[~a].mean(),
        "column_peak": kinds["column"][train].mean(axis=0).max(),
        "fs_all_pixels": gspice.finite_sample_factor(
            model.n_train[0], pixels.size - GUARD
        ),
        "fs_subset": gspice.finite_sample_factor(model.n_train[0], wave.size - GUARD),
    }
    r["clipped"] = mask.mean(axis=0)
    r["artifact_rate"] = a.mean(axis=0)

    return num, r


def axes(fig_height, theme, rows=1, ratios=None):
    fig, ax = plt.subplots(
        rows,
        1,
        figsize=(7.2, fig_height),
        sharex=True,
        squeeze=False,
        gridspec_kw={"height_ratios": ratios, "hspace": 0.08},
        layout="constrained",
    )
    for a in ax[:, 0]:
        a.grid(True, color=theme["grid"], linewidth=0.6)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    return fig, ax[:, 0]


def legend(ax, ncols):
    # Above the axes, so that it hides no data.
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1),
        ncols=ncols,
        frameon=False,
        borderaxespad=0.2,
        handlelength=1.5,
    )


def fitting(r, t):
    fig, (ax,) = axes(2.6, t)
    for rate, color, label in (
        (r["artifact_rate"], t["muted"], "with an injected artifact"),
        (r["clipped"], ACCENT, "clipped by fit"),
    ):
        ax.plot(
            r["wave"],
            100 * np.where(rate > 0, rate, np.nan),
            color=color,
            lw=1,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_xlabel("rest-frame wavelength (Å)")
    ax.set_ylabel("training spectra (%)")
    legend(ax, 2)
    return fig


FIGURES = {"fitting-clipped": fitting}


def style(t):
    return {
        "svg.fonttype": "none",
        "svg.hashsalt": "gspice",
        "font.size": 9,
        "text.color": t["ink"],
        "axes.labelcolor": t["ink"],
        "axes.edgecolor": t["muted"],
        "xtick.color": t["muted"],
        "ytick.color": t["muted"],
        "xtick.labelcolor": t["ink"],
        "ytick.labelcolor": t["ink"],
        "axes.linewidth": 0.6,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    num, r = compute(args.data)
    print(json.dumps(num, indent=1, default=lambda v: v.tolist()))
    args.output.mkdir(parents=True, exist_ok=True)
    for theme, t in THEMES.items():
        with mpl.rc_context(style(t)):
            for name, figure in FIGURES.items():
                fig = figure(r, t)
                fig.savefig(
                    args.output / f"{name}-{theme}.svg",
                    transparent=True,
                    metadata={"Date": None},
                )
                plt.close(fig)


if __name__ == "__main__":
    main()
