"""Make the figures of the tutorials from the simulated survey of spectra.py.

    python docs/scripts/figures.py --data DIR/spectra.npz

It runs the computations of the tutorials on the survey, prints the numbers that the
survey tutorials quote as JSON, and writes each figure to docs/_static/figures/tutorials/ as
NAME-light.svg and NAME-dark.svg. It needs matplotlib, from the dependency group
figures. It took 2 minutes on 4 cores.
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import gspice

OUT = Path(__file__).parents[1] / "_static" / "figures" / "tutorials"
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
    full = {k: data[k] for k in ("flux", "ivar", "clean", "clean_ivar")}
    s = {k: v[:, pixels] for k, v in full.items()}
    kinds = {k: data[k][:, pixels] for k in ("cosmic", "column", "shift")}
    artifacts = kinds["cosmic"] | kinds["column"] | kinds["shift"]
    shift = kinds["shift"]
    flux, ivar, clean, clean_ivar = s["flux"], s["ivar"], s["clean"], s["clean_ivar"]
    n = len(flux)
    train = slice(0, int(0.6 * n))
    blocks = slice(int(0.6 * n), int(0.8 * n))
    test = slice(int(0.8 * n), n)
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

    # Pixelwise prediction.
    p = gspice.pixelwise(model, flux[test], ivar[test])
    zc = gspice.pixelwise(model, clean[test], clean_ivar[test]).z
    zi = gspice.pixelwise(model, clean[test], clean_ivar[test], inflate=True).z
    num["pixelwise"] = {
        "z_std": np.std(zc),
        "z_std_inflate": np.std(zi),
        "far": np.mean(np.abs(zc) > SIGMA),
        "far_inflate": np.mean(np.abs(zi) > SIGMA),
        "strong_far": np.mean(np.abs(p.z[strong[test]]) > SIGMA),
        "missed_in_shift": shift[test][strong[test] & ~(np.abs(p.z) > SIGMA)].mean(),
    }
    cosmic = kinds["cosmic"][test]
    pick = np.flatnonzero((amplitude[test] * cosmic).max(axis=1) > 20)[0]
    r["pixelwise"] = (flux[test][pick], p.flux[pick], p.var[pick], p.z[pick])

    # Infill.
    known = gspice.infill(
        model, flux[test], ivar[test], mask=artifacts[test], inflate=True
    )
    at = artifacts[test]
    pull = (known.flux - clean[test])[at] / np.sqrt(known.var[at])
    gap = (wave > 4840) & (wave < 4890)
    rng = np.random.default_rng(3)
    g = gspice.infill(
        model, clean[test], clean_ivar[test], replace=gap, covariance=True, rng=rng
    )
    num["infill"] = {
        "pull_std": np.std(pull),
        "gap_pixels": int(gap.sum()),
        "gap_ratio": np.mean(g.chi2 / (g.dof * g.fs)),
    }
    r["infill"] = (clean[test][0], g.flux[0], g.var[0], g.draws[0], gap)

    # Repair.
    model = gspice.fit_blocks(model, clean[blocks], clean_ivar[blocks])
    rep = gspice.repair(model, flux[test], ivar[test])
    clean_rep = gspice.repair(model, clean[test], clean_ivar[test])
    st, at = strong[test], artifacts[test]
    has_shift = shift[test].any(axis=1)
    false = rep.replaced & ~at
    without = np.where(at, clean[test], flux[test])
    z_without = gspice.pixelwise(model, without, ivar[test]).z
    num["repair"] = {
        "nblock": int(model.blocks.starts.size),
        "false": (rep.replaced & ~at).sum() / (~at).sum(),
        "strong_replaced": rep.replaced[st].mean(),
        "strong_handled": (rep.replaced | rep.ignored)[st].mean(),
        "clean_spectra_touched": clean_rep.replaced.any(axis=1).mean(),
        "false_spectra": int((rep.replaced & ~at).any(axis=1).sum()),
        "false_spectra_with_artifact": int((false.any(axis=1) & at.any(axis=1)).sum()),
        "false_spectra_with_cosmic": int(
            (false.any(axis=1) & kinds["cosmic"][test].any(axis=1)).sum()
        ),
        "test_spectra_with_cosmic": kinds["cosmic"][test].any(axis=1).mean(),
        "false_z_above_10": np.mean(np.abs(p.z[false]) > 10),
        "false_z_above_10_without_artifacts": np.mean(np.abs(z_without[false]) > 10),
        "strong_replaced_by_type": {
            name: rep.replaced[st & kind[test]].mean() for name, kind in kinds.items()
        },
        "shift_spectra": int(has_shift.sum()),
        "shift_flagged": rep.blocks[has_shift].any(axis=1).mean(),
    }
    flagged = np.flatnonzero(has_shift & rep.blocks.any(axis=1))
    cover = (rep.replaced & shift[test])[flagged].sum(1) / shift[test][flagged].sum(1)
    pick = flagged[np.argmax(cover)]
    inblock = np.zeros(wave.size, dtype=bool)
    for start in model.blocks.starts[rep.blocks[pick]]:
        inblock[start : start + model.blocks.width] = True
    r["repair"] = (
        flux[test][pick],
        rep.flux[pick],
        clean[test][pick],
        ivar[test][pick],
        shift[test][pick],
        inblock,
    )
    edges = np.geomspace(1, 1000, 16)
    which = np.digitize(amplitude[test][at], edges)
    replaced, handled = rep.replaced[at], (rep.replaced | rep.ignored)[at]
    keep = [k for k in range(1, len(edges)) if (which == k).sum() >= 20]
    r["recall"] = (
        np.sqrt(edges[:-1] * edges[1:])[np.array(keep) - 1],
        np.array([replaced[which == k].mean() for k in keep]),
        np.array([handled[which == k].mean() for k in keep]),
    )

    # Calibration and noise audit.
    cal = gspice.calibration(
        model, clean[test], clean_ivar[test], rng=np.random.default_rng(1)
    )
    every = (cal.n > 0).all(axis=1)
    snr = gspice.snr(clean[test], clean_ivar[test])
    third = np.searchsorted(np.quantile(snr, [1 / 3, 2 / 3]), snr)
    binned, _ = gspice.fit(flux[train], ivar[train], snr_bins=3)
    calb = gspice.calibration(
        binned, clean[test], clean_ivar[test], rng=np.random.default_rng(1)
    )
    num["calibration"] = {
        "sizes": cal.n[every][0].tolist(),
        "ratio": cal.ratio(where=every)[0].tolist(),
        "sem": cal.ratio(where=every)[1].tolist(),
        "tail": cal.tail(0.99, where=every).tolist(),
        "snr_edges": np.quantile(snr, [1 / 3, 2 / 3]).tolist(),
        "third": [cal.ratio(where=every & (third == t))[0].tolist() for t in range(3)],
        "binned_edges": binned.edges,
        "binned_n_train": binned.n_train,
        "binned_third": [
            calb.ratio(where=every & (third == t))[0].tolist() for t in range(3)
        ],
        "fs_binned": gspice.finite_sample_factor(
            min(binned.n_train), wave.size - GUARD
        ),
    }
    r["calibration"] = (
        cal.n[every][0],
        [cal.ratio(where=every & (third == t)) for t in range(3)],
        cal.ratio(where=every),
        num["calibration"]["snr_edges"],
    )
    rho = gspice.noise_audit(model, clean[test], clean_ivar[test])
    rho_train = gspice.noise_audit(model, clean[train], clean_ivar[train])
    num["noise_audit"] = {
        "median": np.median(rho),
        "range": [rho.min(), rho.max()],
        "median_train": np.median(rho_train),
    }
    r["rho"] = (rho, rho_train)

    # Low-rank and noise-aware models.
    full_train = [full[k][train] for k in ("flux", "ivar")]
    full_test = [full[k][test] for k in ("clean", "clean_ivar")]
    zs = {}
    for name, rank in (("dense", None), ("rank", 40)):
        m, _ = gspice.fit(*full_train, rank=rank)
        zs[name] = gspice.pixelwise(m, *full_test).z.ravel()
    few = slice(test.start, test.start + n // 50)
    noisy, _ = gspice.fit(flux[train], ivar[train], noise=True)
    caln = gspice.calibration(
        noisy, clean[few], clean_ivar[few], rng=np.random.default_rng(1)
    )
    num["models"] = {
        "z_std_dense": np.std(zs["dense"]),
        "z_std_rank": np.std(zs["rank"]),
        "far_dense": np.mean(np.abs(zs["dense"]) > SIGMA),
        "far_rank": np.mean(np.abs(zs["rank"]) > SIGMA),
        "noise_spectra": n // 50,
        "noise_sizes": caln.n[(caln.n > 0).all(axis=1)][0].tolist(),
        "noise_ratio": caln.ratio(where=(caln.n > 0).all(axis=1))[0].tolist(),
        "noise_sem": caln.ratio(where=(caln.n > 0).all(axis=1))[1].tolist(),
    }
    r["models"] = zs
    r["npix"] = pixels.size
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


def pixelwise(r, t):
    flux, pred, var, z = r["pixelwise"]
    centre = int(np.nanargmax(np.abs(z)))
    w = slice(max(centre - 60, 0), centre + 60)
    wave = r["wave"][w]
    fig, (ax, az) = axes(3.6, t, rows=2, ratios=[2, 1])
    sd = np.sqrt(var[w])
    ax.fill_between(
        wave, pred[w] - sd, pred[w] + sd, color=ACCENT, alpha=0.3, lw=0, label="±1σ"
    )
    ax.plot(wave, flux[w], color=t["ink"], lw=1, label="flux")
    ax.plot(wave, pred[w], color=ACCENT, lw=1.5, label="prediction")
    ax.set_ylabel("normalized flux")
    legend(ax, 3)
    az.axhspan(-SIGMA, SIGMA, color=t["muted"], alpha=0.15, lw=0)
    az.plot(wave, z[w], color=ACCENT, lw=1)
    az.set_yscale("symlog", linthresh=10)
    az.set_yticks([-10, 0, 10, 100], ["−10", "0", "10", "100"])
    az.set_ylabel("z")
    az.set_xlabel("rest-frame wavelength (Å)")
    return fig


def infill(r, t):
    clean, mean, var, draw, gap = r["infill"]
    idx = np.flatnonzero(gap)
    w = slice(idx[0] - 50, idx[-1] + 51)
    wave = r["wave"]
    fig, (ax,) = axes(2.8, t)
    sd = np.sqrt(var[gap])
    ax.axvspan(wave[idx[0]], wave[idx[-1]], color=t["muted"], alpha=0.12, lw=0)
    ax.fill_between(
        wave[gap],
        mean[gap] - sd,
        mean[gap] + sd,
        color=ACCENT,
        alpha=0.3,
        lw=0,
        label="±1σ",
    )
    ax.plot(wave[w], clean[w], color=t["ink"], lw=1, label="flux")
    ax.plot(wave[gap], draw[gap], color=t["blue"], lw=1, label="a draw")
    ax.plot(wave[gap], mean[gap], color=ACCENT, lw=1.5, label="conditional mean")
    ax.set_xlabel("rest-frame wavelength (Å)")
    ax.set_ylabel("normalized flux")
    legend(ax, 4)
    return fig


def repair(r, t):
    flux, fixed, clean, ivar, shift, inblock = r["repair"]
    lo, hi = np.flatnonzero(shift)[[0, -1]]
    w = slice(max(lo - 80, 0), hi + 81)
    wave = r["wave"][w]
    fig, (ax, ar) = axes(3.8, t, rows=2, ratios=[3, 2])
    for a in (ax, ar):
        a.fill_between(
            wave,
            0,
            1,
            where=inblock[w],
            transform=a.get_xaxis_transform(),
            color=t["muted"],
            alpha=0.2,
            lw=0,
            label="flagged blocks",
        )
    ax.plot(
        r["wave"][[lo, hi]],
        [1.1, 1.1],
        color=t["blue"],
        lw=3,
        solid_capstyle="butt",
        label="shifted block",
    )
    sd = np.sqrt(1 / ivar[w])
    for y, color, label in ((flux, t["ink"], "input"), (fixed, ACCENT, "repaired")):
        ax.plot(wave, y[w], color=color, lw=0.9, label=label)
        ar.plot(wave, (y[w] - clean[w]) / sd, color=color, lw=0.9)
    ax.set_ylabel("normalized flux")
    ar.set_ylabel("(f − clean) / σ")
    ar.set_xlabel("rest-frame wavelength (Å)")
    legend(ax, 4)
    return fig


def recall(r, t):
    centre, replaced, handled = r["recall"]
    fig, (ax,) = axes(2.6, t)
    for y, color, label in (
        (handled, t["muted"], "replaced or ignored"),
        (replaced, ACCENT, "replaced"),
    ):
        ax.plot(centre, 100 * y, color=color, lw=1.5, marker="o", ms=4, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("artifact amplitude, |f − clean| / σ")
    ax.set_ylabel("artifact pixels (%)")
    ax.set_ylim(0, 102)
    legend(ax, 2)
    return fig


def calibration(r, t):
    sizes, thirds, overall, edges = r["calibration"]
    labels = (
        f"S/N < {edges[0]:.0f}",
        f"S/N {edges[0]:.0f} to {edges[1]:.0f}",
        f"S/N > {edges[1]:.0f}",
    )
    fig, (ax,) = axes(2.8, t)
    ax.axhline(1, color=t["muted"], lw=0.8)
    for (mean, sem), color, label in zip(
        thirds, (ACCENT, t["blue"], t["aqua"]), labels, strict=True
    ):
        ax.errorbar(
            sizes,
            mean,
            sem,
            color=color,
            lw=1.5,
            marker="o",
            ms=4,
            capsize=0,
            label=label,
        )
    ax.errorbar(
        sizes,
        *overall,
        color=t["ink"],
        lw=1,
        marker="o",
        ms=3,
        capsize=0,
        label="all spectra",
    )
    ax.set_xscale("log")
    ax.set_xticks(sizes, [str(s) for s in sizes])
    ax.minorticks_off()
    ax.set_xlabel("pixels per prediction, n")
    ax.set_ylabel("mean of z² / (n fₛ)")
    legend(ax, 4)
    return fig


def noise_audit(r, t):
    rho, rho_train = r["rho"]
    fig, (ax,) = axes(2.6, t)
    ax.axhline(1, color=t["muted"], lw=0.8)
    ax.plot(r["wave"], rho_train, color=t["muted"], lw=1, label="training spectra")
    ax.plot(r["wave"], rho, color=ACCENT, lw=1, label="held-out spectra")
    ax.set_xlabel("rest-frame wavelength (Å)")
    ax.set_ylabel("ρ")
    legend(ax, 2)
    return fig


def models(r, t):
    edges = np.linspace(-12, 12, 97)
    fig, (ax,) = axes(2.6, t)
    x = np.linspace(-12, 12, 481)
    ax.plot(
        x,
        np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi),
        color=t["muted"],
        lw=1,
        label="N(0, 1)",
    )
    for name, color, label in (
        ("dense", t["blue"], "dense"),
        ("rank", ACCENT, "low rank, K = 40"),
    ):
        density = np.histogram(r["models"][name], edges, density=True)[0]
        ax.stairs(
            np.where(density > 0, density, np.nan),
            edges,
            baseline=None,
            color=color,
            lw=1.5,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1)
    ax.set_xlabel(f"z of held-out spectra without artifacts, all {r['npix']} pixels")
    ax.set_ylabel("density")
    legend(ax, 3)
    return fig


FIGURES = {
    "fitting-clipped": fitting,
    "pixelwise-z": pixelwise,
    "infill-gap": infill,
    "repair-spectrum": repair,
    "repair-recall": recall,
    "calibration-ratio": calibration,
    "noise-audit-rho": noise_audit,
    "models-z": models,
}


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
