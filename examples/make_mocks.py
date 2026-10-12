"""
make_mocks.py -- LAMOST-like mock stellar spectra built from the PHOENIX-ACES grid.

Purpose
-------
Controllable mock data for the validation on simulated spectra in the GSPICE
paper (Finkbeiner, Speagle & Karim, arXiv:2511.16754).  The mocks mimic the
LAMOST DR5 low-resolution data used in the paper: R ~ 1800, vacuum wavelengths,
pixels uniform in log10(lambda) with step 1e-4 (~69 km/s), continuum normalized
by a ~41-pixel running smoother, S/N > 30, a dwarf-dominated population with a
red-giant/clump component.

Pipeline (one function per stage; see README.md for every parameter)
-----------------------------------------------------------------------
0. ``fetch_phoenix_nodes``   download the needed R=10000 PHOENIX nodes
                             (HTTP range requests into the per-[M/H] zip files).
1. ``preprocess_nodes``      air->vacuum, Gaussian LSF to R = 1600/1800/2000,
                             resample onto a 10x oversampled log10-lambda grid,
                             store log flux for every node in an HDF5 cache.
2. ``draw_labels``           LAMOST-like (Teff, logg, [Fe/H]) population.
3. ``synthesize_noiseless``  trilinear interpolation of log flux between nodes,
                             per-star LSF (interpolated between the precomputed
                             LSF planes), per-star velocity offset, 10->1 binning.
4. ``observe``               throughput (blue fall-off, dichroic dip), S/N draw,
                             photon + read noise, ivar.
5. ``continuum_normalize``   LAMOST-like smooth continuum from the noisy flux.
6. ``inject_artifacts``      cosmic rays, bad-column blocks, local wavelength-
                             calibration shifts, with a truth mask (on demand).
7. ``generate_dataset``      chunked driver that writes everything to HDF5.

All randomness flows from a single integer seed; every per-star draw (labels,
S/N, dv, R) is stored with the spectra.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict

import numpy as np
import h5py
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d, percentile_filter
from scipy.interpolate import BSpline

C_KMS = 299792.458
LN10 = np.log(10.0)

PHOENIX_BASE = "https://ftp.gwdg.de/pub/misc/phoenix/MedResFITS/R10000FITS/"  # mirror of phoenix.astro.physik.uni-goettingen.de/data
PHOENIX_R_NATIVE = 10000.0

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "phoenix_raw")
CACHE_FILE = os.path.join(HERE, "cache", "phoenix_nodes_oversampled.h5")

# PHOENIX grid steps in the region we use
TEFF_STEP, LOGG_STEP, FEH_STEP = 100.0, 0.5, 0.5
TEFF_RANGE, LOGG_RANGE, FEH_RANGE = (4000.0, 7000.0), (1.5, 5.0), (-2.0, 0.5)

# Label boxes of the two population components (nodes are only fetched here)
DWARF_BOX = dict(teff=(4500.0, 7000.0), logg=(4.0, 4.7))
GIANT_BOX = dict(teff=(4000.0, 5200.0), logg=(1.5, 3.5))


# --------------------------------------------------------------------------
# Wavelength grids
# --------------------------------------------------------------------------
@dataclass
class Grid:
    """Target pixel grid and its oversampled companion.

    Pixel i has centre log10(lam_i) = log10(lam_min) + i*dlog.  The
    oversampled grid places ``osamp`` sub-pixels uniformly inside every pixel
    and adds ``margin`` pixels on each side so velocity shifts never read off
    the edge.  Binning = mean of the ``osamp`` sub-pixels of each pixel.
    """
    lam_min: float = 3900.0
    lam_max: float = 6800.0
    dlog: float = 1e-4
    osamp: int = 10
    margin: int = 16

    def __post_init__(self):
        l0 = np.log10(self.lam_min)
        self.npix = int(np.floor((np.log10(self.lam_max) - l0) / self.dlog + 1e-9)) + 1
        self.loglam = l0 + self.dlog * np.arange(self.npix)
        self.wave = 10 ** self.loglam
        ntot = self.npix + 2 * self.margin
        sub = (np.arange(self.osamp) - (self.osamp - 1) / 2.0) * self.dlog / self.osamp
        centres = l0 + self.dlog * (np.arange(ntot) - self.margin)
        self.loglam_os = (centres[:, None] + sub[None, :]).ravel()
        self.nos = self.loglam_os.size
        self.dlog_os = self.dlog / self.osamp

    def bin(self, f_os):
        """(nstar, nos) -> (nstar, npix) by averaging sub-pixels and trimming margins."""
        n = f_os.shape[0]
        b = f_os.reshape(n, self.npix + 2 * self.margin, self.osamp).mean(axis=2)
        return b[:, self.margin:self.margin + self.npix]


# --------------------------------------------------------------------------
# Stage 0: PHOENIX download
# --------------------------------------------------------------------------
def _zstr(feh):
    return "-0.0" if abs(feh) < 1e-9 else f"{feh:+.1f}"


def phoenix_member(teff, logg, feh):
    return f"lte{int(round(teff)):05d}-{logg:.2f}{_zstr(feh)}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"


def phoenix_zip_url(feh):
    return PHOENIX_BASE + f"PHOENIX-ACES-AGSS-COND-2011_R10000FITS_Z{_zstr(feh)}.zip"


def needed_nodes():
    """All (Teff, logg, [Fe/H]) nodes that trilinear interpolation inside
    DWARF_BOX and GIANT_BOX x FEH_RANGE can touch."""
    fehs = np.arange(FEH_RANGE[0], FEH_RANGE[1] + 1e-6, FEH_STEP)
    nodes = set()
    for box in (DWARF_BOX, GIANT_BOX):
        t0 = np.floor(box["teff"][0] / TEFF_STEP) * TEFF_STEP
        t1 = np.ceil(box["teff"][1] / TEFF_STEP) * TEFF_STEP
        g0 = np.floor(box["logg"][0] / LOGG_STEP) * LOGG_STEP
        g1 = np.ceil(box["logg"][1] / LOGG_STEP) * LOGG_STEP
        for t in np.arange(t0, t1 + 1, TEFF_STEP):
            for g in np.arange(g0, g1 + 1e-6, LOGG_STEP):
                for z in fehs:
                    nodes.add((float(t), round(float(g), 2), round(float(z), 1)))
    return sorted(nodes)


def fetch_phoenix_nodes(nodes=None, raw_dir=RAW_DIR, nthreads=6):
    """Download PHOENIX R=10000 node spectra by HTTP range requests into the
    per-metallicity zip archives (only the requested members are transferred).
    Files land in raw_dir/Z<feh>/.  Existing files are skipped."""
    from remotezip import RemoteZip
    nodes = needed_nodes() if nodes is None else nodes
    by_z = {}
    for t, g, z in nodes:
        by_z.setdefault(z, []).append((t, g, z))

    def work(z):
        d = os.path.join(raw_dir, "Z" + _zstr(z))
        os.makedirs(d, exist_ok=True)
        todo = [n for n in by_z[z] if not os.path.exists(os.path.join(d, phoenix_member(*n)))]
        missing = []
        if not todo:
            return z, 0, missing
        with RemoteZip(phoenix_zip_url(z)) as zf:
            names = set(zf.namelist())
            for n in todo:
                m = phoenix_member(*n)
                if m not in names:
                    missing.append(n)
                    continue
                zf.extract(m, d)
        return z, len(todo), missing

    with ThreadPoolExecutor(nthreads) as ex:
        res = list(ex.map(work, sorted(by_z)))
    allmissing = [m for r in res for m in r[2]]
    return res, allmissing


# --------------------------------------------------------------------------
# Stage 1: node preprocessing
# --------------------------------------------------------------------------
def air_to_vac(lam_air):
    """Air -> vacuum (Angstrom), refractive index of Piskunov/VALD (Ciddor 1996 based)."""
    s = 1e4 / lam_air
    n = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s ** 2) \
        + 0.0001599740894897 / (38.92568793293 - s ** 2)
    return lam_air * n


def read_phoenix_r10000(path):
    """Returns (air wavelength [A], flux [erg/s/cm^2/cm]).  The R=10000 files are
    sampled uniformly in ln(lambda_air) (CTYPE1 = AWAV-LOG, CRVAL1 = ln 3000 A)."""
    from astropy.io import fits
    with fits.open(path, memmap=False) as h:
        f = h[0].data.astype(np.float64)
        hd = h[0].header
        assert hd["CTYPE1"].strip() == "AWAV-LOG", hd["CTYPE1"]
        lnlam = hd["CRVAL1"] + hd["CDELT1"] * np.arange(f.size)
    return np.exp(lnlam), f


def preprocess_spectrum(lam_air, flux, grid: Grid, R_list=(1600.0, 1800.0, 2000.0),
                        R_native=PHOENIX_R_NATIVE, lam_window=(3800.0, 7000.0)):
    """Air->vacuum, Gaussian LSF of constant resolving power R (FWHM = lam/R,
    accounting for the native R=10000 LSF in quadrature), resample to the
    oversampled log10 grid.  Returns (len(R_list), nos) linear flux."""
    sel = (lam_air > lam_window[0]) & (lam_air < lam_window[1])
    lam_vac = air_to_vac(lam_air[sel])
    f = flux[sel]
    dln = 1e-5  # native sampling in ln(lambda)
    lnv = np.log(lam_vac)
    ln_uni = np.arange(lnv[0], lnv[-1], dln)
    f_uni = np.interp(ln_uni, lnv, f)
    out = np.empty((len(R_list), grid.nos))
    ln_target = grid.loglam_os * LN10
    assert ln_target[0] > ln_uni[0] + 0.01 and ln_target[-1] < ln_uni[-1] - 0.01
    for k, R in enumerate(R_list):
        fwhm_ln = np.sqrt(1.0 / R ** 2 - 1.0 / R_native ** 2)
        sig_pix = fwhm_ln / (2 * np.sqrt(2 * np.log(2))) / dln
        fs = gaussian_filter1d(f_uni, sig_pix, mode="nearest", truncate=5.0)
        out[k] = np.interp(ln_target, ln_uni, fs)
    return out


def preprocess_nodes(grid: Grid, nodes=None, raw_dir=RAW_DIR, cache_file=CACHE_FILE,
                     R_list=(1600.0, 1800.0, 2000.0)):
    """Build the oversampled log-flux cache for all nodes.  log10 flux is
    stored after dividing by each node's median R=1800 flux (a per-node
    constant, irrelevant after S/N scaling and continuum normalization)."""
    nodes = needed_nodes() if nodes is None else nodes
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    logf = np.empty((len(nodes), len(R_list), grid.nos), dtype=np.float32)
    for i, (t, g, z) in enumerate(nodes):
        p = os.path.join(raw_dir, "Z" + _zstr(z), phoenix_member(t, g, z))
        lam, f = read_phoenix_r10000(p)
        fo = preprocess_spectrum(lam, f, grid, R_list)
        fo /= np.median(fo[list(R_list).index(1800.0)])
        logf[i] = np.log10(fo).astype(np.float32)
    with h5py.File(cache_file, "w") as h:
        h["nodes"] = np.array(nodes)
        h["logflux"] = logf
        h["R_list"] = np.array(R_list)
        h["loglam_os"] = grid.loglam_os
        h.attrs["grid"] = json.dumps(dict(lam_min=grid.lam_min, lam_max=grid.lam_max, dlog=grid.dlog,
                                          osamp=grid.osamp, margin=grid.margin))
    return cache_file


class NodeLibrary:
    """In-memory node cache with trilinear interpolation on a regular
    (Teff, logg, [Fe/H]) lattice.  Nodes absent from the cache may appear
    only with zero interpolation weight."""

    def __init__(self, cache_file=CACHE_FILE):
        with h5py.File(cache_file, "r") as h:
            self.nodes = h["nodes"][:]
            self.logf = h["logflux"][:]
            self.R_list = h["R_list"][:]
            self.loglam_os = h["loglam_os"][:]
            self.grid_kw = json.loads(h.attrs["grid"])
        self.t_ax = np.arange(TEFF_RANGE[0], TEFF_RANGE[1] + 1, TEFF_STEP)
        self.g_ax = np.arange(LOGG_RANGE[0], LOGG_RANGE[1] + 1e-6, LOGG_STEP)
        self.z_ax = np.arange(FEH_RANGE[0], FEH_RANGE[1] + 1e-6, FEH_STEP)
        self.index = -np.ones((self.t_ax.size, self.g_ax.size, self.z_ax.size), dtype=np.int64)
        for i, (t, g, z) in enumerate(self.nodes):
            it = int(round((t - self.t_ax[0]) / TEFF_STEP))
            ig = int(round((g - self.g_ax[0]) / LOGG_STEP))
            iz = int(round((z - self.z_ax[0]) / FEH_STEP))
            self.index[it, ig, iz] = i

    @staticmethod
    def _locate(x, ax):
        u = (x - ax[0]) / (ax[1] - ax[0])
        i = np.clip(np.floor(u).astype(int), 0, ax.size - 2)
        w = np.clip(u - i, 0.0, 1.0)
        return i, w

    def corners(self, teff, logg, feh):
        """Return (rows[n,8], weights[n,8])."""
        it, wt = self._locate(np.asarray(teff, float), self.t_ax)
        ig, wg = self._locate(np.asarray(logg, float), self.g_ax)
        iz, wz = self._locate(np.asarray(feh, float), self.z_ax)
        rows, ws = [], []
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    rows.append(self.index[it + a, ig + b, iz + c])
                    ws.append((wt if a else 1 - wt) * (wg if b else 1 - wg) * (wz if c else 1 - wz))
        rows, ws = np.stack(rows, 1), np.stack(ws, 1)
        bad = (rows < 0) & (ws > 1e-12)
        if bad.any():
            j = np.where(bad.any(1))[0][0]
            raise ValueError(f"label ({teff[j]}, {logg[j]}, {feh[j]}) needs a node missing from the cache")
        ws = np.where(rows < 0, 0.0, ws)
        rows = np.where(rows < 0, 0, rows)
        return rows, ws

    def interpolate(self, teff, logg, feh, R=None):
        """Noiseless oversampled linear flux (nstar, nos).  R (per star) is
        handled by linear interpolation in log flux between the two
        bracketing precomputed LSF planes."""
        n = np.size(teff)
        rows, ws = self.corners(teff, logg, feh)
        R = np.full(n, 1800.0) if R is None else np.asarray(R, float)
        Rl = self.R_list
        kR = np.clip(np.searchsorted(Rl, R, side="right") - 1, 0, Rl.size - 2)
        wR = np.clip((R - Rl[kR]) / (Rl[kR + 1] - Rl[kR]), 0.0, 1.0)
        out = np.zeros((n, self.logf.shape[2]), dtype=np.float32)
        for c in range(8):
            w = ws[:, c].astype(np.float32)
            lo = self.logf[rows[:, c], kR]
            exact = wR == 0.0
            if exact.all():
                out += w[:, None] * lo
            else:
                hi = self.logf[rows[:, c], kR + 1]
                wr = wR.astype(np.float32)[:, None]
                out += w[:, None] * ((1 - wr) * lo + wr * hi)
        return 10.0 ** out


# --------------------------------------------------------------------------
# Stage 2: population
# --------------------------------------------------------------------------
@dataclass
class PopulationConfig:
    f_giant: float = 0.25          # fraction of red giants + clump
    f_clump_of_giants: float = 0.35
    dwarf_teff_mu: float = 5900.0  # truncated normal on DWARF_BOX['teff']
    dwarf_teff_sig: float = 600.0
    dwarf_logg_at_4500: float = 4.62  # mild main-sequence Teff-logg relation
    dwarf_logg_at_7000: float = 4.15
    dwarf_logg_scatter: float = 0.12
    rgb_logg_mu: float = 2.6       # truncated normal on GIANT_BOX['logg']
    rgb_logg_sig: float = 0.5
    rgb_teff_at_logg25: float = 4700.0
    rgb_dteff_dlogg: float = 400.0
    rgb_dteff_dfeh: float = -250.0  # metal-poor giants are hotter
    rgb_teff_scatter: float = 80.0
    clump_teff_mu: float = 4800.0
    clump_teff_sig: float = 90.0
    clump_logg_mu: float = 2.45
    clump_logg_sig: float = 0.10
    feh_mu: float = -0.25
    feh_sig: float = 0.30
    f_metal_poor_tail: float = 0.04  # extra component uniform in [-2, -1]
    tail_range: tuple = (-2.0, -1.0)


def _truncnorm(rng, mu, sig, lo, hi, n):
    out = rng.normal(mu, sig, n)
    bad = (out < lo) | (out > hi)
    while bad.any():
        out[bad] = rng.normal(mu, sig, bad.sum())
        bad = (out < lo) | (out > hi)
    return out


def _redraw(rng, fn, lo, hi, n):
    """Draw n values from fn(rng, idx) and redraw those outside [lo, hi]
    (truncation rather than clipping, so no pile-up at the boundaries).
    fn receives the integer indices being (re)drawn."""
    idx = np.arange(n)
    out = np.empty(n)
    while idx.size:
        out[idx] = fn(rng, idx)
        idx = idx[(out[idx] < lo) | (out[idx] > hi)]
    return out


def draw_labels(n, rng, cfg: PopulationConfig = PopulationConfig()):
    """Returns dict teff, logg, feh, pop (0 dwarf, 1 RGB, 2 clump).
    All components are truncated (redrawn) at the boundaries of their label
    boxes, never clipped."""
    feh = _redraw(rng, lambda r, i: r.normal(cfg.feh_mu, cfg.feh_sig, i.size), *FEH_RANGE, n)
    tail = rng.random(n) < cfg.f_metal_poor_tail
    feh[tail] = rng.uniform(*cfg.tail_range, tail.sum())
    u = rng.random(n)
    pop = np.zeros(n, np.int8)
    pop[u < cfg.f_giant] = 1
    pop[u < cfg.f_giant * cfg.f_clump_of_giants] = 2
    teff = np.empty(n)
    logg = np.empty(n)
    d = np.where(pop == 0)[0]
    teff[d] = _truncnorm(rng, cfg.dwarf_teff_mu, cfg.dwarf_teff_sig, *DWARF_BOX["teff"], d.size)
    slope = (cfg.dwarf_logg_at_7000 - cfg.dwarf_logg_at_4500) / 2500.0
    gmean = cfg.dwarf_logg_at_4500 + slope * (teff[d] - 4500.0)
    logg[d] = _redraw(rng, lambda r, i: gmean[i] + r.normal(0, cfg.dwarf_logg_scatter, i.size),
                      *DWARF_BOX["logg"], d.size)
    g = np.where(pop == 1)[0]
    logg[g] = _truncnorm(rng, cfg.rgb_logg_mu, cfg.rgb_logg_sig, *GIANT_BOX["logg"], g.size)
    tmean = cfg.rgb_teff_at_logg25 + cfg.rgb_dteff_dlogg * (logg[g] - 2.5) + cfg.rgb_dteff_dfeh * feh[g]
    teff[g] = _redraw(rng, lambda r, i: tmean[i] + r.normal(0, cfg.rgb_teff_scatter, i.size),
                      *GIANT_BOX["teff"], g.size)
    c = np.where(pop == 2)[0]
    teff[c] = _truncnorm(rng, cfg.clump_teff_mu, cfg.clump_teff_sig, *GIANT_BOX["teff"], c.size)
    logg[c] = _truncnorm(rng, cfg.clump_logg_mu, cfg.clump_logg_sig, *GIANT_BOX["logg"], c.size)
    return dict(teff=teff, logg=logg, feh=feh, pop=pop)


# --------------------------------------------------------------------------
# Stage 3: per-star nuisances on the oversampled grid
# --------------------------------------------------------------------------
def doppler_shift_os(f_os, dv_kms, dlog_os):
    """Shift each oversampled spectrum by dv (km/s, positive = redshift):
    f_obs(log lam) = f_rest(log lam - log10(1 + dv/c)), linear interpolation."""
    n, m = f_os.shape
    s = np.log10(1.0 + np.asarray(dv_kms) / C_KMS) / dlog_os  # shift in sub-pixels
    if np.all(s == 0):
        return f_os
    j = np.arange(m)[None, :] - s[:, None]
    j0 = np.floor(j).astype(np.int64)
    w = (j - j0).astype(np.float32)
    j0c = np.clip(j0, 0, m - 1)
    j1c = np.clip(j0 + 1, 0, m - 1)
    rows = np.arange(n)[:, None]
    return (1 - w) * f_os[rows, j0c] + w * f_os[rows, j1c]


# --------------------------------------------------------------------------
# Stage 4: observation
# --------------------------------------------------------------------------
@dataclass
class ObsConfig:
    snr_min: float = 30.0
    snr_max: float = 400.0
    snr_power: float = 2.5          # p(S/N) ~ (S/N)^-power on [snr_min, snr_max]
    snr_window: tuple = (6100.0, 6700.0)  # S/N reference = median per-pixel S/N here
    read_noise: float = 8.0         # counts per binned pixel (rms)
    ivar_from: str = "model"        # 'model' (true variance) or 'observed' (from noisy counts)
    # throughput shape
    blue_edge: float = 4250.0       # logistic fall-off centre in the blue [A]
    blue_width: float = 220.0
    blue_floor: float = 0.15
    red_slope: float = -0.25        # fractional change per 1000 A redward of 5000 A
    dichroic_centre: float = 5800.0
    dichroic_sigma: float = 60.0
    dichroic_depth: float = 0.75


def throughput(wave, cfg: ObsConfig = ObsConfig()):
    """Smooth relative instrument throughput (max ~1): logistic fall-off in
    the blue, gentle slope in the red, Gaussian dip at the dichroic crossover."""
    blue = cfg.blue_floor + (1 - cfg.blue_floor) / (1 + np.exp(-(wave - cfg.blue_edge) / cfg.blue_width))
    red = 1 + cfg.red_slope * np.clip(wave - 5000.0, 0, None) / 1000.0
    dip = 1 - cfg.dichroic_depth * np.exp(-0.5 * ((wave - cfg.dichroic_centre) / cfg.dichroic_sigma) ** 2)
    T = blue * red * dip
    return T / T.max()


def draw_snr(n, rng, cfg: ObsConfig = ObsConfig()):
    """Inverse-CDF draw from p(S) ~ S^-a on [smin, smax]."""
    a, lo, hi = cfg.snr_power, cfg.snr_min, cfg.snr_max
    u = rng.random(n)
    e = 1 - a
    return (lo ** e + u * (hi ** e - lo ** e)) ** (1 / e)


def observe(f_true, wave, snr, rng, cfg: ObsConfig = ObsConfig()):
    """Scale noiseless flux to counts so the median per-pixel S/N in
    cfg.snr_window equals snr; add Gaussian noise with variance
    counts + read_noise^2.  Returns (counts_noisy, counts_true, ivar)."""
    T = throughput(wave, cfg)
    g = f_true * T[None, :]
    win = (wave > cfg.snr_window[0]) & (wave < cfg.snr_window[1])
    gref = np.median(g[:, win], axis=1)
    S2 = snr ** 2
    cref = 0.5 * (S2 + np.sqrt(S2 ** 2 + 4 * S2 * cfg.read_noise ** 2))  # c/sqrt(c+RN^2) = S
    k = cref / gref
    counts = g * k[:, None]
    var = counts + cfg.read_noise ** 2
    noisy = counts + rng.standard_normal(counts.shape).astype(np.float32) * np.sqrt(var)
    if cfg.ivar_from == "observed":
        var = np.clip(noisy, 0, None) + cfg.read_noise ** 2
    return noisy.astype(np.float32), counts.astype(np.float32), (1.0 / var).astype(np.float32)


# --------------------------------------------------------------------------
# Stage 5: continuum normalization
# --------------------------------------------------------------------------
@dataclass
class ContinuumConfig:
    method: str = "clipped_median"  # 'median' | 'clipped_median' | 'mean' | 'percentile'
    window: int = 41                # running-filter width in pixels (guard window 20 = half width)
    niter: int = 3                  # clipping iterations ('clipped_median')
    clip_low: float = 2.0           # reject r < -clip_low * MAD-sigma (absorption)
    clip_high: float = 3.0          # reject r > +clip_high * MAD-sigma
    percentile: float = 85.0        # for 'percentile' (upper-envelope flavour)
    knot_spacing: int = 20          # cubic B-spline knot spacing in pixels (final smoothing); 0 = none


_SPLINE_CACHE = {}


def spline_projector(npix, knot_spacing):
    """Least-squares cubic B-spline smoother as an (npix, npix) projection matrix."""
    key = (npix, knot_spacing)
    if key not in _SPLINE_CACHE:
        x = np.arange(npix, dtype=float)
        inner = np.arange(0, npix - 1, knot_spacing, dtype=float)[1:]
        t = np.r_[[0.0] * 4, inner, [npix - 1.0] * 4]
        B = BSpline.design_matrix(x, t, 3).toarray()
        P = B @ np.linalg.solve(B.T @ B, B.T)
        _SPLINE_CACHE[key] = P.astype(np.float32)
    return _SPLINE_CACHE[key]


def _running(f, cfg: ContinuumConfig):
    if cfg.method in ("median", "clipped_median"):
        s = median_filter(f, size=(1, cfg.window), mode="nearest")
    elif cfg.method == "mean":
        s = uniform_filter1d(f, cfg.window, axis=1, mode="nearest")
    elif cfg.method == "percentile":
        s = percentile_filter(f, cfg.percentile, size=(1, cfg.window), mode="nearest")
    else:
        raise ValueError(cfg.method)
    if cfg.knot_spacing:
        s = s @ spline_projector(f.shape[1], cfg.knot_spacing).T
    return s


def continuum_normalize(flux, cfg: ContinuumConfig = ContinuumConfig()):
    """Estimate a smooth continuum C from the (noisy) flux, vectorized over
    stars.  For 'clipped_median', pixels deviating by more than
    clip_low/clip_high robust sigmas from the current continuum are replaced
    by the continuum and the smoother is re-run niter times.  Returns C."""
    f = np.asarray(flux, dtype=np.float32)
    fw = f.copy()
    niter = cfg.niter if cfg.method == "clipped_median" else 0
    for _ in range(niter):
        C = _running(fw, cfg)
        r = f / C - 1
        med = np.median(r, axis=1, keepdims=True)
        sig = 1.4826 * np.median(np.abs(r - med), axis=1, keepdims=True)
        rej = (r < med - cfg.clip_low * sig) | (r > med + cfg.clip_high * sig)
        fw = np.where(rej, C, f)
    C = _running(fw, cfg)
    return np.clip(C, 1e-6 * np.abs(C).max(), None).astype(np.float32)


# --------------------------------------------------------------------------
# Stage 6: artifacts
# --------------------------------------------------------------------------
@dataclass
class ArtifactConfig:
    p_cosmic: float = 0.3           # fraction of spectra with >= 1 cosmic ray
    n_cosmic_mean: float = 2.0      # Poisson mean of the count (given >= 1)
    cosmic_width: tuple = (1, 3)    # pixels
    cosmic_amp_sigma: tuple = (5.0, 50.0)  # log-uniform, in units of the pixel noise
    p_badcol: float = 0.1
    badcol_width: tuple = (5, 20)
    badcol_level: tuple = (0.3, 0.8)  # fractional offset from local value, sign random
    p_wavecal: float = 0.05
    wavecal_width: tuple = (200, 600)
    wavecal_shift: tuple = (0.5, 3.0)  # pixels, sign random


MASK_COSMIC, MASK_BADCOL, MASK_WAVECAL = 1, 2, 4


def inject_artifacts(flux, ivar, rng, cfg: ArtifactConfig = ArtifactConfig(), edge=5):
    """Apply artifacts to a copy of normalized spectra (nstar, npix).

    cosmic rays: 1-3 adjacent pixels raised by A*sigma_i (A log-uniform);
    bad columns: a 5-20 pixel block multiplied by (1 +/- level);
    wavecal: a 200-600 pixel block replaced by the same spectrum shifted by
             0.5-3 pixels (linear interpolation of the star's own spectrum).
    ivar is left unchanged (the pipeline does not know about the artifact).
    Returns (flux_art, mask) with bit flags MASK_COSMIC|MASK_BADCOL|MASK_WAVECAL,
    plus a list of per-star event dicts."""
    x = np.array(flux, dtype=np.float32, copy=True)
    n, m = x.shape
    mask = np.zeros((n, m), np.uint8)
    sig = 1.0 / np.sqrt(np.clip(ivar, 1e-30, None))
    events = [[] for _ in range(n)]
    pix = np.arange(m)
    for s in range(n):
        if rng.random() < cfg.p_wavecal:
            w = rng.integers(cfg.wavecal_width[0], cfg.wavecal_width[1] + 1)
            a = rng.integers(edge, m - w - edge)
            sh = rng.uniform(*cfg.wavecal_shift) * rng.choice([-1, 1])
            blk = slice(a, a + w)
            x[s, blk] = np.interp(pix[blk] - sh, pix, flux[s])
            mask[s, blk] |= MASK_WAVECAL
            events[s].append(("wavecal", int(a), int(w), float(sh)))
        if rng.random() < cfg.p_badcol:
            w = rng.integers(cfg.badcol_width[0], cfg.badcol_width[1] + 1)
            a = rng.integers(edge, m - w - edge)
            lev = rng.uniform(*cfg.badcol_level) * rng.choice([-1, 1])
            x[s, a:a + w] *= (1 + lev)
            mask[s, a:a + w] |= MASK_BADCOL
            events[s].append(("badcol", int(a), int(w), float(lev)))
        if rng.random() < cfg.p_cosmic:
            k = 1 + rng.poisson(max(cfg.n_cosmic_mean - 1, 0))
            for _ in range(k):
                w = rng.integers(cfg.cosmic_width[0], cfg.cosmic_width[1] + 1)
                a = rng.integers(edge, m - w - edge)
                amp = np.exp(rng.uniform(*np.log(cfg.cosmic_amp_sigma)))
                prof = amp * np.r_[1.0, rng.uniform(0.2, 0.8, w - 1)]
                x[s, a:a + w] += prof * sig[s, a:a + w]
                mask[s, a:a + w] |= MASK_COSMIC
                events[s].append(("cosmic", int(a), int(w), float(amp)))
    return x, mask, events


# --------------------------------------------------------------------------
# Stage 7: data-set driver
# --------------------------------------------------------------------------
@dataclass
class MockConfig:
    n_star: int = 80000
    seed: int = 1
    sigma_v: float = 0.0            # km/s, Gaussian residual RV offsets
    R_mode: str = "fixed"           # 'fixed' or 'uniform'
    R_fixed: float = 1800.0
    R_range: tuple = (1600.0, 2000.0)
    chunk: int = 1000
    population: PopulationConfig = field(default_factory=PopulationConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    continuum: ContinuumConfig = field(default_factory=ContinuumConfig)


def generate_dataset(out_file, cfg: MockConfig, lib: NodeLibrary = None, verbose=True):
    """Generate cfg.n_star spectra in chunks and write them to HDF5.

    Datasets (float32 unless noted; spectra chunked (256, npix)):
      flux       continuum-normalized noisy flux
      ivar       inverse variance of `flux` (= ivar_counts * C^2)
      s_true     noiseless counts / C  (so flux - s_true is the pure propagated noise)
      continuum  C in counts (flux*C are the un-normalized noisy counts)
      labels     (n,3) Teff, logg, [Fe/H];  pop (int8: 0 dwarf, 1 RGB, 2 clump)
      snr, dv, R (n,)
      wave, loglam (npix,) vacuum Angstrom; throughput (npix,)
    The full configuration is stored as JSON in the file attribute 'config'."""
    lib = lib or NodeLibrary()
    grid = Grid(**lib.grid_kw)
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_star
    lab = draw_labels(n, rng, cfg.population)
    snr = draw_snr(n, rng, cfg.obs)
    dv = rng.normal(0.0, cfg.sigma_v, n) if cfg.sigma_v > 0 else np.zeros(n)
    if cfg.R_mode == "fixed":
        R = np.full(n, cfg.R_fixed)
    elif cfg.R_mode == "uniform":
        R = rng.uniform(*cfg.R_range, n)
    else:
        raise ValueError(cfg.R_mode)
    npix = grid.npix
    t0 = time.time()
    with h5py.File(out_file, "w") as h:
        ch = (min(256, n), npix)
        dsets = {k: h.create_dataset(k, (n, npix), dtype="f4", chunks=ch)
                 for k in ("flux", "ivar", "s_true", "continuum")}
        h["labels"] = np.stack([lab["teff"], lab["logg"], lab["feh"]], 1).astype("f4")
        h["labels"].attrs["columns"] = "teff,logg,feh"
        h["pop"] = lab["pop"]
        h["pop"].attrs["codes"] = "0 dwarf, 1 RGB, 2 red clump"
        h["snr"] = snr.astype("f4")
        h["dv"] = dv.astype("f4")
        h["R"] = R.astype("f4")
        h["wave"] = grid.wave
        h["loglam"] = grid.loglam
        h["throughput"] = throughput(grid.wave, cfg.obs)
        h.attrs["config"] = json.dumps(asdict(cfg))
        for a in range(0, n, cfg.chunk):
            b = min(a + cfg.chunk, n)
            # per-chunk RNG stream so results do not depend on chunk timing
            crng = np.random.default_rng([cfg.seed, a])
            f_os = lib.interpolate(lab["teff"][a:b], lab["logg"][a:b], lab["feh"][a:b], R[a:b])
            f_os = doppler_shift_os(f_os, dv[a:b], grid.dlog_os)
            f_true = grid.bin(f_os)
            del f_os
            noisy, counts, ivar = observe(f_true, grid.wave, snr[a:b], crng, cfg.obs)
            C = continuum_normalize(noisy, cfg.continuum)
            dsets["flux"][a:b] = noisy / C
            dsets["ivar"][a:b] = ivar * C ** 2
            dsets["s_true"][a:b] = counts / C
            dsets["continuum"][a:b] = C
            if verbose and (a // cfg.chunk) % 10 == 0:
                print(f"  {b}/{n} stars, {time.time() - t0:.0f} s", flush=True)
        h.attrs["runtime_s"] = time.time() - t0
    return out_file
