"""Simulate LAMOST-like stellar spectra from PHOENIX models for the documentation figures.

The script downloads its inputs, so it runs outside CI:

    python docs/scripts/spectra.py --cache DIR [--nspec 10000] [--seed 119]

Inputs, cached in DIR and fetched only when missing:

- Stellar models: PHOENIX-ACES-AGSS-COND-2011, medium resolution R = 10000, version
  of 2013-04-19 (Husser et al. 2013, A&A 553, A6, doi:10.1051/0004-6361/201219058),
  https://ftp.gwdg.de/pub/misc/phoenix/MedResFITS/R10000FITS/. The library asks its
  users to cite Husser et al. 2013 and states no other terms. The script reads only the
  models it interpolates from each zip archive, with HTTP range requests, and checks
  the size and CRC-32 of each one: 543 models, about 380 MB, at the default nspec and seed.
- Sky emission lines: the UVES sky atlas (Hanuschik 2003, A&A 407, 1157), the line
  lists gident_*.dat at https://www.eso.org/observing/dfo/quality/UVES/txt/sky/: 139 kB.

Steps:

1. Population: 65% dwarfs, 25% red giants, 10% red clump stars, with [Fe/H] from a
   normal distribution N(-0.2, 0.25) truncated to [-1, 0.5].
   - Dwarfs: Teff ~ N(5800, 600) K truncated to [4000, 7000] K, and
     log g = 4.75 - 0.2 (Teff - 4000 K)/1000 K - U(0, 0.35).
   - Giants: Teff0 ~ U(3900, 5100) K, log g = 2.2 + 2.1 (Teff0 - 4500 K)/1000 K
     + N(0, 0.1), and Teff = Teff0 - 200 K [Fe/H].
   - Red clump: Teff ~ N(4750, 80) K and log g ~ N(2.45, 0.08).
   Radial velocities are N(0, 30) km/s, and S/N has the law p(S/N) ~ (S/N)^-2.5 on
   [30, 400].
2. Models: trilinear interpolation in (Teff, log g, [Fe/H]) of the models, converted
   from air to vacuum wavelengths and convolved with a Gaussian to R = 1800.
3. Instrument, in counts on the observed vacuum grid log10(lambda/A) = 3.57 + 1e-4 k,
   k = 0..3900 (3715 to 9120 A), from 16 spectrographs of 250 fibers:
   - star counts a F_lambda lambda^2 T(lambda), with a throughput T peaked at 6000 A
     and a dichroic dip at 5800 A, and a set so that the median S/N per pixel is the
     drawn S/N;
   - sky: a continuum flat in F_lambda at V = 21.7 mag/arcsec^2, 100 counts per pixel
     at 6000 A, plus the atlas lines at R = 1800, times a level ~ lognormal(0, 0.2);
   - artifacts: cosmic rays (Poisson(1.5) hits per spectrum of 1 to 4 pixels, a charge
     of 100 to 5000 counts per hit and 30% to 100% of it in each pixel), hot columns (Poisson(1) per spectrograph, 1 or 2 pixels
     wide, 30 to 1000 counts, in every fiber of the spectrograph at a position that
     curves by up to 3 pixels across the fibers), and shifted blocks (5% of spectra: a
     block of 100 to 400 pixels of the star recorded 0.2 to 3 pixels off);
   - noise: Gaussian photon noise of the expected counts and a read noise of 10
     counts per pixel;
   - pipeline: sky subtraction with a scale error N(0, 0.03) and a shift error
     N(0, 0.05) pixels per spectrum, variance max(counts, 0) + 10^2, flux calibration
     with a multiplicative error exp(sum_k c_k P_k), Legendre P_1 to P_3 in log lambda
     and c_k ~ N(0, 0.05), linear resampling to the rest frame with a radial velocity
     error N(0, 5) km/s, and continuum normalization.
4. Rest-frame grid: log10(lambda/A) = 3.58 + 1e-4 j, j = 0..3740 (3802 to 8995 A),
   vacuum. The continuum is a cubic spline through the 90th percentile of the flux in
   50 equal bins of pixels.

The output .npz holds, for nspec spectra and npix = 3741 pixels:

- wave (npix,): the rest-frame vacuum wavelength (A);
- flux, ivar (nspec, npix): the continuum-normalized flux and its inverse variance;
- clean, clean_ivar: the same spectra with the same noise and sky-subtraction errors,
  without the artifacts, normalized by the continuum of flux;
- truth: the spectra without noise, sky-subtraction errors or artifacts, normalized by
  the continuum of flux;
- continuum: the continuum, in calibrated flux (arbitrary units);
- cosmic, column, shift: bool truth masks, True at each pixel that a cosmic ray, a
  hot column or a shifted block reaches after the resampling;
- teff, logg, feh, kind (0 dwarf, 1 giant, 2 red clump), rv (km/s), snr (median of
  flux sqrt(ivar), which the resampling raises above the drawn S/N), spectrograph, fiber (nspec,).
"""

import argparse
import struct
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise, product
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import legvander
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.stats import truncnorm

PHOENIX = (
    "https://ftp.gwdg.de/pub/misc/phoenix/MedResFITS/R10000FITS/"
    "PHOENIX-ACES-AGSS-COND-2011_R10000FITS_Z{}.zip"
)
SKY = "https://www.eso.org/observing/dfo/quality/UVES/txt/sky/gident_{}.dat"
# The settings that cover 3750 to 10400 A; 800U fills the gap between 860L and 860U.
SKY_SETTINGS = ("437", "580L", "580U", "860L", "800U", "860U")
FEH_TAG = {-1.0: "-1.0", -0.5: "-0.5", 0.0: "-0.0", 0.5: "+0.5"}
TEFF = np.arange(3700.0, 7001.0, 100.0)
LOGG = np.arange(0.5, 5.01, 0.5)
FEH = np.array(sorted(FEH_TAG))

C = 299792.458  # km/s
R_MODEL = 10000
R_INSTRUMENT = 1800
SIGMA_FWHM = 2 * np.sqrt(2 * np.log(2))
# The models on 10 samples per sigma of the line-spread function (1.02e-4 in log10),
# so that linear interpolation errs by at most h^2 max|f''| / 8 = depth / 800 at a line.
MODEL_LOG = np.log10(3650) + 1e-5 * np.arange(40150)
OBS_LOG = 3.57 + 1e-4 * np.arange(3901)
REST_LOG = 3.58 + 1e-4 * np.arange(3741)
OBS_WAVE = 10**OBS_LOG

SKY_CONTINUUM = 0.076  # 1e-16 erg/s/cm^2/A/arcsec^2, V = 21.7 mag/arcsec^2
SKY_COUNTS = 100.0
READ_NOISE = 10.0
SPECTROGRAPHS, FIBERS = 16, 250
CHUNK = 250


def ranged(url: str, start: int, stop: int) -> bytes:
    request = urllib.request.Request(
        url, headers={"Range": f"bytes={start}-{stop - 1}"}
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data: bytes = response.read()
        except OSError as error:
            failure = error
        else:
            if len(data) == stop - start:
                return data
            failure = OSError(f"{url}: expected {stop - start} bytes, got {len(data)}")
        time.sleep(2**attempt)
    raise failure


def directory(url: str) -> dict[str, tuple[int, int, int, int]]:
    head = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(head, timeout=60) as response:
        size = int(response.headers["Content-Length"])
    # The end record is 22 bytes plus a comment of at most 65535 bytes.
    tail = ranged(url, max(size - 65557, 0), size)
    end = tail[tail.rfind(b"PK\x05\x06") :]
    if end[:4] != b"PK\x05\x06":
        raise OSError(f"{url}: no zip end record")
    count, length, offset = struct.unpack("<HLL", end[10:20])
    listing = ranged(url, offset, offset + length)
    entries, j = {}, 0
    for _ in range(count):
        fields = struct.unpack("<4s6H3L5H2L", listing[j : j + 46])
        method, crc, csize, usize, nname, nextra, ncomment = fields[4], *fields[7:13]
        name = listing[j + 46 : j + 46 + nname].decode()
        if method == 8:
            entries[name] = (crc, csize, usize, fields[16])
        j += 46 + nname + nextra + ncomment
    return entries


def member(url: str, entry: tuple[int, int, int, int], path: Path) -> None:
    crc, csize, usize, offset = entry
    nname, nextra = struct.unpack("<HH", ranged(url, offset + 26, offset + 30))
    start = offset + 30 + nname + nextra
    data = zlib.decompress(ranged(url, start, start + csize), -15)
    if len(data) != usize or zlib.crc32(data) != crc:
        raise OSError(f"{url}: {path.name} fails its size or CRC-32 check")
    partial = path.with_suffix(".part")
    partial.write_bytes(data)
    partial.replace(path)


def model_name(teff: float, logg: float, feh: float) -> str:
    return (
        f"lte{teff:05.0f}-{logg:.2f}{FEH_TAG[feh]}"
        ".PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    )


def download_models(nodes: list[tuple[float, float, float]], cache: Path) -> list[Path]:
    paths = [cache / "phoenix" / model_name(*node) for node in nodes]
    missing = [(node, path) for node, path in zip(nodes, paths) if not path.exists()]
    print(f"models: {len(nodes)} needed, {len(missing)} to download")
    if missing:
        (cache / "phoenix").mkdir(parents=True, exist_ok=True)
        urls = {feh: PHOENIX.format(FEH_TAG[feh]) for feh in {n[2] for n, _ in missing}}
        entries = {feh: directory(url) for feh, url in urls.items()}
        with ThreadPoolExecutor(8) as pool:
            jobs = [
                pool.submit(member, urls[n[2]], entries[n[2]][p.name], p)
                for n, p in missing
            ]
            for job in jobs:
                job.result()
    return paths


def read_fits(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    header, k = {}, 0
    while data[k : k + 3] != b"END":
        card = data[k : k + 80].decode("ascii")
        if card[8:10] == "= ":
            header[card[:8].strip()] = (
                card[10:].split("/")[0].strip().strip("'").strip()
            )
        k += 80
    if header["BITPIX"] != "-32" or header["CTYPE1"] != "AWAV-LOG":
        raise ValueError(f"{path}: expected float32 flux on a log air wavelength grid")
    n = int(header["NAXIS1"])
    # Raises ValueError if the file is shorter than its header states.
    flux = np.frombuffer(data, ">f4", n, offset=-(-(k + 80) // 2880) * 2880)
    wave = np.exp(float(header["CRVAL1"]) + float(header["CDELT1"]) * np.arange(n))
    return wave, flux.astype(np.float64)


def air_to_vacuum(wave: np.ndarray) -> np.ndarray:
    # The inversion of Morton (2000, ApJS 130, 403) by N. Piskunov for VALD3.
    s2 = (1e4 / wave) ** 2
    n = (
        1.00008336624212083
        + 0.02408926869968 / (130.1065924522 - s2)
        + 0.0001599740894897 / (38.92568793293 - s2)
    )
    return wave * n


def degrade(path: Path) -> np.ndarray:
    wave, flux = read_fits(path)
    keep = (wave > 3550) & (wave < 9400)
    step = np.log(wave[1] / wave[0])
    sigma = np.sqrt(R_INSTRUMENT**-2 - R_MODEL**-2) / SIGMA_FWHM / step
    flux = gaussian_filter1d(flux[keep], sigma)
    return np.interp(MODEL_LOG, np.log10(air_to_vacuum(wave[keep])), flux)


def needed_nodes(
    teff: np.ndarray, logg: np.ndarray, feh: np.ndarray
) -> list[tuple[float, float, float]]:
    corners = []
    for grid, values in ((TEFF, teff), (LOGG, logg), (FEH, feh)):
        if values.min() < grid[0] or values.max() > grid[-1]:
            raise ValueError("a star lies outside the model grid")
        lower = np.clip(np.searchsorted(grid, values) - 1, 0, len(grid) - 2)
        corners.append(np.stack([grid[lower], grid[lower + 1]]))
    nodes = set()
    for a, b, c in product(range(2), repeat=3):
        nodes.update(zip(corners[0][a], corners[1][b], corners[2][c]))
    return sorted(nodes)


def model_grid(
    nodes: list[tuple[float, float, float]], paths: list[Path]
) -> RegularGridInterpolator:
    cube = np.full((len(TEFF), len(LOGG), len(FEH), len(MODEL_LOG)), np.nan, np.float32)
    for (teff, logg, feh), path in zip(nodes, paths):
        index = np.searchsorted(TEFF, teff), np.searchsorted(LOGG, logg)
        cube[index + (np.searchsorted(FEH, feh),)] = degrade(path)
    # Nodes that no star uses stay NaN, so an interpolation that reached one would show.
    return RegularGridInterpolator((TEFF, LOGG, FEH), cube)


def sky_lines(cache: Path) -> tuple[np.ndarray, np.ndarray]:
    (cache / "sky").mkdir(parents=True, exist_ok=True)
    lines = []
    for setting in SKY_SETTINGS:
        path = cache / "sky" / f"gident_{setting}.dat"
        if not path.exists():
            with urllib.request.urlopen(SKY.format(setting), timeout=60) as response:
                partial = path.with_suffix(".part")
                partial.write_bytes(response.read())
                partial.replace(path)
        for row in path.read_text().splitlines():
            fields = row.split()
            if len(fields) == 5 and fields[0].isdigit():
                lines.append((float(fields[1]), float(fields[4])))
    wave, flux = np.array(sorted(lines)).T
    # The settings overlap; a line within 0.05 A of the one before it is a repeat.
    keep = np.diff(wave, prepend=0.0) > 0.05
    return air_to_vacuum(wave[keep]), flux[keep]


def throughput(wave: np.ndarray) -> np.ndarray:
    blaze = np.exp(-0.5 * (np.log(wave / 6000) / 0.3) ** 2)
    return blaze * (1 - 0.5 * np.exp(-0.5 * ((wave - 5800) / 40) ** 2))


def sky_spectrum(line_wave: np.ndarray, line_flux: np.ndarray) -> np.ndarray:
    use = (line_wave > OBS_WAVE[0] - 20) & (line_wave < OBS_WAVE[-1] + 20)
    center, flux = line_wave[use], line_flux[use]
    sigma = center / R_INSTRUMENT / SIGMA_FWHM
    profile = np.exp(-0.5 * ((OBS_WAVE[:, None] - center) / sigma) ** 2)
    lines = profile @ (flux / (np.sqrt(2 * np.pi) * sigma))
    scale = response(OBS_WAVE) / response(np.array(6000.0))
    return SKY_COUNTS * (1 + lines / SKY_CONTINUUM) * scale


def response(wave: np.ndarray) -> np.ndarray:
    # Counts per unit F_lambda: lambda (photons) times the pixel width, also ~ lambda.
    return (wave / 6000) ** 2 * throughput(wave)


def population(
    n: int, rng: np.random.Generator
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    kind = rng.choice(3, size=n, p=[0.65, 0.25, 0.10])
    feh = truncnorm.rvs(-3.2, 2.8, loc=-0.2, scale=0.25, size=n, random_state=rng)
    teff, logg = np.empty(n), np.empty(n)
    dwarf, giant, clump = (kind == k for k in range(3))
    nd, ng, nc = dwarf.sum(), giant.sum(), clump.sum()
    teff[dwarf] = truncnorm.rvs(-3, 2, loc=5800, scale=600, size=nd, random_state=rng)
    logg[dwarf] = 4.75 - 2e-4 * (teff[dwarf] - 4000) - rng.uniform(0, 0.35, nd)
    teff0 = rng.uniform(3900, 5100, ng)
    logg[giant] = 2.2 + 2.1e-3 * (teff0 - 4500) + rng.normal(0, 0.1, ng)
    teff[giant] = teff0 - 200 * feh[giant]
    teff[clump] = rng.normal(4750, 80, nc)
    logg[clump] = rng.normal(2.45, 0.08, nc)
    low, high = 30.0**-1.5, 400.0**-1.5
    snr = (low - rng.random(n) * (low - high)) ** (-1 / 1.5)
    stars = {
        "teff": teff,
        "logg": logg,
        "feh": feh,
        "kind": kind,
        "rv": rng.normal(0, 30, n),
        "spectrograph": rng.integers(0, SPECTROGRAPHS, n),
        "fiber": rng.integers(0, FIBERS, n),
    }
    return stars, snr


def hot_columns(rng: np.random.Generator) -> np.ndarray:
    n = rng.poisson(1.0, SPECTROGRAPHS)
    return np.column_stack(
        [
            np.repeat(np.arange(SPECTROGRAPHS), n),
            rng.integers(3, len(OBS_LOG) - 5, n.sum()),
            rng.uniform(-3, 3, n.sum()),
            rng.integers(1, 3, n.sum()),
            np.exp(rng.uniform(np.log(30), np.log(1000), n.sum())),
        ]
    )


def column_counts(
    columns: np.ndarray, spectrograph: np.ndarray, fiber: np.ndarray
) -> np.ndarray:
    counts = np.zeros((len(fiber), len(OBS_LOG)))
    curve = ((fiber - (FIBERS - 1) / 2) / ((FIBERS - 1) / 2)) ** 2
    for unit, center, bend, width, charge in columns:
        rows = np.flatnonzero(spectrograph == unit)
        start = np.rint(center + bend * curve[rows]).astype(int)
        for offset in range(int(width)):
            counts[rows, start + offset] += charge
    return counts


def cosmic_counts(n: int, rng: np.random.Generator) -> np.ndarray:
    counts = np.zeros((n, len(OBS_LOG)))
    rows = np.repeat(np.arange(n), rng.poisson(1.5, n))
    length = rng.choice([1, 2, 3, 4], size=len(rows), p=[0.5, 0.3, 0.15, 0.05])
    start = rng.integers(0, len(OBS_LOG) - length + 1)
    charge = np.exp(rng.uniform(np.log(100), np.log(5000), len(rows)))
    for offset in range(length.max(initial=0)):
        hit = length > offset
        deposit = charge[hit] * rng.uniform(0.3, 1, hit.sum())
        np.add.at(counts, (rows[hit], start[hit] + offset), deposit)
    return counts


def block_shifts(n: int, rng: np.random.Generator) -> np.ndarray:
    shift = np.zeros((n, len(OBS_LOG)))
    for row in np.flatnonzero(rng.random(n) < 0.05):
        length = rng.integers(100, 401)
        start = rng.integers(0, len(OBS_LOG) - length + 1)
        size = np.exp(rng.uniform(np.log(0.2), np.log(3)))
        shift[row, start : start + length] = rng.choice([-1, 1]) * size
    return shift


def resample(
    spectra: np.ndarray, grid: np.ndarray, position: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = (position - grid[0]) / (grid[1] - grid[0])
    lower = np.floor(index).astype(int)
    t = index - lower
    below = np.take_along_axis(spectra, lower, axis=1)
    above = np.take_along_axis(spectra, lower + 1, axis=1)
    return (1 - t) * below + t * above, lower, t


def amplitude(
    target: np.ndarray, star: np.ndarray, background: np.ndarray
) -> np.ndarray:
    # Bisect log a so that the median of a s / sqrt(a s + b) is the target S/N,
    # to 12 / 2^25 = 4e-7 in log a.
    low = np.log(target**2 / np.median(star, axis=1))
    high = low + 12
    for _ in range(25):
        mid = (low + high) / 2
        a = np.exp(mid)[:, None]
        snr = np.median(a * star / np.sqrt(a * star + background), axis=1)
        low, high = np.where(snr < target, mid, low), np.where(snr < target, high, mid)
    return np.exp((low + high) / 2)


def continuum(flux: np.ndarray) -> np.ndarray:
    edges = np.linspace(0, flux.shape[1], 51).astype(int)
    levels = [np.percentile(flux[:, a:b], 90, axis=1) for a, b in pairwise(edges)]
    centers = (edges[:-1] + edges[1:] - 1) / 2
    level = CubicSpline(centers, np.stack(levels, axis=1), axis=1)(
        np.arange(flux.shape[1])
    )
    if not (level > 0).all():
        raise ValueError("a continuum is not positive")
    return level


def observe(
    models: RegularGridInterpolator,
    sky: np.ndarray,
    columns: np.ndarray,
    stars: dict[str, np.ndarray],
    target: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    n = len(stars["teff"])
    points = np.column_stack([stars["teff"], stars["logg"], stars["feh"]])
    model = models(points)
    if not np.isfinite(model).all():
        raise ValueError("an interpolation reached a model that was not downloaded")
    rest = OBS_LOG - np.log10(1 + stars["rv"] / C)[:, None]
    shift = block_shifts(n, rng)
    gain = response(OBS_WAVE)
    unshifted = resample(model, MODEL_LOG, rest)[0] * gain
    shifted = resample(model, MODEL_LOG, rest + 1e-4 * shift)[0] * gain

    background = sky * rng.lognormal(0, 0.2, n)[:, None]
    scale = amplitude(target, unshifted, background + READ_NOISE**2)
    star0, star1 = scale[:, None] * unshifted, scale[:, None] * shifted
    cosmic = cosmic_counts(n, rng)
    column = column_counts(columns, stars["spectrograph"], stars["fiber"])
    photon, read = rng.standard_normal((2, n, len(OBS_LOG)))
    expected = {
        ("flux", "ivar"): star1 + background + cosmic + column,
        ("clean", "clean_ivar"): star0 + background,
    }
    error = rng.normal(0, [0.03, 0.05], (n, 2)).T
    sky_model = background * (1 + error[0, :, None])
    sky_model += error[1, :, None] * np.gradient(background, axis=1)
    x = np.linspace(-1, 1, len(OBS_LOG))
    calibration = gain * np.exp(legvander(x, 3)[:, 1:] @ rng.normal(0, 0.05, (3, n))).T

    # The pipeline resamples to the rest frame with a measured radial velocity.
    measured = REST_LOG + np.log10(1 + (stars["rv"] + rng.normal(0, 5, n)) / C)[:, None]
    out = {}
    for (key, ivar), mean in expected.items():
        counts = mean + np.sqrt(mean) * photon + READ_NOISE * read
        variance = (np.maximum(counts, 0) + READ_NOISE**2) / calibration**2
        out[key], lower, t = resample(
            (counts - sky_model) / calibration, OBS_LOG, measured
        )
        below = np.take_along_axis(variance, lower, axis=1)
        above = np.take_along_axis(variance, lower + 1, axis=1)
        out[ivar] = 1 / ((1 - t) ** 2 * below + t**2 * above)
    out["truth"] = resample(star0 / calibration, OBS_LOG, measured)[0]
    for key, source in (("cosmic", cosmic), ("column", column), ("shift", shift)):
        hit = source != 0
        below = np.take_along_axis(hit, lower, axis=1)
        above = np.take_along_axis(hit, lower + 1, axis=1)
        out[key] = below | (above & (t > 0))

    level = continuum(out["flux"])
    for key in ("flux", "clean", "truth"):
        out[key] /= level
    for key in ("ivar", "clean_ivar"):
        out[key] *= level**2
    out["continuum"] = level
    out["snr"] = np.median(out["flux"] * np.sqrt(out["ivar"]), axis=1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--nspec", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=119)
    parser.add_argument("--output", type=Path, help="default: CACHE/spectra.npz")
    args = parser.parse_args()
    if args.nspec < 1:
        parser.error(f"--nspec must be at least 1, got {args.nspec}")
    begin = time.perf_counter()
    rng = np.random.default_rng(args.seed)

    stars, target = population(args.nspec, rng)
    nodes = needed_nodes(stars["teff"], stars["logg"], stars["feh"])
    paths = download_models(nodes, args.cache)
    print(f"downloaded: {time.perf_counter() - begin:.0f} s")
    models = model_grid(nodes, paths)
    sky = sky_spectrum(*sky_lines(args.cache))
    columns = hot_columns(rng)
    print(f"models read: {time.perf_counter() - begin:.0f} s")

    chunks = []
    for start in range(0, args.nspec, CHUNK):
        chunk = {key: value[start : start + CHUNK] for key, value in stars.items()}
        out = observe(models, sky, columns, chunk, target[start : start + CHUNK], rng)
        chunks.append(
            {
                k: v.astype(np.float32) if v.dtype == np.float64 else v
                for k, v in out.items()
            }
        )
    result = {key: np.concatenate([c[key] for c in chunks]) for key in chunks[0]}
    output = args.output or args.cache / "spectra.npz"
    np.savez(output, wave=10**REST_LOG, **result, **stars)
    print(f"wrote {output}: {time.perf_counter() - begin:.0f} s")


if __name__ == "__main__":
    main()
