# Examples

| File | Content |
|---|---|
| `end_to_end_example.py` | Covariance with iterative clipping, calibration of block thresholds, detection and repair of injected artifacts, and comparison with the truth. Runs on small synthetic spectra by default, or on a subset of the mock spectra below. |
| `make_mocks.py` | Library that generates the mock LAMOST-like spectra of Section 4 of the paper from the PHOENIX-ACES synthetic grid, including artifact injection (`inject_artifacts`). |
| `run_mocks.py` | Command-line driver for `make_mocks.py` (`setup` and `make`). |

## Minimal example

From the repository root, after `pip install .`:

```
python examples/end_to_end_example.py
```

This needs only numpy and scipy and takes a few seconds. It generates 8000 synthetic spectra of 300 pixels, injects cosmic rays, bad columns and local wavelength shifts, and prints the fraction of artifact pixels that are replaced, the chi^2 against the noiseless truth before and after repair, the dispersion of the conditional residuals, and the false-positive rate on clean spectra, with and without the exclusion window around extended artifacts. The synthetic spectra are smooth and low-dimensional and do not reproduce the quantitative results of the paper.

## Mock spectra of Section 4

The mocks are built from PHOENIX-ACES-AGSS-COND-2011 model spectra (Husser et al. 2013) and mimic LAMOST DR5 low-resolution spectra (R about 1800, 2415 pixels with a step of 1e-4 in log10 wavelength).

Requirements: `numpy`, `scipy`, `h5py` and `remotezip`.

1. **Download and preprocess the model grid** (about 0.7 GB downloaded from the PHOENIX server, 0.25 GB cache, a few minutes). The files are written next to the scripts in `phoenix_raw/` and `cache/`, which are excluded from version control.

   ```
   cd examples
   python run_mocks.py setup
   ```

2. **Generate the spectra** (about 10 ms per star on one core; the base set of the paper has 80,000 stars and takes about 13 minutes, 3 GB). A smaller set is enough for a first run.

   ```
   python run_mocks.py make --name base --n 20000 --seed 1
   ```

   Spectra are stored in `data/base.h5` as `float32` arrays `flux`, `ivar` and the noiseless `s_true`, together with the labels, S/N and the other quantities listed in the docstring of `make_mocks.generate_dataset`. Using the same `--seed` and population settings reproduces the same stars and noise. Variants with radial-velocity offsets (`--sigma-v`) and line-spread-function variation (`--R-mode uniform`) are described in the docstring of `run_mocks.py`.

3. **Run the example on a subset** (about five minutes on two cores, mostly the iterative covariance estimate and the pixelwise weights):

   ```
   python end_to_end_example.py --mock data/base.h5 --ntrain 12000 --ntest 3000
   ```

## How the code corresponds to the procedure of Section 4

| Step in the paper | Package function |
|---|---|
| Training covariance with iterative clipping, 20, 8, 6 sigma, one-pixel dilation, guard half-width 20, cumulative masks | `gspice.covar_iter_mask(..., cumulative=True)` |
| Pixelwise detection, |Z| > 10 replaced, 5 < |Z| <= 10 not used for conditioning | `gspice.gp_repair` (step 1), `gspice.gp_weights` |
| Blockwise detection with 150-pixel blocks, thresholds from the 99.9th percentile of clean held-out spectra | `gspice.make_blocks`, `block_operators`, `block_thresholds`, `gp_repair` (step 2) |
| Second pixelwise pass on the repaired spectrum | `gspice.gp_repair` (step 3) |
| Exclusion of 150 pixels around runs of at least 5 flagged pixels, skipped if less than half the spectrum would remain | `gspice.exclude_extended` (`exclude_runs=True` in `gp_repair`) |
| Replacement by the conditional mean with conditional covariance | `gspice.gp_infill` |
| Back to flux units | `gspice.unscale` |

On the mock spectra, `gp_repair` reproduces the replaced, ignored and block masks of the script used for the detection and repair simulation of Section 4, and the repaired fluxes and variances to rounding error. The simulation scripts themselves (artifact injection with log-uniform amplitudes, the diagnosis of the conditional residuals, the figures) are not part of this repository.
