# GSPICE: GausSian PIxelwise Conditional Estimation

GSPICE is a data-driven method for finding and repairing artifacts in large samples of spectra. It models an ensemble of spectra as a multivariate Gaussian with an empirical covariance matrix and estimates the expected value and variance of each pixel of each spectrum conditional on other pixels. Pixels that disagree with their estimate (cosmic rays, bad columns, wavelength-calibration errors) can be masked or replaced by the estimate, and every replaced pixel carries a conditional variance. The method is described in

> Finkbeiner, D. P., Speagle, J. S., & Karim, T., *Data-Driven Stellar Spectral Modelling with GSPICE*, The Astronomical Journal, [AJ reference to be added on publication]; [arXiv:2511.16754](https://arxiv.org/abs/2511.16754)

Version 1.0.0 accompanies the revised paper. See `CHANGELOG.md` for what changed relative to the 2022 code.

## Contents

| Directory | Content |
|---|---|
| `gspice/` | Python package (numpy, scipy): covariance, pixelwise and blockwise conditional estimates, iterative masking, repair |
| `pro/` | IDL implementation of the covariance, conditional estimate and iterative-masking routines |
| `julia/` | Julia implementation of the same routines |
| `examples/` | A runnable end-to-end example and the scripts that generate the mock spectra of the paper's Section 4 |
| `paper/` | Draft of the paper from 2020 (superseded by arXiv:2511.16754) |

## Installation

Python 3.9 or later with numpy and scipy:

```
git clone https://github.com/dfink/gspice.git
cd gspice
pip install .
```

To run the tests, use `pip install ".[test]"` and then `pytest`. The tests generate their own synthetic spectra and need no data files.

## Quick start

Arrays have shape `(nspec, npix)`, one spectrum per row, on a common wavelength grid (de-redshifted if the sample requires it). Masks are nonzero for bad pixels. The complete example below runs in under a minute with synthetic spectra; for real data replace the first two lines by `flux`, `ivar` and optionally `mask` arrays.

```python
import numpy as np
import gspice
from gspice.synthetic import make_synthetic_spectra, inject_outliers

flux, ivar, truth = make_synthetic_spectra(nspec=5000, npix=200, seed=0)
flux, spikes = inject_outliers(flux, ivar, nspike=100, amplitude=30.)

# 1. Covariance of the scaled spectra, estimated with iterative clipping (Algorithm 4).
#    refmean is the mean of the scaled spectra, needed to scale new spectra consistently.
cov, mask, refmean = gspice.covar_iter_mask(flux, ivar, cumulative=True, return_scale=True)

# 2. Detect and replace outlying pixels in any set of spectra.
newflux, newvar, info = gspice.gp_repair(flux, ivar, cov, refmean)

replaced = info['replace']
print(f"{replaced[spikes].mean():.2f} of the injected spike pixels replaced, "
      f"{(replaced & ~spikes).sum()} other pixels replaced (the neighbours of the spikes)")
```

`newflux` is `flux` with the replaced pixels filled in, and `newvar` is the conditional variance of the replaced pixels (zero elsewhere), both in the units of the input. The variance of a replaced pixel is that of the conditional estimate, not the original `1/ivar`, and it is correlated with the variances of the other replaced pixels of the same spectrum. The full conditional covariance is available with `gp_repair(..., return_covar=True)` and `gp_infill(..., return_covar=True)`.

### Building blocks

| Function | Purpose |
|---|---|
| `standard_scale` | Interpolate over masked pixels, scale each spectrum, subtract the mean (see below) |
| `unscale` | Transform predictions and variances back to flux units |
| `covar`, `covar_iter_mask` | Covariance of scaled spectra, with iterative outlier masking (Algorithm 4) |
| `gp_interp`, `gp_weights` | Pixelwise conditional estimate for every pixel, excluding a guard window (Algorithm 3) |
| `gaussian_estimate` | Conditional estimate for a given set of pixels (Algorithm 2) |
| `gp_infill` | Replace masked pixels of each spectrum by their conditional mean, with variance and covariance; optional pixels that are neither used nor replaced (the two-threshold scheme) |
| `exclude_extended` | Exclude the neighbourhood of extended artifacts from the conditioning set |
| `gp_blockwise`, `make_blocks`, `block_operators`, `block_chisq`, `block_thresholds` | Blockwise prediction and the blockwise test |
| `gp_repair` | The detection and repair procedure validated in Section 4 of the paper |

### Scaling convention

`standard_scale` interpolates linearly over masked pixels (`mask != 0`, or `ivar == 0` if no mask is given), multiplies each spectrum by `refscale = sqrt(mean(ivar))` over its unmasked pixels, so that the noise has approximately unit variance, and subtracts the mean of the scaled spectra, `refmean`. Covariances, predictions and variances from `covar`, `gp_interp` and `gp_infill` are in these scaled units. `unscale(pred, predvar, refscale, refmean)` returns `(pred + refmean) / refscale` and `predvar / refscale**2`, which are in the units of the input flux. `gp_repair` does this automatically.

### Choices that affect results

- `covar_iter_mask(cumulative=...)`. The default `False` reproduces the masking of all earlier versions, in which pass *n* uses the input mask and the flags of pass *n*-1 only. `cumulative=True` is Algorithm 4 as written in the paper (masks accumulate over passes) and is what the paper's simulations use. The two give nearly the same final mask.
- `gp_repair` applies the pixelwise test before the blockwise test, because a strong localized artifact otherwise contaminates the prediction of every block. Its defaults are those of the paper's simulations: replace pixels with |Z| > 10, condition only on pixels with |Z| < 5, and exclude 150 pixels on either side of any run of 5 or more flagged pixels from the conditioning set unless that would leave less than half the spectrum.
- Block thresholds are calibrated on held-out clean spectra with `block_thresholds`; they depend on the instrument and sample.

## Which implementation produced the LAMOST results

The LAMOST analysis in the paper was carried out with earlier IDL code of D. P. Finkbeiner, which predates the routines in this repository (to be confirmed by DPF). The IDL, Python and Julia routines here are later implementations of the same algorithms, and the driver scripts used for the LAMOST figures (blockwise and pixelwise runs, repair, and the diffuse-interstellar-band analysis) are not included (to be confirmed by DPF). The repair routines of the Python package were written for version 1.0.0 and were validated on the simulated spectra of Section 4 of the paper; the IDL and Julia routines have not changed and do not include them. The IDL and Julia unit tests need data files that are not part of the repository.

## Citation

If you use GSPICE, please cite the paper above and the archived software release:

> Finkbeiner, D. P., Speagle, J. S., & Karim, T., GSPICE v1.0.0, Zenodo, [Zenodo DOI to be added after the release is archived]

`CITATION.cff` contains the same information in machine-readable form.

## License

GNU General Public License v3.0; see `LICENSE`.
