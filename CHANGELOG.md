# Changelog

## 1.0.0 (release accompanying the revised paper; to be dated when tagged)

The previous state of the repository is commit `1fbab9f` (2022-08-17); there were no tags or releases and the Python package declared version 0.0.1.

### Added

- **Repair and infill (Python).** `gp_infill` replaces the masked pixels of each spectrum by their Gaussian conditional mean and returns the conditional variance and, optionally, the full conditional covariance of the replaced pixels. Pixels passed as `ignore` are neither used for conditioning nor replaced, which gives the two-threshold scheme (replace |Z| > 10, condition on |Z| < 5). `exclude_extended` excludes 150 pixels on either side of any run of 5 or more flagged pixels from the conditioning set (skipped for a spectrum if less than half of it would remain). `gp_repair` combines these with the pixelwise and blockwise tests in the order validated on simulated spectra (pixelwise, blockwise, second pixelwise pass, exclusion window, final infill).
- **Blockwise prediction (Python).** `gp_blockwise`, `make_blocks`, `block_operators`, `block_chisq` and `block_thresholds`.
- **Unscaling.** `unscale` returns predictions and variances in the units of the input flux (`(pred + refmean) / refscale` and `var / refscale**2`). Previously no routine did this, although the paper says that the estimates are scaled back. `standard_scale` accepts a `refmean` argument and `covar_iter_mask(..., return_scale=True)` returns the mean, so that new spectra can be scaled like the training set.
- **Cumulative iterative masking.** `covar_iter_mask(..., cumulative=True)` implements Algorithm 4 of the paper (`M <- M or M_Z`).
- `gp_weights`: the weights and variances of the pixelwise estimate, computed once and reusable (`gp_interp` is built on it).
- `nguard` keyword for `get_chimask` and `covar_iter_mask` (previously fixed at 20) and `verbose` keywords.
- `gspice.synthetic`: small synthetic spectra for tests and examples.
- A test suite that needs no data files, continuous-integration configuration (`.github/workflows/tests.yml`), `pyproject.toml`, `CITATION.cff`, `.zenodo.json`, an expanded README, and `examples/`.

### Fixed

- `gp_interp`: the `irange` check was inverted (`assert irange.size != 2`), so every valid range raised an error. It now requires two elements with `0 <= irange[0] <= irange[1] < npix`.
- `gaussian_estimate` and `gp_interp` with `bruteforce=True`: the brute-force branch returned three items and the default branch four, and `gp_interp` crashed on the brute-force path. Both branches now return `(predoverD, predcovar, predkstar, kstar)`. This changes the return value of direct calls to `gaussian_estimate(..., bruteforce=True)`.
- `gp_interp`: the product `spec @ predoverD` was computed and discarded for every pixel inside the loop and once more at the end. It is now computed once (`compute_pred=False` inside the loop), which removes the dependence of the loop cost on the number of spectra.
- `standard_scale`: a spectrum with no unmasked pixels produced `refscale = NaN` and made the covariance NaN. Such a spectrum now gets `refscale = 0`, is set to the mean spectrum, and a `RuntimeWarning` is issued.
- `covar_iter_mask`: `mask=None` now means `ivar == 0`, as documented.
- Progress messages are printed only with `verbose=True`; the unused `astropy` import was removed (the package depends on numpy and scipy only); the stale `dist/gspice-0.0.1-py3.7.egg` was removed from version control.
- Docstrings corrected (`imask` is a length-N vector; `Y` need not be zero padded at removed rows).

For `standard_scale`, `covar`, `gp_interp`, `get_chimask`, `covar_iter_mask` and `gaussian_estimate` with default arguments the outputs are bit-identical to those of commit `1fbab9f` on a synthetic test sample.

### Unchanged behaviour that differs from the paper or from what one might expect

These were left as they are so that results of earlier versions are reproduced. The choices are documented in the code.

- **Iterative masking is not cumulative by default.** In `covar_iter_mask` (and in the IDL and Julia routines) pass *n* uses `input_mask | chimask_{n-1}`, so pixels flagged in pass *n*-2 or earlier but not again in pass *n*-1 re-enter the covariance of pass *n*, whereas Algorithm 4 of the paper accumulates the masks. The final masks of the two schemes are nearly the same, but the intermediate covariances differ. Use `cumulative=True` for the paper's algorithm; the simulations in the paper use it. Whether to make it the default is a decision for the authors.
- **Covariance normalization.** Covariances use `N_s - 1`, whereas Algorithm 1 of the paper writes `N_s`. The conditional mean is unaffected and the conditional variance changes by `N_s/(N_s - 1)`.
- **Z-scores of input-masked pixels.** The covariance, prediction and Z-score in `get_chimask` use the data with masked pixels linearly interpolated, so pixels masked on input are tested at their interpolated values.
- **Dilation and comparison.** Flags are dilated by one pixel on each side, and the Python test is `|Z| >= nsigma` (IDL and Julia use `>`).
- **Guard window.** `nguard` is a half-width: pixels `j` with `|i - j| <= nguard` are excluded when predicting pixel `i`, i.e. `2 * nguard + 1` pixels.
- **Appendix A.** `submatrix_inv_mult` forms an explicit inverse of the small block `U`, whereas Appendix A describes Cholesky back-substitution (used in the IDL and Julia code). The results agree to rounding error. The IDL "contiguous" fast path of Appendix A is not implemented in Python.

### Known limitations

- The Schur-complement update used by `gp_interp` and `gp_weights` loses accuracy when the covariance is ill conditioned: on a covariance of 2415 pixels with condition number 2e8 the predicted variances have a relative error of about 1e-6 compared with a direct solve, against about 1e-11 for a solve with the precision matrix of the guard window. The Z-scores are unaffected at this level. The method was kept so that the results of earlier versions are reproduced.
- The conditional variance is the same for every spectrum in scaled units, so after `unscale` it follows each spectrum's overall noise level but not its wavelength-dependent noise shape.
- The IDL and Julia implementations are unchanged. They do not contain the repair routines, and their unit tests need data files that are not in the repository.
