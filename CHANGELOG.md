# Changelog

## Unreleased

### Added

- Golden tests in `tests/golden/`: the outputs of every public function of the 2022 package (commit `1fbab9f`) on synthetic inputs, and tests that compare an implementation with them.
- Continuous integration on Linux with Python 3.11 to 3.15, on macOS and Windows with Python 3.11 and 3.15, and on the oldest supported numpy and scipy.
- `pyproject.toml`: `pip install .` installs the package. It requires Python 3.11 or later, numpy 1.26 or later and scipy 1.11 or later.
- `gspice.__version__`, which is `2.0.0.dev0`.
- Type checks with mypy in strict mode for the code outside `gspice.legacy`.
- A test coverage floor of 90% for the code outside `gspice.legacy`.
- pre-commit hooks that run ruff.
- `benchmarks/run.py`, which times `pixelwise`, `infill` and the 2022 `gp_interp`, and prints the timings and the setup as JSON.
- `gspice.scale`, `gspice.unscale` and `gspice.snr`. `scale` interpolates each spectrum over its bad pixels and scales it to unit mean noise, `unscale` returns scaled values and variances to flux units, and `snr` gives the median signal-to-noise ratio of each spectrum.
- `gspice.Model`, which holds the mean, the covariance and the training count of each S/N bin, and `gspice.save` and `gspice.load`, which write and read a model as one .npz file.
- `gspice.pixelwise`, which predicts each pixel of each spectrum from the pixels farther than `nguard` from it, and its result `gspice.Prediction`. A model keeps the precision matrix and the guard weights of each bin after their first use.
- `gspice.simulate`, which draws synthetic continuum-normalized spectra for tests and examples, and its result `gspice.Simulation`. The spectra have absorption lines whose depths follow two latent parameters in two populations, Gaussian noise, and optional cosmic rays, bad columns and shifted blocks. With `gaussian=True` and no artifacts they are exactly Gaussian.
- `gspice.infill`, which replaces chosen pixels of each spectrum by their Gaussian conditional mean given the other pixels, and its result `gspice.Repaired`. It reports the replaced and ignored pixels and χ², and with `covariance=True` the conditional covariance of the replaced pixels of each spectrum.
- `gspice.fit`, which fits a model to training spectra with the iterative outlier clipping of paper Alg. 4, and returns the model and the final training mask.

### Deprecated

- An access to one of the 11 functions of the 2022 package through `gspice`, as `gspice.<name>` or `from gspice import <name>`, emits a `DeprecationWarning`. It names the replacement and the removal of the name in 3.0.0. The access returns `gspice.legacy.<name>`, which keeps the 2022 results, except for the behaviour changes below, and does not warn. Type checkers report `gspice.<name>` as an unknown attribute.
- `gspice.standard_scale`: use `gspice.scale`.
- `gspice.covar` and `gspice.covar_iter_mask`: use `gspice.fit`.
- `gspice.get_chimask` and `gspice.gp_interp`: use `gspice.pixelwise`.
- `gspice.gaussian_estimate`: use `gspice.infill`.
- `gspice.cholesky_inv`, `gspice.submatrix_inv`, `gspice.submatrix_inv_mult`, `gspice.maskinterp1` and `gspice.maskinterp`: use `gspice.legacy.<name>`.

### Removed

- `setup.py`, which `pyproject.toml` replaces.
- The egg in `dist/` and the 2022 tests in `gspice/tests/`, which could not run and which the golden tests replace.
- The import of astropy, which the 2022 code did not use.
- The 10 prints of `gspice.legacy`, which reported progress, timings, shapes and notices such as "imask does not remove any rows or columns." `covar(checkmean=True)` still prints its check of the column means.
- The LAPACK message that `cholesky_inv` printed for a 0 × 0 matrix.

### Behaviour changes

- The golden tests compare 11 outputs at a relative tolerance looser than 1e-10: 1e-9 for 7 outputs and 1e-8 for 4. These outputs come from ill-conditioned inputs, on which the 2022 results differ from a direct Cholesky solve by up to 7.6e-10.
- The 2022 modules `gspice.specUtils`, `gspice.matrixUtils` and `gspice.djs_maskinterp` moved to `gspice.legacy.specUtils`, `gspice.legacy.matrixUtils` and `gspice.legacy.djs_maskinterp`, and the unused `gspice.gspiceMain` is deleted. Imports of the old module paths fail, and pickles that name the old paths no longer load. To migrate, insert `legacy.` into the module path, for example `from gspice.legacy.specUtils import standard_scale`.
- `gspice.__all__` holds the new public names and `legacy`, so `from gspice import *` binds them. It no longer binds the 11 functions and the 3 modules of the 2022 package. It can shadow names such as `load` and `save` of an earlier `from numpy import *`. To migrate, use `from gspice.legacy import *`, which binds the same 14 names as the 2022 star import, without a warning.
- The 2022 asserts of `gspice.legacy` raise `_LegacyError`, a subclass of `AssertionError` and `ValueError`, with a message that names the argument. An `except ValueError` clause now catches them, and the checks also run under `python -O`.
- `gp_interp` accepts `irange` as a list, a tuple or an array of at least two integers of any integer type, and raises unless 0 <= `irange[0]` <= `irange[1]` <= npix - 1. The 2022 code rejected a 2-element `irange` and a uint64 array, and for a negative start with `irange[1]` = npix - 1 it returned leading columns of zeros. A numpy bool in `irange[0]` or `irange[1]` raises `TypeError`, and spectra without pixels raise.
- `gp_interp` raises for a negative `nguard`, where the 2022 code predicted each pixel from itself. Under numpy 2, fixed-width numpy integers now work for `nguard`, where the 2022 code gave wrong windows or raised OverflowError or TypeError. A numpy bool or an array of one or more dimensions raises `TypeError` for `nguard`.
- When `bruteforce` is False and `imask` keeps every row, `submatrix_inv` and `submatrix_inv_mult` return a copy of `Minv` and `MinvY`. The 2022 code returned the argument itself, so a later change to the result changed the caller's array.
- `gp_interp` with `bruteforce=True` returns `pred` and `predvar` in the layout of the default path, where the 2022 code raised ValueError. The golden tests compare them with the 2022 outputs of the default path at the tolerances of those outputs, which are 1e-8 for 4 of them.
- In an array with one good entry, `maskinterp1` sets a non-finite value at a bad entry to the good value, and so do `maskinterp` and `djs_maskinterp.average_bilinear`, which call it. The 2022 code gave NaN there. `standard_scale` spread that NaN to a whole column of Dvec, and `get_chimask` and `covar_iter_mask` then raised ValueError.
- `standard_scale` ignores a non-finite ivar at a bad pixel. The 2022 code made the refscale of that spectrum NaN, and with it every value of Dvec and refmean, so `get_chimask` and `covar_iter_mask` raised ValueError.
- `standard_scale` and `get_chimask` raise `_LegacyError` for a spectrum without a good pixel, and name its row. `covar_iter_mask` raises it for a kept spectrum that its mask or the clipping masks completely, and names its row in `flux`. The 2022 code returned NaN or raised another error.
- `covar_iter_mask` accepts `mask=None`, which marks the pixels where ivar == 0, also for `maxbadpix`. The 2022 code raised numpy's AxisError.
- `scale` raises ValueError for a non-finite flux at a good pixel, a non-finite or negative ivar at a pixel not in `mask`, and zero spectra without `refmean`. For such inputs, the 2022 `standard_scale` returned NaN or a wrong refscale.
- `scale` counts a pixel with ivar = 0 as bad also where `mask` is given. It interpolates the flux of such a pixel and leaves the pixel out of refscale. The 2022 `standard_scale` with a mask used that flux and counted the 0 in the mean ivar.
- `scale` casts flux to float64 before it interpolates. The 2022 `standard_scale` interpolated in the dtype of the flux, so it truncated the interpolated values of integer flux and rounded those of float32 flux to float32.
- `scale` checks and computes a long double flux or ivar as float64. It raises ValueError for a long double beyond the float64 range at a good pixel, and an ivar that rounds to 0 in float64 marks a bad pixel. The 2022 `standard_scale` computed in long double and returned a finite result for such values.
- `scale` and `snr` raise ValueError for a spectrum without a good pixel, and name its row. For such a spectrum, the 2022 `standard_scale` returned NaN in every value of Dvec and refmean.
- `pixelwise` computes the conditional variances through the precision matrix, where the 2022 `gp_interp` used the Schur updates of a precomputed inverse. On the golden data set D3, whose covariance has the condition number 8.6e4, the 2022 predvar differs from a 60-digit reference by up to 7.6e-10 at nguard 5 and 2.7e-10 at nguard 20, relative. `pixelwise` differs by at most 1.7e-13 and 8.7e-14.
- `infill` computes the conditional covariance through the precision matrix, where the 2022 `gaussian_estimate` used the Schur updates of a precomputed inverse. On D3, the 2022 predcovar differs from a 60-digit reference by up to 3.0e-10 relative on the diagonal, and `infill` by at most 8.8e-14.
- `fit` masks the flags of every clipping pass, as paper Alg. 4 does. The 2022 `covar_iter_mask` masked only the flags of the last two passes; `fit(cumulative=False)` keeps that rule. On the golden data sets D1 to D4, the final mask of the default `fit` differs from the 2022 mask in 157, 152, 106 and 8 entries.
- `fit` flags a pixel whose |z| exceeds the threshold. The 2022 `get_chimask` also flagged a pixel whose |z| equals it.
- `fit` drops a spectrum that has no good pixel, or that a clipping mask covers completely. For such a spectrum, the 2022 `covar_iter_mask` returned a covariance of NaN or raised an error.
- `fit` needs at least npix + 1 spectra for a mean-centred model, and a good pixel in one of them at every pixel, and raises ValueError otherwise. The 2022 `covar_iter_mask` accepted npix spectra, whose covariance is singular, and then raised LinAlgError or flagged most pixels. It filled a pixel that is bad in every spectrum by interpolation, which also makes the covariance singular; on D1 it then raised LinAlgError.
- `fit` counts a pixel with ivar = 0 as bad, also against `maxbadpix`. The 2022 `covar_iter_mask` counted only the pixels in its mask.
