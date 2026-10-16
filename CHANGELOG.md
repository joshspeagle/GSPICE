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
- `benchmarks/run.py`, which times the conditioning primitives of pixelwise prediction and infill and the 2022 `gp_interp`, and prints the timings and the setup as JSON.

### Removed

- `setup.py`, which `pyproject.toml` replaces.
- The egg in `dist/` and the 2022 tests in `gspice/tests/`, which could not run and which the golden tests replace.
- The import of astropy, which the 2022 code did not use.
- The 10 prints of `gspice.legacy`, which reported progress, timings, shapes and notices such as "imask does not remove any rows or columns." `covar(checkmean=True)` still prints its check of the column means.
- The LAPACK message that `cholesky_inv` printed for a 0 × 0 matrix.

### Behaviour changes

- The golden tests compare 11 outputs at a relative tolerance looser than 1e-10: 1e-9 for 7 outputs and 1e-8 for 4. These outputs come from ill-conditioned inputs, on which the 2022 results differ from a direct Cholesky solve by up to 7.6e-10.
- The 2022 modules `gspice.specUtils`, `gspice.matrixUtils` and `gspice.djs_maskinterp` moved to `gspice.legacy.specUtils`, `gspice.legacy.matrixUtils` and `gspice.legacy.djs_maskinterp`, and the unused `gspice.gspiceMain` is deleted. Imports of the old module paths fail, `from gspice import *` binds `legacy` instead of the three modules, and pickles that name the old paths no longer load. To migrate, insert `legacy.` into the module path, for example `from gspice.legacy.specUtils import standard_scale`. The top-level names, such as `gspice.standard_scale`, still work.
- The 2022 asserts of `gspice.legacy` raise `_LegacyError`, a subclass of `AssertionError` and `ValueError`, with a message that names the argument. An `except ValueError` clause now catches them, and the checks also run under `python -O`.
- `gp_interp` accepts `irange` as a list, a tuple or an array of at least two integers of any integer type, and raises unless 0 <= `irange[0]` <= `irange[1]` <= npix - 1. The 2022 code rejected a 2-element `irange` and a uint64 array, and for a negative start with `irange[1]` = npix - 1 it returned leading columns of zeros. A numpy bool in `irange[0]` or `irange[1]` raises `TypeError`, and spectra without pixels raise.
- `gp_interp` raises for a negative `nguard`, where the 2022 code predicted each pixel from itself. Under numpy 2, fixed-width numpy integers now work for `nguard`, where the 2022 code gave wrong windows or raised OverflowError or TypeError. A numpy bool or an array of one or more dimensions raises `TypeError` for `nguard`.
- When `bruteforce` is False and `imask` keeps every row, `submatrix_inv` and `submatrix_inv_mult` return a copy of `Minv` and `MinvY`. The 2022 code returned the argument itself, so a later change to the result changed the caller's array.
- `gp_interp` with `bruteforce=True` returns `pred` and `predvar` in the layout of the default path, where the 2022 code raised ValueError. The golden tests compare them with the 2022 outputs of the default path at the tolerances of those outputs, which are 1e-8 for 4 of them.
- In an array with one good entry, `maskinterp1` sets a non-finite value at a bad entry to the good value, and so do `maskinterp` and `djs_maskinterp.average_bilinear`, which call it. The 2022 code gave NaN there. `standard_scale` spread that NaN to a whole column of Dvec, and `get_chimask` and `covar_iter_mask` then raised ValueError.
- `standard_scale` ignores a non-finite ivar at a bad pixel. The 2022 code made the refscale of that spectrum NaN, and with it every value of Dvec and refmean, so `get_chimask` and `covar_iter_mask` raised ValueError.
- `standard_scale` and `get_chimask` raise `_LegacyError` for a spectrum without a good pixel, and name its row. `covar_iter_mask` raises it for a kept spectrum that its mask or the clipping masks completely, and names its row in `flux`. The 2022 code returned NaN or raised another error.
- `covar_iter_mask` accepts `mask=None`, which marks the pixels where ivar == 0, also for `maxbadpix`. The 2022 code raised numpy's AxisError.
