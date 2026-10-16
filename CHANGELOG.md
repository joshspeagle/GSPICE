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

### Removed

- `setup.py`, which `pyproject.toml` replaces.
- The egg in `dist/` and the 2022 tests in `gspice/tests/`, which could not run and which the golden tests replace.
- The import of astropy, which the 2022 code did not use.

### Behaviour changes

- The golden tests compare 11 outputs at a relative tolerance looser than 1e-10: 1e-9 for 7 outputs and 1e-8 for 4. These outputs come from ill-conditioned inputs, on which the 2022 results differ from a direct Cholesky solve by up to 7.6e-10.
- The 2022 modules `gspice.specUtils`, `gspice.matrixUtils` and `gspice.djs_maskinterp` moved to `gspice.legacy.specUtils`, `gspice.legacy.matrixUtils` and `gspice.legacy.djs_maskinterp`, and the unused `gspice.gspiceMain` is deleted. Imports of the old module paths fail, `from gspice import *` binds `legacy` instead of the three modules, and pickles that name the old paths no longer load. To migrate, insert `legacy.` into the module path, for example `from gspice.legacy.specUtils import standard_scale`. The top-level names, such as `gspice.standard_scale`, still work.
