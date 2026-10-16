# Changelog

## Unreleased

### Added

- Golden tests in `tests/golden/`: the outputs of every public function of the 2022 package (commit `1fbab9f`) on synthetic inputs, and tests that compare an implementation with them.

### Behaviour changes

- The golden tests compare 11 outputs at a relative tolerance looser than 1e-10: 1e-9 for 7 outputs and 1e-8 for 4. These outputs come from ill-conditioned inputs, on which the 2022 results differ from a direct Cholesky solve by up to 7.6e-10.
