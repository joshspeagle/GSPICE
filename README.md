# GSPICE: Gaussian Pixelwise Conditional Estimation

GSPICE models spectra on a common wavelength grid as a multivariate Gaussian with an empirical covariance, and predicts any set of pixels from the others. The prediction can replace bad pixels, and the distance of a pixel from its prediction can flag artifacts. The method is in Finkbeiner, Speagle & Karim, [arXiv:2511.16754](https://arxiv.org/abs/2511.16754). The IDL and Julia implementations, notebooks and notes of 2021 and 2022 are in the upstream repository [dfink/gspice](https://github.com/dfink/gspice/tree/1fbab9f).

## Installation

See the [installation page](https://joshspeagle.github.io/GSPICE/installation/) of the documentation.

## Example

The functions take `flux` and `ivar` (inverse variance) arrays of shape (nspec, npix) on a common wavelength grid. An optional `mask` marks bad pixels with nonzero values, and a pixel with `ivar` = 0 is also bad. The example simulates its spectra.

```python
import numpy as np

import gspice

rng = np.random.default_rng(2026)
train = gspice.simulate(3000, 100, rng=rng, artifacts=0.3)
test = gspice.simulate(1000, 100, rng=rng, artifacts=0.3)

# Fit a mean and a covariance with iterative outlier clipping.
model, _ = gspice.fit(train.flux, train.ivar)

# Predict each pixel from the pixels outside a guard window around it.
prediction = gspice.pixelwise(model, test.flux, test.ivar)
far = np.abs(prediction.z) > 6
print(far.sum(), "pixels lie more than 6 standard deviations from their prediction")

# Mask the injected artifacts, and replace them by their conditional mean given the
# other pixels.
repaired = gspice.infill(model, test.flux, test.ivar, mask=test.artifacts)
print(repaired.replaced.sum(), "pixels replaced")

# Write the model to an .npz file and read it back.
gspice.save(model, "model.npz")
model = gspice.load("model.npz")
```

## The 2022 functions

`gspice.legacy` keeps the 2022 functions, with their signatures and results. The 2022 names at the top level of `gspice`, such as `gspice.gp_interp`, are deprecated. `CHANGELOG.md` lists their replacements and the behaviour changes of `gspice.legacy`.

## More

- [The documentation](https://joshspeagle.github.io/GSPICE/).
- [`docs/design.md`](docs/design.md): the design of the package and the plan of its phases.
- [`CHANGELOG.md`](CHANGELOG.md): the changes since the 2022 package.
