# Quickstart

{doc}`installation` installs GSPICE. This example simulates spectra with
{func}`gspice.simulate`, so it needs no download.

```python
import numpy as np

import gspice

rng = np.random.default_rng(2026)
train = gspice.simulate(3000, 500, rng=rng, artifacts=0.3)
clean = gspice.simulate(1000, 500, rng=rng)
test = gspice.simulate(500, 500, rng=rng, artifacts=0.3)
```

Each call returns a {class}`gspice.Simulation`: `flux` and `ivar`, the inverse
variance, of shape (nspec, npix), with the noise-free `truth` and the injected
`artifacts`. `artifacts=0.3` gives an artifact to 30% of the spectra.

Fit a model to the training spectra, then compute the block thresholds from spectra
without artifacts:

```python
model, mask = gspice.fit(train.flux, train.ivar)
model = gspice.fit_blocks(model, clean.flux, clean.ivar)
```

`fit` estimates the mean and the covariance of the spectra, and clips outliers on the
way. `mask` marks the pixels that it clipped. `fit_blocks` returns a new model that
also holds a threshold for each block of 150 pixels.

Find and replace the artifacts of the test spectra:

```python
result = gspice.repair(model, test.flux, test.ivar)

hit = result.replaced.any(axis=1)
has = test.artifacts.any(axis=1)
print(f"{hit[has].mean():.0%} of the spectra with an artifact have replaced pixels")
print(f"{hit[~has].mean():.1%} of the others have replaced pixels")
```

```text
70% of the spectra with an artifact have replaced pixels
0.9% of the others have replaced pixels
```

`result` is a {class}`gspice.Repaired`. `result.flux` holds the conditional mean at
the replaced pixels and the input flux elsewhere, `result.var` the variance of each
pixel, `result.replaced` and `result.ignored` the masks, and `result.blocks` the
blocks that the block scan flagged.

{doc}`tutorials/repair` measures what `repair` finds on a realistic survey, and the
other tutorials work through each step. {doc}`background/method` explains the model.
