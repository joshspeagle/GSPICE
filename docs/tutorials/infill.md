# Infilling

{func}`gspice.infill` replaces chosen pixels of each spectrum by their conditional mean
given the other pixels, with its uncertainty. {doc}`../background/prediction` explains
the method.

The tutorial uses the survey and the model of {doc}`fitting`.
[`docs/scripts/spectra.py`](https://github.com/joshspeagle/GSPICE/blob/main/docs/scripts/spectra.py)
makes `spectra.npz`.

```python
import numpy as np

import gspice

data = np.load("spectra.npz")
pixels = (data["wave"] > 3800) & (data["wave"] < 5370)
wave = data["wave"][pixels]
flux = data["flux"][:, pixels]
ivar = data["ivar"][:, pixels]
clean = data["clean"][:, pixels]
clean_ivar = data["clean_ivar"][:, pixels]
artifacts = (data["cosmic"] | data["column"] | data["shift"])[:, pixels]

n = len(flux)
train = slice(0, int(0.6 * n))
test = slice(int(0.8 * n), n)
model, mask = gspice.fit(flux[train], ivar[train])
```

## Replace bad pixels

By default, `infill` replaces the bad pixels: those with a nonzero `mask` or
`ivar == 0`. Here the mask is the true artifacts of the test spectra:

```python
result = gspice.infill(
    model, flux[test], ivar[test], mask=artifacts[test], inflate=True
)

bad = artifacts[test]
pull = (result.flux - clean[test])[bad] / np.sqrt(result.var[bad])
print(f"standard deviation of the pulls: {np.std(pull):.2f}")
```

`result` is a {class}`gspice.Repaired`. `result.flux` holds the conditional mean at
the replaced pixels and the input flux elsewhere, and `result.var` the conditional
variance there, which includes the noise of the pixel. `inflate=True` includes the
finite-sample factor in the variances, as in {doc}`pixelwise`. The pull compares each
replaced pixel with the same pixel without its artifact. On the 2000 test spectra, its
standard deviation is 1.02: the variances describe the errors of the replaced values.

## Predict a range of pixels

A `replace` mask of shape (npix,) replaces the same pixels in every spectrum. This
predicts the pixels from 4840 to 4890 Å, around Hβ, from the rest of each spectrum:

```python
gap = (wave > 4840) & (wave < 4890)
rng = np.random.default_rng(3)
result = gspice.infill(
    model, clean[test], clean_ivar[test], replace=gap, covariance=True, rng=rng
)
index, cov = result.covariance(0)
print(f"mean chi2 / (dof fs): {np.mean(result.chi2 / (result.dof * result.fs)):.2f}")
```

```{image} ../_static/figures/tutorials/infill-gap-light.svg
:alt: The conditional mean of the pixels around H-beta, with its uncertainty and a draw
:class: only-light
```

```{image} ../_static/figures/tutorials/infill-gap-dark.svg
:alt: The conditional mean of the pixels around H-beta, with its uncertainty and a draw
:class: only-dark
```

The figure shows the first test spectrum. The conditional mean predicts the line from
the other pixels alone, and the band is the square root of `result.var`. With
`covariance=True`, `result.covariance(s)` returns the replaced pixels of spectrum `s`
and their covariance in flux², here of shape (45, 45) on the survey. With `rng`, `result.draws`
holds a draw from the conditional distribution, which follows the correlations of the
replaced pixels.

`result.chi2` compares the input values at the replaced pixels with their prediction,
and `result.dof` counts these pixels. The mean of `chi2 / (dof * fs)` is 1.00 on
the 2000 test spectra, as it is for a correct model. {doc}`calibration` makes this check at several
sizes.
