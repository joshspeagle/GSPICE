# Fitting a model

This tutorial fits a model to a simulated survey, and compares the outliers that the
fit clips with the artifacts that the simulation injected. {doc}`../background/fitting`
explains the method.

## The survey

The tutorials use 10000 simulated spectra of a LAMOST-like survey, made from PHOENIX
stellar models by
[`docs/scripts/spectra.py`](https://github.com/joshspeagle/GSPICE/blob/main/docs/scripts/spectra.py).
The script downloads its inputs and writes `spectra.npz`. Its docstring describes the
spectra and the keys of the file. The spectra have 3741 pixels from 3802 to 8995 Å,
and 91% of them have a cosmic ray, a hot column or a shifted block.

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
blocks = slice(int(0.6 * n), int(0.8 * n))
test = slice(int(0.8 * n), n)
```

The tutorials use the 1500 pixels from 3802 to 5369 Å. A dense model needs many more
training spectra than pixels: the error of its covariance widens the residuals of
held-out spectra by the finite-sample factor $f_s$ of {doc}`../background/prediction`.
For a pixelwise prediction from 6000 training spectra, $f_s$ is 6.8 on all 3741
pixels and 1.7 on these 1500: `gspice.finite_sample_factor(6000, npix - 41)`, with
npix − 41 the pixels outside a guard window of 20 pixels on each side.
{doc}`models` shows the other remedy, a low-rank model of all the pixels.

The spectra are in random order. The first 60% train the model, the next 20%, without
their artifacts (`clean`), set the block thresholds of {doc}`repair`, and the last 20%
test it. `clean` and `clean_ivar` are the same spectra, with the same noise, without
the artifacts; `artifacts` marks the pixels that an artifact reaches.

## Fit

```python
model, mask = gspice.fit(flux[train], ivar[train])
print(model.n_train, model.mean.shape)
```

`fit` returns a {class}`gspice.Model` and the final training mask. `model.n_train`
holds the number of training spectra that the fit used, here all 6000, and
`model.mean` and `model.cov` the mean and the covariance in scaled units, one of each
per S/N bin.

## The clipped pixels

The survey has no bad pixels, so `mask` holds only the pixels that the clipping
passes flagged. The simulation knows the artifacts, so the flags can be checked:

```python
amplitude = np.abs(flux - clean) * np.sqrt(clean_ivar)
strong = artifacts & (amplitude > 5)
print(f"{mask.mean():.2%} of the training pixels are clipped")
print(f"{mask[strong[train]].mean():.0%} of the artifact pixels above 5 sigma")
print(f"{mask[~artifacts[train]].mean():.3%} of the pixels without an artifact")
```

`amplitude` is the change that an artifact made to a pixel, in units of its noise.
The fit clipped 0.10% of the training pixels: 63% of the artifact pixels above 5σ, and
0.031% of the pixels without an artifact. The clipping removes the largest artifacts,
which would widen the covariance. {doc}`repair` finds the artifacts of the test
spectra.

```{image} ../_static/figures/tutorials/fitting-clipped-light.svg
:alt: The fraction of training spectra clipped at each pixel, against the fraction with an injected artifact
:class: only-light
```

```{image} ../_static/figures/tutorials/fitting-clipped-dark.svg
:alt: The fraction of training spectra clipped at each pixel, against the fraction with an injected artifact
:class: only-dark
```

The figure shows, at each pixel, the fraction of the training spectra with an artifact
and the fraction that `fit` clipped. The narrow peaks are hot columns, which reach
nearly the same pixels in all the spectra of one spectrograph. The fit clips few pixels
at these peaks: a hot column recurs in up to 6.4% of the training spectra, so the
covariance describes it as a feature of the spectra.

## Other models

`fit` also fits models with a constant centre, S/N bins, a low rank or the noise of
each spectrum. {doc}`../background/fitting` compares them. {doc}`models` uses the
low-rank and the noise-aware model, and {doc}`calibration` discusses S/N bins.
