---
title: What you get back
description: The fields of Model, Prediction, Repaired and Blocks, with their shapes and units.
---

This page lists the fields of the objects that GSPICE returns, with their shapes and
units. A value without a unit is a count, a mask or a ratio. In the shapes, nspec is the
number of spectra and npix the number of pixels. Every array that these objects hold is
read-only.

## Model

[`gspice.fit()`](/GSPICE/reference/api/#gspice.fit) returns a
[`gspice.Model`](/GSPICE/reference/api/#gspice.Model) and a mask:

```python
model, clipped = gspice.fit(flux, ivar)
```

The mask `clipped` has the shape (nspec, npix). It is True at each pixel that `fit` left
out of its final estimate. It is also True at every pixel of a spectrum that `fit` did
not use.

A model has one mean and one covariance for each
[S/N bin](/GSPICE/background/glossary/#term-S-N-bin). There is one bin unless you call
`fit(..., snr_bins=n)`. The mean and the covariance are in
[scaled units](/GSPICE/background/glossary/#term-scaled-units), not flux units.
[Continuum normalization](/GSPICE/start/spectra/#continuum-normalization) explains the
scaling.

| Field | Type and shape | Holds |
|---|---|---|
| `mean` | array (nbin, npix) | The mean of each bin, in scaled units |
| `cov` | tuple of nbin arrays (npix, npix), or of [`LowRank`](/GSPICE/reference/api/#gspice.LowRank) | The covariance of each bin, in scaled units squared |
| `n_train` | tuple of nbin integers | The number of training spectra of each bin |
| `edges` | tuple of nbin − 1 floats | The signal-to-noise ratios (S/N) between the bins |
| `center` | float or None | The constant centre in flux units, or None for a mean-centred model |
| `noise` | bool | True for a noise-aware model |
| `weighted` | bool | True if `fit` had unequal weights |
| `blocks` | `Blocks` or None | The block thresholds of `fit_blocks`, or None without them |

A [noise-aware model](/GSPICE/background/glossary/#term-noise-aware-model), from
`fit(..., noise=True)`, has one bin. Its `mean` is in flux units, and its `cov` is the
covariance of the flux without noise, in flux².

## Blocks

[`gspice.fit_blocks()`](/GSPICE/reference/api/#gspice.fit_blocks) returns a copy of the
model whose `blocks` is a [`gspice.Blocks`](/GSPICE/reference/api/#gspice.Blocks).
[`gspice.repair()`](/GSPICE/reference/api/#gspice.repair) reads it.

| Field | Type and shape | Holds |
|---|---|---|
| `starts` | array of int64 (nblock,) | The first pixel of each block |
| `width` | int | The number of pixels of each block |
| `nguard` | int | The half-width of the guard window around each block, in pixels |
| `thresholds` | array (nbin, nblock) | The threshold of each bin and block |

## Prediction

[`gspice.pixelwise()`](/GSPICE/reference/api/#gspice.pixelwise) predicts each pixel from
the pixels outside a [guard window](/GSPICE/background/glossary/#term-guard-window)
around it. It returns a [`gspice.Prediction`](/GSPICE/reference/api/#gspice.Prediction).

| Field | Shape | Holds |
|---|---|---|
| `flux` | (nspec, npix) | The prediction of each pixel, in flux units |
| `var` | (nspec, npix) | Its variance, which includes the noise, in flux² |
| `z` | (nspec, npix) | The input flux minus the prediction, divided by √`var`; NaN at bad pixels |

## Repaired

[`gspice.infill()`](/GSPICE/reference/api/#gspice.infill) and
[`gspice.repair()`](/GSPICE/reference/api/#gspice.repair) return a
[`gspice.Repaired`](/GSPICE/reference/api/#gspice.Repaired). The
[replaced pixels](/GSPICE/background/glossary/#term-replaced-pixel) hold their
prediction. The [ignored pixels](/GSPICE/background/glossary/#term-ignored-pixel) are
neither replaced nor used to predict the others.

| Field | Shape | Holds |
|---|---|---|
| `flux` | (nspec, npix) | The prediction at the replaced pixels, and the input flux elsewhere, in flux units |
| `var` | (nspec, npix) | In flux²: the variance of the prediction at the replaced pixels, which includes the noise; 1/`ivar` at the other good pixels; inf at the other bad pixels |
| `replaced` | (nspec, npix), bool | The replaced pixels |
| `ignored` | (nspec, npix), bool | The ignored pixels |
| `chi2` | (nspec,) | The χ² of the input values at the replaced pixels that are good in the input; 0 if there are none |
| `dof` | (nspec,), int64 | The number of those pixels |
| `fs` | (nspec,) | The [finite-sample factor](/GSPICE/background/glossary/#term-finite-sample-factor) of `chi2` |
| `draws` | (nspec, npix), or None | One random draw at the replaced pixels, and the input flux elsewhere, in flux units |
| `blocks` | (nspec, nblock), bool, or None | The blocks that `repair` flagged in each spectrum |

Three fields depend on the call:

- `draws` is None unless you pass `rng`, a `numpy.random.Generator`.
- `blocks` is None for a result of `infill`, and of `repair(..., blockwise=False)`.
- [`Repaired.covariance(s)`](/GSPICE/reference/api/#gspice.Repaired.covariance) returns
  the replaced pixels of spectrum `s` and their covariance, in flux². It needs a call
  with `covariance=True`.
