---
title: Migrate from the 2022 package
description: The new function for each 2022 function, its outputs and their units, and when the old names go.
---

I use the 2022 functions. What replaces each one? The 2022 package had 11 functions at
the top level of `gspice`, such as `gspice.covar_iter_mask` and `gspice.gp_interp`.
They now live in `gspice.legacy`, which keeps their signatures and their arithmetic.
[Legacy interface](/GSPICE/reference/legacy/) documents them.

## From scaled units to flux

The largest change is the units. The 2022 functions took and returned the scaled
spectra of `standard_scale`. A scaled spectrum is the flux times $a_s$, the square root
of the mean `ivar` of the good pixels, minus the mean spectrum. The new prediction functions take flux and
return flux, with variances in flux². The model in between, a
[`gspice.Model`](/GSPICE/reference/api/#gspice.Model), still holds its mean and
covariance in [scaled units](/GSPICE/concepts/glossary/#term-scaled-units).
[Working units](/GSPICE/concepts/model/#working-units) explains the scaling.

This example fits both ways and predicts every pixel. With `cumulative=False`, `fit` uses
the 2022 rule for the masks of its clipping passes. The two models then agree under the
conditions in the docstring of [`gspice.fit()`](/GSPICE/reference/api/#gspice.fit). The
main ones are the default `nguard`, no `center`, `weights`, `rank`, `snr_bins` or
`noise`, and float64 flux:

```python
import warnings

import numpy as np

import gspice

rng = np.random.default_rng(7)
sim = gspice.simulate(1000, 100, rng=rng, artifacts=0.3)
mask = np.zeros(sim.flux.shape, dtype=bool)

# 2022: scaled spectra in, scaled predictions out.
cov, finalmask = gspice.legacy.covar_iter_mask(sim.flux, sim.ivar, mask)
X, refscale, refmean = gspice.legacy.standard_scale(sim.flux, sim.ivar, finalmask)
pred, predvar = gspice.legacy.gp_interp(X, cov)

# New: flux in, flux out.
model, newmask = gspice.fit(sim.flux, sim.ivar, mask, cumulative=False)
result = gspice.pixelwise(model, sim.flux, sim.ivar, newmask)

a = refscale[:, None]
print(np.array_equal(newmask, finalmask != 0))
print(np.allclose(result.flux, (pred + refmean) / a, rtol=1e-7, atol=0))
print(np.allclose(result.var, predvar / a**2, rtol=1e-7, atol=0))
```

```text output
True
True
True
```

The 2022 predictions return to flux as $(\mathrm{pred} + \mathrm{refmean}) / a_s$, and
their variances as $\mathrm{predvar} / a_s^2$.
[`gspice.unscale()`](/GSPICE/reference/api/#gspice.unscale) does this for any scaled
values. The 2022 functions work from explicit inverses of the covariance, so their
variances differ from the new ones by a relative error of up to 8e-9 on one machine.
The comparison allows 1e-7. The default `cumulative=True` keeps the flags of every pass,
as the paper does, so its mask and model differ from the 2022 ones.

## Function by function

Each row gives the outputs and their units as 2022 → new.

| 2022 function | New function | Outputs | Units |
|---|---|---|---|
| `standard_scale` | [`scale`](/GSPICE/reference/api/#gspice.scale) | `Dvec, refscale, refmean` → `X, refscale, refmean` | scaled, flux⁻¹, scaled, in both |
| `covar` | [`fit`](/GSPICE/reference/api/#gspice.fit) with `nsigma=()` | `cov, refmean` → `Model`, mask | units of the input → scaled |
| `covar_iter_mask` | [`fit`](/GSPICE/reference/api/#gspice.fit) | `cov`, float mask → `Model`, bool mask | scaled, in both |
| `get_chimask` | [`fit`](/GSPICE/reference/api/#gspice.fit), then [`pixelwise`](/GSPICE/reference/api/#gspice.pixelwise) | bool mask → `z`; flag `abs(z) >= nsigma`, and each flag also covers its two neighbours | unit-free |
| `gp_interp` | [`pixelwise`](/GSPICE/reference/api/#gspice.pixelwise) | `pred, predvar` of `irange` → [`Prediction`](/GSPICE/reference/api/#gspice.Prediction): `flux, var, z` of every pixel | scaled → flux, flux² |
| `gaussian_estimate` | [`infill`](/GSPICE/reference/api/#gspice.infill) | `predoverD, predcovar, predkstar, kstar` → [`Repaired`](/GSPICE/reference/api/#gspice.Repaired): `flux, var, chi2` and, with `covariance=True`, `covariance(s)` | scaled → flux, flux² |
| `cholesky_inv`, `submatrix_inv`, `submatrix_inv_mult`, `maskinterp1`, `maskinterp` | none: keep `gspice.legacy.<name>` | unchanged | unchanged |

The new functions have no 2022 equivalent for the block scan and the search for
artifacts: [Find and repair artifacts](/GSPICE/guides/repair/) shows
[`gspice.repair()`](/GSPICE/reference/api/#gspice.repair).

## When the old names go

| Version | What changes |
|---|---|
| 2.0.0, 2026-10-29 | The 11 functions move to `gspice.legacy`. `gspice.<name>` still works, with a `DeprecationWarning`. The 2022 modules move, and their old paths fail. |
| 3.0.0 | `gspice.<name>` is removed. `gspice.legacy` stays. |

The warning names the replacement. The example catches it to print it:

```python
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    covar_iter_mask = gspice.covar_iter_mask
print(caught[0].message)
print(covar_iter_mask is gspice.legacy.covar_iter_mask)
```

```text output
gspice.covar_iter_mask is deprecated and will be removed in 3.0.0. Use gspice.fit, or gspice.legacy.covar_iter_mask to keep the 2022 results.
True
```

To keep the 2022 results without the warning, import from `gspice.legacy`, for example
`from gspice.legacy import covar_iter_mask`. For a 2022 module, insert `legacy.` into
its path, as in `from gspice.legacy.specUtils import gp_interp`.

`gspice.legacy` also fixes crashes and wrong results of the 2022 code on some inputs.
The [changelog](/GSPICE/reference/changelog/#200---2026-10-29) lists these behaviour
changes under 2.0.0.
