---
title: Migrating from the 2022 interface
---

The 2022 package had 11 functions at the top level of `gspice`, such as
`gspice.covar_iter_mask` and `gspice.gp_interp`. They now live in `gspice.legacy`,
which keeps their signatures and their arithmetic. [Legacy interface](/GSPICE/reference/legacy/)
documents them, and the
[changelog](https://github.com/joshspeagle/GSPICE/blob/main/CHANGELOG.md) lists the
replacement of each one in the new API and the behaviour changes of `gspice.legacy`.

## The deprecated names

The old names still work, with a `DeprecationWarning` that names the replacement. The
example catches the warning to print it:

```python
import warnings

import numpy as np

import gspice

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    covar_iter_mask = gspice.covar_iter_mask
print(caught[0].message)
print(covar_iter_mask is gspice.legacy.covar_iter_mask)
```

The names will be removed in 3.0.0. To keep the 2022 results without the warning,
import from `gspice.legacy`, for example
`from gspice.legacy import covar_iter_mask`. The 2022 modules moved too: insert
`legacy.` into the module path, as in `from gspice.legacy.specUtils import gp_interp`.

## From covar_iter_mask to fit

`covar_iter_mask` returns a covariance and a float mask. [`gspice.fit()`](/GSPICE/reference/api/#gspice.fit) returns a
[`gspice.Model`](/GSPICE/reference/api/#gspice.Model) and a bool mask. With `cumulative=False`, it uses the 2022 rule
for the masks of its clipping passes ([Fitting](/GSPICE/background/fitting/)), and its
docstring states the conditions under which the two agree:

```python
rng = np.random.default_rng(7)
sim = gspice.simulate(1000, 100, rng=rng, artifacts=0.3)
mask = np.zeros(sim.flux.shape, dtype=bool)

cov, finalmask = gspice.legacy.covar_iter_mask(sim.flux, sim.ivar, mask)
model, newmask = gspice.fit(sim.flux, sim.ivar, mask, cumulative=False)
print(np.allclose(model.cov[0], cov, rtol=0, atol=1e-12 * np.abs(cov).max()))
print(np.array_equal(newmask, finalmask != 0))
```

The default, `cumulative=True`, keeps the flags of every pass, as the paper does.
The model then replaces the 2022 arrays: pass it to [`gspice.pixelwise()`](/GSPICE/reference/api/#gspice.pixelwise) in place
of `gp_interp` and `get_chimask`, and to [`gspice.infill()`](/GSPICE/reference/api/#gspice.infill) in place of
`gaussian_estimate`. These take and return flux, where the 2022 functions took and
returned the scaled spectra of `standard_scale`.
