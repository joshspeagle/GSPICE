---
title: Diagnostics
description: What calibration and noise_audit measure, and when their results hold.
---

What do [`gspice.calibration()`](/GSPICE/reference/api/#gspice.calibration) and
[`gspice.noise_audit()`](/GSPICE/reference/api/#gspice.noise_audit) measure? `calibration`
checks the predicted uncertainties, and `noise_audit` checks the reported noise, `ivar`.
Both work on [held-out spectra](/GSPICE/concepts/glossary/#term-held-out-spectrum), which
the model was not fitted to. A training spectrum has smaller residuals than a held-out
one ([Held-out uncertainty](/GSPICE/concepts/held-out/)), so neither check holds on the
training spectra. [Check your model](/GSPICE/guides/checks/) runs both and reads their
numbers.

## Calibration

A predicted variance is right only if the residuals of real spectra match it. For each
spectrum and each size $n$, `calibration` therefore hides $n$ good pixels, predicts them,
and compares. It draws the set $M$ of hidden pixels independently of the data:

- with `layout="run"`, the default, $M$ is $n$ consecutive good pixels from a random
  start;
- with `layout="scattered"`, $M$ is $n$ good pixels drawn at random.

It then predicts $M$ from the pixels outside $G = M \cup \mathrm{bad}$, as
[`gspice.infill()`](/GSPICE/reference/api/#gspice.infill) does, and computes

$$
z^2 = r_M^\mathsf{T} C_{MM}^{-1} r_M ,
$$

with $r_M$ the residuals and $C_{MM}$ their full conditional covariance. It also gives
the [finite-sample factor](/GSPICE/concepts/glossary/#term-finite-sample-factor)
$f_s = f_s(N_s, \mathrm{npix} - |G|, n)$, or 1 for a model without a derived factor. The
default sizes are the powers of 4 from 1 to 1024 that do not exceed half the median
number of good pixels per spectrum. The `rng` argument, a `numpy.random.Generator`,
draws the masks.

For held-out Gaussian spectra and a correct model, $z^2 / (n f_s)$ has the mean 1. The
methods of [`gspice.Calibration`](/GSPICE/reference/api/#gspice.Calibration) read the
result:

- `ratio()` returns the mean of $z^2 / (n f_s)$ for each size, and its standard error.
  A mean above 1 shows predicted variances that are too small, and a mean below 1
  variances that are too large. A mean that changes with $n$ shows that the covariance
  describes the correlations between the predicted pixels less well than their
  variances.
- `tail(q)` returns the fraction of entries with $z^2 / f_s$ above the quantile $q$ of
  the χ² distribution with $n$ degrees of freedom. The fraction approaches $1 - q$ only
  for $N_s - N_k \gg n$. A fraction above $1 - q$ shows a heavier tail than the Gaussian
  model.
- `qq(i)` returns the points of a quantile-quantile plot of column $i$, one size. The
  points lie on the line of slope 1 where $z^2 / f_s$ follows that χ² distribution.

A spectrum with fewer good pixels than a size has no entry for that size. The `where`
argument of each method selects the spectra, for example those with an entry at every
size.

Two effects can move the mean away from 1 in a correct model. $f_s$ does not hold for
data that are not Gaussian, such as scaled spectra with a spread of S/N. A model without
a derived factor reports $f_s = 1$, so its mean includes the widening that $f_s$ would
describe.

The standard error is conditional on the fitted model. It falls with the number of
held-out spectra, and the spread between training sets does not.

`Calibration` also holds other held-out results, such as the χ² of an `infill` with
masks that do not depend on the data. Its [API entry](/GSPICE/reference/api/#gspice.Calibration)
gives the fields to pass.

## Noise audit

Every model trusts `ivar`. A [scaled model](/GSPICE/concepts/glossary/#term-scaled-model)
takes into its covariance the noise level that its training spectra share. Its z-scores
therefore cannot show an `ivar` that is wrong in all of them. `noise_audit` compares the
residuals of the pixelwise prediction with the reported noise instead. With $\hat f$
the prediction of [`gspice.pixelwise()`](/GSPICE/reference/api/#gspice.pixelwise) in flux
units, and the mean over the spectra whose pixel $i$ is good,

$$
\rho_i^2 = \operatorname{mean}_s\, (f_{si} - \hat f_{si})^2\, \mathrm{ivar}_{si} .
$$

Let $t$ be the [noise-free flux](/GSPICE/concepts/glossary/#term-noise-free-flux). Then
$E[\rho_i^2] = \eta_i + E[e_i]$, with two terms:

- $\eta_i$, the mean of $\mathrm{ivar}_{si} \operatorname{var}(f_{si} - t_{si})$, is 1
  where `ivar` is right.
- $e_i$, the mean of $\mathrm{ivar}_{si} (t_{si} - \hat f_{si})^2$, holds the error of
  the prediction. Its parts are the spectral variation that the model misses,
  the noise of the conditioning pixels and the estimation error of the model. It is at least 0, and
  it grows with the S/N.

So $\rho_i^2 \ge \eta_i$ in expectation. A $\rho_i$ below 1 shows that `ivar` at pixel
$i$ reports more noise than the spectra have. A $\rho_i$ above 1 can come from either
term.

The identity holds where $\hat f_{si}$ does not depend on the noise of $f_{si}$. This
needs held-out spectra, `mask` and `ivar` independent of the noise, and noise
uncorrelated beyond `nguard`. It can fail where the interpolation of a bad pixel carries
$f_{si}$, or a pixel within `nguard` of $i$, into $\hat f_{si}$. Next to a run of bad
pixels longer than `nguard`, $\rho_i^2$ can fall below $\eta_i$. `noise_audit` returns
$\rho$ of shape (npix,), with NaN where no spectrum is good.
