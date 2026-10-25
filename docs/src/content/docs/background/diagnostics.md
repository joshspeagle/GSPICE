---
title: Diagnostics
---

Two functions check a model on [held-out spectra](/GSPICE/background/glossary/#term-held-out-spectrum), which the
model was not fitted to. [`gspice.calibration()`](/GSPICE/reference/api/#gspice.calibration) checks the predicted uncertainties.
[`gspice.noise_audit()`](/GSPICE/reference/api/#gspice.noise_audit) checks the reported noise, `ivar`. A residual of a training
spectrum is smaller than that of a held-out one, so neither check holds on the training
spectra.

## Calibration

For each spectrum and each size $n$, `calibration` draws a set $M$ of $n$ good pixels,
independently of the data. With `layout="run"`, $M$ is $n$ consecutive good pixels from
a random start; with `layout="scattered"`, $M$ is $n$ good pixels drawn at random. It
then predicts $M$ from the pixels outside $G = M \cup \mathrm{bad}$, as
[`gspice.infill()`](/GSPICE/reference/api/#gspice.infill) does, and computes

$$
z^2 = r_M^\mathsf{T} C_{MM}^{-1} r_M ,
$$

with $r_M$ the residuals and $C_{MM}$ their full conditional covariance. It also gives the
finite-sample factor $f_s = f_s(N_s, \mathrm{npix} - |G|, n)$ of [Prediction](/GSPICE/background/prediction/), or 1
for a model without a derived factor. The default sizes are those of 1, 4, 16, 64, 256
and 1024 that are not above half the median number of good pixels per spectrum. The
`rng` argument, a `numpy.random.Generator`, draws the masks.

For held-out Gaussian spectra and a correct model, $z^2 / (n f_s)$ has the mean 1. The
methods of [`gspice.Calibration`](/GSPICE/reference/api/#gspice.Calibration) read the result:

- `ratio()` returns the mean of $z^2 / (n f_s)$ for each size, and its standard error.
  A mean above 1 shows predicted variances that are too small, and a mean below 1
  variances that are too large. A mean that changes with $n$ shows that the
  covariance describes the correlations between the predicted pixels less well than
  their variances.
- `tail(q)` returns the fraction of entries with $z^2 / f_s$ above the quantile $q$ of
  $\chi^2_n$. The fraction approaches $1 - q$ only for $N_s - N_k \gg n$. A fraction
  above $1 - q$ shows a heavier tail than the Gaussian model.
- `qq(i)` returns the points of a quantile-quantile plot of column $i$, one size. The points lie on
  the line of slope 1 where $z^2 / f_s$ follows $\chi^2_n$.

```python
rng = np.random.default_rng(1)
result = gspice.calibration(model, flux, ivar, mask, rng=rng)
mean, sem = result.ratio(where=(result.n > 0).all(axis=1))
```

A spectrum with fewer good pixels than a size has no entry for that size. The
`where` argument above compares all sizes on the same spectra.

Two effects can move the mean away from 1 in a correct model. $f_s$ does not hold for
data that are not Gaussian, such as scaled spectra with a spread of S/N. A model without
a derived factor reports $f_s = 1$, so its mean includes the widening that $f_s$ would
describe.

The standard error is conditional on the fitted model: it falls with the
number of held-out spectra, and the spread between training sets does not.

`Calibration` also holds other held-out results. With `r` from `infill` and masks that
do not depend on the data, use
`Calibration(z2=r.chi2[:, None], n=r.dof[:, None], fs=r.fs[:, None])`. With `p` from
`pixelwise` with `inflate=True`, use `Calibration(z2=p.z**2, n=1, fs=1)`.

## Noise audit

`noise_audit` compares the residuals of the pixelwise prediction with the reported noise.
With $\hat f$ the prediction of [`gspice.pixelwise()`](/GSPICE/reference/api/#gspice.pixelwise) in flux units, and the mean over
the spectra whose pixel $i$ is good,

$$
\rho_i^2 = \operatorname{mean}_s\, (f_{si} - \hat f_{si})^2\, \mathrm{ivar}_{si} .
$$

Let $t$ be the noise-free flux. Then $E[\rho_i^2] = k_i + E[e_i]$, with two terms:

- $k_i$, the mean of $\mathrm{ivar}_{si} \operatorname{var}(f_{si} - t_{si})$, is 1
  where `ivar` is right.
- $e_i$, the mean of $\mathrm{ivar}_{si} (t_{si} - \hat f_{si})^2$, holds the error of
  the prediction: the spectral variation that the model does not predict, the noise of
  the conditioning pixels and the estimation error of the model. It is at least 0, and
  it grows with the S/N.

So $\rho_i^2 \ge k_i$ in expectation. A $\rho_i$ below 1 shows that `ivar` at pixel $i$
reports more noise than the spectra have. A $\rho_i$ above 1 can come from either term.

A model without `noise=True` takes into its covariance a noise level that its training
spectra share, so the pixelwise z-scores do not show an `ivar` that is wrong in all of
them; $\rho$ does.

The identity holds where $\hat f_{si}$ does not depend on the noise of $f_{si}$. This
needs held-out spectra, `mask` and `ivar` independent of the noise, and noise
uncorrelated beyond `nguard`. It can fail where the interpolation of a bad pixel carries
$f_{si}$, or a pixel within `nguard` of $i$, into $\hat f_{si}$. Next to a run of bad
pixels longer than `nguard`, $\rho_i^2$ can fall below $k_i$. `noise_audit` returns
$\rho$ of shape (npix,), with NaN where no spectrum is good.
