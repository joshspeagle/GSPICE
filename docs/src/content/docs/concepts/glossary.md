---
title: Glossary
description: The words and symbols of the GSPICE documentation.
---

What does this word or symbol mean? Each entry gives the meaning in plain words first,
then the symbol or the code name. [Symbols](#symbols) lists every symbol of the pages.

<dl>

<dt id="term-artifact">artifact</dt>
<dd>

  A corruption of the data, such as a cosmic ray, a bad column or a shifted segment.
  [`gspice.repair()`](/GSPICE/reference/api/#gspice.repair) finds artifacts that no mask
  marks.

</dd>

<dt id="term-bad-pixel">bad pixel</dt>
<dd>

  A pixel that the input marks: a nonzero `mask` or `ivar == 0`. GSPICE never reads
  `flux` or `ivar` there, and never conditions on it, except in the pixelwise
  prediction, which uses its interpolated value.

</dd>

<dt id="term-block">block</dt>
<dd>

  A run of `width` consecutive pixels that the block scan tests together, $A$.
  [`gspice.Blocks`](/GSPICE/reference/api/#gspice.Blocks) holds the first pixel of each
  block and their thresholds.

</dd>

<dt id="term-block-scan">block scan</dt>
<dd>

  The comparison of each block of a spectrum with its prediction from the pixels beyond
  its guard window, by the statistic $\chi^2_A$ of
  [Detection](/GSPICE/concepts/detection/#the-block-threshold).

</dd>

<dt id="term-chi2">χ²</dt>
<dd>

  One number that compares several observed values with their prediction:
  $r^\mathsf{T} C^{-1} r$, with $r$ the residuals and $C$ their conditional covariance.
  It counts correlated errors once. Its number of degrees of freedom is the number of
  values, `Repaired.dof` ([Prediction](/GSPICE/concepts/prediction/#χ-and-its-degrees-of-freedom)).

</dd>

<dt id="term-clean-spectrum">clean spectrum</dt>
<dd>

  A spectrum without an artifact.
  [`gspice.fit_blocks()`](/GSPICE/reference/api/#gspice.fit_blocks) sets its thresholds
  from clean spectra.

</dd>

<dt id="term-clipping-ceiling">clipping ceiling</dt>
<dd>

  The value that the in-sample $|z|$ of an outlier of any amplitude approaches in a
  clipping pass, in a spectrum of typical leverage. A pass whose threshold exceeds it
  flags few pixels, or none ([Fitting](/GSPICE/concepts/fitting/#the-clipping-ceiling)).

</dd>

<dt id="term-clipping-pass">clipping pass</dt>
<dd>

  One round of `fit`. It fits a covariance, computes the in-sample z-score of every
  training pixel, and flags the pixels above a threshold, `nsigma[n - 1]` for pass $n$.

</dd>

<dt id="term-conditional-distribution">conditional distribution</dt>
<dd>

  The Gaussian distribution of the pixels $G$ of a spectrum, given the values of its
  conditioning pixels $O$. Its mean is the prediction, and its covariance the
  uncertainty of the prediction.

</dd>

<dt id="term-conditional-mean">conditional mean</dt>
<dd>

  The mean of the conditional distribution: the average of all spectra in the model that
  agree with the conditioning pixels. It is the prediction, and it is smoother than any
  one spectrum. `infill` returns it in `Repaired.flux`.

</dd>

<dt id="term-conditional-variance">conditional variance</dt>
<dd>

  The variance of the conditional distribution at one pixel: the uncertainty of its
  prediction, noise included. `pixelwise` and `infill` return it in `var`, in flux².

</dd>

<dt id="term-conditioning-pixels">conditioning pixels</dt>
<dd>

  The pixels whose values a prediction uses, $O$: all pixels outside $G$. The paper
  calls this set $k$, and $N_k$ counts it.

</dd>

<dt id="term-constant-centre">constant centre</dt>
<dd>

  A constant $c$, in flux units, that a model subtracts from the flux instead of the
  mean of the training spectra (`fit(center=c)`).

</dd>

<dt id="term-correlation">correlation</dt>
<dd>

  The covariance of two pixels divided by the square root of the product of their
  variances, $C_{ii'} / \sqrt{C_{ii} C_{i'i'}}$: a number from −1 to 1.

</dd>

<dt id="term-counting-set">counting set</dt>
<dd>

  The pixels $R$ whose runs step 4 of `repair` counts: $M \cup I$ with `count_on="all"`
  ([Detection](/GSPICE/concepts/detection/#the-procedure)).

</dd>

<dt id="term-covariance">covariance</dt>
<dd>

  How the fluxes of two pixels vary together across the spectra. It is positive if they
  tend to rise together, and negative if one tends to rise as the other falls. The covariance
  matrix $C$ holds it for every pair of pixels, and its diagonal holds the variances.

</dd>

<dt id="term-dilation">dilation</dt>
<dd>

  The widening of a set of flagged pixels $F$ to $\mathrm{dil}(F)$: $F$ plus every pixel
  within a given number of pixels of a pixel of $F$ along the spectrum.

</dd>

<dt id="term-draw">draw</dt>
<dd>

  A random sample of the replaced pixels from their conditional distribution, in
  `Repaired.draws`. It scatters like an observed spectrum, unlike the conditional mean.

</dd>

<dt id="term-finite-sample-factor">finite-sample factor</dt>
<dd>

  $f_s$: the factor by which the error of an estimated mean and covariance widens the
  residuals of a held-out spectrum, $E[Z^2] / N_*$
  ([Held-out uncertainty](/GSPICE/concepts/held-out/#the-finite-sample-factor)).

</dd>

<dt id="term-flagged-pixel">flagged pixel</dt>
<dd>

  A pixel that [`gspice.repair()`](/GSPICE/reference/api/#gspice.repair) detects as an
  artifact. It becomes a replaced or an ignored pixel.

</dd>

<dt id="term-guard-window">guard window</dt>
<dd>

  The pixels within `nguard` of a predicted pixel or block. A prediction does not
  condition on them, and reports only the predicted pixels among them.

</dd>

<dt id="term-held-out-spectrum">held-out spectrum</dt>
<dd>

  A spectrum that the model was not fitted to.

</dd>

<dt id="term-ignored-pixel">ignored pixel</dt>
<dd>

  A pixel in $G$ that is not replaced: it is not conditioned on, and keeps its input
  value. `repair` reports its ignored pixels as $I$.

</dd>

<dt id="term-in-sample">in-sample</dt>
<dd>

  Computed on the training spectra of the model, as the z-scores of the clipping passes
  of `fit` are.

</dd>

<dt id="term-ivar">ivar</dt>
<dd>

  The inverse variance of the noise of each pixel, $1/\sigma^2$, in flux⁻². Every
  function takes it as the array `ivar`, and `ivar == 0` marks a bad pixel.

</dd>

<dt id="term-low-rank-model">low-rank model</dt>
<dd>

  A model whose covariance is $V \Lambda V^\mathsf{T} + D$: $K$ components and a
  diagonal $D$ (`fit(rank=K)`, [`gspice.LowRank`](/GSPICE/reference/api/#gspice.LowRank)).

</dd>

<dt id="term-marginalize">marginalize</dt>
<dd>

  To leave pixels out of a prediction without conditioning on them. Their values do not
  enter, and the covariance of the reported pixels includes their uncertainty.

</dd>

<dt id="term-multivariate-gaussian">multivariate Gaussian</dt>
<dd>

  A Gaussian distribution of several variables at once, here the fluxes of all pixels.
  A mean vector and a covariance matrix define it.

</dd>

<dt id="term-noise-aware-model">noise-aware model</dt>
<dd>

  A model of the noise-free flux, in flux units, whose prediction adds the noise of each
  spectrum from its `ivar` (`fit(noise=True)`). Its opposite is a scaled model.

</dd>

<dt id="term-noise-free-flux">noise-free flux</dt>
<dd>

  The flux that a spectrum would have without noise, $t$. `gspice.simulate` returns it
  as `truth`.

</dd>

<dt id="term-outlier">outlier</dt>
<dd>

  A pixel whose flux is many standard deviations from its prediction: a large $|z|$.

</dd>

<dt id="term-pattern">pattern</dt>
<dd>

  The pixel sets $G$, $M$ and $T$ of an infill. Spectra of one S/N bin with one pattern
  share one factorization within each chunk of spectra.

</dd>

<dt id="term-precision-matrix">precision matrix</dt>
<dd>

  The inverse of the covariance, $P = C^{-1}$. GSPICE computes the conditional
  distribution from its blocks $P_{GG}$ and $P_{GO}$.

</dd>

<dt id="term-prior">prior</dt>
<dd>

  The distribution $\mathcal{N}(0, C)$ of the working vector, without conditioning
  pixels. A prediction whose $G$ holds all pixels returns it.

</dd>

<dt id="term-replaced-pixel">replaced pixel</dt>
<dd>

  A pixel of the set $M$, whose value an infill replaces by its conditional mean.

</dd>

<dt id="term-S-N">S/N</dt>
<dd>

  The signal-to-noise ratio of a spectrum: the median of $f \sqrt{\mathrm{ivar}}$ over its
  good pixels ([`gspice.snr()`](/GSPICE/reference/api/#gspice.snr)).

</dd>

<dt id="term-S-N-bin">S/N bin</dt>
<dd>

  A range of S/N with its own mean and covariance (`fit(snr_bins=...)`), with the index
  $b$.

</dd>

<dt id="term-scaled-model">scaled model</dt>
<dd>

  A model in scaled units, whose one covariance holds both the variation between
  spectra and the noise. Every model is a scaled model except a noise-aware model.

</dd>

<dt id="term-scaled-units">scaled units</dt>
<dd>

  Flux multiplied by $a_s$, the square root of the mean inverse variance of the good
  pixels of the spectrum. The noise of a spectrum with equal `ivar` at every pixel then
  has the variance 1.

</dd>

<dt id="term-shifted-segment">shifted segment</dt>
<dd>

  An artifact in which a run of pixels has a shifted wavelength scale, so that its
  lines are recorded a few pixels off.

</dd>

<dt id="term-threshold">threshold</dt>
<dd>

  For the block scan, the percentile of $\chi^2_A$ of a block over clean spectra of one
  S/N bin, from [`gspice.fit_blocks()`](/GSPICE/reference/api/#gspice.fit_blocks). For
  z-scores, a limit on $|z|$, such as `nsigma_replace`.

</dd>

<dt id="term-training-spectra">training spectra</dt>
<dd>

  The spectra that [`gspice.fit()`](/GSPICE/reference/api/#gspice.fit) estimates the
  model from.

</dd>

<dt id="term-working-units">working units</dt>
<dd>

  The units of the working vector: scaled units, or flux units for a noise-aware model.

</dd>

<dt id="term-working-vector">working vector</dt>
<dd>

  The vector $x_s$ that the model describes: the interpolated flux of spectrum $s$ in
  working units. The mean of its S/N bin, or a constant centre, is subtracted from it.

</dd>

<dt id="term-z-score">z-score</dt>
<dd>

  The difference between the flux of a pixel and its prediction, divided by the square
  root of the conditional variance, $z$.

</dd>

</dl>

## Symbols

The symbols follow the paper [@finkbeiner2025gspice]. A subscript set, as in $C_{GO}$, selects those rows and columns.

### Pixel sets

| Symbol | Meaning |
|---|---|
| $G$ | The pixels that a prediction does not condition on: predicted, ignored, guard and bad pixels |
| $O$ | The conditioning pixels, all pixels outside $G$; the paper's $k$ |
| $M$ | The replaced pixels; in `calibration`, the hidden pixels. $M_1$ and $M_{\mathrm{blk}}$ are those of steps 1 and 2 of `repair` |
| $T$ | The replaced pixels that are good in the input |
| $I$ | The ignored pixels of `repair`; $I_1$ those of step 1 |
| $R$ | The counting set of step 4 of `repair` |
| $A$ | A block of the block scan |
| $g$ | The guard window of one pixel |
| $B_n$ | The mask of clipping pass $n$ of `fit` |
| $F_n$ | The pixels that pass $n$ flags |
| $\mathrm{dil}(F)$ | The dilation of the set $F$ |

### Vectors and matrices

| Symbol | Meaning |
|---|---|
| $f$, $\tilde f$, $\hat f$ | The flux, the flux interpolated over bad pixels, and its prediction; $f_{si}$ is pixel $i$ of spectrum $s$ |
| $t$ | The noise-free flux |
| $x_s$ | The working vector of spectrum $s$; $X$ is the matrix whose rows are the $x_s$, and $\hat x$ a prediction of $x$ |
| $x_{\mathrm{blk}}$ | The working vector after the replacements of the block pass of `repair` |
| $y_s$ | The scaled spectrum $a_s \tilde f_s$ |
| $m_b$ | The mean of S/N bin $b$ |
| $\mu$ | The mean of a noise-aware model, in flux units |
| $C$ | The covariance of the working vector; $C_b$ of bin $b$, $C_s$ of spectrum $s$ of a noise-aware model |
| $S$ | The sample covariance, before a low-rank approximation |
| $P$ | The precision matrix $C^{-1}$ |
| $V$, $\Lambda$, $D$ | The components, their variances and the diagonal of a low-rank covariance |
| $\Sigma$, $\Sigma_+$ | The covariance of the noise-free flux, and the same with its negative eigenvalues set to 0 |
| $\nu_s$ | The noise variances of spectrum $s$ in a noise-aware model |
| $r$ | Residuals: values minus their conditional mean |

### Numbers

| Symbol | Meaning |
|---|---|
| $a_s$ | The scale of spectrum $s$: the square root of its mean `ivar` over good pixels, in flux⁻¹ |
| $c$ | A constant centre, in flux units |
| $N_s$ | The number of training spectra of a model or bin |
| $N_k$ | The number of conditioning pixels |
| $N_*$ | The number of predicted pixels |
| $N_c$ | The number of clean spectra of a bin in `fit_blocks` |
| $K$ | The rank of a low-rank model |
| $n$ | The number of hidden pixels of `calibration`; also the index of a clipping pass, and the number of bins in `snr_bins=n` |
| $p$ | The percentile of the block thresholds |
| $q$ | A quantile of `Calibration.tail` |
| $w_s$ | The weight of spectrum $s$ in `fit` |
| $f_s$ | The finite-sample factor |
| $z$, $Z^2$ | The z-score of one pixel, and the χ² of several pixels of a held-out spectrum |
| $\chi^2_A$ | The block statistic of block $A$ |
| $\tau_r$, $\tau_i$ | The thresholds `nsigma_replace` and `nsigma_ignore` of `repair` |
| $\rho_i$, $\eta_i$, $e_i$ | The noise audit of pixel $i$, its noise term and its prediction term |
| $\kappa$, $\kappa_W$ | A condition number, and its bound for a low-rank covariance |
| $\epsilon$ | The float64 machine epsilon, $2.2 \times 10^{-16}$ |
| $s$, $b$ | The indices of a spectrum and of an S/N bin |
| $i$, $i'$ | The indices of two pixels |
| $j$ | The index of a block, in the docstrings of `Blocks` and `fit_blocks` |
