---
title: Glossary
---

<dl>

<dt id="term-bad-pixel">bad pixel</dt>
<dd>

  A pixel with a nonzero `mask` or `ivar == 0`. GSPICE never reads `flux` or `ivar`
  there, and never conditions on it, except in the pixelwise prediction, which uses its
  interpolated value.

</dd>

<dt id="term-block">block</dt>
<dd>

  `width` consecutive pixels of the block scan. [`gspice.Blocks`](/GSPICE/reference/api/#gspice.Blocks) holds the first
  pixel of each block and their thresholds.

</dd>

<dt id="term-block-scan">block scan</dt>
<dd>

  The comparison of each block of a spectrum with its prediction from the pixels beyond
  its guard window, by the statistic $\chi^2_A$ of [Detection and repair](/GSPICE/background/repair/).

</dd>

<dt id="term-conditional-distribution">conditional distribution</dt>
<dd>

  The Gaussian distribution of the pixels $G$ of a spectrum, given the values of its
  conditioning pixels $O$. Its mean is the prediction, and its covariance the
  uncertainty of the prediction.

</dd>

<dt id="term-conditioning-pixels">conditioning pixels</dt>
<dd>

  The set $O$: the pixels whose values a prediction uses. All pixels outside $G$.

</dd>

<dt id="term-constant-centre">constant centre</dt>
<dd>

  A constant $c$, in flux units, that a model subtracts from the flux instead of the
  mean of the training spectra (`fit(center=c)`).

</dd>

<dt id="term-dilation">dilation</dt>
<dd>

  $\mathrm{dil}(F)$: the set $F$ of flagged pixels plus every pixel within a given
  number of pixels of a pixel of $F$ along the spectrum.

</dd>

<dt id="term-draw">draw</dt>
<dd>

  A random sample of the replaced pixels from their conditional distribution, in
  `Repaired.draws`.

</dd>

<dt id="term-finite-sample-factor">finite-sample factor</dt>
<dd>

  $f_s$: the factor by which the error of an estimated mean and covariance widens the
  residuals of a held-out spectrum, $E[Z^2] / N_*$ (see [Prediction](/GSPICE/background/prediction/)).

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
  value.

</dd>

<dt id="term-in-sample">in-sample</dt>
<dd>

  Computed on the training spectra of the model, as the z-scores of the clipping passes
  of `fit` are.

</dd>

<dt id="term-low-rank-model">low-rank model</dt>
<dd>

  A model with the covariance $V \Lambda V^\mathsf{T} + D$ of $K$ components and a
  diagonal $D$ (`fit(rank=K)`, [`gspice.LowRank`](/GSPICE/reference/api/#gspice.LowRank)).

</dd>

<dt id="term-noise-aware-model">noise-aware model</dt>
<dd>

  A model of the noise-free flux, in flux units, whose prediction adds the noise of each
  spectrum from its `ivar` (`fit(noise=True)`).

</dd>

<dt id="term-pattern">pattern</dt>
<dd>

  The pixel sets $G$, $M$ and $T$ of an infill. Spectra of one S/N bin with one pattern
  share one factorization within each chunk of spectra.

</dd>

<dt id="term-precision-matrix">precision matrix</dt>
<dd>

  $P = C^{-1}$, the inverse of the covariance. GSPICE computes the conditional
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

<dt id="term-S-N-bin">S/N bin</dt>
<dd>

  A range of signal-to-noise ratio with its own mean and covariance
  (`fit(snr_bins=...)`). The S/N of a spectrum is the median of
  $f \sqrt{\mathrm{ivar}}$ over its good pixels.

</dd>

<dt id="term-scaled-units">scaled units</dt>
<dd>

  Flux multiplied by $a_s$, the square root of the mean inverse variance of the good
  pixels of the spectrum. The noise of a spectrum with equal `ivar` at every pixel then
  has the variance 1.

</dd>

<dt id="term-threshold">threshold</dt>
<dd>

  For the block scan, the percentile of $\chi^2_A$ of a block over clean spectra of one
  S/N bin, from [`gspice.fit_blocks()`](/GSPICE/reference/api/#gspice.fit_blocks). For z-scores, a limit on $|z|$, such as
  `nsigma_replace`.

</dd>

<dt id="term-training-spectra">training spectra</dt>
<dd>

  The spectra that [`gspice.fit()`](/GSPICE/reference/api/#gspice.fit) estimates the model from.

</dd>

<dt id="term-working-units">working units</dt>
<dd>

  The units of the working vector: scaled units, or flux units for a noise-aware model.

</dd>

<dt id="term-working-vector">working vector</dt>
<dd>

  The vector $x_s$ that the model describes: the interpolated flux of spectrum $s$ in
  working units, centred by the mean of its S/N bin or by a constant centre.

</dd>

<dt id="term-z-score">z-score</dt>
<dd>

  The difference between the flux of a pixel and its prediction, divided by the square
  root of the conditional variance.

</dd>

</dl>
