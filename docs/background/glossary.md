# Glossary

```{glossary}
bad pixel
  A pixel with a nonzero `mask` or `ivar == 0`. GSPICE never reads `flux` or `ivar`
  there, and never conditions on it, except in the pixelwise prediction, which uses its
  interpolated value.

block
  `width` consecutive pixels of the block scan. {class}`gspice.Blocks` holds the first
  pixel of each block and their thresholds.

block scan
  The comparison of each block of a spectrum with its prediction from the pixels beyond
  its guard window, by the statistic $\chi^2_A$ of {doc}`repair`.

conditional distribution
  The Gaussian distribution of the pixels $G$ of a spectrum, given the values of its
  conditioning pixels $O$. Its mean is the prediction, and its covariance the
  uncertainty of the prediction.

conditioning pixels
  The set $O$: the pixels whose values a prediction uses. All pixels outside $G$.

constant centre
  A constant $c$, in flux units, that a model subtracts from the flux instead of the
  mean of the training spectra (`fit(center=c)`).

dilation
  $\mathrm{dil}(F)$: the set $F$ of flagged pixels plus every pixel within a given
  number of pixels of a pixel of $F$ along the spectrum.

draw
  A random sample of the replaced pixels from their conditional distribution, in
  `Repaired.draws`.

finite-sample factor
  $f_s$: the factor by which the error of an estimated mean and covariance widens the
  residuals of a held-out spectrum, $E[Z^2] / N_*$ (see {doc}`prediction`).

guard window
  The pixels within `nguard` of a predicted pixel or block. A prediction does not
  condition on them, and reports only the predicted pixels among them.

held-out spectrum
  A spectrum that the model was not fitted to.

ignored pixel
  A pixel in $G$ that is not replaced: it is not conditioned on, and keeps its input
  value.

in-sample
  Computed on the training spectra of the model, as the z-scores of the clipping passes
  of `fit` are.

low-rank model
  A model with the covariance $V \Lambda V^\mathsf{T} + D$ of $K$ components and a
  diagonal $D$ (`fit(rank=K)`, {class}`gspice.LowRank`).

noise-aware model
  A model of the noise-free flux, in flux units, whose prediction adds the noise of each
  spectrum from its `ivar` (`fit(noise=True)`).

pattern
  The pixel sets $G$, $M$ and $T$ of an infill. Spectra of one S/N bin with one pattern
  share one factorization within each chunk of spectra.

precision matrix
  $P = C^{-1}$, the inverse of the covariance. GSPICE computes the conditional
  distribution from its blocks $P_{GG}$ and $P_{GO}$.

prior
  The distribution $\mathcal{N}(0, C)$ of the working vector, without conditioning
  pixels. A prediction whose $G$ holds all pixels returns it.

replaced pixel
  A pixel of the set $M$, whose value an infill replaces by its conditional mean.

S/N bin
  A range of signal-to-noise ratio with its own mean and covariance
  (`fit(snr_bins=...)`). The S/N of a spectrum is the median of
  $f \sqrt{\mathrm{ivar}}$ over its good pixels.

scaled units
  Flux multiplied by $a_s$, the square root of the mean inverse variance of the good
  pixels of the spectrum. The noise of a spectrum with equal `ivar` at every pixel then
  has the variance 1.

threshold
  For the block scan, the percentile of $\chi^2_A$ of a block over clean spectra of one
  S/N bin, from {func}`gspice.fit_blocks`. For z-scores, a limit on $|z|$, such as
  `nsigma_replace`.

training spectra
  The spectra that {func}`gspice.fit` estimates the model from.

working units
  The units of the working vector: scaled units, or flux units for a noise-aware model.

working vector
  The vector $x_s$ that the model describes: the interpolated flux of spectrum $s$ in
  working units, centred by the mean of its S/N bin or by a constant centre.

z-score
  The difference between the flux of a pixel and its prediction, divided by the square
  root of the conditional variance.
```
