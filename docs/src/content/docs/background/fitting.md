---
title: Fitting
---

[`gspice.fit()`](/GSPICE/reference/api/#gspice.fit) estimates a [`gspice.Model`](/GSPICE/reference/api/#gspice.Model) from [training spectra](/GSPICE/background/glossary/#term-training-spectra). It
returns the model and the final training mask. Five kinds of model share this function:
dense (the default), constant centre, S/N bins, low rank and noise-aware.

## Training spectra

`fit` uses a spectrum unless it has more than `maxbadpix` bad pixels (64 by default),
has the weight 0, or has every pixel masked by a clipping pass. $N_s$ counts the spectra
that a fit uses. A dense, mean-centred model needs $N_s - 1 \ge \mathrm{npix}$, and a
good pixel in at least one spectrum at every pixel. Close to that minimum, the
covariance can still fail the check of its precision matrix ([The model](/GSPICE/background/method/)).

## Outlier clipping

Artifacts in the training spectra widen the covariance. `fit` therefore flags them in
passes, with thresholds that fall from pass to pass (paper Alg. 4). Pass $n$ uses the
threshold `nsigma[n - 1]`, from `nsigma=(20, 8, 6)` by default, and the mask $M_n$,
with $M_1$ the bad pixels:

1. Interpolate the pixels of $M_n$, and scale each spectrum by $a_s$ over its pixels
   outside $M_n$.
2. Fit the covariance, with unit weights.
3. Compute the pixelwise z-score of every pixel of these spectra, interpolated pixels
   included, with `nguard`.
4. Flag $F_n$: each pixel with $|z|$ above the threshold, and its two neighbours.

With `cumulative=True`, the default, $M_{n+1} = M_n \cup F_n$. The final covariance is
fitted on the last of these masks, with the `weights` if they are given. The returned
mask is that final mask, and True at every pixel of a spectrum that `fit` did not use.
`cumulative=False` uses the 2022 rule, $M_{n+1} = \mathrm{bad} \cup F_n$, with the
final mask $\mathrm{bad} \cup F_{N-1} \cup F_N$ for $N$ passes. `nsigma=()` fits
without clipping.

The z-scores of a pass are [in-sample](/GSPICE/background/glossary/#term-in-sample): each spectrum helped to fit the covariance
that predicts it. An outlier therefore widens the covariance that judges it. For a dense,
mean-centred model, the in-sample $|z|$ of an outlier of any amplitude approaches

$$
\sqrt{\frac{(N_s - 1)(N_s - N_k - 1)}{N_s}} ,
\qquad N_k = \max(\mathrm{npix} - 2\,\mathrm{nguard} - 1, 0) ,
$$

in a spectrum of typical leverage. At $N_s = 2500$, npix 2400 and `nguard=20`, this
ceiling is 11.8, so a pass at 20 flags no pixel. With a constant centre, the ceiling is
$\sqrt{N_s - N_k}$. `fit` logs a warning on the `gspice` logger for each pass whose
threshold exceeds the ceiling. No ceiling is derived for a low-rank model.

## Constant centre

`center=c` subtracts a constant $c$ in flux units, such as a continuum level, instead of
the mean: $x_s = a_s (\tilde f_s - c)$. With unit weights, the
covariance is $\sum_s x_s x_s^\mathsf{T} / N_s$, and the model needs $N_s \ge \mathrm{npix}$.

## S/N bins

In scaled units, the noise of every spectrum has a variance near 1, but the variation of
the spectra scales with $a_s^2$, so it grows with the S/N. Spectra of different S/N
therefore have different covariances in scaled units, and one covariance of all of them
is a mixture. `snr_bins` fits one model for each [S/N bin](/GSPICE/background/glossary/#term-S-N-bin).

The S/N of a spectrum is the median of $f \sqrt{\mathrm{ivar}}$ over its good pixels
([`gspice.snr()`](/GSPICE/reference/api/#gspice.snr)). `snr_bins=n` puts the edges between the bins at quantiles of the S/N,
so that each bin holds about the same number of spectra. A sequence gives the interior
edges instead. Each bin clips and fits on its own spectra: bin $b$ of the model equals
the model that `fit` returns for the spectra of bin $b$ alone. At prediction, each
spectrum goes to the bin of its S/N, with the edges in `Model.edges`.

Each bin needs the spectra of a model of its kind, for example $N_s - 1 \ge \mathrm{npix}$
for a dense bin. A small bin also has a low in-sample ceiling, so it flags fewer
outliers.

## Low-rank model

`rank=K` fits a [`gspice.LowRank`](/GSPICE/reference/api/#gspice.LowRank) covariance:

$$
C = V \Lambda V^\mathsf{T} + D ,
$$

with $(V, \Lambda)$ the $K$ leading eigenvectors and eigenvalues of the sample
covariance $S$, and the diagonal $D = \mathrm{diag}(d)$ with
$d_i = S_{ii} - \sum_k \lambda_k V_{ik}^2$. So $C$ and $S$ have the same variance at
every pixel, and the correlations between pixels come from the $K$ components alone.
The clipping passes use the same rank.

The model computes the precision matrix by the Woodbury identity
[@woodbury1950], in npix² $K$ flops, without forming $C$:

$$
P = D^{-1} - D^{-1} V \left(\Lambda^{-1} + V^\mathsf{T} D^{-1} V\right)^{-1}
V^\mathsf{T} D^{-1} .
$$

A low-rank model needs $N_s - 1 \ge K + 1$, a positive $\lambda_K$ and every $d_i > 0$,
but no good pixel at every pixel. It suits a training set with fewer spectra than
pixels.

## Noise-aware model

`noise=True` estimates $\Sigma$, the covariance of the noise-free flux, in flux units,
from the `ivar` of each pixel. It clips with a scaled model first, as above. At
prediction, spectrum $s$ has the covariance

$$
C_s = \Sigma_+ + \mathrm{diag}(\nu_s) ,
$$

with $\nu_{si} = 1/\mathrm{ivar}_{si}$ at the good pixels and $1/a_s^2$ at the bad
pixels. $\Sigma_+$ is $\Sigma$ with its negative eigenvalues set to 0, which biases it
upward. The working vector is $x_s = \tilde f_s - \mu$, in flux units. `rank=K` also
works here.

$\Sigma$ is unbiased element by element under four conditions:

1. The spectra are independent, with a common mean and a common signal covariance.
2. The noise has zero mean and the variance `1 / ivar`.
3. The noise is uncorrelated between pixels and with the signal.
4. The masks, weights and `ivar` do not depend on the flux.

The outlier flags of the final mask break condition 4, and so does an `ivar` computed
from the flux.

A noise-aware model has one bin and no centre. It needs two spectra good at every pixel,
and its clipping passes need the spectra of a scaled model. Each spectrum has its own
covariance, so each needs its own factorization, of order npix³ flops with a dense
$\Sigma_+$, where the other models share one product for all spectra.

## Choosing a model

| Model | Arguments | Spectra needed | Cost at prediction | Finite-sample factor |
|---|---|---|---|---|
| Dense | default | $N_s - 1 \ge \mathrm{npix}$ | one $P$ per bin | derived |
| Constant centre | `center=c` | $N_s \ge \mathrm{npix}$ | one $P$ per bin | 1 |
| S/N bins | `snr_bins=n` | per bin, as for its kind | one $P$ per bin | derived for dense bins |
| Low rank | `rank=K` | $N_s - 1 \ge K + 1$ | one $P$ per bin, npix² $K$ flops | 1 |
| Noise-aware | `noise=True` | two good spectra at each pixel, and those of a scaled model for clipping | one factorization per spectrum | 1 |

Positive `weights` apply to the final fit only, and the clipping passes use unit
weights. A weighted model has no finite-sample factor either.
[`gspice.save()`](/GSPICE/reference/api/#gspice.save) writes a model to a `.npz` file, and [`gspice.load()`](/GSPICE/reference/api/#gspice.load) reads it
back.
