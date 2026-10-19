# The model

GSPICE treats each spectrum as one draw from a multivariate Gaussian distribution over
its pixels. It estimates the mean and the covariance of that distribution from
{term}`training spectra`, and predicts any set of pixels from the other pixels with the
conditional distribution {cite:p}`finkbeiner2025gspice`. The covariance holds what the
method knows about the spectra and the instrument.

## Spectra and bad pixels

The input is a set of nspec spectra on one grid of npix pixels. Pixel $i$ has the same
wavelength in every spectrum, for example after a shift to the rest frame. The
covariance describes pixels, so the spectra need a similar line-spread function at each
pixel.

Every function that takes spectra takes three arrays of shape (nspec, npix): `flux`, `ivar`, the inverse
variance of the noise in flux⁻², and the optional `mask`. A {term}`bad pixel` has a
nonzero `mask` or `ivar == 0`. The code never reads `flux` or `ivar` at a bad pixel, so
NaN or inf there is allowed.

## Working units

The model describes the {term}`working vector` $x_s$ of spectrum $s$, not its flux.
The model computes it in three steps:

1. Interpolate the flux linearly over the bad pixels, and hold it constant beyond the
   first and the last good pixel. This gives $\tilde f_s$.
2. Multiply by $a_s$, the square root of the mean of `ivar` over the good pixels of the
   spectrum, in flux⁻¹. This gives {term}`scaled units`.
3. Subtract the mean $m_b$ of the training spectra in the {term}`S/N bin` $b$ of the
   spectrum. There is one bin by default.

{func}`gspice.scale` does steps 1 and 2. In step 3 it subtracts its `refmean` argument,
by default the mean of the spectra that it is given.

$$
x_s = a_s \tilde f_s - m_b .
$$

In scaled units, the noise at pixel $i$ has the variance $a_s^2 / \mathrm{ivar}_{si}$.
That variance is 1 where `ivar` is equal at every pixel of the spectrum, whatever its
S/N. A conditional variance divided by $a_s^2$ is a variance in flux units.

The public functions take flux and return flux, flux² or unit-free values. The working
units appear only in {func}`gspice.scale`, {func}`gspice.unscale` and the fields of
{class}`gspice.Model`. Two kinds of model use other working vectors: a model with a
{term}`constant centre` $c$ uses $x_s = a_s (\tilde f_s - c)$, and a
{term}`noise-aware model` uses $x_s = \tilde f_s - \mu$ in flux units. {doc}`fitting`
describes both.

## Mean and covariance

A spectrum in S/N bin $b$ has $x \sim \mathcal{N}(0, C_b)$. {func}`gspice.fit` estimates
$m_b$ and $C_b$ from the $N_s$ training spectra of the bin. With $y_s = a_s \tilde f_s$,

$$
m_b = \frac{1}{N_s} \sum_s y_s, \qquad
C_b = \frac{1}{N_s - 1} \sum_s (y_s - m_b)(y_s - m_b)^\mathsf{T} .
$$

The training spectra carry their noise, so $C_b$ holds the noise as well as the
variation of the spectra. Every predicted variance therefore includes the noise of the
predicted pixel. A {class}`gspice.Model` holds `mean` ($m_b$), `cov` ($C_b$) and
`n_train` ($N_s$) for each bin.

The spectra need not be Gaussian. A population of stars is far from a Gaussian
distribution, but the conditional estimates of a Gaussian fitted to it still predict
its pixels well {cite:p}`finkbeiner2025gspice`.

## The conditional distribution

Split the pixels into two sets. $G$ holds the pixels to predict or to leave out, and $O$
holds the others, the {term}`conditioning pixels`. The conditional distribution of $x_G$
given $x_O$ is Gaussian {cite:p}`rasmussen2006gpml`:

$$
x_G \mid x_O \sim \mathcal{N}\!\left(C_{GO} C_{OO}^{-1} x_O,\;
C_{GG} - C_{GO} C_{OO}^{-1} C_{OG}\right) .
$$

The code writes it with the {term}`precision matrix` $P = C^{-1}$. The density of $x$ is
proportional to $\exp(-\tfrac12 x^\mathsf{T} P x)$. As a function of $x_G$, it is
proportional to

$$
\exp\!\left(-\tfrac12 x_G^\mathsf{T} P_{GG}\, x_G - x_G^\mathsf{T} P_{GO}\, x_O\right) ,
$$

so

$$
x_G \mid x_O \sim \mathcal{N}\!\left(-P_{GG}^{-1} P_{GO}\, x_O,\; P_{GG}^{-1}\right) .
$$

The two forms are equal in exact arithmetic. The figure shows the blocks of $P$ that the
second form uses. Where $G$ holds all pixels, the result is the {term}`prior`
$\mathcal{N}(0, C)$.

```{raw} html
:file: ../_static/figures/precision.svg
```

A function can report only some pixels of $G$. The others are marginalized: their
values do not enter, and the conditional covariance of the reported pixels includes the
uncertainty of the others. {doc}`prediction` uses this for guard windows and ignored
pixels.

## Why the precision form

The precision form has three advantages over the covariance form:

- **Cost.** The model computes $P$ once for each S/N bin, in npix³ flops, and keeps it.
  Each pixel set then needs a Cholesky factor of $P_{GG}$, in $|G|^3/3$ flops. The
  covariance form factors $C_{OO}$ instead, in $|O|^3/3$ flops. $G$ is usually small, a
  guard window or a few bad pixels, so $|O|$ is close to npix.
- **No values from G.** In {func}`gspice.infill`, the mean
  $-P_{GG}^{-1} (P_{GO} x_O)$ reads no value in $G$, so an outlier in $G$ cannot change
  it.
- **Accuracy.** A Cholesky solve with a matrix of condition number $\kappa$ has a
  relative error of order $\kappa \epsilon$, with $\epsilon = 2.2 \times 10^{-16}$
  {cite:p}`higham2002accuracy`. The error of the precision form stayed within about
  $\kappa \epsilon$ in every measured case. The 2022 code instead updates a precomputed
  inverse (paper Appendix A), and the error of that update has no such bound where the
  guard window is coupled to the conditioning pixels. The
  [design document](https://github.com/joshspeagle/GSPICE/blob/main/docs/design.md)
  records the measurements, and the {doc}`legacy interface <../reference/legacy>` keeps
  the 2022 arithmetic.

A covariance fails a check, with `numpy.linalg.LinAlgError`, where its precision matrix
is not reliable. For a dense covariance, the check is a reciprocal condition number below
1e-13, after scaling the covariance to unit diagonal. A {class}`gspice.LowRank` covariance
has its own bound.

Four computations use another route:

- A low-rank model computes $P$ by the Woodbury identity ({doc}`fitting`).
- A dense noise-aware model has its own covariance for each spectrum, and
  {func}`gspice.infill` solves with its Cholesky factor.
- A low-rank noise-aware model forms no $P$. Its infill solves a system of size $K$ for
  each spectrum.
- For a dense model without noise, the infills of the mean alone inside
  {func}`gspice.repair` and {func}`gspice.fit_blocks` factor $C_{OO}$ where $G$ holds
  more than half the pixels, because that factor is then the smaller one.
