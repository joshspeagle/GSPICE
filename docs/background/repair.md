# Detection and repair

{func}`gspice.repair` flags the bad pixels of each spectrum and replaces them. It
extends the pixelwise repair of the paper's Secs. 2.6 and 3.3
{cite:p}`finkbeiner2025gspice` with a scan over blocks. It uses two detectors: the
pixelwise z-score of {doc}`prediction` for narrow defects, and a {term}`block scan` for
broad ones.

## Why a block scan

A defect that covers many pixels, such as a section of a spectrum with a wrong
wavelength solution, corrupts many conditioning pixels of each pixelwise prediction. The
pixelwise z-scores can then miss it. The block scan compares a whole block of pixels with
its prediction from the pixels beyond a guard window.

## The block statistic

A {term}`block` holds `width` consecutive pixels. The blocks start at 0, `stride`,
2 `stride`, ... up to npix − `width`, and one more block ends at the last pixel if the
last start is smaller than npix − `width`. The defaults, `width=150`, `stride=75` and
`nguard=20`, give 31 blocks at npix 2400.

```{raw} html
:file: ../_static/figures/blocks.svg
```

For block $A$, $G$ holds the pixels within `nguard` of $A$, clipped at the ends, and $O$
holds the other pixels. The statistic is

$$
\chi^2_A = \frac{1}{|A|}\, r_A^\mathsf{T} C_{AA}^{-1} r_A , \qquad
r_A = x_A - \mu_A , \qquad C_{AA} = \big(P_{GG}^{-1}\big)_{AA} ,
$$

with $\mu_A$ and $C_{AA}$ the conditional mean and covariance of $x_A$ given $x_O$. The
guard pixels are marginalized.

## Block thresholds

{func}`gspice.fit_blocks` sets a {term}`threshold` for each block and S/N bin: the
percentile `percentile` (99.9 by default) of $\chi^2_A$ over the $m$ spectra of the bin.
Give it spectra without artifacts that the model was not fitted to. It replaces their bad
pixels by the conditional mean first, and flags no pixel by z-score. A percentile of the
data needs no assumption about the distribution of $\chi^2_A$.

Each bin needs $m (100 - p) \ge 100$ spectra for the percentile $p$: 1000 spectra at
99.9. A clean block then exceeds its threshold with a probability of 0.20% at
$m = 1000$, and 0.12% at $m = 5000$. The docstring of `fit_blocks` gives the formula.
`fit_blocks` returns a new model whose `blocks` field, a {class}`gspice.Blocks`, holds
the blocks and the thresholds.

## The procedure

`repair` uses two thresholds of $|z|$. A pixel above `nsigma_replace` (10 by default) is
replaced. A pixel above `nsigma_ignore` (5 by default) and not replaced is ignored: it is
neither replaced nor used as a conditioning pixel {cite:p}`finkbeiner2025gspice`.
$\mathrm{dil}(F)$ is the set $F$ plus every pixel within `dilate` pixels of a pixel of
$F$ along the spectrum. A "conditional mean" is that of {func}`gspice.infill`.

```{raw} html
:file: ../_static/figures/repair.svg
```

The five steps run in this order:

1. **Pixelwise pass.** Compute $z_1$, the pixelwise z-scores of $x$ at every pixel,
   interpolated bad pixels included. Then
   $M_1 = \mathrm{dil}(|z_1| > \mathtt{nsigma\_replace}) \cup \mathrm{bad}$ and
   $I_1 = \mathrm{dil}(|z_1| > \mathtt{nsigma\_ignore}) \setminus M_1$.
2. **Block pass**, if `blockwise=True`. Replace $x$ at $M_1 \cup I_1$ by its conditional
   mean, which gives $x_b$. Then repeat:
   1. For each block $A$ not yet flagged, compute the ratio of $\chi^2_A(x_b)$ to its
      threshold in the S/N bin of the spectrum.
   2. If the largest ratio exceeds 1, flag that block, and add its pixels to $M_b$.
   3. Recompute $x_b$, with $x$ at $M_1 \cup I_1 \cup M_b$ replaced by its conditional
      mean.

   The pass stops when no ratio exceeds 1. A block is flagged at most once, so there are
   at most nblock rounds. The scan runs on $x_b$, so a narrow defect that step 1 found
   does not raise the $\chi^2_A$ of its block.
3. **Second pixelwise pass.** Replace $x$ at $M_1 \cup M_b$ by its conditional mean
   given the pixels outside $M_1 \cup M_b \cup I_1$, which gives $x_1$. Compute $z_2$,
   the pixelwise z-scores of $x_1$. Then
   $M = M_1 \cup M_b \cup \mathrm{dil}(|z_2| > \mathtt{nsigma\_replace})$ and
   $I = \mathrm{dil}(|z_2| > \mathtt{nsigma\_ignore}) \setminus M$. The pixels of $M_1$
   and $M_b$ hold their conditional mean in $x_1$, so they do not corrupt the
   conditioning pixels of $z_2$.
4. **Exclusion**, if `runlength` is not None. Each run of at least `runlength`
   consecutive pixels of the counting set $K$ adds every pixel within `halfwidth` of the
   run to $I$, except the pixels of $M$. $K$ is $M \cup I$ with `count_on="all"`. With
   `count_on="flagged"`, $K$ holds the pixels outside the bad set with
   $|z_1| > \mathtt{nsigma\_replace}$ or $|z_2| > \mathtt{nsigma\_ignore}$, and $M_b$. A
   spectrum keeps its earlier $I$ if fewer than $\lfloor \mathtt{minfrac} \cdot
   \mathrm{npix} \rfloor$ pixels outside $M$ and $I$ would remain.
5. **Final infill.** The result is that of {func}`gspice.infill` with `replace=M` and
   `ignore=I`, and the options `inflate`, `covariance` and `rng`.

## The result

`repair` returns a {class}`gspice.Repaired`, as `infill` does. `replaced` is $M$, which
always holds the bad pixels, and `ignored` is $I$. `blocks` holds the flagged blocks of
each spectrum, or None with `blockwise=False`. The pixels outside $M$ keep their input
values, bit for bit.

$M$ and $I$ depend on the data, so the finite-sample factor does not hold for
`Repaired.chi2` of `repair`. `repair` needs a model with blocks, unless
`blockwise=False`. To replace with another model the pixels that `repair` found, pass
`replace=r.replaced` and `ignore=r.ignored` to `infill`.
