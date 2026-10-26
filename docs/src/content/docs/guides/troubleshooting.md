---
title: Troubleshooting
description: The common errors and warnings of GSPICE, what causes them, and how to fix them.
---

Something failed. What does the error mean? Find the block that starts like your
message. Words in angle brackets, such as `<npix>`, stand for the values in your message.

### Not positive definite

```text wrap
LinAlgError: <covariance> is not positive definite: dpotrf failed at its leading minor of order <k>
```

Cause: two pixels always vary together, for example pixels duplicated by resampling.

Fix: remove the duplicated pixels before you call `fit`.

### Numerically singular

```text wrap
LinAlgError: <covariance> is numerically singular: rcond = <r> for the matrix scaled to unit diagonal, below 1e-13
```

Cause: the covariance is too close to singular for a reliable prediction. This can
happen with about as many training spectra as pixels.

Fix: use more training spectra, or a [low-rank model](/GSPICE/guides/models/#low-rank).

### A pass above the clipping ceiling

```text wrap
fit: pass <n> has nsigma <t> above <c>, the in-sample ceiling of |z| at N_s = <N_s>
```

Cause: `fit` logs this warning when, with few training spectra per pixel, an outlier in
a spectrum of typical leverage cannot reach the threshold of pass `<n>`. So the pass
flags few pixels, or none.

Fix: use more training spectra, or lower `nsigma`. A pass whose threshold is below the
ceiling still clips. [The clipping ceiling](/GSPICE/concepts/fitting/#the-clipping-ceiling)
gives the ceiling.

`fit` logs its warning on the `gspice` logger, so you see it only if your program shows
log records. `logging.basicConfig()` shows them on the standard error stream.

### A model without blocks

```text wrap
ValueError: blockwise: expected False for a model without blocks, got True; fit_blocks gives a model blocks
```

Cause: `repair` runs the block scan by default, and the model has no block thresholds.

Fix: call [`gspice.fit_blocks()`](/GSPICE/reference/api/#gspice.fit_blocks) first, or
call `repair(..., blockwise=False)`.

### Too few training spectra

```text wrap
ValueError: fit: pass 1 needs at least <n> spectra for a mean-centred model of <npix> pixels, got <N_s>: ...
```

Cause: a default model needs npix + 1 usable spectra. The message counts the spectra
that `fit` removed, such as those with more than `maxbadpix` bad pixels.

Fix: use more spectra, or a [low-rank model](/GSPICE/guides/models/#low-rank). If
`maxbadpix` removed many spectra, raise it.

### Too few threshold spectra

```text wrap
ValueError: fit_blocks: percentile <p> needs at least <m> spectra in each S/N bin, got <n> in bin <b>
```

Cause: too few clean spectra for the percentile of the block thresholds: 1000 at the
default 99.9.

Fix: use more clean spectra, or a lower `percentile`.

### A slow noise-aware model

Cause: a noise-aware model predicts much more slowly than a scaled model. Each spectrum
has its own covariance, so each needs its own factorization ([Cost](/GSPICE/concepts/numerics/#cost)).

Fix: fit with `rank=K` too, or use a scaled model with `snr_bins` if the noise keeps its
shape ([Choose a model](/GSPICE/guides/models/#noise-aware)).

### Too little memory

Cause: a `MemoryError`, or a slow computer, comes from large arrays: see
[Memory](#memory).

Fix: pass the spectra to `infill`, `repair`, `pixelwise` or `calibration` in slices, and
use `covariance=True` only where you need it. `fit` and `fit_blocks` need all their
spectra in one call.

## Memory

The memory of a call grows with these arrays, of 8 bytes per value:

- A model keeps the precision matrix of each S/N bin: npix² values, 46 MB at npix 2400.
  [What the file holds](/GSPICE/guides/save/#what-the-file-holds) says when it is
  computed.
- `fit` holds the working vectors of its training spectra: N_s × npix values.
- With `covariance=True`, `infill` and `repair` keep the covariance of the replaced
  pixels of each pattern of replaced pixels.
- `fit_blocks` holds a factor for each block and S/N bin. A small `stride` makes many
  blocks.

`infill` with one `Generator` on consecutive slices of the spectra gives the same draws
as one call, to rounding. The docstring of each function states its cost.
