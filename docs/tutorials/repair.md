# Finding and repairing artifacts

{func}`gspice.repair` flags the artifacts of each spectrum, with pixelwise z-scores
and a block scan, and replaces them. {doc}`../background/repair` explains the
procedure.

The tutorial uses the survey and the model of {doc}`fitting`.
[`docs/scripts/spectra.py`](https://github.com/joshspeagle/GSPICE/blob/main/docs/scripts/spectra.py)
makes `spectra.npz`.

```python
import numpy as np

import gspice

data = np.load("spectra.npz")
pixels = (data["wave"] > 3800) & (data["wave"] < 5370)
flux = data["flux"][:, pixels]
ivar = data["ivar"][:, pixels]
clean = data["clean"][:, pixels]
clean_ivar = data["clean_ivar"][:, pixels]
artifacts = (data["cosmic"] | data["column"] | data["shift"])[:, pixels]
shift = data["shift"][:, pixels]

n = len(flux)
train = slice(0, int(0.6 * n))
blocks = slice(int(0.6 * n), int(0.8 * n))
test = slice(int(0.8 * n), n)
model, mask = gspice.fit(flux[train], ivar[train])
```

## Block thresholds

The block scan compares each block of 150 pixels with its prediction from the pixels
beyond a guard window. {func}`gspice.fit_blocks` sets the threshold of each block from
spectra without artifacts that the model was not fitted to:

```python
model = gspice.fit_blocks(model, clean[blocks], clean_ivar[blocks])
print(model.blocks.starts, model.blocks.thresholds.shape)
```

`model.blocks` is a {class}`gspice.Blocks`: here 19 blocks, 75 pixels apart, with
one threshold for each block in each S/N bin. The thresholds are the 99.9th
percentile of the block statistic over these 2000 spectra.

## Repair

```python
result = gspice.repair(model, flux[test], ivar[test])
```

`result` is a {class}`gspice.Repaired`, as from {func}`gspice.infill`, with the
flagged blocks of each spectrum in `result.blocks`.

```{image} ../_static/figures/tutorials/repair-spectrum-light.svg
:alt: A test spectrum with a shifted block, before and after repair, with its flagged blocks
:class: only-light
```

```{image} ../_static/figures/tutorials/repair-spectrum-dark.svg
:alt: A test spectrum with a shifted block, before and after repair, with its flagged blocks
:class: only-dark
```

The figure shows a test spectrum with a shifted block, whose lines are recorded up to
3 pixels off. The lower panel shows the difference from the same spectrum without the
artifact. The block scan flagged the grey blocks, and `repair` replaced
their pixels.

## Check against the truth

The simulation knows the artifacts, so the result can be checked:

```python
amplitude = np.abs(flux - clean) * np.sqrt(clean_ivar)
strong = (artifacts & (amplitude > 5))[test]
other = ~artifacts[test]
found = result.replaced | result.ignored
print(f"replaced: {result.replaced[strong].mean():.0%} of strong artifact pixels")
print(f"replaced or ignored: {found[strong].mean():.0%}")
print(f"replaced: {result.replaced[other].mean():.2%} of the other pixels")

has_shift = shift[test].any(axis=1)
flagged = result.blocks.any(axis=1)
print(flagged[has_shift].sum(), "of", has_shift.sum(), "shifted spectra flagged")
```

On the 2000 test spectra, `repair` replaced 78% of the artifact pixels above 5σ, and
replaced or ignored 96% of them. It replaced 87% of such pixels in cosmic rays, 73% in
shifted blocks and 6% in hot columns, which the model partly describes
({doc}`fitting`). It replaced 0.06% of the pixels without an artifact. The block scan
flagged a block in 7 of the 45 test spectra with a shifted block.

```{image} ../_static/figures/tutorials/repair-recall-light.svg
:alt: The fraction of artifact pixels that repair replaced, against the amplitude of the artifact
:class: only-light
```

```{image} ../_static/figures/tutorials/repair-recall-dark.svg
:alt: The fraction of artifact pixels that repair replaced, against the amplitude of the artifact
:class: only-dark
```

The figure shows the fraction of artifact pixels that `repair` replaced, and replaced
or ignored, against their amplitude. Most pixels above 10σ are replaced, and most above
5σ are replaced or ignored, as the defaults `nsigma_replace=10` and `nsigma_ignore=5`
lead one to expect.

Spectra without artifacts show the cost of a false flag:

```python
quiet = gspice.repair(model, clean[test], clean_ivar[test])
print(f"{quiet.replaced.any(axis=1).mean():.1%} of the spectra have replaced pixels")
```

1.8% of the test spectra without their artifacts get a replaced pixel.
