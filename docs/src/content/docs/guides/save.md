---
title: Save and load
description: Write a model to one .npz file, and read it back.
---

How do I keep a model? [`gspice.save()`](/GSPICE/reference/api/#gspice.save) writes a
[`gspice.Model`](/GSPICE/reference/api/#gspice.Model) to one `.npz` file, and
[`gspice.load()`](/GSPICE/reference/api/#gspice.load) reads it back.

## Save a model

Fitting a model takes time, so save it once and load it where you predict. This model
also has the block thresholds of [Find and repair artifacts](/GSPICE/guides/repair/):

```python
import numpy as np

import gspice

rng = np.random.default_rng(7)
train = gspice.simulate(1500, 100, rng=rng)
clean = gspice.simulate(1000, 100, rng=rng)
model, _ = gspice.fit(train.flux, train.ivar)
model = gspice.fit_blocks(model, clean.flux, clean.ivar, width=50, stride=25)

gspice.save(model, "model.npz")
```

`save` replaces an existing file, and adds no suffix to the path.

## Load it

```python
loaded = gspice.load("model.npz")
test = gspice.simulate(200, 100, rng=rng, artifacts=0.3)
before = gspice.repair(model, test.flux, test.ivar)
after = gspice.repair(loaded, test.flux, test.ivar)
print(np.array_equal(before.replaced, after.replaced))
print(np.allclose(before.flux, after.flux, rtol=1e-10, atol=0))
```

```text output
True
True
```

The file holds every field of the model exactly, the blocks included. The loaded model
replaces the same pixels, and predicts the same flux to rounding, because it computes its
precision matrix again ([What the file holds](#what-the-file-holds)).

## What the file holds

The file is an ordinary `.npz` archive, so `numpy.load` also reads it:

```python
print(np.load("model.npz").files)
```

```text output
['schema', 'gspice_version', 'mean', 'n_train', 'edges', 'center', 'weighted', 'cov0', 'block_starts', 'block_width', 'block_nguard', 'block_thresholds']
```

The docstring of `save` describes each array. The file holds no precision matrices. A
model computes the precision matrix of a bin at its first use, in npix³ flops, and keeps
it. So the first prediction of a loaded model takes longer than the next ones.
