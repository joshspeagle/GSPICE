# Saving and loading

{func}`gspice.save` writes a {class}`gspice.Model` to one `.npz` file, and
{func}`gspice.load` reads it back. This example simulates its spectra with
{func}`gspice.simulate`.

```python
import numpy as np

import gspice

rng = np.random.default_rng(7)
train = gspice.simulate(1500, 100, rng=rng, artifacts=0.3)
clean = gspice.simulate(1000, 100, rng=rng)
model, mask = gspice.fit(train.flux, train.ivar)
model = gspice.fit_blocks(model, clean.flux, clean.ivar, width=50, stride=25)

gspice.save(model, "model.npz")
loaded = gspice.load("model.npz")
```

The file holds every field of the model, the blocks of {func}`gspice.fit_blocks`
included. The docstring of `save` lists its arrays, which `numpy.load` also reads:

```python
print(np.load("model.npz").files)
```

The file holds no precision matrices. A model computes the precision matrix of a bin at
its first use, in npix³ flops, and keeps it, so the first prediction of a loaded model
takes longer than the next ones. The predictions are the same:

```python
test = gspice.simulate(200, 100, rng=rng, artifacts=0.3)
before = gspice.repair(model, test.flux, test.ivar)
after = gspice.repair(loaded, test.flux, test.ivar)
print(np.array_equal(before.flux, after.flux))
```

`save` replaces an existing file, and adds no suffix to the path.
