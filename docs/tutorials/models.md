# Low-rank and noise-aware models

`fit(rank=K)` fits a {term}`low-rank model`, and `fit(noise=True)` a
{term}`noise-aware model`. {doc}`../background/fitting` describes both and compares
them with the dense model.

[`docs/scripts/spectra.py`](https://github.com/joshspeagle/GSPICE/blob/main/docs/scripts/spectra.py)
makes `spectra.npz`, which {doc}`fitting` describes. The low-rank model uses all its
pixels.

```python
import numpy as np

import gspice

data = np.load("spectra.npz")
flux, ivar = data["flux"], data["ivar"]
clean, clean_ivar = data["clean"], data["clean_ivar"]

n = len(flux)
train = slice(0, int(0.6 * n))
test = slice(int(0.8 * n), n)
```

## Low-rank model

A dense model of all 3741 pixels from 6000 training spectra has a finite-sample
factor of 6.8 for a pixelwise prediction ({doc}`fitting`). A low-rank model has
$K$ components, and needs only $K + 2$ spectra:

```python
lowrank, _ = gspice.fit(flux[train], ivar[train], rank=40)
z = gspice.pixelwise(lowrank, clean[test], clean_ivar[test]).z
print(f"standard deviation {np.std(z):.2f}, |z| > 5: {np.mean(np.abs(z) > 5):.3%}")
```

`lowrank.cov[0]` is a {class}`gspice.LowRank`, which holds the $K$ eigenvectors and
eigenvalues and the diagonal. On the test spectra without artifacts, the z-scores have
the standard deviation 1.02, and 0.062% of them have $|z| > 5$. A dense model of the
same spectra gives 2.64, close to $\sqrt{6.8} = 2.61$, and 5.8%.

```{image} ../_static/figures/tutorials/models-z-light.svg
:alt: Histograms of the held-out z-scores of the dense and the low-rank model on all pixels
:class: only-light
```

```{image} ../_static/figures/tutorials/models-z-dark.svg
:alt: Histograms of the held-out z-scores of the dense and the low-rank model on all pixels
:class: only-dark
```

The low-rank model has no derived finite-sample factor, and rejects `inflate=True`.

## Noise-aware model

A noise-aware model estimates the covariance of the noise-free flux, and adds the noise
of each spectrum, from its `ivar`, at prediction. Each spectrum then has its own
covariance and its own factorization, so the prediction costs of order npix³ flops per
spectrum. This example uses the 1500 pixels of {doc}`fitting` and 2% of the
spectra:

```python
pixels = (data["wave"] > 3800) & (data["wave"] < 5370)
few = slice(test.start, test.start + n // 50)
noisy, _ = gspice.fit(flux[train][:, pixels], ivar[train][:, pixels], noise=True)
rng = np.random.default_rng(1)
result = gspice.calibration(
    noisy, clean[few][:, pixels], clean_ivar[few][:, pixels], rng=rng
)
print(result.ratio(where=(result.n > 0).all(axis=1))[0].round(3))
```

`noisy.noise` is True, and `noisy.cov[0]` is in flux² units. On 200 test spectra
without artifacts, the means of $z^2 / n$ of {doc}`calibration` are 0.86 to 0.92 for
the sizes 1 to 256, with standard errors of 0.10 at $n = 1$ and 0.008 at $n = 256$:
the predicted variances are too large. The estimate of the noise-free covariance
assumes an `ivar` that does not depend on the flux ({doc}`../background/fitting`), and
the `ivar` of this survey comes from the observed counts.
