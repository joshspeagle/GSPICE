# GSPICE

[![CI](https://github.com/joshspeagle/GSPICE/actions/workflows/ci.yml/badge.svg)](https://github.com/joshspeagle/GSPICE/actions/workflows/ci.yml)
[![Docs](https://github.com/joshspeagle/GSPICE/actions/workflows/docs.yml/badge.svg)](https://github.com/joshspeagle/GSPICE/actions/workflows/docs.yml)
[![License](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Fjoshspeagle%2FGSPICE%2Fmain%2Fpyproject.toml&query=%24.project.license&label=license)](https://github.com/joshspeagle/GSPICE/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Fjoshspeagle%2FGSPICE%2Fmain%2Fpyproject.toml&query=%24.project%5B%27requires-python%27%5D&label=python)](https://github.com/joshspeagle/GSPICE/blob/main/pyproject.toml)

Survey spectra contain pixels that you cannot trust, such as cosmic rays, hot columns and pixels that the pipeline masked. GSPICE learns from your own spectra how the pixels vary together, and predicts any pixels from the others, with an uncertainty.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/joshspeagle/GSPICE/main/docs/readme/quickstart-dark.svg">
  <img alt="A simulated spectrum of 200 pixels. Pixels 80 to 109 are masked, and GSPICE predicts them from the other pixels, with bands of ±1σ and ±2σ." src="https://raw.githubusercontent.com/joshspeagle/GSPICE/main/docs/readme/quickstart-light.svg">
</picture>

The figure shows one test spectrum of the example below. Pixels 80 to 109 are masked, and the orange line is their prediction by `infill`.

## Install

```sh
pip install git+https://github.com/joshspeagle/GSPICE
```

[Install](https://joshspeagle.github.io/GSPICE/start/install/) gives the requirements and the other ways to install GSPICE.

## Example

The example simulates spectra, fits a model to them, and predicts a range of pixels that it masks.

```python
import numpy as np

import gspice

rng = np.random.default_rng(2026)
train = gspice.simulate(3000, 200, rng=rng)
test = gspice.simulate(5, 200, rng=rng)
model, _ = gspice.fit(train.flux, train.ivar)

mask = np.zeros(test.flux.shape, dtype=bool)
mask[:, 80:110] = True
result = gspice.infill(model, test.flux, test.ivar, mask=mask)
print(f"{result.replaced.sum()} pixels replaced")
```

[Quickstart](https://joshspeagle.github.io/GSPICE/start/quickstart/) explains each step.

## Documentation

The documentation is at [joshspeagle.github.io/GSPICE](https://joshspeagle.github.io/GSPICE/). Start with [Quickstart](https://joshspeagle.github.io/GSPICE/start/quickstart/), [Install](https://joshspeagle.github.io/GSPICE/start/install/) and the first guide, [Fill known bad pixels](https://joshspeagle.github.io/GSPICE/guides/infill/). The [API reference](https://joshspeagle.github.io/GSPICE/reference/api/) documents every public name, and [`CHANGELOG.md`](https://github.com/joshspeagle/GSPICE/blob/main/CHANGELOG.md) lists the changes since the 2022 package.

## The 2022 package

`gspice.legacy` keeps the 2022 functions, with their signatures and results: see [Legacy interface](https://joshspeagle.github.io/GSPICE/reference/legacy/). The 2022 names at the top level of `gspice`, such as `gspice.gp_interp`, are deprecated. [Migrating from the 2022 interface](https://joshspeagle.github.io/GSPICE/tutorials/migration/) lists their replacements, and `CHANGELOG.md` lists the behaviour changes of `gspice.legacy`.

## Citation

The method is in Finkbeiner, Speagle & Karim, [arXiv:2511.16754](https://arxiv.org/abs/2511.16754). The IDL and Julia implementations, notebooks and notes of 2021 and 2022 are in the upstream repository [dfink/gspice](https://github.com/dfink/gspice/tree/1fbab9f).
