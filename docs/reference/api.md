---
title: API reference
tableOfContents:
  maxHeadingLevel: 4
editUrl: https://github.com/joshspeagle/GSPICE/edit/main/docs/reference/api.md
---

Every name in `gspice.__all__` except the submodule `gspice.legacy`, which has [its own page](/GSPICE/reference/legacy/).

## Models and fitting

Fit a model of the spectra.

::: gspice.Model

::: gspice.LowRank

::: gspice.Blocks

::: gspice.fit

::: gspice.fit_blocks

## Prediction

Predict pixels from other pixels.

::: gspice.Prediction

::: gspice.Repaired

::: gspice.pixelwise

::: gspice.infill

## Repair

Detect artifacts, and replace the flagged pixels and the bad pixels.

::: gspice.repair

## Diagnostics

Check that the predicted uncertainties are calibrated.

::: gspice.Calibration

::: gspice.calibration

::: gspice.finite_sample_factor

::: gspice.noise_audit

## Scaling

Scale spectra before fitting, and undo the scaling.

::: gspice.scale

::: gspice.unscale

::: gspice.snr

## Simulation

Make synthetic spectra for tests and examples.

::: gspice.Simulation

::: gspice.simulate

## Input and output

Write a model to a file, and read it back.

::: gspice.save

::: gspice.load
