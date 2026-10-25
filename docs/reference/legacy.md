---
title: Legacy interface
editUrl: https://github.com/joshspeagle/GSPICE/edit/main/docs/reference/legacy.md
---

The 2022 functions of `gspice.legacy`, with the arithmetic of commit `1fbab9f`, kept for migration. The [changelog](https://github.com/joshspeagle/GSPICE/blob/main/CHANGELOG.md) lists their replacements in the new API. The 2022 modules stay too, as `gspice.legacy.specUtils`, `gspice.legacy.matrixUtils` and `gspice.legacy.djs_maskinterp`, so each function below can also be imported from its 2022 module.

## Scaling and covariance

::: gspice.legacy.standard_scale

::: gspice.legacy.covar

::: gspice.legacy.covar_iter_mask

## Prediction

::: gspice.legacy.get_chimask

::: gspice.legacy.gp_interp

::: gspice.legacy.gaussian_estimate

## Matrix utilities

::: gspice.legacy.cholesky_inv

::: gspice.legacy.submatrix_inv

::: gspice.legacy.submatrix_inv_mult

## Interpolation

::: gspice.legacy.maskinterp1

::: gspice.legacy.maskinterp
