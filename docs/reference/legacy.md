---
title: Legacy interface
editUrl: https://github.com/joshspeagle/GSPICE/edit/main/docs/reference/legacy.md
---

`gspice.legacy` preserves the 2022 functions with the arithmetic of commit `1fbab9f` so existing code continues to run during migration. [Migrate from the 2022 package](/GSPICE/legacy/migrate/) gives their replacements in the new API. The 2022 modules also remain available as `gspice.legacy.specUtils`, `gspice.legacy.matrixUtils` and `gspice.legacy.djs_maskinterp`, so each function below can be imported from its 2022 module path.

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
