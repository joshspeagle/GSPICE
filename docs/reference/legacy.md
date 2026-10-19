# Legacy interface

The 2022 functions of `gspice.legacy`, with the arithmetic of commit `1fbab9f`, kept for migration. The [changelog](https://github.com/joshspeagle/GSPICE/blob/main/CHANGELOG.md) lists their replacements in the new API. The 2022 modules stay too, as `gspice.legacy.specUtils`, `gspice.legacy.matrixUtils` and `gspice.legacy.djs_maskinterp`, so each function below can also be imported from its 2022 module.

```{eval-rst}
.. currentmodule:: gspice.legacy
```

## Scaling and covariance

```{eval-rst}
.. autofunction:: standard_scale

.. autofunction:: covar

.. autofunction:: covar_iter_mask
```


## Prediction

```{eval-rst}
.. autofunction:: get_chimask

.. autofunction:: gp_interp

.. autofunction:: gaussian_estimate
```


## Matrix utilities

```{eval-rst}
.. autofunction:: cholesky_inv

.. autofunction:: submatrix_inv

.. autofunction:: submatrix_inv_mult
```


## Interpolation

```{eval-rst}
.. autofunction:: maskinterp1

.. autofunction:: maskinterp
```
