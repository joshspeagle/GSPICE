# GSPICE

```{image} _static/logo-light.svg
:alt: GSPICE
:width: 420px
:align: center
:class: only-light
```

```{image} _static/logo-dark.svg
:alt: GSPICE
:width: 420px
:align: center
:class: only-dark
```

GSPICE models spectra on a common wavelength grid as a multivariate Gaussian with an
empirical covariance, and predicts any set of pixels from the others
{cite:p}`finkbeiner2025gspice`. The prediction can replace bad pixels, and the distance
of a pixel from its prediction can flag artifacts. The method needs no model of the
spectra and no knowledge of the instrument: the covariance of the training spectra
carries both.

The logo shows the two uses. A masked block cuts the G, and its stroke comes back as the
conditional mean with its uncertainty. The top band of "SPICE" was shifted, and the whole
band is flagged and restored.

```{toctree}
installation
references
```
