# API reference

Every name in `gspice.__all__` except the submodule `gspice.legacy`, which has [its own page](legacy.md).

```{eval-rst}
.. currentmodule:: gspice
```

## Models and fitting

Fit a model of the spectra.

```{eval-rst}
.. autoclass:: Model
   :members:

.. autoclass:: LowRank
   :members:

.. autoclass:: Blocks
   :members:

.. autofunction:: fit

.. autofunction:: fit_blocks
```


## Prediction

Predict pixels from other pixels.

```{eval-rst}
.. autoclass:: Prediction
   :members:

.. autoclass:: Repaired
   :members:

.. autofunction:: pixelwise

.. autofunction:: infill
```


## Repair

Flag and replace bad pixels.

```{eval-rst}
.. autofunction:: repair
```


## Diagnostics

Check that the predicted uncertainties are calibrated.

```{eval-rst}
.. autoclass:: Calibration
   :members:

.. autofunction:: calibration

.. autofunction:: finite_sample_factor

.. autofunction:: noise_audit
```


## Scaling

Scale spectra before fitting, and undo the scaling.

```{eval-rst}
.. autofunction:: scale

.. autofunction:: unscale

.. autofunction:: snr
```


## Simulation

Make synthetic spectra for tests and examples.

```{eval-rst}
.. autoclass:: Simulation
   :members:

.. autofunction:: simulate
```


## Input and output

Write a model to a file, and read it back.

```{eval-rst}
.. autofunction:: save

.. autofunction:: load
```
