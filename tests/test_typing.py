import os
from importlib import resources
from pathlib import Path
from typing import assert_type

import numpy as np
from numpy.typing import NDArray

import gspice
from gspice import (
    Blocks,
    Calibration,
    Model,
    Prediction,
    Repaired,
    Simulation,
    calibration,
    finite_sample_factor,
    fit,
    fit_blocks,
    infill,
    load,
    noise_audit,
    pixelwise,
    save,
    scale,
    simulate,
    snr,
    unscale,
)

# The calls pass the widest inputs that the docstrings allow: float32 and integer
# arrays, lists, a numpy scalar for each scalar, bytes and Path. The test runs them
# too, so that none of these calls fails.


def progress(stage: str, done: int, total: int) -> int:
    # The functions ignore the value, as that of a progress bar's update.
    return done


def test_the_documented_calls_type_check_and_run(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    flux = rng.normal(1.0, 0.1, (6, 4)).astype(np.float32)
    ivar = np.full((6, 4), 100, np.int16)
    mask = np.zeros((6, 4), np.uint8)
    # As an argument, a numpy call gets the static type NDArray[Any]; a local keeps
    # the dtype.
    eye = np.eye(4, dtype=np.float32)
    refscale = np.ones(6, np.float32)
    starts = np.array([0, 2], np.uint8)
    thresholds = np.ones((2, 2), np.float32)
    blocks = Blocks(starts, np.int64(2), np.int64(1), thresholds)
    model = Model(
        [[0.0] * 4] * 2,
        [eye, 2 * np.eye(4)],
        [np.int32(5), 7],
        edges=[np.float32(50.0)],
        center=np.float32(1.0),
        weighted=np.True_,
        blocks=blocks,
    )
    fitted = fit(
        flux,
        ivar,
        mask,
        nsigma=[np.float32(20.0), 8],
        maxbadpix=np.int64(1),
        nguard=np.int64(1),
        cumulative=np.False_,
        center=np.int64(1),
        weights=[1, 2, 1, 2, 1, 2],
        progress=progress,
    )
    assert_type(fitted, tuple[Model, NDArray[np.bool_]])
    save(model, tmp_path / "model.npz")
    assert_type(load(os.fsencode(tmp_path / "model.npz")), Model)
    scaled = scale(flux, ivar, mask, center=np.float32(1.0), refmean=[0.0] * 4)
    assert_type(
        scaled, tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
    )
    flux_var = unscale(
        [[0.0] * 4] * 6, refscale, [0.0] * 4, center=np.int64(1), var=[1.0] * 4
    )
    assert_type(flux_var, tuple[NDArray[np.float64], NDArray[np.float64] | None])
    assert_type(snr(flux, ivar, mask), NDArray[np.float64])
    # inflate needs a mean-centred, unweighted model.
    centred = Model(np.zeros((1, 4)), [eye], [np.int64(50)])
    prediction = pixelwise(
        centred,
        flux,
        ivar,
        mask,
        nguard=np.int64(1),
        inflate=np.True_,
        progress=progress,
    )
    assert_type(prediction, Prediction)
    repaired = infill(
        centred,
        flux,
        ivar,
        mask,
        replace=[False, True, False, False],
        ignore=np.array([True, False, False, False]),
        inflate=np.True_,
        covariance=np.True_,
        rng=rng,
        progress=progress,
    )
    assert_type(repaired, Repaired)
    assert_type(
        repaired.covariance(np.int64(0)), tuple[NDArray[np.int64], NDArray[np.float64]]
    )
    # Two spectra suffice at the percentile 50.
    blocked = fit_blocks(
        centred,
        flux,
        ivar,
        mask,
        width=np.int64(2),
        stride=np.int64(1),
        nguard=np.int64(1),
        percentile=np.float32(50.0),
        progress=progress,
    )
    assert_type(blocked, Model)
    assert_type(blocked.blocks, Blocks | None)
    calibrated = calibration(
        centred,
        flux,
        ivar,
        mask,
        sizes=[np.int64(1), 2],
        layout="scattered",
        rng=rng,
        progress=progress,
    )
    assert_type(calibrated, Calibration)
    columns = tuple[NDArray[np.float64], NDArray[np.float64]]
    assert_type(calibrated.ratio(where=[True, False] * 3), columns)
    assert_type(
        calibrated.tail(np.float32(0.9), where=np.ones(6, bool)), NDArray[np.float64]
    )
    assert_type(calibrated.qq(np.int64(1), where=[True] * 6), columns)
    built = Calibration(z2=[[1.0, 4.0]] * 6, n=np.int32(1), fs=[[1], [2]] * 3)
    assert_type(built, Calibration)
    assert_type(finite_sample_factor(np.int64(200), 40), float)
    factor = finite_sample_factor(np.int64(200), [40, 41], np.ones(2, np.uint8))
    assert_type(factor, float | NDArray[np.float64])
    rho = noise_audit(model, flux, ivar, mask, nguard=np.int64(1), progress=progress)
    assert_type(rho, NDArray[np.float64])
    simulation = simulate(
        np.int64(5), np.int64(30), rng=rng, artifacts=np.float32(0.2), gaussian=np.True_
    )
    assert_type(simulation, Simulation)


def test_the_calls_cover_every_public_name() -> None:
    # ruff rejects an unused import, so each name imported above has a call or a check.
    assert set(gspice.__all__) - {"legacy"} <= globals().keys()


def test_the_package_ships_the_py_typed_marker() -> None:
    assert resources.files("gspice").joinpath("py.typed").is_file()
