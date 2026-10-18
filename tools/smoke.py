"""Smoke test of an installed gspice: fit a tiny model and predict with it."""

import numpy as np

import gspice

assert "site-packages" in gspice.__file__, gspice.__file__
assert gspice.__version__
sim = gspice.simulate(200, 60, rng=np.random.default_rng(0), artifacts=0.2)
model, _ = gspice.fit(sim.flux, sim.ivar)
prediction = gspice.pixelwise(model, sim.flux[:5], sim.ivar[:5])
repaired = gspice.infill(model, sim.flux[:5], sim.ivar[:5], mask=sim.artifacts[:5])
assert np.isfinite(prediction.z).any()
assert np.isfinite(repaired.flux).all()
print(f"gspice {gspice.__version__} from {gspice.__file__}: smoke test passed")
