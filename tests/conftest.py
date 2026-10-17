import os
import sys
from pathlib import Path

from hypothesis import settings

# Runs before numpy is imported. The 2022 per-pixel loop calls BLAS on tiny blocks,
# where more threads give no speedup and slow the suite down on a busy CPU. The tests
# pass at 1 and 2 threads. VECLIB_MAXIMUM_THREADS is for Accelerate, the BLAS of numpy
# on macOS.
for name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(name, "1")

# tests/legacy reads the golden inputs through _golden.
sys.path.insert(0, str(Path(__file__).parent / "golden"))

# Design section 11; deadline=None, as a deadline turns a slow CI runner into a failure.
settings.register_profile(
    "gspice", max_examples=50, derandomize=True, database=None, deadline=None
)
settings.load_profile("gspice")
