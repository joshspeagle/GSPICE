import os

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
