# Golden tests

The golden tests freeze the results of the 2022 Python package (commit `1fbab9f`).
`generate_golden.py` builds deterministic synthetic inputs and calls every 2022 public function on them.
It stores the inputs and outputs here.
`test_golden.py` calls an implementation on the same inputs and compares its results with the stored ones.

| File | Content |
|---|---|
| `manifest.json` | Baseline commit and the sha256 of its source files, environment, seeds, the sha256 of each dataset array, and every case: arguments, stored outputs, behaviour expectations, `baseline_accuracy` |
| `inputs_*.npz`, `golden_*.npz` | Stored inputs and outputs |
| `_golden.py` | Rebuilds the dataset arrays bit for bit, checks their sha256, decodes the outputs (numpy only) |
| `test_golden.py` | The tests and the tolerance table |
| `generate_golden.py` | The generator |

## Run

From the repository root: `python -m pytest tests/golden` (about 10 s).
`tests/conftest.py` sets 1 BLAS thread, for the reason it states.

The tests check, for each golden case: every output, its dtype and shape, the tuple layout, and that no input array changed.
For each `covar_iter_mask` case they replay the 2022 loop and check the mask of every pass.
The behaviour records check 2022 errors, dtypes, aliasing, NaN propagation and the `>=` tie of `get_chimask`.

## Datasets

| Name | Size (spectra x pixels) | Content | cond(cov) |
|---|---|---|---|
| D1 | 2000 x 50 | Spectrum-like data at S/N 1-10, with spikes, bad columns, shifted blocks and masks; also `ivar_z` and `mask_alt` | 5.5e2 |
| D2 | 5000 x 300 | The same model at S/N 0.4-4 | 8.8e2 |
| D3 | 2000 x 50 | The same model at S/N 12-120 | 8.6e4 |
| D4 | 1500 x 150 | The same model at S/N 30-300, for mask cases only | 2.1e6 |
| E | 10 x 24 | Edge cases: a fully masked row, one good pixel, masked ends, an interior block, an alternating mask | |
| MAT | | SPD matrices M50a, M50b, M300a, M300b (n = 50, 300; cond 1e3, 1e6) and right-hand sides | |

`ivar_z` sets ivar to 0 on some unmasked pixels; `mask_alt` also flags pixels whose ivar is not 0.
At the S/N of D1 and D2 the 2022 results agree with a direct Cholesky solve to 1e-13 or better, so a different algorithm can be held to 1e-12.
The comments in `RTOL_OVERRIDES` give the accuracy of the 2022 results where it is worse than 1e-13.

Large outputs keep the norm of every row and column, and most of them also a fixed subset of rows.
`gp_interp(spec=I)` returns its weight matrix W, so each `gp_interp` case also checks the whole prediction against `spec @ W`.
The `gaussian_estimate` cases on D1 and D3 also check `predkstar` against `Dvec @ predoverD` of the default path.
D2 runs only the bruteforce branch, which returns no `predoverD`, so for its row-stored `predkstar` a case with `Dvec = I` supplies the weights.

## Tolerances

The tolerances live in `test_golden.py`: `RTOL`, `RTOL_OVERRIDES`, `MODE` and `ATOL`.

- Masks and `kstar` compare exactly.
- The comment above `RTOL_OVERRIDES` states the rule for an override.
  A test checks the table against the `baseline_accuracy` that the generator stores in `manifest.json`.
- The golden tests cannot detect an error smaller than the tolerance of an output.
  On D3 and on the cond-1e6 matrices the tolerances reach 1e-8 (D3 `predvar`), while a direct solve is accurate to 4e-13 or better on D3 and to about 1e-11 on the matrices.
  On the model of D4 at 5000 x 300 and S/N 30-300, the 2022 `predvar` differs from a direct solve by 1.8e-7 to 2.2e-7 (three seeds, nguard 20), so golden `predvar` cannot check accuracy at realistic S/N.
  Masks change only where |Z| crosses `nsigma`, so the mask cases cannot detect a relative error in Z below the smallest margin, 2.0e-5.
  The tests against a direct solve (`CLAUDE.md`, Tests) must cover all of these.
- `CLAUDE.md` states when a tolerance may change (Pull requests) and when the files are regenerated (Tests).

## Regenerate

The generator needs numpy, scipy and astropy, and takes about 17 s.

```
git archive 1fbab9f gspice | tar -x -C <dir>
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python tests/golden/generate_golden.py --baseline <dir> --out tests/golden
```

It imports the 2022 package only from `<dir>`, and checks each file there against `git show 1fbab9f`.
It reuses the inputs: it rebuilds each dataset from `inputs_<name>.npz` in `--out` and checks it against the dataset hashes of the existing `manifest.json`.
`--new-inputs` builds every dataset from its seed instead, as the generator does for a dataset with no inputs file.
Without `manifest.json`, it refuses an `--out` that holds inputs files.
It checks that every stored input and output decodes bit for bit, and that no |Z| lies within a relative `MIN_MARGIN` of the threshold `nsigma` of a mask.
The outputs are byte-identical only with the Python, numpy and scipy versions and the CPU features that `manifest.json` records, the same OpenBLAS kernel, and 2 BLAS threads.
`manifest.json` cannot record the kernel: numpy reports the kernel of its build, but OpenBLAS selects one at run time (`OPENBLAS_CORETYPE` overrides it).

## Implementation under test

The `implementation` fixture in `test_golden.py` maps a 2022 (module, function) to the callable under test.
