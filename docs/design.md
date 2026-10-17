# GSPICE package design

This document is the requirement for Phases 2 to 5 of the overhaul. Pull requests and their adversarial reviews check against it. A pull request that departs from it edits this file in the same pull request and records the change as a decision. Where it and the kickoff specification differ, this document holds.

Register IDs (`D-01` to `D-80`) and question IDs (`Q1` to `Q15`) refer to the Phase 0 inventory in the checkpoint report on issue #1; `D-81` is new. Decision IDs `S11.1` to `S11.10` are the ten choices that the kickoff specification of the overhaul (not in the repository) asked Phase 1 to make. Paper equations and lines (`l.NN`) refer to Finkbeiner, Speagle & Karim, arXiv:2511.16754 (revised version). r26 is the unmerged branch `revision-2026`; κ is a condition number and ε = 2.2e-16. Section 12 gives the setup of every measured number.

## 1. Purpose and principles

GSPICE models spectra on a common pixel grid as a multivariate Gaussian with an empirical covariance. It predicts any set of pixels from the others, scores each pixel, repairs bad pixels and checks its own calibration.

- **Priorities.** Correctness, then API quality, then performance.
- **One conditioning route.** Every new-API prediction uses the precision identity of section 4.2. The Schur route of paper Appendix A survives only in `gspice.legacy`, which keeps the 2022 code.
- **Frozen model, functions as verbs.** A fitted `Model` is an immutable dataclass. The public functions take it as their first argument. There is no estimator class.
- **Flux in, flux out.** Public functions take flux, inverse variance and mask, and return flux units. Scaled units appear only in `scale`, `unscale` and the fields of `Model`.
- **Two layers.** The new API lives in `gspice`. The 2022 names live in `gspice.legacy`, which reproduces the 2022 results except for the recorded behaviour changes of section 7.

## 2. Package layout and public names

```
src/gspice/
  __init__.py       public names, __version__, deprecated 2022 names (module __getattr__)
  _linalg.py        precision, Woodbury precision, guard weights, pattern solves, grouping
  _spectra.py       validation, interpolation, dilation, runs, scale, unscale, snr
  _model.py         Model, LowRank, Blocks; fit, save, load; covariance estimators
  _blocks.py        fit_blocks, block chi-square
  _predict.py       Prediction, Repaired; pixelwise, infill
  _repair.py        repair
  _diagnostics.py   finite_sample_factor, Calibration, calibration, noise_audit
  _datasets.py      Simulation, simulate
  legacy/           __init__.py, specUtils.py, matrixUtils.py, djs_maskinterp.py
  py.typed
```

Each private module holds one group of things. The import paths are `gspice`, `gspice.legacy` and its 2022 modules `specUtils`, `matrixUtils` and `djs_maskinterp`.

**Public names (21, plus `legacy`).**

| Kind | Names |
|---|---|
| Fitting | `fit`, `fit_blocks`, `save`, `load` |
| Prediction | `pixelwise`, `infill`, `repair` |
| Diagnostics | `finite_sample_factor`, `calibration`, `noise_audit` |
| Helpers | `scale`, `unscale`, `snr`, `simulate` |
| Classes | `Model`, `LowRank`, `Blocks`, `Prediction`, `Repaired`, `Calibration`, `Simulation` |

| Left out | Reason |
|---|---|
| Estimator class | It would repeat the functions as methods and double the docs and tests. A thin wrapper can be added later without breaking anything. |
| `covariance()` | `fit(..., nsigma=())` fits without clipping; a user's own matrix goes into `Model(...)`. |
| `predict_block`, `detect`, `chimask`, `exclude_extended` | `infill` with an `(npix,)` mask predicts a block. The detection steps are internal steps of `fit` and `repair`. |
| `method="schur"`, `chunk=`, `ddof=`, `strict=` | Sections 4.2, 8 and 10. |
| QQ and tail helpers on arrays | Methods of `Calibration`. |
| Iteration of paper l.472-481; rest-frame resampling and continuum normalization (D-62) | `infill` and `pixelwise` compose the iteration. Inputs are continuum-normalized spectra on a common grid. |
| PHOENIX mock (D-74) | It needs h5py, astropy and a 0.7 GB download. Out of scope for Phases 2 to 5; the examples phase may add it, and no test uses it. |

## 3. Units and data model

### 3.1 Inputs

| Argument | Shape | Units | Rule |
|---|---|---|---|
| `flux` | (nspec, npix) | flux | cast to float64; finite at good pixels |
| `ivar` | (nspec, npix) | flux⁻² | finite and ≥ 0 at pixels not in `mask` |
| `mask` | (nspec, npix) or None | none | nonzero means bad |

The bad set is always `bad = (mask != 0) | (ivar == 0)` in the new API (Q11). Values of `flux` and `ivar` at bad pixels are never read, so NaN or inf there is allowed (D-80). A spectrum without a good pixel raises `ValueError` (Q1), except in `fit`, which drops it (section 5.1). Section 8 gives the dtypes.

### 3.2 Working units

| Model kind | Working vector of spectrum s | Units |
|---|---|---|
| Mean-centred (default) | x_s = a_s f̃_s − m_b | scaled |
| Constant centre c | x_s = a_s (f̃_s − c) | scaled |
| Noise-aware | x_s = f̃_s − μ | flux |

- a_s = sqrt(mean of ivar_s over its good pixels), in flux⁻¹ (the `refscale` of 2022).
- f̃_s is the flux with bad pixels linearly interpolated along wavelength and held constant beyond the first and last good pixels. A spectrum with one good pixel becomes constant (D-42).
- m_b is the training mean of a_s f̃_s in S/N bin b (one bin by default), weighted by the user weights ω_s (section 4.1) when they are given.
- c is in flux units, because it is a continuum level such as 0.95 (paper l.722).

The inverse: f = (x + m)/a + c and var_f = var/a² in `unscale`, and cov_f = cov/a² in `Repaired.covariance` (section 6). For the noise-aware model, a = 1, m = μ and c = 0.

### 3.3 Model

```python
@dataclass(frozen=True, eq=False)
class Model:
    """Fitted Gaussian model of spectra on a common pixel grid.

    A spectrum in S/N bin b has working vector x ~ N(0, C_b) (section 3.2).
    With noise=True: x = ftilde - mean[0] ~ N(0, cov[0] + diag(nu_s)) in flux units,
    nu_s as in section 4.1.
    """
    mean: NDArray[np.float64]                 # (nbin, npix) working units: m_b, or mu; zeros if center is set
    cov: tuple[NDArray[np.float64] | LowRank, ...]  # nbin covariances, working units squared
    n_train: tuple[int, ...]                  # (nbin,) N_s; inflate and fs assume an N_s - 1 sample covariance
    _: KW_ONLY                                # the fields with defaults are keyword-only
    edges: tuple[float, ...] = ()             # (nbin - 1,) strictly increasing interior S/N edges
    center: float | None = None               # None: mean-centred; c: constant centre, flux
    noise: bool = False                       # cov[0] is noise-free; nbin must be 1
    weighted: bool = False                    # fitted with positive weights not all equal (A23)
    blocks: Blocks | None = None              # block-scan thresholds (fit_blocks)
```

`__post_init__` checks shapes, finiteness, exact symmetry of dense covariances, n_train ≥ 1, strictly increasing edges, `noise` implies nbin = 1 and `center is None`, a `LowRank` covariance has diag > 0 unless `noise` is set, and `blocks.thresholds` has nbin rows. It stores read-only float64 copies. `LowRank`, `Blocks` and `Calibration` also validate their fields (`LowRank`: shapes, values > 0, diag ≥ 0), store read-only copies and rebuild through `__init__` like `Model`. A private `_cache` dict (`field(init=False, repr=False, default_factory=dict)`) holds the precision P_b per bin and the guard weights for the latest nguard per bin, so there is no global state. Each cached matrix takes npix² × 8 bytes (46 MB at npix 2400). `__reduce__` rebuilds through `__init__`, so pickle, copy and deepcopy validate again, make the arrays read-only and start with an empty cache.

```python
@dataclass(frozen=True, eq=False)
class LowRank:
    """Low-rank plus diagonal covariance C = V diag(values) V^T + diag(diag)."""
    vectors: NDArray[np.float64]   # V (npix, K)
    values: NDArray[np.float64]    # (K,) > 0
    diag: NDArray[np.float64]      # (npix,) >= 0; > 0 unless the model is noise-aware

@dataclass(frozen=True, eq=False)
class Blocks:
    """Block-scan geometry and per-block thresholds (paper 3.2)."""
    starts: NDArray[np.int64]       # (nblock,) first pixel of each block
    width: int                      # pixels per block
    nguard: int                     # guard half-width around each block
    thresholds: NDArray[np.float64] # (nbin, nblock) chi-square per pixel
```

## 4. Covariance models and the conditioning primitive

### 4.1 Estimators

ω_s are the user weights (`weights`, 1 by default). `fit` uses a training spectrum unless it has more than maxbadpix input bad pixels, has ω_s = 0, or a pass mask or the final mask covers it completely; N_s counts the spectra of the final fit. Spectrum s has weight w_s = ω_s, or a_s² ω_s in the noise-free estimator; V1 = Σ w_s and V2 = Σ w_s² (reliability weights).

| Kind | `fit` arguments | Estimator | Requirement |
|---|---|---|---|
| Dense, mean-centred | default | S = Σ_s w_s (x_s − x̄_w)(x_s − x̄_w)ᵀ / (V1 − V2/V1); unweighted: X_cᵀX_c/(N_s − 1) | N_s − 1 ≥ npix (Q2); no pixel bad in every training spectrum |
| Dense, constant centre | `center=c` | S = Σ_s w_s x_s x_sᵀ / V1 (no mean is estimated) | N_s ≥ npix; no pixel bad in every training spectrum |
| Low rank plus diagonal | `rank=K` | (V, Λ) = leading K eigenpairs of S; d = diag(S) − Σ_k λ_k v_k² | λ_K > 0 and every d_i > 0 |
| S/N binned | `snr_bins=` | the chosen kind per bin, with its own m_b and N_s | per bin as above |
| Noise-aware | `noise=True` | Σ₊ from the estimator below; with `rank=K`, (V, Λ) are its leading K eigenpairs and d_i = Σ_{k>K} max(λ_k, 0) Q_ik² | W_i > 0; every denominator below > 0; λ_K > 0 with `rank=K` |

Covariances are accumulated in float64, in row chunks, and are exactly symmetric.

**Noise-free covariance (D-63; the paper gives no estimator).** Let u_si = 1 at good pixels of the final training mask and 0 elsewhere, and let a_s use the same pixels. With w_s = a_s² ω_s the estimate is precision weighted:

- W_i = Σ_s w_s u_si and μ_i = Σ_s w_s u_si f_si / W_i;
- A_si = √w_s u_si (f_si − μ_i), D1 = BᵀB with B_si = √w_s u_si, and D2 = EᵀE with E_si = w_s u_si;
- N_i = Σ_s w_s u_si (1 − w_s/W_i) / ivar_si;
- Σ_ij = ((AᵀA)_ij − δ_ij N_i) / (D1_ij − D2_ij (1/W_i + 1/W_j − D1_ij/(W_i W_j))).

The estimate Σ is unbiased element by element if (1) spectra are independent, with a common mean μ and a common signal covariance; (2) the noise has zero mean and variance 1/ivar; (3) the noise is uncorrelated between pixels and with the signal; and (4) masks, weights and ivar do not depend on the flux. Gaussianity is not needed. The outlier flags of the final mask break condition 4, and so does photon noise with ivar computed from the flux. The derivation goes in the pull request that implements the estimator (5.3). Σ₊ = Q max(λ, 0) Qᵀ, from `eigh` of Σ, is biased upward (trace +17% at npix 200 and N_s 5000, with the paper's S/N law and 2% bad pixels). At prediction, C_s = Σ₊ + diag(ν_s), or VΛVᵀ + diag(d + ν_s), with ν_si = 1/ivar_si at good pixels and 1/a_s² at bad pixels, so that P_s exists. Cost: three products of N_s·npix² and one `eigh`.

**Clipping passes.** The passes of paper Alg. 4 use a scaled model with the requested rank and centre, and unit weights for every spectrum that `fit` uses: dense, or `LowRank(K)`, which clips when N_s − 1 < npix. With `snr_bins`, one pooled model clips all spectra (provisional, A6), and each bin fits its final covariance on the final mask. The noise-aware estimator is fitted once, on the final mask of the scaled clipping; with `rank=None` that clipping needs N_s − 1 ≥ npix. In-sample clipping has a ceiling: for a dense model, the in-sample |Z| of an outlier of any amplitude approaches sqrt((N_s − 1)(N_s − N_k − 1)/N_s) in a spectrum of typical leverage, with N_k = npix − 2·nguard − 1 (6.4 at N_s = npix + 1 = 2401 and nguard 20). `fit` logs a WARNING naming each pass whose nsigma exceeds it.

**S/N statistic.** snr_s = median of f_si·sqrt(ivar_si) over the good pixels of section 3.1, in `fit` and at prediction; it does not depend on the flux units. `snr_bins=n` puts the interior edges at the 1/n, ..., (n − 1)/n quantiles of the unweighted snr of the spectra with at most maxbadpix bad pixels and ω_s > 0, and a sequence gives them. Spectrum s goes to bin `searchsorted(edges, snr_s, side="right")`. `fit` raises ValueError naming a bin that ties leave below its requirement.

### 4.2 Conditioning primitive

Let P = C⁻¹ be the precision matrix. For a pixel set G with complement O, the conditional density of x_G is proportional to exp(−½ x_Gᵀ P_GG x_G − x_Gᵀ P_GO x_O), so

  x_G | x_O ~ N(μ_G, P_GG⁻¹),  μ_G = −P_GG⁻¹ P_GO x_O = −P_GG⁻¹ (P_{G,:} x₀),

where x₀ is x with the G entries set to 0. Values in G are never read, so an outlier in G cannot cause cancellation. Pixels of G that are not reported are marginalized. G = all pixels gives the prior N(0, C).

Every new-API algorithm uses this identity. Schur updates of a precomputed inverse (Appendix A) factorize the same |G|×|G| block of P per pixel set and add products with C, so they are never cheaper by operation count. Their accuracy depends on how strongly G is coupled to the conditioning pixels. At nguard 5, against long-double and 60-digit references (npix 120, random and squared-exponential matrices), the 2022 Schur variances err by 1.1e-4 to 3.0e-2 relative at κ about 2e8 and by 0.47 to 210 at 1e10 to 2e10, where up to 74 of 120 are negative; at 1e12, 40 to 80 of 120 are negative. The error comes from the update itself: with a Cholesky solve of U, as the paper, IDL and Julia do (D-36), it was 3.1e-5 to 3.3e-2 at κ 2e8. The precision route errs by 5.0e-10 to 3.0e-9 at κ about 2e8 and by 1.8e-6 to 1.2e-5 at 1e12 (D-37). Where the guard window is nearly independent of the conditioning pixels, the order reverses for the variances: the Schur variances err by at most 1.3e-11 at nguard 20 and are exact to rounding from nguard 40, while the precision route errs by up to about 0.5κε (squared-exponential kernel of length 5, npix 120: 1.2e-8 to 2.4e-8 at κ 2e8 for nguard 20 to 100, and 6.6e-5 to 1.2e-4 at κ 1e12 for nguard 40). The Schur means still err more at nguard 20 (3.8e-8σ against 2.3e-9σ at κ 2e8). The precision route is chosen because its error stayed within about κε in every measured regime, while the Schur error has no such bound. So there is no `method="schur"`.

**Private primitives (`_linalg`).** All take float64; w = min(2·nguard + 1, npix). Section 12 gives their measured times.

| Primitive | Computes | Cost |
|---|---|---|
| `precision(S)` | P = S⁻¹ by `dpotrf` and `dpotri`, lower triangle mirrored in place; P is then exactly symmetric, so its transpose is a C-contiguous view; `LinAlgError` if `dpotrf` fails or the reciprocal condition number of S scaled to unit diagonal, which `dpocon` estimates in the 1-norm, is below 1e-13 (A26) | npix³ |
| `woodbury(V, values, d)` | P = D⁻¹ − L Lᵀ, L = D⁻¹ V R_K⁻ᵀ, R_K R_Kᵀ = Λ⁻¹ + Vᵀ D⁻¹ V; never forms C; `LinAlgError` if κ_W > 1e13 (A28), with c_ii = d_i + Σ_k λ_k V_ik² the model variances, d̃ = d/c, Ṽ = diag(c)^−½ V and κ_W = (max d̃ + λ_max(Λ^½ ṼᵀṼ Λ^½))/min d̃, a bound on the condition number of C scaled to unit diagonal and on max c_ii/d_i | npix² K |
| `guard_weights(P, c, nguard)` | for each pixel i, with g = `slice(max(0, i−nguard), min(npix, i+nguard+1))`: v = P_gg⁻¹ e_i by Cholesky; W[:, i] = −P[g, :]ᵀ v, then W[g, i] = 0 exactly; var_i = v[i − g.start]. A window that covers all pixels gives W[:, i] = 0 and var_i = c_i, the exact prior, from c = diag(C) and without a factorization. `LinAlgError` if `dpotrf` fails | npix (w³/3 + 2 npix w) |
| `solve_pattern(P, G, M, T, X, normals=None, covariance=False)` | G, M and T are boolean masks with T ⊆ M ⊆ G, and the results follow the pixel order of M; G ordered as (G \ M, M \ T, T), with T the pixels whose χ² is wanted; R = lower Cholesky factor of P_GG; μ_G = −R⁻ᵀR⁻¹ (P_{G,:} X0ᵀ), where X0 is X with its G columns set to 0; C_MM = (R_MM R_MMᵀ)⁻¹, so var_M = column sums of (R_MM⁻¹)²; χ² = ‖R_TTᵀ r_T‖², because C_TT⁻¹ is the Schur complement R_TT R_TTᵀ; draws μ_M + R_MM⁻ᵀ z_M, the M rows of μ_G + R⁻ᵀ z, with z = `normals` (nspec × npix); `covariance=True` adds C_MM, with var_M as its diagonal; `LinAlgError` if `dpotrf` fails. Empty blocks skip LAPACK, and an empty T gives χ² = 0; callers give the exact prior where G covers all pixels (below) | per pattern \|G\|³/3 + \|M\|³/3, and \|M\|³/3 more for C_MM; per spectrum 2\|G\| (npix + \|G\|) |
| `group_rows(keys)` | rows with identical boolean key rows: `np.unique` of a void view of `np.packbits(keys, axis=1)`; the infill key packs G, M and T, and grouping runs per S/N bin | O(nspec log nspec) |

Guard windows are contiguous, so slices replace gathers; this is the IDL "contiguous" fast path (D-47) and needs no separate code. The trailing-block identities of `solve_pattern` give the variances, covariance and χ² from the one factorization (verified to 1.8e-15 at κ 1e2 to 1e8).

**How each operation uses the primitive.**

| Operation | G | Bad pixels | Once per model and bin | Per spectrum |
|---|---|---|---|---|
| Pixelwise | {j : \|j − i\| ≤ nguard} for each i | interpolated, conditioned on as data (2022 and paper) | P, W (cached) | x W: 2 npix² flops |
| Infill | replace ∪ ignore ∪ bad | never conditioned on | P (cached) | per pattern as above |
| Block scan | {j : dist(j, A) ≤ nguard}, block A ordered last | infilled before the scan | R per block; χ²_A = ‖R_AAᵀ r_A‖²/\|A\| | 2\|G\| (npix + \|G\|) per block |

A pattern is the triple (G, M, T). Spectra of one S/N bin with equal patterns share one factorization, except in noise-aware models, where each spectrum has its own P_s. Where G covers all pixels, every operation returns the exact prior from C (μ_G = 0, C_MM = C[M, M], var = diag(C)) without P; diag(P⁻¹) would err by 0.23κε to 0.29κε.

**Noise-aware models.** C_s differs per spectrum, so P_s is formed per spectrum by `precision` or `woodbury`, once per call, and every operation of the call reuses it. This is the exception that CLAUDE.md (Numerics) states. Section 12 gives the cost and the gated optimizations.

### 4.3 Finite-sample factor

For a held-out Gaussian spectrum, E[Z²]/N_* = f_s with

  f_s(N_s, N_k, N_*) = (1 + 1/N_s) (N_s − 2)/(N_s − N_k − 2) · (N_s − 1)/(N_s − N_k − N_* − 2).

N_k counts conditioning pixels only; guard and ignored pixels do not count. The formula is exact for an unweighted, mean-centred sample covariance with divisor N_s − 1, iid Gaussian working vectors, data-independent masks and Z² with the full conditional covariance (inventory section 7). With a spread of S/N the scaled vectors are a scale mixture, and f_s underestimates ⟨Z²⟩ by up to 25% at (200, 40, 1) and 5.3% at (2000, 40, 1). f_s does not hold for non-Gaussian data (D-61), nor for the Z-dependent masks of `repair`.

f_s is derived only for dense full-rank, unweighted, mean-centred models, binned or not (A13), with the N_s of the spectrum's bin. They report f_s, or NaN where N_s − N_k − N_* − 2 ≤ 0 or N_* = 0. Other kinds report 1, as no factor is derived, although their held-out factor reached 1.58 to 2.56 at N_s 200.

`inflate=True` multiplies var, every covariance and the spread of the draws by f_s(N_s, N_k, 1), and χ² uses the inflated covariance; `Repaired.fs` is then f_s(N_s, N_k, dof)/f_s(N_s, N_k, 1), so χ²/(dof·fs) stays the calibration statistic. This factor calibrates every fixed linear combination of the predicted pixels; f_s(N_s, N_k, N_*) calibrates only the joint χ² (a fixed combination measured 2.474 ± 0.003 at (60, 20, 8), where f_s(·, 1) = 2.474 and f_s(·, 8) = 3.052). `inflate=True` raises ValueError for a model without a derived f_s, and, naming N_s and N_k, where f_s(N_s, N_k, 1) is NaN; where only f_s(N_s, N_k, dof) is NaN, `Repaired.fs` is NaN.

## 5. Public API

Common conventions for every function: `ArrayLike` and `NDArray` are from `numpy.typing`; `flux`, `ivar` and `mask` are as in section 3.1; outputs are float64; inputs are never modified. `Progress` abbreviates `Callable[[str, int, int], None]` and is not public. `progress(stage, done, total)` is optional and called once per chunk with the function name as stage; done and total count spectra over the call, once per pass of `fit` and per step of `repair`.

### 5.1 Fitting

```python
def fit(flux: ArrayLike, ivar: ArrayLike, mask: ArrayLike | None = None, *,
        nsigma: Sequence[float] | NDArray[np.floating] = (20.0, 8.0, 6.0),
        maxbadpix: int = 64, nguard: int = 20, cumulative: bool = True,
        center: float | None = None, weights: ArrayLike | None = None,
        rank: int | None = None, noise: bool = False,
        snr_bins: int | Sequence[float] | None = None,
        progress: Progress | None = None) -> tuple[Model, NDArray[np.bool_]]:
    """Fit a model to training spectra with iterative outlier clipping (paper Alg. 4).

    Spectra with more than maxbadpix input bad pixels, or with weight 0, are not used.
    Pass n (1 to N) interpolates the pixels of M_n, scales each spectrum by a_s over its
    pixels outside M_n, fits the clipping model (section 4.1) and computes in-sample
    pixelwise Z with nguard at every pixel, interpolated ones included (2022, Alg. 4).
    F_n = dilate(|Z| > nsigma_n, 1), with nsigma_n the n-th threshold; F_0 = empty and
    M_1 = bad. cumulative=True: M_{n+1} = M_n | F_n, final mask M_{N+1}.
    cumulative=False (2022): M_{n+1} = bad | F_n, final mask bad | F_{N-1} | F_N.
    A spectrum that a pass mask or the final mask covers completely is dropped. The final
    covariance (section 4.1) is fitted on the final mask, with a_s over the pixels outside
    it. nsigma=() fits on the mask bad, without clipping. center: constant centre c (flux).
    weights: (nspec,) >= 0; positive weights act on the final fit only, an advanced option
    (A23; reweighting raised false flags in the paper, l.548), and weight 0 excludes a
    spectrum. cumulative=False equals legacy covar_iter_mask when nguard is 20, center,
    weights, rank, noise and snr_bins keep their defaults, flux is float64, ivar == 0 only
    at masked pixels, nsigma is not empty, no |Z| lies within the 2022 Schur error of a
    threshold (Q10), no pass mask or final mask covers completely a spectrum with at most
    maxbadpix bad pixels, every covariance passes A26, the inputs pass section 8, and the
    requirements of section 4.1 hold.

    Returns the model, and the final training mask (nspec, npix) bool, all True for
    spectra not used. Raises ValueError, naming the counts or pixels, if a requirement
    of section 4.1 fails, and LinAlgError, naming the bin, N_s and rcond, if a
    covariance fails A26, which can happen for N_s - 1 close to npix.
    Cost per pass: N_s npix^2 (covariance) + npix^3 + 2 N_s npix^2 (pixelwise).
    """

def fit_blocks(model: Model, flux: ArrayLike, ivar: ArrayLike, mask: ArrayLike | None = None, *,
               width: int = 150, stride: int = 75, nguard: int = 20,
               percentile: float = 99.9, progress: Progress | None = None) -> Model:
    """Return model with per-block thresholds from artifact-free spectra (paper 3.2).

    Blocks start at 0, stride, 2 stride, ... <= npix - width, plus one at npix - width if
    the last start is smaller (31 blocks at npix 2400). Bad pixels are infilled first, as
    in repair step 2, and no Z-based detection runs (paper l.510). For block A with
    G = A plus its guard:  chi2_A = r_A^T C_AA^-1 r_A / |A|,  r_A = x_A - mu_A,
    C_AA = (P_GG^-1)_AA.  thresholds[b, A] = np.percentile (linear) of chi2_A at p over
    the m spectra of bin b. A clean block then exceeds it with probability
    (m (1 - p/100) + p/100)/(m + 1) when (m - 1) p/100 is an integer, and at most that
    otherwise, for a density that falls beyond the threshold: 0.20% at m = 1000 and 0.12%
    at 5000 for p = 99.9 (the paper used 5000). repair also infills Z-flagged pixels, which
    lowers the rate. Use spectra not used for training. Raises ValueError, naming the bin
    and count, if m (100 - p) < 100 in a bin, with p read as the decimal Fraction(str(p))
    (m >= 1000 at p = 99.9).
    Cost: per block |G|^3/3; per spectrum about 2 sum|G| (npix + |G|) flops.
    """

def save(model: Model, path: str | os.PathLike) -> None:
    """Write model to the .npz file path, adding no suffix (section 9)."""

def load(path: str | os.PathLike) -> Model:
    """Read a model written by save. Raises ValueError unless its schema is the integer 1."""
```

### 5.2 Prediction, infill and repair

```python
def pixelwise(model: Model, flux: ArrayLike, ivar: ArrayLike, mask: ArrayLike | None = None, *,
              nguard: int = 20, inflate: bool = False,
              progress: Progress | None = None) -> Prediction:
    """Predict each pixel from all pixels farther than nguard from it (paper Alg. 3).

    x_i | x_O ~ N(-(P_gg^-1 P_gO x_O)_i, (P_gg^-1)_ii),  g = {j : |j - i| <= nguard}.
    Bad pixels enter as interpolated values; their z is NaN, and var omits the error of
    those values (next to a 60-pixel bad run at correlation length 20, <z^2> measured
    0.65 to 2.45, depending on the noise level). nguard >= npix - 1 gives the prior.
    inflate: var *= f_s(N_s, npix - |g|, 1).
    Returns Prediction in flux units.
    Cost: once per model, bin and nguard, npix^3 + guard weights; then 2 npix^2 flops per
    spectrum (section 12).
    """

def infill(model: Model, flux: ArrayLike, ivar: ArrayLike, mask: ArrayLike | None = None, *,
           replace: ArrayLike | None = None, ignore: ArrayLike | None = None,
           inflate: bool = False, covariance: bool = False,
           rng: np.random.Generator | None = None,
           progress: Progress | None = None) -> Repaired:
    """Replace pixels by their Gaussian conditional mean given the trusted pixels.

    G = replace | ignore | bad, O = the rest:  x_G | x_O ~ N(-P_GG^-1 P_GO x_O, P_GG^-1).
    replace defaults to bad. replace and ignore are each (nspec, npix) or (npix,) bool;
    an (npix,) mask applies to every spectrum (blockwise prediction). Pixels of G that
    are not replaced keep their input value and are reported as ignored; a pixel in both
    replace and ignore is replaced. Spectra of one S/N bin with equal (G, replace,
    replace & bad) share one factorization, except in noise-aware models. Correct for any G, up to all pixels (the prior). inflate: section
    4.3. covariance: keep each pattern's covariance for Repaired.covariance. rng: also
    return draws from the conditional distribution; they widen downstream errors (paper
    l.550), so var and covariance are the uncertainty of replaced pixels. To infill with
    model B what repair found with model A (Q13), pass the same mask to both calls.
    Cost per pattern |G|^3/3 + |M|^3/3; per spectrum 2 |G| (npix + |G|). An (npix,)
    mask on spectra with their own bad pixels makes one large pattern per spectrum
    (section 12).
    """

def repair(model: Model, flux: ArrayLike, ivar: ArrayLike, mask: ArrayLike | None = None, *,
           nguard: int = 20, nsigma_replace: float = 10.0, nsigma_ignore: float = 5.0,
           dilate: int = 1, blockwise: bool = True, runlength: int | None = 5,
           halfwidth: int = 150, minfrac: float = 0.5,
           count_on: Literal["all", "flagged"] = "all", inflate: bool = False,
           covariance: bool = False, rng: np.random.Generator | None = None,
           progress: Progress | None = None) -> Repaired:
    """Detect and replace bad pixels with the procedure the paper validates (l.510).

    Steps in section 5.4. Raises ValueError if blockwise is True and model.blocks is None,
    or if nsigma_ignore > nsigma_replace. runlength=None skips the exclusion step.
    inflate, covariance and rng apply to the final infill. Nothing limits the replaced
    set; the paper measured calibration up to 1024 of 2415 replaced pixels (l.530).
    Cost: two pixelwise passes, the block scans and three infills (section 12).
    """
```

### 5.3 Diagnostics and helpers

```python
def finite_sample_factor(n_train: ArrayLike, n_cond: ArrayLike,
                         n_pred: ArrayLike = 1) -> float | NDArray[np.float64]:
    """Held-out calibration factor E[Z^2]/N_* for Gaussian data (section 4.3).

    f_s = (1 + 1/N_s)(N_s - 2)/(N_s - N_k - 2) (N_s - 1)/(N_s - N_k - N_* - 2), for a
    mean-centred covariance with divisor N_s - 1. Broadcasts. Raises TypeError for
    non-integers, and ValueError unless n_train >= 1, n_cond >= 0, n_pred >= 1 and
    N_s - N_k - N_* - 2 > 0.
    """

def calibration(model: Model, flux: ArrayLike, ivar: ArrayLike, mask: ArrayLike | None = None, *,
                sizes: Sequence[int] | None = None,
                layout: Literal["run", "scattered"] = "run",
                rng: np.random.Generator, progress: Progress | None = None) -> Calibration:
    """Held-out whitened residuals for masks drawn independently of the data.

    For size n and spectrum s, M is n consecutive good pixels from a uniform start ("run",
    consecutive in the list of good pixels) or n random good pixels ("scattered");
    G = M | bad. Z^2 = r_M^T C_MM^-1 r_M with the full conditional covariance, and
    fs = f_s(N_s of the bin, npix - |G|, n) as in section 4.3. sizes=None: those of 1, 4,
    16, 64, 256 and 1024 not above half the good pixels of any spectrum. A spectrum with
    fewer good pixels than n gets n = 0 and z2 NaN. rng draws the masks spectrum by
    spectrum in row order, and size by size within a spectrum, so they do not depend on
    chunking. Use spectra not used for training.
    The sem of Calibration.ratio is conditional on the fitted model, and was 10 times
    below the spread across training sets at N_s 200.
    Returns Calibration with one column per size.
    Cost: one Cholesky of P_GG per spectrum and size, |G|^3/3 flops (section 12).
    """

def noise_audit(model: Model, flux: ArrayLike, ivar: ArrayLike, mask: ArrayLike | None = None, *,
                nguard: int = 20, progress: Progress | None = None) -> NDArray[np.float64]:
    """Ratio of residual scatter to the reported noise, per pixel.

    rho_i^2 = mean over spectra good at i of (f_si - fhat_si)^2 ivar_si, with fhat the
    pixelwise prediction. E[rho_i^2] = k_i + e_i, with k_i the mean of ivar_si times the
    true noise variance and e_i that of ivar_si (t_si - fhat_si)^2, t the noise-free flux.
    e_i holds the astrophysical error, the noise propagated from the conditioning pixels
    and the estimation error, so E[rho_i^2] >= k_i, and e_i grows with S/N and next to
    long bad runs. Use spectra not used for training, with noise uncorrelated beyond
    nguard and ivar independent of the noise. For noise=False models whose training
    spectra share k_i, Z is blind to k_i; rho is not.
    Returns rho (npix,), unit-free, NaN where no spectrum is good. Cost: that of pixelwise.
    """

def scale(flux: ArrayLike, ivar: ArrayLike, mask: ArrayLike | None = None, *,
          center: float = 0.0,
          refmean: ArrayLike | None = None
          ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Scale spectra to unit mean noise:  X_s = a_s (ftilde_s - center) - refmean.

    a_s = sqrt(mean of ivar over good pixels); ftilde interpolates bad pixels (section 3.2).
    refmean None: the column mean of a_s (ftilde_s - center); for held-out spectra pass
    refmean = model.mean[b], because their own mean breaks f_s.
    Returns X (nspec, npix) scaled, refscale a (nspec,) flux^-1, refmean (npix,) scaled.
    scale(flux, ivar, mask) is the 2022 standard_scale with the bad set of section 3.1.
    """

def unscale(x: ArrayLike, refscale: ArrayLike, refmean: ArrayLike, *, center: float = 0.0,
            var: ArrayLike | None = None
            ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Invert scale:  f = (x + refmean)/refscale + center,  var_f = var/refscale^2.

    x (nspec, n) scaled; var (n,) or (nspec, n) scaled^2. Returns f (nspec, n) flux and
    var_f (nspec, n) flux^2, or None if var is None.
    """

def snr(flux: ArrayLike, ivar: ArrayLike, mask: ArrayLike | None = None) -> NDArray[np.float64]:
    """Signal-to-noise per spectrum (nspec,): median over good pixels of flux sqrt(ivar)."""

def simulate(nspec: int, npix: int = 500, *, rng: np.random.Generator,
             artifacts: float = 0.0, gaussian: bool = False) -> Simulation:
    """Synthetic continuum-normalized spectra for tests and examples (no network).

    Two populations (labels 0 and 1) with absorption lines whose depths follow two latent
    parameters; S/N from p(S/N) ~ (S/N)^-2.5 on [30, 400] (paper l.494). artifacts: fraction of
    spectra with one injected cosmic ray (1-3 px, 5-50 sigma), bad column (5-20 px) or
    shifted block (200-600 px, at most npix - 1; 0.5-3 px shift). gaussian=True: flux ~
    N(mu, Sigma) with ivar equal for all spectra, so the scaled data are exactly Gaussian
    (for f_s tests).
    """
```

### 5.4 Repair procedure

The order of the steps is part of the requirement: it is the procedure that the paper validates on simulations (l.510). dil(F, d) adds every pixel within d of a flagged pixel along wavelength, without wrapping. X is the working-unit data with bad pixels interpolated. The pixelwise weights and P are computed once and reused.

1. **Pixelwise pass.** Z₁ = pixelwise Z of X. M₁ = dil(|Z₁| > nsigma_replace, dilate) ∪ bad. I₁ = dil(|Z₁| > nsigma_ignore, dilate) \ M₁.
2. **Block pass** (if `blockwise`). X_b = infill of X with replace = M₁ ∪ I₁. Then repeat:
   1. For each spectrum, compute ratio_A = χ²_A(X_b)/thresholds[b_s, A], with b_s its S/N bin, for its blocks not yet flagged.
   2. If the largest ratio exceeds 1, flag that block and add its pixels to M_b.
   3. Re-infill those spectra from X with replace = M₁ ∪ I₁ ∪ M_b.

   Stop when no spectrum has a ratio above 1. A block is flagged at most once per spectrum, so there are at most nblock iterations (Q5).
3. **Second pixelwise pass.** X₁ = infill of X with replace = M₁ ∪ M_b and ignore = I₁. Z₂ = pixelwise Z of X₁. M = M₁ ∪ M_b ∪ dil(|Z₂| > nsigma_replace, dilate). I = dil(|Z₂| > nsigma_ignore, dilate) \ M.
4. **Exclusion** (if `runlength` is not None). The counting set K is M ∪ I for `count_on="all"`. For `"flagged"`, K is the undilated core of M ∪ I: the pixels outside bad with |Z₁| > nsigma_replace or |Z₂| > nsigma_ignore, and the flagged blocks, K = (((|Z₁| > nsigma_replace) ∪ (|Z₂| > nsigma_ignore)) \ bad) ∪ M_b. Every run of at least `runlength` consecutive pixels of K adds every pixel within `halfwidth` of the run to I, minus M. A spectrum keeps its old I if fewer than floor(minfrac·npix) conditioning pixels would remain (D-57).
5. **Final infill** of X with replace = M and ignore = I, in flux units. Unreplaced pixels are the input, bit for bit.

## 6. Result types

All are frozen dataclasses with `eq=False`. Arrays are read-only.

| Type | Field | Shape | Units and content |
|---|---|---|---|
| `Prediction` | `flux` | (nspec, npix) | flux: prediction of each pixel |
| | `var` | (nspec, npix) | flux²: conditional variance (includes noise) |
| | `z` | (nspec, npix) | (x − x̂)/sqrt(var); NaN at bad pixels |
| `Repaired` | `flux` | (nspec, npix) | flux: conditional mean at replaced pixels, the input elsewhere |
| | `var` | (nspec, npix) | flux²: conditional variance at replaced pixels; 1/ivar at good pixels outside them; inf at bad pixels outside them (Q14) |
| | `replaced`, `ignored` | (nspec, npix) bool | pixels replaced; pixels neither conditioned on nor replaced |
| | `chi2` | (nspec,) | r_Tᵀ C_TT⁻¹ r_T over T = replaced pixels that are good in the input, with the input values; inflated with `inflate=True`; 0 where dof = 0 |
| | `dof` | (nspec,) int | \|T\| |
| | `fs` | (nspec,) | f_s(N_s, N_k, dof), N_k = npix − \|G\|, divided by f_s(N_s, N_k, 1) with `inflate=True`; as in section 4.3 |
| | `draws` | (nspec, npix) or None | flux: like `flux`, with replaced pixels drawn from the conditional distribution |
| | `blocks` | (nspec, nblock) bool or None | blocks flagged (repair only) |
| | `covariance(s: int)` | method | → (pixels (m,) int64, cov (m, m) flux²) for spectrum s; ValueError naming `covariance=True` if it was not requested |
| `Calibration` | `z2`, `n`, `fs` | (nspec, k) each | χ², pixels per prediction, f_s as in section 4.3; `n` and `fs` broadcast at construction; the methods use only entries with n > 0 and finite z2 and fs |
| | `ratio(where=None)` | → (mean (k,), sem (k,)) | mean of z2/(n·fs) and its sem, std(ddof=1)/√m over the m entries used; the mean is NaN for m = 0 and the sem for m < 2; `where` (nspec,) bool selects spectra |
| | `tail(q=0.99, where=None)` | → (k,) | fraction with z2/fs above the χ²_n quantile q, which is 1 − q only for N_s − N_k ≫ n (1.37% at (200, 40, 10) for q = 0.99) |
| | `qq(i=0, where=None)` | → (expected, observed) | column i: normal scores Φ⁻¹((j − ½)/m), and the sorted scores of z2/fs under χ²_n, as Φ⁻¹(cdf) where cdf < ½ and −Φ⁻¹(sf) elsewhere |
| `Simulation` | `flux`, `ivar`, `truth`, `artifacts` | (nspec, npix) | flux, flux⁻², noise-free flux, injected-artifact mask |
| | `snr`, `labels` | (nspec,) | S/N, population label |

- **Draws.** Normals come from `rng.standard_normal((n, npix))` per chunk of n rows, in row order, and nothing else draws from rng in that call. Spectrum s uses its entries at G_s. The normals therefore do not depend on the chunk size or the grouping, and the draws agree to rounding.
- **Covariance storage.** `Repaired` keeps each pattern's C_MM in working units and divides by a_s² (and multiplies by f_s(N_s, N_k, 1) with `inflate=True`) on access. At one 30-pixel pattern per spectrum this takes 7.2 KB per spectrum (720 MB per 1e5), so it is opt-in.
- **Calibration from any result.** `Calibration(z2=r.chi2[:, None], n=r.dof[:, None], fs=r.fs[:, None])` calibrates a held-out infill whose replace and ignore masks do not depend on the data. `Calibration(z2=p.z**2, n=1, fs=1)` calibrates a pixelwise result computed with `inflate=True`.

## 7. Legacy layer and deprecation policy

### 7.1 Implementation

`gspice.legacy` exports the eleven names of the 2022 `__init__.py` with their 2022 signatures, return layouts and dtypes. Its `__all__` also lists the three 2022 modules, so `from gspice.legacy import *` binds what the 2022 `from gspice import *` bound. `average_bilinear` stays reachable as `gspice.legacy.djs_maskinterp.average_bilinear`, unexported as in 2022 (D-38). The golden fixture maps module `gspice.X` to `gspice.legacy.X`.

The rule: legacy keeps every behaviour of the 2022 code run with its asserts enabled, its arithmetic, quirks and errors included, except the behaviour changes of this section and section 7.2. Each change replaces a crash, aliasing, a wrong result caused by a fully masked spectrum, by non-finite values at masked pixels or by fixed-width integer arithmetic, or a result for an argument outside its documented domain, with a correct result or a named error. No change alters a 2022 result that was correct for arguments inside their documented domains, except the API changes of section 7.2: the 2022 modules move and `gspiceMain` is deleted (A24), and the top-level 2022 names warn and leave `__all__` (A25). A defect not listed here is kept; fixing it later is a new behaviour change.

| 2022 name | Implementation | Kept (legacy-keep) | Behaviour changes |
|---|---|---|---|
| `maskinterp1`, `maskinterp`, `average_bilinear` | 2022 code | D-19 (0.5 is good), D-20 (int truncation), D-21 (`interp1d` stays: `np.interp` differs from it in the last bit under scipy 1.18, which moves truncated integer values by ±1), D-39 (axis naming), D-42 (one good pixel), fully masked row unchanged | **BC** in a row with one good pixel, a non-finite flux at a masked pixel becomes the good value, where 2022 gave NaN (`yval*0 + value`), so it no longer spreads to up to npix − 1 columns of `standard_scale` X (D-81). Every other entry keeps the 2022 bits, signed zeros and a ±inf good pixel included, and so does every golden row |
| `standard_scale` | 2022 code; bad = `mask != 0`, or `ivar == 0` if mask is None (D-30) | 2022 mask rule; refscale 0 for a row whose good pixels have ivar 0 (D-30); non-finite values at good pixels propagate | **BC** a row whose 2022 pixmask is nonzero everywhere (NaN and 0.5 count; mask=None: ivar == 0 everywhere) raises, naming the rows, after the shape asserts of `maskinterp`, so 1-D input keeps its AssertionError (D-13, Q1). **BC** refscale sums `np.where(wt != 0, ivar, 0) * wt`, so non-finite ivar at a masked pixel no longer poisons X (D-80); finite inputs, float32 included, give the 2022 result bit for bit |
| `covar` | 2022 code | float32 input raises (record `float32_assert`); D-33 | none |
| `cholesky_inv` | 2022 code | D-23 (only the lower triangle enters the result; a float32 M is factorized in float32 and returned as float64 at float32 accuracy, 7.8e-6 on M50a) | none |
| `submatrix_inv`, `submatrix_inv_mult` | 2022 code with its explicit inverses (D-36) | `pad=True`, D-26 (brute ignores `pad`); every imask as in 2022 (D-24): (N,) and (N, 1) work on every path, some 2-D forms work on some paths, the others raise LinAlgError or return wrong shapes, and a first dimension other than N raises AssertionError | **BC** a copy when nr = 0 (D-25) |
| `gaussian_estimate` | 2022 code: Schur update of `covinv`; brute: `cholesky_inv` of S_kk | 4-tuple fast, 3-tuple brute with predkstar (nspec,) when nk* = 1; D-04 cost; D-23 (a float32 cov gives float64 results at float32 accuracy); D-27; icond and ipred in the imask forms of `submatrix_inv_mult`; empty ipred or icond; an overlap of ipred and icond, which returns the conditioned pixels as data with variance 0 to rounding (D-28) | none |
| `gp_interp` | 2022 code: per pixel, the Schur update of `gaussian_estimate` on `cholesky_inv(cov)` (Q10). Brute: the 2022 code with its three crashes fixed | the Schur route and its accuracy (Q10); D-03 and D-04 cost; D-23 (as in `gaussian_estimate`); predvar a writable (nspec, n) array with identical rows (D-29); inclusive irange end | **BC** irange must be 1-D by `np.ndim`, with at least two elements; i0 = irange[0] and i1 = irange[1] are integers, read from irange itself so that `np.asarray` cannot promote them, later elements are ignored as in 2022 (the goldens use [10, 30, 0]), and it raises, naming irange, unless 0 ≤ i0 ≤ i1 ≤ npix − 1. npix 0 raises, naming spec, whatever irange is. There 2022 raised AttributeError, AssertionError, IndexError, TypeError, UnboundLocalError (at npix 0, from a print) or ValueError, or, for −npix ≤ i0 < 0 with i1 = npix − 1, returned \|i0\| leading columns of zeros (D-01, D-18). **BC** nguard must be an integer ≥ 0. A negative nguard raises (D-17); 2022 predicted each pixel from itself (on D1, pred = X to 3.8e-13 and \|predvar\| ≤ 8.5e-14). Fixed-width numpy integers now work: under numpy 2, 2022 computed i − nguard and i + nguard + 1 in the type of nguard, so np.uint8 to np.uint32 predicted the first nguard pixels from themselves, np.int8 gave wrong windows from pixel 127 − nguard, and np.uint8 and np.int8 raised OverflowError at npix above 256 and 128; np.uint64 raised TypeError under every numpy. **BC** nguard, irange[0] and irange[1] are read with `operator.index` after a check that rejects numpy bools, and a failure raises TypeError naming the argument: numpy bools (2022: 0 or 1) and arrays of one or more dimensions (2022: ValueError for `np.array([5])`) raise, while 0-d integer arrays and Python bools work as in 2022. **BC** bruteforce works (D-02): the three 2022 crashes (a 3-tuple unpacked into 4, a (nspec,) predkstar written into an (nspec, 1) slice, and the unbound predoverD) are fixed, so each pixel uses the 2022 formula with `cholesky_inv` of S_kk, in the 2022 layout and over irange, and an empty k gives the prior without calling dpotri. Its explicit inverse cancels like the fast route: on D3 its predvar errs by 3.9e-10 at nguard 5 (fast route 7.6e-10, a Cholesky solve 1.3e-13), and by 5e-5 to 3e-2 at κ 2e8 to 3.5e8; it passes every `gp_interp` golden record at the table tolerance. Cost per pixel: `dpotrf` and `dpotri` of S_kk, about (npix − w)³ flops, plus 2·nspec·npix |
| `get_chimask` | 2022 composition on legacy functions; `>=`; ±1 dilation; nguard 20 | D-16 (no nguard argument), D-40 | inherits Q1 (the same ValueError class), and D-80 and D-81 (a crash becomes a result) |
| `covar_iter_mask` | 2022 non-cumulative loop on legacy `get_chimask` | float64 finalmask, rejected rows 1.0 (D-31); D-32; D-41; mask required; the 2022 check nmask ≥ npix, so nmask = npix raises LinAlgError or returns a singular covariance silently (D-15) | inherits D-80 and D-81. **BC** an explicit mask=None means ivar == 0, also for maxbadpix (D-14). **BC** a kept spectrum that is fully masked, by its input mask or by clipping, raises `_LegacyError`, naming its input row (Q1; 2022: a ValueError in the first pass for an input mask, and for clipping an all-NaN covariance or an error in the next pass). The check of the input mask runs after the nmask check and before the first pass, so with an empty nsigma it replaces the UnboundLocalError of D-32 for such a row |

**Numerics.** Legacy keeps the 2022 arithmetic, so its results equal those of 1fbab9f bit for bit on the same platform, except where a listed change applies; D-80 and D-81 use new expressions that give the 2022 bits on finite inputs. The 2022 results vary with the BLAS kernel, the thread count and the library versions, by up to 9.7e-10 in the D3 predvar; the tolerances of #9 cover that spread.

**Hygiene without BC** (bit-identical): no prints except the one that `covar(checkmean=True)` asks for (D-05, D-06), and no LAPACK message from a 0 × 0 `cholesky_inv`, which returns its (0, 0) float64 array without calling dpotri (D-05); no astropy import (D-07, in PR 2.1); dead code removed (D-38); docstrings fixed (D-11, D-12); ruff fixes and formatting (A29, in PR 2.4).

**Errors** (A20). The 2022 asserts and the new argument checks raise `_LegacyError(AssertionError, ValueError)`, whose message names the argument and contains the 2022 message where it replaces an assert (D-22). **BC**: an `except ValueError` clause now also catches the 2022 assert failures. Integer conversion raises TypeError, as above, and an argument that an assert dereferences, such as M=None, still raises AttributeError, as with the 2022 asserts enabled. **BC** under `python -O`: 2022 skipped its asserts there and returned results, or other errors, for inputs they reject, such as a float32 covariance from `covar`, a copy from `maskinterp` for 1-D input without bad pixels at axis 0 or for an axis other than 0 and 1 (taken as axis 1), `MinvY` from `submatrix_inv_mult` for a 1-D Y when nothing is removed, and a result from `submatrix_inv` for a non-square M on the fast path. Legacy always checks.

**Golden records affected.** PR 2.3: `behaviour.gp_interp.irange_two_elements_assert` (D-01) and `.irange_empty` (D-18; 2022 raised UnboundLocalError from a print). PR 2.5: `standard_scale.E.with_fully_masked`, `behaviour.standard_scale.fully_masked_nan` (Q1) and `behaviour.covar_iter_mask.mask_none` (D-14). PR 2.6: `behaviour.submatrix_inv.nr0_returns_input` and `behaviour.submatrix_inv_mult.nr0_returns_input` (D-25). PR 2.7: `behaviour.gp_interp.bruteforce_unpack` (D-02). The `gp_interp.D1.ng5.irange10_30` cases stay valid. `behaviour.covar_iter_mask.not_enough_spectra` and `behaviour.covar.float32_assert` still hold: AssertionError is in the MRO of `_LegacyError`, and the message contains the 2022 one.

### 7.2 Deprecation policy

- `gspice.legacy` is the frozen reproducibility layer. It emits no warning of its own, gets bug fixes only and stays indefinitely; numpy warnings of the kept 2022 arithmetic, such as the cast warning of D-19, pass through (A21).
- The top-level 2022 names resolve through the module `__getattr__` to `gspice.legacy`, with a `DeprecationWarning` (stacklevel 2) that names the replacement: `standard_scale` → `scale`; `covar` and `covar_iter_mask` → `fit`; `get_chimask` and `gp_interp` → `pixelwise`; `gaussian_estimate` → `infill`; `cholesky_inv`, `submatrix_inv`, `submatrix_inv_mult`, `maskinterp1` and `maskinterp` → `gspice.legacy.<name>`. They are removed in 3.0.0, the next major release after 2.0.0 (A25).
- **BC (API).** `__all__` holds the new names and `legacy`. So `from gspice import *` binds the new names, which can shadow names such as `load` and `save` of an earlier `from numpy import *`, and no longer binds the 2022 names (A25); the CHANGELOG gives `from gspice.legacy import *`.
- **BC (API).** The modules `gspice.specUtils`, `gspice.matrixUtils` and `gspice.djs_maskinterp` move to `gspice.legacy.*`, and `gspice.gspiceMain` is deleted (A24, D-38). `import gspice.specUtils` and `from gspice.specUtils import x` raise ModuleNotFoundError, `from gspice import specUtils` raises ImportError, the attribute `gspice.specUtils` raises AttributeError, and `from gspice import *` no longer binds the three modules. For `gspiceMain` the three import forms give the same errors; its attribute already raised AttributeError in 2022, and the star import never bound it. Pickles that name the old module paths no longer load. `average_bilinear`, which 2022 exposed only as `gspice.djs_maskinterp.average_bilinear`, is `gspice.legacy.djs_maskinterp.average_bilinear`. The CHANGELOG gives the migration.

## 8. Validation, errors, logging, progress, memory

**Validation at every public entry** of the new API, before any computation. `gspice.legacy` validates only as section 7.1 states.

| Check | Error |
|---|---|
| `flux`, `ivar`, `mask` 2-D with equal shapes; npix equal to the model's | ValueError |
| `flux`, `ivar` integer or floating, not bool or complex; `mask` bool, integer or floating; `replace`, `ignore` bool, so that no integer array is read as indices (D-70) | TypeError |
| finite `flux` at good pixels; finite `ivar` ≥ 0 at pixels not in `mask` | ValueError |
| no spectrum without a good pixel (names the count and the first 10 rows; `fit` drops such spectra instead, section 4.1) | ValueError |
| integers are `numbers.Integral`, not bool; 1 ≤ `width` ≤ npix; `stride`, `runlength`, sizes ≥ 1; `nguard`, `halfwidth`, `dilate`, `maxbadpix` ≥ 0; 1 ≤ `rank` < npix | TypeError or ValueError |
| `nsigma` a 1-D sequence or array of positive finite numbers; thresholds positive; 0 ≤ `minfrac` ≤ 1; 0 < `percentile` < 100; 0 < `q` < 1 | TypeError or ValueError |
| `weights` (nspec,), finite, ≥ 0; `snr_bins` an integer ≥ 1 or a strictly increasing finite sequence; `center` finite; `count_on`, `layout` one of their literals; 0 ≤ `artifacts` ≤ 1; `where` (nspec,) bool | TypeError or ValueError |
| `replace`, `ignore` of shape (nspec, npix) or (npix,) | ValueError |
| `rng` a `numpy.random.Generator` | TypeError |
| incompatible options: `noise` with `snr_bins` or `center`; `inflate` without a derived f_s, or where it is NaN (section 4.3; `repair` checks the model kind at entry, and the NaN case before its final infill, when G is known) | ValueError |

Messages name the argument and what was expected, for example `"ivar: expected shape (2000, 50) like flux, got (2000, 49)"`. A covariance that is not positive definite or fails the rcond check (A26) raises `numpy.linalg.LinAlgError` (a ValueError subclass) with the bin in the message. There is no public exception class.

**Logging and progress.** `logging.getLogger("gspice")` with a `NullHandler`. Each fit pass logs at INFO the nsigma, fraction flagged, N_s and spectra dropped, and a weighted fit logs V1²/V2; timings go to DEBUG. No `print`. The package never sets thread counts.

**Memory.** Rows go in chunks of max(1, 2²⁴ // npix) spectra (6990 at npix 2400, 134 MB per float64 array), with a few temporaries per chunk. There is no `chunk` argument. Outputs are O(nspec·npix): `pixelwise` returns 3 × 1e5 × 2400 × 8 B = 5.76 GB for 1e5 spectra, and that run peaked at 8.5 GB with float32 inputs. The model cache adds npix² × 8 bytes per bin for P and for W, also for `LowRank`. Users split very large surveys by calling `pixelwise`, `infill` and `repair` on slices, whose results agree with one call to rounding (3.2e-14 relative measured; a one-row slice uses a matrix-vector product).

**Dtypes.** float32 and non-contiguous inputs are cast to float64 per chunk. Covariances, factorizations and solves are float64.

## 9. Serialization

- **Format.** One `.npz` file that `save` writes with `np.savez` through an open file, so the file is exactly `path`; `load` reads it with `np.load(allow_pickle=False)`. No extra dependency.
- **Keys.**
  - `schema` (int, 1) and `gspice_version` (str);
  - `mean`, `n_train`, `edges`, `center` (NaN for None), `noise` (only if True), `weighted`;
  - per bin b: `cov{b}`, or `vectors{b}`, `values{b}` and `diag{b}`;
  - with blocks: `block_starts`, `block_width`, `block_nguard`, `block_thresholds`.
- **Not stored.** P and W: they are recomputed on first use (0.39 + 0.10 s at npix 2400).
- **Load.** It raises ValueError, naming the path, for a file that is not a zip archive or is a damaged one, for a schema other than the integer 1, for a missing array and for an array that `save` does not write. Other errors, such as that of a missing file, pass through, and `Model.__post_init__` validates the fields.
- **Compatibility.** A later version keeps writing and reading every array of schema 1, writes a new array only where its value differs from the default, and changes the meaning of an existing array only with a new schema. An older version then reads every later file whose new fields hold their defaults, and rejects every other.
- **Size.** 46 MB per dense bin at npix 2400.

## 10. Decisions

BC marks a change of a 2022 result in `gspice.legacy` (section 7.1). BC-new marks a new-API default or behaviour that differs from its 2022 equivalent, which also changes a 2022 result. The pull request that implements a BC or BC-new item adds a CHANGELOG line under "Behaviour changes" and a test that shows the 2022 and the new result (section 11). The adversarial review of these items happened in this design (PR #12); a pull request that departs from them gets the review that CLAUDE.md states. The lead's decisions on Q3, Q4, Q8 and Q9: "Okay, so I actually do not know the answer for these, and so I'm happy to go with the recommended defaults."

| ID | Decision | Reason | Alternative |
|---|---|---|---|
| S11.1 | `fit(cumulative=True)`; legacy non-cumulative. **BC-new** | paper Alg. 4 | False everywhere |
| S11.2 | `>` in the new API with no flag; legacy `>=`. **BC-new** | paper, IDL and Julia; golden masks are tie-free | `strict=` flag |
| S11.3 | Divisor N_s − 1 (mean-centred), no `ddof` argument | code, revised paper and f_s agree; deleting a concept beats a flag | `ddof=` argument |
| S11.4 | Precision route only, in the new API. **BC-new**: on D3, `pixelwise` moves the 2022 `gp_interp` predvar by 7.6e-10 and 2.7e-10 at nguard 5 and 20 toward the 60-digit reference, and `infill` moves the D3 `gaussian_estimate` covariance by 6.4e-11 normwise in the same way | never cheaper by operation count; its error stayed within about κε in every measured regime, while the Schur error has no bound where the window is coupled to the conditioning pixels (section 4.2) | `method="schur"` |
| S11.5 | Flux in and out; scaled units through `scale`/`unscale` | one unit system per layer, hard to misuse | `scaled=` flag |
| S11.6 | (nspec, npix) | all existing Python and Julia code | none |
| S11.7 | Import and distribution name `gspice` | continuity | none |
| S11.8 | `license = "GPL-3.0-only"`, provisional; the authors choose "-only" or "-or-later" before release | permission can be widened later, not narrowed | "-or-later" now |
| S11.9 | Sphinx; the docs phase picks the theme after it compares the comparables the lead named, PyData by default | all six comparables use Sphinx, with different themes | MkDocs |
| S11.10 | Python ≥ 3.11; PR 2.1 drops 3.10 from CI, which changes the CI decision of #9 | 3.10 reached end of life on 2026-10-01; the oldest-stack CI job runs 3.11, so the floors of Python, numpy and scipy are tested together | 3.10 |
| Q1 | A spectrum without a good pixel raises (sections 3.1 and 7.1); `fit` checks only the spectra it uses, and drops a spectrum that a pass mask or the final mask covers completely, where legacy raises (section 5.1); no `on_empty` option. **BC**, **BC-new** | NaN in every row helps nobody; the filter is one line, for the new API `bad = (ivar == 0) if mask is None else (np.asarray(mask) != 0) \| (ivar == 0); keep = ~bad.all(axis=1)`, and for legacy without the `\| (ivar == 0)` term | `on_empty="raise"\|"drop"`, default "raise" (inventory default) |
| Q2 | The requirements of section 4.1, per bin; legacy keeps the 2022 check nmask ≥ npix (section 7.1). Near the minimum, A26 can still refuse the covariance. **BC-new** | otherwise S is singular: 2022 accepted nmask = npix, then raised LinAlgError or flagged most pixels silently (D-15) | the same requirement in legacy; a higher minimum |
| Q3 | `repair(dilate=1)` as a parameter (lead) | r26 simulations | 2 (paper Fig. 8) |
| Q4 | `count_on="all"` default (lead); `"flagged"` counts the undilated core of section 5.4 | "all" reproduces r26 and the kickoff specification. K for "flagged" is a subset of M ∪ I, so before the minfrac check it never widens I more than "all"; the fallback is per spectrum, so "all" can skip a spectrum that "flagged" still widens | `"flagged"` default |
| Q5 | Worst block = largest χ²/threshold; iterate until none exceeds; each block at most once | paper "until no block exceeds"; the cap of nblock follows | cap 6 (r26) |
| Q6 | `repair` raises without `model.blocks` unless `blockwise=False` | the default must be the validated pipeline (D-68) | thresholds from the training set, or a warning |
| Q7 | The golden tolerance table of #9 stays. A legacy output that new code computes would be a BC if its golden override is looser than 1e-10, or if it differs from 2022 by more than 1e-10 on a measured input; under Q10 legacy keeps the 2022 arithmetic, and the new expressions of D-80 and D-81 give the 2022 bits on finite inputs | CLAUDE.md; the overrides cover the spread of the 2022 results across platforms | per-environment goldens with bit identity |
| Q8 | revision-2026 is not merged or tagged, and "1.0.0" is not reused (lead); version `2.0.0.dev0` until release preparation | distinct from r26 "1.0.0" and 2022 "0.0.1"; installs over r26 as an upgrade | 0.1.0.dev0 |
| Q9 | No vendored `gspice_ref.py`; in-house oracle (lead) | licence unknown (D-77) | vendor |
| Q10 | Legacy keeps the 2022 code: `gp_interp`, `gaussian_estimate` and the matrix utilities use the Schur updates of Appendix A, and the precision route serves the new API | the Schur variances cancel where the guard window is coupled to the conditioning pixels, but are exact to rounding where it is nearly independent, and there the precision route errs by up to about 0.5κε (section 4.2). A move would change correct 2022 results without a signal, against the first constraint of the overhaul, no silent regressions; `pixelwise` gives the bounded route | legacy `gp_interp` on the precision route (a numeric BC) |
| Q11 | New API: bad = mask ∪ (ivar == 0), also counted against maxbadpix. **BC-new** | ivar = 0 carries no information (D-30) | mask only |
| Q12 | The kickoff specification is not edited; this document supersedes it. Sections 4.2 (D-37) and 12 (D-65) hold the measured replacements of its figures; its statements that are not in the paper (D-58) or hold only for an unstated N_k (D-60) are not requirements | one home per fact; the specification is planning material of one session | correct the specification |
| Q13 | Detect with model A, infill with model B: two calls | no new concept | `infill_model=` |
| Q14 | `Repaired.var` of section 6: 1/ivar at good pixels outside the replaced set, inf at bad ones | a complete flux² error array that never reads ivar at bad pixels | 0 (r26) or NaN |
| Q15 | New API: nonzero mask means bad; flux is cast to float64 before interpolation, which changes integer flux (D-20) and float32 flux (2022 interpolates in float32: up to 5.9e-8 relative, bound 2⁻²⁴); legacy keeps D-19 and D-20. **BC-new** | consistency; legacy fidelity | none |
| A1 | No estimator class | section 2 | `GSPICE` class |
| A2 | Low rank uses the Woodbury precision; C is never formed | npix²K. For nearly uniform d, more accurate than a dense inverse in every measured case with \|O\| ≥ 2K, usually for K < \|O\| < 2K, and about κε like it for \|O\| ≤ K (section 12). Its error grows as ε·κ_W where some d_i ≪ c_ii, and a dense inverse can then be exact (A28) | dense S_K |
| A3 | All conditioning works on a dense P; no operator abstraction | one code path; an operator is optimization (b) | `block(g)`/`dot(Y, g)` operator |
| A4 | Noise-free estimator of section 4.1 | the raw Σ is unbiased element by element with per-pixel masks under its conditions; Σ₊ is not | interpolate masked pixels |
| A5 | S/N = median of f·sqrt(ivar) over the input good pixels; equal-count bins | independent of flux units; equal counts maximize the smallest N_s | refscale |
| A6 | With `snr_bins`, one pooled scaled model of the requested rank runs the clipping passes (provisional), and each bin fits its final covariance on the final mask. PR 5.2 runs `fit(snr_bins=3)` on `simulate(3n, 500, rng=rng, artifacts=0.4)` for n = 2·npix and 8·npix spectra per bin, with 10 seeds. Completeness at A is the fraction of injected cosmic rays with peak in [A − 1, A + 1]σ whose peak pixel is in the final mask. Per-bin clipping becomes the default only if, at n = 8·npix, its completeness at 12σ and 25σ is not below that of pooled clipping by more than 2 standard errors of the paired difference over the seeds, and in every bin \|rms(Z/f_s^½) − 1\| is smaller by more than 2 standard errors, with Z pixelwise at the good pixels of held-out `simulate(artifacts=0)` spectra and f_s = f_s(N_s,b, npix − \|g\|, 1). The noise-aware estimator is fitted once, on the final mask | per-bin clipping cannot flag in small bins, because the in-sample ceiling of section 4.1 falls below the thresholds: with 125 spectra per bin at npix 120 and nguard 5 it found none of the spikes up to 40σ, where pooled clipping found 0.38 to 0.99 of them. Per-bin clipping flagged fewer genuine pixels at high S/N (0.000 to 0.023 per clean spectrum, pooled 0.029 to 0.178, at npix 120 with 125 to 600 spectra per bin). rank=K clips when N_s − 1 < npix; noise-aware in-sample Z costs 0.5 to 0.7 s per spectrum and pass | per-bin clipping; `clip_model=` |
| A7 | Block thresholds from clean spectra, with bad pixels infilled and no Z-based detection; per bin at least the smallest m with m (100 − p) ≥ 100, with the percentile p read as the decimal `Fraction(str(p))` (1000 at 99.9) | paper l.510 and r26; repair step 2 infills bad pixels too; with fewer spectra the threshold is near the maximum; in float arithmetic 99.9 gives 1001, and under numpy 2 `repr` of a numpy scalar is not a decimal | step-1 detection first |
| A8 | Block statistic χ²_A/\|A\| | r26; ratios are unchanged | χ²_A |
| A9 | Blocks 150/75/20, last block flush with the end | r26 (D-54) | none |
| A10 | Exclusion compares with floor(minfrac·npix) | r26 reproducibility (D-57) | no floor |
| A11 | Pixelwise conditions on interpolated bad pixels; infill never does | 2022 semantics and one shared W, though Z is miscalibrated next to long bad runs (section 5.2); infill correctness | per-spectrum W |
| A12 | `Prediction.z` is NaN at bad pixels; `fit` and `repair` flag Z at every pixel of the interpolated data, as 2022 and Alg. 4 do for `fit` and r26 `gp_repair` does for `repair` | interpolated values are not data; NaN Z in `fit` would change 10, 7, 24, 2 and 20 entries of the five `covar_iter_mask` golden masks | 2022 Z |
| A13 | `inflate` only for dense full-rank, unweighted, mean-centred models, binned or not; one factor f_s(N_s, N_k, 1) | f_s is derived only there; that factor calibrates every fixed linear combination of the predicted pixels (section 4.3) | per-pixel and joint factors |
| A14 | `finite_sample_factor` has no ddof argument | every covariance for which the package reports f_s has divisor N_s − 1 (S11.3), so a ddof argument could only disagree with the model | (N_s − ddof)/(N_s − 1) factor |
| A15 | Constant centre: divisor N_s, no mean estimated | unbiased second moment about c; the in-sample mean Z²/N_* is then exactly 1 | N_s − 1 |
| A16 | Draws from per-chunk `standard_normal((n, npix))` in row order, with no other use of rng in that call | the normals are bit-identical for any chunking or grouping; the draws agree to rounding | per-group draws |
| A17 | Pattern grouping by a void view of packed masks | 15 to 200 times faster than `unique(axis=0)` (section 12) | `unique(axis=0)` |
| A18 | The cache and `__reduce__` of section 3.3; fixed chunk size | no global state; repeated calls stay cheap; the cache is bounded, and copies never carry a stale cache | `lru_cache`; `chunk=` |
| A19 | `nsigma` is a 1-D sequence or array, () for no clipping, never a scalar; integer arguments accept `numbers.Integral` except bool; `nguard` is any integer ≥ 0 | a scalar is ambiguous (one pass, or one threshold per pass); numpy integers and the 2022 ndarray nsigma must work; nguard ≥ npix − 1 is the prior | scalar nsigma; nguard ≤ npix − 2 |
| A20 | The 2022 asserts and the new legacy argument checks raise `_LegacyError(AssertionError, ValueError)`, whose message names the argument and contains the 2022 message; integer conversion raises TypeError naming the argument. **BC**: `except ValueError` also catches them, and legacy also checks under `python -O` | old `except` clauses keep working; legacy also checks under `-O` | plain AssertionError |
| A21 | `gspice.legacy` emits no warning of its own; numpy warnings pass through; no D-16 nguard argument | it is the reproducibility layer, frozen | warn on every call |
| A22 | Noise models form P_s per spectrum (the stated exception to the nspec rule) | one code path; gated optimizations in section 12 | residual route now |
| A23 | Positive `weights` apply to the final fit only, and clipping uses unit weights; weight 0 excludes a spectrum | weighted clipping found 0 of 40 injected 30σ spikes in up-weighted spectra, unit weights 40 of 40 | every pass |
| A24 | The 2022 modules move to `gspice.legacy.*`, `gspiceMain` is deleted, and the old module paths raise (section 7.2). **BC** (API) | no file outside `tests/golden` imports the modules (the 2022 tests and notebooks use the top-level names), PR 2.1 retargets the golden fixture, and keeping the paths needs a `sys.modules` alias per module | aliases, or stub modules that warn |
| A25 | The top-level 2022 names warn with a `DeprecationWarning` (stacklevel 2) until 3.0.0 and stay out of `__all__`, so `from gspice import *` no longer binds them. **BC** (API) | numpy keeps deprecated names out of `__all__`. With them in it, a star import of the new API warned 11 times, 22 times under the "always" filter, and failed under warnings-as-errors, and no file in the repository needs the star import to bind a 2022 name. stacklevel 2 shows the warning at the caller's line; 3.0.0 is the next major release (Q8) | 2022 names in `__all__`; FutureWarning |
| A26 | `precision` raises LinAlgError when the reciprocal condition number of S scaled to unit diagonal, which `dpocon` estimates in the 1-norm, is below 1e-13 | a numerically singular S otherwise passes silently (max\|PS − I\| = 0.06 to 0.52 at κ 1.5e16 to 3.2e16). At κ₂ about 1e13 and nguard 5, the precision route erred by 3.1e-6 to 4.3e-4 in variance and 1.3e-6σ to 1.8e-3σ in the mean, below κε = 2.2e-3. 1/rcond was 1.7 to 34 times κ₂ at npix 120 to 2400, so the check fires from κ₂ about 3e11 to 6e12; the golden covariances have scaled κ₂ up to 2.9e6. At N_s − 1 = npix = 2400 a sample covariance can fail the check (1/rcond 4.0e13 to 2.5e14 for a population κ of 1e6; at most 2.2e11 with 1% more spectra) | no check |
| A27 | Benchmarks in `benchmarks/run.py`, run by hand, with JSON output | no added dependency; it records the setup that CLAUDE.md requires | pytest-benchmark or asv |
| A28 | `woodbury` raises LinAlgError where κ_W > 1e13 (section 4.2) | κ_W bounds the condition number of C scaled to unit diagonal and max c_ii/d_i, which drive the error of P_ii = 1/d_i − (LLᵀ)_ii. In a sweep it fired on every case with an error above 2.4e-3 and accepted at most 6.5e-4; K = 50 models of simulated spectra gave κ_W 7.2e6 and 2.9e8. min d_i/c_ii alone missed spread models at κ₂ 1e13 to 1e15, where Woodbury erred by up to 0.59 in variance at \|O\| ≤ K | min d_i/c_ii; no check |
| A29 | ruff lints and formats `src/gspice/legacy/` from PR 2.4, which fixes what it flags without a behaviour change. Per-file ignores keep N999 for `specUtils.py` and `matrixUtils.py`, and a `# noqa: B008` on the signature of `covar_iter_mask` keeps its ndarray default, because these module names and that default are 2022 API (section 7.1) | PRs 2.3 and 2.5 to 2.7 add new code to legacy, which needs the checks of other code. On the 2022 modules, ruff 0.17.1 flags only unused imports and variables, import order, and these three; PR 2.4 checks that the formatted code gives every golden output bit for bit | skip legacy as vendored code; lint it without formatting |

## 11. Testing strategy

**Golden tests** (`tests/golden/`).

- PR 2.1 retargets the fixture: `importlib.import_module(module.replace("gspice.", "gspice.legacy.", 1))`. The manifest and the tolerance table do not change, and nothing is regenerated.
- A changed golden record is marked `xfail(strict=True, reason=<D-ID>)` in the pull request that changes it. `tests/legacy/test_behaviour_changes.py` shows the 2022 and the new result of each BC; without a manifest record, the 2022 result is a literal whose comment cites the 1fbab9f line or the run that produced it. `tests/conftest.py` puts `tests/golden` on `sys.path`.
- Anchors without a record, with the 2022 result as a literal from a cited run of 1fbab9f:
  - D-80: D1 with NaN or inf ivar at the masked pixels against legacy on the unmodified D1 input, bit for bit, also in float32 (2022: X all NaN); refscale on finite inputs against the 2022 expression `np.sqrt(np.sum(ivar*wt, axis=1)/np.sum(wt, axis=1))`, evaluated in the test.
  - D-81: E row 2 with NaN at its masked pixels against `maskinterp1.E.row2`; `get_chimask` on D1 with NaN flux at the masked pixels of row 1930 against the clean run (2022: ValueError); a ±inf good pixel and a −0.0 good value keep the 2022 result.
  - D-14: `covar_iter_mask` with mask=None against the run with the mask, bit for bit, on the input of `covar_iter_mask.D1.nsig10_5p5.maxbad8`, where maxbadpix drops 8 spectra.
  - Inherited Q1: D1 with row 11 good only at pixels 24 to 26 (ivar 0 at the newly masked pixels) and flux += 20/sqrt(ivar) at pixel 25, nsigma=[6] (2022: an all-NaN covariance); the same with maxbadpix 47 and row 3 given 48 bad pixels, so that row 11 is not row 11 of the kept subset; a kept row fully masked by its input mask (2022: a ValueError in the first pass).
  - D-01 and D-18: irange [10, 30] as array, list and tuple against the [10, 30, 0] outputs; irange [−1, 49, 0] on D1 raises (2022: a (2000, 51) result whose first column is 0); npix 0 raises (2022: UnboundLocalError from a print).
  - D-17 and the integer rule: nguard −1 on D1 (2022: pred = X to 3.8e-13); `np.uint8(5)` and `np.uint64(5)` equal nguard 5 (2022 under numpy 2: pixels 0 to 4 predicted from themselves, and TypeError); `np.int8(5)` on D2 equals nguard 5 (2022 under numpy 2: OverflowError); `np.True_` and `np.array([5])` raise TypeError naming nguard on every CI job (2022: nguard 1, and ValueError).
  - D-02: bruteforce on the 20 `gp_interp` golden cases at the table tolerance, in `tests/golden/test_golden.py`, which holds the table (2022: the crash of `bruteforce_unpack`).
  - A20: `covar` of a float32 input raises under `python -O`, in a subprocess (2022: a float32 covariance).
  - A24: each old import form of section 7.2, `gspiceMain` included, raises its error, and a 1fbab9f pickle of `standard_scale`, kept as a bytes literal, fails to load.

  PR 2.5 drops the RuntimeWarning filter of `standard_scale.E.with_fully_masked`.
- `tests/golden/test_new_api.py` checks, at the table tolerance where it is at most 1e-10: `scale` against `standard_scale`, except the two ivar0 cases with a mask and `E.with_fully_masked`; `pixelwise` on `Model(mean=zeros, cov=(golden cov,), n_train=(N,))` with ivar = 1 against the stored `gp_interp` outputs of D1 and D2, and on D3 within 1e-12 of the 60-digit reference, where the 2022 record is more than 1e-10 from it (measured 1.7e-13 against 7.6e-10 and 2.7e-10); `fit(cumulative=False)` against `covar_iter_mask`, masks exactly; `infill` on that model, with replace = ipred and ignore = ~(icond | ipred), against `gaussian_estimate` on D1 and D2.
- Its BC-new tests, on golden inputs (none are `slow`):
  - S11.1: default `fit` equals an Alg. 4 replay on legacy `get_chimask` and differs from the stored final masks in 157, 152, 106 and 8 entries on D1 to D4.
  - S11.2: the `ge_tie` input, with nsigma equal to |Z| from the private function that computes Z in `fit`: `fit` does not flag the pixel, and flags it with nsigma one ulp lower.
  - Q11: the two ivar0 cases of `scale` with a mask; `fit(cumulative=False)` against live legacy `covar_iter_mask` on D1 with `ivar_z` and maxbadpix 8 (1990 spectra used against 1992 kept).
  - Q15: integer flux and float32 flux.
  - Q1: `E.with_fully_masked`; `fit` on the inherited-Q1 input (2022: an all-NaN covariance; legacy raises; `fit` drops row 11 and fits 1999 spectra).
  - Q2: `fit` on D1[:50] raises ValueError (2022: LinAlgError).
  - S11.4: the D3 test of `pixelwise` above; `infill` on the input of `gaussian_estimate.D3.many21.guard5` within 1e-12 of the 60-digit reference, where the record errs by 6.4e-11 normwise and 3.1e-10 on the diagonal (precision route: 2.8e-14 and 8.8e-14).

**Oracle** (`tests/oracle.py`, about 40 lines).

- Reference: a 60-digit Cholesky solve of S_kk with the standard-library `decimal` module, for n ≤ 60, the same on every platform. It takes 5 to 30 ms per solve at n 30 to 60, so tests subsample pixels. Low-rank cases at κ ≥ 1e5 may use an exact `fractions` reference at npix ≤ 36 (0.8 s per pattern).
- Matrices: random symmetric positive definite (SPD) with log-spaced eigenvalues and κ = 1e2, 1e5 and 1e8; squared-exponential kernels of lengths 5 and 15 plus a nugget; low rank plus diagonal. Data are drawn from N(0, C).
- Paths: precision, with A26 raising at κ 1e14; Woodbury, with A28; guard weights (nguard 0, 1, 5, about npix/3 on the kernel of length 5 at npix ≤ 60, where the error relative to κε is largest, and ≥ npix − 1, the exact prior); pattern solve (means, variances, covariance, χ², draws through R); block χ²; noise-model C_s.
- Tolerances, with κ the condition number of the matrix scaled to unit diagonal: dense routes 10κε in |Δμ|/σ, |Δvar|/var and |Δχ²|/(χ² + |T|) (worst measured 0.98κε over 86 matrices); Woodbury routes 10ε·κ_W, and 1e-9 when |O| ≥ 2K and κ_W ≤ 1e8 (10κε failed 108 of 526 swept results; 10ε·κ_W none).

**Property tests** (hypothesis, at most 50 examples, npix ≤ 60, a profile with `derandomize=True`, `database=None` and `deadline=None`; `.hypothesis/` in `.gitignore`).

- C → αC leaves the conditional mean unchanged and multiplies the covariance by α.
- Conditional covariances are symmetric and positive semi-definite.
- Grouped infill equals one-spectrum-at-a-time infill to 10κε in |Δμ|/σ.
- `infill(replace={i}, ignore=window \ {i})` equals `pixelwise` at pixel i of a spectrum without bad pixels.
- `unscale(scale(f))` returns f at good pixels to 8ε·max(|f|, |refmean/a_s|, |center|); unreplaced pixels are bit-identical; inputs are unchanged; float32 equals its float64 cast; non-contiguous inputs work.
- The normals do not depend on the chunk size, bit for bit; the draws agree to 1e-12 relative.
- `save` then `load` gives bit-identical arrays and equal scalars, also in `LowRank` and `Blocks`.
- Edge cases: masks at the edges, one good pixel, nguard = 0, empty masks, masks over half the spectrum, G = all pixels, empty G, M or T.

**Exact identities** (fast, heavy-tailed data, `fit(nsigma=())` without bad pixels, κ ≤ 1e4; at κ 1.3e10 the deviation reached 8.9e-9). In-sample mean Z²/N_* = (N_s − 1)/N_s to 1e-12, per pixel for pixelwise and for one fixed infill pattern. It is exactly 1 for a constant centre (measured 1 ± 3e-14). The w-weighted mean of a weighted fit is 1 − V2/V1², or 1 for a constant centre.

**Statistical tests** (`slow`, seeded, inventory section 7 conditions), each in the pull request of the code it checks.

- Held-out ⟨Z²⟩/N_* = f_s within 3σ, with `simulate(gaussian=True)`, at (N_s, N_k, N_*) = (30, 10, 2), (200, 40, 10) and (200, 40, 1) pixelwise, also with ignored pixels (3.7). It averages over training sets, each with `fit(nsigma=())` and its own held-out spectra, and σ is the standard error across sets (section 5.3).
- Two-population mixture: each population's ⟨Z²⟩ matches tr(Ĉ⁻¹ E_p[r rᵀ]) from its own mean and covariance within 3σ, and the average matches f_s for N_s ≥ 50 N_k (measured 1.130 at N_s 200 and 1.004 at 2000, N_k 40) (4.4).
- `noise_audit`: with the reported σ 1/1.2 of the true σ on 200 pixels of the training and test spectra, ρ_i² − e_i equals k_i within 3σ, e_i from the noise-free flux (4.5).
- The noise-free estimator before the eigenvalue floor is unbiased with per-pixel masks, weights and per-spectrum ivar: every element within 4σ (5.3).

**End to end** (4.3, `slow`). `simulate(artifacts=0.4)`, then `fit`, `fit_blocks` and `repair`: completeness ≥ 98% for peaks above 25σ, false replacements per clean spectrum, repaired-pixel Z dispersion near f_s^½, and a larger one with `runlength=None`, with thresholds from the Monte Carlo error. As a check, not an assertion, since `simulate` is not the PHOENIX mock of the paper, the report lists each statistic next to the paper value: 2.8 replaced pixels per clean spectrum, 1.7% of clean spectra affected, Z dispersion 1.02 (f_s^½ = 1.04) and 1.29 with `runlength=None` (paper l.512-514); rms (repaired − truth)/σ 0.65, calibration 0.993 to 1.004 for 1 to 1024 pixels, and 50% completeness at 12σ for cosmic rays and at a 1.9-pixel shift (kickoff specification). A fast version checks that one strong cosmic ray and one bad column are replaced.

**Contracts.** One parametrized test per row of the section 8 table checks the error class and that the message names the argument. Each top-level 2022 name warns, naming its replacement and the caller's file, and the tests check that a warning is raised, not how many (the module `__getattr__` can run twice per access); `gspice.legacy` adds no warning. `from gspice import *` binds exactly `__all__`, with no 2022 name and none of the three 2022 modules, without a warning (2022: 11 functions and 3 modules). `progress` gets its stage name and ends with done == total. `finite_sample_factor(60000, 2400, 1)` equals 1.0851274682919 to 1e-12.

**Regressions.** Each fixed D-item gets a test named as a sentence, for example `test_fully_masked_spectrum_is_rejected`.

**CI budget.** The fast suite stays within the CLAUDE.md budget. `slow` tests run in the `slow` job of `ci.yml`, started by hand, and locally before a merge that touches their modules. From PR 2.2, CI checks mypy strict on `src/gspice` except `legacy/`, and core coverage ≥ 90%.

## 12. Performance plan and benchmarks

**Setup of every measured number.** Intel Xeon @ 2.10 GHz (4 cores, 16 GB, shared with other agents), Python 3.13.16, numpy 2.5.4 on OpenBLAS 0.3.34 SkylakeX, scipy 1.18.1 with LAPACK on its own OpenBLAS 0.3.31.dev, `OPENBLAS_NUM_THREADS=1`. Timings are the minimum of at least 5 repeats at load average 0.1 to 0.9, unless stated. Accuracy figures use 60-digit, long-double or exact rational references, some repeated with numpy 1.26.4 and scipy 1.11.4, and the goldens of 036aaf2. Covariance unless stated: squared-exponential kernel (length 15 pixels) plus rank 30 plus identity, npix 2400. The scripts are working files outside the repository; PRs 2.9 and 5.5 rebuild the measurements as `benchmarks/run.py` cases.

| Quantity | Measured |
|---|---|
| P by `dpotrf` + `dpotri` + mirror | 0.39 s (median 0.41 s); P is exactly symmetric, so its C-contiguous transpose avoids the 0.04 s of a copy |
| Guard weights, nguard 20 | 0.24 s; 0.10 s with row slices of a C-contiguous P |
| Product of 1e4 × 2400 by W | 1.57 s: 2·2400²·1e4/1.57 s = 73 GFLOP/s |
| Pixelwise chain (scale, product, Z, unscale) at 2e4 × 2400 | 0.39 ms per spectrum |
| Covariance X_cᵀX_c at 1e4 × 2400; `eigh`, top 50 of 2400 | 0.85 s (3 repeats); 1.24 s (2 repeats) |
| Infill per spectrum at large \|G\| | 1.5 to 1.6 ms at \|G\| 310 and 5.7 to 7.5 ms at 620; χ² only at 1036, 18 to 24 ms, of which `dpotrf` takes 13 ms at 1024 (load 6.4); about 50 ms per pattern at 1200, so 80 minutes for 1e5 unique patterns (computed) |
| Woodbury P, K = 50 | 41 ms; max\|PS − I\| 4.7e-13 to 1.0e-12 |
| Low rank, n 36 (K 4) and 60 (K 8), κ 9.7e7 to 1.8e8, x ~ N(0, C), reference from (V, Λ, d) | mean error with more than K pixels conditioned on: Woodbury 3.5e-13σ to 5.0e-10σ, dense inverse 1.9e-9σ to 1.4e-8σ; at \|G\| = npix − 1: 2.8e-8σ and 2.2e-9σ |
| Noise-model infill, full rank, \|G\| 30 | 0.44 to 0.50 s per spectrum through P_s; 0.18 s through chol(C_s) and two triangular solves; forming C_s takes 0.01 s |
| Grouping 1e5 rows of 4800 bits | void view 0.17 to 0.19 s; `unique(axis=0)` 2.8 to 3.2 s on packed bytes, 37.5 s on bool (numpy 1.26.4 and 2.5.4) |
| S/N of 1e5 spectra | 4.2 s with a sort-based median; 12.8 s with `np.nanmedian` |

**Targets.**

| Target | Design | Figure |
|---|---|---|
| Pixelwise, 1e5 × 2400, < 60 s single core | P and W once, one chunked product | 52 to 54 s end to end with float32 inputs, 7.9 s of it for the first write to the outputs (measured) |
| Infill about 0.5 ms per spectrum, 30-pixel masks | per-pattern Cholesky of P_GG on a C-contiguous P | 0.10 ms (median 0.18 ms, 3 repeats), 0.58 ms on the Fortran-ordered P of `dpotri`; 3.6e-14σ in mean and 7e-15 in var from a direct solve (measured) |
| No per-pixel work that scales with nspec | W and block factors per model and bin | holds except noise models (A22) |
| Noise-aware pixelwise, per spectrum | C_s, P_s and W_s per spectrum | 0.01 + 0.39 + 0.10 = 0.50 s at full rank, 0.01 + 0.04 + 0.10 = 0.15 s at K = 50: 14 h and 4.2 h per 1e5 spectra (computed) |
| Against 2022 at 4000 × 2400 | as above | 16.1 to 16.4 s against 1.75 to 2.06 s for P, W, product and Z (measured; D-65) |
| Block scan per spectrum | 31 blocks, \|G\| 190, R per block | 0.79 ms for 31·2·2400·190 = 2.8e7 GEMM flops (measured) |
| Repair per spectrum | two passes, scans, three infills | 2 × 0.39 + 0.79 + 3 × (0.10 to 0.18) = 1.9 to 2.1 ms without flagged blocks or exclusions, and 1.5 ms more after one exclusion (computed) |
| `fit`, 60,000 × 2400, three passes | one covariance and one pixelwise pass per clipping pass | 3 × (5.1 + 0.49 + 23.4) + 5.1 = 92 s, with 6 × 0.85 = 5.1 s per covariance and 60,000 × 0.39 ms = 23.4 s per pass (computed); `rank=K` adds 1.24 s per pass |

**Benchmarks.** `benchmarks/run.py` (A27) writes JSON with the commit, machine, thread settings, library versions and load average, and reports the minimum and median of at least 5 repeats. PR 2.9 adds cases 1, 2 and 7, and PR 5.5 the rest:

1. pixelwise at {4000, 20,000, 100,000} × 2400;
2. infill with \|G\| ∈ {11, 30, 100, 1200}, unique and shared patterns;
3. block scan;
4. repair per spectrum;
5. `fit` at 20,000 × 2400;
6. noise models, pixelwise and infill, full rank and K = 50;
7. legacy `gp_interp`, the 2022 code, at 4000 × 2400;
8. the Schur route (legacy `submatrix_inv_mult` per pixel set) against `solve_pattern` at \|G\| 11, 41 and 190.

**Gated optimizations.** Each one is its own pull request, and only if the benchmark shows a gain. (a) and (c) read x_g, where an outlier can cancel, so they must also pass the oracle with 1e4σ outliers in g.

- (a) Residual pixelwise for one-spectrum groups: x̂_i = x_i − (P_gg⁻¹ (Px)_g)_i, which avoids building W per spectrum for noise models.
- (b) Infill for low-rank noise models from V and d + ν_s without forming P_s.
- (c) Block χ² from r_G = P_GG⁻¹ (Px)_G after one product per spectrum.
- (d) Batched small-pattern infill.
- (e) Full-rank noise-model infill from chol(C_s) and P_{:,G} = L⁻ᵀ(L⁻¹ E_G), without `dpotri`.
- (f) Blockwise prediction on spectra with their own bad pixels: one factor for the shared part of G and a rank-|E_s| update for the extra bad pixels E_s of each spectrum.
- (g) No copy of X in `solve_pattern`: a sum of products over the runs of pixels outside G, or a zeroing of G in the rows that the caller already gathered. PR 2.9 (#31) measured the sum over runs faster with one shared pattern and not faster with one spectrum per pattern.

## 13. Packaging

```toml
[build-system]
requires = ["hatchling>=1.27"]
build-backend = "hatchling.build"

[project]
name = "gspice"
dynamic = ["version"]
requires-python = ">=3.11"
dependencies = ["numpy>=1.26", "scipy>=1.11"]   # the oldest stack that CI tests
license = "GPL-3.0-only"                  # provisional (S11.8)
authors = [{name = "Douglas P. Finkbeiner"}, {name = "Joshua S. Speagle"}, {name = "Tanveer Karim"}]

[dependency-groups]
test = ["pytest>=8", "pytest-cov>=5", "hypothesis>=6"]
dev = [{include-group = "test"}, "ruff==0.17.1", "mypy==2.4.0", "scipy-stubs==1.18.1.1; python_version >= '3.12'", "pre-commit"]

[tool.hatch.version]
path = "src/gspice/__init__.py"           # __version__ = "2.0.0.dev0" (Q8)

```

- The pytest and ruff settings of `pytest.ini` and `ruff.toml` (#9) move into `pyproject.toml`.
- ruff (lint and format) checks `tests` from #9. PR 2.1 extends it to `src` except `src/gspice/legacy/`, and updates `ci.yml` and the lint command of CLAUDE.md; PR 2.4 adds `src/gspice/legacy/` (A29), and PR 2.9 adds `benchmarks/`. mypy, with `scipy-stubs` because scipy ships no type information, coverage and pre-commit run from PR 2.2; mypy runs in the lint job on Python 3.13. The `dev` group replaces `requirements-dev.txt`; PR 2.1 removes the unused astropy import (D-07), so no group needs astropy. `pip install --group` needs pip 25.1, so CI upgrades pip first.
- CI (#9): Linux on Python 3.10 to 3.15; macOS and Windows on 3.10 and 3.15; an oldest-stack job on Python 3.11 with numpy 1.26 and scipy 1.11. PR 2.1 drops 3.10 (S11.10): Linux then runs 3.11 to 3.15, and macOS and Windows 3.11 and 3.15. Phase 5 adds a wheel and sdist build check.
- No h5py, matplotlib or tqdm, not even as extras: `save` writes .npz, `progress` is a callback, and the package draws no plots. A `docs` group arrives with the docs phase.
- Deleted: `setup.py`, `dist/`, `gspiceMain.py`, the 2022 tests (D-08, D-09, D-10).

## 14. Phase plan

Each line is one issue, one branch and one pull request with a single goal; line 5.6 is one such line per gated optimization of section 12. Brackets give dependencies. "BC" marks an item with a behaviour change or a BC-new item, which adds its CHANGELOG line and its old-and-new test (section 10). Every pull request keeps the golden suite green. A phase ends when its items are merged and its "Done when" condition holds.

**Phase 2: package and legacy.** Done when every golden test passes through `gspice.legacy`, with only the records of section 7.1 marked xfail, and `benchmarks/run.py` has cases 1, 2 and 7.

1. Src layout and `pyproject.toml`; `git mv` the 2022 modules into `src/gspice/legacy/` unchanged, then remove their unused astropy import (D-07); top-level re-exports; retarget the golden fixture; delete `setup.py`, `dist/`, `gspiceMain.py` and the 2022 tests (D-08, D-09, D-10, D-38); drop Python 3.10; CHANGELOG with the module-path change (A24; BC). [golden PR #9]
2. mypy, coverage and pre-commit in the commands and CI. [2.1]
3. Legacy checks raise `_LegacyError`: the 2022 asserts (D-22) and new checks of irange and nguard (D-01, D-17, D-18, A20; BC). [2.1]
4. Legacy hygiene without BC: prints, the LAPACK message of a 0 × 0 `cholesky_inv`, dead code, docstrings, ruff on legacy (D-05, D-06, D-11, D-12, D-38, A29). [2.3]
5. Legacy input handling: fully masked spectra, non-finite values at masked pixels, mask=None (D-13, D-80, D-81, D-14; BC). [2.4]
6. Legacy copies when nothing is removed (D-25; BC). [2.4]
7. Legacy brute-force `gp_interp` works (D-02; BC). [2.4]
8. `_linalg`: precision, guard weights, pattern solve, grouping; `tests/oracle.py` and oracle tests. [2.1, 2.2]
9. `benchmarks/run.py` with cases 1, 2 and 7, on the `_linalg` primitives until the public functions exist. [2.8]

**Phase 3: model and prediction.** Done when the equivalence, BC-new and exact-identity tests of section 11 pass and the top-level 2022 names warn.

1. `_spectra`: validation, interpolation, `scale`, `unscale`, `snr`; equivalence with `standard_scale`; BC-new tests of Q1, Q11 and Q15 (BC). [2.1]
2. `Model` (dense), `save`, `load`. [3.1, 2.8]
3. `pixelwise`, `Prediction`, chunking, cache, logging, progress; equivalence with `gp_interp`; the D3 test of S11.4 (BC); benchmark case 1 on `pixelwise`. [3.2]
4. `fit` (dense, cumulative and 2022 modes, centre, weights), with the dilation of `_spectra`; equivalence with `covar_iter_mask`; BC-new tests of S11.1, S11.2, Q2, and the `fit` parts of Q1 and Q11 (BC). [3.3]
5. `infill`, `Repaired` (replace, ignore, chi2, covariance); equivalence with `gaussian_estimate`; the D3 test of S11.4 (BC); benchmark case 2 on `infill`. [3.3]
6. Draws in `infill`. [3.5]
7. `finite_sample_factor`, `inflate`, exact in-sample identity tests, held-out f_s test. [3.4, 3.5, 3.8]
8. `simulate`. [3.1]
9. Deprecation warnings on the top-level 2022 names, which leave `__all__` (section 7.2, A25; BC); README. [3.4, 3.5]

**Phase 4: detection, repair and diagnostics.** Done when `repair` matches r26 as item 3 states and the `slow` suite passes.

1. `Blocks`, `fit_blocks`, block χ², with oracle tests. [3.5]
2. Extended-artifact exclusion with `count_on` (private). [3.1]
3. `repair` with the end-to-end tests. Against r26 `gp_repair` with `max_block_iter` = nblock, `count_on="all"`, `dilate=1` and ivar = 0 only at masked pixels, on test data where no |Z| lies within 1e-6 relative of a threshold (r26 computes its weights by the Schur route), the replaced and ignored sets agree exactly and flux to 1e-12 relative; only `Repaired.var` differs (Q14; r26 gives 0 outside M). [4.1, 4.2, 3.6, 3.7]
4. `Calibration` and `calibration`; mixture test. [3.7]
5. `noise_audit` and its `slow` test. [3.3, 3.8]

**Phase 5: covariance models, performance and packaging.** Done when `benchmarks/run.py` meets the targets of section 12 and CI checks the wheel and sdist.

1. Low rank (`rank=`, `LowRank`, Woodbury with A28), clipping included. [3.4]
2. S/N bins, with the clipping measurement of A6 and block thresholds. [3.4, 4.1]
3. Noise-aware model, full rank; `slow` test of the noise-free estimator. [3.4]
4. Noise-aware model, rank K. [5.1, 5.3]
5. `benchmarks/run.py` cases 3 to 6 and 8, and a report against 2022. [5.4, 4.3]
6. Each gated optimization (x) of section 12, in its own pull request 5.6x, only if 5.5 shows a gain. [5.5]
7. Wheel and sdist check. [5.1 to 5.4]

**Later phases**

- Docs: Sphinx with the theme chosen in the docs phase (S11.9), built in CI with `sphinx-build -W`; background, glossary, SVG schematics in currentColor, figures built from data.
- Tutorials and examples, after the docs.
- Release preparation: CITATION.cff, .zenodo.json, version, licence wording, the Python floor against end-of-life dates; the lead approves before any tag, release or upload.
- IDL (`pro/`) and Julia (`julia/`) updated and modernized, last.

## 15. Worked examples

```python
import numpy as np
import gspice

rng = np.random.default_rng(2026)
train = gspice.simulate(20_000, 400, rng=rng, artifacts=0.3)
clean = gspice.simulate(5_000, 400, rng=rng)                 # held out, artifact free
test = gspice.simulate(5_000, 400, rng=rng, artifacts=0.3)

# Fit with iterative clipping (Alg. 4, cumulative, |Z| > 20, 8, 6); N_s is stored.
model, train_mask = gspice.fit(train.flux, train.ivar)
model = gspice.fit_blocks(model, clean.flux, clean.ivar)
fs = gspice.finite_sample_factor(model.n_train[0], 400 - 41, 1)

# Detect outliers with pixelwise Z (NaN at bad pixels).
pred = gspice.pixelwise(model, test.flux, test.ivar, inflate=True)
suspect = np.abs(pred.z) > 6

# Repair with the validated procedure; covariance and draws are opt-in.
rep = gspice.repair(model, test.flux, test.ivar, covariance=True, rng=rng)
pixels, cov = rep.covariance(0)                             # flux^2, spectrum 0
flux, sigma = rep.flux, np.sqrt(rep.var)

# Detect with one model, infill with another.
lowrank, _ = gspice.fit(train.flux, train.ivar, rank=50)
refill = gspice.infill(lowrank, test.flux, test.ivar, replace=rep.replaced, ignore=rep.ignored)

# Predict the red half from the blue half beyond a 20-pixel guard.
j = np.arange(400)
red = j >= 200
block = gspice.infill(model, test.flux, test.ivar, replace=red,
                      ignore=(j >= 180) & ~red, covariance=True)

# Calibration on held-out spectra, overall and for one population.
cal = gspice.calibration(model, clean.flux, clean.ivar, sizes=(1, 16, 64), rng=rng)
mean, sem = cal.ratio()
giants, _ = cal.ratio(where=clean.labels == 1)
tails = cal.tail(0.99)
expected, observed = cal.qq(1)
rho = gspice.noise_audit(model, clean.flux, clean.ivar)    # sqrt(k + e): above 1 even if ivar is right

# Other covariance models.
binned, _ = gspice.fit(train.flux, train.ivar, snr_bins=3)
noisy, _ = gspice.fit(train.flux, train.ivar, noise=True, rank=50)

# Save and load.
gspice.save(model, "model.npz")
model = gspice.load("model.npz")

# 2022 results, reproduced.
from gspice import legacy
cov2022, finalmask = legacy.covar_iter_mask(train.flux, train.ivar, train.ivar == 0)
```
