// The conditional Gaussian of the explainers: predict the masked pixels M of a spectrum x
// from its other pixels O, given the mean m and the covariance C of the spectra:
//   mu_M = m_M + C_MO C_OO^-1 (x_O - m_O),
//   sd_M = sqrt(diag(C_MM - C_MO C_OO^-1 C_OM)).
// It factors C_OO by Cholesky on each call. tests/gauss.test.mjs compares it with numpy
// and with gspice.infill.

// The lower Cholesky factor of the n x n matrix A, both row-major.
function cholesky(A: Float64Array, n: number): Float64Array {
  const L = new Float64Array(n * n)
  for (let j = 0; j < n; j++) {
    let d = A[j * n + j]
    for (let k = 0; k < j; k++) d -= L[j * n + k] ** 2
    if (d <= 0) throw new Error('the covariance is not positive definite')
    const ljj = Math.sqrt(d)
    L[j * n + j] = ljj
    for (let i = j + 1; i < n; i++) {
      let s = A[i * n + j]
      for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k]
      L[i * n + j] = s / ljj
    }
  }
  return L
}

// Solves L y = b in place.
function forward(L: Float64Array, n: number, b: Float64Array): Float64Array {
  for (let i = 0; i < n; i++) {
    let s = b[i]
    for (let k = 0; k < i; k++) s -= L[i * n + k] * b[k]
    b[i] = s / L[i * n + i]
  }
  return b
}

// Solves L^T y = b in place.
function backward(L: Float64Array, n: number, b: Float64Array): Float64Array {
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i]
    for (let k = i + 1; k < n; k++) s -= L[k * n + i] * b[k]
    b[i] = s / L[i * n + i]
  }
  return b
}

// mean and x have length npix, cov is npix rows of npix, and M holds the pixels start to
// start + width - 1. Returns mu and sd of length npix, NaN outside M. O must not be empty.
export function condition(
  mean: readonly number[],
  cov: readonly (readonly number[])[],
  x: readonly number[],
  start: number,
  width: number,
) {
  const npix = mean.length
  const M = Array.from({ length: width }, (_, k) => start + k)
  const O = Array.from({ length: npix - width }, (_, k) => (k < start ? k : k + width))
  const mu = new Float64Array(npix).fill(NaN)
  const sd = new Float64Array(npix).fill(NaN)
  const n = O.length
  const A = new Float64Array(n * n)
  for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) A[a * n + b] = cov[O[a]][O[b]]
  const L = cholesky(A, n)
  // w = C_OO^-1 (x_O - m_O)
  const w = backward(L, n, forward(L, n, Float64Array.from(O, (j) => x[j] - mean[j])))
  for (const i of M) {
    // v = L^-1 C_Oi, so that C_iO C_OO^-1 C_Oi = |v|^2.
    const v = forward(L, n, Float64Array.from(O, (j) => cov[i][j]))
    let s = mean[i]
    let q = 0
    for (let a = 0; a < n; a++) {
      s += cov[i][O[a]] * w[a]
      q += v[a] * v[a]
    }
    mu[i] = s
    // Rounding can make the variance slightly negative where it is near 0.
    sd[i] = Math.sqrt(Math.max(cov[i][i] - q, 0))
  }
  return { mu, sd }
}
