// The conditional Gaussian of the explainers: predict the masked pixels M of a spectrum x
// from its other pixels O, given the mean m and the covariance C of the spectra:
//   mu_M = m_M + C_MO C_OO^-1 (x_O - m_O),
//   sd_M = sqrt(diag(C_MM - C_MO C_OO^-1 C_OM)).
// condition factors C_OO by Cholesky on each call. pixelwise predicts each pixel from
// the pixels beyond its guard window, as gspice.pixelwise does. tests/gauss.test.mjs and
// tests/explainers.test.mjs compare them with numpy, gspice.infill and gspice.pixelwise.

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

// The precision matrix P = C^-1 of the npix x npix covariance cov, row-major.
export function precision(cov: readonly (readonly number[])[]): Float64Array {
  const n = cov.length
  const L = cholesky(Float64Array.from(cov.flat()), n)
  const P = new Float64Array(n * n)
  for (let k = 0; k < n; k++) {
    const e = new Float64Array(n)
    e[k] = 1
    // P is symmetric, so its column k is its row k.
    P.set(backward(L, n, forward(L, n, e)), k * n)
  }
  return P
}

// The guard window g = {j : |j - i| <= nguard} of each pixel i, clipped at the ends:
// its first pixel, the Cholesky factor of P_gg, and sd_i = sqrt((P_gg^-1)_ii), which
// depends only on the window.
export interface Window {
  first: number
  size: number
  L: Float64Array
  sd: number
}

export function guardWindows(P: Float64Array, nguard: number): Window[] {
  const npix = Math.sqrt(P.length)
  return Array.from({ length: npix }, (_, i) => {
    const first = Math.max(0, i - nguard)
    const size = Math.min(npix, i + nguard + 1) - first
    const A = new Float64Array(size * size)
    for (let a = 0; a < size; a++) for (let b = 0; b < size; b++) A[a * size + b] = P[(first + a) * npix + first + b]
    const L = cholesky(A, size)
    // (P_gg^-1)_kk = |L^-1 e_k|^2.
    const e = new Float64Array(size)
    e[i - first] = 1
    return { first, size, L, sd: Math.sqrt(forward(L, size, e).reduce((sum, v) => sum + v * v, 0)) }
  })
}

// The prediction of each pixel i of x from the pixels O outside its guard window g,
// with the windows of guardWindows (paper Alg. 3):
//   mu_g = x_g - P_gg^-1 (P (x - m))_g,  sd_i^2 = (P_gg^-1)_ii,
// equal to the conditional mean and variance of x_g given x_O, and z_i = (x_i - mu_i) /
// sd_i. Returns mu, sd and z of length npix. Every window must leave O not empty.
export function pixelwise(mean: readonly number[], P: Float64Array, windows: readonly Window[], x: readonly number[]) {
  const npix = mean.length
  const r = new Float64Array(npix)
  for (let i = 0; i < npix; i++) {
    let s = 0
    for (let j = 0; j < npix; j++) s += P[i * npix + j] * (x[j] - mean[j])
    r[i] = s
  }
  const mu = Float64Array.from(windows, ({ first, size, L }, i) => {
    const y = backward(L, size, forward(L, size, r.slice(first, first + size)))
    return x[i] - y[i - first]
  })
  const sd = Float64Array.from(windows, (window) => window.sd)
  return { mu, sd, z: Float64Array.from(x, (v, i) => (v - mu[i]) / sd[i]) }
}

// The conditional mean and standard deviation of pixel i given pixel j at value, from
// the 2 x 2 model of the two pixels.
export function conditionPair(
  mean: readonly number[],
  cov: readonly (readonly number[])[],
  i: number,
  j: number,
  value: number,
) {
  const pair = [
    [cov[i][i], cov[i][j]],
    [cov[j][i], cov[j][j]],
  ]
  const { mu, sd } = condition([mean[i], mean[j]], pair, [NaN, value], 0, 1)
  return { mu: mu[0], sd: sd[0] }
}

// n + 1 points of the ellipse at k standard deviations of a Gaussian with the mean
// (mx, my) and the covariance [[cxx, cxy], [cxy, cyy]]: m + k L (cos t, sin t), with L
// its Cholesky factor. The last point repeats the first. For a correlation of ±1, as
// of a pixel with itself, the ellipse is a line.
export function ellipse(mx: number, my: number, cxx: number, cxy: number, cyy: number, k: number, n = 72) {
  const l11 = Math.sqrt(cxx)
  const l21 = cxy / l11
  // Rounding can make l22^2 slightly negative where it is near 0.
  const l22 = Math.sqrt(Math.max(cyy - l21 * l21, 0))
  return Array.from({ length: n + 1 }, (_, m) => {
    const t = (2 * Math.PI * m) / n
    return [mx + k * l11 * Math.cos(t), my + k * (l21 * Math.cos(t) + l22 * Math.sin(t))] as const
  })
}
