// Compares the maths of the covariance explorer, the two-pixel explainer and the guard
// window explainer with numpy and gspice on the stored data. docs/scripts/explainers.py
// writes the files.
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'

import { withDefect } from '../src/lib/defect.ts'
import { conditionPair, ellipse, guardWindows, pixelwise, precision } from '../src/lib/gauss.ts'
import { heatSteps } from '../src/lib/heatmap.ts'

const read = (name) => JSON.parse(readFileSync(new URL(`../src/data/${name}`, import.meta.url)))
const data = read('explainer.json')
const guard = read('guard.json')
const close = (actual, expected, tolerance, what) =>
  assert.ok(Math.abs(actual - expected) < tolerance, `${what}: ${actual} against ${expected}`)

test('each cell of the heat map takes the step that numpy gives its correlation', () => {
  assert.deepEqual(heatSteps(data.cov), read('covariance-check.json').steps)
})

test('the ellipse at k standard deviations lies at Mahalanobis distance k', () => {
  // A correlation of -0.98, as the most anti-correlated pair of the model; the
  // distance of each point carries rounding near 1e-16 times the condition number of
  // the 2 x 2 covariance, about 100.
  const [cxx, cxy, cyy] = [4, -0.98 * 2 * 0.5, 0.25]
  const det = cxx * cyy - cxy * cxy
  for (const k of [1, 2]) {
    for (const [x, y] of ellipse(1, -1, cxx, cxy, cyy, k)) {
      const [u, v] = [x - 1, y + 1]
      close(Math.sqrt((cyy * u * u - 2 * cxy * u * v + cxx * v * v) / det), k, 1e-12, 'distance')
    }
  }
})

test('the ellipse of a pixel with itself is a line of finite points', () => {
  // The covariance explorer draws it on the diagonal of the map, where cyy - l21^2
  // rounds to a value that can be slightly negative.
  // Pixel 2 is one of the 42 pixels of the model where it is negative.
  const v = data.cov[2][2]
  for (const point of ellipse(1, 1, v, v, v, 2)) assert.ok(point.every(Number.isFinite))
})

// gspice.infill conditions in scaled units through the precision matrix of the full
// model, and the explainer conditions on the 2 x 2 model of the pair; both are exact
// for a marginal of a Gaussian. As in gauss.test.mjs, the covariance has a condition
// number near 1e3, so rounding stays near 1e3 times the float64 epsilon, or 2e-13; the
// differences measured 3e-15 or less.
test('the two-pixel prediction matches gspice.infill from the fixed pixel alone', () => {
  for (const { pair, value, mu, sd } of read('two-pixel-check.json').cases) {
    const [i, j] = pair
    const result = conditionPair(data.mean, data.cov, i, j, value)
    close(result.mu, mu, 1e-12, `mean of pixel ${i} given pixel ${j}`)
    close(result.sd, sd, 1e-12, `sd of pixel ${i} given pixel ${j}`)
  }
})

// gspice.pixelwise computes the same means and variances from its own precision matrix,
// in scaled units, by another route where 2 nguard + 1 <= npix: rounding near 2e-13, as
// above. The z-scores divide by sd near 0.02, which scales it by 50, to 1e-11. The
// differences measured 3e-14 for the mean, 2e-16 for sd and 2e-12 for z.
const P = precision(data.cov)
for (const { nguard, width, mu, sd, z } of read('guard-check.json').cases) {
  test(`the guard-window prediction matches gspice.pixelwise at nguard ${nguard} and width ${width}`, () => {
    const x = withDefect(data.examples[guard.spectrum].flux, guard.start, width, guard.height)
    const result = pixelwise(data.mean, P, guardWindows(P, nguard), x)
    for (let i = 0; i < data.npix; i++) {
      close(result.mu[i], mu[i], 1e-12, `mean at pixel ${i}`)
      close(result.sd[i], sd[i], 1e-12, `sd at pixel ${i}`)
      close(result.z[i], z[i], 1e-10, `z at pixel ${i}`)
    }
  })
}
