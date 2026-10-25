// Compares the conditioning of the explainers with numpy and with gspice.infill on the
// stored data. docs/scripts/explainers.py writes both files.
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'

import { condition } from '../src/lib/gauss.ts'

const read = (name) => JSON.parse(readFileSync(new URL(`../src/data/${name}`, import.meta.url)))
const data = read('explainer.json')
const { cases } = read('explainer-check.json')

// numpy solves the same system. gspice.infill conditions in scaled units through the
// precision matrix, which adds the rounding of the scaling and of that matrix. The
// covariance has a condition number near 1e3, so rounding errors stay near 1e3 times the
// float64 epsilon of 2.2e-16, or 2e-13; both differences measured 2e-15 or less.
const tolerance = 1e-12
for (const reference of ['numpy', 'infill']) {
  for (const { spectrum, start, width, ...expected } of cases) {
    test(`the prediction matches ${reference} for pixels ${start} to ${start + width - 1} of spectrum ${spectrum}`, () => {
      const { mu, sd } = condition(data.mean, data.cov, data.examples[spectrum].flux, start, width)
      for (let k = 0; k < width; k++) {
        assert.ok(Math.abs(mu[start + k] - expected[reference].mu[k]) < tolerance, `mean at pixel ${start + k}`)
        assert.ok(Math.abs(sd[start + k] - expected[reference].sd[k]) < tolerance, `sd at pixel ${start + k}`)
      }
    })
  }
}
