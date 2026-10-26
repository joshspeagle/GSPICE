// The heat map of a covariance as docs/brand.md draws it: its correlation, cell for
// cell, in the five steps of the ramp at -1, -0.5, 0, 0.5 and 1. Each cell takes the
// nearest step. The SVG has one path per step, of the runs of cells of that step in each
// row, and spans the npix x npix pixels in data units, row 0 at the top. The pixel labels
// are HTML, on both axes.

import { legend, pixelTicks, tickLabels } from './spectrum-plot.ts'

// The edges between the steps, as HEAT_EDGES of docs/scripts/explainers.py.
const EDGES = [-0.75, -0.25, 0.25, 0.75]

export function correlation(cov: readonly (readonly number[])[], i: number, j: number): number {
  return cov[i][j] / Math.sqrt(cov[i][i] * cov[j][j])
}

// The step of each cell, from 0 to 4, as rows of digits.
export function heatSteps(cov: readonly (readonly number[])[]): string[] {
  return cov.map((row, i) =>
    row.map((_, j) => EDGES.filter((edge) => edge <= correlation(cov, i, j)).length).join(''),
  )
}

export function heatmapMarkup(cov: readonly (readonly number[])[]): string {
  const npix = cov.length
  const paths = ['', '', '', '', '']
  heatSteps(cov).forEach((row, i) => {
    let start = 0
    for (let j = 1; j <= npix; j++) {
      if (j < npix && row[j] === row[start]) continue
      paths[+row[start]] += `M${start},${i}h${j - start}v1h${start - j}z`
      start = j
    }
  })
  const ticks = pixelTicks(npix).map((i) => [(100 * i) / npix, String(i)] as const)
  return (
    `<svg class="gs-heat" viewBox="0 0 ${npix} ${npix}" preserveAspectRatio="none" aria-hidden="true">` +
    paths.map((d, k) => `<path class="h${k}" d="${d}"/>`).join('') +
    '</svg>' +
    tickLabels(ticks, ticks)
  )
}

// The legend of the five steps.
export const heatLegend = legend(['−1', '−0.5', '0', '0.5', '1'].map((text, k) => [`h${k}`, text] as const))
