// The markup of the two-pixel explainer, V-13. Pixel j of a pair is fixed at the value
// m_j + k sd_j, and the other pixel i follows its conditional distribution. The joint
// plot shows the 1σ and 2σ ellipses of the two pixels, the fixed value and the
// conditional mean with its ±1σ and ±2σ cells; the side plot shows the distribution of
// pixel i without pixel j and given it. docs/scripts/explainers.py picks the pairs.
import data from '../data/explainer.json'
import pairs from '../data/two-pixel.json'
import { clamp, countLabel } from './explainer.ts'
import { conditionPair } from './gauss.ts'
import { correlation } from './heatmap.ts'
import { jointPlot, sd } from './joint-plot.ts'
import { bandCell, line, plotSvg, scaleFor, signed, type Scale } from './spectrum-plot.ts'

const { mean, cov } = data
// The slider runs over k = -LIMIT to LIMIT.
export const LIMIT = 3
// The bins of the side plot.
const BINS = 60

export function pair(index: number) {
  const [i, j] = pairs.pairs[index]
  return { i, j, r: correlation(cov, i, j) }
}

export function value(index: number, k: number) {
  const { j } = pair(index)
  return mean[j] + k * sd(j)
}

function conditional(index: number, k: number) {
  const { i, j } = pair(index)
  return conditionPair(mean, cov, i, j, value(index, k))
}

function scales(index: number): [Scale, Scale] {
  const { i, j } = pair(index)
  // Pixel j across, over the slider and the 2σ ellipse; pixel i up, over the ±2σ cells of
  // every conditional mean, at most sqrt(9 + 4) < 3.7 sd_i from the mean.
  return [
    scaleFor([mean[j] - (LIMIT + 0.2) * sd(j), mean[j] + (LIMIT + 0.2) * sd(j)]),
    scaleFor([mean[i] - 3.7 * sd(i), mean[i] + 3.7 * sd(i)]),
  ]
}

// The k, on the 0.05 steps of the slider, at a fraction of the width of the joint plot.
export function kAt(index: number, fraction: number): number {
  const { j } = pair(index)
  const [sx] = scales(index)
  const k = (sx.low + fraction * (sx.high - sx.low) - mean[j]) / sd(j)
  return clamp(Math.round(k / 0.05) * 0.05, -LIMIT, LIMIT)
}

export function jointMarkup(index: number, k: number): string {
  const { i, j } = pair(index)
  const [sx, sy] = scales(index)
  const { X, Y, grid, contours, labels } = jointPlot(j, i, sx, sy)
  const { mu, sd: s } = conditional(index, k)
  const x = X(value(index, k))
  return (
    plotSvg(
      grid +
        contours +
        line('fix', `M${x.toFixed(2)},0V100`) +
        bandCell((x - 1.5).toFixed(2), mu, s, 2, Y, 3) +
        bandCell((x - 1.5).toFixed(2), mu, s, 1, Y, 3) +
        line('pred', `M${(x - 3).toFixed(2)},${Y(mu).toFixed(2)}H${(x + 3).toFixed(2)}`),
      100,
    ) +
    labels +
    `<span class="xname">pixel ${j}, fixed</span><span class="yname">pixel ${i}</span>`
  )
}

export function sideMarkup(index: number, k: number): string {
  const { i } = pair(index)
  const [, sy] = scales(index)
  const { mu, sd: s } = conditional(index, k)
  const pdf = (v: number, m: number, sigma: number) => Math.exp(-0.5 * ((v - m) / sigma) ** 2) / sigma
  // The densities at the centre of each bin, from the top; the conditional has the
  // higher peak, 1 / s.
  const centre = (b: number) => sy.high - ((b + 0.5) * (sy.high - sy.low)) / BINS
  const path = (m: number, sigma: number) =>
    Array.from({ length: BINS }, (_, b) => {
      const x = (90 * s * pdf(centre(b), m, sigma)).toFixed(2)
      const y = ((100 * (b + 1)) / BINS).toFixed(2)
      return b ? `H${x}V${y}` : `M${x},0V${y}`
    }).join('')
  return (
    plotSvg(line('ref', path(mean[i], sd(i))) + line('pred', path(mu, s)), 100)
  )
}

export function readout(index: number, k: number): string {
  const { i, j } = pair(index)
  const { mu, sd: s } = conditional(index, k)
  return (
    `Pixel ${j} fixed at ${signed(value(index, k), 3)}. Pixel ${i}: ${signed(mu, 3)} ± ${signed(s, 3)}, ` +
    `against ${signed(mean[i], 3)} ± ${signed(sd(i), 3)} without pixel ${j}.`
  )
}

export const pairLabel = (index: number) =>
  `${countLabel('Pair', index, pairs.pairs.length)}: correlation ${signed(pair(index).r, 2)}`
