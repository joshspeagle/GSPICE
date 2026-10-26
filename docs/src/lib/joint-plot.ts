// A joint plot of two pixels of the model of docs/scripts/explainers.py, pixel across
// against pixel up, on a square of 100 x 100 SVG units: its grid, the 1σ and 2σ
// ellipses of the model and its tick labels. The covariance explorer and the two-pixel
// explainer draw on it.
import data from '../data/explainer.json'
import { ellipse } from './gauss.ts'
import { gridPath, line, signed, tickLabels, xScale, yScale, type Scale } from './spectrum-plot.ts'

const { mean, cov } = data

// The standard deviation of pixel i in the model.
export const sd = (i: number) => Math.sqrt(cov[i][i])

export function jointPlot(across: number, up: number, sx: Scale, sy: Scale) {
  const X = xScale(sx)
  const Y = yScale(sy)
  const point = ([u, v]: readonly [number, number], m: number) => `${m ? 'L' : 'M'}${X(u).toFixed(2)},${Y(v).toFixed(2)}`
  const contours = [1, 2]
    .map((k) => ellipse(mean[across], mean[up], cov[across][across], cov[across][up], cov[up][up], k).map(point).join(''))
    .join('')
  return {
    X,
    Y,
    grid: line('grid', sx.ticks.map((v) => `M${X(v).toFixed(2)},0V100`).join('')) + gridPath(sy.ticks, Y, 100),
    contours: line('ref', contours),
    labels: tickLabels(
      sy.ticks.map((v) => [Y(v), signed(v, sy.digits)]),
      sx.ticks.map((v) => [X(v), signed(v, sx.digits)]),
    ),
  }
}
