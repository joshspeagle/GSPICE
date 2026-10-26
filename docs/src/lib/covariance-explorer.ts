// The markup of the covariance explorer, V-12: for a pixel pair (x, y), the scatter of
// the training spectra at the two pixels with the 1σ and 2σ ellipses of the model, and
// the mean spectrum with the two pixels marked. docs/scripts/explainers.py writes the
// data.
import scatter from '../data/covariance.json'
import data from '../data/explainer.json'
import { correlation } from './heatmap.ts'
import { jointPlot, sd } from './joint-plot.ts'
import { axesFor, axisLabels, gridPath, line, plotSvg, scaleFor, signed, steps, yScale } from './spectrum-plot.ts'

const { mean, cov, npix } = data
// The flux of each training spectrum at pixel i.
const fluxAt = (i: number) => scatter.spectra.map((spectrum) => mean[i] + spectrum[i] * scatter.unit)

export function scatterMarkup(x: number, y: number): string {
  const xs = fluxAt(x)
  const ys = fluxAt(y)
  // The axes hold every point and the 2σ ellipse.
  const sx = scaleFor([...xs, mean[x] - 2 * sd(x), mean[x] + 2 * sd(x)])
  const sy = scaleFor([...ys, mean[y] - 2 * sd(y), mean[y] + 2 * sd(y)])
  const { X, Y, grid, contours, labels } = jointPlot(x, y, sx, sy)
  const points = xs
    .map((u, m) => `<rect class="sample" x="${(X(u) - 0.8).toFixed(2)}" y="${(Y(ys[m]) - 0.8).toFixed(2)}" width="1.6" height="1.6"/>`)
    .join('')
  return (
    plotSvg(grid + points + contours, 100) +
    labels +
    `<span class="xname">pixel ${x}</span><span class="yname">pixel ${y}</span>`
  )
}

const axes = axesFor(mean, npix)

export function meanMarkup(x: number, y: number): string {
  const Y = yScale(axes)
  const left = (i: number) => ((100 * (i + 0.5)) / npix).toFixed(3)
  return (
    plotSvg(
      gridPath(axes.yticks, Y, npix) +
        [x, y].map((i) => `<rect class="pick" x="${i}" y="0" width="1" height="100"/>`).join('') +
        line('obs', steps(mean, 0, npix, Y)),
      npix,
    ) +
    axisLabels(axes, npix, Y) +
    // The label of the pixel on the right sits higher, so that two near pixels stay legible.
    [Math.min(x, y), Math.max(x, y)]
      .map((i, k) => `<span class="pick-label${k ? ' second' : ''}" style="left:${left(i)}%">${i}</span>`)
      .join('')
  )
}

export function readout(x: number, y: number): string {
  return `Pixels ${x} and ${y}: correlation ${signed(correlation(cov, x, y), 2)}.`
}
