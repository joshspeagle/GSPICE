// The guard window explainer, V-8: a defect adds a constant to `width` pixels of a
// held-out spectrum, and every pixel is predicted from the pixels farther than nguard
// from it, as gspice.pixelwise does. The pixel next to the defect is the one shown with
// its guard window. docs/scripts/explainers.py writes the data.
import data from '../data/explainer.json'
import guard from '../data/guard.json'
import { withDefect } from './defect.ts'
import { clamp } from './explainer.ts'
import { guardWindows, pixelwise, precision, type Window } from './gauss.ts'
import {
  axesFor,
  axisLabels,
  bandCell,
  gridPath,
  line,
  plotSvg,
  signed,
  steps,
  tickLabels,
  yScale,
} from './spectrum-plot.ts'

const { mean, npix } = data
// The axes hold every flux and prediction over all nguard and widths.
const axes = axesFor(guard.range, npix)
// z beyond ±ZMAX is drawn at ±ZMAX.
const ZMAX = 12
let P: Float64Array | undefined
// The guard windows of each nguard, from the first frame that uses it.
const windows = new Map<number, Window[]>()

// The state of one frame: the flux with the defect, the guard windows, and the
// prediction, sd and z of each pixel.
export type Frame = ReturnType<typeof frame>

export function frame(nguard: number, width: number) {
  P ??= precision(data.cov)
  if (!windows.has(nguard)) windows.set(nguard, guardWindows(P, nguard))
  const x = withDefect(data.examples[guard.spectrum].flux, guard.start, width, guard.height)
  const guards = windows.get(nguard)!
  return { nguard, width, x, guards, ...pixelwise(mean, P, guards, x) }
}

export function spectrumMarkup({ width, x, guards, mu, sd }: Frame): string {
  const y = yScale(axes)
  const p = guard.start + width
  const { first, size } = guards[p]
  return (
    plotSvg(
      gridPath(axes.yticks, y, npix) +
        `<rect class="guard" x="${first}" y="0" width="${size}" height="100"/>` +
        bandCell(p, mu[p], sd[p], 2, y) +
        bandCell(p, mu[p], sd[p], 1, y) +
        line('defect', `M${guard.start},1.5H${p}`) +
        line('truth', steps(data.examples[guard.spectrum].truth, 0, npix, y)) +
        line('obs', steps(x, 0, npix, y)) +
        line('pred', steps(mu, 0, npix, y)),
      npix,
    ) +
    `<span class="pick-label" style="left:${((100 * (guard.start + width / 2)) / npix).toFixed(3)}%">defect</span>` +
    axisLabels(axes, npix, y)
  )
}

export function zMarkup({ width, z }: Frame): string {
  const t = guard.threshold
  const y = (v: number) => (50 * (ZMAX - clamp(v, -ZMAX, ZMAX))) / ZMAX
  const bars = Array.from(z, (v, i) => {
    const cls = Math.abs(v) > t ? 'zbar flag' : 'zbar'
    return `<rect class="${cls}" x="${i}" y="${Math.min(y(v), 50).toFixed(2)}" width="1" height="${Math.abs(y(v) - 50).toFixed(2)}"/>`
  }).join('')
  return (
    plotSvg(
      line('grid', `M0,50H${npix}`) +
        bars +
        line('edge', `M0,${y(t)}H${npix}M0,${y(-t)}H${npix}`) +
        line('defect', `M${guard.start},1.5H${guard.start + width}`),
      npix,
    ) +
    tickLabels(
      [ZMAX - 2, t, 0, -t, 2 - ZMAX].map((v) => [y(v), signed(v, 0)]),
      [],
    ) +
    '<span class="yname">z</span>'
  )
}

export function readout({ nguard, width, z }: Frame): string {
  const t = guard.threshold
  const p = guard.start + width
  const others = z.filter((v, i) => Math.abs(v) > t && (i < guard.start || i >= p)).length
  const defect = z.slice(guard.start, p).filter((v) => Math.abs(v) > t).length
  return (
    `nguard ${nguard}, defect of ${width} pixel${width > 1 ? 's' : ''}. ` +
    `Pixel ${p}, next to the defect: z = ${signed(z[p], 1)}. ` +
    `|z| > ${t} at ${defect} of the ${width} defect pixels, and at ${others} of the ${npix - width} other pixels.`
  )
}
