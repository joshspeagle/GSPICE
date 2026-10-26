// The markup of the repair stepper, V-9: a spectrum of Find and repair artifacts, its
// replaced and ignored pixels and flagged blocks after each step of gspice.repair, and
// after step 5 the prediction at the replaced pixels. Step 0 is the input.
// docs/scripts/explainers.py takes the masks of each step from gspice.pixelwise and
// gspice.repair.
import data from '../data/repair-steps.json'
import { countLabel } from './explainer.ts'
import { axesFor, axisLabels, bandCell, gridPath, line, plotSvg, steps, yScale } from './spectrum-plot.ts'

const { npix, options, starts, width } = data
type Runs = readonly (readonly number[])[]
const none = { replaced: [], ignored: [], blocks: [] }
const at = (example: number, step: number) => (step ? data.examples[example].steps[Math.min(step, 4) - 1] : none)
const count = (runs: Runs) => runs.reduce((n, [a, b]) => n + b - a, 0)

export const STEPS = ['Input', '1 Pixelwise', '2 Blocks', '3 Second pass', '4 Exclusion', '5 Infill']

const axes = axesFor(
  data.examples.flatMap(({ flux, truth, mu, sd }) => [
    ...flux,
    ...truth,
    ...mu.map((m, k) => m - 2 * sd[k]),
    ...mu.map((m, k) => m + 2 * sd[k]),
  ]),
  npix,
)

export function plotMarkup(example: number, step: number): string {
  const { flux, truth, mu, sd } = data.examples[example]
  const y = yScale(axes)
  let prediction = ''
  if (step === 5) {
    const { replaced } = at(example, 5)
    // mu and sd hold the replaced pixels in increasing order.
    const pixels = replaced.flatMap(([a, b]) => Array.from({ length: b - a }, (_, k) => a + k))
    const full = new Float64Array(npix)
    pixels.forEach((i, k) => (full[i] = mu[k]))
    prediction =
      pixels.map((i, k) => bandCell(i, mu[k], sd[k], 2, y) + bandCell(i, mu[k], sd[k], 1, y)).join('') +
      replaced.map(([a, b]) => line('pred', steps(full, a, b, y))).join('')
  }
  return (
    plotSvg(
      gridPath(axes.yticks, y, npix) + line('truth', steps(truth, 0, npix, y)) + line('obs', steps(flux, 0, npix, y)) + prediction,
      npix,
    ) +
    axisLabels(axes, npix, y)
  )
}

// One strip of cells per set: the replaced pixels, the ignored pixels and the flagged
// blocks.
export function masksMarkup(example: number, step: number): string {
  const masks = at(example, step)
  const cells = (runs: Runs, cls: string) =>
    runs.map(([a, b]) => `<rect class="${cls}" x="${a}" y="0" width="${b - a}" height="1"/>`).join('')
  const strip = (label: string, body: string) =>
    `<div class="gs-strip"><span>${label}</span><svg viewBox="0 0 ${npix} 1" preserveAspectRatio="none" aria-hidden="true">${body}</svg></div>`
  return (
    strip('replaced', cells(masks.replaced, 'replaced')) +
    strip('ignored', cells(masks.ignored, 'ignored')) +
    strip(
      'flagged blocks',
      cells(
        masks.blocks.map((k) => [starts[k], starts[k] + width]),
        'newflag',
      ),
    )
  )
}

export function readout(example: number, step: number): string {
  const masks = at(example, step)
  const sets = `${count(masks.replaced)} pixels to replace and ${count(masks.ignored)} to ignore`
  const run = `each run of ${options.runlength} or more flagged pixels`
  const step4 = data.examples[example].too_few
    ? `Step 4 would ignore the ${options.halfwidth} pixels on each side of ${run}. ` +
      `That would leave fewer than ${Math.floor(options.minfrac * npix)} pixels to predict from, so it changes nothing: ${sets}.`
    : `Step 4 ignores the ${options.halfwidth} pixels on each side of ${run}: ${sets}.`
  const text = [
    'The input. The thin line is the flux without noise.',
    `Step 1 flags each pixel whose |z| exceeds ${options.nsigma_replace} to replace, or ${options.nsigma_ignore} to ignore, and widens each flag by ${options.dilate} pixel: ${sets}.`,
    `Step 2 predicts the flagged pixels, then flags each block whose χ² exceeds its threshold: ${masks.blocks.length} of ${starts.length} blocks. ` +
      `Their pixels join those to replace: ${sets}.`,
    `Step 3 computes z again, against a prediction from the pixels outside the flags. ` +
      `A flag stays only where z still exceeds a threshold: ${sets}.`,
    step4,
    `Step 5 replaces ${count(masks.replaced)} pixels by their conditional mean, as infill does.`,
  ]
  return text[step]
}

export const exampleLabel = (example: number) => countLabel('Spectrum', example, data.examples.length)
