// The markup of a spectrum plot with a masked block and its prediction, as
// docs/brand.md draws it: the spectrum in steps, the block as a flat tint with a solid
// edge and two grips, and the prediction with its ±1σ and ±2σ bands as cells per pixel,
// or with draws in place of the bands.
// The SVG spans the plot area in data units, stretched to its box, so the same markup
// serves the build and the browser; the tick labels are HTML.

export interface Spectrum {
  flux: readonly number[]
  // The noise-free flux, drawn as a thin line when given.
  truth?: readonly number[]
  // The masked pixels start to start + width - 1.
  start: number
  width: number
  // The prediction and its standard deviation at each pixel, indexed from 0; only the
  // masked pixels are read. Without sd, the plot has no bands.
  mu: ArrayLike<number>
  sd?: ArrayLike<number>
  // Draws of the masked pixels, indexed like mu, drawn as thin steps.
  draws?: readonly ArrayLike<number>[]
}

export interface Axes {
  low: number
  high: number
  yticks: readonly number[]
  // The decimals of the flux labels.
  digits: number
  xticks: readonly number[]
}

// The scale of an axis that holds every value of values with a margin of 5% of their
// range, with ticks at the step of 1, 2 or 5 times a power of ten nearest to a fifth of
// the range.
export interface Scale {
  low: number
  high: number
  ticks: readonly number[]
  // The decimals of the tick labels.
  digits: number
}

export function scaleFor(values: readonly number[]): Scale {
  let min = Infinity
  let max = -Infinity
  for (const v of values) {
    min = Math.min(min, v)
    max = Math.max(max, v)
  }
  const low = min - 0.05 * (max - min)
  const high = max + 0.05 * (max - min)
  const rough = (high - low) / 5
  const power = 10 ** Math.floor(Math.log10(rough))
  const step = [1, 2, 5, 10]
    .map((m) => m * power)
    .reduce((best, s) => (Math.abs(Math.log(s / rough)) < Math.abs(Math.log(best / rough)) ? s : best))
  const ticks: number[] = []
  for (let k = Math.ceil(low / step); k * step <= high; k++) ticks.push(k * step)
  return { low, high, ticks, digits: Math.max(0, -Math.floor(Math.log10(step))) }
}

// The ends and quarters of npix pixels.
export const pixelTicks = (npix: number) => [0, 1, 2, 3, 4].map((k) => Math.round((k * npix) / 4))

// Axes that hold every value of values on the scale of scaleFor, with pixel ticks at the
// ends and quarters of the npix pixels.
export function axesFor(values: readonly number[], npix: number): Axes {
  const { low, high, ticks, digits } = scaleFor(values)
  return { low, high, yticks: ticks, digits, xticks: pixelTicks(npix) }
}

// A step path of values[from] to values[to - 1], with y(v) in the units of the SVG.
export function steps(values: ArrayLike<number>, from: number, to: number, y: (v: number) => number) {
  let d = ''
  for (let i = from; i < to; i++) d += `${i === from ? `M${i},` : 'V'}${y(values[i]).toFixed(2)}H${i + 1}`
  return d
}

// The range of an axis, from low to high.
export interface Range {
  low: number
  high: number
}

// y(v) in the units of the SVG: 100 at low, at the bottom of the plot area, and 0 at high.
export function yScale(range: Range) {
  return (v: number) => (100 * (range.high - v)) / (range.high - range.low)
}

// x(v) as a percentage of the width of the plot area: 0 at low and 100 at high.
export function xScale(range: Range) {
  return (v: number) => (100 * (v - range.low)) / (range.high - range.low)
}

// The SVG of a plot area width units wide from x0, stretched to its box.
export function plotSvg(body: string, width: number, x0 = 0) {
  return `<svg viewBox="${x0} 0 ${width} 100" preserveAspectRatio="none" aria-hidden="true">${body}</svg>`
}

// The horizontal grid lines at the ticks, across a plot area width units wide.
export function gridPath(ticks: readonly number[], y: (v: number) => number, width: number) {
  return ticks.length ? `<path class="grid" d="${ticks.map((v) => `M0,${y(v).toFixed(2)}H${width}`).join('')}"/>` : ''
}

// v with digits decimals and a minus sign, U+2212, for negative values.
export function signed(v: number, digits: number) {
  return v.toFixed(digits).replace('-', '−')
}

// The tick labels of axes, with the pixel ticks along npix pixels.
export function axisLabels(axes: Axes, npix: number, y: (v: number) => number) {
  return tickLabels(
    axes.yticks.map((v) => [y(v), signed(v, axes.digits)]),
    axes.xticks.map((i) => [(100 * i) / npix, String(i)]),
  )
}

// The ±k sd cell of a prediction mu at x, width units wide.
export function bandCell(x: number | string, mu: number, sd: number, k: number, y: (v: number) => number, width = 1) {
  const top = y(mu + k * sd)
  return `<rect class="b${k}" x="${x}" y="${top.toFixed(2)}" width="${width}" height="${(y(mu - k * sd) - top).toFixed(2)}"/>`
}

// The masked block of pixels start to start + width - 1 in a plot of npix pixels: its
// fill, which goes under the cells, its edges, which go over them, and its two grips, in
// HTML.
export function blockMarkup(start: number, width: number, npix: number) {
  const end = start + width
  const left = (i: number) => ((100 * i) / npix).toFixed(3)
  return {
    fill: `<rect class="blk" x="${start}" y="0" width="${width}" height="100"/>`,
    edge: `<path class="blk-edge" d="M${start},0V100M${end},0V100"/>`,
    grips: `<span class="grip" style="left:${left(start)}%"></span><span class="grip" style="left:${left(end)}%"></span>`,
  }
}

// A path of class cls, or nothing for an empty d.
export function line(cls: string, d: string) {
  return d ? `<path class="${cls}" d="${d}"/>` : ''
}

export function plotMarkup(spectrum: Spectrum, axes: Axes): string {
  const { flux, truth, start, width, mu, sd, draws = [] } = spectrum
  const npix = flux.length
  const end = start + width
  const y = yScale(axes)
  const cells: string[] = []
  if (sd) for (let i = start; i < end; i++) cells.push(bandCell(i, mu[i], sd[i], 2, y), bandCell(i, mu[i], sd[i], 1, y))
  const gaps: string[] = []
  for (let i = start + 1; i < end; i++) gaps.push(`M${i},0V100`)
  const block = blockMarkup(start, width, npix)
  const svg = plotSvg(
    gridPath(axes.yticks, y, npix) +
      block.fill +
      cells.join('') +
      line('cellgap', gaps.join('')) +
      block.edge +
      line('obs', steps(flux, 0, start, y)) +
      line('obs', steps(flux, end, npix, y)) +
      (truth ? line('truth', steps(truth, 0, npix, y)) : '') +
      draws.map((draw) => line('draw', steps(draw, start, end, y))).join('') +
      line('pred', steps(mu, start, end, y)),
    npix,
  )
  return svg + block.grips + axisLabels(axes, npix, y)
}

type Tick = readonly [position: number, text: string]

// The HTML labels of the y ticks, at a percentage from the top of the plot area, and of
// the x ticks, at a percentage from its left; the first and last x labels stay inside.
export function tickLabels(yticks: readonly Tick[], xticks: readonly Tick[]): string {
  const last = xticks.length - 1
  return (
    yticks.map(([top, text]) => `<span class="ytick" style="top:${top.toFixed(2)}%">${text}</span>`).join('') +
    xticks
      .map(([left, text], k) => {
        const cls = k === 0 ? 'xtick first' : k === last ? 'xtick last' : 'xtick'
        return `<span class="${cls}" style="left:${left.toFixed(3)}%">${text}</span>`
      })
      .join('')
  )
}

// The masked range, the mean standard deviation of the prediction there, and its RMS
// error against the noise-free flux.
export function readout(spectrum: Spectrum & Required<Pick<Spectrum, 'truth' | 'sd'>>): string {
  const { start, width } = spectrum
  let sd = 0
  let squares = 0
  for (let i = start; i < start + width; i++) {
    sd += spectrum.sd[i]
    squares += (spectrum.mu[i] - spectrum.truth[i]) ** 2
  }
  const range = width === 1 ? `Pixel ${start} masked` : `Pixels ${start} to ${start + width - 1} masked (${width})`
  return `${range}. Mean σ ${(sd / width).toFixed(3)}. RMS error ${Math.sqrt(squares / width).toFixed(3)}.`
}

export const labels = {
  obs: 'observed flux',
  blk: 'masked',
  pred: 'prediction',
  draw: 'draw',
  b1: '±1σ',
  b2: '±2σ',
  truth: 'flux without noise',
}

// The data of the quickstart figure, which docs/scripts/explainers.py writes: a spectrum
// with its noise-free flux, the masked pixels start to start + width - 1, and the
// prediction and its standard deviation at each of them.
export interface QuickstartData {
  flux: readonly number[]
  truth: readonly number[]
  start: number
  width: number
  mu: readonly number[]
  sd: readonly number[]
}

// The spectrum of the quickstart figure, with the prediction placed at its pixel
// numbers, and axes that hold the flux and the ±2σ bands.
export function quickstartPlot(data: QuickstartData) {
  const npix = data.flux.length
  const mu = new Float64Array(npix)
  const sd = new Float64Array(npix)
  mu.set(data.mu, data.start)
  sd.set(data.sd, data.start)
  const spectrum = { flux: data.flux, truth: data.truth, start: data.start, width: data.width, mu, sd }
  const bands = data.mu.flatMap((m, k) => [m - 2 * data.sd[k], m + 2 * data.sd[k]])
  return { spectrum, axes: axesFor([...data.flux, ...data.truth, ...bands], npix) }
}

// The legend items of the quickstart figure, in order.
export const quickstartLegend = ['obs', 'blk', 'pred', 'b1', 'b2', 'truth'] as const

// The legend of a plot, as cell swatches, with the items in this order. An item is a
// key of labels, or a class and its HTML text. The text is one span, so that a subscript
// in it stays against its letter in the flex row of the item.
export function legend(items: readonly (keyof typeof labels | readonly [cls: string, text: string])[]): string {
  const item = (cls: string, text: string) => `<li><i class="${cls}"></i><span>${text}</span></li>`
  return `<ul class="gs-legend">${items.map((i) => (typeof i === 'string' ? item(i, labels[i]) : item(...i))).join('')}</ul>`
}
