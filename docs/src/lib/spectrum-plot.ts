// The markup of a spectrum plot with a masked block and its prediction, as
// docs/brand.md draws it: the spectrum in steps, the block as a flat tint with a solid
// edge and two grips, and the prediction with its ±1σ and ±2σ bands as cells per pixel.
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
  // masked pixels are read.
  mu: ArrayLike<number>
  sd: ArrayLike<number>
}

export interface Axes {
  low: number
  high: number
  yticks: readonly number[]
  // The decimals of the flux labels.
  digits: number
  xticks: readonly number[]
}

// Axes that hold every value of values with a margin of 5% of their range, with flux
// ticks at the step of 1, 2 or 5 times a power of ten nearest to a fifth of the range, and
// pixel ticks at the ends and quarters of the npix pixels.
export function axesFor(values: readonly number[], npix: number): Axes {
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
  const yticks: number[] = []
  for (let k = Math.ceil(low / step); k * step <= high; k++) yticks.push(k * step)
  return {
    low,
    high,
    yticks,
    digits: Math.max(0, -Math.floor(Math.log10(step))),
    xticks: [0, 1, 2, 3, 4].map((k) => Math.round((k * npix) / 4)),
  }
}

// A step path of values[from] to values[to - 1], with y(v) in the units of the SVG.
function steps(values: ArrayLike<number>, from: number, to: number, y: (v: number) => number) {
  let d = ''
  for (let i = from; i < to; i++) d += `${i === from ? `M${i},` : 'V'}${y(values[i]).toFixed(2)}H${i + 1}`
  return d
}

export function plotMarkup(spectrum: Spectrum, axes: Axes): string {
  const { flux, truth, start, width, mu, sd } = spectrum
  const npix = flux.length
  const end = start + width
  // 100 units from the top of the plot area to its bottom.
  const y = (v: number) => (100 * (axes.high - v)) / (axes.high - axes.low)
  const cells: string[] = []
  const cell = (i: number, k: number) => {
    const top = y(mu[i] + k * sd[i])
    return `<rect class="b${k}" x="${i}" y="${top.toFixed(2)}" width="1" height="${(y(mu[i] - k * sd[i]) - top).toFixed(2)}"/>`
  }
  for (let i = start; i < end; i++) cells.push(cell(i, 2), cell(i, 1))
  const gaps: string[] = []
  for (let i = start + 1; i < end; i++) gaps.push(`M${i},0V100`)
  const grid = axes.yticks.map((v) => `M0,${y(v).toFixed(2)}H${npix}`).join('')
  const line = (cls: string, d: string) => (d ? `<path class="${cls}" d="${d}"/>` : '')
  const svg =
    `<svg viewBox="0 0 ${npix} 100" preserveAspectRatio="none" aria-hidden="true">` +
    line('grid', grid) +
    `<rect class="blk" x="${start}" y="0" width="${width}" height="100"/>` +
    cells.join('') +
    line('cellgap', gaps.join('')) +
    `<path class="blk-edge" d="M${start},0V100M${end},0V100"/>` +
    line('obs', steps(flux, 0, start, y)) +
    line('obs', steps(flux, end, npix, y)) +
    (truth ? line('truth', steps(truth, 0, npix, y)) : '') +
    line('pred', steps(mu, start, end, y)) +
    '</svg>'
  const left = (i: number) => `${((100 * i) / npix).toFixed(3)}%`
  const label = (cls: string, style: string, text: string) => `<span class="${cls}" style="${style}">${text}</span>`
  return (
    svg +
    `<span class="grip" style="left:${left(start)}"></span><span class="grip" style="left:${left(end)}"></span>` +
    axes.yticks.map((v) => label('ytick', `top:${y(v).toFixed(2)}%`, v.toFixed(axes.digits))).join('') +
    axes.xticks
      .map((i, k) => label(k === 0 ? 'xtick first' : k === axes.xticks.length - 1 ? 'xtick last' : 'xtick', `left:${left(i)}`, String(i)))
      .join('')
  )
}

// The masked range, the mean standard deviation of the prediction there, and its RMS
// error against the noise-free flux.
export function readout(spectrum: Required<Spectrum>): string {
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

// The legend of a plot, as cell swatches; masked adds the swatch of the masked block.
export function legend(masked: boolean): string {
  const items: [string, string][] = [
    ['obs', 'observed flux'],
    ...(masked ? [['blk', 'masked'] as [string, string]] : []),
    ['pred', 'prediction'],
    ['b1', '±1σ'],
    ['b2', '±2σ'],
    ['truth', 'flux without noise'],
  ]
  return `<ul class="gs-legend">${items.map(([cls, text]) => `<li><i class="${cls}"></i>${text}</li>`).join('')}</ul>`
}
