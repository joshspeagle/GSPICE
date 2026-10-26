// The client of the mask explainer, V-1: drag or resize the masked block over a
// spectrum, and redraw the prediction of the hidden pixels from the others. In the full
// version, the covariance square beside the plot marks the rows and columns of the block.
import data from '../data/explainer.json'
import { announcer, clamp, countLabel, redrawer } from './explainer.ts'
import { condition } from './gauss.ts'
import { plotMarkup, readout, type Axes } from './spectrum-plot.ts'

const npix = data.npix

// The position of the block in the covariance square, as percentages of its side.
export function squareStyle(start: number, width: number): string {
  return `--start:${(100 * start) / npix}%;--width:${(100 * width) / npix}%`
}

export function mountMaskExplainer(root: HTMLElement) {
  const axes: Axes = JSON.parse(root.dataset.axes!)
  const plot = root.querySelector<HTMLElement>('[data-spectrum]')!
  const square = root.querySelector<HTMLElement>('.gs-square')
  const count = root.querySelector<HTMLElement>('[data-count]')!
  const truthButton = root.querySelector<HTMLButtonElement>('[data-action="truth"]')!
  const announce = announcer(root.querySelector<HTMLElement>('[data-readout]')!)
  const state = { ...JSON.parse(root.dataset.first!), truth: false }

  const schedule = redrawer(() => {
    const { flux, truth } = data.examples[state.example]
    const { mu, sd } = condition(data.mean, data.cov, flux, state.start, state.width)
    const spectrum = { flux, start: state.start, width: state.width, mu, sd }
    plot.innerHTML = plotMarkup(state.truth ? { ...spectrum, truth } : spectrum, axes)
    square?.setAttribute('style', squareStyle(state.start, state.width))
    announce(readout({ ...spectrum, truth }))
    count.textContent = countLabel('Spectrum', state.example, data.examples.length)
  })
  // Keeps the block inside the spectrum, from 1 to data.widest pixels wide.
  function place(start: number, width: number) {
    state.width = clamp(width, 1, data.widest)
    state.start = clamp(start, 0, npix - state.width)
    schedule()
  }

  // The pixel under the pointer, and the edge of the block within 8 px of it, if any. The
  // reach is at most a quarter of the block, so that the middle of a narrow block moves it.
  function position(event: PointerEvent) {
    const box = plot.getBoundingClientRect()
    const x = ((event.clientX - box.left) / box.width) * npix
    const reach = Math.min((8 / box.width) * npix, state.width / 4)
    const edge =
      Math.abs(x - state.start) < reach ? 'start' : Math.abs(x - state.start - state.width) < reach ? 'end' : null
    return { pixel: Math.floor(x), edge }
  }
  let drag: { mode: 'start' | 'end' } | { mode: 'move'; offset: number } | null = null
  plot.addEventListener('pointerdown', (event) => {
    const { pixel, edge } = position(event)
    // Only the main button, and only on the plot, not on its tick labels.
    if (event.button !== 0 || pixel < 0 || pixel >= npix) return
    if (edge) drag = { mode: edge }
    else {
      // A press outside the block moves its centre there.
      if (pixel < state.start || pixel >= state.start + state.width) place(pixel - (state.width >> 1), state.width)
      drag = { mode: 'move', offset: pixel - state.start }
    }
    plot.setPointerCapture(event.pointerId)
  })
  plot.addEventListener('pointermove', (event) => {
    const { pixel, edge } = position(event)
    if (!drag) {
      const inside = pixel >= state.start && pixel < state.start + state.width
      plot.style.cursor = edge ? 'ew-resize' : inside ? 'grab' : 'pointer'
      return
    }
    const p = clamp(pixel, 0, npix - 1)
    const end = state.start + state.width
    if (drag.mode === 'start') {
      const start = Math.max(Math.min(p, end - 1), end - data.widest)
      place(start, end - start)
    } else if (drag.mode === 'end') place(state.start, Math.max(p + 1, state.start + 1) - state.start)
    else place(p - drag.offset, state.width)
  })
  const release = () => (drag = null)
  plot.addEventListener('pointerup', release)
  plot.addEventListener('pointercancel', release)
  plot.addEventListener('keydown', (event) => {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return
    event.preventDefault()
    const step = event.key === 'ArrowRight' ? 1 : -1
    if (event.shiftKey) place(state.start, state.width + step)
    else place(state.start + step, state.width)
  })
  truthButton.addEventListener('click', () => {
    state.truth = !state.truth
    truthButton.setAttribute('aria-pressed', String(state.truth))
    schedule()
  })
  root.querySelector('[data-action="next"]')!.addEventListener('click', () => {
    state.example = (state.example + 1) % data.examples.length
    schedule()
  })
}
