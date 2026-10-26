// The client helpers that every explainer shares.

// Returns a function that runs draw once in the next animation frame, however often it
// is called before then.
export function redrawer(draw: () => void): () => void {
  let frame = 0
  return () => {
    frame ||= requestAnimationFrame(() => {
      frame = 0
      draw()
    })
  }
}

// Returns a function that sets the text of a live region 300 ms after its last call, so
// that a screen reader announces the state once the input stops, not every frame of a
// drag.
export function announcer(region: HTMLElement): (text: string) => void {
  let timer = 0
  return (text) => {
    clearTimeout(timer)
    timer = setTimeout(() => (region.textContent = text), 300)
  }
}

export const clamp = (value: number, low: number, high: number) => Math.max(low, Math.min(high, value))

// The label of item index of count, such as "Spectrum 2 of 8".
export const countLabel = (noun: string, index: number, count: number) => `${noun} ${index + 1} of ${count}`
