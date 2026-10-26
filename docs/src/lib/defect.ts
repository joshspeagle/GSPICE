// The defect of the guard window explainer, V-8: flux with height added to the width
// pixels from start.
export const withDefect = (flux: readonly number[], start: number, width: number, height: number) =>
  flux.map((f, i) => (i >= start && i < start + width ? f + height : f))
