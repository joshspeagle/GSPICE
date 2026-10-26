// Writes the figure of README.md, in a light and a dark version, from the quickstart data:
//
//     node docs/scripts/readme.mjs
//
// figures() returns both SVG files as strings, which docs/tests/readme.test.mjs compares
// with the files in docs/readme/.
// It draws the spectrum with plotMarkup of src/lib/spectrum-plot.ts, as the quickstart
// figure does, and styles it with the rules of src/styles/plot.css. A README image
// cannot read the theme of the site, so each file gets the colours of one theme from
// src/styles/custom.css: docs/readme/quickstart-light.svg and quickstart-dark.svg. The
// tick labels and the legend, which are HTML on the site, are SVG text here.
import { readFileSync, writeFileSync } from 'node:fs'

import { labels, plotMarkup, quickstartLegend, quickstartPlot } from '../src/lib/spectrum-plot.ts'

const read = (path) => readFileSync(new URL(path, import.meta.url), 'utf8')
const data = JSON.parse(read('../src/data/quickstart.json'))

// The custom properties of a block of declarations.
const properties = (block) => Object.fromEntries([...block.matchAll(/(--[\w-]+):\s*([^;]+);/g)].map((m) => [m[1], m[2]]))
const custom = read('../src/styles/custom.css')
const dark = properties(custom.match(/^:root \{([^}]*)\}/m)[1])
const themes = { dark, light: { ...dark, ...properties(custom.match(/^:root\[data-theme='light'\] \{([^}]*)\}/m)[1]) } }

// The rules of plot.css outside its @container blocks, which hide the pixel boundaries
// in a narrow plot: [selector, declarations].
const rules = [
  ...read('../src/styles/plot.css')
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/@container[^{]*\{[^{}]*\{[^}]*\}\s*\}/g, '')
    .matchAll(/([^{}]+)\{([^}]*)\}/g),
].map(([, selector, body]) => [selector.trim(), body.trim().replace(/\s+/g, ' ')])

const CELL = 4 // px per pixel, so that the pixel boundaries in the masked block show
const LEFT = 48
const TOP = 12
const HEIGHT = 240 // the plot area, px
const npix = data.flux.length
const right = LEFT + CELL * npix
const bottom = TOP + HEIGHT
const size = [right + 12, bottom + 64] // room for the pixel labels and the legend

const { spectrum, axes } = quickstartPlot(data)
const markup = plotMarkup(spectrum, axes)
// The SVG of the plot, in pixel units and 100 units of height, without the HTML after it.
// Each σ band becomes one path of per-pixel rectangles instead of a rect per pixel, at
// the place of the first rect.
const band = { b1: '', b2: '' }
let first = true
const plot = markup
  .slice(markup.indexOf('>') + 1, markup.indexOf('</svg>'))
  .replace(/<rect class="(b[12])" x="([^"]+)" y="([^"]+)" width="1" height="([^"]+)"\/>/g, (_, cls, x, y, h) => {
    band[cls] += `M${x},${y}h1v${h}h-1z`
    return '\0'
  })
  .replace(/\0+/g, () => {
    const paths = first ? `<path class="b2" d="${band.b2}"/><path class="b1" d="${band.b1}"/>` : ''
    first = false
    return paths
  })
// The rules of plot.css for the paths and for the classes that the plot uses.
const used = new Set([...plot.matchAll(/class="([^"]+)"/g)].map((m) => m[1]))
const drawn = ([selector]) =>
  selector === '.gs-plot path' ||
  (selector.startsWith('.gs-plot ') && [...selector.matchAll(/\.([\w-]+)/g)].some((m) => used.has(m[1])))

function svg(theme) {
  // The declarations of a rule, with the values of the theme in place of its variables.
  const colour = ([selector, body]) => {
    const text = body.replace(/var\((--[\w-]+)\)/g, (all, name) => theme[name] ?? all)
    if (text.includes('var(')) throw new Error(`readme.mjs: an unknown variable in the rule ${selector} of plot.css`)
    return text
  }
  const style = rules
    .filter(drawn)
    .map((rule) => `${rule[0]}{${colour(rule)}}`)
    .join('')
  // A legend swatch has the size and colour of the rules of .gs-legend i and its class.
  const swatch = (cls) => {
    const body = rules
      .filter(([s]) => s === '.gs-legend i' || (s.startsWith('.gs-legend') && new RegExp(`\\.${cls}\\b`).test(s)))
      .map(colour)
      .join(';')
    const value = (name) => [...body.matchAll(new RegExp(`(?:^|;)\\s*${name}:\\s*([^;]+)`, 'g'))].at(-1)?.[1].trim()
    return { w: parseFloat(value('width')), h: parseFloat(value('height')), fill: value('background'), edge: value('outline')?.split(' ').at(-1) }
  }
  const ink2 = theme['--gs-ink-2']
  const mono = `font-family="${theme['--sl-font-mono']}" font-size="12" fill="${ink2}"`
  const sans = `font-family="${theme['--sl-font']}" font-size="13" fill="${ink2}"`
  const y = (v) => TOP + (HEIGHT * (axes.high - v)) / (axes.high - axes.low)
  const out = [
    `<style>${style}</style>`,
    `<rect width="${size[0]}" height="${size[1]}" fill="${theme['--gs-bg']}"/>`,
    `<g class="gs-plot" transform="translate(${LEFT},${TOP}) scale(${CELL},${HEIGHT / 100})">${plot}</g>`,
  ]
  for (const i of [data.start, data.start + data.width]) {
    out.push(`<rect x="${LEFT + CELL * i - 2}" y="${TOP + HEIGHT / 2 - 9}" width="4" height="18" fill="${ink2}"/>`)
  }
  for (const v of axes.yticks) {
    out.push(`<text x="${LEFT - 6}" y="${y(v).toFixed(2)}" dy="0.35em" text-anchor="end" ${mono}>${v.toFixed(axes.digits)}</text>`)
  }
  axes.xticks.forEach((i, k) => {
    const anchor = k === 0 ? 'start' : k === axes.xticks.length - 1 ? 'end' : 'middle'
    out.push(`<text x="${LEFT + CELL * i}" y="${bottom + 18}" text-anchor="${anchor}" ${mono}>${i}</text>`)
  })
  let left = LEFT
  const top = bottom + 44
  for (const cls of quickstartLegend) {
    const { w, h, fill, edge } = swatch(cls)
    const stroke = edge ? ` stroke="${edge}"` : ''
    out.push(`<rect x="${left + (edge ? 0.5 : 0)}" y="${top - h / 2 + (edge ? 0.5 : 0)}" width="${w - (edge ? 1 : 0)}" height="${h - (edge ? 1 : 0)}" fill="${fill}"${stroke}/>`)
    out.push(`<text x="${left + w + 7}" y="${top}" dy="0.35em" ${sans}>${labels[cls]}</text>`)
    left += w + 7 + 7.5 * labels[cls].length + 18
  }
  const head = `width="${size[0]}" height="${size[1]}" viewBox="0 0 ${size[0]} ${size[1]}"`
  return `<svg xmlns="http://www.w3.org/2000/svg" ${head}>\n${out.join('\n')}\n</svg>\n`
}

// The files docs/readme/quickstart-{light,dark}.svg, by name, and their contents.
export function figures() {
  return Object.fromEntries(Object.entries(themes).map(([name, theme]) => [`quickstart-${name}.svg`, svg(theme)]))
}

if (import.meta.main) {
  for (const [name, text] of Object.entries(figures())) writeFileSync(new URL(`../readme/${name}`, import.meta.url), text)
}
