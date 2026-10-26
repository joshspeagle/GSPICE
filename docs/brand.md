# The GSPICE identity

This document states the design of the documentation site: its tokens, type, logo, motif and components. It is not part of the site. The lead chose this direction, "pixels and covariance", in #137.

The site is drawn in the method's own units. A pixel is a square cell, a covariance is a heat map, and orange marks only what GSPICE fills in.

## Palette

Dark mode is designed first, on a cool blue-black ground. Light mode keeps the navy header, so the brand block is the same in both themes.

| Token | Dark | Light | Use |
|---|---|---|---|
| `--gs-bg` | #0B111C | #FFFFFF | ground |
| `--gs-bg-2` | #111A2A | #F2F5F9 | surfaces: code, callouts, cards |
| `--gs-bg-3` | #18233A | #E6EBF2 | raised: labels, hover |
| `--gs-rule` | #26324A | #D6DDE6 | hairlines |
| `--gs-rule-2` | #3A4868 | #A9B4C4 | solid edges, inactive cells |
| `--gs-ink` | #E8ECF2 | #111827 | text |
| `--gs-ink-2` | #A7B1C2 | #4A5568 | secondary text |
| `--gs-ink-3` | #7F8AA0 | #5F6B7F | tertiary text, interpolated flux, P_GO cells, joint-plot points |
| `--gs-accent` | #F0592A | #F0592A | predicted cells and marks; never text |
| `--gs-accent-text` | #FF8A5C | #C2410C | links and orange text |
| `--gs-pred` | #FF7A45 | #E2481A | prediction line |
| `--gs-band-1` | #A8401E | #F9A47F | ±1σ cells |
| `--gs-band-2` | #4A2219 | #FDD9C8 | ±2σ cells |
| `--gs-mask-fill` | #1A2438 | #E9EDF3 | masked block |
| `--gs-mask-edge` | #6B7A99 | #7A879C | edge of the masked block |
| `--gs-header` | #0B111C | #111A2A | header band |
| `--gs-strip-lo`, `--gs-strip-hi` | #3B465C, #C9D1DE | #6B7588, #E1E6EE | ends of the footer cell strip; `--gs-strip-lo` also fills observed cells in diagrams |

The header band is navy in both themes, so its text and edges keep the dark values. Its focus ring is #FF8A5C, because #C2410C is too dark on navy. The masked block of the header lockup is #1E2A42, because the mask fill #1A2438 nearly vanishes on the light theme's #111A2A band.

### Contrast

These ratios follow WCAG 2.

| Pair | Dark | Light | Minimum |
|---|---|---|---|
| ink on bg | 15.94 | 17.74 | 4.5 |
| ink-2 on bg | 8.74 | 7.53 | 4.5 |
| ink-3 on bg-2 | 5.02 | 4.93 | 4.5 |
| accent-text on bg | 8.14 | 5.18 | 4.5 |
| accent-text on bg-2 | 7.50 | 4.74 | 4.5 |
| ink-2 on header | 8.74 | 8.06 | 4.5 |
| prediction line on bg-2 | 6.74 | 3.71 | 3 (graphic) |
| mask edge on bg-2 | 4.04 | 3.33 | 3 (graphic) |

### Code

Keywords and constants are steel blue, strings are green, comments are ink-3, and all other code is ink.

| Role | Dark | Light | Contrast on bg-2 |
|---|---|---|---|
| keyword, constant | #9DB8E0 | #2F5B99 | 8.60 / 6.24 |
| string | #A9D3B0 | #2E6B3F | 10.48 / 5.84 |

### Heat map

The heat map shows correlation from −1 to +1 in five steps.

| Theme | −1 | −0.5 | 0 | +0.5 | +1 |
|---|---|---|---|---|---|
| Dark | #22355E | #3E5276 | #5D6679 | #D2653A | #FFB892 |
| Light | #D5DFEE | #A6B4C9 | #8A909C | #C0501C | #7E2A0C |

Its lightness runs in one direction only: L* from 23 to 81 in dark mode and from 88 to 30 in light mode. The ramp therefore reads in greyscale. Strong positive correlation always has the most contrast with the ground.

## Type

| Role | Face | Setting |
|---|---|---|
| Body | Archivo | 16.5 px, line height 1.62, width 100 |
| Page title | Archivo | 40 px, weight 720, width 112 |
| Section heading | Archivo | 26 px, weight 660, width 108 |
| Subheading | Archivo | 19 px, weight 620 |
| Wordmark | Archivo | weight 760, width 118, outlined in the logo files |
| Code, API names, numbers and units | IBM Plex Mono | 14 px |

- Both faces are self-hosted from @fontsource. Archivo is variable, with a width axis.
- KaTeX renders the maths.

## Logo

The logo is a G on a 5 × 5 grid of cells. A masked block covers its left stroke, and the stroke returns as three orange cells. "SPICE" follows in Archivo Expanded, so the lockup reads "GSPICE" with the G as its symbol.

| Size | Form |
|---|---|
| 32 px and smaller | Closed cells with no gaps, so the letter keeps its weight. This covers the favicon and the header lockup. |
| 48 px and larger | Open cells with 1 px gaps. Hollow orange cells beside the stroke show the uncertainty, and the block has a solid edge. |

- The gap between the G and "SPICE" is fixed, 3 px at header size, so the name never reads "G SPICE".
- The files are outlined SVG, with no live text. `docs/scripts/logo.py` writes them:
  - the full logo, for the dark and the light ground;
  - the header lockup, for the navy band;
  - the favicon, a 32 px design that follows the colour scheme of the browser. One SVG favicon serves every size, so there is no separate 16 or 64 px file.

## Motif

1. A cell is one pixel: a square on a grid, with a 1 px gap where the cell is wider than 4 px.
2. Solid orange cells are predicted values, and hollow orange cells are their uncertainty. Orange marks nothing else, except the current page.
3. A masked block is a flat cool tint with a solid edge and two square grips. It has no dashes.
4. Every spectrum is drawn as steps, one per pixel. Bands are cells per pixel, not smooth ribbons.
5. A heat map uses the ramp above, cell for cell, and is never smoothed.
6. Corners are square everywhere. Nothing has a shadow, a gradient or a glow.

The motif appears in these places:
- the logo and the favicon;
- the sidebar's square group markers;
- the "On this page" list, drawn as a column of cells with the current one orange;
- the cell strip of the mean spectrum in the footer;
- the glyphs of the page cards, drawn from cells;
- the covariance square beside the full explainer.

## Components

The theme of #142 builds the header, the sidebar, "On this page", the page title, code and output, and the figure frame. The callout, the term card, the page card, the label and the covariance square come with the pages and explainers that use them (#137).

| Component | Form |
|---|---|
| Header | A navy band with the lockup, a square search box with a `/` key label, text links, and a theme toggle drawn in cells that cycles auto, dark and light. |
| Sidebar | Square group markers. The current page has a 3 px orange left edge. |
| On this page | The column of cells. |
| Page title | A breadcrumb of the page's group, then the title. API pages set the title in mono. |
| Callout | A surface with a square label, "Note" or "Warning", and a 3 px left edge in ink-3 or orange. |
| Code and output | The code sits on a surface with a header bar that names the language and holds the word "Copy". An output block has the flag `output` on its fence, as in `text output`: it joins the code block above it on the ground, with a square "Output" tag and no header bar. |
| Term | A dotted underline. On hover or focus, a card shows the glossary entry as plain text; the term itself links to the entry. |
| Page card | A surface with a solid top edge, a cell glyph, the title, one line, and the name of the page it opens. |
| Label | A small square tag in mono: function, class, the version. |
| Previous and next | Two tiles with a solid top edge. |
| Figure | A rule above, a hairline below, and the caption in secondary text. The legend uses cell swatches. |
| Decision chart | Questions on surfaces with a 3 px top rule, model tiles with a 3 px ink left edge, and "no" connectors from each question down to the next (ModelChart). |
| Step list | Numbered rows, each with a strip of cells (CalibrationSteps). |
| Explainer | A surface with a solid top edge: a hint, the plots, the controls, a live readout and a caption (`src/components/explainers/`). Sliders are the browser's range inputs, coloured ink. The current step of a stepper has a 3 px ink left edge, as a pressed button. |
| Covariance square | The heat map of the correlation of a model, one cell per pixel pair. A masked block marks its rows and columns as flat tints with a solid edge; a selected pair is a 2 px ink outline. |
| Joint plot | Two pixels against each other on a square plot: training spectra as ink-3 square points, the 1σ and 2σ ellipses of the model in ink-2, a fixed value as an ink line, and the conditional mean and its ±1σ and ±2σ as orange cells. |
| Mask strip | One row of cells per pixel set under a spectrum, in the cell states below: replaced pixels in `--gs-accent`, ignored pixels in the mask fill with its edge, and flagged blocks in `--gs-rule-2`. |

## Figures

- Scripts draw every figure from data, as SVG in the theme tokens, so one file serves both themes. The figures of tutorials/fitting still come in light and dark pairs until they are redrawn.
- All text in a figure is HTML at 12 px or more, placed over the plot, so it does not shrink on a phone.
- Pixel boundaries inside the masked block are hidden when a pixel is narrower than 4 px.
- A draw is a thin orange step line, drawn under the prediction.
- The reference curve of a histogram or a plot, class `ref`, and a threshold line, class `edge`, are thin ink-2 lines.
- In the guard window explainer, a defect is marked by a thick ink-2 bar along the top of the plot.
- z-scores are bars per pixel: ink-3, and ink beyond the threshold, whose lines are ink-2.

The figures of the Concepts pages add these marks:

| Mark | Class | Form |
|---|---|---|
| Interpolated flux | `interp` | An ink-3 step line, 1.5 px |
| Guard window | `guard` | A flat `--gs-rule` fill, without an edge |
| Artifact | `span-mark` | A 4 px ink-2 bar along the top of the plot area |
| Pixels a clipping pass flags | `newflag` | A flat `--gs-rule-2` fill |
| Scatter point | `pt held`, `pt in` | An 8 px square, solid ink, or hollow with a 1.5 px ink-2 outline |

A strip of pixel cells, class `gs-cells`, gives each cell one of these states:

| State | Class | Fill |
|---|---|---|
| Predicted or replaced | `replaced` | `--gs-accent` |
| Guard window | `guard` | `--gs-rule` |
| Ignored, or a value not read | `ignored`, `unread` | `--gs-mask-fill` with a 1 px `--gs-mask-edge` outline |
| Bad | `bad` | `--gs-mask-edge` |
| Conditioned on | `kept` | `--gs-strip-lo` |
