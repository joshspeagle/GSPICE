"""Write the logo files of the site and the cell strip of its footer.

    python docs/scripts/logo.py

It needs fontTools with brotli, from the dependency group logo, and the Archivo font
of docs/node_modules (npm --prefix docs ci). The wordmark "SPICE" is Archivo at weight
760 and width 118, outlined, so the files hold no live text. docs/brand.md states the
design.

It writes to docs/src/assets/: logo-dark.svg and logo-light.svg, the full logo with
open cells, for a dark and a light ground; lockup-dark.svg, the lockup with closed
cells for the navy header, which is navy in both themes; and strip.json, the mean of
8000 noise-free simulated spectra of 120 pixels, scaled to [0, 1]. It also writes
docs/public/favicon.svg, which follows the colour scheme of the browser.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

import gspice

DOCS = Path(__file__).parents[1]
ASSETS = DOCS / "src" / "assets"
FONT = DOCS / "node_modules/@fontsource-variable/archivo/files"
FONT /= "archivo-latin-wdth-normal.woff2"
WORD = "SPICE"
TRACKING = 0.02  # em, between the letters of the wordmark

# The G on a 5 x 5 grid: "#" is ink, "P" a predicted cell and "B" its uncertainty. The
# masked block covers the P and B cells: the stroke of the G returns as the prediction.
G = (".####", "PB...", "PB.##", "PB..#", ".####")

# The colours of the classes on each ground; the orange is shared.
DARK = ".ink{fill:#E8ECF2}.block{fill:#1A2438}.edge{fill:none;stroke:#6B7A99}"
LIGHT = ".ink{fill:#111827}.block{fill:#E9EDF3}.edge{fill:none;stroke:#7A879C}"
# The header band is navy in both themes. Its block is lighter than mask-fill so
# that it stays visible on the light theme's #111A2A.
NAVY = ".ink{fill:#E8ECF2}.block{fill:#1E2A42}"
ORANGE = ".pred{fill:#F0592A}.band{fill:none;stroke:#F0592A}"


@dataclass(frozen=True)
class Grid:
    pitch: int  # cell plus gap, px
    gap: int  # px; open cells (gap > 0) also show the band and the block edge

    @property
    def cell(self) -> int:
        return self.pitch - self.gap

    @property
    def span(self) -> int:
        return 4 * self.pitch + self.cell


FULL = Grid(pitch=12, gap=1)  # 64 px design: 59 px of cells
LOCKUP = Grid(pitch=5, gap=0)  # 24 px design: 25 px of cells, the header size
FAVICON = Grid(pitch=6, gap=0)  # 32 px design: 30 px of cells


def rect(name: str, x: float, y: float, width: float, height: float) -> str:
    return f'<rect class="{name}" x="{x:g}" y="{y:g}" width="{width:g}" height="{height:g}"/>'


# The SVG elements of the pixel G, with its top-left cell at (x, y).
def mark(grid: Grid, x: int, y: int) -> list[str]:
    p, c, g = grid.pitch, grid.cell, grid.gap
    # The block covers the P and B cells, with a margin of one gap.
    block = (x - g, y + p - g, p + c + 2 * g, 2 * p + c + 2 * g)
    shapes = [rect("block", *block)]
    if g:
        # A 1 px stroke covers whole pixels when it runs through their centres.
        bx, by, bw, bh = block
        shapes.append(rect("edge", bx + 0.5, by + 0.5, bw - 1, bh - 1))
    for row, cells in enumerate(G):
        for col, kind in enumerate(cells):
            cx, cy = x + col * p, y + row * p
            if kind == "#":
                shapes.append(rect("ink", cx, cy, c, c))
            elif kind == "P":
                shapes.append(rect("pred", cx, cy, c, c))
            elif kind == "B" and g:
                shapes.append(rect("band", cx + 0.5, cy + 0.5, c - 1, c - 1))
    return shapes


# The path of the wordmark at cap height cap, and its ink bounds (xmin, ymin, xmax,
# ymax) in SVG units. The ink of the S starts at x = left, on the baseline y = baseline.
def wordmark(
    font: TTFont, cap: int, left: int, baseline: int
) -> tuple[str, tuple[float, ...]]:
    glyphs = font.getGlyphSet()
    names = [font.getBestCmap()[ord(letter)] for letter in WORD]
    first, capital = BoundsPen(glyphs), BoundsPen(glyphs)
    glyphs[names[0]].draw(first)
    glyphs[names[WORD.index("E")]].draw(capital)
    scale = cap / capital.bounds[3]
    path = SVGPathPen(glyphs, ntos=lambda v: f"{v:.2f}".rstrip("0").rstrip("."))
    ink = BoundsPen(glyphs)
    origin = left - scale * first.bounds[0]
    for name in names:
        # Font units point up and SVG units down.
        transform = (scale, 0, 0, -scale, origin, baseline)
        glyphs[name].draw(TransformPen(path, transform))
        glyphs[name].draw(TransformPen(ink, transform))
        origin += scale * (glyphs[name].width + TRACKING * font["head"].unitsPerEm)
    return path.getCommands(), ink.bounds


def document(box: tuple[int, int, int, int], style: str, body: list[str]) -> str:
    view = " ".join(map(str, box))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view}" width="{box[2]}" '
        f'height="{box[3]}" shape-rendering="crispEdges"><title>GSPICE</title>'
        f"<style>{style}</style>{''.join(body)}</svg>\n"
    )


# The pixel G and the wordmark, with gap px between their inks.
def logo(font: TTFont, grid: Grid, gap: int, colours: str) -> str:
    x = y = grid.gap
    letters, (_, top, right, bottom) = wordmark(
        font, grid.span, x + grid.span + gap, y + grid.span
    )
    # The S and the C overshoot the cap height and the baseline by under 1 px.
    ymin, ymax = min(0, math.floor(top)), max(grid.span + 2 * y, math.ceil(bottom))
    letters = f'<path class="ink" d="{letters}" shape-rendering="auto"/>'
    return document(
        (0, ymin, math.ceil(right), ymax - ymin),
        colours + ORANGE,
        [*mark(grid, x, y), letters],
    )


def favicon() -> str:
    style = (
        f".ground{{fill:#0B111C}}{NAVY}{ORANGE}@media (prefers-color-scheme: light)"
        "{.ground{fill:#FFFFFF}.ink{fill:#111827}.block{fill:#E9EDF3}}"
    )
    return document(
        (0, 0, 32, 32), style, [rect("ground", 0, 0, 32, 32), *mark(FAVICON, 1, 1)]
    )


def strip() -> str:
    mean = gspice.simulate(8000, 120, rng=np.random.default_rng(0)).truth.mean(axis=0)
    scaled = (mean - mean.min()) / (mean.max() - mean.min())
    return "[" + ",".join(f"{v:.3f}" for v in scaled) + "]\n"


def main() -> None:
    font = instantiateVariableFont(TTFont(FONT), {"wght": 760, "wdth": 118})
    files = {
        ASSETS / "logo-dark.svg": logo(font, FULL, 7, DARK),
        ASSETS / "logo-light.svg": logo(font, FULL, 7, LIGHT),
        ASSETS / "lockup-dark.svg": logo(font, LOCKUP, 3, NAVY),
        ASSETS / "strip.json": strip(),
        DOCS / "public" / "favicon.svg": favicon(),
    }
    for path, text in files.items():
        path.write_text(text)


if __name__ == "__main__":
    main()
