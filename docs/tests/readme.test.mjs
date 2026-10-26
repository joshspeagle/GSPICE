// Checks that the figure of README.md is current: docs/scripts/readme.mjs writes it from
// the quickstart data and the styles of the site.
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'

import { figures } from '../scripts/readme.mjs'

for (const [name, text] of Object.entries(figures())) {
  test(`docs/readme/${name} is current`, () => {
    const file = readFileSync(new URL(`../readme/${name}`, import.meta.url), 'utf8')
    assert.equal(file, text, `docs/readme/${name} is out of date: run node docs/scripts/readme.mjs`)
  })
}
