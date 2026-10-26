// A rehype plugin that gives each page the term cards of the glossary entries it links
// to: for each link to /GSPICE/concepts/glossary/#term-<name>, a hidden card with the
// entry, which describes the link (aria-describedby) and which
// src/components/TermCards.astro shows on hover or focus. The card is a tooltip, so it
// holds nothing focusable: the links of the entry become plain text, and the term link
// itself leads to the entry.
// The cards come from the glossary page, rendered once per build with the remark and
// rehype plugins of the site.
import { readFileSync } from 'node:fs'
import { createMarkdownProcessor } from '@astrojs/markdown-remark'
import { fromHtml } from 'hast-util-from-html'

const LINK = /\/glossary\/#(term-[A-Za-z0-9-]+)$/

function walk(node, visit) {
  visit(node)
  for (const child of node.children ?? []) walk(child, visit)
}

const text = (node) => (node.type === 'text' ? node.value : (node.children ?? []).map(text).join(''))

// A copy of nodes with each link turned into a span of its content.
const unlinked = (nodes) =>
  nodes.map((node) =>
    node.type !== 'element'
      ? node
      : node.tagName === 'a'
        ? { type: 'element', tagName: 'span', properties: {}, children: unlinked(node.children) }
        : { ...node, children: unlinked(node.children ?? []) },
  )

// The entries of the glossary, by id: the term of each <dt id="term-..."> and the
// content of the <dd> after it.
async function entries(file, plugins) {
  const source = readFileSync(file, 'utf-8').replace(/^---\n[\s\S]*?\n---\n/, '')
  const processor = await createMarkdownProcessor(plugins)
  const tree = fromHtml((await processor.render(source)).code, { fragment: true })
  const found = new Map()
  let term = null
  walk(tree, (node) => {
    if (node.tagName === 'dt') term = { id: node.properties.id, name: text(node).trim() }
    if (node.tagName === 'dd' && term) {
      found.set(term.id, { name: term.name, body: node.children })
      term = null
    }
  })
  if (!found.size) throw new Error(`term cards: no glossary entries in ${file}`)
  return found
}

// glossary is the path of the glossary page, a Markdown file; remarkPlugins and
// rehypePlugins are those of the site, without this one.
export function rehypeTermCards({ glossary, remarkPlugins, rehypePlugins }) {
  let glossaryEntries
  return async (tree, file) => {
    const ids = new Set()
    walk(tree, (node) => {
      const match = node.tagName === 'a' && String(node.properties?.href ?? '').match(LINK)
      if (!match) return
      ids.add(match[1])
      node.properties.ariaDescribedBy = `gs-card-${match[1]}`
    })
    if (!ids.size) return
    glossaryEntries ??= entries(glossary, { remarkPlugins, rehypePlugins })
    const found = await glossaryEntries
    const cards = [...ids].map((id) => {
      const entry = found.get(id)
      if (!entry) file.fail(`term cards: the glossary has no entry ${id}`)
      return {
        type: 'element',
        tagName: 'div',
        properties: { className: ['gs-term-card'], id: `gs-card-${id}`, role: 'tooltip', hidden: true },
        children: [
          { type: 'element', tagName: 'p', properties: { className: ['gs-term-name'] }, children: [{ type: 'text', value: entry.name }] },
          ...unlinked(entry.body),
        ],
      }
    })
    tree.children.push({
      type: 'element',
      tagName: 'div',
      properties: { className: ['gs-term-cards', 'not-content'], dataPagefindIgnore: 'all' },
      children: cards,
    })
  }
}
