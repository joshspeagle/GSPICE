// Rehype plugins that run after rehype-citation, which renders [@key] and @key from a
// BibTeX file.
import { citationRE } from 'rehype-citation/node/src/regex.js'

// Elements whose text is code, where an @ is not a citation.
const CODE = new Set(['code', 'pre', 'script', 'style'])

function walk(node, visit, parent) {
  visit(node, parent)
  for (const child of node.children ?? []) walk(child, visit, node)
}

// rehype-citation leaves as text a citation with an unknown key, and one outside a
// paragraph, a list item or a table cell, such as in a heading or in emphasis. Fail the
// build on any citation left in text that is not code.
export function rehypeKnownCitations() {
  const pattern = new RegExp(citationRE.source, 'u')
  function check(node, file) {
    if (node.type === 'element' && CODE.has(node.tagName)) return
    if (node.type === 'text') {
      const match = node.value.match(pattern)
      if (match) file.fail(`Unrendered citation ${match[0].trim()}: an unknown key, or a place that rehype-citation skips`, node)
    }
    for (const child of node.children ?? []) check(child, file)
  }
  return (tree, file) => check(tree, file)
}

// The references page lists every entry, and each citation links there; rehype-citation
// would append the cited entries to each page instead.
export function rehypeReferencesPage({ href }) {
  return (tree, file) => {
    if (file.data.astro?.frontmatter?.noCite) return
    tree.children = tree.children.filter((node) => node.properties?.id !== 'refs')
    walk(tree, (node) => {
      if (node.tagName === 'a' && node.properties.href?.startsWith('#bib-')) {
        node.properties.href = href + node.properties.href
      }
    })
  }
}
