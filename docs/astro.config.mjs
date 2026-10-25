import { unified } from '@astrojs/markdown-remark'
import starlight from '@astrojs/starlight'
import { globSync } from 'node:fs'
import { defineConfig } from 'astro/config'
import rehypeCitation from 'rehype-citation'
import rehypeKatex from 'rehype-katex'
import remarkHeadingId from 'remark-heading-id'
import remarkMath from 'remark-math'
import linksValidator from 'starlight-links-validator'
import { rehypeKnownCitations, rehypeReferencesPage } from './plugins/citations.mjs'

const base = '/GSPICE'
// The Sphinx site served each page at <page>.html: keep those addresses working.
const redirects = Object.fromEntries(
  globSync('**/*.{md,mdx}', { cwd: 'src/content/docs' })
    .map((file) => file.replace(/\.mdx?$/, ''))
    .filter((page) => page !== 'index')
    .map((page) => [`/${page}.html`, `${base}/${page}/`]),
)
const bibliography = './references.bib'

export default defineConfig({
  site: 'https://joshspeagle.github.io',
  base,
  trailingSlash: 'always',
  redirects,
  markdown: {
    processor: unified({
      // remark-heading-id sets the ids of the API headings: {#gspice.fit}.
      remarkPlugins: [remarkMath, remarkHeadingId],
      rehypePlugins: [
        rehypeKatex,
        [rehypeCitation, { bibliography, linkCitations: true }],
        rehypeKnownCitations,
        [rehypeReferencesPage, { href: `${base}/references/` }],
      ],
    }),
  },
  integrations: [
    starlight({
      title: 'GSPICE',
      logo: {
        light: './src/assets/logo-light.svg',
        dark: './src/assets/logo-dark.svg',
        replacesTitle: true,
      },
      editLink: { baseUrl: 'https://github.com/joshspeagle/GSPICE/edit/main/docs/' },
      customCss: ['katex/dist/katex.min.css', './src/styles/custom.css'],
      sidebar: [
        'installation',
        'quickstart',
        {
          label: 'Tutorials',
          items: [
            'tutorials/fitting',
            'tutorials/pixelwise',
            'tutorials/infill',
            'tutorials/repair',
            'tutorials/calibration',
            'tutorials/models',
            'tutorials/saving',
            'tutorials/migration',
          ],
        },
        {
          label: 'Background',
          items: [
            'background/method',
            'background/prediction',
            'background/repair',
            'background/fitting',
            'background/diagnostics',
            'background/glossary',
          ],
        },
        { label: 'Reference', items: ['reference/api', 'reference/legacy', 'references'] },
      ],
      plugins: [linksValidator({ sameSitePolicy: 'error' })],
    }),
  ],
})
