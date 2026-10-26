import { unified } from '@astrojs/markdown-remark'
import starlight from '@astrojs/starlight'
import { defineConfig } from 'astro/config'
import rehypeCitation from 'rehype-citation'
import rehypeKatex from 'rehype-katex'
import remarkHeadingId from 'remark-heading-id'
import remarkMath from 'remark-math'
import linksValidator from 'starlight-links-validator'
import { rehypeKnownCitations, rehypeReferencesPage } from './plugins/citations.mjs'

const base = '/GSPICE'
// The Sphinx site served each page at <page>.html. Each old page maps to its current
// page; a page that moved also keeps its old address <page>/.
const sphinx = {
  installation: 'start/install',
  quickstart: 'start/quickstart',
  references: 'references',
  'reference/api': 'reference/api',
  'reference/legacy': 'reference/legacy',
  'tutorials/calibration': 'guides/checks',
  'tutorials/fitting': 'tutorials/fitting',
  'tutorials/infill': 'guides/infill',
  'tutorials/migration': 'tutorials/migration',
  'tutorials/models': 'guides/models',
  'tutorials/pixelwise': 'guides/outliers',
  'tutorials/repair': 'guides/repair',
  'tutorials/saving': 'guides/save',
  'background/diagnostics': 'concepts/diagnostics',
  'background/fitting': 'concepts/fitting',
  'background/glossary': 'concepts/glossary',
  'background/method': 'concepts/model',
  'background/prediction': 'concepts/prediction',
  'background/repair': 'concepts/detection',
}
const redirects = Object.fromEntries(
  Object.entries(sphinx).flatMap(([old, page]) => [
    [`/${old}.html`, `${base}/${page}/`],
    ...(old === page ? [] : [[`/${old}/`, `${base}/${page}/`]]),
  ]),
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
      // The header band is navy in both themes.
      logo: { src: './src/assets/lockup-dark.svg', replacesTitle: true },
      social: [
        { icon: 'document', label: 'Paper', href: 'https://arxiv.org/abs/2511.16754' },
        { icon: 'github', label: 'GitHub', href: 'https://github.com/joshspeagle/GSPICE' },
      ],
      editLink: { baseUrl: 'https://github.com/joshspeagle/GSPICE/edit/main/docs/' },
      routeMiddleware: './src/route-middleware.ts',
      components: {
        Header: './src/components/Header.astro',
        Hero: './src/components/Hero.astro',
        PageTitle: './src/components/PageTitle.astro',
        Footer: './src/components/Footer.astro',
      },
      customCss: [
        '@fontsource-variable/archivo/wdth.css',
        '@fontsource-variable/archivo/wdth-italic.css',
        '@fontsource/ibm-plex-mono/400.css',
        '@fontsource/ibm-plex-mono/500.css',
        'katex/dist/katex.min.css',
        './src/styles/custom.css',
      ],
      sidebar: [
        {
          label: 'Getting started',
          items: ['start/fit', 'start/install', 'start/quickstart', 'start/spectra', 'start/results'],
        },
        {
          label: 'Guides',
          items: [
            'guides/infill',
            'guides/outliers',
            'guides/repair',
            'guides/models',
            'guides/checks',
            'guides/save',
            'guides/troubleshooting',
          ],
        },
        { label: 'Tutorials', items: ['tutorials/fitting', 'tutorials/migration'] },
        {
          label: 'Concepts',
          items: [
            'concepts/model',
            'concepts/fitting',
            'concepts/prediction',
            'concepts/held-out',
            'concepts/detection',
            'concepts/diagnostics',
            'concepts/numerics',
            'concepts/glossary',
          ],
        },
        { label: 'Reference', items: ['reference/api', 'reference/legacy', 'references'] },
      ],
      plugins: [linksValidator({ sameSitePolicy: 'error' })],
    }),
  ],
})
