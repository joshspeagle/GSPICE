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
import * as codeThemes from './plugins/code-themes.mjs'

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
      // The header band is navy in both themes.
      logo: { src: './src/assets/lockup-dark.svg', replacesTitle: true },
      social: [
        { icon: 'document', label: 'Paper', href: 'https://arxiv.org/abs/2511.16754' },
        { icon: 'github', label: 'GitHub', href: 'https://github.com/joshspeagle/GSPICE' },
      ],
      editLink: { baseUrl: 'https://github.com/joshspeagle/GSPICE/edit/main/docs/' },
      components: {
        Header: './src/components/Header.astro',
        PageTitle: './src/components/PageTitle.astro',
        Footer: './src/components/Footer.astro',
      },
      expressiveCode: {
        themes: [codeThemes.dark, codeThemes.light],
        // The tab of each block names its language; shell commands get the same frame.
        defaultProps: {
          overridesByLang: {
            python: { title: 'Python' },
            sh: { title: 'Shell', frame: 'code' },
            text: { title: 'Text' },
          },
        },
        styleOverrides: {
          borderColor: 'var(--gs-rule)',
          frames: {
            frameBoxShadowCssValue: 'none',
            editorTabsMarginInlineStart: '0',
            editorTabBarBackground: 'var(--gs-bg-2)',
            editorTabBarBorderBottomColor: 'var(--gs-rule)',
            editorActiveTabBackground: 'var(--gs-bg-2)',
            editorActiveTabForeground: 'var(--gs-ink-2)',
            editorActiveTabBorderColor: 'transparent',
            editorActiveTabIndicatorTopColor: 'transparent',
            editorActiveTabIndicatorBottomColor: 'transparent',
          },
        },
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
