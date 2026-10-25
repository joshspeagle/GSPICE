// The options of Expressive Code, the code blocks of Starlight. They live in their own
// file because the <Code> component of the home page needs them, and a plugin is not
// serializable in astro.config.mjs.
import { defineEcConfig } from '@astrojs/starlight/expressive-code'
import * as codeThemes from './plugins/code-themes.mjs'
import { outputBlocks } from './plugins/output-blocks.mjs'

export default defineEcConfig({
  themes: [codeThemes.dark, codeThemes.light],
  plugins: [outputBlocks],
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
})
