// The syntax colours of the code blocks (docs/brand.md): keywords and constants in
// steel blue, strings in green, comments in ink-3, everything else in ink. Each pair
// has a contrast of at least 4.5 on the code surface, bg-2.
function theme(name, type, { ink, ink3, surface, keyword, string }) {
  return {
    name,
    type,
    colors: { 'editor.background': surface, 'editor.foreground': ink },
    tokenColors: [
      { scope: ['comment', 'punctuation.definition.comment'], settings: { foreground: ink3 } },
      {
        scope: ['keyword', 'storage', 'constant.language', 'support.type', 'variable.language'],
        settings: { foreground: keyword },
      },
      { scope: ['keyword.operator', 'punctuation'], settings: { foreground: ink } },
      { scope: ['string', 'constant.character.escape'], settings: { foreground: string } },
    ],
  }
}

export const dark = theme('gspice-dark', 'dark', {
  ink: '#E8ECF2',
  ink3: '#7F8AA0',
  surface: '#111A2A',
  keyword: '#9DB8E0',
  string: '#A9D3B0',
})

export const light = theme('gspice-light', 'light', {
  ink: '#111827',
  ink3: '#5F6B7F',
  surface: '#F2F5F9',
  keyword: '#2F5B99',
  string: '#2E6B3F',
})
