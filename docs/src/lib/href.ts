// Addresses of the site's pages under its base, /GSPICE/, for components; Markdown links
// keep their literal addresses, which the links validator checks.
export const href = (path: string) => `${import.meta.env.BASE_URL.replace(/\/$/, '')}/${path}`

// The entry of a public name of gspice in the API reference.
export const api = (name: string) => href(`reference/api/#gspice.${name}`)
