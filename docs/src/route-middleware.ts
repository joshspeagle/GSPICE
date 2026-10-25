import { defineRouteMiddleware } from '@astrojs/starlight/route-data'

// Starlight renders the phone menu only with a sidebar, which a splash page lacks, so
// every page gets one. custom.css hides it on wide screens on the splash pages.
export const onRequest = defineRouteMiddleware(({ locals }) => {
  locals.starlightRoute.hasSidebar = true
})
