// Runtime configuration for the SPA.
//
// API_BASE is the prefix every fetch() prepends. The empty string is
// the same-origin default -- the SPA and the FastAPI service live on
// the same host:port. For split-origin deploys (SPA on a CDN, api on
// a separate domain), build with VITE_API_BASE_URL set:
//
//     VITE_API_BASE_URL=https://api.mynl.com npm run build
//
// Vite inlines import.meta.env.VITE_* values at build time, so the
// resulting bundle bakes the base URL in -- no runtime config file.

export const API_BASE = import.meta.env.VITE_API_BASE_URL || '';
