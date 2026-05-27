// Vite build config for the aggregate web SPA.
//
// Build output lands at ../src/aggregate/api/static/ -- the directory
// the FastAPI `StaticFiles` mount in src/aggregate/api/app.py serves
// at '/' when present. That gives the same-origin deploy mode out of
// the box: a single `aggregate-api` process serves both the SPA and
// the /v1 endpoints.
//
// `base: './'` keeps asset URLs relative so the bundle works whether
// it's mounted at '/' (same-origin) or at a sub-path (e.g. a reverse
// proxy installing it under /aggregate/).
//
// The dev server proxies /v1/* to a local backend on :8000 -- the
// SPA itself talks to /v1 via fetch() without any base URL juggling,
// matching the production behavior.

import { defineConfig } from 'vite';

export default defineConfig({
    base: './',
    build: {
        outDir: '../src/aggregate/api/static',
        emptyOutDir: true,
        target: 'es2020',
        // Keep the assets/ subdir name stable for the FastAPI mount.
        assetsDir: 'assets',
        sourcemap: false,
    },
    server: {
        port: 5173,
        proxy: {
            '/v1': {
                target: 'http://127.0.0.1:8000',
                changeOrigin: false,
            },
        },
    },
});
