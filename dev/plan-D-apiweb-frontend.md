# Plan D — web SPA frontend

**Status:** ready to execute after Plan C (needs a real backend to talk to).
**Depends on:** Plan C endpoints. Implicitly Plans A and B (via Plan C).
**Unblocks:** Plan E (polish, deploy, package).

## Goal

A single-page Bootstrap 5 web UI for the aggregate API. Standalone SPA — Vite-bundled HTML+JS+CSS that loads in any browser and talks to the API via `fetch('/v1/...')`. Deployable:

- **Same-origin** — copied into `src/aggregate/api/static/` and served by the FastAPI `StaticFiles` mount. Default.
- **Split-origin** — uploaded to a static host (e.g. `mynl.com/aggregate/`); calls the API at a configured base URL with CORS enabled on the backend.

Visual style matches the existing mynl.com sites: light navbar, serif body (STIX Two Text), sans-serif headings (Inter), restrained tables, `#a81313` for code.

Deliberately minimal. This is a *demo tool* and a thin convenience over the API; users who want power go to Jupyter.

## Tree

Top-level `web/` folder, sibling to `src/`:

```
web/                                  # SOURCE — Node project, top-level, NOT in src/
    package.json
    package-lock.json                 # committed
    vite.config.js
    index.html                        # Bootstrap 5 SPA shell
    public/                           # static assets copied to bundle root
        favicon.ico                   # user-supplied
        favicon.svg                   # user-supplied (preferred)
        apple-touch-icon.png          # user-supplied (180×180)
        logo.png                      # user-supplied; navbar-brand
    src/
        main.js                       # entry: bootstraps editor + UI + API client
        api.js                        # fetch() wrappers for /v1/*
        config.js                     # API_BASE_URL resolution (env-aware)
        editor.js                     # CM6 editor setup + extensions
        decl-mode.js                  # CM6 StreamLanguage for DecL highlighting
        completion.js                 # autocomplete source hitting /v1/decl/complete
        history.js                    # localStorage ring buffer + nav
        examples.js                   # examples dropdown loader
        actions.js                    # per-button fetch dispatch
        renderers.js                  # JSON/table → DOM (Bootstrap table classes)
        plot-pane.js                  # <img src=...svg|png> render
        pricing-pane.js               # pricing-at form + result table
        error-pane.js                 # renders Plan B ErrorReport JSON
        styles/
            site.css                  # serif body, Inter headings, code color
            cm6.css                   # CodeMirror theme overrides
        utils/
            dom.js
            format.js
            debounce.js
```

Build output lands at `src/aggregate/api/static/` for the same-origin install path. Both `web/node_modules/` and the build target are `.gitignored`; the Plan E packaging step rebuilds before `uv build`.

`scripts/build-web.{ps1,sh}` wraps `npm install && npm run build`.

## Tech stack

- **Bootstrap 5.3** for CSS + components (no jQuery dependency; native ES module imports).
- **Vite** for the build (modern, fast, ESM-native).
- **CodeMirror 6** for the DecL editor.
- **No JS framework** (no React/Vue/Svelte). Vanilla ES modules.

`package.json`:
```json
{
  "name": "aggregate-web",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build --outDir ../src/aggregate/api/static --emptyOutDir"
  },
  "devDependencies": {
    "vite": "^5.4.0"
  },
  "dependencies": {
    "bootstrap": "^5.3.0",
    "@popperjs/core": "^2.11.0",
    "codemirror": "^6.0.1",
    "@codemirror/view": "^6.30.0",
    "@codemirror/state": "^6.4.0",
    "@codemirror/commands": "^6.6.0",
    "@codemirror/language": "^6.10.0",
    "@codemirror/autocomplete": "^6.18.0",
    "@codemirror/lint": "^6.8.0",
    "@codemirror/search": "^6.5.0"
  }
}
```

`vite.config.js`:
```js
import { defineConfig } from 'vite';
export default defineConfig({
  base: './',                          // relative paths so bundle works at any subpath
  build: {
    outDir: '../src/aggregate/api/static',
    emptyOutDir: true,
    target: 'es2020',
  },
  server: {
    proxy: { '/v1': 'http://127.0.0.1:8000' },
  },
});
```

The `/v1` proxy makes `npm run dev` work standalone against a backend on :8000. For production same-origin deploys the proxy is irrelevant — the SPA and API share a host.

For split-origin deploys: `src/config.js` reads `import.meta.env.VITE_API_BASE_URL` at build time, falling back to '' (same-origin). Build for split deploy: `VITE_API_BASE_URL=https://api.mynl.com npm run build`.

## Layout (Bootstrap 5)

```
┌─────────────────────────────────────────────────────────────────────┐
│ <nav class="navbar navbar-expand-lg bg-light">                      │
│   [logo.png] aggregate · DecL playground       v1.0.0a14 · [ⓘ help] │
├─────────────────────────────────────────────────────────────────────┤
│ container-fluid px-4 py-3                                           │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ row gx-2 align-items-center                                    │ │
│  │  [Examples ▾]   [Knowledge base ▾]   [⚙ Options ▾]   [Build]   │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ col-lg-10 offset-lg-1   (~83% width, centered on wide screens) │ │
│  │  ┌──────────────────────────────────────────────────────────┐  │ │
│  │  │  <CodeMirror DecL editor>                                │  │ │
│  │  │                                                          │  │ │
│  │  └──────────────────────────────────────────────────────────┘  │ │
│  │  <div class="btn-toolbar mt-2">                                │ │
│  │    [info] [describe] [stats_df] [density_df] [plot] [more ▾]   │ │
│  │  </div>                                                        │ │
│  │  more ▾ →  [kappa]  [pricing at…]  [cdf]  [qq]                 │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ col-lg-10 offset-lg-1                                          │ │
│  │  <div id="output">  ← replaced on each action button click     │ │
│  │     <table class="table table-sm"> ... </table>                │ │
│  │     or <img src="/v1/.../plot?...&format=svg">                 │ │
│  │     or <pre class="parse-error">...</pre>                      │ │
│  │  </div>                                                        │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

Notes:
- `col-lg-10 offset-lg-1` is the Bootstrap idiom for "~83% width centered" — what you wanted as "reasonable size, not 100%".
- Single output region below; each action button replaces its contents. No tabs.
- Dropdowns use Bootstrap 5 native (no jQuery) — `data-bs-toggle="dropdown"`.
- `more ▾` is a `btn-group` with a dropdown menu listing secondary actions.
- The build button is right-aligned in the header row; clicking it submits whatever's in the editor.

## Action button model

Per-button fetches against the cached object. After `POST /v1/objects` returns `{id}`, the SPA stores `currentId`. Each action button:

1. Disables itself + shows a tiny spinner glyph.
2. Fires its endpoint via `api.js`.
3. Pipes the response through the right renderer in `renderers.js` / `plot-pane.js` / `pricing-pane.js`.
4. Writes the result into `#output`.
5. Re-enables itself.

```js
// actions.js sketch
const ACTIONS = {
    info:        () => render.kv     (api.info(currentId)),
    describe:    () => render.table  (api.description(currentId)),
    stats_df:    () => render.table  (api.stats_df(currentId)),
    density_df:  () => render.table  (api.density_df(currentId, {downsample: 200})),
    kappa:       () => render.table  (api.kappa(currentId, {downsample: 200})),
    plot:        () => render.image  (api.plotUrl(currentId, {kind: 'density', format: 'svg'})),
    cdf:         () => render.image  (api.plotUrl(currentId, {kind: 'cdf', format: 'svg'})),
    qq:          () => render.image  (api.plotUrl(currentId, {kind: 'qq', format: 'svg'})),
    pricing:     () => renderPricingForm(currentId),
};
```

Buttons for Portfolio-only actions (`kappa`, multi-unit pricing rows) get disabled when `currentKind === 'agg'`. Determined from the BuildResponse `kind` field.

## Renderers (Bootstrap table classes)

`renderers.js`:
```js
function tableFromFrame({columns, rows}) {
    const t = document.createElement('table');
    t.className = 'table table-sm table-borderless border-top border-bottom';
    // <thead><tr><th>col</th>...</tr></thead><tbody>...</tbody>
    // numeric cells get class="text-end font-monospace"
    return t;
}
function kvList(info) {
    // <dl class="row"><dt class="col-sm-3">k</dt><dd class="col-sm-9">v</dd></dl>
}
function imageView(url) {
    // <img src={url} class="img-fluid" alt="aggregate plot">
}
```

Numeric formatting: integer | decimal | scientific picked by magnitude. `format.js` carries a single `fmt(value)` that does this.

## DecL syntax mode

CM6's `StreamLanguage` is a regex-state-machine that's perfect for a small DSL. The mode lives in `decl-mode.js` and recognizes:

- **Keywords** — seeded from `decl.lark`: `agg`, `port`, `sev`, `dfreq`, `dsev`, `claims`, `mixed`, `poisson`, `negbin`, `binomial`, `bernoulli`, `geometric`, `lognorm`, `gamma`, `pareto`, `cv`, `mean`, `occurrence`, `aggregate`, `net of`, `ceded to`, `note`, `distortion`, `tvar`, `wang`, `ph`, etc.
- **Distortion / severity names** — separate token class (different color).
- **Numbers** — integer, float, negative, `1/4` fractions.
- **Strings** — single- and double-quoted, plus `note{...}` braced strings.
- **Brackets** — `[`, `]`, `(`, `)`.
- **Comments** — `#` to end of line.
- **Identifiers** — anything else.

Seed list comes from a hand-grouped JSON file at `web/src/decl-keywords.json`; mirrored against backend's `_TERMINAL_LABELS`.

## Autocomplete

CM6's autocomplete extension with a debounced async source hitting `/v1/decl/complete`:

```js
async function declCompletions(context) {
    const word = context.matchBefore(/[\w.-]*/);
    if (!word || (word.from === word.to && !context.explicit)) return null;
    const decl = context.state.doc.toString();
    const cursor = context.pos;
    const res = await api.complete(decl, cursor);
    return {
        from: word.from,
        options: res.completions.map(c => ({ label: c.label, type: c.kind })),
    };
}
```

Debounced (~150 ms).

## Examples dropdown

Loads `/v1/examples` once on startup. Renders as a Bootstrap dropdown with a nested submenu by category letter:

```html
<div class="dropdown">
  <button class="btn btn-outline-secondary dropdown-toggle" data-bs-toggle="dropdown">
    Examples
  </button>
  <ul class="dropdown-menu">
    <li><h6 class="dropdown-header">A. Frequency</h6></li>
    <li><a class="dropdown-item" href="#" data-example="Dice">Dice</a></li>
    <li><a class="dropdown-item" href="#" data-example="Poisson">Poisson</a></li>
    <li><hr class="dropdown-divider"></li>
    <li><h6 class="dropdown-header">B. Severity</h6></li>
    ...
  </ul>
</div>
```

Click → load the DecL text into the editor (does not auto-build). Hover shows the `note{...}` text as a tooltip via `data-bs-toggle="tooltip"`.

## Pricing pane

When the "pricing at" item is clicked, `#output` is replaced with an inline form:

```html
<form id="pricing-form" class="row g-3">
  <div class="col-sm-3"><label>p value</label><input name="p" class="form-control" value="0.995"></div>
  <div class="col-sm-3"><label>ccoc</label><input name="ccoc" class="form-control" value="0.10"></div>
  <div class="col-sm-2 align-self-end"><button class="btn btn-primary">Calculate</button></div>
</form>
<div id="pricing-result" class="mt-3"></div>
```

Submit → `POST /v1/objects/{id}/pricing_at` → render result table into `#pricing-result`. Form stays visible so you can iterate on p / ccoc.

## API client (`api.js`)

```js
import { API_BASE } from './config.js';

class ApiError extends Error {
    constructor(status, body) { super(body.message || `HTTP ${status}`); this.status = status; this.body = body; }
}

async function _json(method, path, body) {
    const r = await fetch(API_BASE + path, {
        method,
        headers: body ? {'Content-Type': 'application/json'} : {},
        body: body ? JSON.stringify(body) : undefined,
    });
    const data = await r.json();
    if (!r.ok) throw new ApiError(r.status, data);
    return data;
}

export const api = {
    build:       (decl, opts={}) => _json('POST', '/v1/objects', {decl, ...opts}),
    info:        (id)            => _json('GET',  `/v1/objects/${id}/info`),
    description: (id)            => _json('GET',  `/v1/objects/${id}/description`),
    stats_df:    (id)            => _json('GET',  `/v1/objects/${id}/stats_df`),
    density_df:  (id, p={})      => _json('GET',  `/v1/objects/${id}/density_df?${qs(p)}`),
    kappa:       (id, p={})      => _json('GET',  `/v1/objects/${id}/kappa?${qs(p)}`),
    pricing_at:  (id, body)      => _json('POST', `/v1/objects/${id}/pricing_at`, body),
    complete:    (decl, cursor)  => _json('POST', '/v1/decl/complete', {decl, cursor}),
    examples:    ()              => _json('GET',  '/v1/examples'),
    meta:        ()              => _json('GET',  '/v1/meta'),
    plotUrl:     (id, p={})      => `${API_BASE}/v1/objects/${id}/plot?${qs(p)}`,
};
```

`config.js`:
```js
export const API_BASE = import.meta.env.VITE_API_BASE_URL || '';
```

Build with `VITE_API_BASE_URL=https://api.mynl.com npm run build` for split-origin deploys.

## Site CSS

`web/src/styles/site.css` brings back the mynl visual language on top of Bootstrap 5:

```css
:root {
  --agg-code: #a81313;
  --agg-rule: #1a1a1a;
}
html { font-size: 17px; }
body {
  font-family: "STIX Two Text", "Calisto MT", serif;
  hyphens: auto;
}
h1, h2 { font-family: Inter, sans-serif; }
h3, h4, h5, h6 { font-family: Helvetica, sans-serif; }
code, pre, .font-monospace {
  font-family: "Cascadia Mono", Menlo, Consolas, monospace;
  color: var(--agg-code);
}
.table { border-top: 1px solid var(--agg-rule); border-bottom: 1px solid var(--agg-rule); }
.table thead th { border-bottom: 1px solid var(--agg-rule); font-family: Helvetica, sans-serif; }
.navbar-brand { font-family: Helvetica, sans-serif; }
```

`cm6.css` themes the editor with the same palette as `aggregate.mplstyle` (aliceblue / lightsteelblue accents) so the editor visually rhymes with rendered plots.

## Error rendering

Plan B's `ErrorReport` JSON rendered as:

```
DecL parse error at line 1, column 41

   agg X 100 claims sev lognorm 100 cv 2 mixedd poisson
                                           ^^^^^^

Unexpected identifier 'mixedd'.

Did you mean: mixed?

Expected: 'mixed', 'note', end of input.
```

In a `<pre class="parse-error">` with the editor's monospace font; caret red. CM6 `lint` extension also marks the error position in the editor margin.

## History

`history.js` stores last 20 successful DecL programs in `localStorage` under key `aggregate-web:history`:

```js
{ entries: [...], cursor: -1 }
```

Keybindings (CM6 keymap):
- `Mod-Enter` — build current text.
- `Mod-ArrowUp` — previous in history.
- `Mod-ArrowDown` — next in history.

(`Mod` is Ctrl on Windows/Linux, Cmd on macOS — CM6 handles the aliasing.)

Updated on successful build (deduplicated against latest). Failed builds don't add entries.

## Favicons & branding — where you drop files

All in `web/public/`:

| File | Purpose | Notes |
|---|---|---|
| `favicon.ico` | classic favicon (fallback) | optional if you ship the SVG |
| `favicon.svg` | preferred favicon | scales for all sizes; referenced by `<link rel="icon" type="image/svg+xml" href="/favicon.svg">` |
| `apple-touch-icon.png` | iOS home-screen icon | 180×180 |
| `logo.png` | navbar-brand image | ~96×40 like `sjmm.png` on mynl.com |
| `og-image.png` | (optional) Open Graph preview | 1200×630 |

Vite copies `public/` verbatim into the build root, so `/favicon.svg` resolves regardless of subpath. The HTML `<link>` tags in `index.html` reference them by absolute path; I wire those once you drop the files.

## Standalone SPA / CORS

Confirmed: the frontend is a standalone SPA. After `npm run build` the contents of `src/aggregate/api/static/` are a complete static site — `index.html` + hashed `assets/`. Deploy targets:

- **Same-origin (default)** — backend's `StaticFiles` mount serves it at `/`. No CORS needed.
- **Split-origin** — copy `dist` to any static host. Set `VITE_API_BASE_URL` at build time so the bundle calls the right API. Backend `AGGAPI_CORS_ORIGINS` env var lists the SPA origin(s); Plan C's `cors.py` middleware kicks in.
- **`file://`** — works for the SPA itself, but CORS rules around `null` origin make calling the API from `file://` flaky. Not a supported deploy mode; document it as "use a local server".

## Verification steps

1. **Dev loop:**
   - Terminal 1: `uv run aggregate-api --port 8000`
   - Terminal 2: `cd web && npm install && npm run dev`
   - Browser: `http://localhost:5173` — Bootstrap navbar, editor, dropdowns render.
2. **Manual demo flow:** click `Examples ▾ → K. Tweedie / Tweedie01`; press Ctrl-Enter; click `plot` — Tweedie density renders as SVG.
3. **Autocomplete:** type `agg X 100 ` and pause; suggestions appear including `claims`.
4. **Error case:** type `agg X 100 claims mixedd poisson`; build; output shows formatted parse-error with caret + suggestion.
5. **History:** build 3 things, press Ctrl-↑ twice; previous program loads in the editor.
6. **Pricing pane:** Build a Portfolio, click `more ▾ → pricing at`; enter p=0.995 ccoc=0.10; click Calculate; result table renders.
7. **Production build:** `npm run build`; `uv run aggregate-api` (no Vite); `http://localhost:8000/` serves the same UI same-origin.
8. **Split-origin build:** `VITE_API_BASE_URL=http://localhost:8000 npm run build`; serve the `static/` dir from a different port (e.g. `python -m http.server 5500 -d src/aggregate/api/static`); set `AGGAPI_CORS_ORIGINS=http://localhost:5500`; verify the SPA at :5500 talks to the API at :8000.
9. **Plot format toggle:** add a temporary URL or button to fetch `?format=png`; confirm raster comes back and renders.
10. **Narrow window sanity check** (Bootstrap should reflow at <992px).

## Open knobs for execution

- **DecL keyword list source.** Hand-curated `web/src/decl-keywords.json` mirrored against `_TERMINAL_LABELS`. Alternative: server endpoint `/v1/decl/keywords` returns the list dynamically (drifts less, adds page-load latency).
- **Plot interactivity.** `<img>` with no zoom/pan. Adequate for v0. If you want pan/zoom later: switch to Plotly for selected kinds, or use [svg-pan-zoom](https://github.com/ariutta/svg-pan-zoom) on inline SVG. Out of scope for v0.
- **Persisted UI state.** Active example, last DecL — store in localStorage? Default to "no, fresh each load".
- **CodeMirror theme.** Aiming for an "aggregate-ey" theme using aliceblue/lightsteelblue from the mplstyle. v1.1 polish.
- **Example search box.** Pure client-side filter over the loaded examples list — easy add at v1.1.

## Out of scope

- React / Preact / Vue / Svelte.
- Service worker / offline mode.
- Login / user accounts.
- Per-user history (it's localStorage = per-browser).
- Multi-buffer editor.
- Plot interactivity (zoom, pan, tooltips).
- Internationalization.
- Theme switcher.

## File-by-file checklist (for execution)

1. Create `web/` skeleton: `package.json`, `vite.config.js`, `index.html`, `public/`, `src/main.js`.
2. `cd web && npm install` — populate `package-lock.json`, commit it.
3. Implement `config.js` and `api.js`.
4. Implement `decl-mode.js` — `StreamLanguage` definition + keyword list.
5. Implement `editor.js` — CM6 setup, mode, autocomplete, keymap.
6. Implement `completion.js` — debounced async completion source.
7. Implement `history.js`.
8. Implement `examples.js` — dropdown loader.
9. Implement `actions.js` — per-button fetch dispatch.
10. Implement `renderers.js`, `plot-pane.js`, `pricing-pane.js`, `error-pane.js`.
11. Implement `main.js` — wire navbar, editor, dropdowns, output area, global hotkeys.
12. Style pass: `site.css`, `cm6.css`.
13. `npm run build` → outputs to `src/aggregate/api/static/`.
14. Wire `StaticFiles` mount in backend `app.py` (covered by Plan C — confirm it resolves to the built bundle).
15. Manual verification per steps above.

## Recovery / rollback

`web/` and `src/aggregate/api/static/` are entirely net-new. Delete both, revert the `StaticFiles` mount in `app.py` (if added in this plan rather than C), and the backend continues to work API-only.
