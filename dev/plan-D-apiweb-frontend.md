# Plan D — apiweb frontend

**Status:** ready to execute after Plan C (needs a real backend to talk to).
**Depends on:** Plan C endpoints. Implicitly Plans A and B (via Plan C).
**Unblocks:** Plan E (polish, deploy, package).

## Goal

A single-page web UI for the apiweb. Three columns: examples sidebar (left), CodeMirror 6 DecL editor (center), tabbed result panes (right). One process serves it via the FastAPI app's static-files mount; no separate web server.

Deliberately minimal. This is a *demo tool* and a thin convenience over the API; users who want power go to Jupyter.

## Deliverables

```
src/aggregate/apiweb/frontend/        # SOURCE — not shipped in the wheel
    package.json
    package-lock.json                 # committed
    vite.config.js
    index.html
    src/
        main.js                       # entry: bootstraps editor + UI + API client
        api.js                        # fetch() wrappers for /v1/*
        editor.js                     # CM6 editor setup + extensions
        decl-mode.js                  # CM6 StreamLanguage for DecL highlighting
        completion.js                 # autocomplete source hitting /v1/decl/complete
        history.js                    # localStorage ring buffer + nav
        examples.js                   # left sidebar: load /v1/examples, render tree
        panes.js                      # right column: tabs (info/desc/stats/density/plot/pricing/error)
        error-pane.js                 # renders Plan B ErrorReport JSON
        styles/
            app.css
            cm6.css                   # theme overrides
        utils/
            dom.js
            format.js

src/aggregate/apiweb/static/          # BUILD OUTPUT — shipped in wheel
    index.html
    assets/*.{js,css}                 # hashed bundles from Vite

scripts/build-frontend.{ps1,sh}       # convenience wrapper around npm
```

`Makefile`-equivalent at repo root or in `scripts/`:
```
build-frontend.ps1:
    cd src/aggregate/apiweb/frontend
    npm install
    npm run build
```

`.gitignore` additions: `src/aggregate/apiweb/frontend/node_modules/`, `src/aggregate/apiweb/static/` (built artifacts) — though `static/` *is* shipped in the wheel, so it's a build-time artifact rebuilt by `scripts/build-frontend`. (Open knob: commit `static/` for users who install from a git tag, or always require `npm run build` first? See "Open knobs" below.)

## Tech stack

- **Vite** as the build tool. Modern, fast, ESM-native.
- **CodeMirror 6** as the editor.
- **No framework**. Vanilla JS modules. Small enough that React/Preact would be overhead.
- **No CSS framework**. Hand-written CSS, ~200 lines.

`package.json` deps:
```json
{
  "name": "aggregate-apiweb",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build --outDir ../static --emptyOutDir"
  },
  "devDependencies": {
    "vite": "^5.4.0"
  },
  "dependencies": {
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
  base: '/',
  build: {
    outDir: '../static',
    emptyOutDir: true,
    target: 'es2020',
  },
  server: {
    proxy: { '/v1': 'http://127.0.0.1:8000' },
  },
});
```

The proxy makes `npm run dev` work standalone against a backend on :8000.

## DecL syntax mode

CM6's `StreamLanguage` is a regex-state-machine that's perfect for a small DSL. The mode lives in `decl-mode.js` and recognizes:

- **Keywords** — pulled from a constant list seeded from `decl.lark`: `agg`, `port`, `sev`, `dfreq`, `dsev`, `claims`, `mixed`, `poisson`, `negbin`, `binomial`, `bernoulli`, `geometric`, `lognorm`, `gamma`, `pareto`, `cv`, `mean`, `occurrence`, `aggregate`, `net of`, `ceded to`, `note`, `distortion`, `tvar`, `wang`, `ph`, etc.
- **Distortion / severity names** — same list as a separate token class (different color).
- **Numbers** — integer, float, negative, `1/4` fractions (decl-specific).
- **Strings** — single- and double-quoted, plus `note{...}` braced strings.
- **Brackets** — `[`, `]`, `(`, `)` paired.
- **Comments** — `#` to end of line.
- **Identifiers** — anything else.

Seed list comes from a hand-grouped JSON file checked into `src/decl-keywords.json` — easy to keep in sync with `decl.lark` and the backend's `_TERMINAL_LABELS`. Single source-of-truth-by-convention; flagged as an open knob.

## Autocomplete

CM6's autocomplete extension accepts an async `CompletionSource`. Implementation:

```js
async function declCompletions(context) {
    const word = context.matchBefore(/[\w.-]*/);
    if (!word || (word.from === word.to && !context.explicit)) return null;
    const decl = context.state.doc.toString();
    const cursor = context.pos;
    const res = await api.complete(decl, cursor);
    return {
        from: word.from,
        options: res.completions.map(c => ({
            label: c.label,
            type: c.kind,           // 'keyword' | 'identifier' | 'literal'
        })),
    };
}
```

Debounced (~150ms) so each keystroke doesn't hit the server.

## Layout

```
┌──────────────────────────────────────────────────────────────────────┐
│ aggregate · DecL playground                            v1.0.0a13 · ⓘ │
├────────────┬────────────────────────────────┬────────────────────────┤
│ Examples   │                                │ Result                 │
│            │  agg Dice                      │ ┌─────────────────────┐│
│ ▸ A. ...   │      dfreq [3]                 │ │info│desc│stats│df│  ││
│ ▸ B. Basic │      dsev [1:6]                │ │plot│pricing│error│  ││
│ ▾ C. Freq  │                                │ └─────────────────────┘│
│   Poisson  │                                │                        │
│   NegBin   │                                │   <selected pane>      │
│   ...      │                                │                        │
│ ▸ D. Sev   │                                │                        │
│ ▸ E. ...   │                                │                        │
│            │                                │                        │
│            │ [Build (Ctrl-↩)] [Clear] ↑ ↓   │                        │
└────────────┴────────────────────────────────┴────────────────────────┘
```

- Three-column flex layout, columns resize with the window.
- Sidebar collapsible (header click).
- Right column: a tab bar across the top, pane content below.
- Editor footer: Build button, Clear button, history nav (Ctrl-↑ / Ctrl-↓ also bind).
- Header row: title, version (from `/v1/meta`), info button → modal with keyboard shortcuts.

## Result panes

Tabs along the top of the right column. Disabled if the data isn't available for the current object kind (e.g., `pricing` greyed out for a bare `Aggregate`, `kappa` greyed out unless Portfolio).

| Tab | Source | Rendering |
|---|---|---|
| info | `BuildResponse.info` | `<dl>` of key/value pairs |
| description | `BuildResponse.description` | HTML table |
| stats | `BuildResponse.stats` | HTML table |
| density_df | `GET .../density_df?downsample=200` | HTML table, first/last rows + sample |
| plot | `<img src="/v1/objects/{id}/plot?kind=density">` | image, kind selector above |
| pricing | form (`p`, `coc`) + `POST .../pricing_at` | HTML table response |
| error | last `BuildResponse` error detail | structured ErrorReport rendering |

All HTML tables: hand-rolled `<table>` with sticky header, no DataTables-style JS. Cap visible rows; offer "show all" link if truncated.

### Error pane detail

Renders the Plan B `ErrorReport` JSON as:
```
DecL parse error at line 1, column 41

   agg X 100 claims sev lognorm 100 cv 2 mixedd poisson
                                           ^^^^^^

Unexpected identifier 'mixedd'.

Did you mean: mixed?

Expected: 'mixed', 'note', end of input.
```
- Monospace font matching the editor.
- Caret color matches an error highlight color (red-ish).
- Editor itself also gets a `lint` annotation pointing at the error position, so the user sees it in the editor margin too.

## History

`history.js` stores the last 20 successful DecL programs in `localStorage` under key `aggregate-apiweb:history`:

```js
{
  entries: ["agg Dice ...", "agg Lognorm ...", ...],
  cursor: -1   // -1 = composing, 0..N-1 = browsing
}
```

Keybindings (inside editor):
- `Ctrl-Enter` — build current text.
- `Ctrl-↑` — previous in history (replaces editor content).
- `Ctrl-↓` — next in history (or restore current draft if at top).

History updated on every successful build (deduplicated against the latest entry). Failed builds don't add entries.

**On macOS use Cmd-Enter / Cmd-↑ / Cmd-↓**. CM6's keymap supports modKey aliasing.

## API client (`api.js`)

Plain wrapper around `fetch` returning parsed JSON. One function per endpoint. Error responses surface as thrown `ApiError` with status + body, so the UI can switch the active pane to "error" on parse-error responses.

```js
class ApiError extends Error {
    constructor(status, body) { super(body.message || `HTTP ${status}`); this.status = status; this.body = body; }
}

async function build(decl, opts = {}) {
    const r = await fetch('/v1/objects', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({decl, ...opts}),
    });
    const body = await r.json();
    if (!r.ok) throw new ApiError(r.status, body);
    return body;
}

// + complete(), examples(), densityDf(), pricingAt(), plotUrl(), meta()
```

## Examples sidebar

Loads `/v1/examples` once at startup, renders as a collapsible tree by category letter. Each leaf is a button: click → load DecL into editor (does not auto-build). Hover shows the `note{...}` text as a tooltip.

Each category collapsed by default except `A`, to keep the sidebar scannable at first paint.

## Static-files mount

FastAPI's `StaticFiles` mounts `apiweb/static/` at `/`. The Vite-built `index.html` is the root. Bundle assets land under `/assets/`. In `app.py`:

```python
from fastapi.staticfiles import StaticFiles
from importlib.resources import files

def create_app():
    app = FastAPI(...)
    register_routes(app)
    static_dir = files("aggregate.apiweb").joinpath("static")
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
    return app
```

`html=True` makes `/` serve `index.html` and falls back to `index.html` for unknown paths (SPA-style).

## Verification steps

1. **Dev loop:**
   - Terminal 1: `uv run aggregate-server --port 8000`
   - Terminal 2: `cd src/aggregate/apiweb/frontend && npm install && npm run dev` (proxies `/v1` to :8000)
   - Browser: `http://localhost:5173` — editor renders, example sidebar populates.
2. **Manual demo flow:** open the page; click `K. Tweedie / Tweedie01`; press Ctrl-Enter; tabs populate; plot tab shows the Tweedie density.
3. **Autocomplete:** type `agg X 100 ` and pause; suggestions appear including `claims`.
4. **Error case:** type `agg X 100 claims mixedd poisson`; build; error tab activates with caret + suggestion.
5. **History:** build 3 things, press Ctrl-↑ twice; previous program loads.
6. **Production build:** `npm run build`; `uv run aggregate-server` (no Vite); `http://localhost:8000/` serves the bundled SPA.
7. **Mobile / narrow window sanity check** (it's a desktop tool but should at least not break at 1280×720).

## Open knobs for execution

- **Commit built `static/` or not.** Two options:
  - *Commit it.* Anyone cloning the repo gets a working `aggregate-server` immediately. Cost: bundles in git history, churn on every frontend change.
  - *Don't commit, build in CI / packaging step.* Lighter repo. Requires `npm` available at packaging time. Plan E ties the frontend build into the wheel-building step.
  - I recommend **don't commit, build during packaging.** Cleanest.
- **DecL keyword list source.** I propose hand-curated `src/decl-keywords.json` mirrored against `_TERMINAL_LABELS`. Alternative: server endpoint `/v1/decl/keywords` returns the list dynamically. The dynamic option drifts less but adds latency at page load.
- **Plot in browser.** Currently `<img src=...>`. No interactivity (no zoom/pan). Adequate for a demo; if you want pan/zoom later, swap for an `<svg>` from matplotlib or move to Plotly. Out of scope for v0.
- **CodeMirror theme.** Default light theme works; if you want it to feel "aggregate-ey" we can author a small theme using the `aliceblue`/`lightsteelblue` palette to match the plot style. Maybe a v1.1 polish item.
- **Persisted UI state.** Sidebar collapsed state, active tab, etc. — store in localStorage? Trivial to add; default to "no, fresh each load" for simplicity.

## Out of scope

- React / Preact / Svelte / any framework.
- Service worker / offline mode.
- Login / user accounts.
- Per-user history (it's localStorage = per-browser).
- Multi-buffer editor or file management.
- Plot interactivity (zoom, pan, tooltips).
- Internationalization.
- Theme switcher.

## File-by-file checklist (for execution)

1. Create `src/aggregate/apiweb/frontend/` skeleton: `package.json`, `vite.config.js`, `index.html`, `src/main.js`.
2. `cd src/aggregate/apiweb/frontend && npm install` — populate `package-lock.json`, commit it.
3. Implement `api.js` — endpoint wrappers + `ApiError`.
4. Implement `decl-mode.js` — `StreamLanguage` definition + keyword list.
5. Implement `editor.js` — CM6 setup, mode, autocomplete, keymap.
6. Implement `completion.js` — debounced async completion source.
7. Implement `history.js` — localStorage ring buffer + nav functions.
8. Implement `examples.js` — sidebar tree.
9. Implement `panes.js` — tab manager + pane renderers.
10. Implement `error-pane.js` — structured ErrorReport renderer.
11. Implement `main.js` — wires everything together, registers global hotkeys.
12. Style pass: `app.css`, `cm6.css`.
13. `npm run build` → outputs to `src/aggregate/apiweb/static/`.
14. Wire `StaticFiles` mount in backend `app.py`. (May be partly done in Plan C; this step ensures the path resolves to the built bundle.)
15. Manual verification per steps above.

## Recovery / rollback

Frontend tree is entirely net-new. Delete `src/aggregate/apiweb/frontend/` and `src/aggregate/apiweb/static/`, revert the `StaticFiles` mount in `app.py` (if added in this plan rather than C), and the backend continues to work API-only.
