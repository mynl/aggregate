# Plan E — api/web polish, deploy, package

**Status:** ready to execute after Plans C and D.
**Depends on:** Plans C (backend at `src/aggregate/api/`) and D (web SPA at `web/`, built into `src/aggregate/api/static/`).
**Unblocks:** end-to-end shipping — `pip install aggregate[api]` → `aggregate-api` → demo or team install.

## Goal

Take the api + web from "works on my dev box" to "installs cleanly on Windows and Linux, runs as a service, can be put behind Caddy, has the small polish details that make a demo feel finished."

## Deliverables

- Modified: `pyproject.toml` — `package-data` declaration for the built web bundle.
- New: `docs/2_user_guides/2_x_api.rst` — user-facing docs page covering install, run, deploy.
- New: `dev/Caddyfile.example` — copy-paste reference config.
- New: `dev/aggregate-api.service.example` — systemd unit reference (Linux).
- New: `dev/aggregate-api-windows.md` — Windows service notes (NSSM or scheduled task).
- New: `scripts/build-web.ps1` and `scripts/build-web.sh` — wrappers around `npm run build`.
- Modified: `README.rst` — short api/web mention with link to the new docs page.
- Small polish edits in the web SPA (page title, keyboard help modal, version display from `/v1/meta`, examples search box).

## Packaging — the wheel must ship the built web bundle

`pyproject.toml`:
```toml
[tool.setuptools.package-data]
aggregate = [
    "*.lark",
    "agg/*.agg",
    "data/*.mplstyle",
    "api/static/**/*",
]
```

Build ordering — the web bundle has to exist before `uv build` runs. Two options:

- **Option A — Build step is a manual prerequisite.** Document "run `scripts/build-web.ps1` before `uv build`." Simple, no build automation. If you forget, `uv build` succeeds but ships an empty `static/` and the same-origin web is broken (the API still works fine).
- **Option B — Build step automated via a build backend hook.** Use `setuptools` with a `cmdclass` override, or switch to `hatchling` with a build hook. Adds complexity but bullet-proof.

Recommend **Option A for v0**. Document loudly in the release checklist. Revisit if a release ever ships an empty bundle.

`scripts/build-web.ps1`:
```powershell
$ErrorActionPreference = 'Stop'
Push-Location "$PSScriptRoot/../web"
try {
    npm install
    npm run build
}
finally {
    Pop-Location
}
```

`scripts/build-web.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../web"
npm install
npm run build
```

For split-origin deploys (frontend on mynl.com, API on api.mynl.com), build with:
```powershell
$env:VITE_API_BASE_URL = "https://api.mynl.com"; npm run build
```
Document this in `docs/2_user_guides/2_x_api.rst`.

## Web SPA polish

Final touches in `web/src/`:

- **Page title.** `<title>aggregate · DecL playground</title>`.
- **Favicon wiring.** `index.html` gets the `<link rel="icon">` tags once you drop the files in `web/public/` (per Plan D).
- **Version banner.** Fetch `/v1/meta` once on load, display version in the navbar (`<span class="navbar-text small">v1.0.0a14</span>`).
- **Keyboard shortcuts modal.** Bootstrap 5 modal triggered by `?` or `Ctrl-/`. Lists: `Ctrl-Enter` (build), `Ctrl-↑/↓` (history), `Ctrl-K` (focus examples), `Esc` (close modals).
- **Inline examples search.** Type-ahead filter at the top of the examples dropdown. Pure client-side filter over the already-loaded list. ~20 lines.
- **Build state UI.** While `POST /v1/objects` is in flight, the Build button shows a Bootstrap spinner and is disabled. Cancel only by waiting for timeout.
- **Plot format toggle.** Small SVG/PNG toggle next to the plot button (defaults to SVG; PNG via `?format=png`).
- **"Copy DecL" affordance.** Single button in the navbar or near the editor that copies the current editor content to the clipboard. Useful when sharing.

## User-facing docs page

New `docs/2_user_guides/2_x_api.rst`, roughly:

```rst
The aggregate API and web demo
==============================

``aggregate.api`` ships a small FastAPI service plus an optional Bootstrap 5
single-page web UI for interactive DecL exploration. It is intended for:

* Demos — quickly show what a Tweedie / Lognormal / Portfolio looks like.
* Small-team installs behind a VPN, fronted by Caddy + auth.
* As a stable JSON API consumed by downstream pricing systems.

It is **not** intended as a public-internet service.

Installation
------------

.. code-block:: bash

    pip install aggregate[api]
    aggregate-api --port 8000

Open ``http://127.0.0.1:8000/`` for the web UI. The OpenAPI / Swagger UI is at
``http://127.0.0.1:8000/docs``.

The web bundle is packaged with the wheel. To rebuild from source:

.. code-block:: bash

    scripts/build-web.ps1     # Windows
    scripts/build-web.sh      # Linux / macOS

Configuration
-------------

Environment variables (prefix ``AGGAPI_``):

.. (table from Plan C config section)

Split-origin deployment
-----------------------

The web SPA can run at a different origin from the API (e.g. ``mynl.com/aggregate``
calling ``api.mynl.com``). Rebuild the web bundle with the API base URL set:

.. code-block:: bash

    cd web
    VITE_API_BASE_URL=https://api.mynl.com npm run build

And on the API side, allow the SPA origin:

.. code-block:: bash

    AGGAPI_CORS_ORIGINS=https://mynl.com aggregate-api

API surface
-----------

.. (table of v1 endpoints; link to /openapi.json)

Plot output formats
-------------------

Plots default to SVG (resolution-independent, ~30–150 KB for typical aggregate
densities). Add ``?format=png`` to any plot URL for a PNG raster instead.

Deploying behind Caddy
----------------------

.. (Caddyfile example, basic-auth notes)

Running as a service
--------------------

Linux (systemd) ...
Windows (NSSM or Task Scheduler) ...

Limits and caveats
------------------

* Single uvicorn worker — concurrent builds queue.
* In-memory cache — restarts wipe state. Audit log persists.
* No built-in authentication. Use Caddy / nginx / wireguard.
* Build timeouts cannot cancel CPU-bound work mid-FFT.
* log2 capped at 18 by default, hard ceiling 20.

Audit log
---------

SQLite at ``$AGGAPI_AUDIT_DB`` (default ``~/.aggregate/api/audit.db``).
Query directly:

.. code-block:: bash

    sqlite3 ~/.aggregate/api/audit.db \
      "SELECT ts, ip, status, decl FROM builds ORDER BY ts DESC LIMIT 20"
```

## Reference deploy configs

`dev/Caddyfile.example` (same-origin):
```
# Demo deployment behind Caddy with basic auth.
agg.internal {
    basicauth {
        # generate with: caddy hash-password
        stephen $2a$14$REPLACE_WITH_HASHED_PASSWORD
    }
    reverse_proxy 127.0.0.1:8000

    log {
        output file /var/log/caddy/agg.log
        format json
    }
}
```

`dev/Caddyfile.example` (split-origin variant in comments):
```
# Split-origin: web at mynl.com/aggregate, API at api.mynl.com
# mynl.com {
#     handle_path /aggregate/* {
#         root * /var/www/aggregate-web
#         file_server
#     }
# }
# api.mynl.com {
#     reverse_proxy 127.0.0.1:8000
# }
```

`dev/aggregate-api.service.example` (Linux systemd):
```ini
[Unit]
Description=aggregate api server
After=network.target

[Service]
Type=simple
User=aggregate
Group=aggregate
WorkingDirectory=/opt/aggregate
Environment=AGGAPI_HOST=127.0.0.1
Environment=AGGAPI_PORT=8000
Environment=AGGAPI_AUDIT_DB=/var/lib/aggregate/audit.db
ExecStart=/opt/aggregate/.venv/bin/aggregate-api
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

`dev/aggregate-api-windows.md`:
- Recommend [NSSM](https://nssm.cc/) for "real" Windows-service install.
- Alternative: Task Scheduler + "At system startup" trigger + `aggregate-api --host 127.0.0.1`.
- WSL2 alternative: run the Linux systemd unit inside WSL.

## Cross-OS verification

Manual checklist before tagging a release that includes api/web:

**Linux (Ubuntu 22.04 or Debian 12):**
- [ ] `pip install aggregate[api]` succeeds.
- [ ] `aggregate-api` starts; `curl http://127.0.0.1:8000/v1/health` returns ok.
- [ ] `curl -X POST http://127.0.0.1:8000/v1/objects -d '{"decl":"agg D dfreq [3] dsev [1:6]"}' -H 'Content-Type: application/json'` returns id.
- [ ] Browser loads UI, examples dropdown populates, build works, plot renders (SVG).
- [ ] PNG plot format works via `?format=png`.
- [ ] Audit DB file appears, row inserted.
- [ ] `systemctl --user start aggregate-api` works given the example unit.
- [ ] Split-origin: SPA built with `VITE_API_BASE_URL` set, served from a separate port, calls succeed with `AGGAPI_CORS_ORIGINS` configured.

**Windows 11:**
- [ ] `pip install aggregate[api]` succeeds (no compilation issues).
- [ ] `aggregate-api` starts; same curl/PowerShell-Invoke-WebRequest checks pass.
- [ ] Browser end-to-end works.
- [ ] Audit DB created at `$env:USERPROFILE\.aggregate\api\audit.db`.
- [ ] uvicorn warning about missing uvloop appears at startup (expected; document in deploy notes).
- [ ] NSSM-installed service starts and survives a reboot.

**Demo dry-run:**
- [ ] Cold start → first build (Dice) completes in <2s.
- [ ] Switching examples and rebuilding feels snappy.
- [ ] Tweedie example from `K.` renders a recognizable Tweedie density plot in SVG.
- [ ] Portfolio example produces a `kappa` plot with multiple lines, plus a kappa frame view.
- [ ] Pricing-at form computes and renders for a Portfolio.

## Open knobs for execution

- **Where exactly do the reference deploy files live?** Plan E puts them in `dev/`. Alternative: `docs/_static/deploy/` so they're served as part of the docs. Lean `dev/` for source-of-truth + copy-paste from the docs page.
- **README mention.** I lean *short mention with link* to the new user-guide page.
- **`pip install aggregate[api]` vs separate package.** Single package with extra for now. Split to a sibling `aggregate-api` package on PyPI only if install footprint becomes an issue.
- **Web bundle in git.** Don't commit `src/aggregate/api/static/`; rebuild during packaging. Listed under `.gitignore` to be safe.
- **Build-step automation (Option B above).** Revisit if a release ever ships an empty bundle.

## Out of scope

- CI pipeline for the web SPA (lint, tests). Manual `npm run build` only.
- Frontend tests / Playwright. Manual smoke testing.
- Telemetry / metrics endpoints.
- Multi-version OpenAPI publishing.
- Docker image.
- Helm chart, Kubernetes manifests.

## File-by-file checklist (for execution)

1. Run `scripts/build-web.ps1` (or `.sh`) — ensure the bundle exists in `src/aggregate/api/static/`.
2. Update `pyproject.toml` `[tool.setuptools.package-data]` to include `api/static/**/*`.
3. SPA polish edits (title, version banner, keyboard help modal, examples search, plot format toggle, copy-DecL button).
4. `cd web && npm run build` again to capture polish changes.
5. Write `docs/2_user_guides/2_x_api.rst`.
6. Write `dev/Caddyfile.example`, `dev/aggregate-api.service.example`, `dev/aggregate-api-windows.md`.
7. Add a short mention + link in `README.rst`.
8. Run the cross-OS verification checklist on both Windows and Linux.
9. `uv build` — produce a wheel. Inspect with `python -m zipfile -l dist/*.whl` to confirm `api/static/index.html` and `api/static/assets/*` are present.
10. `pip install dist/aggregate-*.whl` in a fresh venv; run `aggregate-api`; smoke test.
11. Docs rebuild (manual, outside the iteration loop per CLAUDE.md) to confirm the new RST page renders.

## Recovery / rollback

Each Plan E item is independent. If wheel build is the problem, the rollback is reverting the `package-data` line. If a polish change misbehaves, revert that single web file and rebuild. Reference deploy files are inert — they exist to be read.
