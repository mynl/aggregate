# Plan E — apiweb polish, deploy, package

**Status:** ready to execute after Plans C and D.
**Depends on:** Plans C (backend) and D (frontend built).
**Unblocks:** end-to-end shipping — `pip install aggregate[apiweb]` → `aggregate-server` → demo or team install.

## Goal

Take the apiweb from "works on my dev box" to "installs cleanly on Windows and Linux, runs as a service, can be put behind Caddy, has the small polish details that make a demo feel finished."

## Deliverables

- Modified: `pyproject.toml` — package-data declarations for the built frontend, console script (if not added in Plan C).
- New: `docs/2_user_guides/2_x_apiweb.rst` — user-facing docs page covering install, run, deploy.
- New: `dev/Caddyfile.example` — copy-paste reference config.
- New: `dev/aggregate-server.service.example` — systemd unit reference (Linux).
- New: `dev/aggregate-server-windows.md` — Windows service notes (NSSM or scheduled task).
- New: `scripts/build-frontend.ps1` and `scripts/build-frontend.sh` — wrappers around `npm run build`.
- Modified: `README.rst` — short apiweb mention with link to the new docs page.
- Small polish edits in the frontend (favicon, page title, keyboard help modal, version display from `/v1/meta`).

## Packaging — the wheel must ship the built frontend

`pyproject.toml`:
```toml
[tool.setuptools.package-data]
aggregate = [
    "*.lark",
    "agg/*.agg",
    "data/*.mplstyle",
    "apiweb/static/**/*",
]
```

Build ordering — the frontend bundle has to exist before `uv build` runs. Two options:

- **Option A — Build step is a manual prerequisite.** Document "run `scripts/build-frontend.ps1` before `uv build`." Simple, no build automation. If you forget, `uv build` succeeds but ships an empty `static/` and the website is broken.
- **Option B — Build step automated via a build backend hook.** Use `setuptools` with a `cmdclass` override, or switch to `hatchling` with a build hook. Adds complexity but bullet-proof.

Recommend **Option A for v0**. Document loudly in the release checklist. Revisit if a release ever ships an empty bundle.

`scripts/build-frontend.ps1`:
```powershell
$ErrorActionPreference = 'Stop'
Push-Location "$PSScriptRoot/../src/aggregate/apiweb/frontend"
try {
    npm install
    npm run build
}
finally {
    Pop-Location
}
```

`scripts/build-frontend.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../src/aggregate/apiweb/frontend"
npm install
npm run build
```

## Frontend polish

Final touches in `src/aggregate/apiweb/frontend/src/`:

- **Favicon.** A small monochrome glyph — could be a stylized histogram or a Greek letter that's already an aggregate motif (look at existing docs imagery for a candidate). Ship as `public/favicon.svg`; Vite copies to the bundle automatically.
- **Page title.** `<title>aggregate · DecL playground</title>`.
- **Version banner.** Fetch `/v1/meta` once on load, display version in the header.
- **Keyboard shortcuts modal.** Ctrl-/ or `?` opens a modal listing: Ctrl-Enter (build), Ctrl-↑/↓ (history), Ctrl-K (focus example search if added), Esc (close modals).
- **Inline example search.** Type-ahead filter at the top of the examples sidebar. Pure client-side filter over the already-loaded list. ~20 lines.
- **Build state UI.** While `POST /v1/objects` is in flight, the Build button shows a spinner and is disabled. Cancel only by waiting for timeout (no client-side cancellation since server can't truly cancel).
- **"Copied!" toast on shareable actions** (if any are added later — none in v0).

## User-facing docs page

New `docs/2_user_guides/2_x_apiweb.rst`, roughly:

```rst
The apiweb demo server
======================

``aggregate.apiweb`` ships a small FastAPI service plus a single-page
demo UI for interactive DecL exploration. It is intended for:

* Demos — quickly show what a Tweedie / Lognormal / Portfolio looks like.
* Small-team installs behind a VPN, fronted by Caddy + auth.
* As a stable JSON API consumed by downstream pricing systems.

It is **not** intended as a public-internet service.

Installation
------------

.. code-block:: bash

    pip install aggregate[apiweb]
    aggregate-server --port 8000

Open ``http://127.0.0.1:8000/`` for the UI. The OpenAPI / Swagger UI is at
``http://127.0.0.1:8000/docs``.

Configuration
-------------

Environment variables (prefix ``AGGWEB_``):

.. (table from Plan C config section)

API surface
-----------

.. (table of v1 endpoints; link to /openapi.json)

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

SQLite at ``$AGGWEB_AUDIT_DB`` (default ``~/.aggregate/apiweb/audit.db``).
Query directly:

.. code-block:: bash

    sqlite3 ~/.aggregate/apiweb/audit.db \
      "SELECT ts, ip, status, decl FROM builds ORDER BY ts DESC LIMIT 20"
```

## Reference deploy configs

`dev/Caddyfile.example`:
```
# Demo deployment behind Caddy with basic auth.
# Adjust the hostname to match your DNS or internal CA.

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

`dev/aggregate-server.service.example` (Linux systemd):
```ini
[Unit]
Description=aggregate apiweb server
After=network.target

[Service]
Type=simple
User=aggregate
Group=aggregate
WorkingDirectory=/opt/aggregate
Environment=AGGWEB_HOST=127.0.0.1
Environment=AGGWEB_PORT=8000
Environment=AGGWEB_AUDIT_DB=/var/lib/aggregate/audit.db
ExecStart=/opt/aggregate/.venv/bin/aggregate-server
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

`dev/aggregate-server-windows.md`:
- Recommend [NSSM](https://nssm.cc/) for "real" Windows-service install (sets up restart-on-failure, log redirection).
- Alternative: Task Scheduler + "At system startup" trigger + `aggregate-server --host 127.0.0.1` action. Simpler, less robust.
- WSL2 alternative: run the Linux systemd unit inside WSL, expose port to host via `wsl --exec`. Worth noting.

## Cross-OS verification

Manual checklist before tagging a release that includes apiweb:

**Linux (Ubuntu 22.04 or Debian 12):**
- [ ] `pip install aggregate[apiweb]` succeeds.
- [ ] `aggregate-server` starts; `curl http://127.0.0.1:8000/v1/health` returns ok.
- [ ] `curl -X POST http://127.0.0.1:8000/v1/objects -d '{"decl":"agg D dfreq [3] dsev [1:6]"}' -H 'Content-Type: application/json'` returns id.
- [ ] Browser loads UI, example sidebar populates, build works, plot renders.
- [ ] Audit DB file appears, row inserted.
- [ ] `systemctl --user start aggregate-server` works given the example unit.

**Windows 11:**
- [ ] `pip install aggregate[apiweb]` succeeds (no compilation issues).
- [ ] `aggregate-server` starts; same curl/PowerShell-Invoke-WebRequest checks pass.
- [ ] Browser end-to-end works.
- [ ] Audit DB created at `$env:USERPROFILE\.aggregate\apiweb\audit.db`.
- [ ] uvicorn warning about missing uvloop appears at startup (expected; document in deploy notes).
- [ ] NSSM-installed service starts and survives a reboot.

**Demo dry-run:**
- [ ] Cold start → first build (Dice) completes in <2s.
- [ ] Switching examples and rebuilding feels snappy.
- [ ] Tweedie example from `K.` renders a recognizable Tweedie density plot.
- [ ] Portfolio example produces a `kappa` plot with multiple lines.

## Open knobs for execution

- **Where exactly do the reference deploy files live?** Plan E puts them in `dev/` (the new planning folder). Alternative: `docs/_static/deploy/` so they're served as part of the docs and linkable from the user guide. I'd suggest `dev/` for the source-of-truth + copy-paste from the docs page; lower friction.
- **Should the apiweb docs page be referenced from the README, or kept as a "you have to know it's there" guide?** I lean *short README mention with link* — it's a feature of the package and people should be able to discover it.
- **`pip install aggregate[apiweb]` vs separate package.** Currently a single package with an optional extra. If install footprint becomes an issue (FastAPI + uvicorn + websockets + ~20MB), splitting to a sibling `aggregate-apiweb` package on PyPI is an option. Defer until there's a reason.
- **Branding / favicon.** Pure aesthetic. Pick when we get there.
- **Build-step automation (Option B above).** If `Option A` proves error-prone, revisit. For v0, manual + checklist.

## Out of scope

- CI pipeline for the frontend (lint, tests, etc.). Manual `npm run build` only.
- Frontend tests / Playwright. Manual smoke testing.
- Telemetry / metrics endpoints.
- Multi-version OpenAPI publishing.
- Docker image. (Could add later; team installs can write their own Dockerfile in 8 lines.)
- Helm chart, Kubernetes manifests. Out of scope by ~3 orders of magnitude.

## File-by-file checklist (for execution)

1. Run `scripts/build-frontend.ps1` (or `.sh`) — ensure the bundle exists in `apiweb/static/`.
2. Update `pyproject.toml` `[tool.setuptools.package-data]` to include `apiweb/static/**/*`.
3. Frontend polish edits (favicon, title, version banner, keyboard help modal, example search).
4. `cd src/aggregate/apiweb/frontend && npm run build` again to capture polish changes.
5. Write `docs/2_user_guides/2_x_apiweb.rst`.
6. Write `dev/Caddyfile.example`, `dev/aggregate-server.service.example`, `dev/aggregate-server-windows.md`.
7. Add a short mention + link in `README.rst`.
8. Run the cross-OS verification checklist on both Windows and Linux.
9. `uv build` — produce a wheel. Inspect with `python -m zipfile -l dist/*.whl` to confirm `apiweb/static/index.html` and `apiweb/static/assets/*` are present.
10. `pip install dist/aggregate-*.whl` in a fresh venv; run `aggregate-server`; smoke test.
11. Docs rebuild (manual, outside the iteration loop per CLAUDE.md) to confirm the new RST page renders.

## Recovery / rollback

Each Plan E item is independent. If wheel build is the problem, the rollback is reverting the `package-data` line. If a polish change misbehaves, revert that single frontend file and rebuild. Reference deploy files are inert — they exist to be read.
