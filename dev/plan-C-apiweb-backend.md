# Plan C — api backend

**Status:** ready to execute after Plans A and B (uses both).
**Depends on:** Plan A (`aggregate.style.context()`), Plan B (`aggregate.parser_errors.format_error()`).
**Unblocks:** Plan D (web SPA needs a real API), API-only consumers (downstream pricing servers).

## Goal

Stand up a FastAPI service at `src/aggregate/api/` that exposes the aggregate library over HTTP/JSON. Two consumers:

1. The Bootstrap SPA in `web/` (Plan D), served either by this backend's `StaticFiles` mount or independently.
2. Any downstream Python or non-Python pricing system that wants to call `build()` and read back `info`, `description`, `stats_df`, `density_df`, plots, `kappa`, `pricing_at` over a versioned, schema-stable API.

KISS shape: single uvicorn worker, in-memory LRU object cache (Option X), SQLite audit log, content-hash object IDs, log2 ceiling, build timeout, CORS middleware for split-origin deploys.

## Naming changes from earlier draft

Previous draft used the umbrella name `apiweb`. With Plan D now living at top-level `web/`, this plan covers only the backend, so:

- Folder: `src/aggregate/apiweb/` → **`src/aggregate/api/`**.
- Console script: `aggregate-server` → **`aggregate-api`**.
- Optional-deps group: `[apiweb]` → **`[api]`**.
- Env-var prefix: `AGGWEB_` → **`AGGAPI_`**.
- Docs page filename: `2_x_apiweb.rst` → **`2_x_api.rst`** (Plan E).

## Deliverables

New tree under `src/aggregate/api/`:

```
src/aggregate/api/
    __init__.py            # exposes create_app(), __version__
    __main__.py            # python -m aggregate.api → uvicorn run
    app.py                 # FastAPI app factory, route mounts, CORS, static
    config.py              # env-var driven settings
    cache.py               # single LRU object cache
    audit.py               # SQLite audit log
    cors.py                # CORSMiddleware setup helper
    models.py              # Pydantic schemas (requests + responses)
    examples.py            # test_suite.agg loader/grouper
    completion.py          # Lark interactive-parser-driven completions
    plotting.py            # style.context() wrapper + plot dispatch (PNG + SVG)
    pricing.py             # pricing_at / price_ccoc dispatch
    serializers.py         # DataFrame → records, info dict normalization
    routes/
        __init__.py
        objects.py         # /v1/objects/*
        decl.py            # /v1/decl/*
        examples.py        # /v1/examples
        meta.py            # /v1/health, /v1/meta, /v1/grammar (alias)
    static/                # populated by Plan D's web build (gitignored)
```

Plus:
- New: `tests/api/` with `conftest.py`, `test_objects.py`, `test_decl.py`, `test_examples.py`, `test_audit.py`, `test_cors.py`.
- Modified: `pyproject.toml` — add `[project.optional-dependencies] api`, `[project.scripts] aggregate-api`.

No changes to existing `aggregate.*` modules apart from the optional-dependency block.

## Endpoint surface (v1)

All paths under `/v1/`. Response media type is JSON unless noted.

### Objects

| Method | Path | Body / params | Response |
|---|---|---|---|
| POST   | `/v1/objects` | `{decl, log2?, bs?}` | `{id, kind, name, cached, elapsed_ms, warnings}` (slim — heavier panes are fetched explicitly) |
| GET    | `/v1/objects` | — | `[{id, kind, name, ts}]` — cache contents |
| GET    | `/v1/objects/{id}` | — | manifest `{id, kind, name, decl, log2, bs, created_at}` |
| DELETE | `/v1/objects/{id}` | — | `{ok: true}` |
| GET    | `/v1/objects/{id}/info` | — | `{info: {...}}` |
| GET    | `/v1/objects/{id}/description` | — | `{columns, rows}` from `obj.describe` |
| GET    | `/v1/objects/{id}/stats_df` | — | `{columns, rows}` from `obj.stats_df` |
| GET    | `/v1/objects/{id}/density_df` | `?cols=&start=&stop=&downsample=` | `{columns, rows}` |
| GET    | `/v1/objects/{id}/kappa` | `?downsample=` | `{columns, rows}` — Portfolio only; the `exeqa_*` slice of density_df |
| GET    | `/v1/objects/{id}/plot` | `?kind=density\|cdf\|qq\|kappa&format=svg\|png&width=&height=&dpi=` | `image/svg+xml` (default) or `image/png` |
| POST   | `/v1/objects/{id}/pricing_at` | `{p?, a?, ccoc?, distortion?}` | `{rows: [...], assets, premium, equity, ...}` |

Per-button-fetch UX: the SPA's action buttons (`info`, `describe`, `stats_df`, `density_df`, `plot`, kappa, pricing) each hit their own endpoint. The cache makes second hits effectively O(1) because the object is already built. Build itself returns only `{id, kind, name, ...}` so the post-build page doesn't pay for data nobody clicked through.

### DecL

| Method | Path | Body / params | Response |
|---|---|---|---|
| POST   | `/v1/decl/complete` | `{decl, cursor}` | `{completions: [{label, terminal, kind}]}` |
| POST   | `/v1/decl/lex` | `{decl}` | `{tokens: [{type, value, start, end, line, column}]}` |
| GET    | `/v1/decl/grammar` | — | `text/plain` (contents of `decl.lark`) |

### Examples

| Method | Path | Body / params | Response |
|---|---|---|---|
| GET    | `/v1/examples` | — | `{categories: [{letter, title, items: [{name, decl, note}]}]}` |

### Meta

| Method | Path | Response |
|---|---|---|
| GET    | `/v1/health` | `{ok: true, version}` |
| GET    | `/v1/meta` | `{version, log2_cap, log2_default, build_timeout_s, cache_max, plot_default_format}` |

OpenAPI is auto-generated by FastAPI at `/openapi.json`; Swagger UI at `/docs`.

## SVG vs PNG plot output

Default response format is `image/svg+xml`. Rationale:

- Aggregate plots are smooth line/curve plots (densities, CDFs, kappas). Matplotlib's default `path.simplify=True` keeps SVG payloads in the 30–150 KB range for typical 2^16 grids.
- SVG is resolution-independent — crisp on retina/4K without us shipping 2× rasters.
- Users can save SVG and embed in PDFs/LaTeX cleanly.

PNG is the alternative for raster use cases (paste into Word/Slack), selected via `?format=png`. Both formats go through the same `aggregate.style.context(**WEB_OVERRIDES)` so visually identical.

Implementation note: matplotlib SVG embeds glyphs by default — adds a few KB but immunizes against missing-font rendering on the client. Leave as default.

## CORS

The web SPA (Plan D) can be deployed at the same origin (FastAPI `StaticFiles` mount) or at a different origin (e.g. `mynl.com/aggregate/` calling `api.mynl.com`). For the second case the backend needs CORS.

`cors.py`:
```python
from fastapi.middleware.cors import CORSMiddleware

def install_cors(app, allowed_origins: list[str]) -> None:
    if not allowed_origins:
        return
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )
```

`config.py` carries `cors_origins: list[str] = []`, parsed from `AGGAPI_CORS_ORIGINS` as comma-separated. Empty → middleware skipped (same-origin deploys don't pay for it).

## Object IDs — content-hash

```python
def object_id(decl_canonical: str, log2: int, bs: float) -> str:
    payload = f"{decl_canonical}|{log2}|{bs!r}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
```

`decl_canonical` is the DecL text stripped of comments and trailing whitespace; whitespace inside the program is preserved. Building a stronger canonical form (whitespace-insensitive via Lark tree → re-emit) is a future enhancement — flagged as an open knob below.

POST `/v1/objects` with the same `(decl, log2, bs)` returns the same `id` and is idempotent — downstream pricing servers get free dedup and safe retries.

## Cache (Option X)

`cache.py` holds an `OrderedDict[str, CacheEntry]` with LRU semantics. Single cache, single eviction policy.

```python
@dataclass
class CacheEntry:
    obj: Aggregate | Portfolio
    decl: str
    log2: int
    bs: float
    kind: str
    name: str
    created_at: datetime

class ObjectCache:
    def __init__(self, max_entries: int = 50): ...
    def get(self, oid: str) -> CacheEntry | None: ...      # moves to MRU
    def put(self, oid: str, entry: CacheEntry) -> None: ...# evicts LRU
    def delete(self, oid: str) -> bool: ...
    def list(self) -> list[CacheEntry]: ...
    def __contains__(self, oid: str) -> bool: ...
```

Eviction policy is plain LRU bounded by entry count (default 50, configurable). Plots, `density_df` pages, kappa frames, pricing results are *not* cached separately — they're recomputed from the live cached object on demand.

## Audit log (SQLite)

`audit.py` writes one row per build attempt.

Schema (created on startup, idempotent):
```sql
CREATE TABLE IF NOT EXISTS builds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,            -- ISO 8601 UTC
    ip TEXT NOT NULL,
    object_id TEXT,              -- content hash, NULL on failure
    kind TEXT,                   -- 'agg' | 'port' | NULL on failure
    decl TEXT NOT NULL,
    log2 INTEGER,
    bs REAL,
    status TEXT NOT NULL,        -- 'ok' | 'parse_error' | 'build_error' | 'timeout' | 'limit_exceeded'
    error_msg TEXT,
    elapsed_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS builds_ts ON builds(ts);
CREATE INDEX IF NOT EXISTS builds_ip ON builds(ip);
```

API:
```python
class AuditLog:
    def __init__(self, db_path: Path): ...
    def record_build(self, *, ip: str, decl: str, log2: int, bs: float,
                     status: str, object_id: str | None = None,
                     kind: str | None = None, error_msg: str | None = None,
                     elapsed_ms: int = 0) -> None: ...
    def recent(self, n: int = 100) -> list[dict]: ...
    def by_ip(self, ip: str, n: int = 100) -> list[dict]: ...
```

Connection uses `sqlite3` (stdlib) with `journal_mode=WAL` and `synchronous=NORMAL`. Path is configurable; default `~/.aggregate/api/audit.db` (created if missing).

No DB endpoint in v1 — audit log is read by SQL CLI or a future admin route. The user just wants the data captured.

## Build pipeline

```python
def build_endpoint(req: BuildRequest, client_ip: str) -> BuildResponse:
    if req.log2 > settings.log2_cap:
        audit.record_build(..., status='limit_exceeded', ...)
        raise HTTPException(422, "log2 exceeds cap")

    oid = object_id(canonicalize(req.decl), req.log2, req.bs)
    if oid in cache:
        entry = cache.get(oid)
        audit.record_build(..., status='ok', object_id=oid, ...)
        return BuildResponse(id=oid, kind=entry.kind, name=entry.name, cached=True, ...)

    with build_semaphore, timeout(settings.build_timeout_s):
        try:
            obj = build(req.decl, log2=req.log2, bs=req.bs)
        except UnexpectedInput as e:
            report = format_error(req.decl, e)
            audit.record_build(..., status='parse_error', error_msg=report.message, ...)
            raise HTTPException(422, detail=report.to_dict())
        except TimeoutError:
            audit.record_build(..., status='timeout', ...)
            raise HTTPException(504, "build timeout")
        except Exception as e:
            audit.record_build(..., status='build_error', error_msg=str(e), ...)
            raise HTTPException(500, str(e))

    cache.put(oid, CacheEntry(obj=obj, ...))
    audit.record_build(..., status='ok', object_id=oid, ...)
    return BuildResponse(id=oid, kind=..., name=..., cached=False, ...)
```

`build_semaphore` is a `threading.Semaphore(1)` — at most one heavy build in flight. Reads don't acquire it.

Timeout via `concurrent.futures.ThreadPoolExecutor` + `future.result(timeout=...)`. Python can't truly cancel a CPU-bound thread; documented caveat.

## Plotting

```python
WEB_OVERRIDES = {
    "figure.figsize": (5.5, 3.5),
    "figure.dpi": 100,
    "savefig.dpi": 100,
}

def render_plot(obj, kind: str, fmt: str = "svg", **kwargs) -> tuple[bytes, str]:
    buf = io.BytesIO()
    with aggregate.style.context(**WEB_OVERRIDES, **kwargs.pop("rc", {})):
        fig = _dispatch(obj, kind, **kwargs)
        fig.savefig(buf, format=fmt)
        plt.close(fig)
    media_type = "image/svg+xml" if fmt == "svg" else "image/png"
    return buf.getvalue(), media_type
```

Plot kinds (v1):
- `density` — PMF / PDF over loss support.
- `cdf` — F over loss.
- `qq` — quantile-quantile vs normal.
- `kappa` — Portfolio only; line plot of `exeqa_*` columns vs `loss`.

## DecL completions

```python
def complete(decl: str, cursor: int) -> list[Completion]:
    prefix = decl[:cursor]
    try:
        interactive = _PARSER.parse_interactive(prefix)
        interactive.exhaust_lexer()
        accepted = interactive.accepts()
    except UnexpectedInput:
        return []
    return [_to_completion(t) for t in sorted(accepted)]
```

Reuses Plan B's `_TERMINAL_LABELS` for human-readable labels. Identifier completions (severity names, frequency names from the knowledge base) are a v1.1 enhancement.

## Examples loader

Parses `src/aggregate/agg/test_suite.agg` once on startup. Recognizes the contents block (`# A. Title ...`) and section headers (`# A. Title` followed by `# ====` underline). For each non-comment, non-blank line in a section, emit `{name, decl, note}` where `note` is extracted from a trailing `note{...}` if present.

Cached as a module-level dict; reloaded only on server restart.

## Pydantic models

Pydantic v2. Field names `snake_case`. All response models have `model_config = ConfigDict(extra="forbid")` to keep clients honest.

```python
class BuildRequest(BaseModel):
    decl: str = Field(..., min_length=1)
    log2: int | None = Field(None, ge=4)
    bs: float | None = Field(None, gt=0)

class BuildResponse(BaseModel):
    id: str
    kind: Literal["agg", "port"]
    name: str
    warnings: list[str]
    cached: bool
    elapsed_ms: int

class FrameResponse(BaseModel):
    columns: list[str]
    rows: list[list]                 # list-of-lists, not list-of-dicts (smaller)

class InfoResponse(BaseModel):
    info: dict

class DeclCompleteRequest(BaseModel):
    decl: str
    cursor: int = Field(..., ge=0)

class Completion(BaseModel):
    label: str
    terminal: str
    kind: Literal["keyword", "identifier", "literal"]

class PricingRequest(BaseModel):
    p: float | None = None
    a: float | None = None
    ccoc: float | None = None
    distortion: str | None = None

class PricingResponse(BaseModel):
    p: float | None
    a: float
    ccoc: float | None
    premium: float
    equity: float
    expected_loss: float
    rows: list[dict]                 # per-unit breakdown for Portfolios
```

## Config

`config.py` uses `pydantic-settings`. Env vars prefixed `AGGAPI_`:

| Var | Default | Meaning |
|---|---|---|
| `AGGAPI_HOST` | `127.0.0.1` | uvicorn bind |
| `AGGAPI_PORT` | `8000` | uvicorn port |
| `AGGAPI_LOG2_DEFAULT` | `16` | default if request omits |
| `AGGAPI_LOG2_CAP` | `18` | hard cap |
| `AGGAPI_BUILD_TIMEOUT_S` | `10` | per-request timeout |
| `AGGAPI_CACHE_MAX` | `50` | LRU entries |
| `AGGAPI_AUDIT_DB` | `~/.aggregate/api/audit.db` | SQLite path |
| `AGGAPI_KNOWLEDGE_BASE` | `default` | knowledge base name or path |
| `AGGAPI_CORS_ORIGINS` | `` | comma-separated list; empty = no CORS middleware |
| `AGGAPI_PLOT_DEFAULT_FORMAT` | `svg` | `svg` or `png` |
| `AGGAPI_STATIC_DIR` | (auto: `<pkg>/api/static`) | override path to SPA bundle |

## Console script + module entry

`pyproject.toml`:
```toml
[project.optional-dependencies]
api = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "pydantic>=2.7",
    "pydantic-settings>=2.4",
]

[project.scripts]
aggregate-api = "aggregate.api.__main__:main"
```

`__main__.py`:
```python
def main():
    import argparse, uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--reload", action="store_true")
    args = p.parse_args()
    from .config import settings
    uvicorn.run(
        "aggregate.api.app:create_app",
        factory=True,
        host=args.host or settings.host,
        port=args.port or settings.port,
        reload=args.reload,
    )

if __name__ == "__main__":
    main()
```

## Static-files mount (same-origin deploy)

`app.py`:
```python
from fastapi.staticfiles import StaticFiles
from importlib.resources import files

def create_app():
    app = FastAPI(...)
    install_cors(app, settings.cors_origins)
    register_routes(app)
    static_dir = settings.static_dir or files("aggregate.api").joinpath("static")
    if Path(static_dir).exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
    return app
```

`html=True` makes `/` serve `index.html` and falls back to `index.html` for unknown paths (SPA-style). If `static_dir` doesn't exist (no web build present), the mount is skipped and the server is API-only.

## Verification steps

1. `uv sync --extra api` — pulls FastAPI + uvicorn.
2. `uv run pytest tests/api/ -v` — backend test suite green.
3. `uv run aggregate-api --port 8001` then in another shell:
   ```
   curl -s -X POST http://127.0.0.1:8001/v1/objects \
        -H "Content-Type: application/json" \
        -d '{"decl":"agg Dice dfreq [3] dsev [1:6]"}'
   ```
   → `{"id": "...", "kind": "agg", "name": "Dice", "cached": false, ...}`.
4. `curl http://127.0.0.1:8001/v1/objects/{id}/info` → `{"info": {...}}`.
5. `curl "http://127.0.0.1:8001/v1/objects/{id}/plot?kind=density" -o density.svg` → opens cleanly in browser.
6. Open `http://127.0.0.1:8001/docs` — Swagger UI renders all routes.
7. Force a parse error, confirm the response body is Plan B's `ErrorReport` JSON.
8. CORS smoke: with `AGGAPI_CORS_ORIGINS=http://localhost:5173` set, `curl -i -H 'Origin: http://localhost:5173' http://127.0.0.1:8001/v1/health` includes `access-control-allow-origin`.
9. Inspect `audit.db` via `sqlite3 audit.db "SELECT ts, ip, status, decl FROM builds ORDER BY ts DESC LIMIT 5"`.

## Test sketch (`tests/api/`)

```python
# tests/api/conftest.py
import pytest
from fastapi.testclient import TestClient
from aggregate.api.app import create_app

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("AGGAPI_AUDIT_DB", str(tmp_path / "audit.db"))
    monkeypatch.setenv("AGGAPI_LOG2_CAP", "20")
    return TestClient(create_app())

# tests/api/test_objects.py
def test_build_dice(client):
    r = client.post("/v1/objects", json={"decl": "agg Dice dfreq [3] dsev [1:6]"})
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == "agg"
    assert body["name"] == "Dice"
    assert len(body["id"]) == 16

def test_build_is_idempotent(client):
    decl = "agg Dice dfreq [3] dsev [1:6]"
    r1 = client.post("/v1/objects", json={"decl": decl})
    r2 = client.post("/v1/objects", json={"decl": decl})
    assert r1.json()["id"] == r2.json()["id"]
    assert r2.json()["cached"] is True

def test_info_endpoint(client):
    r = client.post("/v1/objects", json={"decl": "agg Dice dfreq [3] dsev [1:6]"})
    oid = r.json()["id"]
    r2 = client.get(f"/v1/objects/{oid}/info")
    assert "info" in r2.json()

def test_density_df_paginated(client):
    r = client.post("/v1/objects", json={"decl": "agg X dfreq [3] dsev [1:6]"})
    oid = r.json()["id"]
    r2 = client.get(f"/v1/objects/{oid}/density_df", params={"cols": "loss,p_total", "downsample": 20})
    body = r2.json()
    assert body["columns"] == ["loss", "p_total"]
    assert len(body["rows"]) <= 20

def test_plot_svg_default(client):
    r = client.post("/v1/objects", json={"decl": "agg X dfreq [3] dsev [1:6]"})
    oid = r.json()["id"]
    r2 = client.get(f"/v1/objects/{oid}/plot", params={"kind": "density"})
    assert r2.status_code == 200
    assert r2.headers["content-type"].startswith("image/svg+xml")
    assert r2.content.lstrip().startswith(b"<?xml") or r2.content.lstrip().startswith(b"<svg")

def test_plot_png_explicit(client):
    r = client.post("/v1/objects", json={"decl": "agg X dfreq [3] dsev [1:6]"})
    oid = r.json()["id"]
    r2 = client.get(f"/v1/objects/{oid}/plot", params={"kind": "density", "format": "png"})
    assert r2.headers["content-type"] == "image/png"
    assert r2.content[:8] == b"\x89PNG\r\n\x1a\n"

def test_parse_error_returns_report(client):
    r = client.post("/v1/objects", json={"decl": "agg X 100 claims mixedd poisson"})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "line" in detail

# tests/api/test_cors.py
def test_cors_headers_present(tmp_path, monkeypatch):
    monkeypatch.setenv("AGGAPI_AUDIT_DB", str(tmp_path / "audit.db"))
    monkeypatch.setenv("AGGAPI_CORS_ORIGINS", "http://localhost:5173")
    from aggregate.api.app import create_app
    from fastapi.testclient import TestClient
    client = TestClient(create_app())
    r = client.get("/v1/health", headers={"Origin": "http://localhost:5173"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"

# tests/api/test_decl.py
def test_grammar_serves_lark(client):
    r = client.get("/v1/decl/grammar")
    assert r.status_code == 200
    assert "start:" in r.text or "%import" in r.text

# tests/api/test_examples.py
def test_examples_grouped_by_category(client):
    r = client.get("/v1/examples")
    cats = r.json()["categories"]
    letters = {c["letter"] for c in cats}
    assert "A" in letters and "B" in letters

# tests/api/test_audit.py
def test_audit_row_written(client, tmp_path):
    client.post("/v1/objects", json={"decl": "agg X dfreq [3] dsev [1:6]"})
    import sqlite3
    db = tmp_path / "audit.db"
    rows = sqlite3.connect(db).execute("SELECT status, decl FROM builds").fetchall()
    assert any(r[0] == "ok" for r in rows)
```

## Open knobs for execution

- **DecL canonicalization for hashing.** Default to "strip comments and trim". Stronger normalization (whitespace-insensitive via tree round-trip) is a future enhancement.
- **Identifier completions.** Plan C ships keyword/terminal completions only. Knowledge-base identifier completions deferred to v1.1.
- **Plot `kind=kappa` rendering.** Builds a fresh matplotlib figure from `density_df[['loss', 'exeqa_*']]`. Default: linear-x, color cycle for lines, legend on right.
- **Pricing endpoint surface.** `pricing_at` (distortion-based) and `price_ccoc` (constant CoC) union'd into one request; server picks based on which fields are present.
- **Build cancellation.** Threading + timeout can't truly cancel CPU-bound work. Acceptable for trusted team. Hard cancellation requires subprocess-per-build (out of scope).
- **SVG glyph embedding.** Default-on, slightly larger payload but no font surprises. Switch off via `mpl.rcParams['svg.fonttype'] = 'none'` if file size becomes a concern.

## Out of scope

- Authentication / authorization. The team-install gets `caddy reverse_proxy` + basic auth.
- Multi-tenant cache isolation.
- Redis or disk-backed cache.
- Plot JSON delivery.
- WebSocket / streaming endpoints.
- An admin route for the audit log.
- Bulk build endpoint.

## File-by-file checklist (for execution)

1. Add `[project.optional-dependencies] api` and `[project.scripts] aggregate-api` to `pyproject.toml`.
2. `uv sync --extra api` — pull deps.
3. Create the file skeleton under `src/aggregate/api/`.
4. Implement `config.py` and `cache.py`.
5. Implement `audit.py`.
6. Implement `cors.py`.
7. Implement `models.py`.
8. Implement `serializers.py`.
9. Implement `examples.py`.
10. Implement `completion.py`.
11. Implement `plotting.py` (SVG + PNG paths).
12. Implement `pricing.py`.
13. Implement `routes/objects.py`, `routes/decl.py`, `routes/examples.py`, `routes/meta.py`.
14. Implement `app.py` — `create_app()` factory, CORS, static mount (conditional), exception handlers.
15. Implement `__main__.py` — uvicorn launcher.
16. Tests under `tests/api/`. `uv run pytest tests/api/ -v`.
17. Manual smoke test per "Verification steps".

## Recovery / rollback

`src/aggregate/api/` and `tests/api/` are net-new and can be deleted wholesale. The only edit to existing code is the `pyproject.toml` block; revert that to restore the prior state. No changes to library modules.
