"""Object lifecycle routes -- the heart of the api.

Endpoints
---------

The /v1/objects/* family covers everything object-shaped:

* ``POST   /v1/objects``         -- build + cache; returns slim manifest.
* ``GET    /v1/objects``         -- list cache contents.
* ``GET    /v1/objects/{id}``    -- per-object manifest.
* ``DELETE /v1/objects/{id}``    -- evict from cache.
* ``GET    /v1/objects/{id}/info``        -- text summary.
* ``GET    /v1/objects/{id}/description`` -- describe DataFrame.
* ``GET    /v1/objects/{id}/stats_df``    -- stats_df DataFrame.
* ``GET    /v1/objects/{id}/density_df``  -- paginated density frame.
* ``GET    /v1/objects/{id}/kappa``       -- Portfolio exeqa_* slice.
* ``GET    /v1/objects/{id}/plot``        -- SVG/PNG image.
* ``POST   /v1/objects/{id}/pricing_at``  -- distortion / ccoc pricing.

Build pipeline (POST /v1/objects)
---------------------------------

1. Validate ``log2`` against the cap. Reject early.
2. Compute the content-hash id for the (canonical_decl, log2, bs)
   triple.
3. If cached: bump LRU, audit ``status='ok'``, return slim response.
4. Otherwise: acquire the build semaphore (single concurrent
   build), submit to a thread-pool with a wall-clock timeout, and
   either store the result + audit ``ok`` or audit
   ``parse_error / build_error / timeout``.

Why a thread-pool + future timeout
----------------------------------

``concurrent.futures.ThreadPoolExecutor`` is the simplest way to
get a hard wall-clock cap on a synchronous library call. Python
can't truly cancel a CPU-bound thread, but the api stops waiting
on it and returns 504 so the SPA doesn't hang. The thread keeps
running until ``build()`` returns -- documented caveat in the plan.

Per-button-fetch UX
-------------------

The build response carries only ``id``, ``kind``, ``name``,
``warnings``, ``cached``, ``elapsed_ms``. The SPA shows that
immediately and only fetches info/describe/plot/pricing when the
user clicks the matching button. Second visits hit the cache and
return in milliseconds.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response

from lark.exceptions import UnexpectedInput

from aggregate import build as _build_singleton
from aggregate.parser_errors import format_error

from .. import models
from ..audit import AuditLog
from ..cache import CacheEntry, ObjectCache, canonicalize_decl, object_id
from ..config import Settings, get_settings
from ..plotting import render_plot
from ..pricing import run_pricing
from ..serializers import frame_to_payload, info_to_payload, reset_index_safe


router = APIRouter()


# ----------------------------------------------------------------------
# Process-wide singletons
# ----------------------------------------------------------------------
# The cache and audit log are created lazily on first use. They're
# *not* created at import time because tests rely on env-var-driven
# config (audit-db location) being read after monkeypatching.
# ``_get_cache`` / ``_get_audit`` are pulled via Depends so the
# objects stay in module-level state where production code wants
# them, but they're reachable for monkeypatching in tests.
#
# Build-side concurrency: a single semaphore caps in-flight heavy
# builds at 1, regardless of how many requests are queued up. Reads
# (info / describe / plot) don't touch it -- they're O(ms) lookups
# on the already-built object.
_cache_lock = threading.Lock()
_cache_singleton: ObjectCache | None = None
_audit_singleton: AuditLog | None = None

# Single-slot semaphore = only one heavy build runs at a time.
# Heavy builds happen rarely (most requests are cache hits); the
# semaphore prevents an accidental "build a 2**18 portfolio four
# times" pile-up from saturating the box.
_build_semaphore = threading.Semaphore(1)

# A small thread pool, one worker, used solely to enforce build
# timeouts. ``future.result(timeout=T)`` is the cleanest pattern
# for "give up on a synchronous call after N seconds" in Python.
_build_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agg-build")


def _get_cache(settings: Settings = Depends(get_settings)) -> ObjectCache:
    """Lazy-init the cache singleton with the configured max size."""
    global _cache_singleton
    with _cache_lock:
        if _cache_singleton is None or _cache_singleton._max != settings.cache_max:
            _cache_singleton = ObjectCache(max_entries=settings.cache_max)
    return _cache_singleton


def _get_audit(settings: Settings = Depends(get_settings)) -> AuditLog:
    """Lazy-init the audit-log singleton at the configured DB path."""
    global _audit_singleton
    with _cache_lock:
        if _audit_singleton is None or str(_audit_singleton.db_path) != settings.audit_db:
            _audit_singleton = AuditLog(settings.audit_db)
    return _audit_singleton


def reset_singletons() -> None:
    """Drop the cached cache + audit so the next request re-inits.

    Hook for tests that swap env vars across cases -- the
    ``client`` fixture in ``tests/api/conftest.py`` calls this.
    """
    global _cache_singleton, _audit_singleton
    with _cache_lock:
        _cache_singleton = None
        _audit_singleton = None


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _client_ip(request: Request) -> str:
    """Best-effort client IP. Returns ``"-"`` if FastAPI didn't capture one."""
    if request.client and request.client.host:
        return request.client.host
    return "-"


def _now_iso() -> str:
    """ISO 8601 timestamp with millisecond precision."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _run_build(decl: str, log2: int, bs: float):
    """Invoke the underlying ``build()``.

    Pulled into a helper so the thread-pool target is a plain
    function -- closures over ``log2=0`` / ``bs=0`` are the
    library's "let me pick" signal, so we forward the request's
    values verbatim.
    """
    # log2=0 / bs=0 are the underlying ``build()``'s "use defaults"
    # sentinels; pass them through when the request omitted those
    # knobs.
    return _build_singleton(decl, log2=log2, bs=bs)


def _resolve_object(oid: str, cache: ObjectCache) -> CacheEntry:
    """Fetch an entry or raise 404."""
    entry = cache.get(oid)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"object {oid} not in cache")
    return entry


# ----------------------------------------------------------------------
# POST /v1/objects
# ----------------------------------------------------------------------

@router.post("/objects", response_model=models.BuildResponse)
def post_object(
    req: models.BuildRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
    cache: ObjectCache = Depends(_get_cache),
    audit: AuditLog = Depends(_get_audit),
) -> dict:
    """Build (or retrieve from cache) an aggregate object.

    Returns the slim manifest; the SPA fetches heavier panes on
    demand via the per-button GETs. Same (decl, log2, bs) is
    idempotent -- the second call returns ``cached=True`` with
    the same ``id``.
    """
    # Resolve effective knobs: a missing log2 / bs from the request
    # means "use library defaults" -- which the underlying build()
    # signals via 0. We hash the *requested* values (0 included)
    # so two callers asking for "defaults" share the same cache slot.
    eff_log2 = req.log2 if req.log2 is not None else 0
    eff_bs = req.bs if req.bs is not None else 0.0
    ip = _client_ip(request)
    t0 = time.monotonic()

    # Cap check is cheap; do it before the cache lookup so an
    # over-cap request never reaches the build path.
    if eff_log2 and eff_log2 > settings.log2_cap:
        elapsed = int((time.monotonic() - t0) * 1000)
        audit.record_build(
            ip=ip, decl=req.decl, log2=eff_log2, bs=eff_bs,
            status="limit_exceeded",
            error_msg=f"log2 {eff_log2} exceeds cap {settings.log2_cap}",
            elapsed_ms=elapsed,
        )
        raise HTTPException(
            status_code=422,
            detail=f"log2 {eff_log2} exceeds AGGAPI_LOG2_CAP={settings.log2_cap}",
        )

    canonical = canonicalize_decl(req.decl)
    oid = object_id(canonical, eff_log2, eff_bs)

    # Cache hit -- return slim manifest immediately.
    cached_entry = cache.get(oid)
    if cached_entry is not None:
        elapsed = int((time.monotonic() - t0) * 1000)
        audit.record_build(
            ip=ip, decl=req.decl, log2=eff_log2, bs=eff_bs,
            status="ok", object_id=oid, kind=cached_entry.kind,
            elapsed_ms=elapsed,
        )
        return {
            "id": oid,
            "kind": cached_entry.kind,
            "name": cached_entry.name,
            "warnings": [],
            "cached": True,
            "elapsed_ms": elapsed,
        }

    # Cache miss -- fire the build, gated by the semaphore +
    # wall-clock timeout. Note: the semaphore only serializes the
    # *future submission*, not the wait. With one worker the
    # semaphore is technically redundant (the worker serializes
    # naturally), but it makes intent explicit.
    with _build_semaphore:
        future = _build_executor.submit(_run_build, req.decl, eff_log2, eff_bs)
        try:
            obj = future.result(timeout=settings.build_timeout_s)
        except FuturesTimeout:
            elapsed = int((time.monotonic() - t0) * 1000)
            audit.record_build(
                ip=ip, decl=req.decl, log2=eff_log2, bs=eff_bs,
                status="timeout",
                error_msg=f"build exceeded {settings.build_timeout_s}s",
                elapsed_ms=elapsed,
            )
            raise HTTPException(status_code=504, detail="build timeout")
        except ValueError as exc:
            # The parser wraps Lark exceptions in ValueError via
            # ``raise ... from``. Use format_error to recover a
            # structured ErrorReport when the underlying cause was
            # a parse failure; otherwise it's a build-time validation
            # error and we fall through.
            unwrapped = exc.__cause__
            if isinstance(unwrapped, UnexpectedInput):
                report = format_error(req.decl, exc)
                elapsed = int((time.monotonic() - t0) * 1000)
                audit.record_build(
                    ip=ip, decl=req.decl, log2=eff_log2, bs=eff_bs,
                    status="parse_error", error_msg=report.message,
                    elapsed_ms=elapsed,
                )
                raise HTTPException(status_code=422, detail=report.to_dict())
            # Library-side validation error (e.g. invalid spec).
            elapsed = int((time.monotonic() - t0) * 1000)
            audit.record_build(
                ip=ip, decl=req.decl, log2=eff_log2, bs=eff_bs,
                status="build_error", error_msg=str(exc),
                elapsed_ms=elapsed,
            )
            raise HTTPException(status_code=422, detail=str(exc))
        except UnexpectedInput as exc:
            # Defensive: if the parser ever surfaces a raw Lark
            # exception without the ValueError wrap, handle it the
            # same way.
            report = format_error(req.decl, exc)
            elapsed = int((time.monotonic() - t0) * 1000)
            audit.record_build(
                ip=ip, decl=req.decl, log2=eff_log2, bs=eff_bs,
                status="parse_error", error_msg=report.message,
                elapsed_ms=elapsed,
            )
            raise HTTPException(status_code=422, detail=report.to_dict())
        except Exception as exc:
            elapsed = int((time.monotonic() - t0) * 1000)
            audit.record_build(
                ip=ip, decl=req.decl, log2=eff_log2, bs=eff_bs,
                status="build_error", error_msg=str(exc),
                elapsed_ms=elapsed,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # Classify the result. ``ParsedProgram.kind`` is 'agg' | 'port' |
    # 'sev' | 'distortion' -- the api only stores agg/port (sev and
    # distortion don't have density_df / pricing). Reject the rest.
    kind = _classify_object(obj)
    if kind not in ("agg", "port"):
        elapsed = int((time.monotonic() - t0) * 1000)
        audit.record_build(
            ip=ip, decl=req.decl, log2=eff_log2, bs=eff_bs,
            status="build_error",
            error_msg=f"unsupported kind {kind!r}",
            elapsed_ms=elapsed,
        )
        raise HTTPException(
            status_code=422,
            detail=f"api supports 'agg' and 'port' only; got {kind!r}",
        )

    entry = CacheEntry(
        obj=obj,
        decl=req.decl,
        log2=eff_log2,
        bs=eff_bs,
        kind=kind,
        name=getattr(obj, "name", "<anonymous>"),
        created_at=datetime.now(timezone.utc),
    )
    cache.put(oid, entry)
    elapsed = int((time.monotonic() - t0) * 1000)
    audit.record_build(
        ip=ip, decl=req.decl, log2=eff_log2, bs=eff_bs,
        status="ok", object_id=oid, kind=kind, elapsed_ms=elapsed,
    )
    return {
        "id": oid,
        "kind": kind,
        "name": entry.name,
        "warnings": [],
        "cached": False,
        "elapsed_ms": elapsed,
    }


def _classify_object(obj: Any) -> str:
    """Return 'agg' | 'port' | <type-name> for ``obj``.

    Uses class-name discrimination because Aggregate / Portfolio
    don't carry an explicit .kind attribute on the instances --
    that's only on ParsedProgram, which we don't see here (we
    called ``build()``, which unwraps).
    """
    cls = type(obj).__name__
    if cls == "Portfolio":
        return "port"
    if cls == "Aggregate":
        return "agg"
    return cls.lower()


# ----------------------------------------------------------------------
# GET /v1/objects -- cache listing
# ----------------------------------------------------------------------

@router.get("/objects", response_model=models.ObjectListResponse)
def list_objects(cache: ObjectCache = Depends(_get_cache)) -> dict:
    """Return a snapshot of cache contents, MRU last."""
    # Recover the (id, entry) pairing by scanning the cache.
    # The cache holds the OrderedDict internally; we expose it via
    # .list() but lose the id. Walk the internal dict directly
    # *with* the lock through a small helper.
    items = []
    # Internal access: read the OrderedDict items under lock.
    with cache._lock:  # noqa: SLF001 -- intentional cross-module use
        for oid, entry in cache._store.items():
            items.append({
                "id": oid,
                "kind": entry.kind,
                "name": entry.name,
                "ts": entry.created_at.isoformat(timespec="milliseconds"),
            })
    return {"objects": items}


# ----------------------------------------------------------------------
# GET /v1/objects/{id} -- manifest
# ----------------------------------------------------------------------

@router.get("/objects/{oid}", response_model=models.ObjectManifest)
def get_manifest(oid: str, cache: ObjectCache = Depends(_get_cache)) -> dict:
    entry = _resolve_object(oid, cache)
    return {
        "id": oid,
        "kind": entry.kind,
        "name": entry.name,
        "decl": entry.decl,
        "log2": entry.log2,
        "bs": entry.bs,
        "created_at": entry.created_at.isoformat(timespec="milliseconds"),
    }


# ----------------------------------------------------------------------
# DELETE /v1/objects/{id}
# ----------------------------------------------------------------------

@router.delete("/objects/{oid}", response_model=models.DeleteResponse)
def delete_object(oid: str, cache: ObjectCache = Depends(_get_cache)) -> dict:
    if not cache.delete(oid):
        raise HTTPException(status_code=404, detail=f"object {oid} not in cache")
    return {"ok": True}


# ----------------------------------------------------------------------
# GET /v1/objects/{id}/info
# ----------------------------------------------------------------------

@router.get("/objects/{oid}/info", response_model=models.InfoResponse)
def get_info(oid: str, cache: ObjectCache = Depends(_get_cache)) -> dict:
    entry = _resolve_object(oid, cache)
    return info_to_payload(entry.obj)


# ----------------------------------------------------------------------
# GET /v1/objects/{id}/description
# ----------------------------------------------------------------------

@router.get("/objects/{oid}/description", response_model=models.FrameResponse)
def get_description(oid: str, cache: ObjectCache = Depends(_get_cache)) -> dict:
    """Theoretical-vs-empirical moment table."""
    entry = _resolve_object(oid, cache)
    df = getattr(entry.obj, "describe", None)
    if df is None:
        raise HTTPException(
            status_code=400,
            detail=f"describe not available for {entry.kind!r}",
        )
    # ``describe`` is a property returning a DataFrame; we want its
    # named index in the payload too, so promote it to a column when
    # possible (reset_index_safe handles index/column collisions).
    df = reset_index_safe(df)
    return frame_to_payload(df)


# ----------------------------------------------------------------------
# GET /v1/objects/{id}/stats_df
# ----------------------------------------------------------------------

@router.get("/objects/{oid}/stats_df", response_model=models.FrameResponse)
def get_stats_df(oid: str, cache: ObjectCache = Depends(_get_cache)) -> dict:
    entry = _resolve_object(oid, cache)
    df = getattr(entry.obj, "stats_df", None)
    if df is None:
        raise HTTPException(
            status_code=400,
            detail=f"stats_df not available for {entry.kind!r}",
        )
    return frame_to_payload(reset_index_safe(df))


# ----------------------------------------------------------------------
# GET /v1/objects/{id}/density_df
# ----------------------------------------------------------------------

@router.get("/objects/{oid}/density_df", response_model=models.FrameResponse)
def get_density_df(
    oid: str,
    cols: str | None = Query(
        None,
        description="Comma-separated subset of column names.",
    ),
    start: int | None = Query(None, ge=0),
    stop: int | None = Query(None, ge=0),
    downsample: int | None = Query(None, ge=1, le=10_000),
    cache: ObjectCache = Depends(_get_cache),
) -> dict:
    """Paginated density_df slice.

    Without filters this is a 2**N-row table (potentially big);
    typical SPA use sets ``cols`` and ``downsample`` to limit
    payload size.
    """
    entry = _resolve_object(oid, cache)
    df = getattr(entry.obj, "density_df", None)
    if df is None:
        raise HTTPException(
            status_code=400,
            detail=f"density_df not available for {entry.kind!r}",
        )
    col_list = [c.strip() for c in cols.split(",")] if cols else None
    # density_df is indexed by loss; surface that as a column for
    # the SPA so it can render the x-axis without a separate query.
    # ``loss`` is already a column on the frame so reset_index_safe
    # avoids the collision.
    df = reset_index_safe(df)
    return frame_to_payload(
        df, cols=col_list, start=start, stop=stop, downsample=downsample,
    )


# ----------------------------------------------------------------------
# GET /v1/objects/{id}/kappa  -- Portfolio only
# ----------------------------------------------------------------------

@router.get("/objects/{oid}/kappa", response_model=models.FrameResponse)
def get_kappa(
    oid: str,
    downsample: int | None = Query(None, ge=1, le=10_000),
    cache: ObjectCache = Depends(_get_cache),
) -> dict:
    """Per-unit conditional expected losses (the ``exeqa_*`` slice)."""
    entry = _resolve_object(oid, cache)
    if entry.kind != "port":
        raise HTTPException(status_code=400, detail="kappa is Portfolio-only")
    df = entry.obj.density_df
    # Build the kappa-slice: loss + every exeqa_* column.
    exeqa = [c for c in df.columns if c.startswith("exeqa_")]
    if not exeqa:
        raise HTTPException(status_code=400, detail="no exeqa_* columns on density_df")
    df = reset_index_safe(df)[["loss", *exeqa]]
    return frame_to_payload(df, downsample=downsample)


# ----------------------------------------------------------------------
# GET /v1/objects/{id}/plot
# ----------------------------------------------------------------------

@router.get("/objects/{oid}/plot")
def get_plot(
    oid: str,
    kind: str = Query("density", description="density|cdf|qq|kappa"),
    format: str = Query("svg", description="svg|png"),
    width: float | None = Query(None, gt=0, le=30),
    height: float | None = Query(None, gt=0, le=30),
    dpi: float | None = Query(None, gt=0, le=600),
    cache: ObjectCache = Depends(_get_cache),
) -> Response:
    """Render the requested plot.

    Returns the image bytes directly (no JSON wrapper) with the
    matching ``Content-Type``. Streamed as a single ``Response``
    rather than ``StreamingResponse`` because plot bytes are
    already in-memory.
    """
    entry = _resolve_object(oid, cache)
    try:
        payload, media_type = render_plot(
            entry.obj, kind, fmt=format, width=width, height=height, dpi=dpi,
        )
    except ValueError as exc:
        # ValueError from render_plot is a 400 (bad request param).
        raise HTTPException(status_code=400, detail=str(exc))
    return Response(content=payload, media_type=media_type)


# ----------------------------------------------------------------------
# POST /v1/objects/{id}/pricing_at
# ----------------------------------------------------------------------

@router.post("/objects/{oid}/pricing_at", response_model=models.PricingResponse)
def post_pricing(
    oid: str,
    req: models.PricingRequest,
    cache: ObjectCache = Depends(_get_cache),
) -> dict:
    entry = _resolve_object(oid, cache)
    try:
        return run_pricing(
            entry.obj,
            p=req.p,
            a=req.a,
            ccoc=req.ccoc,
            distortion=req.distortion,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
