"""Meta routes -- health check and config introspection.

Routes
------

* ``GET /v1/health`` -- liveness probe (``{ok: true, version}``).
  Used by Caddy / k8s health checks and by the SPA's "server up?"
  splash logic.
* ``GET /v1/meta`` -- runtime config (log2_cap, build_timeout, etc.)
  so the SPA can configure its form widgets (e.g. set the log2
  slider's max to ``log2_cap``).

These are cheap reads; they don't touch the cache or audit log.
"""

from __future__ import annotations

from importlib.metadata import version as _pkg_version

from fastapi import APIRouter, Depends

from .. import models
from ..config import Settings, get_settings


# APIRouter is FastAPI's analogue of a Flask Blueprint -- a group
# of routes that the app factory mounts at a prefix.
router = APIRouter()


@router.get("/health", response_model=models.HealthResponse)
def health() -> dict:
    """Trivial liveness check.

    Returns the package version so a curl-from-prod debugging
    session can confirm which build it's talking to without an
    OpenAPI fetch.
    """
    return {"ok": True, "version": _pkg_version("aggregate")}


@router.get("/meta", response_model=models.MetaResponse)
def meta(settings: Settings = Depends(get_settings)) -> dict:
    """Echo the live config knobs the SPA needs.

    ``Depends(get_settings)`` is FastAPI's dependency-injection
    primitive -- the function parameter declared with ``Depends``
    is filled by calling the dependency, with result caching
    handled automatically. Flask users: closest analogue is
    ``flask.current_app.config``, but typed and validated.
    """
    return {
        "version": _pkg_version("aggregate"),
        "log2_cap": settings.log2_cap,
        "log2_default": settings.log2_default,
        "build_timeout_s": settings.build_timeout_s,
        "cache_max": settings.cache_max,
        "plot_default_format": settings.plot_default_format,
    }
