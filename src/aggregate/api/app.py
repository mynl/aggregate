"""FastAPI application factory.

``create_app()`` is the single entry point: it builds a fresh
:class:`fastapi.FastAPI`, installs CORS if configured, mounts the
``v1`` routers, and -- if the Plan D web bundle is present --
serves it as static files at ``/``.

Flask users
-----------

The factory pattern (``create_app``) lets uvicorn launch the app
lazily (one app per worker) and lets tests build a fresh instance
per test if they want. The Flask-equivalent is the standard
"application factory" used by larger Flask apps.

Routing model
-------------

Each ``routes/*.py`` module defines an :class:`APIRouter`
(``router`` global). The factory mounts all of them under
``/v1``. The OpenAPI schema (``/openapi.json``) and the
Swagger UI (``/docs``) are added by FastAPI automatically.
"""

from __future__ import annotations

from importlib.metadata import version as _pkg_version
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .config import Settings, get_settings
from .cors import install_cors
from .routes import decl as decl_routes
from .routes import examples as examples_routes
from .routes import meta as meta_routes
from .routes import objects as objects_routes


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app.

    Parameters
    ----------
    settings : Settings | None
        Optional pre-built settings object. ``None`` means "read
        from environment via :func:`get_settings`" -- the normal
        path. Tests can inject a custom ``Settings`` to bypass
        env-var setup.

    Returns
    -------
    FastAPI
        Configured app, routers mounted, ready for uvicorn.
    """
    if settings is None:
        # Force a fresh read so config-affecting monkeypatches in
        # tests are honored. Cache the result so subsequent
        # ``Depends(get_settings)`` calls return the same object.
        get_settings.cache_clear()
        settings = get_settings()
    # Drop cached cache+audit singletons -- the next request
    # rebuilds them against the (possibly just-changed) config.
    objects_routes.reset_singletons()

    app = FastAPI(
        title="aggregate api",
        description=(
            "HTTP/JSON wrapper around the aggregate library. "
            "Stand up DecL parsing, FFT-based compound distributions, "
            "and risk-pricing as a web service."
        ),
        version=_pkg_version("aggregate"),
        # Disable the default Pydantic-validation 422 schema in
        # OpenAPI -- it's noisy and we override the parse path
        # with our own ErrorReport response model.
    )

    install_cors(app, settings.cors_origins)

    # Mount all routers under /v1. FastAPI's include_router accepts
    # a prefix, similar to Flask's blueprint url_prefix kwarg.
    app.include_router(meta_routes.router, prefix="/v1", tags=["meta"])
    app.include_router(objects_routes.router, prefix="/v1", tags=["objects"])
    app.include_router(decl_routes.router, prefix="/v1", tags=["decl"])
    app.include_router(examples_routes.router, prefix="/v1", tags=["examples"])

    # Static-file mount for the SPA (Plan D). Conditional because
    # the api ships independently of the web build; if the web
    # bundle isn't present, the api is api-only and ``/`` returns
    # 404 (which is fine for backend-only deploys).
    static_dir = _resolve_static_dir(settings)
    if static_dir is not None and static_dir.exists():
        # ``html=True`` tells StaticFiles to serve index.html for /
        # *and* fall back to it for unknown paths -- the SPA-style
        # client-side routing behavior the web app needs.
        app.mount(
            "/",
            StaticFiles(directory=str(static_dir), html=True),
            name="static",
        )

    return app


def _resolve_static_dir(settings: Settings) -> Path | None:
    """Locate the SPA bundle directory.

    Order of precedence:

    1. ``AGGAPI_STATIC_DIR`` env var (handed via ``settings.static_dir``).
       Useful for developing the SPA out of a separate tree.
    2. ``src/aggregate/api/static`` inside the installed package.

    Returns None if neither is set / exists.
    """
    if settings.static_dir:
        return Path(settings.static_dir)
    # importlib.resources style: the static dir lives next to
    # this file. We use Path rather than files() because StaticFiles
    # needs a real filesystem path, not a Traversable.
    pkg_root = Path(__file__).resolve().parent
    return pkg_root / "static"
