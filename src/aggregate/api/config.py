"""Settings for the api, driven by environment variables.

The :class:`Settings` class is built on ``pydantic-settings`` --
a small wrapper around Pydantic that reads env vars (prefixed
``AGGAPI_``) and validates them through the usual Pydantic
machinery. The end result is a typed, validated config object
loaded once at process start.

Flask users: this replaces ``app.config[...]``. The pattern of
"build a settings object once, inject it everywhere" is a FastAPI
idiom -- routes pull the live settings via the
:func:`get_settings` dependency (used with ``Depends``), not via
a global.

To override a value at runtime, set the env var before launching
``aggregate-api`` (e.g. ``AGGAPI_LOG2_CAP=20 aggregate-api``) or
pass it through ``monkeypatch.setenv`` in tests.

The defaults are the team-deploy preset: localhost-only bind,
modest log2 cap, 10 s build timeout, 50 objects in the cache.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-wide configuration.

    All knobs come from env vars beginning with ``AGGAPI_``.
    Empty strings on list-typed fields mean "unset" rather than
    "single empty element" -- see :meth:`_parse_cors_origins`.
    """

    # ``model_config`` is the Pydantic v2 idiom for class-level
    # config (it replaces v1's inner ``class Config``). ``env_prefix``
    # tells pydantic-settings to look for AGGAPI_HOST -> ``host``,
    # AGGAPI_PORT -> ``port``, etc. ``extra='ignore'`` lets us run
    # in an environment with unrelated env vars without complaint.
    model_config = SettingsConfigDict(
        env_prefix="AGGAPI_",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Network / server
    # ------------------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000

    # ------------------------------------------------------------------
    # Build pipeline
    # ------------------------------------------------------------------
    # Default log2 if the client omits it.
    log2_default: int = 16
    # Hard cap: requests above this fail with HTTP 422 (limit_exceeded).
    # The cap exists because 2**N grid points * sizeof(float) is the
    # FFT memory cost; 2**20 is ~8 MB per density array and is well
    # past anything legitimate users need on a team-deploy box.
    log2_cap: int = 18
    # Per-build wall-clock timeout; expires via threadpool.future.
    build_timeout_s: float = 10.0
    # Knowledge-base name forwarded to Underwriter(databases=...). The
    # api uses the default "test_suite" so builtins like ``agg.X`` and
    # named severities load.
    knowledge_base: str = "default"

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------
    cache_max: int = 50

    # ------------------------------------------------------------------
    # Audit log
    # ------------------------------------------------------------------
    # Default is per-user data dir; resolved lazily in audit.py so the
    # directory is only created when an AuditLog is actually opened.
    audit_db: str = str(Path.home() / ".aggregate" / "api" / "audit.db")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    plot_default_format: Literal["svg", "png"] = "svg"

    # ------------------------------------------------------------------
    # CORS
    # ------------------------------------------------------------------
    # Comma-separated list in the env var. Empty -> middleware skipped
    # (same-origin deploys don't pay the per-request CORS handling).
    # ``validation_alias`` overrides the auto-derived AGGAPI_CORS_ORIGINS_RAW
    # name so the env var stays AGGAPI_CORS_ORIGINS (matches docs).
    cors_origins_raw: str = Field(
        default="",
        validation_alias="AGGAPI_CORS_ORIGINS",
    )

    # ------------------------------------------------------------------
    # Static files (Plan D web build)
    # ------------------------------------------------------------------
    # When set, overrides the importlib-resources discovery in
    # app.py. Lets a developer point the running api at a Vite dev
    # build sitting in a sibling tree without reinstalling.
    static_dir: str = ""

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------
    @property
    def cors_origins(self) -> list[str]:
        """Parse ``AGGAPI_CORS_ORIGINS`` into a list of origins."""
        raw = self.cors_origins_raw.strip()
        if not raw:
            return []
        return [o.strip() for o in raw.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached process-wide settings.

    ``lru_cache`` makes this an effective singleton without the
    pitfalls of a module-level mutable -- and FastAPI's
    ``Depends(get_settings)`` knows how to use it directly.

    Tests that flip env vars via ``monkeypatch.setenv`` must call
    :func:`get_settings.cache_clear` (handled centrally by the
    ``client`` fixture in ``tests/api/conftest.py``) so the next
    read re-evaluates the environment.
    """
    return Settings()


# Module-level alias for code that doesn't need DI -- e.g. ``__main__.py``
# launching uvicorn off a single ``settings.host``/``port`` read.
# Reaches through the cache so tests with cleared cache still see
# the right object.
def _settings_attr(name: str):
    """Pass-through accessor that always reads from the cached object."""
    return getattr(get_settings(), name)


class _SettingsProxy:
    """Lazy attribute proxy so ``settings.host`` always hits live config.

    Avoids the trap of ``settings = Settings()`` at import time, which
    would lock in env-var values from before tests had a chance to
    monkeypatch them.
    """

    def __getattr__(self, name):
        return _settings_attr(name)


settings = _SettingsProxy()
