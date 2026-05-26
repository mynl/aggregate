"""Shared fixtures for the api test suite.

The ``client`` fixture builds a fresh FastAPI app per test against a
temp-directory audit DB and a clean cache. Each test gets isolation
from cross-test cache leakage and pollution of the user's
``~/.aggregate/api/audit.db``.

FastAPI's :class:`fastapi.testclient.TestClient` is a thin
synchronous wrapper around the ASGI app -- no actual HTTP server,
no port binding, no event-loop choreography. ``client.post(...)``
returns a ``httpx.Response``.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Fresh app + cache per test, audit DB in tmp_path."""
    # Push env vars *before* importing the app modules so
    # ``get_settings`` reads the patched values.
    monkeypatch.setenv("AGGAPI_AUDIT_DB", str(tmp_path / "audit.db"))
    monkeypatch.setenv("AGGAPI_LOG2_CAP", "20")
    # Default to no CORS for the common case; ``test_cors`` overrides.
    monkeypatch.setenv("AGGAPI_CORS_ORIGINS", "")

    # Import after monkeypatching so the cache_clear inside
    # create_app() picks up the new env.
    from aggregate.api.app import create_app

    app = create_app()
    with TestClient(app) as client_obj:
        yield client_obj
