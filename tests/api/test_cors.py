"""Tests for CORS middleware behavior."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def _build_client(tmp_path, monkeypatch, *, origins: str):
    """Build a fresh client with an explicit AGGAPI_CORS_ORIGINS value."""
    monkeypatch.setenv("AGGAPI_AUDIT_DB", str(tmp_path / "audit.db"))
    monkeypatch.setenv("AGGAPI_CORS_ORIGINS", origins)
    from aggregate.api.app import create_app

    return TestClient(create_app())


def test_cors_headers_present_when_origin_listed(tmp_path, monkeypatch):
    client = _build_client(
        tmp_path, monkeypatch, origins="http://localhost:5173",
    )
    r = client.get("/v1/health", headers={"Origin": "http://localhost:5173"})
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_absent_when_empty_origins(tmp_path, monkeypatch):
    """Empty origins -> middleware not installed -> no CORS headers."""
    client = _build_client(tmp_path, monkeypatch, origins="")
    r = client.get("/v1/health", headers={"Origin": "http://localhost:5173"})
    assert r.status_code == 200
    # Without middleware, no access-control-allow-origin header is added.
    assert "access-control-allow-origin" not in r.headers


def test_cors_unlisted_origin_blocked(tmp_path, monkeypatch):
    """A request from an origin not in the allowlist gets no CORS allow."""
    client = _build_client(
        tmp_path, monkeypatch, origins="http://localhost:5173",
    )
    r = client.get("/v1/health", headers={"Origin": "https://evil.example"})
    # The response still succeeds (CORS is browser-side enforcement);
    # the missing header is what stops the browser from delivering it.
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") != "https://evil.example"
