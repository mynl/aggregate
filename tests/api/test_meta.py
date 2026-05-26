"""Tests for /v1/health and /v1/meta."""

from __future__ import annotations


def test_health(client):
    r = client.get("/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert isinstance(body["version"], str) and body["version"]


def test_meta(client):
    r = client.get("/v1/meta")
    assert r.status_code == 200
    body = r.json()
    # log2_cap was set to 20 by the conftest fixture.
    assert body["log2_cap"] == 20
    assert body["plot_default_format"] in ("svg", "png")
    assert body["cache_max"] >= 1
