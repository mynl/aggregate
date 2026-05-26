"""Tests for the SQLite audit log."""

from __future__ import annotations

import sqlite3


def test_audit_row_written_on_success(client, tmp_path):
    """A successful build leaves a status='ok' row in audit.db."""
    client.post("/v1/objects", json={"decl": "agg X dfreq [3] dsev [1:6]"})
    db = tmp_path / "audit.db"
    assert db.exists()
    rows = sqlite3.connect(db).execute(
        "SELECT status, decl FROM builds"
    ).fetchall()
    assert any(r[0] == "ok" for r in rows)


def test_audit_row_written_on_parse_error(client, tmp_path):
    """A parse error leaves a status='parse_error' row."""
    client.post(
        "/v1/objects",
        json={"decl": "agg X 100 claims sev lognorm 100 cv 2 mixd poisson 0.5"},
    )
    db = tmp_path / "audit.db"
    statuses = [
        r[0]
        for r in sqlite3.connect(db).execute(
            "SELECT status FROM builds"
        ).fetchall()
    ]
    assert "parse_error" in statuses


def test_audit_captures_elapsed_ms(client, tmp_path):
    client.post("/v1/objects", json={"decl": "agg X dfreq [3] dsev [1:6]"})
    db = tmp_path / "audit.db"
    rows = sqlite3.connect(db).execute(
        "SELECT elapsed_ms FROM builds WHERE status='ok'"
    ).fetchall()
    assert all(r[0] >= 0 for r in rows)
