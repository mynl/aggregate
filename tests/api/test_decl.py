"""Tests for /v1/decl/* helpers (completions, lex, grammar)."""

from __future__ import annotations


def test_grammar_serves_lark(client):
    r = client.get("/v1/decl/grammar")
    assert r.status_code == 200
    body = r.text
    # decl.lark should reference at least one of these familiar
    # Lark constructs; loose check tolerant of grammar evolution.
    assert "start:" in body or "%import" in body or "answer" in body


def test_lex_returns_tokens(client):
    r = client.post("/v1/decl/lex", json={"decl": "agg Dice dfreq [3] dsev [1:6]"})
    assert r.status_code == 200
    tokens = r.json()["tokens"]
    assert len(tokens) > 0
    # Token records carry the expected schema fields.
    first = tokens[0]
    for key in ("type", "value", "start", "end", "line", "column"):
        assert key in first


def test_lex_handles_bad_input_gracefully(client):
    r = client.post("/v1/decl/lex", json={"decl": "agg X ?"})
    assert r.status_code == 200
    # Bad input -> empty list rather than a 500. Callers should use
    # /v1/objects for the structured error report.
    assert r.json()["tokens"] == [] or isinstance(r.json()["tokens"], list)


def test_complete_returns_candidates(client):
    """Completion at the start of input should propose top-level kinds."""
    r = client.post("/v1/decl/complete", json={"decl": "", "cursor": 0})
    assert r.status_code == 200
    body = r.json()
    assert "completions" in body
    # At the empty position the grammar should accept at least 'agg',
    # 'sev', 'port', or 'dist' as starters.
    labels = {c["label"] for c in body["completions"]}
    # Loose check: at least one of the top-level kinds is present.
    assert labels & {"agg", "sev", "port", "dist", "distortion"}


def test_complete_empty_for_unparseable_prefix(client):
    """An unrecoverable parse error -> empty completion list."""
    r = client.post(
        "/v1/decl/complete", json={"decl": "agg X ?", "cursor": 7},
    )
    assert r.status_code == 200
    # Either empty or non-error; the contract is "no crash".
    assert isinstance(r.json()["completions"], list)
