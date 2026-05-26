"""Tests for /v1/examples (test_suite.agg loader)."""

from __future__ import annotations


def test_examples_grouped_by_category(client):
    r = client.get("/v1/examples")
    assert r.status_code == 200
    body = r.json()
    cats = body["categories"]
    letters = {c["letter"] for c in cats}
    # At least the canonical A-O categories from the Contents block
    # should be present (or close to it).
    assert "A" in letters
    assert "B" in letters


def test_examples_contain_dice(client):
    """The A.Dice00 example from test_suite.agg should appear in section A."""
    r = client.get("/v1/examples")
    cats = {c["letter"]: c for c in r.json()["categories"]}
    a = cats["A"]
    names = {item["name"] for item in a["items"]}
    assert any(n.startswith("A.Dice") for n in names)


def test_example_items_have_decl(client):
    """Each example carries non-empty DecL text."""
    r = client.get("/v1/examples")
    for cat in r.json()["categories"]:
        for item in cat["items"]:
            assert item["decl"].strip(), f"empty decl for {item['name']}"
            assert item["name"]
            # note may be None; if present it's a string.
            assert item["note"] is None or isinstance(item["note"], str)
