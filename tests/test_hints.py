"""Tests for the ``note{...}`` / ``hints{...}`` split (1.0.0a25).

``note{...}`` is pure free-text annotation; ``hints{...}`` carries
``key=value;`` build settings parsed by the underwriter under *caller-wins*
rules (explicit ``build()`` kwargs override in-program hints). The two clauses
are optional and order-free, at most one of each.

The DecL programs exercised here are mirrored in
``src/aggregate/agg/decl-testers.agg`` under the HINTS section.
"""

import logging

import pytest

from aggregate import build
from aggregate.parser import UnderwritingParser
from aggregate.underwriter import _coerce_hint_value, _parse_hints

PARSER = UnderwritingParser(lambda x: {})


def _spec(program):
    """Parse a single DecL program and return its spec dict."""
    _kind, _name, spec = PARSER.parse(program)
    return spec


# ----------------------------------------------------------------------
# Grammar: note / hints are optional and order-free
# ----------------------------------------------------------------------
BASE = "agg A 5 claims sev lognorm 10 cv 2 poisson"


@pytest.mark.parametrize(
    "program, note, hints",
    [
        (BASE, "", ""),
        (BASE + " note{hi}", "hi", ""),
        (BASE + " hints{log2=18}", "", "log2=18"),
        (BASE + " note{hi} hints{log2=18}", "hi", "log2=18"),
        (BASE + " hints{log2=18} note{hi}", "hi", "log2=18"),
    ],
    ids=["neither", "note-only", "hints-only", "note-first", "hints-first"],
)
def test_trailer_combinations(program, note, hints):
    """All four present/absent combinations parse, in either order, to the
    expected ``note`` / ``hints`` spec values."""
    spec = _spec(program)
    assert spec["note"] == note
    assert spec["hints"] == hints


def test_sev_and_port_carry_hints():
    """``hints`` flows through severity and (mid-rule) portfolio trailers."""
    s = _spec("sev S lognorm 100 cv 1 note{a sev} hints{bs=1}")
    assert s["note"] == "a sev" and s["hints"] == "bs=1"
    # a port's note/hints sit between the name and the agg list; the trailing
    # hints here attach to the last agg UNIT (grammar position is preserved).
    p = _spec("port P note{prose} agg U1 1 claim sev lognorm 10 cv 1 fixed "
              "agg U2 1 claim sev lognorm 20 cv 1 fixed hints{log2=8}")
    assert p["note"] == "prose" and p["hints"] == ""
    assert p["spec"][-1][2]["hints"] == "log2=8"


# ----------------------------------------------------------------------
# Value coercion / hint parsing
# ----------------------------------------------------------------------
def test_coerce_hint_value():
    assert _coerce_hint_value("18") == 18 and isinstance(_coerce_hint_value("18"), int)
    assert _coerce_hint_value("0.5") == 0.5
    assert _coerce_hint_value("1/64") == pytest.approx(0.015625)
    assert _coerce_hint_value("True") is True
    assert _coerce_hint_value("False") is False
    assert _coerce_hint_value("discrete") == "discrete"


def test_parse_hints_typed_dict():
    d = _parse_hints("log2=18; bs=1/64")
    assert d == {"log2": 18, "bs": pytest.approx(0.015625)}


def test_parse_hints_unknown_key_warns_and_drops(caplog):
    with caplog.at_level(logging.WARNING, logger="aggregate.underwriter"):
        d = _parse_hints("frobnicate=3")
    assert d == {}
    assert "unknown key" in caplog.text


def test_parse_hints_duplicate_last_wins(caplog):
    with caplog.at_level(logging.WARNING, logger="aggregate.underwriter"):
        d = _parse_hints("bs=1; bs=2")
    assert d == {"bs": 2}
    assert "duplicate key" in caplog.text


def test_parse_hints_malformed_clause_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="aggregate.underwriter"):
        d = _parse_hints("this has no equals")
    assert d == {}
    assert "malformed" in caplog.text


# ----------------------------------------------------------------------
# Build-time behaviour
# ----------------------------------------------------------------------
def test_hint_applied_when_caller_unset():
    a = build(BASE.replace("A", "HintLog2") + " hints{log2=12}")
    assert a.log2 == 12


def test_caller_wins_over_hint():
    a = build(BASE.replace("A", "HintCaller") + " hints{log2=12}", log2=14)
    assert a.log2 == 14


def test_fraction_bs_hint():
    a = build("agg HF 1 claim sev gamma 50 cv 0.1 fixed hints{bs=1/64; log2=10}")
    assert a.bs == pytest.approx(0.015625)
    assert a.log2 == 10


def test_prose_with_equals_builds():
    """The original failing program: `=`/`<=`/`;` in note prose no longer
    crashes the build (notes are pure text)."""
    a = build("agg HintProse dfreq [3] dsev [-2 5] [.5 .5] "
              "note{closed form support -6,1,8,15; needs x_min<=-6}")
    assert a.note == "closed form support -6,1,8,15; needs x_min<=-6"
    assert a.hints == ""


def test_unknown_hint_key_still_builds(caplog):
    with caplog.at_level(logging.WARNING, logger="aggregate.underwriter"):
        a = build(BASE.replace("A", "HintUnknown") + " hints{frobnicate=3}")
    assert a is not None
    assert "unknown key" in caplog.text


def test_settings_in_note_warns_deprecation(caplog):
    """A note that still looks like it carries settings triggers the
    deprecation warning (but the note is treated as pure text, not acted on)."""
    with caplog.at_level(logging.WARNING, logger="aggregate.underwriter"):
        a = build(BASE.replace("A", "HintDeprec") + " note{log2=20}")
    assert "no longer sets build options" in caplog.text
    # the note's log2=20 was NOT applied (it is pure text now)
    assert a.log2 != 20
