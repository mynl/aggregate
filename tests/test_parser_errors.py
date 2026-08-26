"""Unit tests for :mod:`aggregate.parser_errors`.

The formatter is pure: it takes a Lark exception (or a ValueError
wrapping one) and returns an :class:`ErrorReport`. These tests
exercise the public entry point against real
``UnderwritingParser.parse`` failures plus targeted unit tests for the
private helpers.
"""

import json

import pytest

from aggregate.parser import UnderwritingParser
from aggregate.parser_errors import (
    ErrorReport,
    _caret,
    _contextual_hint,
    _did_you_mean,
    _extract_word,
    _label,
    _preceding_text,
    _source_line,
    format_error,
)


@pytest.fixture
def parser():
    """A parser with a no-op safe_lookup; sufficient for parse-failure tests."""
    return UnderwritingParser(lambda x: x)


def _try_parse(parser, text):
    """Helper: parse ``text`` and return the exception that was raised."""
    with pytest.raises(Exception) as info:
        parser.parse(text)
    return info.value


# ----------------------------------------------------------------------
# format_error against real parser output
# ----------------------------------------------------------------------

# With Earley + Lark's dynamic lexer, the vast majority of DecL parse
# failures surface as UnexpectedCharacters rather than UnexpectedToken
# (the lexer refuses to emit a token when none in the parser-state's
# allowed set matches). These tests exercise that path against real
# parser output.

# A keyword typo at a position whose `allowed` set contains the intended
# keyword -- triggers the "did you mean" suggestion.
_TYPO_INPUT = "agg X 100 claims sev lognorm 100 cv 2 mixd poisson 0.5"
_TYPO_WORD = "mixd"


def test_unexpected_chars_extracts_bad_word(parser):
    exc = _try_parse(parser, _TYPO_INPUT)
    report = format_error(_TYPO_INPUT, exc)
    assert isinstance(report, ErrorReport)
    assert report.line == 1
    assert report.column > 0
    assert report.got == _TYPO_WORD


def test_unexpected_chars_did_you_mean(parser):
    exc = _try_parse(parser, _TYPO_INPUT)
    report = format_error(_TYPO_INPUT, exc)
    # MIXED is in the allowed set at this parser state, so "mixed"
    # should appear as a suggestion for "mixd".
    assert "mixed" in report.suggestions


def test_unexpected_chars_caret_spans_word(parser):
    exc = _try_parse(parser, _TYPO_INPUT)
    report = format_error(_TYPO_INPUT, exc)
    assert report.caret.count("^") >= len(_TYPO_WORD)
    assert report.caret.lstrip(" ").startswith("^")


def test_unexpected_chars_source_echoed(parser):
    exc = _try_parse(parser, _TYPO_INPUT)
    report = format_error(_TYPO_INPUT, exc)
    assert report.source == _TYPO_INPUT
    assert report.source_line == _TYPO_INPUT


def test_unexpected_chars_message_mentions_word(parser):
    exc = _try_parse(parser, _TYPO_INPUT)
    report = format_error(_TYPO_INPUT, exc)
    assert _TYPO_WORD in report.message


def test_unexpected_chars_punctuation_fallback(parser):
    # When the bad character is punctuation rather than identifier-like,
    # `got` is just the single char and there's no word to suggest from.
    bad = "agg X 100 claims sev lognorm 100 cv 2 ?"
    exc = _try_parse(parser, bad)
    report = format_error(bad, exc)
    assert report.got == "?"
    assert report.suggestions == []
    assert "character" in report.message


def test_format_error_unwraps_value_error(parser):
    # UnderwritingParser wraps Lark exceptions in ValueError via
    # `raise ... from e`. format_error must still find the Lark
    # exception via __cause__ and produce a structured report.
    exc = _try_parse(parser, _TYPO_INPUT)
    assert isinstance(exc, ValueError)
    report = format_error(_TYPO_INPUT, exc)
    # Structured report, not the empty fallback.
    assert report.got == _TYPO_WORD
    assert report.expected  # non-empty allowed set was recovered


# ----------------------------------------------------------------------
# _did_you_mean
# ----------------------------------------------------------------------

def test_did_you_mean_finds_mixed():
    assert "mixed" in _did_you_mean("mixedd", ["MIXED", "NET", "CEDED"])


def test_did_you_mean_empty_for_nonsense():
    assert _did_you_mean("zzzzzz", ["MIXED", "NET", "CEDED"]) == []


def test_did_you_mean_empty_string_returns_empty():
    assert _did_you_mean("", ["MIXED", "NET", "CEDED"]) == []


def test_did_you_mean_ignores_anonymous_terminals():
    # Anonymous terminals (__ANON_*) must not contaminate the suggestion
    # pool.
    assert _did_you_mean("anon_0", ["__ANON_0", "__ANON_1"]) == []


def test_did_you_mean_case_insensitive():
    # 'MiXeD' should still nominate 'mixed' as the close match.
    assert "mixed" in _did_you_mean("MiXeD", ["MIXED", "NET"])


# ----------------------------------------------------------------------
# _label
# ----------------------------------------------------------------------

def test_label_known_generic_terminal():
    assert _label("NUMBER") == "a number"
    assert _label("ID") == "an identifier"


def test_label_anonymous_skipped():
    assert _label("__ANON_0") == "token"


def test_label_known_keyword_terminal():
    # MIXED is in the curated table.
    assert _label("MIXED") == "'mixed'"


def test_label_unknown_keyword_falls_back_to_lowercase():
    # Unknown all-caps terminals get the lowercase-in-quotes treatment.
    assert _label("NEWWORD") == "'newword'"


def test_label_none_returns_generic_token():
    assert _label(None) == "token"


# ----------------------------------------------------------------------
# _source_line / _caret
# ----------------------------------------------------------------------

def test_source_line_multiline():
    src = "line one\nline two\nline three"
    assert _source_line(src, 2) == "line two"


def test_source_line_out_of_range_returns_empty():
    assert _source_line("only one", 5) == ""


def test_caret_column_one_is_first_char():
    assert _caret(1, 1) == "^"


def test_caret_width_at_least_one():
    assert _caret(3, 0) == "  ^"


def test_caret_multi_char_token():
    assert _caret(5, 4) == "    ^^^^"


# ----------------------------------------------------------------------
# _extract_word
# ----------------------------------------------------------------------

def test_extract_word_recovers_identifier():
    assert _extract_word("agg X 100 mixd poisson", 10) == "mixd"


def test_extract_word_stops_at_whitespace():
    assert _extract_word("foo bar", 0) == "foo"


def test_extract_word_empty_for_punctuation():
    assert _extract_word("agg ? extra", 4) == ""


def test_extract_word_handles_decl_id_chars():
    # The DecL ID regex character class is [A-Za-z0-9._:~\-].
    assert _extract_word("name.with-dashes:and~tilde rest", 0) == \
        "name.with-dashes:and~tilde"


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------

def test_render_contains_caret_and_line_marker(parser):
    exc = _try_parse(parser, _TYPO_INPUT)
    rendered = format_error(_TYPO_INPUT, exc).render()
    assert "^" in rendered
    assert "line" in rendered.lower()


def test_render_includes_suggestion_when_present(parser):
    exc = _try_parse(parser, _TYPO_INPUT)
    rendered = format_error(_TYPO_INPUT, exc).render()
    assert "Did you mean" in rendered
    assert "mixed" in rendered


def test_render_truncates_long_expected_lists():
    # Synthesize a report with > 10 expected entries to exercise the
    # truncation branch in ``render``.
    report = ErrorReport(
        line=1,
        column=1,
        source="x",
        source_line="x",
        caret="^",
        got="x",
        got_type=None,
        expected=[f"label_{i}" for i in range(20)],
        expected_terminals=[f"T{i}" for i in range(20)],
        suggestions=[],
        message="msg",
    )
    rendered = report.render()
    assert "..." in rendered


# ----------------------------------------------------------------------
# Contextual hints
# ----------------------------------------------------------------------

# ``as`` closes an expense group, so a following ``and`` cannot attach.
# The token-level report ("Unexpected 'and'") is correct but says nothing
# about the two-level expense list, which is what the hint supplies.
_ENGINE = (
    'xpnl E.X 10000 premium as "GWP" less agg B.A 6500 loss 1000 xs 0 '
    'sev lognorm 100 cv 2 mixed ig 0.25 less '
)
_AND_AFTER_EXPENSE_LABEL = (
    _ENGINE + "500 fixed expense as FE and 15% premium expense as Comm"
)


def test_hint_fires_on_and_after_expense_label(parser):
    exc = _try_parse(parser, _AND_AFTER_EXPENSE_LABEL)
    report = format_error(_AND_AFTER_EXPENSE_LABEL, exc)
    assert report.hint is not None
    assert "expense group" in report.hint


def test_hint_fires_on_quoted_expense_label(parser):
    text = _ENGINE + '500 fixed expense as "Fixed Expense" and 15% premium expense'
    report = format_error(text, _try_parse(parser, text))
    assert report.hint is not None


def test_hint_quiet_on_and_outside_the_expense_clause(parser):
    # ``and`` is equally unexpected here, but no expense label precedes it,
    # so the expense hint must not fire.
    text = "xpnl E.X 10000 premium less agg.A and 5"
    report = format_error(text, _try_parse(parser, text))
    assert report.hint is None


def test_hint_quiet_on_plain_keyword_typo(parser):
    report = format_error(_TYPO_INPUT, _try_parse(parser, _TYPO_INPUT))
    assert report.hint is None


def test_render_includes_hint_when_present(parser):
    exc = _try_parse(parser, _AND_AFTER_EXPENSE_LABEL)
    rendered = format_error(_AND_AFTER_EXPENSE_LABEL, exc).render()
    assert "Hint:" in rendered


def test_summary_includes_hint_when_present(parser):
    exc = _try_parse(parser, _AND_AFTER_EXPENSE_LABEL)
    report = format_error(_AND_AFTER_EXPENSE_LABEL, exc)
    assert report.hint in report.summary


def test_render_omits_hint_when_absent(parser):
    rendered = format_error(_TYPO_INPUT, _try_parse(parser, _TYPO_INPUT)).render()
    assert "Hint:" not in rendered


def test_contextual_hint_needs_a_got_literal():
    assert _contextual_hint("500 fixed expense as FE and", 1, 28, None) is None


def test_preceding_text_single_line():
    assert _preceding_text("abcdef", 1, 4) == "abc"


def test_preceding_text_spans_earlier_lines():
    assert _preceding_text("one\ntwo\nthree", 3, 3) == "one\ntwo\nth"


def test_preceding_text_out_of_range_is_empty():
    assert _preceding_text("one", 9, 1) == ""


# ----------------------------------------------------------------------
# to_dict / JSON-safety
# ----------------------------------------------------------------------

def test_to_dict_keys_match_dataclass_fields(parser):
    exc = _try_parse(parser, _TYPO_INPUT)
    d = format_error(_TYPO_INPUT, exc).to_dict()
    assert set(d) == {
        "line", "column", "source", "source_line", "caret",
        "got", "got_type", "expected", "expected_terminals",
        "suggestions", "message", "hint",
    }


def test_to_dict_json_safe(parser):
    exc = _try_parse(parser, _TYPO_INPUT)
    d = format_error(_TYPO_INPUT, exc).to_dict()
    json.dumps(d)  # must not raise


# ----------------------------------------------------------------------
# Fallback path for non-Lark exceptions
# ----------------------------------------------------------------------

def test_format_error_fallback_for_non_lark_exception():
    # A bare ValueError with no UnexpectedInput in its cause chain
    # should still yield a usable (if minimal) report.
    report = format_error("some source", ValueError("boom"))
    assert isinstance(report, ErrorReport)
    assert "boom" in report.message
