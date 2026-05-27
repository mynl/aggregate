"""End-to-end tests for the parser-errors promotion.

These cover the wiring added in ``parser.py`` and ``underwriter.py``:

- ``UnderwritingParser.parse`` attaches ``.report`` (an
  :class:`~aggregate.parser_errors.ErrorReport`) to the wrapping
  :class:`ValueError` on every parse failure path.
- ``args[0]`` of the wrapping ``ValueError`` is the human-readable
  one-line summary (so ``str(e)`` is useful at the traceback tail).
- ``Underwriter._interpret_program`` logs the rendered report.
- Successful builds are untouched.

Unit-level coverage of the formatter itself lives in
:mod:`tests.test_parser_errors`.
"""

import logging

import pytest

from aggregate import build
from aggregate.parser import UnderwritingParser
from aggregate.parser_errors import ErrorReport


@pytest.fixture
def parser():
    return UnderwritingParser(lambda x: x)


# ----------------------------------------------------------------------
# .report is attached on each Lark-failure branch
# ----------------------------------------------------------------------

# Keyword typo: hits UnexpectedCharacters under the Earley + dynamic lexer.
_TYPO = "agg X 100 claims sev lognorm 100 cv 2 mixd poisson 0.5"


def test_report_attached_on_unexpected_characters(parser):
    with pytest.raises(ValueError) as info:
        parser.parse(_TYPO)
    report = getattr(info.value, "report", None)
    assert isinstance(report, ErrorReport)
    assert report.line == 1
    assert "mixed" in report.suggestions
    assert "mixd" in report.message


def test_report_attached_on_unexpected_token(parser):
    # Stray punctuation in a position where a number is expected --
    # depending on the grammar state this may surface as either
    # UnexpectedToken or UnexpectedCharacters. Either way .report
    # must be attached.
    with pytest.raises(ValueError) as info:
        parser.parse("agg X 100 claims sev lognorm 100 cv 2 + +")
    report = getattr(info.value, "report", None)
    assert isinstance(report, ErrorReport)
    assert report.line >= 1
    assert report.column >= 1


def test_report_attached_on_unexpected_eof(parser):
    # Truncated program -- ends mid-clause, triggers UnexpectedEOF or
    # an equivalent UnexpectedInput branch.
    with pytest.raises(ValueError) as info:
        parser.parse("agg X 100 claims sev lognorm")
    report = getattr(info.value, "report", None)
    assert isinstance(report, ErrorReport)
    # Rendered text should be safe to print even on the EOF path.
    assert isinstance(report.render(), str)


# ----------------------------------------------------------------------
# Legacy back-compat surface unchanged
# ----------------------------------------------------------------------

def test_str_is_human_readable_summary(parser):
    """``str(e)`` and ``e.args[0]`` carry the one-line summary."""
    with pytest.raises(ValueError) as info:
        parser.parse(_TYPO)
    s = str(info.value)
    # No legacy SimpleNamespace repr -- a plain human-readable line.
    assert "namespace" not in s
    assert "DecL parse error" in s
    assert "mixd" in s
    assert "Did you mean: mixed" in s
    # args[0] is the same string.
    assert info.value.args[0] == s


# ----------------------------------------------------------------------
# Underwriter._interpret_program logs the rendered report
# ----------------------------------------------------------------------

def test_interpret_program_logs_rendered_report(caplog):
    """When build() fails, the rendered report appears in the log."""
    caplog.set_level(logging.ERROR, logger="aggregate.underwriter")
    with pytest.raises(ValueError):
        build(_TYPO)
    # The new log message should contain the report's caret line.
    combined = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "DecL parse error" in combined
    assert "^" in combined  # the caret marker
    # And the suggestion should be present.
    assert "mixed" in combined.lower()


# ----------------------------------------------------------------------
# Successful builds are untouched
# ----------------------------------------------------------------------

def test_successful_build_unchanged():
    """The wrapping only activates on failure -- happy path still works."""
    a = build("agg Dice dfreq [3] dsev [1:6]")
    assert a is not None
    # Sanity: no spurious .report attribute on the agg.
    assert not hasattr(a, "report")
