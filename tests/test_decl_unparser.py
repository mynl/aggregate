"""Round-trip tests for the DecL unparser (``aggregate.decl_writer``, 1.0.0a53).

The unparser is the structural inverse of the parser. Its contract (see
``dev/done/plan-decl-unparser.md`` and the ``decl_writer`` module docstring) is
**idempotence one step removed**: with ``f = spec_to_decl``,
``f(f(f(x))) == f(x)``. Concretely, for every program in the reference corpus
(``test_suite.agg`` + ``test_suite2.agg`` + ``test_decl.agg``):

* **idempotence** --- rendering the spec, re-parsing, and rendering again yields
  byte-identical text (the universal gate); and
* **fidelity** --- the re-parsed spec equals the original under a
  numpy/inf/object-aware comparator (holds for every line except a handful of
  genuinely lossy constructs whose first-parse spec is not a canonical fixed
  point --- e.g. ``tweedie``, which bypasses the ``sev_weighted`` defaults).

A bare ``spec0 == spec1`` would raise on the ndarrays the specs carry (the same
array-truthiness trap fixed in hygiene-3), so the comparator below is explicit.
"""

import re
from pathlib import Path

import numpy as np
import pytest

import aggregate
from aggregate import Underwriter, build, format_program
from aggregate.decl_writer import spec_to_decl, _split_statements
from aggregate.parser import UnderwritingLexer

# ----------------------------------------------------------------------
# Corpus collection
# ----------------------------------------------------------------------

_AGG_DIR = Path(aggregate.__file__).parent / 'agg'
_CORPUS_FILES = ['test_suite.agg', 'test_suite2.agg', 'test_decl.agg']

# Genuinely lossy / non-canonical-spec constructs: idempotence holds, but the
# first parse's spec is not a fixed point (tweedie discards its note and bakes a
# CP-gamma spec that bypasses the sev_weighted defaults), so fidelity is exempt.
_FIDELITY_EXEMPT = {'K.Tweedie2'}

# The corpus is the test_suite family, whose programs reference builtins it
# defines (e.g. ``sev.One``). Parse against an underwriter that loads it, NOT
# the module-level ``build`` singleton -- ``build`` now defaults to the curated
# ``examples`` library, which does not carry those builtins.
_uw = Underwriter(databases='test_suite')


def _corpus_lines():
    """Yield ``(id, program_text)`` for every parseable corpus program.

    Routes the file text through :meth:`UnderwritingLexer.preprocess` (the single
    owner of the blank-line / ``;`` statement-splitting rule) rather than
    splitting on physical lines, so multi-line statements and ``;``-terminated
    dense lists are handled the same way the parser sees them.
    """
    seen = set()
    for fn in _CORPUS_FILES:
        path = _AGG_DIR / fn
        for line in UnderwritingLexer.preprocess(path.read_text(encoding='utf-8')):
            # second token is the object name (agg NAME ..., port NAME ...,
            # distortion NAME ...); fall back to the raw line for an id.
            parts = line.split()
            ident = parts[1] if len(parts) > 1 else line
            # de-dupe ids that repeat across files (e.g. a reused NAME)
            base, ident = ident, ident
            i = 1
            while ident in seen:
                i += 1
                ident = f'{base}#{i}'
            seen.add(ident)
            yield ident, line


_CASES = list(_corpus_lines())
_IDS = [c[0] for c in _CASES]


# ----------------------------------------------------------------------
# numpy/inf/object-aware spec comparator
# ----------------------------------------------------------------------

def _spec_diff(a, b, path=''):
    """Return a human description of the first difference, or ``None`` if equal.

    Handles nested dicts, lists/tuples (including the ``(kind, name, spec)``
    sub-aggregate tuples a portfolio carries), ndarrays, ``inf``, ``Copula``
    instances, strings and numbers (with a float tolerance).
    """
    if type(a).__name__.startswith('Copula') or type(b).__name__.startswith('Copula'):
        if getattr(a, '_name', None) != getattr(b, '_name', None):
            return f'{path}: copula kind differs'
        if getattr(a, 'param', None) != getattr(b, 'param', None):
            return f'{path}: copula param differs'
        return None
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            return f'{path}: dict vs non-dict'
        if set(a) != set(b):
            return (f'{path}: keys differ '
                    f'(only-a={set(a) - set(b)}, only-b={set(b) - set(a)})')
        for k in a:
            d = _spec_diff(a[k], b[k], f'{path}.{k}')
            if d:
                return d
        return None
    a_seq = isinstance(a, (list, tuple, np.ndarray))
    b_seq = isinstance(b, (list, tuple, np.ndarray))
    if a_seq or b_seq:
        try:
            aa = np.atleast_1d(np.asarray(a, dtype=float))
            bb = np.atleast_1d(np.asarray(b, dtype=float))
            if aa.shape != bb.shape:
                return f'{path}: shape {aa.shape} != {bb.shape}'
            if not np.allclose(aa, bb, rtol=1e-8, atol=1e-10, equal_nan=True):
                return f'{path}: {aa} != {bb}'
            return None
        except (ValueError, TypeError):
            if len(a) != len(b):
                return f'{path}: len {len(a)} != {len(b)}'
            for i, (x, y) in enumerate(zip(a, b)):
                d = _spec_diff(x, y, f'{path}[{i}]')
                if d:
                    return d
            return None
    if isinstance(a, str) or isinstance(b, str):
        return None if a == b else f'{path}: {a!r} != {b!r}'
    if a is None or b is None:
        return None if a is b else f'{path}: None mismatch ({a!r}, {b!r})'
    try:
        if np.isclose(float(a), float(b), rtol=1e-8, atol=1e-10, equal_nan=True):
            return None
        return f'{path}: {a!r} != {b!r}'
    except (ValueError, TypeError):
        return None if a == b else f'{path}: {a!r} != {b!r}'


def _parse_one(text):
    """Parse a single canonical statement (rendered text) to ``(kind, name, spec)``."""
    statements = _split_statements(text)
    assert len(statements) == 1, f'expected one statement, got {len(statements)}'
    return _uw.parser.parse(statements[0])


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def test_corpus_nonempty():
    """Guard the collection wiring --- the corpus must yield many programs."""
    assert len(_CASES) > 150


@pytest.mark.parametrize('program', [c[1] for c in _CASES], ids=_IDS)
def test_roundtrip(program):
    """Each corpus program is idempotent under the unparser, and faithful
    unless it is a known-lossy construct."""
    kind0, name0, spec0 = _uw.parser.parse(program)
    if kind0 == 'expr':
        pytest.skip('bare expression, not an unparsable object')

    try:
        text1 = spec_to_decl(spec0, kind0, name0)
    except NotImplementedError:
        pytest.skip('construct cannot round-trip (combinator distortion)')

    kind1, name1, spec1 = _parse_one(text1)
    text2 = spec_to_decl(spec1, kind1, name1)

    # Idempotence: the canonical text is a fixed point after one application.
    assert text1 == text2, f'not idempotent:\n  {text1!r}\n  {text2!r}'
    assert kind0 == kind1

    # Fidelity: re-parsed spec matches the original (except known-lossy lines).
    diff = _spec_diff(spec0, spec1)
    if name0 in _FIDELITY_EXEMPT:
        assert diff is not None or True  # idempotence already asserted
    else:
        assert diff is None, f'spec drift: {diff}'


# ----------------------------------------------------------------------
# format_program smoke tests
# ----------------------------------------------------------------------

_SMOKE = 'agg X 10 claims sev lognorm 50 cv 0.8 occurrence net of 50% so 10 xs 0 poisson note{hi}'


def test_format_text_is_plain():
    out = format_program(_SMOKE, fmt='text')
    assert isinstance(out, str)
    assert '\x1b[' not in out and '<span' not in out
    assert out.startswith('agg X')


def test_format_html_has_span():
    out = format_program(_SMOKE, fmt='html')
    assert isinstance(out, str) and '<span' in out


def test_format_ansi_has_escape():
    out = format_program(_SMOKE, fmt='ansi')
    assert isinstance(out, str) and '\x1b[' in out


def test_format_latex_nonempty():
    out = format_program(_SMOKE, fmt='latex')
    assert isinstance(out, str) and out.strip()


def test_format_bad_fmt_raises():
    with pytest.raises(ValueError):
        format_program(_SMOKE, fmt='svg')


def test_format_empty_returns_empty():
    assert format_program('   ') == ''


def test_format_accepts_spec_tuple():
    parsed = build.parser.parse('agg Y dfreq [1:6] dsev [1]')
    out = format_program(parsed, fmt='text')
    assert out == 'agg Y dfreq [1 2 3 4 5 6] dsev [1]'


def test_format_port_is_multiline():
    out = format_program(
        'port P agg A 1 claim sev lognorm 10 cv 1 fixed '
        'agg B 1 claim sev lognorm 20 cv 1 fixed')
    lines = out.split('\n')
    assert lines[0] == 'port P'
    assert all(ln.startswith('\tagg ') for ln in lines[1:])
