"""Round-trip tests for the DecL unparser (``aggregate.decl_writer``, 1.0.0a53).

The unparser is the structural inverse of the parser. Its contract (see
``dev/done/plan-decl-unparser.md`` and the ``decl_writer`` module docstring) is
**idempotence one step removed**: with ``f = spec_to_decl``,
``f(f(f(x))) == f(x)``. Concretely, for every program in the reference corpus
(``_test_suite.agg`` + ``_test_suite2.agg`` + ``decl-testers.agg``):

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
_CORPUS_FILES = ['_test_suite.agg', '_test_suite2.agg', 'decl-testers.agg']

# Genuinely lossy / non-canonical-spec constructs: idempotence holds but the
# first parse's spec is not a fixed point, so fidelity is exempt. Empty since
# 1.0.0a231, when the tweedie clause (its only member: it discarded its note and
# baked a CP-gamma spec that bypassed the sev_weighted defaults) started
# round-tripping through the ``_tweedie`` provenance key.
_FIDELITY_EXEMPT: set[str] = set()

# The corpus programs reference builtins the corpus itself defines (``sev.One``
# from ``_test_suite.agg``, the ``agg.ASV.*`` severity references in
# ``decl-testers.agg``), so parse against an underwriter that carries the WHOLE
# corpus, not the module-level ``build`` singleton -- ``build`` defaults to the
# curated ``library``, which has none of them.
#
# Preloaded line by line rather than through ``databases=`` because
# ``decl-testers.agg`` deliberately does not load clean: its section X is
# intentional parse-error fixtures. Failures are swallowed here exactly as in
# ``conftest.underwriter``, so they surface as the individual test failures they
# are meant to be. File order is definition-before-use, and the three files have
# no ``(kind, name)`` collisions between them, so nothing shadows anything.
# Preloaded LAZILY, on first use: parsing the corpus costs about 11 seconds, and
# an import-time preload pays that in every xdist worker whether or not the
# worker got any of these tests. ``Underwriter(databases=…)`` is lazy for the
# same reason, but cannot be used here: ``decl-testers.agg`` deliberately does
# not load clean (its section X is intentional parse-error fixtures) and
# ``load`` aborts on the first error. Failures are swallowed exactly as in
# ``conftest.underwriter``, so they surface as the individual test failures they
# are meant to be. File order is definition-before-use, and the three files have
# no ``(kind, name)`` collisions between them, so nothing shadows anything.
_uw = Underwriter()
_uw_ready = False


def _corpus_underwriter():
    global _uw_ready
    if not _uw_ready:
        for fn in _CORPUS_FILES:
            path = _AGG_DIR / fn
            for line in UnderwritingLexer.preprocess(path.read_text(encoding='utf-8')):
                try:
                    kind, name, spec = _uw.parser.parse(_uw.lexer.tokenize(line))
                except Exception:
                    continue
                if kind != 'expr':
                    _uw.add_recipe(kind, name, spec, line)
        _uw_ready = True
    return _uw


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
    return _corpus_underwriter().parser.parse(statements[0])


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
    kind0, name0, spec0 = _corpus_underwriter().parser.parse(program)
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

_SMOKE = 'agg X 10 claims sev lognorm 50 cv 0.8 occurrence net of 50% po 10 xs 0 poisson note{hi}'


def test_format_program_has_no_width_parameter():
    """a227: ``width`` was accepted and ignored, a documented lie.

    ``layout`` is structural, one clause per line, not width driven, and
    nothing ever needed the parameter. A signature that takes an argument and
    discards it is worse than one that does not take it.
    """
    import inspect

    assert 'width' not in inspect.signature(format_program).parameters
    with pytest.raises(TypeError):
        format_program(_SMOKE, width=40)


def test_format_text_is_plain():
    # default layout is now 'spread': multiline, each clause on its own line
    out = format_program(_SMOKE, fmt='text')
    assert isinstance(out, str)
    assert '\x1b[' not in out and '<span' not in out
    assert out.startswith('agg X')
    assert '\n  sev ' in out             # severity clause on its own indented line


def test_format_text_terse_is_single_line():
    out = format_program(_SMOKE, fmt='text', layout='terse')
    assert '\n' not in out
    assert out.startswith('agg X 10 claims sev ')


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
    # terse reproduces the historical single-line form byte-for-byte
    assert (format_program(parsed, fmt='text', layout='terse')
            == 'agg Y dfreq [1 2 3 4 5 6] dsev [1]')
    # spread heads with `agg Y` and indents the two clauses
    spread = format_program(parsed, fmt='text')
    lines = spread.split('\n')
    assert lines[0] == 'agg Y'
    assert lines[1] == '  dfreq [1 2 3 4 5 6]'
    assert lines[2] == '  dsev [1]'


def test_format_port_is_multiline():
    prog = ('port P agg A 1 claim sev lognorm 10 cv 1 fixed '
            'agg B 1 claim sev lognorm 20 cv 1 fixed')
    # spread: two-space-indented unit heads, their clauses one level deeper
    spread = format_program(prog).split('\n')
    assert spread[0] == 'port P'
    assert '  agg A' in spread
    assert '  agg B' in spread
    assert '    sev lognorm 10 cv 1' in spread
    # terse: the historical tab-indented one-line-per-unit form
    terse = format_program(prog, layout='terse').split('\n')
    assert terse[0] == 'port P'
    assert all(ln.startswith('\tagg ') for ln in terse[1:])


def test_format_reins_cessions_indent():
    prog = ('agg R 10 claims sev lognorm 50 cv 0.8 occurrence net of '
            '75% po 100 xs 200 and 50% po 100 xs 300 poisson')
    lines = format_program(prog).split('\n')
    assert '  occurrence net of' in lines
    i = lines.index('  occurrence net of')
    # cessions indented one level past the clause keyword; first ends with ' and'
    assert lines[i + 1] == '    75% po 100 xs 200 and'
    assert lines[i + 2] == '    50% po 100 xs 300'


def test_format_picks_gets_its_own_line():
    """[Format-Program-Picks-Line]: picks is a block child, not an inline tail.

    The severity clause with picks was the one clause the spread layout left
    running off the page. It is now a ``_Block``: the distribution heads the
    clause and the picks fragment is its child, one level deeper.
    """
    prog = 'agg P dfreq [1] sev 100 * uniform picks [50 75 100] [30 12 5]'
    lines = format_program(prog).split('\n')
    assert '  sev 100 * uniform' in lines
    i = lines.index('  sev 100 * uniform')
    assert lines[i + 1] == '    picks [50 75 100] [30 12 5]'
    # terse space-joins the block back, byte for byte the historical form
    assert format_program(prog, layout='terse') == prog


def test_format_picks_carries_the_unconditional_bang_and_label():
    """The trailing ``!`` and the interior label ride on the last fragment.

    Both close the whole severity clause, so terse byte order is picks, then
    bang, then label; putting them anywhere else would change what re-parses.
    """
    bang = 'agg PB dfreq [1] sev 100 * uniform picks [50 75 100] [30 12 5] !'
    assert format_program(bang, layout='terse') == bang
    assert format_program(bang).split('\n')[-1] == \
        '    picks [50 75 100] [30 12 5] !'
    labeled = ('agg PL dfreq [1] sev 100 * uniform '
               'picks [50 75 100] [30 12 5] as Picked')
    assert format_program(labeled).split('\n')[-1] == \
        '    picks [50 75 100] [30 12 5] as Picked'


def test_format_picks_free_severity_keeps_the_bang_on_its_own_line():
    """No picks means no block: the ``!`` stays where it always was."""
    prog = 'agg PN 5 claims 1000 xs 0 sev lognorm 100 cv 2 ! poisson'
    assert '  sev lognorm 100 cv 2 !' in format_program(prog).split('\n')


def test_format_html_spread_preserves_newlines():
    out = format_program(_SMOKE, fmt='html')           # default spread
    assert '<span' in out and '\n' in out


def test_format_bad_layout_raises():
    with pytest.raises(ValueError):
        format_program(_SMOKE, layout='zigzag')


# ----------------------------------------------------------------------
# [DecL-Paren-Arithmetic] (1.0.0a268): expressions collapse to their value
# ----------------------------------------------------------------------
# `+`, `-` and `*` inside parentheses evaluate at parse time exactly as `/`,
# `**` and `exp` always have, so the writer never sees a formula. That is the
# "canonical, not verbatim" contract stated in the ``decl_writer`` module
# docstring, and it means `format_program` over user source replaces the
# formula with its evaluated literal. Formula-preserving reformatting would be
# a token-level text tool, not a spec change.

_PAREN_MATH_PROGRAMS = [
    'agg PaExposure (4 + 3*2) claims sev lognorm 100 cv 2 poisson',
    'agg PaPremium 5 claims (100_000/(1-.25)) premium sev lognorm 100 cv 2 poisson',
    'agg PaScale 10 claims sev (exp(-1 * .4**2/2)) * lognorm .4 poisson',
    'agg PaLayer 5 claims (500*2) xs (0+0) sev lognorm 100 cv 2 poisson',
    'agg PaReins 5 claims 1000 xs 0 sev lognorm 100 cv 2 '
    'occurrence net of (100*2) xs (50 + 50) poisson',
]


@pytest.mark.parametrize('program', _PAREN_MATH_PROGRAMS)
def test_paren_math_collapses_and_is_a_fixed_point(program):
    """Paren arithmetic decompiles to evaluated literals, and settles at once."""
    text1 = format_program(program)
    assert '(' not in text1.split('note{')[0], f'formula survived: {text1!r}'
    assert format_program(text1) == text1


def test_paren_math_exposure_renders_its_value():
    """The worked example from the plan: ``(4 + 3*2) claims`` reads ``10 claims``."""
    text = format_program(_PAREN_MATH_PROGRAMS[0])
    assert '10 claims' in text
