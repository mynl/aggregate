"""Rewrite ``agg/library.agg`` in the canonical ``spread`` layout.

One clause per line at a two-space indent, a ``;`` terminating each statement
and a blank line between statements. That is the layout ``Recipe.decl`` renders
and every generated cookbook page already shows, so after this run the source
stops being the one place the library is still written dense (median statement
length before: 153 characters, 40 statements over 200).

Run from the repo root::

    python dev/reflow_library.py            # rewrite in place
    python dev/reflow_library.py --check    # report, write nothing

How it works
------------
Comment runs and blank lines pass through verbatim, so the file's header and
every section divider survive. Each statement is parsed, re-rendered by
:func:`aggregate.decl_writer.format_program`, and the result is re-parsed and
compared against the original spec. A statement is **held back** in its source
form when the writer cannot invert it:

* a named object reference (``sev.UnitSeverity``) is resolved and inlined at
  parse time, with nothing on the spec recording that a reference was written;
* the ``tweedie`` clause expands to its compound-Poisson-gamma equivalent;
* ``minimum`` / ``mixture`` distortion combinators drop their child names, so
  rendering raises.

A statement is also held back when the canonical form is **more ambiguous** than
the source. The writer renders ``ssev 100 - lognorm 80 cv .2`` in the general
affine form ``ssev -1 * lognorm 80 cv 0.2 + 100``, and a leading ``-1 *`` parses
two ways (``scaled(-1, X)`` or ``negate(scaled(1, X))``). The two are
algebraically identical, so nothing computes differently, but the shipped
library should not grow ambiguous statements for a cosmetic gain, and the
compact spelling is what entries named ``PremiumMinusLoss`` exist to show.
``tests/test_grammar_ambiguity.py`` is what would otherwise catch this.

See ``[Unparser-Reference-Gaps]`` in ``dev/TODO.md``. Held-back statements are
reflowed by hand afterwards, in the same style.

Correctness is not a matter of choosing break points well: whitespace between
tokens is insignificant to the lexer, so the only real check is that every
statement still parses to the same ``(kind, name, spec)``. That is asserted for
the whole file before anything is written.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

from aggregate import Underwriter
from aggregate.decl_writer import format_program

LIBRARY = Path(__file__).resolve().parents[1] / 'src' / 'aggregate' / 'agg' / 'library.agg'

#: Constructs :mod:`aggregate.decl_writer` cannot render back to what was
#: written. Matched against the *source* statement, because the spec no longer
#: carries the evidence.
NON_INVERTIBLE = re.compile(
    r'\b(?:sev|dist|distortion|agg|port)\.[A-Za-z]'   # named object reference
    r'|\btweedie\b'                                   # tweedie clause
    r'|\b(?:minimum|mixture)\s+dist'                  # distortion combinator
)


def ambiguity_count(statement):
    """How many ambiguous spans a statement parses with.

    Uses a second Lark built with ``ambiguity='explicit'``, which keeps the
    whole parse forest and marks each ambiguous span with an ``_ambig`` node,
    rather than the shipped parser's ``ambiguity='resolve'``, which silently
    picks one. Same instrument ``tests/test_grammar_ambiguity.py`` uses.
    """
    from lark import Lark

    from aggregate.parser import GRAMMAR_FILE

    global _EXPLICIT_PARSER
    if _EXPLICIT_PARSER is None:
        _EXPLICIT_PARSER = Lark.open(str(GRAMMAR_FILE), start='answer',
                                     parser='earley', ambiguity='explicit',
                                     lexer='dynamic', maybe_placeholders=True)
    tree = _EXPLICIT_PARSER.parse(statement)
    return sum(1 for st in tree.iter_subtrees() if st.data == '_ambig')


_EXPLICIT_PARSER = None


def split_chunks(text):
    """Split raw file text into ``(kind, lines)`` chunks in file order.

    ``kind`` is ``'verbatim'`` for a comment or blank line and ``'statement'``
    for everything else. A statement runs until a line ending in ``;`` that is
    not inside a ``doc{{{`` block, or until the blank line that terminates it.
    """
    chunks, current, in_doc = [], [], False
    for line in text.split('\n'):
        stripped = line.strip()
        if not current and not in_doc and (not stripped or stripped.startswith('#')):
            if chunks and chunks[-1][0] == 'verbatim':
                chunks[-1][1].append(line)
            else:
                chunks.append(['verbatim', [line]])
            continue
        current.append(line)
        if stripped.startswith('doc{{{'):
            in_doc = True
            continue
        if in_doc:
            if stripped.rstrip(';') == '}}}':
                in_doc = False
                if stripped.endswith(';'):
                    chunks.append(['statement', current])
                    current = []
            continue
        if stripped.endswith(';'):
            chunks.append(['statement', current])
            current = []
    if current:
        chunks.append(['statement', current])
    return chunks


def normalize(value):
    """Reduce a spec value to something comparable with ``==``.

    numpy scalars and arrays become plain Python. A ``Copula`` or a
    ``Distortion`` becomes its repr: neither defines ``__eq__``, so parsing one
    program twice yields two objects that compare unequal even though they say
    the same thing. Both reach a spec, a copula on ``bivariate`` and a list of
    child distortions on a ``minimum`` / ``mixture`` combinator.
    """
    from aggregate.copula import Copula
    from aggregate.spectral import Distortion

    if isinstance(value, (Copula, Distortion)):
        return repr(value)
    if isinstance(value, np.generic):
        return normalize(value.item())
    if isinstance(value, np.ndarray):
        return tuple(normalize(v) for v in value.tolist())
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(normalize(v) for v in value)
    return value


#: Keys the writer emits explicitly that the parser leaves absent when they hold
#: their default. Comparing them would report a difference where there is none.
DEFAULTS = {'sev_lb': 0.0, 'sev_ub': float('inf'), 'sev_wt': 1.0}


def comparable(spec):
    """Normalized spec with default-valued optional keys dropped."""
    spec = normalize(spec)
    return {k: v for k, v in spec.items()
            if not (k in DEFAULTS and v == DEFAULTS[k])}


def main(check=False):
    uw = Underwriter(databases='library')
    uw.load()
    text = LIBRARY.read_text(encoding='utf-8')

    out, canonical, held = [], [], []
    for kind, lines in split_chunks(text):
        if kind == 'verbatim':
            out.extend(lines)
            continue
        source = '\n'.join(lines)
        statements = uw.lexer.preprocess(source)
        assert len(statements) == 1, f'{len(statements)} statements in:\n{source}'
        statement = statements[0]
        parsed = uw.parser.parse(uw.lexer.tokenize(statement))

        rendered = None
        if not NON_INVERTIBLE.search(statement):
            try:
                candidate = format_program(parsed, fmt='text', layout='spread',
                                           trailer=True)
                flat = uw.lexer.preprocess(candidate)[0]
                round_trip = uw.parser.parse(uw.lexer.tokenize(flat))
                if (comparable(round_trip[2]) == comparable(parsed[2])
                        and ambiguity_count(flat) <= ambiguity_count(statement)):
                    rendered = candidate
            except Exception as exc:                       # noqa: BLE001
                print(f'  render failed  {parsed[1]}: {str(exc)[:70]}')

        if rendered is None:
            held.append(parsed[1])
            out.extend(lines)
        else:
            canonical.append(parsed[1])
            out.extend((rendered + ';').split('\n'))
        out.append('')

    # Collapse the blank line appended after a statement into whatever blank
    # lines the source already had, so section dividers keep their spacing.
    result = re.sub(r'\n{3,}', '\n\n', '\n'.join(out))
    if not result.endswith('\n'):
        result += '\n'

    print(f'{len(canonical)} canonical, {len(held)} held back: {", ".join(held)}')

    # Verify before writing: every statement must parse to the same triple.
    before = [uw.parser.parse(uw.lexer.tokenize(s))
              for s in uw.lexer.preprocess(text)]
    after = [uw.parser.parse(uw.lexer.tokenize(s))
             for s in uw.lexer.preprocess(result)]
    assert len(before) == len(after), f'{len(before)} statements became {len(after)}'
    for (k0, n0, s0), (k1, n1, s1) in zip(before, after):
        assert (k0, n0) == (k1, n1), f'{n0} became {n1}'
        assert comparable(s0) == comparable(s1), f'spec moved: {n0}'
    print(f'verified: {len(after)} statements, specs identical')

    if check:
        print('--check: nothing written')
    else:
        LIBRARY.write_text(result, encoding='utf-8')
        print(f'wrote {LIBRARY}')


if __name__ == '__main__':
    main(check='--check' in sys.argv)
