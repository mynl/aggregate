"""Rewrite ``agg/library.agg`` in the canonical ``spread`` layout.

One clause per line at a two-space indent, a ``;`` terminating each statement
and a blank line between statements. That is the layout ``Recipe.decl`` renders
and every generated cookbook page already shows, so the source is not the one
place the library is still written dense (median statement length before the
first run: 153 characters, 40 statements over 200).

Run from the repo root::

    python dev/done/reflow_library.py            # rewrite in place
    python dev/done/reflow_library.py --check    # report, write nothing

**Filed under ``dev/done/`` but not retired.** What is closed is the one-shot
``[Library-Canonical-Layout]`` migration this was written for; the script itself
is the library's formatter and stays runnable. Re-run it whenever entries are
added or edited, and whenever an unparser gap closes, since a construct that
starts inverting can leave the held-back set and be laid out by machine (three
Tweedie entries did exactly that at 1.0.0a231). ``--check`` is the read-only
form, and ``tests/test_agg_libraries.py`` fails if the shipped file drifts from
what this renders, layout included: ``test_library_reflow_is_a_no_op`` runs
:func:`reflow` over the file and requires it unchanged. Before 2026-10-28 the
test compared whitespace-normalized content only, so eight entries lost their
``;`` and the script, unable to split them, went unrun for two months unnoticed.

How it works
------------
Comment runs and blank lines pass through verbatim, so the file's header and
every section divider survive. Each statement is parsed, re-rendered by
:func:`aggregate.decl_writer.format_program`, and the result is re-parsed and
compared against the original spec. A statement is **held back** in its source
form when the writer cannot invert it:

* a named object reference (``sev.UnitSeverity``) is resolved and inlined at
  parse time, with nothing on the spec recording that a reference was written;
* ``minimum`` / ``mixture`` distortion combinators drop their child names, so
  rendering raises.

The ``tweedie`` clause was a third case until 1.0.0a231, when it gained the
``_tweedie`` provenance key and started inverting like everything else. A
``sev agg.NAME`` severity reference looks like the first case and is not: since
1.0.0a293 the spec records the reference rather than the object, so the writer
renders it and those statements are laid out by machine like any other.

A statement is also held back when the canonical form is **more ambiguous** than
the source. The writer renders ``ssev 100 - lognorm 80 cv .2`` in the general
affine form ``ssev -1 * lognorm 80 cv 0.2 + 100``, and a leading ``-1 *`` parses
two ways (``scaled(-1, X)`` or ``negate(scaled(1, X))``). The two are
algebraically identical, so nothing computes differently, but the shipped
library should not grow ambiguous statements for a cosmetic gain, and the
compact spelling is what entries named ``PremiumMinusLoss`` exist to show.
``tests/test_grammar_ambiguity.py`` is what would otherwise catch this.

A statement is held back, finally, when its name is in ``UNPARSER_EXEMPT`` in
``tests/test_agg_libraries.py``: an entry whose authored spelling (``[1:6]``,
``2/3``, arithmetic, digit separators) is the point of the entry and renders to
a spec-identical but different canonical form. See :func:`unparser_exempt`.

See ``[Unparser-Reference-Gaps]`` in ``dev/TODO.md``. Held-back statements are
reflowed by hand afterwards, in the same style.

Correctness is not a matter of choosing break points well: whitespace between
tokens is insignificant to the lexer, so the only real check is that every
statement still parses to the same ``(kind, name, spec)``. That is asserted for
the whole file before anything is written.
"""
from __future__ import annotations

import ast
import difflib
import re
import sys
from pathlib import Path

import numpy as np

from aggregate import Underwriter
from aggregate.decl_writer import format_program

def _repo_root() -> Path:
    """The repo root, found by walking up to the directory holding ``pyproject.toml``.

    Was ``parents[1]``, a fixed hop that silently assumed this file sat directly
    in ``dev/``. Moving it to ``dev/done/`` made it resolve to ``dev/src/...``
    and the script died on a ``FileNotFoundError``. Searching for the marker
    costs nothing and does not care how deep the script is filed.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / 'pyproject.toml').exists():
            return parent
    raise RuntimeError(
        f'reflow_library: no pyproject.toml above {Path(__file__).resolve()}; '
        f'run this from inside the repo.')


LIBRARY = _repo_root() / 'src' / 'aggregate' / 'agg' / 'library.agg'

#: A severity reference, ``sev agg.NAME`` / ``ssev port.NAME`` (with an optional
#: trailing ``!``). Stripped from the statement *before* the search below,
#: because it is the one dotted reference the writer CAN render: since
#: 1.0.0a293 the spec records the reference instead of inlining the object.
#: Removing it here rather than complicating the pattern keeps the bare
#: ``agg.`` / ``port.`` alternative catching an ENGINE reference
#: (``less agg.USXOLTower``), which still cannot be inverted.
SEV_REFERENCE = re.compile(
    r'\bs?sev\s+(?:agg|port)\.[A-Za-z][\w.:~\-]*\s*!?')

#: Constructs :mod:`aggregate.decl_writer` cannot render back to what was
#: written. Matched against the *source* statement, because the spec no longer
#: carries the evidence.
NON_INVERTIBLE = re.compile(
    r'\b(?:sev|dist|distortion|agg|port)\.[A-Za-z]'   # named object reference
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
        if current and not in_doc and not stripped:
            # A statement missing its ``;`` ends at the blank line, as it does
            # for the library loader. Without this it ran on into the next
            # statement, and an unterminated entry stopped the script cold.
            chunks.append(['statement', current])
            current = []
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


def unparser_exempt() -> frozenset:
    """The names whose authored spelling the library keeps.

    ``UNPARSER_EXEMPT`` in ``tests/test_agg_libraries.py`` is the one list, so
    this reads it from there rather than keeping a copy that would drift. Read
    with :func:`ast.literal_eval` rather than imported, so loading the set runs
    none of the test module.

    Notes
    -----
    The script's own hold-back rules (``NON_INVERTIBLE`` and the ambiguity
    count) are not enough without it. An entry like ``dsev [1:6]``,
    ``ph 2/3`` or ``10000 exposure at 0.125 rate`` renders to a spec-identical
    canonical form, so it passes the round trip and was rewritten to
    ``dsev [1 2 3 4 5 6]``, ``ph 0.6666666666666666`` or
    ``10000 premium at 0.125 lr``, discarding the spelling the entry exists to
    show.
    """
    path = _repo_root() / 'tests' / 'test_agg_libraries.py'
    for node in ast.parse(path.read_text(encoding='utf-8')).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and getattr(node.targets[0], 'id', None) == 'UNPARSER_EXEMPT'):
            return frozenset(ast.literal_eval(node.value))
    raise RuntimeError(f'reflow_library: no UNPARSER_EXEMPT in {path}')


def reflow(text, exempt=frozenset(), uw=None):
    """The library text in canonical layout, with the names reflowed and held back.

    Parameters
    ----------
    text : str
        The library file's text.
    exempt : collection of str
        Statement names kept exactly as written (see :func:`unparser_exempt`).
    uw : Underwriter, optional
        A loaded library underwriter; one is built when omitted.

    Returns
    -------
    result : str
        The reflowed text. Equal to ``text`` when the file is already canonical.
    canonical, held : list of str
        Names laid out by machine, and names left in their source form.
    """
    if uw is None:
        uw = Underwriter(databases='library')
        uw.load()

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
        if parsed[1] in exempt:
            pass
        elif not NON_INVERTIBLE.search(SEV_REFERENCE.sub('', statement)):
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
            # Held back keeps the spelling, not a missing terminator.
            last = lines[-1].rstrip()
            if not last.endswith(';') and not last.lstrip().startswith('#'):
                lines = [*lines[:-1], last + ';']
            out.extend(lines)
        else:
            canonical.append(parsed[1])
            out.extend((rendered + ';').split('\n'))
        out.append('')

    # Collapse the blank line appended after a statement into whatever blank
    # lines the source already had, so section dividers keep their spacing.
    result = re.sub(r'\n{3,}', '\n\n', '\n'.join(out)).rstrip('\n') + '\n'
    return result, canonical, held


def main(check=False):
    uw = Underwriter(databases='library')
    uw.load()
    text = LIBRARY.read_text(encoding='utf-8')
    result, canonical, held = reflow(text, unparser_exempt(), uw)

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

    if result == text:
        print('already canonical: nothing to write')
    elif check:
        moved = sum(1 for line in difflib.unified_diff(
                        text.split('\n'), result.split('\n'), lineterm='', n=0)
                    if line[:1] in '+-' and line[:3] not in ('+++', '---'))
        print(f'--check: not canonical, {moved} diff lines; nothing written')
    else:
        LIBRARY.write_text(result, encoding='utf-8')
        print(f'wrote {LIBRARY}')


if __name__ == '__main__':
    main(check='--check' in sys.argv)
