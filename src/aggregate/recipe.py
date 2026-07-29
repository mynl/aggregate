"""Recipes: one library entry, described / tested / audited.

A **recipe** is a DecL entry: its identity (``kind``, ``name``), its parsed
``spec``, the source ``program`` it came from, its provenance, and -- once
the factory has run -- the constructed ``object``. :class:`Recipe` is what the
underwriter's *recipe base* stores, what
:meth:`aggregate.Underwriter.recipe` hands back, and what
:meth:`aggregate.Underwriter.build_many` returns one of per top-level output.

Most entries carry only a one-line ``note{...}`` abstract. The cookbook-worthy
few also carry a ``doc{{{...}}}`` trailer holding markdown on the *Python
Cookbook* (Beazley & Jones) rhythm, plus a check:

``## Problem``
    What you are trying to do, and why you would.
``## Solution``
    The code that does it. Written **self-contained** (it opens with its own
    ``build(...)``) so a reader can copy-paste it out of the page.
``## Discussion``
    Why it works, what to watch for.
``## Check``
    Pure ``assert``s -- the invariant, made visible. Runs in the same namespace
    the Solution left behind, so it can reach the objects that code built.

Those sections are parsed **lazily**, on first access: only a handful of the
shipped entries have a doc at all, and reading the libraries sits on the import
path for :data:`aggregate.build`.

One source, three consumers: a recipe can be *rendered* (the cookbook page),
*run* (the pytest harness), and *audited*
(:attr:`aggregate.Underwriter.recipes`).

Anything outside the four recognised headings is preserved in
:attr:`Recipe.extra` and rendered verbatim, so a recipe is free to add its own
sections without this parser needing to know about them.

.. warning::
   :meth:`Recipe.run` **executes the code in the doc**. For the shipped library
   that is the same trust level as the rest of the package, but a ``.agg`` file
   from elsewhere should be treated exactly like a Python file from elsewhere:
   read it before you run it. See ``dev/plan-meta-data.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

__all__ = ['Recipe', 'parse_doc', 'SECTIONS', 'DECL_PLACEHOLDER']

#: Placeholder a doc writes instead of retyping its own DecL declaration.
#:
#: A recipe's Solution needs to show the program it is about. Copying the
#: declaration into the doc would mean two copies of the same thing in the same
#: statement, guaranteed to drift the first time the program is edited. So the
#: doc writes::
#:
#:     ```python
#:     a = build('''<<decl>>''')
#:     ```
#:
#: and :attr:`Recipe.decl` -- the entry's own declaration -- is substituted in.
#:
#: **The substituted declaration carries ``hints{...}`` and nothing else.**
#: Hints change how the object builds, so a program without them would not
#: reproduce the recipe. The note, tags and doc are the surrounding page, so
#: repeating them in the program is redundant -- and dropping the doc is also
#: what stops it from quoting itself.
DECL_PLACEHOLDER = '<<decl>>'

#: The recognised section headings, in canonical order.
SECTIONS = ('problem', 'solution', 'discussion', 'check')

# Any level-2 heading. ``##[ \t]+`` requires whitespace after the hashes, so
# ``### Sub`` is not matched. Trailing punctuation is tolerated, so
# "## Solution" and "## solution:" agree.
_ANY_HEADING_RE = re.compile(r'^[ \t]*##[ \t]+(.+?)[ \t]*:?[ \t]*$', re.M)

# A fenced python block. ``python`` / ``py`` / bare fences all count as code;
# a fence with any other language tag (```text) is prose and is left alone.
_FENCE_RE = re.compile(
    r'^[ \t]*```[ \t]*(python|py|)[ \t]*\n(.*?)^[ \t]*```[ \t]*$',
    re.S | re.M)

# Any fenced block, used only to mask code before hunting for headings: a
# Python comment written ``## note`` at the start of a line inside a fence is
# code, not a section heading.
_ANY_FENCE_RE = re.compile(r'^[ \t]*```.*?^[ \t]*```[ \t]*$', re.S | re.M)


def _headings_outside_fences(text):
    """Yield level-2 heading matches that are not inside a fenced code block."""
    fences = [(m.start(), m.end()) for m in _ANY_FENCE_RE.finditer(text)]
    for m in _ANY_HEADING_RE.finditer(text):
        if not any(lo <= m.start() < hi for lo, hi in fences):
            yield m


def _code_of(section_text):
    """Concatenate every fenced python block in *section_text*.

    Multiple blocks in one section run as one script, in document order, so a
    Solution may narrate across several fences.
    """
    return '\n'.join(m.group(2) for m in _FENCE_RE.finditer(section_text))


def _split_sections(text, decl=''):
    """Split a doc body into ``({section: body}, extra)``.

    Splits on **every** level-2 heading, then classifies: splitting only on the
    canonical four would silently glue a ``## References`` section onto
    whichever section preceded it. A repeated heading concatenates rather than
    overwrites -- losing the second copy silently would be worse than showing
    both.

    ``decl``, when non-empty, is substituted for every
    :data:`DECL_PLACEHOLDER` **before** splitting, so the expansion works in any
    section (a Discussion may quote the declaration too).
    """
    text = (text or '').strip('\n')
    if decl and DECL_PLACEHOLDER in text:
        # ``decl`` is the doc-free rendering, so this cannot recurse.
        text = text.replace(DECL_PLACEHOLDER, decl)
    bodies = {s: [] for s in SECTIONS}
    extra = []

    matches = list(_headings_outside_fences(text))
    # Anything before the first heading is preamble.
    head = text[:matches[0].start()] if matches else text
    if head.strip():
        extra.append(head.strip())

    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title = m.group(1).strip().lower()
        body = text[m.end():end].strip('\n')
        if title in bodies:
            bodies[title].append(body)
        else:
            # Preserved verbatim, heading included, so a recipe may add its own
            # sections without this parser needing to know about them.
            extra.append(f'{text[m.start():m.end()].strip()}\n\n{body}'.strip())

    merged = {s: '\n\n'.join(b).strip('\n') for s, b in bodies.items()}
    return merged, '\n\n'.join(extra)


@dataclass
class Recipe:
    """One DecL declaration: identity, spec, source, docs, built object.

    The single entry record. Stored in the underwriter's recipe base under
    ``(kind, name)``, returned by :meth:`aggregate.Underwriter.recipe` (or its
    subscript spelling ``uw[name]``), and produced one per top-level output by
    :meth:`aggregate.Underwriter.build_many`.

    Attributes
    ----------
    kind : str
        ``'agg'`` | ``'sev'`` | ``'port'`` | ``'distortion'`` | ``'bvagg'`` |
        ``'pnl'`` | ``'xpnl'`` | ``'expr'``.
    name : str
        The user-given name (e.g. ``'Dice'``, ``'MyBook'``).
    spec : dict
        Constructor kwargs from the parser, including the DecL trailer keys
        ``note`` / ``tags`` / ``hints`` / ``doc`` when written.
    program : str
        The DecL statement as the parser received it: one line, comments
        stripped, any ``doc{{{...}}}`` body base64-encoded. Verbatim relative
        to what was parsed, not to the ``.agg`` file. See :attr:`decl` for the
        canonical re-rendering.
    object : Any
        ``None`` after parsing; populated by
        :meth:`aggregate.Underwriter._factory` once the corresponding
        Aggregate / Severity / Portfolio / Distortion is built.
    source : pathlib.Path or str
        Provenance: the ``.agg`` file the entry was read from, or the sentinel
        ``'session'`` for an entry created by an in-session ``build(...)``
        call. Backs the ``source`` filter on
        :meth:`aggregate.Underwriter.to_agg`.

    Notes
    -----
    The documentation surface -- :attr:`note`, :attr:`tags`, :attr:`doc`, the
    four section properties, :attr:`decl` -- is **derived from** :attr:`spec`,
    not stored alongside it, so there is exactly one copy of every fact. The
    doc is parsed and the declaration re-rendered on first access and cached;
    :func:`dataclasses.replace` drops those caches, which is correct since a
    replaced recipe may have a different spec.
    """

    kind: str = ''
    name: str = ''
    spec: Any = field(default_factory=dict)
    program: str = ''
    object: Any = None
    source: Any = 'session'

    # Derived caches. init=False keeps them off __init__ and out of
    # dataclasses.replace(), which is what makes replace() re-derive.
    _sections: Any = field(default=None, init=False, repr=False, compare=False)
    _extra: Any = field(default=None, init=False, repr=False, compare=False)
    _decl: Any = field(default=None, init=False, repr=False, compare=False)

    # ------------------------------------------------------------------
    # The DecL trailer, read straight off the spec
    # ------------------------------------------------------------------
    @property
    def note(self):
        """The entry's one-line ``note{...}`` abstract ('' if none)."""
        return (self.spec or {}).get('note', '') or ''

    @property
    def tags(self):
        """The entry's ``tags{...}`` slugs as a tuple (empty if none)."""
        return tuple((self.spec or {}).get('tags', ()) or ())

    @property
    def hints(self):
        """The entry's ``hints{...}`` build settings, raw ('' if none)."""
        return (self.spec or {}).get('hints', '') or ''

    @property
    def doc(self):
        """The entry's ``doc{{{...}}}`` body, raw markdown ('' if none)."""
        return (self.spec or {}).get('doc', '') or ''

    @property
    def decl(self):
        """The entry's declaration, carrying ``hints{...}`` and nothing else.

        What :data:`DECL_PLACEHOLDER` expands to, and what a cookbook page
        shows as "the program". Canonical (re-rendered from the spec by
        :func:`aggregate.decl_writer.format_program`, the parser's inverse)
        rather than the verbatim :attr:`program`.

        ``hints`` survives because it **changes how the object builds** -- a
        copy-pasteable program without it would not reproduce the recipe. The
        rest of the trailer does not: inside a recipe the ``note`` is the
        Problem, the ``tags`` are the page it sits on, and the ``doc`` is the
        page itself, so repeating any of them in the program would be
        redundant, and repeating the doc would make it quote itself.

        ``''`` when the entry cannot be unparsed (a ``minimum`` / ``mixture``
        combinator distortion references its children by name, and those
        references are not retained on the spec).
        """
        if self._decl is None:
            self._decl = self._render_decl()
        return self._decl

    def _render_decl(self):
        """Re-render the declaration with hints only; '' if it cannot be unparsed."""
        from .decl_writer import format_program
        if not self.kind or not isinstance(self.spec, dict):
            return ''
        try:
            return format_program((self.kind, self.name, self.spec), fmt='text',
                                  trailer=('hints',))
        except Exception:
            # A recipe for a construct the writer cannot invert simply gets no
            # <<decl>> expansion rather than failing to load at all.
            return ''

    # ------------------------------------------------------------------
    # Doc sections -- parsed lazily on first access
    # ------------------------------------------------------------------
    def _parse(self):
        if self._sections is None:
            self._sections, self._extra = _split_sections(self.doc, self.decl)
        return self._sections

    @property
    def problem(self):
        """The ``## Problem`` body ('' when absent)."""
        return self._parse()['problem']

    @property
    def solution(self):
        """The ``## Solution`` body ('' when absent)."""
        return self._parse()['solution']

    @property
    def discussion(self):
        """The ``## Discussion`` body ('' when absent)."""
        return self._parse()['discussion']

    @property
    def check(self):
        """The ``## Check`` body ('' when absent)."""
        return self._parse()['check']

    @property
    def extra(self):
        """Anything outside the four recognised sections, in document order."""
        self._parse()
        return self._extra

    @property
    def solution_code(self):
        """The Solution section's fenced python, concatenated."""
        return _code_of(self.solution)

    @property
    def check_code(self):
        """The Check section's fenced python, concatenated."""
        return _code_of(self.check)

    @property
    def sections(self):
        """The canonical sections actually present, in canonical order."""
        parsed = self._parse()
        return tuple(s for s in SECTIONS if parsed[s].strip())

    @property
    def n_asserts(self):
        """Number of ``assert`` statements in the Check code.

        The audit metric for "is this entry actually checked?" -- a recipe with
        a Check section but no asserts documents an invariant without testing
        it, which is precisely what the audit is meant to surface.
        """
        return len(re.findall(r'(?m)^\s*assert\b', self.check_code))

    @property
    def is_runnable(self):
        """True when there is any code to execute."""
        return bool(self.solution_code or self.check_code)

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    def markdown(self):
        """Reassemble the doc as markdown, canonical sections in order.

        Round-trips a well-formed doc body; a doc whose sections were written
        out of order comes back in canonical order, which is the point.
        """
        parts = []
        for name in SECTIONS:
            body = getattr(self, name)
            if body:
                parts.append(f'## {name.capitalize()}\n\n{body.strip()}')
        if self.extra:
            parts.append(self.extra.strip())
        return '\n\n'.join(parts)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def namespace(self):
        """A fresh namespace seeded with the names a recipe may assume.

        Deliberately small: ``build`` and ``qd`` (the two verbs every recipe
        uses), ``np`` / ``pd`` for the checks, and ``aggregate`` itself for
        anything else. A recipe that needs more imports them itself, exactly as
        a reader copy-pasting the Solution would have to.
        """
        import numpy as np
        import pandas as pd

        import aggregate
        from aggregate import build, qd

        return {'__name__': '__recipe__', 'aggregate': aggregate,
                'build': build, 'qd': qd, 'np': np, 'pd': pd}

    def run(self, ns=None):
        """Execute the Solution then the Check, in one namespace.

        The two share a namespace so a Check can assert against the objects the
        Solution built -- that is what makes a recipe self-testing rather than
        merely self-describing.

        This is the **pytest** consumer of a doc (``tests/test_library_recipes.py``);
        the cookbook consumer is :mod:`aggregate.cookbook`, which emits the
        same code as native Quarto cells. Both read the same doc, which is what
        makes "the page and the test run the same program" true by construction
        rather than by discipline.

        Parameters
        ----------
        ns : dict, optional
            Namespace to execute in; a fresh :meth:`namespace` by default.
            Pass one in to pre-seed values or to inspect what the code left
            behind.

        Returns
        -------
        dict
            The namespace after execution.

        Raises
        ------
        Exception
            Whatever the code raises. An ``AssertionError`` means the recipe's
            own stated invariant does not hold.

        Warnings
        --------
        This executes code carried in a ``.agg`` file. Treat a third-party
        library exactly as you would a third-party Python module.
        """
        ns = self.namespace() if ns is None else ns
        for label, code in (('solution', self.solution_code),
                            ('check', self.check_code)):
            if not code:
                continue
            try:
                exec(compile(code, f'<recipe {self.name or "?"}:{label}>',
                             'exec'), ns)
            except Exception as exc:
                raise type(exc)(
                    f'{self.name or "recipe"}: {label} block failed -- {exc}'
                ).with_traceback(exc.__traceback__) from None
        return ns

    def __repr__(self):
        # Deliberately cheap: `doc` and `object` are dict/attribute lookups,
        # and the section list is only computed when there is a doc to parse.
        bits = [repr(self.kind), repr(self.name)]
        if self.doc:
            bits.append(f'doc={"+".join(self.sections) or "empty"}')
        if self.object is not None:
            bits.append(f'object={type(self.object).__name__}')
        return f'Recipe({", ".join(bits)})'


def parse_doc(text, *, name='', kind='', note='', tags=(), decl=''):
    """Build a :class:`Recipe` from a bare ``doc{{{...}}}`` body.

    The no-underwriter entry point: use it to parse doc text that is not (yet)
    a library entry. A recipe read from the recipe base -- via
    :meth:`aggregate.Underwriter.recipe` or ``build[name]`` -- already carries
    its doc on ``spec['doc']`` and parses it itself, so this function is not
    needed there.

    Parameters
    ----------
    text : str
        The doc body (markdown).
    name, kind, note, tags
        Identity and metadata to attach, as if they had come from the entry's
        own trailer.
    decl : str, optional
        A declaration to substitute for every :data:`DECL_PLACEHOLDER` in
        *text*. Default ``''`` leaves placeholders visible -- better a visible
        ``<<decl>>`` than a silently empty Solution.

    Returns
    -------
    Recipe

    Examples
    --------
    >>> r = parse_doc('## Problem\\n\\nWhy.\\n\\n## Check\\n\\n```python\\nassert 1\\n```')
    >>> r.sections
    ('problem', 'check')
    >>> r.n_asserts
    1
    """
    r = Recipe(kind=kind, name=name,
               spec={'note': note, 'tags': tuple(tags), 'doc': text or ''})
    # '' (not None) so `decl` is taken as given rather than re-rendered from
    # the synthetic spec, which holds no structure to render.
    r._decl = decl or ''
    return r
