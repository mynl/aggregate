"""Recipes: the notes-driven describe / test / audit surface.

A library entry's ``doc{{{...}}}`` trailer carries a **recipe** -- markdown on
the *Python Cookbook* (Beazley & Jones) rhythm, plus a check:

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

This module turns that text into a :class:`Recipe`, which can be *rendered*
(the cookbook page), *run* (the pytest harness), and *audited*
(:attr:`aggregate.Underwriter.recipes`). One source, three consumers.

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

import re
from dataclasses import dataclass, field

__all__ = ['Recipe', 'parse_doc', 'SECTIONS']

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


@dataclass
class Recipe:
    """One library entry's documentation, parsed into its sections.

    Attributes
    ----------
    name, kind : str
        Identity of the entry this recipe documents ('' when parsed from bare
        text via :func:`parse_doc`).
    note : str
        The entry's one-line ``note{...}`` abstract.
    tags : tuple of str
        The entry's ``tags{...}`` slugs.
    problem, solution, discussion, check : str
        Section bodies, markdown, '' when the section is absent.
    extra : str
        Anything outside the four recognised sections, in document order.
    """

    name: str = ''
    kind: str = ''
    note: str = ''
    tags: tuple = ()
    problem: str = ''
    solution: str = ''
    discussion: str = ''
    check: str = ''
    extra: str = ''
    #: The entry's DecL program, for beat 1 of the page.
    program: str = ''
    _sections_present: tuple = field(default_factory=tuple)

    # ------------------------------------------------------------------
    # Derived views
    # ------------------------------------------------------------------
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
        return self._sections_present

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
        """Reassemble the recipe as markdown, canonical sections in order.

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
        got = '+'.join(self.sections) or 'empty'
        return (f'Recipe({self.name!r}, sections={got}, '
                f'asserts={self.n_asserts})')


def parse_doc(text, *, name='', kind='', note='', tags=(), program=''):
    """Parse a ``doc{{{...}}}`` body into a :class:`Recipe`.

    Splits on level-2 headings naming the canonical sections
    (:data:`SECTIONS`), case-insensitively. Text before the first recognised
    heading, and any section under an unrecognised heading, is collected into
    :attr:`Recipe.extra` so nothing is silently dropped.

    Parameters
    ----------
    text : str
        The doc body (markdown).
    name, kind, note, tags, program
        Identity and metadata from the owning knowledge entry, copied onto the
        recipe so a page can render beat 1 without a second lookup.

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
    text = (text or '').strip('\n')
    bodies = {s: [] for s in SECTIONS}
    extra = []

    # Split on EVERY level-2 heading, then classify. Splitting only on the
    # canonical four would silently glue a ``## References`` section onto
    # whichever section preceded it.
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

    # A repeated heading concatenates rather than overwrites -- losing the
    # second copy silently would be worse than showing both.
    merged = {s: '\n\n'.join(b).strip('\n') for s, b in bodies.items()}
    present = tuple(s for s in SECTIONS if merged[s].strip())

    return Recipe(
        name=name, kind=kind, note=note, tags=tuple(tags), program=program,
        problem=merged['problem'], solution=merged['solution'],
        discussion=merged['discussion'], check=merged['check'],
        extra='\n\n'.join(extra), _sections_present=present)
