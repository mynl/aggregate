"""Recipes: one library entry, as a record.

A **recipe** is a DecL entry: its identity (``kind``, ``name``), its parsed
``spec``, the source ``program`` it came from, its provenance, and -- once
the factory has run -- the constructed ``object``. :class:`Recipe` is what the
underwriter's *recipe base* stores, what
:meth:`aggregate.Underwriter.recipe` hands back, and what
:meth:`aggregate.Underwriter.build_many` returns one of per top-level output.

An entry describes itself with its DecL trailer: a one-line ``note{...}``
abstract, ``tags{...}`` slugs for grouping and selection, and ``hints{...}``
build settings. Each is a fact about the entry, and all three are read
straight off the spec so there is exactly one copy of every fact.

Long-form write-ups are not facts about an entry. A ``doc{{{...}}}`` clause
carried them here, in markdown with executable fenced code, until 1.0.0a301,
when it was retired along with the Quarto cookbook that rendered it
(``dev/done/plan-decommission-docs.md``): it mixed a book into a declaration
language, the Python inside it was invisible to ruff, to editors and to
tracebacks, and the DecL stable-tier promise would have frozen the whole
arrangement for the life of the major version. The write-ups became notes in
the presentations repository and the invariants they asserted became
``tests/test_library_entries.py``.

What survives is this record plus :attr:`Recipe.decl`, the canonical
re-rendering of the declaration, which is the only part of the surface a
downstream consumer reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ['Recipe']


@dataclass
class Recipe:
    """One DecL declaration: identity, spec, source, built object.

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
        ``note`` / ``tags`` / ``hints`` when written.
    program : str
        The DecL statement as the parser received it: one line, comments
        stripped. Verbatim relative to what was parsed, not to the ``.agg``
        file. See :attr:`decl` for the canonical re-rendering.
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
    The description surface -- :attr:`note`, :attr:`tags`, :attr:`hints`,
    :attr:`decl` -- is **derived from** :attr:`spec`, not stored alongside it,
    so there is exactly one copy of every fact. The declaration is re-rendered
    on first access and cached; :func:`dataclasses.replace` drops that cache,
    which is correct since a replaced recipe may have a different spec.
    """

    kind: str = ''
    name: str = ''
    spec: Any = field(default_factory=dict)
    program: str = ''
    object: Any = None
    source: Any = 'session'

    # Derived cache. init=False keeps it off __init__ and out of
    # dataclasses.replace(), which is what makes replace() re-derive.
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
    def decl(self):
        """The entry's declaration, carrying ``hints{...}`` and nothing else.

        Canonical: re-rendered from the spec by
        :func:`aggregate.decl_writer.format_program`, the parser's inverse,
        rather than the verbatim :attr:`program`. This is the form to show a
        reader, and the form a note elsewhere should print rather than retype,
        since it reads the live entry and so cannot go stale.

        ``hints`` survives because it **changes how the object builds** -- a
        copy-pasteable program without it would not reproduce the entry. The
        rest of the trailer does not: the ``note`` describes the entry and the
        ``tags`` file it, and neither belongs inside the program.

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
            # An entry for a construct the writer cannot invert simply gets no
            # canonical rendering rather than failing to load at all.
            return ''

    def __repr__(self):
        # Deliberately cheap: every bit is a dict or attribute lookup.
        bits = [repr(self.kind), repr(self.name)]
        if self.object is not None:
            bits.append(f'object={type(self.object).__name__}')
        return f'Recipe({", ".join(bits)})'
