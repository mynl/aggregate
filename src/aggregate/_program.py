"""``ProgramMixin`` -- the shared DecL round-trip surface.

Every object the DecL language can declare carries the text it was declared
with (``program``) and can render that text back in canonical form
(``pprogram`` / ``pprogram_html``). Before this mixin the render half was
**ten near-identical properties across five classes and five files** --
``Aggregate``, ``Portfolio``, ``Severity``, ``PnL``, ``BivariateAggregate`` --
each a three-line delegation to :func:`aggregate.decl_writer.format_program`.
``pprogram_html`` was byte-identical on ``Aggregate`` / ``Portfolio`` and again
on ``PnL`` / ``BivariateAggregate``; the only real variation was a redundant
``if self.program`` guard on ``Severity`` and whether the import was deferred.

Collapsing them here is what made the ``trailer`` axis affordable. A render
option that every host hard-codes in its own property body cannot grow a new
setting without an edit per class -- which is why ``pprogram`` could not omit
``note{...}`` / ``hints{...}`` until the copies were gone. One definition, one
place to add the next axis.

It also closed a hole, exactly as :class:`~aggregate._help.HelpMixin` did for
``Frequency`` and ``GridDistribution``:
:class:`~aggregate.spectral.Distortion` is DecL-creatable -- the writer has a
``'distortion'`` kind renderer and ``Underwriter`` stamps ``obj.program`` on it
-- yet it had no ``pprogram`` and never declared ``program`` at all.

Notes
-----
Like :class:`~aggregate._help.HelpMixin` and
:class:`~aggregate._labeled.LabeledMixin` this defines **no** ``__init__``, so
it stays transparent to each host's ``super()`` chain -- which matters because
the hosts range from ``object`` to ``scipy.stats.rv_continuous``. It needs no
initializer: ``program`` is a class-level default that each stamped host
shadows with a plain instance attribute, and ``PnL`` shadows with a property of
its own (falling through to ``engine.program``). Nothing here converts an
existing attribute into a property, so ``__dict__`` / ``spec`` / ``__reduce__``
behaviour is unchanged for every host.

:func:`~aggregate.decl_writer.format_program` is imported at module scope --
``decl_writer`` imports nothing from ``aggregate`` at module scope (its
``parser`` / ``underwriter`` / ``spectral`` imports are all deferred), so this
closes no cycle. It is aliased to ``_format_program`` because the mixin's own
method carries the same name: the two never actually collide (class scope is
not in a method's name-lookup chain) but reading the body should not require
knowing that.
"""

from .decl_writer import format_program as _format_program


class ProgramMixin:
    """Shared DecL round-trip surface: ``program`` / :meth:`format_program` /
    :attr:`pprogram` / :attr:`pprogram_html`.

    Hosts supply one thing -- the ``program`` text. Four of the six store it as
    a plain instance attribute stamped by ``build``; :class:`~aggregate._pnl.PnL`
    overrides it with a property that falls through to its engine; anything
    never stamped inherits the empty class-level default and renders ``''``.
    """

    #: The DecL source text this object was declared with, stamped by ``build``.
    #: A class-level default so a host that is never stamped still answers;
    #: stamped hosts shadow it with an instance attribute, and ``PnL`` shadows
    #: it with a property that falls through to ``engine.program``. Empty for an
    #: object built programmatically, and for an inline ``sev`` clause (whose
    #: enclosing ``Aggregate`` owns the text).
    program = ''

    def format_program(self, *, fmt='text', layout='spread', trailer=True):
        """Render :attr:`program` in canonical form, with the render axes exposed.

        The object-bound twin of
        :func:`aggregate.decl_writer.format_program`, which it forwards
        ``self.program`` to; :attr:`pprogram` and :attr:`pprogram_html` are this
        method at fixed defaults. Canonical rather than verbatim --- the text is
        re-parsed and rendered from the resulting spec, so equivalent
        declarations share one form. Returns ``''`` when there is no program.

        Parameters
        ----------
        fmt : {'text', 'html', 'ansi', 'latex'}, default 'text'
            Output markup. ``text`` is plain; the others colorize via Pygments.
        layout : {'spread', 'terse'}, default 'spread'
            Line layout. ``spread`` puts each clause on its own two-space
            indented line; ``terse`` is the single-line-per-statement form.
        trailer : bool, default True
            Emit the ``note{...}`` / ``hints{...}`` trailer. ``False`` gives the
            bare declaration --- the form to print in a paper, a docstring or an
            exhibit. Applies through the whole tree, so a portfolio's units and
            a bivariate's components lose theirs too.

        Returns
        -------
        str
            Canonical DecL for the requested axes, or ``''``.

        Notes
        -----
        ``fmt`` and ``layout`` are round-trip safe; ``trailer=False`` is not ---
        its output re-parses to the same spec with ``note`` and ``hints``
        blanked. The semantic ``!`` markers (unconditional severity, the
        zero-modified mean pin, defective ``dwait``) are clause syntax rather
        than trailer and are never suppressed.

        Examples
        --------
        >>> from aggregate import build
        >>> a = build('agg Doc 10 claims sev lognorm 50 cv 1 poisson '
        ...           'note{a stored note}')
        >>> print(a.format_program(layout='terse'))
        agg Doc 10 claims sev lognorm 50 cv 1 poisson note{a stored note}
        >>> print(a.format_program(layout='terse', trailer=False))
        agg Doc 10 claims sev lognorm 50 cv 1 poisson
        """
        if not self.program:
            return ''
        return _format_program(self.program, fmt=fmt, layout=layout,
                               trailer=trailer)

    @property
    def pprogram(self):
        """Canonical DecL program text, rendered from the parsed spec.

        Derived by re-parsing :attr:`program` and rendering it back through
        :func:`aggregate.decl_writer.format_program` (the inverse of the
        parser), so it is *canonical* rather than verbatim --- equivalent
        declarations share one form. Rendered in the default ``spread`` layout
        (each clause on its own two-space-indented line).

        For the raw input as supplied to ``build`` use :attr:`program`; for any
        other layout, markup or the ``note`` / ``hints``-free form use
        :meth:`format_program`. An object built programmatically --- and an
        inline ``sev`` clause, whose enclosing ``Aggregate`` owns the text ---
        returns ``''``.
        """
        return self.format_program(fmt='text')

    @property
    def pprogram_html(self):
        """Syntax-highlighted DecL program for IPython / Jupyter display."""
        return self.format_program(fmt='html')
