"""``ProgramMixin`` -- the shared DecL round-trip surface.

Every object the DecL language can declare carries the statement the *parser*
was handed (``program``) and can render that statement back in canonical form
(``pprogram`` / ``pprogram_html``). The two are not the same text and the gap
between them is the point: ``program`` is what went in, ``pprogram`` is what
the parser understood. Before this mixin the render half was
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

Derived programs
----------------
The mixin answers "what is this object", from the text it was declared with.
The module also answers the neighbouring question, "what is the program that
would build the object I arrived at", for the three ways of arriving that come
up often enough to deserve text: :func:`sharpen_program` (the grid moved),
:func:`pnl_program` (wrapped in a P&L) and :func:`reins_program` (a cession
added). See dev/done/plan-derived-programs.md.

These are functions here rather than mixin members because the mixin's six
hosts include :class:`~aggregate._severity.Severity`,
:class:`~aggregate.spectral.Distortion` and
:class:`~aggregate.bivariate.BivariateAggregate`, none of which can answer any
of the three: a member that raises on four of six hosts is a member in the
wrong place. :class:`~aggregate.distributions.Aggregate` and
:class:`~aggregate.portfolio.Portfolio` carry thin delegations, the pattern
``sharpen`` itself already follows.

All three work the same way: parse the stored program back to its spec, mutate
the spec, render the mutation through the writer. Never string surgery. A
cession clause cannot be appended, because an occurrence cession sits before
the frequency clause and an aggregate cession after it, and a trailer clause
cannot be appended either, because a spec holds one ``note`` and one ``hints``
and a second copy would silently win or lose. Both facts are grammar knowledge,
so the library holds them rather than every caller.

The renders ask for the trailer explicitly. :meth:`ProgramMixin.format_program`
defaults to ``trailer=False``, on the reasoning that formatting a program is
usually about the math rather than the metadata, and these three are the
exception: for :func:`sharpen_program` the trailer is the entire payload.
"""

import logging
import re

import numpy as np

from .decl_writer import format_program as _format_program

logger = logging.getLogger(__name__)


class ProgramMixin:
    """Shared DecL round-trip surface: ``program`` / :meth:`format_program` /
    :attr:`pprogram` / :attr:`pprogram_html`.

    Hosts supply one thing -- the ``program`` text. Four of the six store it as
    a plain instance attribute stamped by ``build``; :class:`~aggregate._pnl.PnL`
    overrides it with a property that falls through to its engine; anything
    never stamped inherits the empty class-level default and renders ``''``.
    """

    #: The DecL statement as the **parser received it**, stamped by ``build``.
    #:
    #: Not the text you typed. ``Underwriter._interpret_program`` stamps the
    #: output of :meth:`aggregate.parser.UnderwritingLexer.preprocess`, so four
    #: things have already happened to it:
    #:
    #: * it is **one line**, however the source was laid out, because a
    #:   statement may span as many physical lines as its author likes;
    #: * comments are gone, including full lines between clauses;
    #: * runs of whitespace survive in places as double spaces (the bracket
    #:   collapse step), so ``dfreq [3]`` can come back as ``dfreq  [3]``;
    #: * a ``doc{{{...}}}`` body is **URL-safe base64**, not markdown. That is
    #:   deliberate, not corruption: encoding it first (step 0) is what lets a
    #:   doc carry ``#`` headings, blank lines and fenced code through the
    #:   later steps. Read it back decoded via ``.doc``.
    #:
    #: Nobody keeps the keystrokes. For the file as written, read the ``.agg``.
    #:
    #: **One thing rewrites it.** :meth:`~aggregate.distributions.Aggregate.sharpen`
    #: moves the grid, and pins the grid it moved to onto this text through
    #: :func:`pin_sharpen`, so a probed object's program is the program that
    #: **builds** it rather than merely the one that built it. Everything else
    #: about the statement is untouched, the rewrite is the trailer only, and it
    #: is still preprocessor output, so the shape above still holds. Nothing
    #: else in the library writes here after ``build``.
    #:
    #: A class-level default so a host that is never stamped still answers;
    #: stamped hosts shadow it with an instance attribute, and ``PnL`` shadows
    #: it with a property that falls through to ``engine.program``. Empty for an
    #: object built programmatically, and for an inline ``sev`` clause (whose
    #: enclosing ``Aggregate`` owns the text).
    program = ''

    def format_program(self, *, fmt='text', layout='spread', trailer=False):
        """Render :attr:`program` in canonical form, with the render axes exposed.

        The object-bound twin of
        :func:`aggregate.decl_writer.format_program`, which it forwards
        ``self.program`` to; :attr:`pprogram` and :attr:`pprogram_html` are this
        method at fixed defaults. Canonical rather than as-received: the stored
        statement is re-parsed and rendered back from the resulting spec, so
        equivalent declarations share one form and any canonicalization the
        parser applied becomes visible. Returns ``''`` when there is no program.

        Parameters
        ----------
        fmt : {'text', 'html', 'ansi', 'latex'}, default 'text'
            Output markup. ``text`` is plain; the others colorize via Pygments.
        layout : {'spread', 'terse'}, default 'spread'
            Line layout. ``spread`` puts each clause on its own two-space
            indented line, the trailer clauses included; ``terse`` is the
            single-line-per-statement form.
        trailer : bool or iterable of str, default False
            Emit the ``note{...}`` / ``tags{...}`` / ``hints{...}`` /
            ``doc{{{...}}}`` trailer. The default ``False`` gives the bare
            declaration: formatting a program is almost always about the math
            and the insurance, not the metadata around it. ``True`` emits all
            four; an iterable names the ones to keep. Applies through the whole
            tree, so a portfolio's units and a bivariate's components follow.

        Returns
        -------
        str
            Canonical DecL for the requested axes, or ``''``.

        Notes
        -----
        ``fmt`` and ``layout`` are round-trip safe; ``trailer`` is not: the
        output re-parses to the same spec minus whatever was suppressed. The
        semantic ``!`` markers (unconditional severity, the zero-modified mean
        pin, defective ``dwait``) are clause syntax rather than trailer and are
        never suppressed.

        A ``doc{{{...}}}`` is emitted as readable markdown here even though
        :attr:`program` stores it base64-encoded: the writer renders from the
        spec, which holds the decoded body.

        Examples
        --------
        >>> from aggregate import build
        >>> a = build('agg Doc 10 claims sev lognorm 50 cv 1 poisson '
        ...           'note{a stored note}')
        >>> print(a.format_program(layout='terse'))
        agg Doc 10 claims sev lognorm 50 cv 1 poisson
        >>> print(a.format_program(layout='terse', trailer=True))
        agg Doc 10 claims sev lognorm 50 cv 1 poisson note{a stored note}
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
        parser), so equivalent declarations share one form. Rendered in the
        default ``spread`` layout (each clause on its own two-space-indented
        line) and **without the trailer**: this is the declaration as a reader
        wants to see it, the math and the insurance, not the ``note`` /
        ``tags`` / ``hints`` / ``doc`` metadata around it.

        **Which of the two to reach for.** :attr:`program` is what the parser
        was *handed*; this is what it *understood*. Use ``program`` when you
        need text to feed back to ``build``, and ``pprogram`` when you want to
        read what the object actually is. The difference between them is
        informative rather than cosmetic: ``[1:6]`` comes back as
        ``[1 2 3 4 5 6]``, ``50% so`` on a $10 line as ``5 so``, and a
        ``sev.X`` reference resolved inline. What you see is the parse, which
        is not always what the author thought they wrote.

        For the metadata, read ``.note`` / ``.tags`` / ``.doc`` directly, or
        pass ``trailer=True`` to :meth:`format_program`, which also exposes the
        layout and markup axes. Neither property is the source file; for that,
        read the ``.agg``. An object built programmatically, and an inline
        ``sev`` clause whose enclosing ``Aggregate`` owns the text, returns
        ``''``.
        """
        return self.format_program(fmt='text')

    @property
    def pprogram_html(self):
        """Syntax-highlighted DecL program for IPython / Jupyter display."""
        return self.format_program(fmt='html')


# ======================================================================
# Derived programs (dev/done/plan-derived-programs.md)
# ======================================================================

#: Spec keys owned by one reinsurance tier. ``occ_reins``, ``occ_kind`` and
#: their per-layer companions (``occ_reins_premium`` / ``_cede`` / ``_label`` /
#: ``_reinst``), and the aggregate twins including the variable-rating feature
#: keys (``agg_reins_swing`` / ``agg_reins_swing_layer`` ...). Group 1 is the
#: tier, which is what :func:`reins_program` replaces a tier's worth of.
_REINS_KEY = re.compile(r'^(occ|agg)_(?:reins|kind)')

#: Name of the throwaway carrier used to parse a bare cession clause. A
#: fragment is not a program, so it is slotted into the smallest aggregate that
#: has both reinsurance slots and parsed there. The name never reaches the
#: caller and the parse registers nothing.
_CESSION_PROBE = 'DerivedProgramCessionProbe'

#: Suffix for the wrapping P&L's name in :func:`pnl_program`.
_PNL_SUFFIX = '_PnL'

#: Premium above which a sized consideration rounds to whole currency units.
#: Above it the cents are noise against the quote; at or below it they are the
#: number. See :func:`_round_consideration`.
_CONSIDERATION_WHOLE_ABOVE = 100.0


def _parse_program(program):
    """Parse a stored DecL statement back to its ``(kind, name, spec)`` triple.

    The raw transformer spec, which is what :mod:`aggregate.decl_writer`
    renders, not the dense ``Aggregate._spec`` constructor dict.

    Parameters
    ----------
    program : str
        One DecL statement, typically an object's :attr:`ProgramMixin.program`.

    Returns
    -------
    (str, str, dict)
        Kind, name and spec, as :meth:`aggregate.parser.UnderwritingParser.parse`
        returns them.

    Notes
    -----
    Deferred import of the default underwriter, exactly as
    :func:`aggregate.decl_writer.format_program` does: it carries the configured
    recipe base, so ``sev.X`` and ``agg.X`` references in the text resolve, and
    a builtin reference resolves **inline**, which is what makes a derived
    program self-contained.
    """
    from .underwriter import build as _build
    return _build.parser.parse(program)


def _require_program(ob, caller):
    """Return ``ob``'s parsed program, or raise saying why there is none.

    Parameters
    ----------
    ob : Aggregate or Portfolio
        The object to derive from.
    caller : str
        Name of the calling member, for the message.

    Returns
    -------
    (str, str, dict)
        The parsed triple.

    Raises
    ------
    ValueError
        When the object carries no DecL program, which is the case for one
        built programmatically rather than through ``build``.
    """
    program = getattr(ob, 'program', '')
    if not program:
        raise ValueError(
            f"{caller}: {getattr(ob, 'name', ob)!r} carries no DecL program, so "
            "there is nothing to derive one from. Only objects built from DecL "
            "(through build) can answer; one constructed directly in Python "
            "has no text to start from.")
    return _parse_program(program)


def _merge_hints(hints, updates):
    """Merge ``key=value`` settings into an existing ``hints{...}`` body.

    A spec holds **one** ``hints`` clause, so a second cannot be appended: the
    duplicate would silently win or lose. Each setting is therefore replaced
    where it already appears and appended where it does not, and every other
    chunk survives untouched, in its original position.

    Parameters
    ----------
    hints : str
        The existing clause body (``'log2=18; padding=2'``), or ``''``.
    updates : dict
        ``{key: formatted_value}``, the values already rendered as the strings
        they should appear as.

    Returns
    -------
    str
        The merged body, ready to become ``hints{...}``.

    Notes
    -----
    Textual rather than parsed: :func:`aggregate.underwriter._parse_hints`
    drops keys outside its allow-list with a warning, so round-tripping through
    it would quietly delete anything it did not recognize. Chunks it would drop
    are still the author's, so they are carried through verbatim.

    Examples
    --------
    >>> _merge_hints('bs=1/32; padding=2', {'log2': '18', 'bs': '1/64'})
    'bs=1/64; padding=2; log2=18'
    """
    done = set()
    out = []
    for chunk in (hints or '').split(';'):
        chunk = chunk.strip()
        if not chunk:
            continue
        key = chunk.split('=')[0].strip() if chunk.count('=') == 1 else None
        if key in updates:
            if key in done:
                continue                # a duplicate of a key already replaced
            out.append(f'{key}={updates[key]}')
            done.add(key)
        else:
            out.append(chunk)
    out.extend(f'{k}={v}' for k, v in updates.items() if k not in done)
    return '; '.join(out)


def with_hints(ob, **extra):
    """The object's program with its realized grid pinned into a ``hints{...}``.

    The worker behind :meth:`aggregate.distributions.Aggregate.with_hints` and
    :meth:`aggregate.portfolio.Portfolio.with_hints`. See either for the user
    story; the mechanics are here so both read from one implementation.

    Parameters
    ----------
    ob : Aggregate or Portfolio
        An **updated** object carrying a DecL program.
    **extra
        Further ``hints{}`` settings, any member of
        :data:`aggregate.underwriter._HINT_KEYS`, merged over the three the
        object supplies.

    Returns
    -------
    str
        One line of DecL, ready to hand back to ``build``.

    Raises
    ------
    ValueError
        If the object has not been updated (there is no grid to pin), carries
        no DecL program, or is passed a hint key the language does not have.

    Notes
    -----
    The clause is **merged**, not appended: a spec holds one ``hints``, so the
    three keys are replaced where the author already wrote them and appended
    where they did not, and every other setting survives in place
    (:func:`_merge_hints`). Pinning the grid therefore does not discard a
    declared ``padding`` or ``sev_calc``.
    """
    from ._bucket_window import _fmt_bs
    from .underwriter import _HINT_KEYS

    who = getattr(ob, 'name', ob)
    bad = sorted(set(extra) - _HINT_KEYS)
    if bad:
        raise ValueError(
            f"with_hints: {', '.join(bad)} is not a hints key; DecL knows "
            f"{', '.join(sorted(_HINT_KEYS))}.")
    bs = getattr(ob, 'bs', 0) or 0
    log2 = getattr(ob, 'log2', 0) or 0
    if not bs or not log2:
        raise ValueError(
            f'with_hints: {who} has not been updated, so there is no grid to '
            'pin. Build or update it at the resolution you want first -- that '
            'is the whole point of the method.')
    kind, name, spec = _require_program(ob, 'with_hints')
    spec = dict(spec)
    updates = {'log2': str(int(log2)), 'bs': _fmt_bs(bs),
               'normalize': str(bool(getattr(ob, 'normalize', True)))}
    for k, v in extra.items():
        updates[k] = _fmt_bs(v) if k == 'bs' else str(v)
    spec['hints'] = _merge_hints(spec.get('hints', ''), updates)
    return _format_program((kind, name, spec), fmt='text', layout='terse',
                           trailer=True)


def _merge_note(note, addition, replace_prefix=None):
    """Append a sentence to an existing ``note{...}`` body.

    One ``note`` per spec, same as :func:`_merge_hints`, so an addition joins
    the existing text rather than opening a second clause.

    Parameters
    ----------
    note : str
        The existing note body, or ``''``.
    addition : str
        Text to add. Must not contain ``}``, which closes the clause.
    replace_prefix : str, optional
        When given, drop any existing ``;``-separated chunk that starts with it
        before appending. That is what makes a generated record **replaceable**
        rather than cumulative: sharpen twice and you want the latest verdict
        once, not both verdicts in sequence. The author's own prose never
        matches, because the prefixes are namespaced (``'sharpen: '``).

    Returns
    -------
    str
        The merged body.

    Examples
    --------
    >>> _merge_note('mine', 'sharpen: b', replace_prefix='sharpen: ')
    'mine; sharpen: b'
    >>> _merge_note('mine; sharpen: a', 'sharpen: b', replace_prefix='sharpen: ')
    'mine; sharpen: b'
    """
    if replace_prefix:
        kept = [c.strip() for c in (note or '').split(';')]
        note = '; '.join(c for c in kept
                         if c and not c.startswith(replace_prefix))
    if not note:
        return addition
    if not addition:
        return note
    return f'{note}; {addition}'


def _render_derived(kind, name, spec):
    """Render a mutated spec as canonical DecL, trailer and all.

    Parameters
    ----------
    kind : str
        ``'agg'``, ``'pnl'`` or ``'port'``.
    name : str
        The object name.
    spec : dict
        A raw transformer spec.

    Returns
    -------
    str
        The program, in the ``spread`` layout that :attr:`ProgramMixin.pprogram`
        uses: each clause on its own two-space indented line. The result is
        itself a program, so the rest of the render surface applies to it, for
        instance ``format_program(a.sharpen_program, layout='terse')``.
    """
    return _format_program((kind, name, spec), fmt='text', layout='spread',
                           trailer=True)


#: Prefix that marks sharpen's own sentence in a ``note{...}``. Namespaced so a
#: re-probe replaces its previous verdict instead of stacking a second one, and
#: so the author's own prose is never mistaken for it.
_SHARPEN_NOTE = 'sharpen: '


def _sharpen_trailer(ob, spec):
    """Merge the last probe's outcome into ``spec``'s ``note`` / ``hints``.

    Mutates ``spec`` in place. Three outcomes, three records.

    **The grid moved.** ``hints{log2=...; bs=...}`` pins it, and nothing else is
    added, because the hints are the record.

    **The grid was confirmed**, either because the probe gate waved it through
    or because the probe ran and the center cell won:
    ``note{sharpen: grid confirmed, no change}``, and deliberately **no hints**.
    Pinning a grid the automatic selector would have picked anyway adds noise to
    a program someone is going to read and share, and implies the selector is
    not trusted. The note is the record that the audit ran, which is what saves
    running it twice.

    **The probe ran under** ``execute=False``, found a better cell and did not
    take it. The object still sits on its original grid, so pinning the
    recommendation would describe an object that does not exist; the note
    carries the recommendation instead.

    Parameters
    ----------
    ob : Aggregate or Portfolio
        A sharpened object, read for ``_sharpen_state`` and ``_sharpen_df``.
    spec : dict
        A raw transformer spec, mutated in place.

    Notes
    -----
    Whatever the program already carried survives: the settings merge key by
    key (:func:`_merge_hints`) and the note is appended to. Sharpen's own
    sentence is *replaced* rather than repeated, so probing twice leaves one
    verdict, the latest.
    """
    from ._bucket_window import _fmt_bs
    state = getattr(ob, '_sharpen_state', None)
    if state is None:
        return
    df = getattr(ob, '_sharpen_df', None)
    won = None if df is None else df[df['selected']]
    if not state['ran'] or won is None or not len(won):
        # The probe gate: the grid was already at or under target and sound, so
        # nothing was run and nothing changed.
        moved = False
    else:
        win = won.iloc[0]
        moved = (float(win['bs']) != float(state['bs0'])
                 or int(win['log2']) != int(state['log20']))
    if not moved:
        addition = 'grid confirmed, no change'
    elif state['execute']:
        spec['hints'] = _merge_hints(
            spec.get('hints', ''),
            {'log2': str(int(win['log2'])), 'bs': _fmt_bs(win['bs'])})
        addition = None
    else:
        addition = (f'probe not executed, recommends log2 '
                    f'{int(win["log2"])} and bs {_fmt_bs(win["bs"])}')
    if addition is not None:
        spec['note'] = _merge_note(spec.get('note', ''),
                                   _SHARPEN_NOTE + addition,
                                   replace_prefix=_SHARPEN_NOTE)
    else:
        # The moved case pins the grid, so any stale verdict from an earlier
        # probe has to go: it would otherwise contradict the hints beside it.
        spec['note'] = _merge_note(spec.get('note', ''), '',
                                   replace_prefix=_SHARPEN_NOTE)


def pin_sharpen(ob):
    """Write the last probe's outcome onto ``ob``'s own program and trailer.

    Called by :func:`aggregate._bucket_window.sharpen` on the way out, which is
    what makes ``program`` mean *the program that builds this object* rather
    than merely the one that built it. Before this, an object whose grid the
    probe had moved carried a ``program`` that rebuilt it somewhere else, and a
    ``hints`` that was worse than empty: a declared ``hints{log2=17}`` survived
    a probe that changed ``bs``, so it read as a complete record of a grid it
    only half described.

    Parameters
    ----------
    ob : Aggregate or Portfolio
        The sharpened object, mutated in place.

    Notes
    -----
    Sets :attr:`ProgramMixin.program`, ``note`` and ``hints`` together, so the
    three never disagree. ``program`` is stamped the way ``build`` stamps it,
    through :meth:`aggregate.parser.UnderwritingLexer.preprocess`, so it stays
    one line with any ``doc{{{...}}}`` body base64 encoded.

    A no-op for an object built programmatically, which has no text to merge
    into, and for one whose program cannot be re-parsed (a reference resolvable
    only by a custom underwriter). Neither is worth failing a probe over: the
    grid still moved, and ``sharpen_description`` still says so.

    A grid the *caller* pinned at build time, ``build(program, log2=20)``, is
    not in the program text and not visible to the probe. Sharpen confirming it
    therefore writes the confirmation note and no hints, and ``program``
    continues to rebuild on the automatic grid. Pass the grid through a
    ``hints{}`` clause rather than a ``build`` keyword when it has to be
    durable.
    """
    from .parser import UnderwritingLexer
    if not getattr(ob, 'program', ''):
        return
    try:
        kind, name, spec = _parse_program(ob.program)
        spec = dict(spec)
        _sharpen_trailer(ob, spec)
        text = _format_program((kind, name, spec), fmt='text', layout='terse',
                               trailer=True)
        program = UnderwritingLexer.preprocess(text)[0]
    except Exception:                   # noqa: BLE001 a probe must not fail here
        logger.info('sharpen: could not pin the outcome onto %r; its program '
                    'is left as declared', getattr(ob, 'name', ob))
        return
    ob.program = program
    ob.note = spec.get('note', '')
    ob.hints = spec.get('hints', '')


def sharpen_program(ob):
    """The program that rebuilds ``ob`` on the grid its last probe chose.

    The fourth thing :func:`aggregate._bucket_window.sharpen` produces, beside
    ``sharpen_df`` / ``sharpen_description`` / ``sharpen_explanation``. Since
    :func:`pin_sharpen` writes that outcome onto ``ob.program`` as the probe
    finishes, this is that program rendered to read, the ``spread`` layout with
    the trailer left in. Ask for it when you want the text; read ``ob.program``
    when you want the one-line stamp, and ``ob.hints`` / ``ob.note`` when you
    want the settings and the verdict on their own.

    Parameters
    ----------
    ob : Aggregate or Portfolio
        A sharpened object.

    Returns
    -------
    str
        DecL, or ``''`` before :meth:`sharpen` has run (and for an object with
        no program at all).

    Notes
    -----
    The result is itself a program, so the rest of the render surface applies
    to it, for instance
    ``format_program(a.sharpen_program, layout='terse')``.
    """
    if getattr(ob, '_sharpen_state', None) is None:
        return ''
    if not getattr(ob, 'program', ''):
        return ''
    return _format_program(ob.program, fmt='text', layout='spread',
                           trailer=True)


def _pnl_consideration(ob, loss_ratio, caller):
    """Resolve the P&L premium: derive it, or size it from the loss ratio.

    Parameters
    ----------
    ob : Aggregate or Portfolio
        The engine.
    loss_ratio : float
        Target loss ratio, used only when the engine has no premium.
    caller : str
        Name of the calling member, for the messages.

    Returns
    -------
    DERIVE_PREMIUM or float
        The sentinel when the engine carries a technical premium, so the
        program says ``derive premium`` and the gross up for the expense
        clause is resolved at build; otherwise expected loss divided by
        ``loss_ratio``, rounded by :func:`_round_consideration`.

    Raises
    ------
    ValueError
        When the engine has no premium, and either ``loss_ratio`` is zero
        or the object has not been updated so its expected loss is unknown.

    Notes
    -----
    ``derive premium`` rather than ``inherit premium`` since 1.0.0a270: the
    engine premium is read as technical, so the wrapping P&L grosses it up
    for the expense clause the program writes, and premium net of expenses
    returns the technical premium exactly.

    Expected loss is the **empirical** mean ``est_m``, read off the computed
    density rather than from the analytic moments, so the P&L's realized loss
    ratio answers the one asked for rather than the analytic approximation to
    it. The rounding then moves it by at most half a currency unit on the
    premium, which is the price of a program a reader can keep.
    """
    from .parser import DERIVE_PREMIUM
    premium = getattr(ob, 'exp_premium', 0.0)
    total = (0.0 if premium is None
             else float(np.sum(np.asarray(premium, dtype=float))))
    if total:
        return DERIVE_PREMIUM
    if not loss_ratio:
        raise ValueError(
            f"{caller}: {getattr(ob, 'name', ob)!r} carries no premium to "
            "derive from (its exposure is stated as claims or loss), so the "
            "premium has to be sized from loss_ratio, and loss_ratio is "
            f"{loss_ratio!r}. Give a positive loss ratio.")
    e_loss = float(getattr(ob, 'est_m', 0.0) or 0.0)
    if not np.isfinite(e_loss) or e_loss <= 0:
        raise ValueError(
            f"{caller}: {getattr(ob, 'name', ob)!r} has no expected loss to "
            "size a premium from. Call update() first (build does it for you); "
            "the premium is the computed mean divided by loss_ratio.")
    return _round_consideration(e_loss / loss_ratio)


def _round_consideration(premium):
    """Round a sized premium to a number someone would write down.

    Parameters
    ----------
    premium : float
        Expected loss divided by the target loss ratio.

    Returns
    -------
    float
        Rounded to whole currency units above
        :data:`_CONSIDERATION_WHOLE_ABOVE`, to two decimals at or below it.

    Notes
    -----
    Only a **sized** premium is rounded. An inherited one is a number the
    program already stated, and restating it differently would make
    :func:`pnl_program` disagree with the exposure clause it wrapped.

    The unrounded quotient put ``1428.5840984231345 premium`` into a program
    the reader is meant to read, keep and edit: sixteen digits derived from an
    input of "about 70 percent". Sizing is a convention, so the answer carries
    a convention's worth of precision and no more.

    **The threshold has one joint on purpose.** Above 100 the cents are noise
    against a premium quoted in whole units; at or below it they are the
    number. A rule with more joints in it stops being predictable from the
    outside. Rounding here rather than in the writer also keeps the number and
    its printed form the same fact: :func:`aggregate.decl_writer._fmt_num`
    renders an integral float without its trailing zero, so the program reads
    ``1429 premium``, and the spec carries the value the program states.
    """
    value = float(premium)
    if value > _CONSIDERATION_WHOLE_ABOVE:
        return float(round(value))
    return round(value, 2)


def pnl_program(ob, loss_ratio=0.70, expense_ratio=0.25):
    """The program that wraps ``ob`` in a P&L.

    Writing this out by hand means knowing that the trailer belongs to the
    wrapping ``pnl`` rather than to the engine it swallows, and knowing what to
    do when the engine carries no premium. Both are grammar knowledge, so the
    library answers.

    Parameters
    ----------
    ob : Aggregate or Portfolio
        The loss engine to wrap.
    loss_ratio : float, default 0.70
        Sizes the premium as expected loss divided by this, and **only** when
        the engine has no premium of its own. An engine declared with a
        ``premium at lr`` exposure always derives from its own, and this is
        then unused. A convention rather than a fact, which is why it sits in
        the signature where the docstring puts it in front of you.
    expense_ratio : float, default 0.25
        Gross expense as a fraction of premium. ``0`` omits the expense clause
        rather than writing a zero.

    Returns
    -------
    str
        ``pnl NAME_PnL <premium> less <engine> less <expense>``.

    Raises
    ------
    ValueError
        When ``ob`` carries no DecL program, when its program is already a P&L,
        or when the premium cannot be resolved (see
        :func:`_pnl_consideration`).

    Notes
    -----
    **Self-contained either way, which is the load-bearing choice.** An
    aggregate engine is the object's own body verbatim; a portfolio engine is
    its units written out, ``less port PNAME <units>``. Both are stripped of
    their trailer, which the engine slot has no room for and the wrapping
    ``pnl`` now owns, and an aggregate's own ``as`` label becomes the engine
    label, which is what names the P&L's loss leg.

    The alternative for a portfolio was ``less port.NAME``, which is
    grammatical and shorter, but resolves only against the underwriter holding
    NAME: the returned text would build in the session that wrote it and
    nowhere else, and a shared server would be writing every user's books into
    one knowledge base to make it resolve. The same argument as
    :func:`reins_program`'s, and the reason the inline portfolio engine was
    added to the grammar at ``1.0.0a216``. A portfolio's own trailer is dropped
    rather than carried up, since it describes the book and not the P&L over
    it.

    Examples
    --------
    >>> from aggregate import build
    >>> a = build('agg PnlProgEx 1000 premium at 0.65 lr '
    ...           'sev lognorm 100 cv 1 poisson')
    >>> print(a.pnl_program())
    pnl PnlProgEx_PnL derive premium less
      agg PnlProgEx
        1000 premium at 0.65 lr
        sev lognorm 100 cv 1
        poisson
      less 0.25 premium expenses
    """
    kind, name, spec = _require_program(ob, 'pnl_program')
    if kind in ('pnl', 'xpnl'):
        raise ValueError(
            f"pnl_program: {name!r} is already a P&L, so there is nothing to "
            "wrap. Read its program directly.")
    if kind not in ('agg', 'port'):
        raise ValueError(
            f"pnl_program: cannot wrap a {kind!r} declaration in a P&L; the "
            "engine must be an aggregate or a portfolio.")
    pnl_name = f'{name}{_PNL_SUFFIX}'
    consideration = _pnl_consideration(ob, loss_ratio, 'pnl_program')
    if kind == 'port':
        # The units written out, never a ``port.NAME`` reference: the reference
        # resolves only against the underwriter holding the name, so the text
        # would build in the session that wrote it and nowhere else. The
        # portfolio's own trailer is dropped rather than carried up, since the
        # engine slot has no trailer and the metadata describes the book, not
        # the P&L over it.
        out = {'name': pnl_name,
               '_engine_port_spec': {'name': name, 'spec': spec['spec'],
                                     'label': spec.get('label')}}
    else:
        out = dict(spec)
        out['name'] = pnl_name
        out['engine_name'] = name
        # the aggregate's own label names the P&L's loss leg; the P&L itself
        # starts unlabeled.
        out['engine_label'] = out.pop('label', None)
    out['consideration'] = consideration
    if expense_ratio:
        out['expense_spec'] = [(None, [('premium', float(expense_ratio))])]
    return _render_derived('pnl', pnl_name, out)


def _cession_spec(cession):
    """Parse one or two bare cession clauses into their spec keys.

    A cession clause is a fragment, not a program, so it is slotted into a
    throwaway carrier aggregate that has both reinsurance slots and parsed
    there. Which slot is read off the leading keyword, and that is the whole
    point of the exercise: an ``occurrence`` cession sits **before** the
    frequency clause and an ``aggregate`` cession **after** it, so text spliced
    onto the end of a program lands in the wrong place or does not parse.

    Parameters
    ----------
    cession : str or iterable of str
        ``'occurrence net of 500 xs 500'``, or several such, at most one per
        tier.

    Returns
    -------
    (dict, set)
        The lifted ``occ_*`` / ``agg_*`` reinsurance keys, and the set of tiers
        (``'occ'`` / ``'agg'``) the caller is replacing.

    Raises
    ------
    ValueError
        For an empty cession, a clause that does not open with ``occurrence``
        or ``aggregate``, two clauses on the same tier, or a clause that does
        not parse.
    """
    fragments = [cession] if isinstance(cession, str) else list(cession)
    slots = {}
    for fragment in fragments:
        fragment = str(fragment).strip()
        head = fragment.split(None, 1)[0].lower() if fragment else ''
        if head not in ('occurrence', 'aggregate'):
            raise ValueError(
                f"reins_program: a cession clause opens with 'occurrence' or "
                f"'aggregate', naming the tier it applies to; got "
                f"{fragment!r}. For example 'occurrence net of 500 xs 500'.")
        tier = 'occ' if head == 'occurrence' else 'agg'
        if tier in slots:
            raise ValueError(
                f"reins_program: two {head} cessions given, and a program has "
                "one clause per tier. Put every layer in the one clause, "
                f"joined by 'and': '{head} net of 500 xs 500 and 1000 xs 1000'.")
        slots[tier] = fragment
    if not slots:
        raise ValueError(
            'reins_program: no cession given. Pass a clause such as '
            "'occurrence net of 500 xs 500'.")
    probe = (f'agg {_CESSION_PROBE} 1 claims dsev [1] '
             f'{slots.get("occ", "")} fixed {slots.get("agg", "")}')
    try:
        _, _, spec = _parse_program(probe)
    except Exception as e:                # noqa: BLE001 report the fragment
        raise ValueError(
            f'reins_program: could not parse the cession '
            f'{cession!r}: {e}') from None
    return ({k: v for k, v in spec.items() if _REINS_KEY.match(k)}, set(slots))


def reins_program(ob, cession):
    """The program that rebuilds ``ob`` with ``cession`` added.

    Parameters
    ----------
    ob : Aggregate
        The aggregate to cede from.
    cession : str or iterable of str
        One cession clause per tier, each opening with ``occurrence`` or
        ``aggregate``: ``'occurrence net of 500 xs 500'``.

    Returns
    -------
    str
        A **self-contained** program: it builds in any session, with nothing
        registered first.

    Raises
    ------
    ValueError
        When ``ob`` carries no DecL program, when its program is not a plain
        aggregate, when the cession is malformed (see :func:`_cession_spec`),
        or when an occurrence cession would join an ``approximate`` clause.

    Notes
    -----
    The cession is authoritative for **its own tier**: an occurrence clause
    replaces whatever occurrence program the object had and leaves the
    aggregate tier alone. That is the tier's whole cession program in one
    clause, which is how the grammar reads it too.

    **Self-contained, and this is the load-bearing choice.**
    ``agg NEW agg.OLD occurrence net of ...`` is grammatical and is the obvious
    first idea, but ``agg.OLD`` resolves only against an underwriter's
    knowledge base, so the returned text would build in the session that made
    it and nowhere else. A shared server would be writing every user's builds
    into one knowledge base besides, with the name collisions and unbounded
    growth that implies. Text that carries its own body has neither problem.

    Examples
    --------
    >>> from aggregate import build
    >>> a = build('agg ReinsProgEx 10 claims sev lognorm 100 cv 1 poisson')
    >>> print(a.reins_program('occurrence net of 500 xs 500'))
    agg ReinsProgEx
      10 claims
      sev lognorm 100 cv 1
      occurrence net of
        500 xs 500
      poisson
    """
    kind, name, spec = _require_program(ob, 'reins_program')
    if kind != 'agg':
        raise ValueError(
            f"reins_program: {name!r} is a {kind!r} declaration; a cession "
            'clause belongs to an aggregate. Cede the units of a portfolio '
            'individually.')
    lifted, tiers = _cession_spec(cession)
    if 'occ' in tiers and spec.get('approximate', 'exact') != 'exact':
        raise ValueError(
            f"reins_program: {name!r} carries 'approximate "
            f"{spec['approximate']}', which is incompatible with occurrence "
            'reinsurance (the method-of-moments fit bypasses the '
            'per-occurrence convolution); cede on the aggregate tier instead.')
    out = {k: v for k, v in spec.items()
           if not (_REINS_KEY.match(k) and _REINS_KEY.match(k).group(1) in tiers)}
    out.update(lifted)
    return _render_derived(kind, name, out)
