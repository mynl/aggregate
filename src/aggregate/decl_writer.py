"""DecL unparser and program formatter --- the inverse of the parser.

This module is the structural inverse of :mod:`aggregate.parser`: where the
``UnderwritingParser`` turns DecL text into a ``(kind, name, spec)`` tuple, the
``decl_writer`` turns a parsed *spec* back into canonical DecL text. That one
function (:func:`spec_to_decl`) backs program display (``pprogram`` /
``pprogram_html``), the ``to_agg`` exporter, and any future web ``format``
endpoint --- all rendering from the spec dictionary, never by regex surgery on
the original text (the approach the deleted ``utilities.decl_pprint`` used).

Three layers
------------
1. :func:`spec_to_decl` --- the unparser. A pure function from a **raw
   transformer spec** dict (``parsed.spec`` / the knowledge entry's ``pp.spec``,
   *not* the dense ``Aggregate._spec`` constructor-argument dict) to canonical
   DecL text. Built from small clause renderers (``_render_*``) that mirror the
   transformer rules one-for-one.
2. Colorization --- :func:`spec_to_decl` output is plain text; colorize by
   lexing it with the existing :class:`aggregate.decl_pygments.AggLexer` and
   rendering through a Pygments formatter selected by ``fmt`` (no new lexer, no
   second keyword list).
3. :func:`format_program` --- the public entry. Accepts a spec dict (preferred)
   or a program string (which it parses first), and returns a ``str`` for the
   requested ``fmt`` (``text`` / ``html`` / ``ansi`` / ``latex``) and ``layout``
   (``spread`` --- the multiline indented default --- or ``terse``, the single
   line / tab-indented-portfolio form that ``spec_to_decl`` also produces). The
   two axes are orthogonal: ``fmt`` is markup, ``layout`` is line structure.
   Pure: it returns a string and never prints.

Canonical, not verbatim
------------------------
The unparser is the inverse of the parser, not of the user's keystrokes. Many
distinct programs collapse to one spec (``50% so`` vs ``5 so`` for a $10 line,
``[1:6]`` vs ``[1 2 3 4 5 6]``, ``exp(.5)`` vs its evaluated float, a builtin
reference vs its resolved body), and several transformer rules are deliberately
lossy or normalizing (Tweedie discards its note and bakes a CP-gamma spec;
builtin scaling mangles the name; a negative multiplier folds into
``sev_reflect``). So the achievable invariant is **idempotence one step
removed**: with ``f = spec_to_decl``, ``f(f(f(x))) == f(x)``. The original text
``x`` need not equal ``f(x)`` --- it is only functionally equivalent (it builds
the same object). The canonical text is a fixed point reached after one
application. The round-trip test pins that fixed point and uses a
numpy/inf/object-aware spec comparator (a bare ``==`` would raise on the arrays
the specs carry).
"""

from __future__ import annotations

import re

import numpy as np

__all__ = ['spec_to_decl', 'format_program']

# Comment stripping for the text path of format_program. Mirrors
# UnderwritingLexer.preprocess EXCEPT for the leading bracket-newline-collapse
# step: that step unconditionally inserts a space after every ``]`` (to fold
# multiline numpy-array vectors), which silently rewrites ``[`` inside a
# ``note{...}`` and so is non-idempotent on bracketed notes. format_program's
# inputs (a stored single-line program, a spec) never carry multiline vectors,
# so omitting that step keeps re-rendering idempotent.
_FULL_LINE_COMMENT_RE = re.compile(r"(?m)^[ \t]*(?://|#)[^\n]*\n?")
_COMMENT_RE = re.compile(r"(//|#)[^\n]*")
_SEMICOLON_RE = re.compile(r";[ \t]*(\r?\n|$)")


def _split_statements(text: str) -> list[str]:
    """Split program text into logical statements under the blank-line / ``;`` rule.

    Removes full-line comments transparently (so they never separate
    statements), strips trailing comments, turns a line-final ``;`` into a
    paragraph break, then splits on runs of blank lines and flattens each
    paragraph (newlines and indentation collapse to single spaces, so a
    tab-indented portfolio folds into one statement). Mirrors
    :meth:`UnderwritingLexer.preprocess` minus its bracket-newline step, so
    bracketed notes survive intact.
    """
    text = _FULL_LINE_COMMENT_RE.sub("", text)
    text = _COMMENT_RE.sub("", text)
    text = _SEMICOLON_RE.sub("\n\n", text)
    statements = (re.sub(r"\s*\n\s*", " ", p).strip() for p in re.split(r"\n\s*\n", text))
    return [s for s in statements if s]

# Direct frequency families (the FREQ terminal in decl.lark). A spec whose
# ``freq_name`` is NOT in this set (and is not 'empirical') came from a
# ``mixed <id> ...`` clause, so it renders with the ``mixed`` keyword.
_FREQ_WORDS = frozenset({
    'binomial', 'pascal', 'poisson', 'bernoulli', 'geometric', 'fixed',
    'neymanA', 'neymana', 'neyman', 'logarithmic', 'negbin',
})


# ======================================================================
# Number formatting
# ======================================================================

def _fmt_num(x) -> str:
    """Format a scalar as a DecL number literal that re-parses to the same value.

    Integral floats render without a trailing ``.0`` (``5`` not ``5.0``);
    infinities render as ``inf`` / ``-inf``; everything else uses ``repr`` (the
    shortest string that round-trips to the same float).
    """
    xf = float(x)
    if np.isinf(xf):
        return 'inf' if xf > 0 else '-inf'
    if xf == int(xf) and abs(xf) < 1e15:
        return str(int(xf))
    return repr(xf)


def _is_seq(x) -> bool:
    """True if ``x`` should render as a bracketed vector rather than a scalar."""
    if isinstance(x, np.ndarray):
        return x.ndim > 0
    return isinstance(x, (list, tuple))


def _fmt_seq(x) -> str:
    """Format a scalar or vector as DecL: ``5`` or ``[1 2 3]``.

    Vectors render space-separated inside square brackets. Range/step sugar
    (``[1:6]``) is not reconstructed --- the explicit list re-parses to the same
    array, so re-folding would be cosmetic only.
    """
    if _is_seq(x):
        arr = np.atleast_1d(np.asarray(x, dtype=float))
        return '[' + ' '.join(_fmt_num(v) for v in arr) + ']'
    return _fmt_num(x)


def _fmt_name(name) -> str:
    """Format a severity name (or a bracketed list of names for a mixture)."""
    if isinstance(name, (list, tuple, np.ndarray)):
        return '[' + ' '.join(str(n) for n in name) + ']'
    return str(name)


def _all_one(x) -> bool:
    """True if every element of ``x`` equals 1 (a no-op scale factor)."""
    return bool(np.all(np.asarray(x, dtype=float) == 1.0))


def _all_zero(x) -> bool:
    """True if every element of ``x`` equals 0 (a no-op location shift)."""
    return bool(np.all(np.asarray(x, dtype=float) == 0.0))


def _negate(x):
    """Return ``-x`` preserving scalar-vs-vector shape (for a reflected scale)."""
    if _is_seq(x):
        return [-float(v) for v in np.atleast_1d(np.asarray(x, dtype=float))]
    return -float(x)


def _elt_div(num, den):
    """Element-wise ``num / den``; return a Python scalar when both are scalar.

    Used to recover the entered mean of a scaled ``mean cv`` severity: the
    transformer stores the post-scale mean, so ``entered = sev_mean / sev_scale``
    (see :func:`_render_dist`).
    """
    n = np.asarray(num, dtype=float)
    d = np.asarray(den, dtype=float)
    res = n / d
    if n.ndim == 0 and d.ndim == 0:
        return float(res)
    return res


# ======================================================================
# Severity
# ======================================================================

def _render_dist(spec: dict) -> str:
    """Render a continuous severity distribution body (the ``sev`` of an agg).

    Inverts the ``sev0`` / ``sev1`` / ``sev2`` / ``sev_weighted`` / ``sev_picks``
    / ``sev_unconditional`` transformer chain. Emits, in order:
    ``<scale> * <name> <params> + <loc>`` then ``wts``, ``splice``, ``picks`` and
    a trailing ``!`` for an unconditional severity.

    Notes
    -----
    Scale and reflection are folded into ``sev_scale`` / ``sev_reflect`` by the
    parser. A reflected severity (negative multiplier, or ``shift - X``) is
    reproduced by emitting a negative scale factor: ``-mag * name params + loc``
    re-parses to ``sev_reflect=True`` with the same magnitude, mean and shift.
    A scaled ``mean cv`` severity stores the post-scale mean, so the entered mean
    is recovered as ``sev_mean / sev_scale``.
    """
    name = _fmt_name(spec['sev_name'])
    scale = spec.get('sev_scale', 1.0)
    reflect = spec.get('sev_reflect', False)

    if 'sev_xs' in spec:
        # Empirical via the continuous `sev` rule: `<name> xps [xs] [ps]`
        # (dhistogram / chistogram). The bare `dsev [..]` clause is handled in
        # _render_sev_clause and never reaches here.
        core = f'{name} xps {_fmt_seq(spec["sev_xs"])} {_fmt_seq(spec["sev_ps"])}'
    elif 'sev_mean' in spec:
        inner = _elt_div(spec['sev_mean'], scale)
        core = f'{name} {_fmt_seq(inner)} cv {_fmt_seq(spec["sev_cv"])}'
    elif 'sev_b' in spec:
        core = f'{name} {_fmt_seq(spec["sev_a"])} {_fmt_seq(spec["sev_b"])}'
    elif 'sev_a' in spec:
        core = f'{name} {_fmt_seq(spec["sev_a"])}'
    else:
        core = name

    # Scale / reflection prefix. Only emitted when it does work: a non-unit
    # magnitude, or a reflection (which needs an explicit negative factor).
    if reflect or not _all_one(scale):
        factor = _negate(scale) if reflect else scale
        core = f'{_fmt_seq(factor)} * {core}'

    # Location shift (applied after scale). A negative shift renders `+ -10`,
    # which the `sev2_add` rule reads identically to a subtraction.
    loc = spec.get('sev_loc', 0.0)
    if not _all_zero(loc):
        core += f' + {_fmt_seq(loc)}'

    core += _render_weights(spec)
    core += _render_splice(spec)
    core += _render_picks(spec)
    if spec.get('sev_conditional') is False:
        core += ' !'
    return core


def _render_weights(spec: dict) -> str:
    """Render a mixture-weights clause (`` wts [...]`` / `` wts=n``) or ``''``.

    A scalar ``sev_wt`` of 1.0 is the unweighted default and renders nothing.
    Uniform weights collapse to the ``wts=n`` equal-weight sugar (which
    re-parses to the same vector).
    """
    wt = spec.get('sev_wt', 1.0)
    if np.isscalar(wt) and float(wt) == 1.0:
        return ''
    arr = np.atleast_1d(np.asarray(wt, dtype=float))
    if len(arr) > 1 and np.allclose(arr, arr[0]):
        return f' wts={len(arr)}'
    return f' wts {_fmt_seq(wt)}'


def _render_splice(spec: dict) -> str:
    """Render a `` splice [lb] [ub]`` clause, or ``''`` for the default support.

    The default ``(sev_lb=0, sev_ub=inf)`` renders nothing. A real splice always
    uses the two-bracket-list form, which re-parses identically to either the
    one-list (``splice [a b]``) or two-list source sugar.
    """
    lb = spec.get('sev_lb', 0.0)
    ub = spec.get('sev_ub', np.inf)
    is_default = (np.isscalar(lb) and float(lb) == 0.0
                  and np.isscalar(ub) and np.isinf(ub))
    if is_default:
        return ''
    return f' splice {_fmt_seq(lb)} {_fmt_seq(ub)}'


def _render_picks(spec: dict) -> str:
    """Render a `` picks [attachments] [losses]`` clause, or ``''`` if absent."""
    if 'sev_pick_attachments' not in spec:
        return ''
    return (f' picks {_fmt_seq(spec["sev_pick_attachments"])} '
            f'{_fmt_seq(spec["sev_pick_losses"])}')


def _is_dsev(spec: dict) -> bool:
    """True if the severity is a bare ``dsev`` clause (not the ``sev`` form).

    A ``dsev`` clause transforms straight to ``{sev_name: 'dhistogram', sev_xs,
    sev_ps}`` and never passes through ``sev_weighted``, so it lacks the
    ``sev_wt`` key that the continuous ``sev dhistogram xps ...`` form carries ---
    that key-set difference is the discriminator.
    """
    return (spec.get('sev_name') == 'dhistogram'
            and 'sev_xs' in spec and 'sev_wt' not in spec)


def _render_dsev(spec: dict) -> str:
    """Render a discrete ``dsev [outcomes] [probs]`` clause.

    Uniform probabilities are omitted (``dsev_main`` defaults to uniform when the
    probability list is empty). A trailing ``!`` marks an unconditional dsev.
    """
    s = f'dsev {_fmt_seq(spec["sev_xs"])}'
    ps = np.atleast_1d(np.asarray(spec['sev_ps'], dtype=float))
    if not np.allclose(ps, ps[0]):
        s += f' {_fmt_seq(spec["sev_ps"])}'
    if spec.get('sev_conditional') is False:
        s += ' !'
    return s


def _render_sev_clause(spec: dict) -> str:
    """Render the severity clause of an aggregate: ``dsev``, ``sev`` or ``ssev``."""
    if _is_dsev(spec):
        return _render_dsev(spec)
    # A reflected severity is signed (negative support) and the parser now
    # rejects it under the clamped ``sev`` clause, so it must round-trip as
    # ``ssev`` regardless of the explicit ``sev_signed`` flag.
    kw = 'ssev' if (spec.get('sev_signed') or spec.get('sev_reflect')) else 'sev'
    return f'{kw} {_render_dist(spec)}'


# ======================================================================
# Frequency
# ======================================================================

def _render_freq(spec: dict) -> str:
    """Render the frequency clause, or ``''`` for an empirical (``dfreq``) freq.

    A ``freq_name`` in :data:`_FREQ_WORDS` is a direct family
    (``poisson`` / ``negbin 3`` / ``pascal a b``); any other name is a mixing
    distribution and renders with the ``mixed`` keyword. ``zm`` / ``zt``
    zero-modification rides on the end. An ``empirical`` freq is the ``dfreq``
    exposure head and is rendered there, not here.
    """
    fn = spec.get('freq_name')
    if fn is None or fn == 'empirical':
        return ''

    if fn in _FREQ_WORDS:
        s = fn
    else:
        s = f'mixed {fn}'
    if 'freq_b' in spec:
        s += f' {_fmt_seq(spec["freq_a"])} {_fmt_seq(spec["freq_b"])}'
    elif 'freq_a' in spec:
        s += f' {_fmt_seq(spec["freq_a"])}'

    if spec.get('freq_zm'):
        p0 = spec.get('freq_p0', 0.0)
        s += ' zt' if float(p0) == 0.0 else f' zm {_fmt_num(p0)}'
    return s


# ======================================================================
# Exposure + layers
# ======================================================================

def _render_dfreq(spec: dict) -> str:
    """Render an empirical-frequency exposure head: ``dfreq [outcomes] [probs]``."""
    s = f'dfreq {_fmt_seq(spec["freq_a"])}'
    b = np.atleast_1d(np.asarray(spec['freq_b'], dtype=float))
    if not np.allclose(b, b[0]):
        s += f' {_fmt_seq(spec["freq_b"])}'
    return s


def _render_exposure(spec: dict) -> str:
    """Render the exposure clause: ``dfreq``, ``claims``, ``loss`` or ``premium at lr``.

    Notes
    -----
    ``premium at lr`` and ``exposure at rate`` produce identical spec keys
    (``exp_premium``, ``exp_lr``, ``exp_el``), so both canonicalize to the
    ``premium at lr`` form. The ``loss`` form sets only ``exp_el``; the premium
    form is detected first by the presence of ``exp_premium``.
    """
    if spec.get('freq_name') == 'empirical':
        return _render_dfreq(spec)
    if 'exp_premium' in spec:
        return (f'{_fmt_seq(spec["exp_premium"])} premium '
                f'at {_fmt_seq(spec["exp_lr"])} lr')
    if 'exp_el' in spec:
        return f'{_fmt_seq(spec["exp_el"])} loss'
    if 'exp_en' in spec:
        return f'{_fmt_seq(spec["exp_en"])} claims'
    return ''


def _render_layers(spec: dict) -> str:
    """Render a ``<limit> xs <attachment>`` layer clause, or ``''`` if ground-up.

    A tower (``tower [breaks]``) is stored as limit/attachment vectors and
    renders in the explicit ``xs`` form, which re-parses to the same vectors.
    """
    if 'exp_limit' in spec:
        return f'{_fmt_seq(spec["exp_limit"])} xs {_fmt_seq(spec["exp_attachment"])}'
    return ''


# ======================================================================
# Reinsurance
# ======================================================================

def _render_reins_clause(clause) -> str:
    """Render one ``(share, limit, attach)`` cession tuple.

    A full-line share (1.0) renders as ``limit xs attach``; a partial share
    renders as a percentage ``share of limit`` clause (``50% so limit xs
    attach``). The ``%`` marker tells the parser to read the leading number as a
    share directly, so the percentage form round-trips regardless of how the
    original was written (``so`` / ``po``, percent or absolute amount).
    """
    share, limit, attach = clause
    if float(share) == 1.0:
        return f'{_fmt_num(limit)} xs {_fmt_num(attach)}'
    return f'{_fmt_num(float(share) * 100)}% so {_fmt_num(limit)} xs {_fmt_num(attach)}'


def _render_reins(spec: dict, prefix: str, list_key: str, kind_key: str):
    """Render an occurrence or aggregate reinsurance clause, or ``''`` if absent.

    Returns a :class:`_Block` whose head is the clause keyword
    (``occurrence net of`` / ``aggregate ceded to``) and whose children are the
    cession strings, joined by ``and``. Terse it flattens to
    ``occurrence net of c1 and c2``; spread puts the keyword on its own line and
    each cession one indent deeper.

    Parameters
    ----------
    prefix : str
        ``'occurrence'`` or ``'aggregate'``.
    list_key, kind_key : str
        Spec keys for the cession list and the ``'net of'`` / ``'ceded to'``
        direction (``'occ_reins'`` / ``'occ_kind'`` or ``'agg_reins'`` /
        ``'agg_kind'``).
    """
    if list_key not in spec:
        return ''
    cessions = [_render_reins_clause(c) for c in spec[list_key]]
    return _Block(f'{prefix} {spec[kind_key]}', cessions, sep='and')


# ======================================================================
# Trailer + approximate
# ======================================================================

def _render_trailer(spec: dict) -> str:
    """Render the ``note{...}`` / ``hints{...}`` trailer, preserved verbatim."""
    parts = []
    if spec.get('note'):
        parts.append(f'note{{{spec["note"]}}}')
    if spec.get('hints'):
        parts.append(f'hints{{{spec["hints"]}}}')
    return ' '.join(parts)


def _render_approx(spec: dict) -> str:
    """Render the ``approximate <kind>`` clause, or ``''`` for the exact default."""
    kind = spec.get('approximate', 'exact')
    return '' if kind == 'exact' else f'approximate {kind}'


def _render_orientation(spec: dict) -> str:
    """Render the trailing ``payoff`` / ``loss`` orientation suffix.

    Emitted only when the spec carries an explicit ``value_type`` (the DecL
    orientation suffix set it); the omitted clause is the ``loss`` default and
    renders nothing, so a default ``agg`` round-trips unchanged. The ``pnl``
    path synthesizes ``value_type`` via ``_attach_pnl`` and renders through
    :func:`_render_pnl`, so this is the ``agg``-only path. See dev/plan-pnl.md.
    """
    vt = spec.get('value_type')
    if vt is None:
        return ''
    from .utilities import value_type_role
    return 'payoff' if value_type_role(vt) is False else 'loss'


def _join(parts) -> str:
    """Join non-empty clause fragments with single spaces."""
    return ' '.join(p for p in parts if p)


# ======================================================================
# Layout: terse (one line) vs spread (multiline, indented)
# ======================================================================

class _Block:
    """A head line plus indented children, rendered terse or spread.

    The structural renderers (``_render_agg`` / ``_render_pnl`` /
    ``_render_port`` / ``_render_bvagg`` and ``_render_reins``) return a tree of
    these instead of pre-joined strings, so both layouts share one ordering of
    clauses. A clause fragment is either a plain ``str`` (renders the same on its
    own line in both layouts) or a nested ``_Block``.

    Attributes
    ----------
    head : str
        The line that introduces the block (e.g. ``agg NAME``). Empty heads emit
        no line in spread (the children render directly).
    children : list of (str or _Block)
        The clause fragments, in render order. Empty fragments are dropped.
    sep : str
        Connective placed between children: ``''`` (plain space in terse) or
        ``'and'`` (reinsurance cessions --- ``c1 and c2`` terse, a trailing
        ``and`` on every non-final child line in spread).
    tab : bool
        Terse only: when set, children render on their own ``\\t``-indented lines
        instead of flattening to one line. Used by ``port`` so its terse form is
        byte-for-byte the historical tab-indented layout. Ignored in spread.
    """
    __slots__ = ('head', 'children', 'sep', 'tab')

    def __init__(self, head, children, sep='', tab=False):
        self.head = head
        self.children = children
        self.sep = sep
        self.tab = tab


def _render_terse(node) -> str:
    """Flatten a ``_Block`` / ``str`` tree to a single logical statement.

    Reproduces the historical single-line rendering byte-for-byte: a plain
    space-join of the (recursively flattened) non-empty children, with the
    ``sep`` connective between them, prefixed by the head. A ``tab`` block keeps
    its children on ``\\t``-indented continuation lines (the ``port`` form), which
    the preprocessor folds back into one statement on re-parse.
    """
    if isinstance(node, str):
        return node
    if node.tab:
        lines = [node.head] if node.head else []
        for child in node.children:
            text = _render_terse(child)
            if text:
                lines.append('\t' + text)
        return '\n'.join(lines)
    join = f' {node.sep} ' if node.sep else ' '
    body = join.join(t for c in node.children if (t := _render_terse(c)))
    return _join([node.head, body])


def _render_spread(node, depth: int = 0, indent: str = '  ') -> str:
    """Render a ``_Block`` / ``str`` tree as multiline, indented text.

    The head sits at ``indent * depth``; each child renders one level deeper, on
    its own line. For an ``and``-separated block every child except the last gets
    a trailing `` and`` so the cession list reads naturally. ``tab`` is ignored
    here --- spread indents uniformly with ``indent``.
    """
    pad = indent * depth
    if isinstance(node, str):
        return pad + node
    lines = [pad + node.head] if node.head else []
    # drop empty string fragments so they never emit a whitespace-only line
    kids = [c for c in node.children if not (isinstance(c, str) and not c)]
    last = len(kids) - 1
    for i, child in enumerate(kids):
        text = _render_spread(child, depth + 1, indent)
        if node.sep and i != last and text:
            text += f' {node.sep}'
        lines.append(text)
    return '\n'.join(lines)


# ======================================================================
# Top-level kind renderers
# ======================================================================

def _render_agg(name: str, spec: dict) -> _Block:
    """Render an ordinary loss aggregate (``agg NAME ...``).

    Clause order mirrors ``agg_out_full`` / ``agg_out_dfreq``: exposure (or
    ``dfreq``), layers, severity, occurrence reinsurance, frequency (omitted for
    ``dfreq``), aggregate reinsurance, ``approximate``, trailer. Returns a
    :class:`_Block` (head ``agg NAME``, the clauses its children) so it renders
    terse on one line or spread with each clause on its own indented line.
    """
    return _Block(f'agg {name}', [
        _render_exposure(spec),
        _render_layers(spec),
        _render_sev_clause(spec),
        _render_reins(spec, 'occurrence', 'occ_reins', 'occ_kind'),
        _render_freq(spec),
        _render_reins(spec, 'aggregate', 'agg_reins', 'agg_kind'),
        _render_approx(spec),
        _render_orientation(spec),
        _render_trailer(spec),
    ])


def _render_pnl(name: str, spec: dict) -> _Block:
    """Render a profit-and-loss aggregate (``pnl NAME <premium> premium - ...``).

    Inverts ``pnl_out_*`` / ``_attach_pnl``. The premium is ``consideration``;
    the exposure head after ``premium -`` is ``claims`` (``exp_en``), ``lr``
    (``exp_lr``, the bare loss-ratio form that binds to the premium) or ``loss``
    (``exp_el``).

    Returns a :class:`_Block` whose head keeps ``pnl NAME <premium> premium -``
    intact (so it re-parses to a :class:`aggregate.PnL`); the loss-head fragment
    is the first child clause.
    """
    premium = _fmt_seq(spec['consideration'])

    if spec.get('freq_name') == 'empirical':
        head = _render_dfreq(spec)
    elif 'exp_en' in spec:
        head = f'{_fmt_seq(spec["exp_en"])} claims'
    elif 'exp_lr' in spec:
        head = f'{_fmt_seq(spec["exp_lr"])} lr'
    elif 'exp_el' in spec:
        head = f'{_fmt_seq(spec["exp_el"])} loss'
    else:
        head = ''

    return _Block(f'pnl {name} {premium} premium -', [
        head,
        _render_layers(spec),
        _render_sev_clause(spec),
        _render_reins(spec, 'occurrence', 'occ_reins', 'occ_kind'),
        _render_freq(spec),
        _render_reins(spec, 'aggregate', 'agg_reins', 'agg_kind'),
        _render_approx(spec),
        _render_trailer(spec),
    ])


def _render_agg_or_pnl(kind: str, name: str, spec: dict) -> _Block:
    """Render an aggregate, dispatching to ``pnl`` when a consideration is set.

    A ``pnl`` declaration transforms to ``("pnl", name, spec)`` with a
    ``consideration`` key (set by ``_attach_pnl``); an ``agg`` has neither.
    """
    if kind == 'pnl' or 'consideration' in spec:
        return _render_pnl(name, spec)
    return _render_agg(name, spec)


def _render_sev_out(name: str, spec: dict) -> str:
    """Render a standalone severity definition (``sev NAME ...``)."""
    body = _render_dsev(spec) if _is_dsev(spec) else _render_dist(spec)
    return _join([f'sev {name}', body, _render_trailer(spec)])


def _render_port(name: str, spec: dict) -> _Block:
    """Render a portfolio: ``port NAME [trailer]`` then one indented unit per line.

    Sub-units are ``agg`` / ``pnl`` tuples; each becomes a child :class:`_Block`.
    The returned block is marked ``tab=True`` so its *terse* form is the
    historical tab-indented layout (head line, then ``\\t`` + each unit on one
    line) byte-for-byte; spread indents the units two spaces and lets their
    clauses spread one level deeper. Either way the preprocessor folds the
    indented continuation back into one logical statement on re-parse.
    """
    head = _join([f'port {name}', _render_trailer(spec)])
    units = [_render_agg_or_pnl(kind, sub_name, sub_spec)
             for kind, sub_name, sub_spec in spec['spec']]
    return _Block(head, units, tab=True)


def _render_copula(copula) -> str:
    """Render a ``copula <kind> [param]`` clause from a :class:`Copula` instance.

    The independence copula has no natural parameter and renders ``copula
    independent``; every other kind renders its single natural parameter.
    """
    kind = getattr(copula, '_name', None) or getattr(copula, 'name', '')
    param = getattr(copula, 'param', None)
    if param is None:
        return f'copula {kind}'
    return f'copula {kind} {_fmt_num(param)}'


def _render_dbvsev(spec: dict) -> str:
    """Render a discrete bivariate severity as the canonical dense form.

    ``dbvsev [xs] [ys] [[row] [row] ...]`` --- ``S[i][j] = P(X=xs[i], Y=ys[j])``.
    The sparse / uniform / range surface forms all normalise to the same dense
    ``dbv_*`` spec, so they canonicalise to this dense rendering (which re-parses
    to the same matrix).
    """
    xs, ys = spec['dbv_xs'], spec['dbv_ys']
    S = np.atleast_2d(np.asarray(spec['dbv_S'], dtype=float))
    rows = ' '.join('[' + ' '.join(_fmt_num(v) for v in row) + ']' for row in S)
    return f'dbvsev {_fmt_seq(xs)} {_fmt_seq(ys)} [{rows}]'


def _render_bvagg(name: str, spec: dict) -> _Block:
    """Render a bivariate (copula-coupled) aggregate or a ``netceded`` agg.

    Inverts ``bv_out_copula`` / ``bv_out_copula_nofreq`` / ``bv_out_copula_dfreq``,
    the discrete ``dbvsev`` forms (``bv_out_discrete*``), the three occurrence
    view-pair prefixes (``bv_out_netceded`` / ``bv_out_grossceded`` /
    ``bv_out_grossnet``), and the ``clash`` statement (``clash_out``). The two
    components are rendered inline (whitespace-insensitive within a program); the
    shared frequency is always emitted (the no-freq source form defaults to
    ``poisson``, which re-parses to the same spec).
    """
    if spec.get('mode') == 'discrete':
        return _Block(f'bivariate {name}', [
            _render_exposure(spec),
            _render_dbvsev(spec),
            _render_freq(spec),
            _render_trailer(spec),
        ])
    if 'clash' in spec:
        # Re-derive the clash surface from the stored (na, nb, nc); each
        # component renders as its limit + severity (the solved Bernoulli dfreq
        # is implied by the counts, so it is not emitted).
        cl = spec['clash']
        (_, _, sa), (_, _, sb) = spec['units']
        return _Block(f'clash {_fmt_name(name)}', [
            f"{_fmt_num(cl['na'])} {_fmt_num(cl['nb'])} {_fmt_num(cl['nc'])} claims",
            _join([_render_layers(sa), _render_sev_clause(sa)]),
            _join([_render_layers(sb), _render_sev_clause(sb)]),
            _render_freq(spec),
            _render_trailer(spec),
        ])

    if spec.get('mode') == 'netceded':
        # the keyword is the (x, y) view pair, names x-then-y
        _kw_for = {('net', 'ceded'): 'netceded', ('gross', 'ceded'): 'grossceded',
                   ('gross', 'net'): 'grossnet'}
        views = tuple(spec.get('nc_views') or ('net', 'ceded'))
        kw = _kw_for.get(views, 'netceded')
        kind, sub_name, sub_spec = spec['units'][0]
        unit = _render_agg_or_pnl(kind, sub_name, sub_spec)
        # prepend the view keyword to the unit's head line
        return _Block(f'{kw} {unit.head}', unit.children, unit.sep, unit.tab)

    components = [_render_agg_or_pnl(k, n, s) for k, n, s in spec['units']]
    return _Block(f'bivariate {name}', [
        _render_exposure(spec),
        *components,
        _render_copula(spec['copula']),
        _render_freq(spec),
        _render_trailer(spec),
    ])


def _render_distortion(name: str, spec: dict) -> str:
    """Render a distortion definition (``distortion NAME kind n1 n2 ...``).

    Inverts ``Distortion.decl_spec``: the flat number list is recovered from the
    kind's ``decl_params`` ordering. The ``minimum`` / ``mixture`` combinators
    take distortion *references* whose names are not retained on the constructed
    children, so they cannot round-trip and raise here.
    """
    from .spectral import Distortion

    kind = spec['name']
    if 'distortions' in spec:
        raise NotImplementedError(
            f"distortion {name!r}: the {kind!r} combinator references child "
            "distortions by name, which are not retained on the spec, so it "
            "cannot be rendered back to DecL.")
    subclass = Distortion._registry.get(kind)
    if subclass is None:
        raise ValueError(f"distortion {name!r}: unknown kind {kind!r}.")
    params = subclass.decl_params or (subclass.param_name,)
    numbers = ' '.join(_fmt_num(spec[p]) for p in params)
    return f'distortion {name} {kind} {numbers}'


_KIND_RENDERERS = {
    'agg': lambda name, spec: _render_agg_or_pnl('agg', name, spec),
    'pnl': lambda name, spec: _render_agg_or_pnl('pnl', name, spec),
    'sev': _render_sev_out,
    'port': _render_port,
    'bvagg': _render_bvagg,
    'distortion': _render_distortion,
}


def _spec_to_node(spec: dict, kind: str = 'agg', name: str | None = None):
    """Dispatch a spec to its kind renderer, returning a ``_Block`` or ``str``.

    The shared front half of :func:`spec_to_decl` (terse) and
    :func:`format_program` (either layout): it resolves the name default and the
    kind renderer, but does *not* flatten --- the caller picks the layout walker.
    Renderers with sub-clause nesting (``agg`` / ``pnl`` / ``port`` / ``bvagg``)
    return a :class:`_Block`; the flat ones (``sev`` / ``distortion``) return a
    ``str``, which both walkers pass through unchanged.

    Raises
    ------
    ValueError
        For an unknown kind.
    """
    if name is None:
        name = spec.get('name')
    try:
        renderer = _KIND_RENDERERS[kind]
    except KeyError:
        raise ValueError(
            f"spec_to_decl: unknown kind {kind!r}; expected one of "
            f"{sorted(_KIND_RENDERERS)}.") from None
    return renderer(name, spec)


def spec_to_decl(spec: dict, kind: str = 'agg', name: str | None = None) -> str:
    """Render a parsed spec back to canonical DecL text (the unparser).

    The structural inverse of :class:`aggregate.parser.UnderwritingParser`.

    Parameters
    ----------
    spec : dict
        A **raw transformer spec** --- ``parsed.spec`` or a knowledge entry's
        ``pp.spec``. *Not* the dense ``Aggregate._spec`` constructor-argument
        dict (which is a different, defaulted shape; see the module docstring).
    kind : str, default 'agg'
        One of ``'agg'``, ``'sev'``, ``'port'``, ``'bvagg'``, ``'distortion'``
        --- the first element of the parser's ``(kind, name, spec)`` tuple.
    name : str, optional
        The object name. Defaults to ``spec['name']`` when present; required for
        distortions (whose ``spec['name']`` holds the *kind*, not the object
        name).

    Returns
    -------
    str
        Canonical DecL, **always terse** (one logical statement; a portfolio
        renders as a head line plus tab-indented units). This is the byte-for-byte
        form the ``to_agg`` exporter and the round-trip tests depend on ---
        :func:`format_program` is where the multiline ``spread`` layout lives.

    Raises
    ------
    NotImplementedError
        For a ``minimum`` / ``mixture`` combinator distortion (references cannot
        round-trip).
    ValueError
        For an unknown kind.

    Notes
    -----
    The output is canonical, not verbatim --- see the module docstring on the
    "idempotence one step removed" contract.
    """
    return _render_terse(_spec_to_node(spec, kind, name))


# ======================================================================
# Colorization (Layer 2) + public formatter (Layer 3)
# ======================================================================

def _colorize(text: str, fmt: str) -> str:
    """Colorize plain DecL text by lexing with ``AggLexer`` and a Pygments formatter.

    Reuses the existing :class:`aggregate.decl_pygments.AggLexer` --- no second
    keyword list. ``fmt='text'`` is a passthrough.
    """
    if fmt == 'text':
        return text
    from pygments import highlight
    from .decl_pygments import AggLexer

    if fmt == 'html':
        from pygments.formatters import HtmlFormatter
        formatter = HtmlFormatter(style='friendly', full=False)
    elif fmt == 'ansi':
        from pygments.formatters import Terminal256Formatter
        formatter = Terminal256Formatter(style='friendly')
    elif fmt == 'latex':
        from pygments.formatters import LatexFormatter
        formatter = LatexFormatter(style='friendly')
    else:
        raise ValueError(
            f"format_program: unknown fmt {fmt!r}; expected 'text', 'html', "
            "'ansi' or 'latex'.")
    # highlight appends a trailing newline; strip it so the return is tight.
    return highlight(text, AggLexer(), formatter).rstrip('\n')


def _render_statement(underwriter, statement: str):
    """Parse one statement and return its render node, or the text verbatim.

    Returns a :class:`_Block` / ``str`` node (which the caller renders in the
    requested layout). The verbatim ``str`` fallback keeps :func:`format_program`
    (hence ``pprogram``) from raising when a program references a builtin that
    the default underwriter cannot resolve --- e.g. a ``sev.X`` defined only in a
    custom knowledge base. A plain ``str`` renders identically in both layouts.
    """
    try:
        kind, name, spec = underwriter.parser.parse(statement)
        return _spec_to_node(spec, kind, name)
    except Exception:
        # Display fallback: a parse error (malformed text) or an unresolved
        # builtin reference (defined only in a custom underwriter, or a lark
        # VisitError wrapping the KeyError) must not crash pprogram.
        return statement


def format_program(spec_or_text, *, fmt: str = 'text', layout: str = 'spread',
                   width=None) -> str:
    """Render a DecL program in canonical form, optionally colorized.

    The public entry point backing ``pprogram`` / ``pprogram_html`` and the
    ``to_agg`` exporter. Pure: returns a ``str`` and never prints (an HTML / ANSI
    payload is still a ``str`` --- printing is the caller's job, mirroring stdlib
    ``pprint.pformat`` vs ``pprint.pprint``).

    Parameters
    ----------
    spec_or_text : dict, tuple or str
        A raw spec ``dict`` (rendered as ``kind='agg'`` unless it is a tuple), a
        ``(kind, name, spec)`` tuple, or a DecL program string (which is parsed
        first --- so doc snippets that pass ``obj.program`` keep working). An
        empty / whitespace-only string returns ``''``.
    fmt : {'text', 'html', 'ansi', 'latex'}, default 'text'
        Output *markup*. ``text`` is plain; the others colorize via Pygments.
        Orthogonal to ``layout``.
    layout : {'spread', 'terse'}, default 'spread'
        Line *layout*. ``spread`` (the default) puts each clause on its own
        two-space-indented line, with reinsurance cessions and portfolio /
        bivariate sub-aggregates nested one level deeper. ``terse`` is the
        historical single-line-per-statement form (a portfolio keeps its
        tab-indented units) and is byte-for-byte what ``spec_to_decl`` /
        ``to_agg`` produce. Both layouts re-parse to the same spec --- the
        preprocessor collapses intra-statement newlines and indentation to a
        single space.
    width : int, optional
        Reserved for future per-line wrapping of long clauses; currently ignored
        (``layout`` is structural --- one clause per line, not width-driven).

    Returns
    -------
    str
        Canonical DecL for the requested format and layout. Top-level statements
        are separated by a blank line (a lone newline is not a statement
        separator on re-parse).
    """
    if layout not in ('spread', 'terse'):
        raise ValueError(
            f"format_program: unknown layout {layout!r}; expected 'spread' or "
            "'terse'.")
    render = _render_terse if layout == 'terse' else _render_spread

    if isinstance(spec_or_text, str):
        text = spec_or_text.strip()
        if not text:
            return ''
        # Lazy import: avoids a parser <-> writer import cycle and keeps the
        # ~1s Lark/pygments cost off the bare `import aggregate` path. The
        # default underwriter carries the configured knowledge base so most
        # builtin references in the text resolve.
        from .underwriter import build as _build
        nodes = [_render_statement(_build, line)
                 for line in _split_statements(text)]
    elif isinstance(spec_or_text, tuple):
        kind, name, spec = spec_or_text
        nodes = [_spec_to_node(spec, kind, name)]
    else:
        nodes = [_spec_to_node(spec_or_text)]

    # Join top-level statements with a blank line: a lone '\n' is not a statement
    # separator (the preprocessor folds it into the preceding statement), so the
    # multi-statement path must use '\n\n' to re-parse correctly.
    rendered = '\n\n'.join(render(n) for n in nodes)
    return _colorize(rendered, fmt)
