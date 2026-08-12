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
   transformer spec** dict (``parsed.spec`` / a recipe's ``spec``,
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
lossy or normalizing (builtin scaling mangles the name; a negative multiplier
folds into ``sev_reflect``). Tweedie was on that list until 1.0.0a231, when it
baked a CP-gamma spec and overwrote the author's note; it now records what was
declared under ``_tweedie`` and round-trips. So the achievable invariant is **idempotence one step
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

def _split_statements(text: str) -> list[str]:
    """Split program text into logical statements under the blank-line / ``;`` rule.

    One line, delegating to :meth:`UnderwritingLexer.preprocess`, so the writer
    and the reader agree on what a statement is by construction.

    Notes
    -----
    This was a near-copy of ``preprocess`` until 1.0.0a177, forked to omit the
    bracket-newline step: that step pads every ``[`` and ``]`` with a space,
    which silently rewrote the brackets inside a ``note{...}`` and made
    re-rendering non-idempotent on a bracketed note. The fork treated the
    symptom and left the ``#`` / ``//`` half of the same bug in place, so a note
    holding either still truncated here. ``preprocess`` now lifts trailer bodies
    out (step 0b), which fixes the cause, and the copy has no reason to exist.

    The returned text is fed straight back to the parser, so the padding the
    bracket step still applies *outside* a trailer body is invisible: the lexer
    ignores whitespace, and nothing rendered by ``format_program`` comes from
    here.
    """
    from .parser import UnderwritingLexer

    return UnderwritingLexer.preprocess(text)

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
    # Interior ``severity`` label rides in ``label_map`` (dev/plan-labels.md S3),
    # appended after the whole clause.
    label = _render_label(spec.get('label_map', {}).get('severity'))
    if _is_dsev(spec):
        return _render_dsev(spec) + label
    # The keyword follows ``sev_signed`` alone. Reflection is orthogonal: both
    # ``sev 100 - lognorm`` (clamps at 0) and ``ssev 100 - lognorm`` (keeps the
    # negative support) are legal and mean different things, so keying the
    # keyword off ``sev_reflect`` would rewrite the former as the latter and
    # silently change the declaration. See
    # dev/done/plan-reflected-loss-severity.md ([Reflect-Unparser-Round-Trip]).
    kw = 'ssev' if spec.get('sev_signed') else 'sev'
    return f'{kw} {_render_dist(spec)}{label}'


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
    if fn == 'renewal':
        # the wait clause renders in the freq slot via _render_wait; the
        # 'renewal' name itself is never a DecL keyword (never in _FREQ_WORDS)
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
        # trailing ! = pin the realized mean to the exposure clause
        if spec.get('freq_pin_mean'):
            s += ' !'
    return s


def _is_dwait(spec: dict) -> bool:
    """True if the wait law is a bare ``dwait`` clause (not the ``wait`` form).

    Mirrors :func:`_is_dsev`: a ``dwait`` transforms straight to
    ``{wait_name: 'dhistogram', wait_xs, wait_ps}`` and never passes through
    the weighted ``sev`` chain, so it lacks ``wait_wt``.
    """
    return (spec.get('wait_name') == 'dhistogram'
            and 'wait_xs' in spec and 'wait_wt' not in spec)


def _render_wait(spec: dict) -> str:
    """Render the renewal waiting-time clause (``wait ...`` / ``dwait ...``).

    Sits in the freq slot of an ``agg_body_renewal``. The continuous form
    builds a ``sev_*`` view of the ``wait_*`` keys and reuses
    :func:`_render_dist`, so every severity-mini-language feature (scale,
    mixtures, splice, ``!``) round-trips through one code path. Returns ``''``
    for a non-renewal spec.
    """
    if spec.get('freq_name') != 'renewal':
        return ''
    label = _render_label(spec.get('label_map', {}).get('wait'))
    if _is_dwait(spec):
        s = f'dwait {_fmt_seq(spec["wait_xs"])}'
        ps = np.atleast_1d(np.asarray(spec['wait_ps'], dtype=float))
        if not np.allclose(ps, ps[0]) or spec.get('wait_conditional') is False:
            # defective probs must always render explicitly -- a uniform
            # sub-stochastic vector (e.g. [.45 .45] !) cannot fall back to
            # the uniform-default sugar, which re-parses to sum 1
            s += f' {_fmt_seq(spec["wait_ps"])}'
        if spec.get('wait_conditional') is False:
            s += ' !'
        return s + label
    # layered wait (``wait y xs a <dist>``): the layer term renders between
    # ``wait`` and the distribution, mirroring the exposures layer clause
    layer = ''
    if 'wait_limit' in spec or 'wait_attachment' in spec:
        attach = spec.get('wait_attachment')
        layer = (f'{_fmt_seq(spec.get("wait_limit", np.inf))} xs '
                 f'{_fmt_seq(0 if attach is None else attach)} ')
    view = {'sev_' + k[5:]: v for k, v in spec.items()
            if k.startswith('wait_') and k not in ('wait_limit',
                                                   'wait_attachment')}
    return f'wait {layer}{_render_dist(view)}{label}'


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
    *sizing* form is detected by ``exp_premium`` together with ``exp_lr``. An
    ``exp_premium`` with no ``exp_lr`` is the informational suffix on a claims
    or loss head ([FYI-Premium-Exposure-Head]) and renders after that head:
    ``5 claims 20000 premium``. Its optional label rides on the ``premium``
    site of ``label_map``.

    A claim count of exactly one renders ``1 claim``, the spelling the grammar
    also accepts and the one a reader expects. It is cosmetic to the parser
    (``CLAIMS`` covers both) and matters to the shipped library, where 46
    single-claim severity entries would otherwise read ``1 claims``.
    """
    if spec.get('freq_name') == 'empirical':
        return _render_dfreq(spec)
    # Interior ``exposure`` label rides in ``label_map`` (dev/plan-labels.md S1);
    # it prints right after the amount/keyword unit, before any ``at ... lr/rate``.
    label = _render_label(spec.get('label_map', {}).get('exposure'))
    if spec.get('freq_name') == 'renewal':
        # FIRST: a ``years at rate`` spec also carries ``exp_premium`` and
        # would otherwise mis-render as ``premium at lr``. ``exp_rate`` is
        # stored by the parser precisely for this byte-exact round-trip.
        s = f'{_fmt_seq(spec["exp_years"])} years{label}'
        if 'exp_rate' in spec:
            s += f' at {_fmt_seq(spec["exp_rate"])} rate'
        return s
    if 'exp_premium' in spec and 'exp_lr' in spec:
        return (f'{_fmt_seq(spec["exp_premium"])} premium{label} '
                f'at {_fmt_seq(spec["exp_lr"])} lr')
    # informational premium suffix on a claims / loss head (no exp_lr)
    fyi = ''
    if 'exp_premium' in spec:
        plabel = _render_label(spec.get('label_map', {}).get('premium'))
        fyi = f' {_fmt_seq(spec["exp_premium"])} premium{plabel}'
    if 'exp_el' in spec:
        return f'{_fmt_seq(spec["exp_el"])} loss{label}{fyi}'
    if 'exp_en' in spec:
        count = _fmt_seq(spec['exp_en'])
        return f'{count} {"claim" if count == "1" else "claims"}{label}{fyi}'
    return ''


def _render_layers(spec: dict) -> str:
    """Render a ``<limit> xs <attachment>`` layer clause, or ``''`` if ground-up.

    A tower (``tower [breaks]``) is stored as limit/attachment vectors and
    renders in the explicit ``xs`` form, which re-parses to the same vectors.
    """
    if 'exp_limit' in spec:
        # Interior ``layer`` label rides in ``label_map`` (dev/plan-labels.md S2),
        # appended after the ``xs`` clause.
        label = _render_label(spec.get('label_map', {}).get('layer'))
        return (f'{_fmt_seq(spec["exp_limit"])} xs '
                f'{_fmt_seq(spec["exp_attachment"])}{label}')
    return ''


# ======================================================================
# Reinsurance
# ======================================================================

def _render_collar(collar: dict) -> str:
    """Render a keyword-first collar ``basic <b> lcm <m> [min <lo>] [max <hi>]``.

    Shared by ``swing`` (ceded premium) and ``retro`` (gross premium); ``min`` /
    ``max`` render only when present (``None`` = omitted), preserving the spec.
    """
    s = f"basic {_fmt_num(collar['basic'])} lcm {_fmt_num(collar['lcm'])}"
    if collar.get('minimum') is not None:
        s += f" min {_fmt_num(collar['minimum'])}"
    if collar.get('maximum') is not None:
        s += f" max {_fmt_num(collar['maximum'])}"
    return s


def _render_variable_feature(spec: dict, list_key: str, layer_idx: int) -> str:
    """Render the variable-rating decorator on layer ``layer_idx``, or ``''``.

    Mirrors the four Phase-3 spec keys ``<list_key>_swing`` / ``_slide`` / ``_pc``
    / ``_corridor`` (each a params dict, with a parallel ``..._layer`` index) back
    to their DecL surface so a variable-rating program round-trips. ``min`` / ``max``
    on a swing collar render only when present (``None`` = omitted), preserving the
    parsed spec exactly.
    """
    for feat in ('swing', 'slide', 'pc', 'corridor'):
        params = spec.get(f'{list_key}_{feat}')
        if params is None or spec.get(f'{list_key}_{feat}_layer') != layer_idx:
            continue
        if feat == 'swing':
            return f'swing {_render_collar(params)}'
        if feat == 'slide':
            anchors = ' and '.join(f'{_fmt_num(c)} at {_fmt_num(lr)}'
                                   for c, lr in params['anchors'])
            return f'slide {anchors}'
        if feat == 'pc':
            return (f"pc {_fmt_num(params['share'])} "
                    f"after {_fmt_num(params['allowance'])}")
        if feat == 'corridor':
            return (f"corridor {_fmt_num(params['share'])} po "
                    f"{_fmt_num(params['width'])} xs {_fmt_num(params['attachment'])}")
    return ''


def _render_reins_clause(clause, premium=None, cede=None, reinst=None,
                         variable='', label=None) -> str:
    """Render one ``(share, limit, attach)`` cession tuple with its decorators.

    A full-line share (1.0) renders as ``limit xs attach``; a partial share
    renders as a percentage ``part of limit`` clause (``50% po limit xs
    attach``). The ``%`` marker tells the parser to read the leading number as a
    share directly.

    The spec stores only the resolved fraction, so this function has no record
    of how the original was written and the percentage form is the only form the
    library emits. A partial placement entered as a bare amount (``5 po 15 xs
    5``) therefore round-trips to ``33.3333% po 15 xs 5``: same layer, canonical
    spelling. Before 1.0.0a249 the canonical keyword was ``so``.

    An optional ceded-premium ``premium`` (``(basis, value)`` with basis
    ``deposit`` / ``rol`` / ``rate``) and ceding-commission ``cede`` (a fraction)
    render after the layer. ``rol`` / ``rate`` / ``cede`` are fractions rendered
    as bare numbers (re-parse identically, no ``%`` float dust). An optional
    reinstatement schedule ``reinst`` (a tuple of price multipliers) renders in
    the canonical explicit-list form ``reinstatements [a1 a2 ...]`` (the
    treaty-language group chain is input sugar that canonicalises to the list);
    the **empty** tuple renders as ``no reinstatements`` (zero reinstatements, a
    single annual limit).
    """
    share, limit, attach = clause
    if float(share) == 1.0:
        base = f'{_fmt_num(limit)} xs {_fmt_num(attach)}'
    else:
        base = (f'{_fmt_num(float(share) * 100)}% po '
                f'{_fmt_num(limit)} xs {_fmt_num(attach)}')
    parts = [base]
    if premium is not None:
        basis, value = premium
        parts.append(f'{basis} {_fmt_num(value)}')
    if cede is not None:
        parts.append(f'cede {_fmt_num(cede)}')
    if reinst is not None:
        if len(reinst) == 0:
            parts.append('no reinstatements')
        else:
            parts.append(
                f'reinstatements [{" ".join(_fmt_num(a) for a in reinst)}]')
    if variable:
        parts.append(variable)
    return ' '.join(parts) + _render_label(label)


def _render_reins(spec: dict, prefix: str, list_key: str, kind_key: str):
    """Render an occurrence or aggregate reinsurance clause, or ``''`` if absent.

    Returns a :class:`_Block` whose head is the clause keyword
    (``occurrence net of`` / ``aggregate ceded to``) and whose children are the
    cession strings, joined by ``and``. Terse it flattens to
    ``occurrence net of c1 and c2``; spread puts the keyword on its own line and
    each cession one indent deeper. Per-layer ceded-premium / ``cede`` decorators
    (parallel ``<list_key>_premium`` / ``<list_key>_cede`` lists) ride along.

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
    layers = spec[list_key]
    prem = spec.get(f'{list_key}_premium')
    cede = spec.get(f'{list_key}_cede')
    reinst = spec.get(f'{list_key}_reinst')
    label = spec.get(f'{list_key}_label')
    cessions = [
        _render_reins_clause(c,
                             prem[i] if prem is not None else None,
                             cede[i] if cede is not None else None,
                             reinst[i] if reinst is not None else None,
                             _render_variable_feature(spec, list_key, i),
                             label[i] if label is not None else None)
        for i, c in enumerate(layers)]
    return _Block(f'{prefix} {spec[kind_key]}', cessions, sep='and')


# ======================================================================
# Trailer + approximate
# ======================================================================

#: The trailer items, in render order.
TRAILER_ITEMS = ('note', 'tags', 'hints', 'doc')


def _render_trailer(spec: dict, trailer=True) -> list:
    """Render the trailer as **one fragment per clause**, in render order.

    Returns a list, not a joined string, so each of ``note`` / ``tags`` /
    ``hints`` / ``doc`` becomes its own child of the enclosing
    :class:`_Block`. That is what puts them on separate lines, each at its
    statement's indentation level, in the ``spread`` layout; ``terse``
    space-joins them back, so the single-line form is unchanged.

    ``trailer`` is ``True`` (all four), ``False`` (none), or an iterable naming
    the items to keep, e.g. ``('hints',)`` renders the declaration with only
    the clause that changes how it *builds*. That is what
    :attr:`aggregate.recipe.Recipe.decl` uses to expand a ``<<decl>>``
    placeholder: inside a recipe the note, tags and doc are the surrounding
    page, so repeating them in the program would be redundant, and repeating
    the doc would make it quote itself.

    Suppressing anything means the result no longer re-parses to the same spec.
    That is the caller's choice, made explicitly at the call site.

    Notes
    -----
    ``doc{{{...}}}`` is emitted last and spans lines: its opening fence ends a
    line and its closing fence sits alone on one, which is exactly the shape
    step 0 of :meth:`aggregate.parser.UnderwritingLexer.preprocess` looks for.
    That is what makes the round trip work: the writer emits a raw markdown
    body, and re-parsing re-encodes it. Only the fragment's *first* line is
    indented by the layout walker, so the markdown body keeps its own column
    positions and its fenced code blocks survive.
    """
    if trailer is True:
        wanted = TRAILER_ITEMS
    elif not trailer:
        return []
    else:
        wanted = tuple(trailer)
        unknown = set(wanted) - set(TRAILER_ITEMS)
        if unknown:
            raise ValueError(
                f'unknown trailer item(s) {sorted(unknown)}; '
                f'expected a subset of {TRAILER_ITEMS}')
    parts = []
    if 'note' in wanted and spec.get('note'):
        parts.append(f'note{{{spec["note"]}}}')
    if 'tags' in wanted and spec.get('tags'):
        parts.append(f'tags{{{", ".join(spec["tags"])}}}')
    if 'hints' in wanted and spec.get('hints'):
        parts.append(f'hints{{{spec["hints"]}}}')
    if 'doc' in wanted and spec.get('doc'):
        # Newline-delimited: the closing fence MUST be alone on its line.
        parts.append(f'doc{{{{{{\n{spec["doc"]}\n}}}}}}')
    return parts


def _render_approx(spec: dict) -> str:
    """Render the ``approximate <kind>`` clause, or ``''`` for the exact default."""
    kind = spec.get('approximate', 'exact')
    return '' if kind == 'exact' else f'approximate {kind}'


def _render_peel(spec: dict) -> str:
    """Render the ``peel <direction>`` clause, or ``''`` for the tier default.

    The absent clause is the tier walk, so a P&L that does not peel renders
    unchanged ([Layer-Peeling-Shorthand]).
    """
    direction = spec.get('peel')
    return '' if not direction else f'peel {direction}'


def _render_label(label) -> str:
    """Render an ``as <label>`` clause (``''`` when ``label`` is ``None``).

    A bareword label (a valid identifier with no spaces) renders unquoted; any
    other label is double-quoted. See dev/plan-decl-labels.md.
    """
    if not label:
        return ''
    if re.fullmatch(r'[a-zA-Z][\w.\-]*', str(label)):
        return f' as {label}'
    return f' as "{label}"'


def _render_expense(spec: dict) -> str:
    """Render a ``pnl`` gross-expense clause, or ``''`` if absent.

    ``expense_spec`` is a list of ``(label, [(basis, value), ...])`` **groups**
    (legacy flat / bare shapes are accepted via :func:`_normalize_expense_groups`).
    Within a group, terms join with ``and``; the group's optional ``as`` label
    follows; juxtaposed groups are space-separated. Each term renders as
    ``<amount> fixed expenses`` / ``<fraction> premium|loss expenses``.
    """
    from ._pnl_builders import _normalize_expense_groups
    groups = _normalize_expense_groups(spec.get('expense_spec'))
    if not groups:
        return ''
    rendered = []
    for label, terms in groups:
        body = ' and '.join(
            f'{_fmt_num(value)} {basis} expenses' for basis, value in terms)
        rendered.append(body + _render_label(label))
    return ' '.join(rendered)


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

def _render_tweedie(spec: dict) -> str:
    """Render the ``tweedie`` clause, or ``''`` when the spec is not one.

    Inverts ``agg_body_tweedie``. The transformer expands a ``tweedie`` clause
    into its compound-Poisson-gamma equivalent and the engine sees nothing else,
    so the clause can only be recovered from the ``_tweedie`` provenance key the
    transformer records beside the expansion.

    Notes
    -----
    Provenance, deliberately, not recognition. Every unlayered poisson x gamma
    aggregate *is* a Tweedie, so this could be reconstructed from the engine
    parameters alone; doing that would rewrite ``K.Tweedie0`` and
    ``K.Tweedie1``, which their author wrote the long way on purpose, into a
    spelling they did not choose. The unparser is the inverse of the parser, not
    a canonicalizer with opinions. ``Aggregate.as_tweedie`` is where recognition
    lives, because there the generosity costs nothing.
    """
    tw = spec.get('_tweedie')
    if not tw:
        return ''
    p, mean, dispersion = tw
    return f'tweedie {_fmt_num(p)} {_fmt_num(mean)} {_fmt_num(dispersion)}'


def _render_agg(name: str, spec: dict, trailer: bool = True) -> _Block:
    """Render an ordinary loss aggregate (``agg NAME ...``).

    Clause order mirrors ``agg_out_full`` / ``agg_out_dfreq``: exposure (or
    ``dfreq``), layers, severity, occurrence reinsurance, frequency (omitted for
    ``dfreq``), aggregate reinsurance, ``approximate``, trailer. Returns a
    :class:`_Block` (head ``agg NAME``, the clauses its children) so it renders
    terse on one line or spread with each clause on its own indented line.

    A ``tweedie`` body short-circuits all of that: the clause is the whole body,
    the grammar gives it no reinsurance or layer slots, and its expansion is
    exactly what must *not* be rendered.

    ``trailer=False`` drops the ``note{...}`` / ``tags{...}`` / ``hints{...}``
    / ``doc{{{...}}}`` tail; each surviving clause is its own child, so spread
    puts it on its own line.
    """
    tweedie = _render_tweedie(spec)
    if tweedie:
        return _Block(f'agg {name}{_render_label(spec.get("label"))}',
                      [tweedie, *_render_trailer(spec, trailer)])
    return _Block(f'agg {name}{_render_label(spec.get("label"))}', [
        _render_exposure(spec),
        _render_layers(spec),
        _render_sev_clause(spec),
        _render_reins(spec, 'occurrence', 'occ_reins', 'occ_kind'),
        _render_freq(spec) or _render_wait(spec),
        _render_reins(spec, 'aggregate', 'agg_reins', 'agg_kind'),
        _render_approx(spec),
        _render_orientation(spec),
        *_render_trailer(spec, trailer),
    ])


def _render_pnl(name: str, spec: dict, kind: str = 'pnl',
                trailer: bool = True) -> _Block:
    """Render a profit-and-loss aggregate (``pnl NAME <premium> premium less ...``).

    Inverts ``pnl_out_*`` / ``_attach_pnl``. The premium is ``consideration``;
    the exposure head after ``less`` is ``claims`` (``exp_en``), ``lr``
    (``exp_lr``, the bare loss-ratio form that binds to the premium) or ``loss``
    (``exp_el``).

    Under the current (engine-wrapped) grammar a P&L declares a complete backing
    ``agg`` engine::

        pnl NAME <premium> premium less agg NAME_e <exposure> <body> [less <exp>]

    Returns a :class:`_Block` headed by ``pnl NAME`` alone. The premium head is
    its own child, and each ``less`` heads a sub-block over what it takes away:
    the loss engine (an ``agg NAME_e`` block sharing the ordinary
    :func:`_render_exposure` / severity / reinsurance / frequency /
    ``approximate`` renderers, or a ``port.NAME`` reference), then any
    gross-expense clause. So ``spread`` reads as the subtraction it is::

        pnl NAME
          <premium> premium
          less
            agg NAME_e
              <exposure>
              ...
          less
            <expense>

    ``terse`` is unaffected, and byte-identical to what it has always been:
    :func:`_render_terse` space-joins a head back onto its children, so
    ``pnl NAME <premium> premium less agg NAME_e ... less <expense>`` is what
    both the nesting and the old flat head flatten to. That is what keeps
    :func:`spec_to_decl` and the ``to_agg`` exporter untouched by the layout.

    The engine name round-trips when the source carried one
    (``spec['engine_name']``); otherwise a synthetic ``NAME_e`` is used. Either
    way it is cosmetic --- discarded at build --- but must stay a valid
    identifier.
    """
    from .parser import INHERIT_PREMIUM
    retro = spec.get('retro_terms')
    consideration = spec.get('consideration')
    if retro is not None:
        premium_head = f'retro {_render_collar(retro)} premium'
    elif consideration is INHERIT_PREMIUM:
        # ``inherit premium`` -- copy the engine's technical premium; the sentinel
        # never renders as a number.
        premium_head = 'inherit premium'
    else:
        premium_head = f'{_fmt_seq(consideration)} premium'
    premium_head += _render_label(spec.get('consideration_label'))
    obj_label = _render_label(spec.get('label'))

    port_engine = spec.get('_engine_port')
    port_spec = spec.get('_engine_port_spec')
    if port_engine is not None:
        # A ``port.NAME``-sourced P&L: the source referenced a stored portfolio,
        # so it renders back as the reference it was, not as the units the
        # parser resolved it to.
        engine = f'port.{port_engine}'
    elif port_spec is not None:
        # An inline portfolio engine: the units written out, the twin of the
        # inline agg below and self-contained for the same reason. No trailer,
        # which the wrapping pnl owns, so this reuses the unit rendering of
        # :func:`_render_port` rather than the whole thing.
        engine = _Block(
            f'port {port_spec.get("name", f"{name}_e")}'
            f'{_render_label(port_spec.get("label"))}',
            [_render_agg_or_pnl(sub_kind, sub_name, sub_spec, trailer)
             for sub_kind, sub_name, sub_spec in port_spec['spec']])
    else:
        # the loss leg is a complete ``agg NAME_e`` engine (new grammar); its
        # exposure head (claims / loss / ``premium at lr`` / dfreq) and body reuse
        # the ordinary agg renderers, so every exposure form round-trips through
        # one code path. (An ``agg.NAME``-sourced P&L renders as the equivalent
        # inline engine -- the merged loss structure round-trips identically.)
        # The engine's own ``as`` label names the P&L's loss leg
        # (dev/plan-yapnl.md label plumbing), so it must round-trip.
        engine_label = _render_label(spec.get('engine_label'))
        engine_name = spec.get('engine_name', f'{name}_e')
        # A tweedie engine short-circuits the body the same way `_render_agg`
        # does; `_merge_engine_spec` copies `_tweedie` into the pnl spec because
        # it is structure, not metadata, so it is not on the skip list there.
        tweedie = _render_tweedie(spec)
        engine = _Block(f'agg {engine_name}{engine_label}', [tweedie] if tweedie else [
            _render_exposure(spec),
            _render_layers(spec),
            _render_sev_clause(spec),
            _render_reins(spec, 'occurrence', 'occ_reins', 'occ_kind'),
            _render_freq(spec) or _render_wait(spec),
            _render_reins(spec, 'aggregate', 'agg_reins', 'agg_kind'),
            _render_approx(spec),
        ])
    expense = _render_expense(spec)
    keyword = 'xpnl' if kind == 'xpnl' else 'pnl'
    return _Block(f'{keyword} {name}{obj_label}', [
        premium_head,
        _Block('less', [engine]),
        _Block('less', [expense]) if expense else '',
        _render_peel(spec),
        *_render_trailer(spec, trailer),
    ])


def _render_agg_or_pnl(kind: str, name: str, spec: dict,
                       trailer: bool = True) -> _Block:
    """Render an aggregate, dispatching to ``pnl`` when a consideration is set.

    A ``pnl`` declaration transforms to ``("pnl", name, spec)`` with a
    ``consideration`` key (set by ``_attach_pnl``); an ``agg`` has neither.
    """
    if kind in ('pnl', 'xpnl') or 'consideration' in spec:
        return _render_pnl(name, spec, kind=kind, trailer=trailer)
    return _render_agg(name, spec, trailer)


def _render_sev_out(name: str, spec: dict, trailer: bool = True) -> _Block:
    """Render a standalone severity definition (``sev NAME ...``).

    The declaration is the block head (a severity has no clause nesting worth
    spreading); the trailer clauses are its children, so ``spread`` lands each
    on its own indented line while ``terse`` flattens to the historical single
    line.
    """
    body = _render_dsev(spec) if _is_dsev(spec) else _render_dist(spec)
    head = _join([f'sev {name}{_render_label(spec.get("label"))}', body])
    return _Block(head, _render_trailer(spec, trailer))


def _render_port(name: str, spec: dict, trailer: bool = True) -> _Block:
    """Render a portfolio: ``port NAME``, its trailer, then one unit per line.

    Sub-units are ``agg`` / ``pnl`` tuples; each becomes a child :class:`_Block`.
    The returned block is marked ``tab=True`` so its *terse* form is the
    historical tab-indented layout (head line, then ``\\t`` + each child on one
    line); spread indents the children two spaces and lets the units' clauses
    spread one level deeper. Either way the preprocessor folds the indented
    continuation back into one logical statement on re-parse.

    The portfolio's own trailer clauses come **before** the units, which is
    where the grammar binds them: a trailer written after the last unit binds
    to that unit instead, because the portfolio's trailer slot closes before
    the units are read.
    """
    head = f'port {name}{_render_label(spec.get("label"))}'
    units = [_render_agg_or_pnl(kind, sub_name, sub_spec, trailer)
             for kind, sub_name, sub_spec in spec['spec']]
    return _Block(head, [*_render_trailer(spec, trailer), *units], tab=True)


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


def _render_bvagg(name: str, spec: dict, trailer: bool = True) -> _Block:
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
            *_render_trailer(spec, trailer),
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
            *_render_trailer(spec, trailer),
        ])

    if spec.get('mode') == 'netceded':
        # the keyword is the (x, y) view pair, names x-then-y
        _kw_for = {('net', 'ceded'): 'netceded', ('gross', 'ceded'): 'grossceded',
                   ('gross', 'net'): 'grossnet'}
        views = tuple(spec.get('nc_views') or ('net', 'ceded'))
        kw = _kw_for.get(views, 'netceded')
        kind, sub_name, sub_spec = spec['units'][0]
        unit = _render_agg_or_pnl(kind, sub_name, sub_spec, trailer)
        # the view keyword is the block head and the whole unit its single
        # child, so spread puts ``grossceded`` on its own line with ``agg NAME``
        # indented under it and the unit's clauses one level deeper again;
        # terse space-joins the two back into the historical single line
        return _Block(kw, [unit])

    components = [_render_agg_or_pnl(k, n, s, trailer) for k, n, s in spec['units']]
    return _Block(f'bivariate {name}', [
        _render_exposure(spec),
        *components,
        _render_copula(spec['copula']),
        _render_freq(spec),
        *_render_trailer(spec, trailer),
    ])


def _render_distortion(name: str, spec: dict, trailer: bool = True) -> _Block:
    """Render a distortion definition (``distortion NAME kind n1 n2 ...``).

    Inverts ``Distortion.decl_spec``: the flat number list is recovered from the
    kind's ``decl_params`` ordering. The ``minimum`` / ``mixture`` combinators
    take distortion *references* whose names are not retained on the constructed
    children, so they cannot round-trip and raise here.

    Since 1.0.0a157 a distortion carries the same trailer as every other
    statement, so ``trailer`` is honoured rather than ignored. Like ``sev``,
    the declaration is the block head and the trailer clauses are its children.
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
    return _Block(f'distortion {name} {kind} {numbers}',
                  _render_trailer(spec, trailer))


_KIND_RENDERERS = {
    'agg': lambda name, spec, trailer: _render_agg_or_pnl('agg', name, spec, trailer),
    'pnl': lambda name, spec, trailer: _render_agg_or_pnl('pnl', name, spec, trailer),
    'xpnl': lambda name, spec, trailer: _render_agg_or_pnl('xpnl', name, spec, trailer),
    'sev': _render_sev_out,
    'port': _render_port,
    'bvagg': _render_bvagg,
    'distortion': _render_distortion,
}


def _spec_to_node(spec: dict, kind: str = 'agg', name: str | None = None,
                  trailer: bool = True):
    """Dispatch a spec to its kind renderer, returning a ``_Block`` or ``str``.

    The shared front half of :func:`spec_to_decl` (terse) and
    :func:`format_program` (either layout): it resolves the name default and the
    kind renderer, but does *not* flatten --- the caller picks the layout walker.
    Every renderer returns a :class:`_Block`; ``sev`` and ``distortion`` have no
    clause nesting worth spreading, so their whole declaration is the head and
    only the trailer clauses are children.

    ``trailer=False`` drops the whole ``note`` / ``tags`` / ``hints`` / ``doc``
    trailer throughout the tree, including on a portfolio's units and a
    bivariate's components.

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
    return renderer(name, spec, trailer)


def spec_to_decl(spec: dict, kind: str = 'agg', name: str | None = None) -> str:
    """Render a parsed spec back to canonical DecL text (the unparser).

    The structural inverse of :class:`aggregate.parser.UnderwritingParser`.

    Parameters
    ----------
    spec : dict
        A **raw transformer spec** --- ``parsed.spec`` or a recipe's
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


def _render_statement(underwriter, statement: str, trailer: bool = True):
    """Parse one statement and return its render node, or the text verbatim.

    Returns a :class:`_Block` / ``str`` node (which the caller renders in the
    requested layout). The verbatim ``str`` fallback keeps :func:`format_program`
    (hence ``pprogram``) from raising when a program references a builtin that
    the default underwriter cannot resolve --- e.g. a ``sev.X`` defined only in a
    custom recipe base. A plain ``str`` renders identically in both layouts.

    Note the fallback is *verbatim*, so it also escapes ``trailer=False``: a
    statement that cannot be parsed keeps whatever trailer its source text had.
    """
    try:
        kind, name, spec = underwriter.parser.parse(statement)
        return _spec_to_node(spec, kind, name, trailer)
    except Exception:
        # Display fallback: a parse error (malformed text) or an unresolved
        # builtin reference (defined only in a custom underwriter, or a lark
        # VisitError wrapping the KeyError) must not crash pprogram.
        return statement


def format_program(spec_or_text, *, fmt: str = 'text', layout: str = 'spread',
                   trailer=False) -> str:
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
    trailer : bool or iterable of str, default False
        Emit the ``note{...}`` / ``tags{...}`` / ``hints{...}`` /
        ``doc{{{...}}}`` trailer. ``False`` (the default) gives the bare
        declaration, which is what you almost always want when formatting: the
        math and the insurance, not the metadata around it. ``True`` emits all
        four; an iterable names the ones to keep, e.g. ``('hints',)`` for the
        clauses that actually change how the object builds.

        In ``spread`` each surviving clause takes its own line at the
        statement's indentation level; in ``terse`` they space-join onto the
        one line.

        Unlike ``fmt`` and ``layout``, this axis is **not** round-trip safe:
        the output re-parses to the same spec minus whatever was suppressed.
        Use :func:`spec_to_decl` (which always emits the full trailer) when the
        result has to reload identically --- that is what ``to_agg`` does. The
        semantic ``!`` markers (unconditional severity, the zero-modified mean
        pin, defective ``dwait``) are clause syntax, not trailer, and are never
        affected.
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
        # default underwriter carries the configured recipe base so most
        # builtin references in the text resolve.
        from .underwriter import build as _build
        nodes = [_render_statement(_build, line, trailer)
                 for line in _split_statements(text)]
    elif isinstance(spec_or_text, tuple):
        kind, name, spec = spec_or_text
        nodes = [_spec_to_node(spec, kind, name, trailer)]
    else:
        nodes = [_spec_to_node(spec_or_text, trailer=trailer)]

    # Join top-level statements with a blank line: a lone '\n' is not a statement
    # separator (the preprocessor folds it into the preceding statement), so the
    # multi-statement path must use '\n\n' to re-parse correctly.
    rendered = '\n\n'.join(render(n) for n in nodes)
    return _colorize(rendered, fmt)
