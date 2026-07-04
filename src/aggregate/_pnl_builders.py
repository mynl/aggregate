r"""The insurance builders over the generic P&L kernel.

All insurance semantics live here (and in the DecL grammar); the kernel
(:mod:`aggregate._pnl`) never sees the words gross / ceded / net. The builders
translate insurance programs -- premiums, losses, expenses, cessions with
their economics -- into a **source** plus signed **groups**
(``dev/plan-yapnl.md``):

* :func:`build_plain_pnl` -- a plain book: one ``sell`` group (premium /
  loss / expense legs) over the engine's own density.
* :func:`build_gcn_pnl` -- a reinsurance program as a per-atom **group
  ledger**: an aggregate-only cession is a real ``buy`` group over the gross
  marginal; an occurrence guaranteed-cost program books over the net-of-occ
  marginal (the deepest source its cash flows are jointly measurable on) with
  the occ ceded premium as a constant leg and any aggregate cession as a real
  ``f(x)`` group. The gross perspective of an occurrence program lives on
  ``xpnl``.
* :func:`build_xpnl_stack` -- the ``xpnl`` marginal perspective stack (the
  onion peel across ``reins_density_df`` marginals: gross / net-occ /
  net-agg, each perspective its own one-group :class:`PnL`), assembled by
  :func:`aggregate._pnl.stack_marginal_pnls`.

The expense resolvers (:func:`resolve_expense` / :func:`_resolve_expense_split`)
relocated here from ``_pnl.py`` unchanged: ``_resolve_expense_split`` is the
only resolver feeding legs (loss-basis expense is stochastic ``rate * x``
wherever the loss is on-source); ``resolve_expense`` survives for scalar needs
(the analyses' deterministic economics) but never feeds a leg.
"""
from __future__ import annotations

import numpy as np

from ._pnl import Leg, Group, PnL, stack_marginal_pnls

__all__ = ['build_plain_pnl', 'build_gcn_pnl', 'build_xpnl_stack',
           'build_variable_pnl', 'build_reinstatement_pnl', 'resolve_expense']


# ----------------------------------------------------------------------
# Expense resolution (relocated from _pnl.py, unchanged semantics)
# ----------------------------------------------------------------------
def _normalize_expense_groups(expense_spec):
    """Normalize a DecL ``expense`` spec to grouped form.

    Returns ``[(label, [(basis, value), ...]), ...]`` -- a list of expense
    **groups** (one obligation leg each). ``label`` is the group's ``as`` display
    label or ``None``. Accepts three input shapes for backward compatibility:

    * the **grouped** parser form ``[(label, [terms]), ...]`` (each element's
      second item is a list of terms);
    * a **legacy flat** list of ``(basis, value)`` terms -- one implicit unlabeled
      group (each element's second item is a number);
    * a **bare** single ``(basis, value)`` tuple -- one implicit unlabeled group.

    ``None`` / empty -> ``[]``. A group is distinguished from a term structurally
    (a group's payload is a list; a term's value is a number), so a label reading
    ``"premium"`` cannot be mistaken for a basis.
    """
    if not expense_spec:
        return []
    first = expense_spec[0]
    if isinstance(first, str):                       # bare ('basis', value)
        return [(None, [tuple(expense_spec)])]
    if len(first) == 2 and not isinstance(first[1], (list, tuple)):
        # legacy flat list of (basis, value) terms -> one implicit group
        return [(None, [tuple(t) for t in expense_spec])]
    return [(lbl, [tuple(t) for t in terms]) for lbl, terms in expense_spec]


def _default_group_name(terms):
    """Default reported name for an unlabeled expense group.

    A single-basis group is ``'<basis> expense'`` (e.g. ``'premium expense'``); a
    mixed-basis group is the generic ``'expense'``. See dev/done/plan-decl-labels.md.
    """
    bases = {basis for basis, _ in terms}
    return f'{next(iter(bases))} expense' if len(bases) == 1 else 'expense'


def _expected_gross_loss(agg):
    """The expected **gross** loss E[X] read off the engine's exact density.

    Works for an :class:`Aggregate` (reads the gross marginal when reinsurance
    is present) and a :class:`Portfolio` (its net-net total).
    """
    if getattr(agg, 'occ_reins', None) is not None \
            or getattr(agg, 'agg_reins', None) is not None:
        rd = agg.reins_density_df
        return float((rd['loss'].to_numpy() * rd['p_agg_gross'].to_numpy()).sum())
    dd = agg.density_df
    x = (dd['loss'].to_numpy(dtype=float) if 'loss' in dd.columns
         else dd.index.to_numpy(dtype=float))
    return float((x * dd['p_total'].to_numpy(dtype=float)).sum())


def resolve_expense(agg, expense_spec, gross_premium):
    """The gross expense ``E_G`` as a scalar from a DecL ``expense`` spec.

    Sums **all** terms across **all** groups. ``expense_spec`` may be the
    grouped form, a legacy flat list of ``(basis, value)`` terms, or a bare
    ``(basis, value)`` tuple (see :func:`_normalize_expense_groups`). For each term
    ``'fixed'`` is a currency amount, ``'premium'`` a fraction of ``gross_premium``,
    ``'loss'`` a fraction of the **expected gross loss** (deterministic).
    ``None`` / empty -> ``0``.

    Scalar needs only (the analyses' deterministic economics); this never
    feeds a leg -- :func:`_resolve_expense_split` is the only leg resolver.
    """
    total = 0.0
    for _label, terms in _normalize_expense_groups(expense_spec):
        for basis, val in terms:
            if basis == 'fixed':
                total += float(val)
            elif basis == 'premium':
                total += float(val) * float(gross_premium)
            elif basis == 'loss':
                total += float(val) * _expected_gross_loss(agg)
            else:
                raise ValueError(f'unknown expense basis {basis!r}')
    return float(total)


def _resolve_expense_split(agg, expense_spec, gross_premium):
    """Split a DecL ``expense`` spec into per-group ``(name, scalar, loss_rate)``.

    Returns a **list** with one entry per expense group (``and``-joined terms make
    one group; juxtaposed groups stay separate). Within a group ``fixed`` and
    ``premium`` terms are **deterministic** -- a currency amount, or a fraction of
    the fixed gross premium -- and fold into ``scalar``; a ``loss`` term is **loss
    adjustment expense**, a fraction of the *actual* loss, returned as ``loss_rate``
    to be applied per atom (``rate * x``) rather than collapsed to ``rate *
    E[loss]``. The caller builds one obligation leg per group as ``loss_rate * x +
    scalar`` -- a scaled-loss distribution, not a point mass.

    ``name`` is the group's ``as`` label if given, else a basis-derived default
    (``'premium expense'`` / ``'loss expense'`` / ``'fixed expense'``, or
    ``'expense'`` for a mixed group); a lone unlabeled group keeps the historical
    ``'expense'`` leg name. ``None`` / empty -> ``[]``.
    """
    groups = _normalize_expense_groups(expense_spec)
    out = []
    single_unlabeled = len(groups) == 1 and groups[0][0] is None
    for label, terms in groups:
        scalar = 0.0
        loss_rate = 0.0
        for basis, val in terms:
            if basis == 'fixed':
                scalar += float(val)
            elif basis == 'premium':
                scalar += float(val) * float(gross_premium)
            elif basis == 'loss':
                loss_rate += float(val)
            else:
                raise ValueError(f'unknown expense basis {basis!r}')
        if label is not None:
            name = label
        elif single_unlabeled:
            # preserve the historical single-leg name for back-compat
            name = 'expense'
        else:
            name = _default_group_name(terms)
        out.append((name, scalar, loss_rate))
    return out


def _expense_legs(agg, expense_spec, gross_premium, *, on_source_loss=True,
                  taken=()):
    """One obligation :class:`Leg` per expense group.

    A ``loss``-basis (LAE) term scales with the *actual* loss when the loss is
    an on-source observable (``on_source_loss=True`` -- the leg is
    ``rate * x + scalar``, stochastic); when the exhibit's source is a *net*
    marginal the gross loss is not measurable there, so the loss-basis part
    degrades to the deterministic ``rate * E[gross loss]``. Colliding default
    names (vs ``taken`` and each other) are de-duplicated with a ``(n)``
    suffix.
    """
    legs = []
    used = set(taken)
    for gname, scalar_exp, loss_rate in _resolve_expense_split(
            agg, expense_spec, gross_premium):
        key, n = gname, 2
        while key in used:
            key = f'{gname} ({n})'
            n += 1
        used.add(key)
        if loss_rate and on_source_loss:
            legs.append(Leg(key, lambda x, r=loss_rate, s=scalar_exp: r * x + s))
        elif loss_rate:
            legs.append(Leg(key, loss_rate * _expected_gross_loss(agg)
                            + scalar_exp))
        elif scalar_exp:
            legs.append(Leg(key, scalar_exp))
    return legs


# ----------------------------------------------------------------------
# Reinsurance economics helpers (relocated from _pnl.py)
# ----------------------------------------------------------------------
def _gcn_magnitudes(agg, gross, ceded, net, expense_spec, econ):
    """Per-perspective premium / expense magnitudes for the marginal stack.

    Premium per perspective (a DecL ceded-premium clause gives a per-side
    split; a scalar API GCN books its single ceded amount on the side
    present), gross expense on the gross leg, a commission credit on each
    cession, so net expense ``= E_G - C_occ - C_agg`` and means add across the
    split. Feeds :func:`build_xpnl_stack`.
    """
    has_occ = agg.occ_reins is not None
    has_agg = agg.agg_reins is not None
    p_gross = float(gross)
    if econ is not None:
        pc_occ = float(econ.get('pc_occ', 0.0))
        pc_agg = float(econ.get('pc_agg', 0.0))
    else:
        ceded_total = float(ceded)
        pc_agg = ceded_total if has_agg else 0.0
        pc_occ = ceded_total if (has_occ and not has_agg) else 0.0
    prem_mag = {
        'gross': p_gross,
        'ceded_occ': pc_occ, 'net_occ': p_gross - pc_occ,
        'ceded_agg': pc_agg, 'net_agg': p_gross - pc_occ - pc_agg,
    }
    e_gross = resolve_expense(agg, expense_spec, p_gross)
    c_occ = float(econ.get('c_occ', 0.0)) if econ is not None else 0.0
    c_agg = float(econ.get('c_agg', 0.0)) if econ is not None else 0.0
    exp_mag = {
        'gross': e_gross,
        'ceded_occ': c_occ, 'net_occ': e_gross - c_occ,
        'ceded_agg': c_agg, 'net_agg': e_gross - c_occ - c_agg,
    }
    final_net = 'net_agg' if has_agg else 'net_occ'
    prem_mag[final_net] = float(net) if net is not None \
        else p_gross - pc_occ - (pc_agg if has_agg else 0.0)
    return prem_mag, exp_mag, has_occ, has_agg


#: Each waterfall perspective reads its own exact aggregate **marginal** from
#: ``Aggregate.reins_density_df``.
_GCN_LOSS_MARGINAL = {
    'gross': 'p_agg_gross',
    'ceded_occ': 'p_agg_ceded_occ',
    'net_occ': 'p_agg_net_occ',
    'ceded_agg': 'p_agg_ceded',
    'net_agg': 'p_agg_net',
}
_GCN_CEDED = frozenset({'ceded_occ', 'ceded_agg'})


def _first_reins_label(agg, site):
    """The lowest-indexed labeled layer's label on a cession basis, or ``None``.

    A basis's cessions consolidate into a single group / perspective, so the
    first labeled layer names it. Reads the sparse ``{layer_index: label}``
    dict pooled into the engine's ``label_map`` (``agg.labels.occ_reins`` /
    ``.agg_reins``); ``site`` is ``'occ_reins'`` or ``'agg_reins'``.
    """
    d = getattr(agg, 'label_map', None) or {}
    d = d.get(site) or {}
    return d[min(d)] if d else None


def _ledger_economics(agg, gross, ceded, econ):
    """Resolve the per-side cession economics for the group ledger.

    Returns ``(p_gross, pc_occ, pc_agg, c_occ, c_agg)``. A DecL economics dict
    gives the per-side split; the scalar API books its single ceded amount on
    the side present (aggregate when an aggregate cover exists, else
    occurrence) with no commission.
    """
    has_occ = agg.occ_reins is not None
    has_agg = agg.agg_reins is not None
    p_gross = float(gross)
    if econ is not None:
        return (p_gross, float(econ.get('pc_occ', 0.0)),
                float(econ.get('pc_agg', 0.0)),
                float(econ.get('c_occ', 0.0)), float(econ.get('c_agg', 0.0)))
    ceded_total = float(ceded)
    pc_agg = ceded_total if has_agg else 0.0
    pc_occ = ceded_total if (has_occ and not has_agg) else 0.0
    return p_gross, pc_occ, pc_agg, 0.0, 0.0


def _marginal_gd(agg, perspective):
    """One ``reins_density_df`` marginal as an irregular :class:`GridDistribution`."""
    from ._grid_distribution import GridDistribution
    rd = agg.reins_density_df
    return GridDistribution(
        rd['loss'].to_numpy(dtype=float),
        rd[_GCN_LOSS_MARGINAL[perspective]].to_numpy(dtype=float), bs=None,
        name=perspective)


# ----------------------------------------------------------------------
# The builders
# ----------------------------------------------------------------------
def build_plain_pnl(engine, *, consideration, consideration_label=None,
                    loss_label=None, expense_spec=None, name=None,
                    label=None):
    """A plain book as a one-group :class:`PnL` over the engine's own density.

    Parameters
    ----------
    engine : Aggregate, Portfolio, or (values, probs)
        The stochastic engine; its density is the source (for a
        reinsurance-bearing aggregate that is the net density -- what comes
        out of the engine).
    consideration : float or callable
        The book premium: a constant, or a loss-sensitive ``f(x)``.
    consideration_label : str, optional
        The premium leg label (the DecL premium ``as`` clause); default
        ``'consideration'``.
    loss_label : str, optional
        The loss leg label (the engine ``as`` clause); default ``'loss'``.
    expense_spec : list or tuple, optional
        A DecL ``expense`` spec; one obligation leg per group, loss-basis
        terms stochastic (``rate * x``).
    name, label : str, optional
        The P&L handle and its optional human label.

    Returns
    -------
    PnL
    """
    if callable(consideration):
        cons = consideration
        gp = float(np.sum(np.asarray(consideration(0.0), dtype=float)))
    else:
        cons = float(np.sum(np.asarray(consideration, dtype=float)))
        gp = cons
    loss_key = loss_label or 'loss'
    obligation = [Leg(loss_key, lambda x: x)]
    obligation += _expense_legs(engine, expense_spec, gp,
                                on_source_loss=True, taken={loss_key})
    return PnL(name=name, role='sell', source=engine,
               consideration=[Leg(consideration_label or 'consideration',
                                  cons)],
               obligation=obligation, result_name='margin',
               label=label)


def build_gcn_pnl(agg, *, gross, ceded, gcn_economics=None, expense_spec=None,
                  consideration_label=None, loss_label=None, name=None,
                  label=None):
    """A reinsurance program as a per-atom **group ledger** :class:`PnL`.

    Source selection (``dev/plan-yapnl.md``): an aggregate-only program books
    over the **gross** 1-D marginal -- gross ``sell`` group plus the aggregate
    cession as a real ``buy`` group (`f(x)` recovery), an exact per-atom
    waterfall. An occurrence program (guaranteed cost, with or without an
    aggregate cover) books over the **net-of-occurrence** marginal -- the
    deepest source its cash flows are jointly measurable on: the occurrence
    recovery is not a function of that observable, so the occ cession enters
    through its deterministic ceded premium (a constant leg) and the occ
    cession's risk transfer is visible through ``xpnl``
    (:func:`build_xpnl_stack`); an aggregate cover attaches on the net-of-occ
    loss and stays a real group.

    The resolved economics ride on the returned P&L as ``pnl.economics``.

    Returns
    -------
    PnL
    """
    from . import _reinsurance
    rd = agg.reins_density_df
    if rd is None:
        raise ValueError(
            'the Gross/Ceded/Net view requires reinsurance on the risky leg; '
            'the aggregate carries no occurrence / aggregate treaty.')
    has_occ = agg.occ_reins is not None
    has_agg = agg.agg_reins is not None
    p_gross, pc_occ, pc_agg, c_occ, c_agg = _ledger_economics(
        agg, gross, ceded, gcn_economics)
    occ_base = _first_reins_label(agg, 'occ_reins') \
        or 'ceded occ'
    agg_base = _first_reins_label(agg, 'agg_reins') \
        or 'ceded agg'
    prem_key = consideration_label or 'premium'
    loss_key = loss_label or 'loss'

    groups = []
    if has_occ:
        # net-of-occ marginal: book the parts measurable there -- gross
        # premium in, net-of-occ loss out, the occ cession's deterministic
        # premium / commission as constants.
        source = _marginal_gd(agg, 'net_occ')
        cons = [Leg(prem_key, p_gross)]
        if c_occ:
            cons.append(Leg(f'{occ_base} commission', c_occ))
        obl = [Leg(f'{loss_key} (net occ)', lambda x: x),
               Leg(f'{occ_base} premium', pc_occ)]
        obl += _expense_legs(agg, expense_spec, p_gross, on_source_loss=False,
                             taken={r.label for r in cons + obl})
        groups.append(Group('net occ', 'sell', cons, obl))
    else:
        # aggregate-only: the true gross marginal, an exact per-atom waterfall.
        source = _marginal_gd(agg, 'gross')
        cons = [Leg(prem_key, p_gross)]
        obl = [Leg(loss_key, lambda x: x)]
        obl += _expense_legs(agg, expense_spec, p_gross, on_source_loss=True,
                             taken={r.label for r in cons + obl})
        groups.append(Group('gross', 'sell', cons, obl))
    if has_agg:
        # the aggregate cover attaches on the source observable (net-of-occ
        # when an occurrence stage inures, else gross).
        agg_ceder, _netter = _reinsurance.make_ceder_netter(agg.agg_reins)
        obl = [Leg(f'{agg_base} recovery', agg_ceder)]
        if c_agg:
            obl.append(Leg(f'{agg_base} commission', c_agg))
        groups.append(Group(agg_base, 'buy',
                            [Leg(f'{agg_base} premium', pc_agg)], obl))
    pnl = PnL(name=name or agg.name, source=source, groups=groups,
              result_name='margin', label=label)
    #: the resolved cession economics (a DecL clause's per-side split, or the
    #: scalar API's single amounts) -- the observable of the DecL premium /
    #: commission resolution (``deposit`` / ``rol`` / ``rate`` / ``cede``).
    pnl.economics = dict(gcn_economics) if gcn_economics is not None \
        else {'gross': float(gross), 'ceded': float(ceded)}
    return pnl


def build_xpnl_stack(agg, *, gross, ceded, gcn_economics=None,
                     expense_spec=None, name=None):
    """The ``xpnl`` marginal perspective stack (the guaranteed-cost onion).

    No joint exists (the cessions are not functions of one observable), so
    each waterfall perspective is a one-group :class:`PnL` over its **own**
    exact aggregate marginal from ``Aggregate.reins_density_df``, and the
    stack is assembled by :func:`aggregate._pnl.stack_marginal_pnls`: means
    add across the perspective rows; SDs / percentiles are per row.

    Returns
    -------
    pandas.DataFrame
        Rows = perspectives plus the inuring impact deltas; columns the
        standard metric set.
    """
    rd = agg.reins_density_df
    if rd is None:
        raise ValueError(
            "'xpnl' requires reinsurance on the wrapped engine; the aggregate "
            'carries no occurrence / aggregate treaty.')
    prem_mag, exp_mag, has_occ, has_agg = _gcn_magnitudes(
        agg, gross, ceded, None, expense_spec, gcn_economics)
    both = has_occ and has_agg

    def persp_pnl(persp):
        gd = _marginal_gd(agg, persp)
        prem, exp = prem_mag[persp], exp_mag[persp]
        if persp in _GCN_CEDED:
            obl = [Leg('recovery', lambda x: x)]
            if exp:
                obl.append(Leg('commission', exp))
            return PnL(name=persp, role='buy', source=gd,
                       consideration=[Leg('ceded premium', prem)],
                       obligation=obl, result_name=persp)
        obl = [Leg('loss', lambda x: x)]
        if exp:
            obl.append(Leg('expense', exp))
        return PnL(name=persp, role='sell', source=gd,
                   consideration=[Leg('premium', prem)],
                   obligation=obl, result_name=persp)

    # A reins-clause ``as`` label (first labeled layer on each basis) names
    # that basis's cession row; the ``net`` rows keep their structural names.
    occ_label = _first_reins_label(agg, 'occ_reins')
    agg_label = _first_reins_label(agg, 'agg_reins')
    perspectives = [('gross', persp_pnl('gross'))]
    impacts = []
    if has_occ:
        q = ' occ' if both else ''
        perspectives += [(occ_label or f'ceded{q}', persp_pnl('ceded_occ')),
                         (f'net{q}', persp_pnl('net_occ'))]
        if both:
            impacts.append(('occ impact', f'net{q}', 'gross'))
    if has_agg:
        q = ' agg' if both else ''
        perspectives += [(agg_label or f'ceded{q}', persp_pnl('ceded_agg')),
                         (f'net{q}', persp_pnl('net_agg'))]
        if both:
            impacts.append(('agg impact', f'net{q}', 'net occ'))
    impacts.append(('impact', perspectives[-1][0], 'gross'))
    return stack_marginal_pnls(perspectives, impacts=impacts, name=name)


# ----------------------------------------------------------------------
# Variable-rating / reinstatement feature builders
# ([Builders-Variable-Features]: a feature never changes the machinery --
#  it changes one leg's function)
# ----------------------------------------------------------------------
def build_variable_pnl(agg, *, expense_spec=None, consideration_label=None,
                       loss_label=None, name=None, label=None):
    """A retro / swing / slide / pc / corridor program as a group ledger.

    Reads the feature attached to ``agg`` by the underwriter
    (``variable_terms`` / ``variable_layer`` / ``variable_gross_premium`` /
    ``variable_ceded_premium`` / ``variable_commission``) and books the
    cash-flow parts over the **gross** 1-D density:

    * **retro** (account-level, no layer): one ``sell`` group whose premium
      leg is the stochastic ``terms.phi`` of the account loss -- the same
      ledger shape as a plain book (the acceptance pair).
    * **swing / slide / pc / corridor** (one decorated aggregate layer): the
      gross ``sell`` group plus a real cession ``buy`` group, with exactly
      **one** leg swapped for the feature's ``terms.phi``-driven map
      (stochastic ceded premium; sliding / profit commission; the
      corridor-adjusted recovery).

    Expense groups book on the gross group (loss-basis LAE stochastic
    ``rate * x`` -- the loss is on-source). Attach the drill-down analysis
    afterwards (``pnl.analysis``).

    Returns
    -------
    PnL
    """
    from . import _reinsurance
    terms = agg.variable_terms
    P_G = float(agg.variable_gross_premium)
    P_C = float(getattr(agg, 'variable_ceded_premium', 0.0))
    C = float(getattr(agg, 'variable_commission', 0.0))
    layer = getattr(agg, 'variable_layer', None)
    prem_key = consideration_label or 'premium'
    loss_key = loss_label or 'loss'
    tl = terms.target_leg
    # retro varies the gross premium (phi of the net account loss = the gross
    # loss in the supported no-inuring-reinsurance case); everything else
    # keeps the fixed P_G.
    cons = [Leg(prem_key, terms.phi if tl == 'gross_premium' else P_G)]
    obl = [Leg(loss_key, lambda x: x)]
    obl += _expense_legs(agg, expense_spec, P_G, on_source_loss=True,
                         taken={leg.label for leg in cons + obl})
    groups = [Group('gross', 'sell', cons, obl)]
    if layer is not None:
        g_ceder, _netter = _reinsurance.make_ceder_netter([layer])
        base = _first_reins_label(agg, 'agg_reins') \
            or 'ceded agg'
        if tl == 'ceded_premium':            # swing: stochastic ceded premium
            c_cons = [Leg(f'{base} premium',
                          lambda x: terms.phi(g_ceder(x)))]
        else:
            c_cons = [Leg(f'{base} premium', P_C)]
        if tl == 'ceded_loss':               # corridor: adjusted recovery
            c_obl = [Leg(f'{base} recovery',
                         lambda x: terms.phi(g_ceder(x) / P_C) * P_C)]
        else:
            c_obl = [Leg(f'{base} recovery', g_ceder)]
        if tl == 'expense':                  # slide / profit commission
            comm_key = ('sliding commission'
                        if type(terms).__name__ == 'SlideTerms'
                        else 'profit commission')
            c_obl.append(Leg(comm_key,
                             lambda x: terms.phi(g_ceder(x) / P_C) * P_C))
        elif C:
            c_obl.append(Leg(f'{base} commission', C))
        groups.append(Group(base, 'buy', c_cons, c_obl))
    return PnL(name=name or agg.name, source=agg, groups=groups,
               result_name='margin', label=label)


def build_reinstatement_pnl(agg, analysis, *, expense_spec=None,
                            consideration_label=None, loss_label=None,
                            name=None, label=None):
    """An occurrence-reinstatements program as a 2-D group ledger.

    Books over the one shared ``(L, R)`` joint ([One-2D-Source]): the gross
    ``sell`` group (fixed premium; the gross loss reads axis 0, so loss-basis
    LAE stays stochastic ``rate * l``), the occurrence cession ``buy`` group
    with the genuinely stochastic ceded premium ``D + h(R)`` and recovery
    ``A(R)``, and -- for a subsequent aggregate cover -- a third ``buy``
    group whose recovery ``g(max(L - A(R), 0))`` rides the same joint (no new
    dimension). The committed ``Scaled`` denominator is the deterministic
    ``gross - deposit - pc_agg``.

    Parameters
    ----------
    agg : Aggregate
        The engine (label source: ``occ_reins_label`` / ``agg_reins_label``).
    analysis : ReinstatementAnalysis
        The constructed analysis carrying the joint, terms and economics;
        attached to the returned P&L as ``pnl.analysis``.

    Returns
    -------
    PnL
    """
    t = analysis.terms
    P_G, D = analysis.gross_premium, t.deposit
    A, h = t.recovery, t.reinstatement_premium
    occ_base = _first_reins_label(agg, 'occ_reins') \
        or 'ceded occ'
    agg_base = _first_reins_label(agg, 'agg_reins') \
        or 'ceded agg'
    prem_key = consideration_label or 'premium'
    loss_key = loss_label or 'loss'
    cons = [Leg(prem_key, P_G)]
    obl = [Leg(loss_key, lambda l: l)]       # 1-D: reads axis 0 = gross loss
    obl += _expense_legs(agg, expense_spec, P_G, on_source_loss=True,
                         taken={leg.label for leg in cons + obl})
    groups = [Group('gross', 'sell', cons, obl)]
    occ_obl = [Leg(f'{occ_base} recovery', lambda l, r: A(r), is2d=True)]
    if analysis.occ_commission:
        occ_obl.append(Leg(f'{occ_base} commission', analysis.occ_commission))
    groups.append(Group(
        occ_base, 'buy',
        [Leg(f'{occ_base} premium', lambda l, r: D + h(r), is2d=True)],
        occ_obl))
    pc = 0.0
    if analysis.agg_recovery is not None:
        g0, pc = analysis.agg_recovery, analysis.agg_ceded_premium

        def g_rec(l, r):
            # clip at 0: off-support joint cells (r > l) carry ~no mass but
            # still evaluate, and the piecewise-linear ceder is only defined
            # on [0, inf).
            return g0(np.maximum(l - A(r), 0.0))

        a_obl = [Leg(f'{agg_base} recovery', g_rec, is2d=True)]
        if analysis.agg_commission:
            a_obl.append(Leg(f'{agg_base} commission',
                             analysis.agg_commission))
        groups.append(Group(agg_base, 'buy',
                            [Leg(f'{agg_base} premium', pc)], a_obl))
    pnl = PnL(name=name or agg.name, source=analysis.source, groups=groups,
              result_name='margin', scale=float(P_G - D - pc),
              label=label)
    pnl.analysis = analysis
    return pnl
