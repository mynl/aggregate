r"""The insurance builders over the generic P&L kernel.

All insurance semantics live here (and in the DecL grammar); the kernel
(:mod:`aggregate._pnl`) never sees the words gross / ceded / net. The builders
translate insurance programs -- premiums, losses, expenses, cessions with
their economics -- into a **source** plus signed **groups**. Two faces, two
questions (``dev/plan-pnl-consolidated-xpnl-walk.md``):

* **``pnl`` -- "what is my position?"** The consolidated net-in-to-net-out
  view: always one group, always the flat three-row card. Ceded economics are
  netted out and not shown -- a reinsured aggregate's default output is its
  net.

  * :func:`build_plain_pnl` -- a plain book: one ``sell`` group (premium /
    loss / expense legs) over the engine's own density.
  * :func:`build_consolidated_pnl` -- a guaranteed-cost reinsurance program
    consolidated over the engine's **deepest net marginal**: one net-premium
    consideration leg (gross - ceded premiums + commissions), the net loss,
    the pnl's own expenses ([Decision-PnL-Is-Consolidated]).
  * :func:`build_variable_pnl` (``walk=False``) -- a variable-rating program
    consolidated over the engine's density (the feature's subject: gross, or
    net-of-occ when an occurrence program inures): the net premium / net
    loss legs are the feature's maps folded together.
  * :func:`build_reinstatement_pnl` (``walk=False``) -- a reinstatements
    program consolidated as one sell group of 2-D legs over the (L, R)
    joint ([2D-Deferred] closed); agrees with the walk exactly (one joint).

* **``xpnl`` -- "how did I get there?"** The walk: gross -> each cover ->
  All, a multi-group :class:`PnL` ([Decision-XPnL-Is-A-Recipe] -- a way of
  building, not a new type).

  * :func:`build_xpnl_walk` -- the guaranteed-cost walk, **per-atom** over
    one shared source: the exact gross marginal when only an aggregate
    cover exists, the occurrence ``(gross, ceded)`` joint when an
    occurrence program does (the ceded-occ aggregate is not a function of
    the gross aggregate). Scenario (``κ``) ladder; every column foots.
  * :func:`build_variable_pnl` (``walk=True``) -- the variable-rating
    two-group per-atom ledger over the shared atoms, or the per-atom tower
    over the occurrence joint when a guaranteed-cost occurrence program
    inures ([Var-Feature-Composed-With-Occ-Program]).
  * :func:`build_reinstatement_pnl` (``walk=True``) -- the
    occurrence-reinstatements per-atom 2-D step tower over the ``(L, R)``
    joint.

The expense resolvers (:func:`resolve_expense` / :func:`_resolve_expense_split`)
relocated here from ``_pnl.py`` unchanged: ``_resolve_expense_split`` is the
only resolver feeding legs (loss-basis expense is stochastic ``rate * x``
wherever the loss is on-source); ``resolve_expense`` survives for scalar needs
(the analyses' deterministic economics) but never feeds a leg.
"""
from __future__ import annotations

import numpy as np

from ._pnl import Leg, Group, PnL

__all__ = ['build_plain_pnl', 'build_consolidated_pnl', 'build_xpnl_walk',
           'build_xpnl_peel', 'build_variable_pnl', 'build_reinstatement_pnl',
           'build_reinstatement_source', 'resolve_expense']


#: Grid-adequacy floor ([Reinst-Joint-Grid-Adequacy],
#: dev/done/plan-pnl-faces-punchlist.md): warn
#: with :class:`~aggregate.constants.CoarseJointGridWarning` when a treaty
#: kink region (the occurrence fill width ``y``; an aggregate cover's layer
#: width) spans fewer than this many buckets of the 2-D joint. A kinked map
#: on too few buckets carries a Jensen-type O(bs) bias the internal audits
#: cannot see. Author decision 2026-07-05: 20.
JOINT_KINK_MIN_BUCKETS = 20


def check_joint_grid_adequacy(bs_x, bs_y, terms, agg_reins=None, *,
                              stacklevel=3):
    """Warn when the joint grid under-resolves a treaty kink region.

    Parameters
    ----------
    bs_x, bs_y : float
        The joint's axis-0 (gross loss) / axis-1 (unlimited ceded) bucket
        sizes.
    terms : ReinstatementTerms or None
        The occurrence basis; its fill width ``y = terms.limit`` is the occ
        tier's kink scale (read on axis 1). ``None`` for a guaranteed-cost
        occurrence joint (no reinstatement kink -- the per-atom ``xpnl``
        walks reuse this check for their aggregate tier).
    agg_reins : list of (share, limit, attach), optional
        A subsequent aggregate cover's layers; each finite width
        ``share * limit`` is a kink scale of the derived net ``L - A(R)``
        (or ``L - C``), resolved no better than ``max(bs_x, bs_y)``.
    """
    import warnings as _warnings
    from .constants import CoarseJointGridWarning
    short = []
    if terms is not None:
        n_occ = terms.limit / bs_y
        if np.isfinite(terms.limit) and n_occ < JOINT_KINK_MIN_BUCKETS:
            short.append(f'occurrence fill width {terms.limit:g} spans '
                         f'{n_occ:.1f} buckets (axis-1 bs {bs_y:g})')
    bs_net = max(bs_x, bs_y)
    for share, limit, _attach in (agg_reins or []):
        width = share * limit
        n_agg = width / bs_net
        if np.isfinite(width) and n_agg < JOINT_KINK_MIN_BUCKETS:
            short.append(f'aggregate-cover layer width {width:g} spans '
                         f'{n_agg:.1f} buckets (bs {bs_net:g})')
    if short:
        _warnings.warn(
            'coarse reinstatement joint: ' + '; '.join(short)
            + f'. Kinked treaty maps need >= {JOINT_KINK_MIN_BUCKETS} '
            'buckets across each kink region for grid-accurate recoveries; '
            'refine the joint grid (bs / log2_x / log2_y forwarded to '
            'occ_bivariate).',
            CoarseJointGridWarning, stacklevel=stacklevel)


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
            legs.append(Leg(key, lambda x, r=loss_rate, s=scalar_exp: r * x + s, kind='expense'))
        elif loss_rate:
            legs.append(Leg(key, loss_rate * _expected_gross_loss(agg)
                            + scalar_exp, kind='expense'))
        elif scalar_exp:
            legs.append(Leg(key, scalar_exp, kind='expense'))
    return legs


# ----------------------------------------------------------------------
# Reinsurance economics helpers (relocated from _pnl.py)
# ----------------------------------------------------------------------
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


#: Cession-basis prefix used in a default step label, keyed by ``label_map`` site.
_SITE_PREFIX = {'occ_reins': 'occ', 'agg_reins': 'agg'}


def _layer_descriptor(site, clause):
    """One layer as its DecL descriptor, e.g. ``'occ 300 xs 200'``.

    Reuses the writer's cession renderer, so the descriptor is exactly the DecL
    the layer round-trips to (a bare share renders ``limit xs attach``, a
    partial share ``share% so limit xs attach``).
    """
    from .decl_writer import _render_reins_clause
    return f'{_SITE_PREFIX[site]} {_render_reins_clause(clause)}'


def _reins_layer_label(agg, site, index, clause):
    """The step label for one reinsurance layer ([Walk-Step-Default-Labels]).

    The layer's declared ``as`` label wins, read from the sparse
    ``{layer_index: label}`` dict pooled into the engine's ``label_map``
    (``agg.labels.occ_reins`` / ``.agg_reins``); an undeclared layer falls back
    to its DecL descriptor, which is distinct per layer because the validator
    rejects overlapping cessions. ``site`` is ``'occ_reins'`` or
    ``'agg_reins'``.
    """
    d = (getattr(agg, 'label_map', None) or {}).get(site) or {}
    if index in d and d[index]:
        return str(d[index])
    return _layer_descriptor(site, clause)


def _tier_label(agg, site, generic):
    """The step label for a whole cession **tier** ([Walk-Step-Default-Labels]).

    A tier consolidates its layers into one group, so the first declared label
    names it. Undeclared, a **single-layer** tier is named by that layer's DecL
    descriptor (``'occ 4750 xs 250'``), which says what the step actually is; a
    multi-layer tier has no one descriptor and keeps the generic name (``peel``
    is how you see those layers separately).
    """
    declared = _first_reins_label(agg, site)
    if declared:
        return declared
    layers = getattr(agg, site, None) or []
    if len(layers) == 1:
        return _layer_descriptor(site, layers[0])
    return generic


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
                    label=None, walk=False):
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
        ``'premium'`` (one name across faces -- the walk's gross step uses
        the same default).
    loss_label : str, optional
        The loss leg label (the engine ``as`` clause); default ``'loss'``.
    expense_spec : list or tuple, optional
        A DecL ``expense`` spec; one obligation leg per group, loss-basis
        terms stochastic (``rate * x``).
    name, label : str, optional
        The P&L handle and its optional human label.
    walk : bool, default False
        ``True`` presents the same one-group ledger as a **one-step walk**
        ([Decision-XPnL-Plain-Is-One-Step-Walk]): the (Step, Side) card and
        (Step, Side, Label) sheet with the single gross step and closing
        grand rows (which duplicate it; the impact is identically zero).
        The ``xpnl`` face of a plain engine -- boring but uniform.

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
    # the source is the SIMPLEST sufficient object ([Engine-Reference-On-PnL]):
    # the engine's own density as a GridDistribution -- what comes out of the
    # engine, uniform with the reinsurance faces. The full engine stays
    # reachable via ``pnl.engine`` (set by the DecL dispatch / make_pnl).
    source = engine
    if hasattr(engine, 'density_df'):
        from ._grid_distribution import GridDistribution
        dd = engine.density_df
        xs = (dd['loss'].to_numpy(dtype=float) if 'loss' in dd.columns
              else dd.index.to_numpy(dtype=float))
        source = GridDistribution(
            xs, dd['p_total'].to_numpy(dtype=float),
            bs=getattr(engine, 'bs', None) or None,
            name=getattr(engine, 'name', None))
    loss_key = loss_label or 'Loss'
    obligation = [Leg(loss_key, lambda x: x, kind='loss')]
    obligation += _expense_legs(engine, expense_spec, gp,
                                on_source_loss=True, taken={loss_key})
    prem_key = consideration_label or 'premium'
    if walk:
        pnl = PnL(name=name, source=source,
                  groups=[Group('Gross', 'sell', [Leg(prem_key, cons, kind='premium')],
                                obligation)],
                  result_name='margin', label=label, force_tower=True)
    else:
        pnl = PnL(name=name, role='sell', source=source,
                  consideration=[Leg(prem_key, cons, kind='premium')],
                  obligation=obligation, result_name='margin',
                  label=label)
    face = 'one-step walk (xpnl)' if walk else 'consolidated'
    pnl._construction_description = (
        f'Plain pnl, {face} face: one sell group over the engine\'s own '
        f'density ({type(engine).__name__}); consideration '
        f'{"loss-sensitive f(x)" if callable(cons) else f"{cons:g}"}; '
        'scenario (κ) ladder (single shared source).')
    pnl._construction_explanation = (
        pnl._construction_description + '\n'
        'Booking: sell -- +consideration, -obligation; loss-basis LAE is '
        'stochastic rate * x (the loss is on-source). For a '
        'reinsurance-bearing engine the density here is the net -- what '
        'comes out of the engine.'
        + ('\nOne-step walk: no cover to step through -- the grand rows '
           'duplicate the gross step and the impact is zero '
           '([Decision-XPnL-Plain-Is-One-Step-Walk]).' if walk else '')
        + '\n\n' + pnl._replay_block())
    return pnl


def build_consolidated_pnl(agg, *, gross, ceded, gcn_economics=None,
                           expense_spec=None, consideration_label=None,
                           loss_label=None, name=None, label=None):
    """A guaranteed-cost reinsurance program as the **consolidated**
    single-group :class:`PnL` ([Decision-PnL-Is-Consolidated]).

    The pnl is the P&L of *what the object does*: a reinsured aggregate's
    default output is its net, so the consolidated view books

    * **consideration** -- one net-premium leg: gross premium minus the ceded
      premiums plus the ceding commissions received (constant for a
      guaranteed-cost program);
    * **obligation** -- the engine's **net loss** (the deepest net marginal
      of ``reins_density_df``: net-of-agg when an aggregate cover exists,
      else net-of-occ) plus the pnl's own expense legs. The gross loss is not
      measurable on the net marginal, so a loss-basis (LAE) expense books as
      its deterministic ``rate * E[gross loss]``.

    Ceded economics are netted out and not shown; the walk -- per-step
    results, running nets, the closing impact -- is one ``xpnl`` away
    (:func:`build_xpnl_walk`). The resolved economics ride on
    ``pnl.economics``. Leg labels follow [Flag-Net-Premium-Leg-Label]:
    default ``'net premium'`` / ``'loss (net)'``; a declared label qualifies
    as ``'<label> (net)'``.

    Returns
    -------
    PnL
    """
    rd = agg.reins_density_df
    if rd is None:
        raise ValueError(
            'a consolidated reinsurance pnl requires reinsurance on the '
            'risky leg; the aggregate carries no occurrence / aggregate '
            'treaty.')
    has_agg = agg.agg_reins is not None
    p_gross, pc_occ, pc_agg, c_occ, c_agg = _ledger_economics(
        agg, gross, ceded, gcn_economics)
    net_premium = p_gross - pc_occ - pc_agg + c_occ + c_agg
    prem_key = (f'{consideration_label} (net)' if consideration_label
                else 'net premium')
    loss_key = f'{loss_label or "Loss"} (net)'
    persp = 'net_agg' if has_agg else 'net_occ'
    source = _marginal_gd(agg, persp)
    cons = [Leg(prem_key, net_premium, kind='premium')]
    obl = [Leg(loss_key, lambda x: x, kind='loss')]
    obl += _expense_legs(agg, expense_spec, p_gross, on_source_loss=False,
                         taken={prem_key, loss_key})
    pnl = PnL(name=name or agg.name, role='sell', source=source,
              consideration=cons, obligation=obl, result_name='margin',
              label=label)
    #: the resolved cession economics (a DecL clause's per-side split, or the
    #: scalar API's single amounts) -- the observable of the DecL premium /
    #: commission resolution (``deposit`` / ``rol`` / ``rate`` / ``cede``).
    pnl.economics = dict(gcn_economics) if gcn_economics is not None \
        else {'gross': float(gross), 'ceded': float(ceded)}
    pnl._construction_description = (
        f'Consolidated pnl over the engine\'s deepest net marginal '
        f'(reins_density_df[{_GCN_LOSS_MARGINAL[persp]!r}]): one sell group; '
        f'net premium {net_premium:g} = gross {p_gross:g} - ceded premiums '
        f'{pc_occ + pc_agg:g} + commissions {c_occ + c_agg:g}; scenario (κ) '
        f'ladder (single shared source). The walk is one xpnl away.')
    pnl._construction_explanation = _consolidated_explanation(
        agg, pnl, persp, p_gross, pc_occ, pc_agg, c_occ, c_agg, net_premium)
    return pnl


def _consolidated_explanation(agg, pnl, persp, p_gross, pc_occ, pc_agg,
                              c_occ, c_agg, net_premium):
    """The full [Construction-Introspection] story for a consolidated pnl."""
    lines = [
        f'Consolidated pnl {pnl.label!r} ([Decision-PnL-Is-Consolidated]).',
        f'Engine: {agg.name!r}'
        + (f' -- occurrence reinsurance {agg.occ_reins}'
           if agg.occ_reins is not None else '')
        + (f' -- aggregate reinsurance {agg.agg_reins}'
           if agg.agg_reins is not None else '') + '.',
        'Economics resolution: '
        f'gross premium {p_gross:g}; ceded premium occ {pc_occ:g} / '
        f'agg {pc_agg:g}; commissions occ {c_occ:g} / agg {c_agg:g} '
        f'-> net premium {net_premium:g} (gross - ceded premiums '
        '+ commissions, one constant consideration leg).',
        f'Source: the deepest net marginal, '
        f"reins_density_df[{_GCN_LOSS_MARGINAL[persp]!r}] -- what comes out "
        'of the engine. The net loss leg reads it per atom; a loss-basis '
        '(LAE) expense is off-source there and books as its deterministic '
        'rate * E[gross loss].',
        'Booking: sell role -- +consideration, -obligation; the EX column '
        'adds down the sheet.',
        'Ladder: scenario (κ) -- the pnl is a single-source ledger, so the '
        'stats columns condition on the grand result '
        '([Decision-Kappa-Shared-Source-Rule]).',
        'Ceded economics are netted out and not shown: the per-step walk '
        "(gross -> each cover -> Total) is the xpnl face of the same "
        'engine.',
        '',
        pnl._replay_block(),
    ]
    return '\n'.join(lines)


def build_xpnl_walk(agg, *, gross, ceded, gcn_economics=None,
                    expense_spec=None, consideration_label=None,
                    loss_label=None, name=None, label=None):
    """The guaranteed-cost ``xpnl`` **walk**: a per-atom multi-group
    :class:`PnL` over one shared source ([XPnL-Walk-Recipe]).

    The step tower -- gross -> each cover -> All -- books every leg as a
    function of the ONE shared source, so the stats ladder is the scenario
    (``\u03ba``) pass and every column **foots** exactly (legs -> totals ->
    result add down the sheet; [Decision-Kappa-Shared-Source-Rule]). The
    source is picked by the occurrence tier:

    * **no occurrence program** -- the engine's exact gross marginal
      (1-D atoms off ``reins_density_df``); an aggregate cover's legs are
      functions ``g(x)`` of it.
    * **occurrence program** -- the occurrence ``(gross, ceded)`` joint from
      :meth:`~aggregate.Aggregate.occ_bivariate`: the ceded-occ aggregate is
      NOT a function of the gross aggregate (the random claim count
      decouples them), so no 1-D source can carry a footing walk. The joint
      runs on a budget-sized common bucket size -- coarser than the engine
      grid; :class:`~aggregate.constants.CoarseJointGridWarning` guards
      kinked aggregate-tier maps ([Reinst-Joint-Grid-Adequacy]). A
      subsequent aggregate cover recovers ``g(max(L - C, 0))`` on the same
      joint.

    Loss-basis LAE books stochastic ``rate * l`` (the gross loss is
    on-source -- an improvement over the retired marginal stitch, which was
    all-marginal). Step labels: the base step is the engine's declared label
    (fallback ``'Gross'``); cover steps are the reins ``as`` labels
    (a single-layer tier falls back to that layer's DecL descriptor, e.g.
    ``'occ 4750 xs 250'``; a multi-layer tier to ``'ceded occ'`` /
    ``'ceded agg'`` -- see ``peel`` to walk those layers); the closing grand step's
    index key is ``'All'`` (a140 rename).

    Returns
    -------
    PnL
        A plain multi-group :class:`PnL` -- xpnl is a way of building, not a
        new type ([Decision-XPnL-Is-A-Recipe]).
    """
    from . import _reinsurance
    rd = agg.reins_density_df
    if rd is None:
        raise ValueError(
            "'xpnl' requires reinsurance on the wrapped engine; the aggregate "
            'carries no occurrence / aggregate treaty.')
    has_occ = agg.occ_reins is not None
    has_agg = agg.agg_reins is not None
    p_gross, pc_occ, pc_agg, c_occ, c_agg = _ledger_economics(
        agg, gross, ceded, gcn_economics)
    occ_base = _tier_label(agg, 'occ_reins', 'ceded occ')
    agg_base = _tier_label(agg, 'agg_reins', 'ceded agg')
    base_step = loss_label or 'Gross'
    prem_key = consideration_label or 'premium'
    loss_key = loss_label or 'Loss'

    gross_obl = [Leg(loss_key, lambda l: l, kind='loss')]     # reads axis 0 on a joint
    gross_obl += _expense_legs(agg, expense_spec, p_gross,
                               on_source_loss=True,
                               taken={prem_key, loss_key})
    groups = [Group(base_step, 'sell', [Leg(prem_key, p_gross, kind='premium')], gross_obl)]
    if has_occ:
        # the occurrence (gross, ceded) joint -- built per walk
        biv = agg.occ_bivariate(views=('gross', 'ceded'))
        check_joint_grid_adequacy(biv.bivariate.bs_ceded,
                                  biv.bivariate.bs_net, None, agg.agg_reins)
        source = biv.bivariate
        occ_obl = [Leg(f'{occ_base} recovery', lambda l, c: c, is2d=True, kind='recovery')]
        if c_occ:
            occ_obl.append(Leg(f'{occ_base} commission', c_occ, kind='commission'))
        groups.append(Group(occ_base, 'buy',
                            [Leg(f'{occ_base} premium', pc_occ, kind='premium')], occ_obl))
        if has_agg:
            g_agg, _netter = _reinsurance.make_ceder_netter(agg.agg_reins)

            def agg_rec(l, c):
                # clip at 0: off-support joint cells (c > l) carry ~no mass
                # but still evaluate.
                return g_agg(np.maximum(l - c, 0.0))

            a_obl = [Leg(f'{agg_base} recovery', agg_rec, is2d=True, kind='recovery')]
            if c_agg:
                a_obl.append(Leg(f'{agg_base} commission', c_agg, kind='commission'))
            groups.append(Group(agg_base, 'buy',
                                [Leg(f'{agg_base} premium', pc_agg, kind='premium')], a_obl))
    else:
        # aggregate cover only: everything is a function of the gross atoms
        source = _marginal_gd(agg, 'gross')
        g_agg, _netter = _reinsurance.make_ceder_netter(agg.agg_reins)
        a_obl = [Leg(f'{agg_base} recovery', g_agg, kind='recovery')]
        if c_agg:
            a_obl.append(Leg(f'{agg_base} commission', c_agg, kind='commission'))
        groups.append(Group(agg_base, 'buy',
                            [Leg(f'{agg_base} premium', pc_agg, kind='premium')], a_obl))
    pnl = PnL(name=name or agg.name, source=source, groups=groups,
              result_name='margin', label=label)
    pnl.economics = dict(gcn_economics) if gcn_economics is not None \
        else {'gross': float(gross), 'ceded': float(ceded)}
    src_desc = ('the occurrence (gross, ceded) joint' if has_occ
                else 'the exact gross marginal')
    pnl._construction_description = (
        f'xpnl walk: {len(groups)}-step per-atom tower '
        f'({" -> ".join(g.label for g in groups)} -> All) over '
        f'{src_desc}; scenario (\u03ba) ladder -- every column foots.')
    pnl._construction_explanation = _walk_explanation(agg, pnl, has_occ,
                                                      src_desc)
    return pnl


def _walk_explanation(agg, pnl, has_occ, src_desc):
    """The full [Construction-Introspection] story for a per-atom walk."""
    econ = pnl.economics
    lines = [
        f'xpnl walk {pnl.label!r} ([XPnL-Walk-Recipe], per-atom).',
        f'Engine: {agg.name!r}'
        + (f' -- occurrence reinsurance {agg.occ_reins}'
           if agg.occ_reins is not None else '')
        + (f' -- aggregate reinsurance {agg.agg_reins}'
           if agg.agg_reins is not None else '') + '.',
        f'Economics resolution: {econ!r}.',
        f'Source: {src_desc}'
        + (' (occ_bivariate; the ceded-occ aggregate is not a function of '
           'the gross aggregate, so a footing walk needs the joint -- built '
           'on a budget-sized common bucket size, coarser than the engine '
           'grid; CoarseJointGridWarning guards kinked aggregate-tier maps).'
           if has_occ else
           ' (1-D atoms; every cover leg is a function g(x) of it).'),
        'Booking: sell books +consideration / -obligation, buy the contra; '
        'every ledger row is a per-atom partial sum, so the EX column and '
        'every \u03ba column foot exactly; loss-basis LAE is stochastic '
        'rate * l (the gross loss is on-source).',
        'Ladder: scenario (\u03ba) -- one shared source '
        '([Decision-Kappa-Shared-Source-Rule]).',
        '',
        pnl._replay_block(),
    ]
    return '\n'.join(lines)


# ----------------------------------------------------------------------
# Layer peeling ([Layer-Peeling-Shorthand], dev/done/plan-layer-peeling.md)
#
# ``xpnl ... peel top-down|bottom-up`` books one group per reinsurance LAYER
# instead of one per tier, so every layer shows its own ceded premium,
# commission and marginal impact, and the running net builds the program up
# one layer at a time.
#
# The two tiers are not symmetric, which picks the route:
#
# * **aggregate layers** are disjoint intervals on the ONE aggregate subject
#   axis, so single-layer ceders sum identically to the cumulative ceder
#   (``ceder([L1..Lk])(x) == sum_i ceder([Li])(x)``). Each layer is therefore
#   just another per-atom ``Leg`` on the shared source: the scenario (kappa)
#   ladder survives and every column foots.
# * **occurrence layers** are not. The ceded-occurrence aggregate is not a
#   function of the gross aggregate (the random claim count decouples them),
#   which is why the tier walk reaches for the 2-D ``occ_bivariate``; splitting
#   m layers per-atom would need an (m+1)-axis joint, and the total ceded axis
#   cannot be decomposed after the fact because ``sum_i ceder(X_i)`` does not
#   determine the per-layer allocation. So two or more peeled occurrence layers
#   route through the kernel's stitched seam, where every row is supplied as
#   its own exact marginal. Each such row is the aggregate of a deterministic
#   per-claim severity transform, so the marginals ARE exact and the EX column
#   foots exactly by linearity; only the dispersion columns are marginal
#   (plain ``P`` headers, not kappa).
# ----------------------------------------------------------------------
def _peel_steps(agg, direction):
    """The peel order as ``(site, index, clause, label)`` per layer.

    Occurrence layers precede aggregate ones, preserving the tier walk's step
    order. Within a tier ``bottom-up`` introduces layers in stored (ascending
    attachment) order and ``top-down`` in reverse. Zero-share layers are gap
    fillers, structural rather than cessions, and get no step.
    """
    steps = []
    for site in ('occ_reins', 'agg_reins'):
        layers = getattr(agg, site, None) or []
        idx = [i for i, (share, _y, _a) in enumerate(layers)
               if float(share) != 0.0]
        if direction == 'top-down':
            idx = list(reversed(idx))
        for i in idx:
            steps.append((site, i, layers[i],
                          _reins_layer_label(agg, site, i, layers[i])))
    return steps


def _peel_side_economics(econ, which, n_layers, total_pc, total_comm):
    """Per-layer ceded premium / commission for one cession basis.

    ``Underwriter._resolve_reins_economics`` supplies the per-layer lists that
    produced the side totals. A scalar-API or zero-fallback economics dict
    carries only the totals; book those on the lowest-indexed layer so the
    ledger still foots against the consolidated face.
    """
    out = []
    for key, total in ((f'pc_{which}_by_layer', total_pc),
                       (f'c_{which}_by_layer', total_comm)):
        vals = list((econ or {}).get(key) or [])
        if len(vals) != n_layers:
            vals = [0.0] * n_layers
            if n_layers and total:
                vals[0] = float(total)
        out.append([float(v) for v in vals])
    return out[0], out[1]


def _affine_row(x, p, *, sign=1.0, shift=0.0, label=''):
    """One stitched ledger row: the marginal of ``sign * X + shift``.

    Returns the ``(gd, exact_mean, exact_sd)`` triple
    :meth:`PnL._init_stitched` expects. The mean carries the sign and shift;
    the standard deviation is invariant to both. ``GridDistribution`` requires
    an increasing index, so a negative sign reverses the pair.
    """
    from ._grid_distribution import GridDistribution
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)
    tot = float(p.sum())
    if tot > 0:
        m1 = float((x * p).sum()) / tot
        m2 = float((x * x * p).sum()) / tot
    else:                                             # pragma: no cover
        m1 = m2 = 0.0
    sd = float(np.sqrt(max(m2 - m1 * m1, 0.0)))
    v = sign * x + shift
    if sign < 0:
        v, p = v[::-1], p[::-1]
    return (GridDistribution(v, p, bs=None, name=label, is_loss_value=False),
            sign * m1 + shift, sd)


def _const_row(value, label=''):
    """A stitched ledger row that is a known constant (premium, commission)."""
    from ._grid_distribution import GridDistribution
    value = float(value)
    return (GridDistribution(np.array([value]), np.array([1.0]), bs=None,
                             name=label, is_loss_value=False), value, 0.0)


def _sev_transform_marginal(agg, f):
    """The exact aggregate marginal of ``sum_j f(X_j)`` over the claim count.

    ``f`` is a non-negative per-claim transform of the gross severity, so the
    compound is one FFT of the transformed severity discretized back onto the
    model grid. This is how a peeled occurrence layer's recovery, and every
    running net through a set of occurrence layers, are computed exactly: both
    are per-claim deterministic functions, even though neither is a function of
    the gross *aggregate*.
    """
    xs = agg.xs
    sev = agg.sev_density_gross if agg.sev_density_gross is not None \
        else agg.sev_density
    moved = agg._rebucket_to_grid(np.asarray(f(xs), dtype=float),
                                  np.asarray(sev, dtype=float))
    p_agg, _ft = agg._fft_aggregate(moved, agg.padding)
    return xs, np.asarray(p_agg, dtype=float)


def _agg_transform_marginal(agg, f, subject_p):
    """The exact marginal of ``f(S)`` for a deterministic map of aggregate ``S``.

    An aggregate cover acts on the one aggregate subject, so its layers are
    plain pushforwards: rebucket ``f(xs)`` carrying the subject's mass. No FFT.
    """
    xs = agg.xs
    return xs, np.asarray(
        agg._rebucket_to_grid(np.asarray(f(xs), dtype=float),
                              np.asarray(subject_p, dtype=float)), dtype=float)


def build_xpnl_peel(agg, *, direction, gross, ceded, gcn_economics=None,
                    expense_spec=None, consideration_label=None,
                    loss_label=None, name=None, label=None):
    """The layer-peeled ``xpnl``: one group per reinsurance **layer**.

    ``xpnl ... peel top-down`` / ``peel bottom-up`` replaces the tier walk's
    single lumped occurrence and aggregate groups with one group per layer, so
    each layer reports its own ceded premium, ceding commission and marginal
    impact, and ``net through <layer>`` builds the program up a layer at a time
    ([Layer-Peeling-Shorthand]). Occurrence layers are peeled before aggregate
    ones, preserving the tier walk's step order; within a tier ``top-down``
    introduces the highest-attaching layer first and ``bottom-up`` the lowest.
    Zero-share gap fillers get no step.

    Two routes, chosen by how many occurrence layers are actually peeled:

    * **per-atom** (at most one occurrence step). Aggregate layers are disjoint
      intervals on the one aggregate subject, so ``ceder([L1..Lk])`` equals
      ``sum_i ceder([Li])`` identically and each layer is simply another
      per-atom :class:`~aggregate._pnl.Leg` on the shared source. The scenario
      (kappa) ladder holds and every column foots exactly, as in the tier walk.
    * **stitched** (two or more occurrence steps). The ceded-occurrence
      aggregate is not a function of the gross aggregate, and the total ceded
      axis of the 2-D joint cannot be split after the fact, so no shared source
      can carry a per-layer occurrence walk. Every ledger row is supplied
      instead as its own exact marginal: each is the compound of a
      deterministic per-claim severity transform (one FFT), or a pushforward of
      the aggregate subject for the aggregate tier. The **EX column still foots
      exactly** by linearity (the default ``'linear'`` rebucketing preserves
      first moments), but the dispersion columns are marginal, so the ladder
      carries plain ``P`` headers rather than kappa, ``evaluate`` is
      unavailable and loss-basis LAE degrades to the deterministic
      ``rate * E[gross loss]`` (the shared-source loss it would scale with does
      not exist here; the same trade the consolidated face makes).

    Returns
    -------
    PnL
        A plain multi-group :class:`~aggregate._pnl.PnL` -- peeling is a way of
        building, not a new type ([Decision-XPnL-Is-A-Recipe]).
    """
    rd = agg.reins_density_df
    if rd is None:
        raise ValueError(
            "'xpnl ... peel' requires reinsurance on the wrapped engine; the "
            'aggregate carries no occurrence / aggregate treaty.')
    steps = _peel_steps(agg, direction)
    if not steps:
        raise ValueError(
            f'{agg.name}: nothing to peel -- every reinsurance layer on this '
            'engine has zero share (gap fillers are structural, not '
            "cessions). Drop the 'peel' clause.")
    occ_steps = [s for s in steps if s[0] == 'occ_reins']
    agg_steps = [s for s in steps if s[0] == 'agg_reins']
    p_gross, pc_occ, pc_agg, c_occ, c_agg = _ledger_economics(
        agg, gross, ceded, gcn_economics)
    pc_occ_by, c_occ_by = _peel_side_economics(
        gcn_economics, 'occ', len(agg.occ_reins or []), pc_occ, c_occ)
    pc_agg_by, c_agg_by = _peel_side_economics(
        gcn_economics, 'agg', len(agg.agg_reins or []), pc_agg, c_agg)
    econ_by = {'occ_reins': (pc_occ_by, c_occ_by),
               'agg_reins': (pc_agg_by, c_agg_by)}
    # One subtotal block per tier that peels into two or more steps
    # ([Tier-Subtotal-Rows]). Groups run [gross] + occ covers + agg covers, so
    # the spans are the cover ranges offset by the one gross group. A tier that
    # peels into a single step already *is* its own subtotal, and
    # ``_ledger_plan`` drops such a span, so this needs no length test here.
    tier_spans = (('All occurrence', 1, 1 + len(occ_steps)),
                  ('All aggregate', 1 + len(occ_steps),
                   1 + len(occ_steps) + len(agg_steps)))
    common = dict(base_step=loss_label or 'Gross',
                  prem_key=consideration_label or 'premium',
                  loss_key=loss_label or 'Loss',
                  expense_spec=expense_spec, econ_by=econ_by,
                  p_gross=p_gross, tier_spans=tier_spans,
                  name=name, label=label)
    if len(occ_steps) >= 2:
        pnl, route = _peel_stitched(agg, occ_steps, agg_steps, **common)
    else:
        pnl, route = _peel_per_atom(agg, occ_steps, agg_steps, **common)
    pnl.economics = dict(gcn_economics) if gcn_economics is not None \
        else {'gross': float(gross), 'ceded': float(ceded)}
    order = ' -> '.join(s[3] for s in steps)
    pnl._construction_description = (
        f'xpnl peel ({direction}): {len(steps)}-layer '
        f'{"per-atom" if route == "per-atom" else "marginal"} tower '
        f'(Gross -> {order} -> All); '
        + ('scenario (κ) ladder -- every column foots.'
           if route == 'per-atom' else
           'EX foots exactly by linearity, dispersion columns are marginal.'))
    pnl._construction_explanation = _peel_explanation(
        agg, pnl, direction, steps, route)
    return pnl


def _peel_per_atom(agg, occ_steps, agg_steps, *, base_step, prem_key, loss_key,
                   expense_spec, econ_by, p_gross, tier_spans, name, label):
    """Peel over ONE shared source: the kappa ladder survives and columns foot.

    Reached when at most one occurrence layer is peeled, so the joint's ceded
    axis already *is* that layer's cession and nothing has to be split. Each
    aggregate layer books its own ceder as a per-atom leg, exact because
    aggregate layers are disjoint intervals on the same subject.

    Tier subtotals need nothing extra here: they are per-atom partial sums over
    a group span, assembled by the kernel ([Tier-Subtotal-Rows]).
    """
    from . import _reinsurance
    gross_obl = [Leg(loss_key, lambda l: l, kind='loss')]         # reads axis 0 on a joint
    gross_obl += _expense_legs(agg, expense_spec, p_gross,
                               on_source_loss=True,
                               taken={prem_key, loss_key})
    groups = [Group(base_step, 'sell', [Leg(prem_key, p_gross, kind='premium')], gross_obl)]

    def cover(lbl, pc, comm, recovery, is2d):
        obl = [Leg(f'{lbl} recovery', recovery, is2d=is2d, kind='recovery')]
        if comm:
            obl.append(Leg(f'{lbl} commission', comm, kind='commission'))
        return Group(lbl, 'buy', [Leg(f'{lbl} premium', pc, kind='premium')], obl)

    if occ_steps:
        site, i, _clause, lbl = occ_steps[0]
        biv = agg.occ_bivariate(views=('gross', 'ceded'))
        check_joint_grid_adequacy(biv.bivariate.bs_ceded,
                                  biv.bivariate.bs_net, None, agg.agg_reins)
        source = biv.bivariate
        pc_by, c_by = econ_by[site]
        groups.append(cover(lbl, pc_by[i], c_by[i],
                            lambda l, c: c, True))
        for site_j, j, clause, lbl_j in agg_steps:
            ceder_j, _n = _reinsurance.make_ceder_netter([clause])
            pc_by, c_by = econ_by[site_j]

            def rec(l, c, g=ceder_j):
                # the aggregate cover's subject is the net-of-occurrence
                # aggregate; clip at 0 for off-support joint cells (c > l),
                # which carry ~no mass but still evaluate.
                return g(np.maximum(l - c, 0.0))

            groups.append(cover(lbl_j, pc_by[j], c_by[j], rec, True))
    else:
        # aggregate layers only: everything is a function of the gross atoms
        source = _marginal_gd(agg, 'gross')
        for site_j, j, clause, lbl_j in agg_steps:
            ceder_j, _n = _reinsurance.make_ceder_netter([clause])
            pc_by, c_by = econ_by[site_j]
            groups.append(cover(lbl_j, pc_by[j], c_by[j], ceder_j, False))
    pnl = PnL(name=name or agg.name, source=source, groups=groups,
              result_name='margin', tier_spans=tier_spans, label=label)
    return pnl, 'per-atom'


def _peel_stitched(agg, occ_steps, agg_steps, *, base_step, prem_key, loss_key,
                   expense_spec, econ_by, p_gross, tier_spans, name, label):
    """Peel two or more occurrence layers: exact marginals, no shared source.

    Every ledger row is supplied to :meth:`PnL._init_stitched` as its own exact
    marginal. Occurrence rows are compounds of deterministic per-claim severity
    transforms (one FFT each); aggregate rows are pushforwards of the
    net-of-occurrence aggregate. The chain is computed end to end from the
    engine's gross severity rather than read off ``reins_density_df``, so the
    rows are internally consistent and the EX column foots exactly.

    A tier subtotal ([Tier-Subtotal-Rows]) needs the tier's **whole** cession as
    its own marginal: summing the per-layer recovery vectors would add densities
    rather than variables, and the tier recovery is not an affine of any
    cumulative net already in hand. That costs one further FFT for the
    occurrence tier (the cumulative ceder from its last step) and no FFT at all
    for the aggregate tier, which is a pushforward.
    """
    from . import _reinsurance
    from ._pnl import _ledger_plan
    xs = agg.xs

    # ----- gross book -------------------------------------------------
    expense_legs = _expense_legs(agg, expense_spec, p_gross,
                                 on_source_loss=False,
                                 taken={prem_key, loss_key})
    # every expense leg is a constant on this route (no shared source to
    # scale a loss-basis term against), so its magnitude is its value at 0
    expense_vals = [float(leg.func(0.0)) if callable(leg.func)
                    else float(leg.func) for leg in expense_legs]
    expense_total = float(sum(expense_vals))
    gross_obl = [Leg(loss_key, lambda l: l, kind='loss')] + expense_legs
    groups = [Group(base_step, 'sell', [Leg(prem_key, p_gross, kind='premium')], gross_obl)]

    # ----- the peel chain: cumulative cessions, tier by tier ----------
    _xs, p_gross_agg = _sev_transform_marginal(agg, lambda x: x)
    occ_rows = []          # per occurrence step: (label, pc, comm, R, N_cum)
    taken = []
    occ_cum = None
    for site, i, clause, lbl in occ_steps:
        taken.append(i)
        occ_cum = _reinsurance.make_ceder_netter(
            [agg.occ_reins[k] for k in sorted(taken)])[0]
        one = _reinsurance.make_ceder_netter([clause])[0]
        _x, recovery = _sev_transform_marginal(agg, one)
        _x, net_cum = _sev_transform_marginal(
            agg, lambda x, g=occ_cum: x - g(x))
        pc_by, c_by = econ_by[site]
        occ_rows.append((lbl, pc_by[i], c_by[i], recovery, net_cum))
    subject = occ_rows[-1][4]        # the net-of-occurrence aggregate

    agg_rows = []          # per aggregate step: (label, pc, comm, R, N_cum)
    taken = []
    agg_cum = None
    for site, j, clause, lbl in agg_steps:
        taken.append(j)
        agg_cum = _reinsurance.make_ceder_netter(
            [agg.agg_reins[k] for k in sorted(taken)])[0]
        one = _reinsurance.make_ceder_netter([clause])[0]
        _x, recovery = _agg_transform_marginal(agg, one, subject)
        _x, net_cum = _agg_transform_marginal(
            agg, lambda s, g=agg_cum: s - g(s), subject)
        pc_by, c_by = econ_by[site]
        agg_rows.append((lbl, pc_by[j], c_by[j], recovery, net_cum))

    # the final net after every cover, and the closing running totals
    final_net = agg_rows[-1][4] if agg_rows else subject
    covers = occ_rows + agg_rows
    total_pc = float(sum(c[1] for c in covers))
    total_comm = float(sum(c[2] for c in covers))

    # ----- the whole-tier cession, for the subtotal blocks -------------
    # Keyed by group span; only computed for a tier that actually peels into
    # two or more steps, since ``_ledger_plan`` drops the others.
    n_occ = len(occ_rows)
    tier_recovery = {}
    if n_occ >= 2:
        tier_recovery[(1, 1 + n_occ)] = _sev_transform_marginal(
            agg, occ_cum)[1]
    if len(agg_rows) >= 2:
        tier_recovery[(1 + n_occ, 1 + n_occ + len(agg_rows))] = \
            _agg_transform_marginal(agg, agg_cum, subject)[1]

    def span_economics(lo, hi):
        """``(ceded premium, commission)`` summed over a span's covers."""
        block = covers[lo - 1:hi - 1]        # group 0 is Gross
        return (float(sum(c[1] for c in block)),
                float(sum(c[2] for c in block)))

    # ----- groups, one per peeled layer -------------------------------
    # The legs carry only their labels and the ledger shape here: the stitched
    # route supplies each row's distribution directly, so no leg function is
    # ever evaluated (the recovery's placeholder is never called).
    for lbl, pc, comm, _r, _n in covers:
        obl = [Leg(f'{lbl} recovery', 0.0, kind='recovery')]
        if comm:
            obl.append(Leg(f'{lbl} commission', comm, kind='commission'))
        groups.append(Group(lbl, 'buy', [Leg(f'{lbl} premium', pc, kind='premium')], obl))

    # ----- one entry per plan row -------------------------------------
    # The plan is re-derived here and must match the one ``PnL`` builds below,
    # so the same ``tier_spans`` goes to both; a mismatch surfaces as the
    # "no entry supplied for ledger row" error.
    entries = {}
    plan = _ledger_plan(groups, 'margin', tier_spans)
    # running consideration / commission net through each cover step
    running = []
    acc = 0.0
    for _lbl, pc, comm, _r, _n in covers:
        acc += comm - pc
        running.append(acc)
    for row_label, kind, payload in plan:
        if kind == 'leg':
            gi, side, li = payload
            if gi == 0:
                if side == 'cons':
                    entries[row_label] = _const_row(p_gross, row_label)
                elif li == 0:
                    entries[row_label] = _affine_row(
                        xs, p_gross_agg, sign=-1.0, label=row_label)
                else:
                    entries[row_label] = _const_row(
                        -expense_vals[li - 1], row_label)
                continue
            lbl, pc, comm, recovery, _n = covers[gi - 1]
            if side == 'cons':
                entries[row_label] = _const_row(-pc, row_label)
            elif li == 0:
                entries[row_label] = _affine_row(xs, recovery, label=row_label)
            else:
                entries[row_label] = _const_row(comm, row_label)
        elif kind == 'group_total':
            gi, side = payload
            if gi == 0:
                entries[row_label] = _affine_row(
                    xs, p_gross_agg, sign=-1.0, shift=-expense_total,
                    label=row_label)
            else:
                _lbl, _pc, comm, recovery, _n = covers[gi - 1]
                entries[row_label] = _affine_row(
                    xs, recovery, shift=comm, label=row_label)
        elif kind == 'group_result':
            gi = payload
            if gi == 0:
                entries[row_label] = _affine_row(
                    xs, p_gross_agg, sign=-1.0,
                    shift=p_gross - expense_total, label=row_label)
            else:
                _lbl, pc, comm, recovery, _n = covers[gi - 1]
                entries[row_label] = _affine_row(
                    xs, recovery, shift=comm - pc, label=row_label)
        elif kind == 'running_net':
            gi = payload
            _lbl, _pc, _comm, _r, net_cum = covers[gi - 1]
            entries[row_label] = _affine_row(
                xs, net_cum, sign=-1.0,
                shift=p_gross - expense_total + running[gi - 1],
                label=row_label)
        elif kind == 'tier_total':
            lo, hi, side = payload
            pc_span, comm_span = span_economics(lo, hi)
            if side == 'cons':
                entries[row_label] = _const_row(-pc_span, row_label)
            else:
                entries[row_label] = _affine_row(
                    xs, tier_recovery[(lo, hi)], shift=comm_span,
                    label=row_label)
        elif kind == 'tier_result':
            lo, hi = payload
            pc_span, comm_span = span_economics(lo, hi)
            entries[row_label] = _affine_row(
                xs, tier_recovery[(lo, hi)], shift=comm_span - pc_span,
                label=row_label)
        elif kind == 'grand_total':
            if payload == 'cons':
                entries[row_label] = _const_row(p_gross - total_pc, row_label)
            else:
                entries[row_label] = _affine_row(
                    xs, final_net, sign=-1.0,
                    shift=total_comm - expense_total, label=row_label)
        elif kind == 'grand_result':
            entries[row_label] = _affine_row(
                xs, final_net, sign=-1.0,
                shift=p_gross - total_pc + total_comm - expense_total,
                label=row_label)
        elif kind == 'total_impact':
            entries[row_label] = ('delta', 'margin', f'{base_step} result')
        else:                                        # pragma: no cover
            raise ValueError(
                f'unknown ledger row kind {kind!r} for row {row_label!r}; '
                '_peel_stitched must supply an entry for every plan row.')
    pnl = PnL(name=name or agg.name, source=None, groups=groups,
              result_name='margin', tier_spans=tier_spans, label=label,
              stitched_rows=entries)
    return pnl, 'stitched'


def _peel_explanation(agg, pnl, direction, steps, route):
    """The full [Construction-Introspection] story for a layer-peeled walk."""
    per_atom = route == 'per-atom'
    lines = [
        f'xpnl peel {pnl.label!r} ([Layer-Peeling-Shorthand], {route}).',
        f'Engine: {agg.name!r}'
        + (f' -- occurrence reinsurance {agg.occ_reins}'
           if agg.occ_reins is not None else '')
        + (f' -- aggregate reinsurance {agg.agg_reins}'
           if agg.agg_reins is not None else '') + '.',
        f'Economics resolution: {pnl.economics!r}.',
        f'Peel order ({direction}): '
        + ' -> '.join(f'{s[3]}' for s in steps)
        + '. Occurrence layers precede aggregate ones, preserving the tier '
          'walk step order; zero-share gap fillers get no step.',
    ]
    if per_atom:
        lines += [
            'Source: one shared source -- at most one occurrence layer is '
            'peeled, so the joint ceded axis already is that layer, and each '
            'aggregate layer rides the same atoms as its own ceder (aggregate '
            'layers are disjoint intervals on one subject, so single-layer '
            'ceders sum to the cumulative ceder identically).',
            'Ladder: scenario (κ) -- every column foots exactly '
            '([Decision-Kappa-Shared-Source-Rule]).',
        ]
    else:
        lines += [
            'Source: none -- the ceded-occurrence aggregate is not a function '
            'of the gross aggregate (the random claim count decouples them), '
            'and the total ceded axis of the 2-D joint cannot be split after '
            'the fact, so peeling two or more occurrence layers per-atom '
            'would need an (m+1)-axis joint ([One-2D-Source] forbids it).',
            'Rows: each supplied as its own exact marginal. An occurrence row '
            'is the compound of a deterministic per-claim severity transform '
            '(one FFT); an aggregate row is a pushforward of the '
            'net-of-occurrence aggregate.',
            'Ladder: marginal (plain P headers). The EX column foots exactly '
            'by linearity, the dispersion columns are deltas of row '
            'statistics rather than statistics of the delta, evaluate is '
            'unavailable, and loss-basis LAE books the deterministic '
            'rate * E[gross loss].',
        ]
    lines += ['', pnl._replay_block()]
    return '\n'.join(lines)


# ----------------------------------------------------------------------
# Variable-rating / reinstatement feature builders
# ([Builders-Variable-Features]: a feature never changes the machinery --
#  it changes one leg's function)
# ----------------------------------------------------------------------
def _feature_maps(terms, tl, g_ceder, P_C, C):
    """The three signed-magnitude cash-flow maps of an aggregate feature.

    Returns ``(prem_f, rec_f, comm_f)`` -- positive-magnitude vectorized
    functions of the feature's subject loss ``x`` (the aggregate the
    decorated layer attaches to): the ceded premium (swing's
    ``phi(g(x))``, else the constant deposit), the recovery (corridor's
    adjusted cession, else the layer ceder), and the commission credit
    (slide / pc's ``phi(LR) * P_C``, else the constant ``cede``
    commission). Constants are wrapped as functions so callers can compose
    without case analysis.
    """
    def _const(v):
        return lambda x: np.full(np.shape(x), float(v), dtype=float)

    if tl == 'ceded_premium':                # swing supplies the premium
        prem_f = lambda x: terms.phi(g_ceder(x))
    else:
        prem_f = _const(P_C)
    if tl == 'ceded_loss':                   # corridor: adjusted recovery
        rec_f = lambda x: terms.phi(g_ceder(x) / P_C) * P_C
    else:
        rec_f = g_ceder
    if tl == 'expense':                      # slide / pc: stochastic credit
        comm_f = lambda x: terms.phi(g_ceder(x) / P_C) * P_C
    else:
        comm_f = _const(C)
    return prem_f, rec_f, comm_f


def build_variable_pnl(agg, *, walk=False, econ=None, expense_spec=None,
                       consideration_label=None, loss_label=None, name=None,
                       label=None):
    """A retro / swing / slide / pc / corridor program as a :class:`PnL`.

    Reads the feature attached to ``agg`` by the underwriter
    (``variable_terms`` / ``variable_layer`` / ``variable_gross_premium`` /
    ``variable_ceded_premium`` / ``variable_commission``) and books the
    cash-flow parts over the engine's 1-D density. The feature's subject is
    **what the engine emits**: the gross aggregate on an unreinsured book,
    the net-of-occurrence aggregate when a guaranteed-cost occurrence
    program inures ([Var-Feature-Composed-With-Occ-Program],
    ``dev/done/plan-pnl-faces-punchlist.md``). Two faces
    ([Decision-2D-Is-Computation-Only] --
    these features stay 1-D):

    * ``walk=False`` (the ``pnl`` face) -- the **consolidated** single-group
      net view: one net-premium consideration leg (the feature's map folded
      in, plus the inuring occurrence program's constants
      ``- pc_occ + c_occ``) and the net loss (``x - recovery(x)``,
      corridor-adjusted where applicable). Retro has no cession, so its
      consolidated ledger is the plain shape with the stochastic
      ``terms.phi`` premium (the acceptance pair).
    * ``walk=True`` (the ``xpnl`` face) -- the step tower. With no
      occurrence program: the per-atom two-group ledger over the shared
      atoms. With an inuring occurrence program: the per-atom tower over
      the occurrence ``(gross, ceded)`` joint, gross -> ceded occ ->
      feature cover -> All (the feature's subject is the net-of-occ
      ``max(L - C, 0)``). Scenario (``κ``) ladder either way -- every
      column foots.

    Parameters
    ----------
    econ : dict, optional
        The resolved cession economics (``pc_occ`` / ``c_occ`` for the
        inuring occurrence program; ``pc_agg`` / ``c_agg`` mirror the
        feature deposit / commission). ``None`` -> all zero.

    Returns
    -------
    PnL
    """
    terms = agg.variable_terms
    P_G = float(agg.variable_gross_premium)
    P_C = float(getattr(agg, 'variable_ceded_premium', 0.0))
    C = float(getattr(agg, 'variable_commission', 0.0))
    layer = getattr(agg, 'variable_layer', None)
    pc_occ = float(econ.get('pc_occ', 0.0)) if econ else 0.0
    c_occ = float(econ.get('c_occ', 0.0)) if econ else 0.0
    has_occ = getattr(agg, 'occ_reins', None) is not None
    prem_key = consideration_label or 'premium'
    loss_key = loss_label or 'Loss'
    tl = terms.target_leg
    feature = type(terms).__name__
    if layer is None or not walk:
        pnl = _build_variable_consolidated(
            agg, terms, P_G, P_C, C, layer, tl, expense_spec,
            consideration_label, prem_key, loss_key, name, label,
            pc_occ=pc_occ, c_occ=c_occ)
    elif has_occ:
        pnl = _build_variable_walk_occ(
            agg, terms, P_G, P_C, C, layer, tl, expense_spec, prem_key,
            loss_key, name, label, pc_occ=pc_occ, c_occ=c_occ)
    else:
        pnl = _build_variable_walk(
            agg, terms, P_G, P_C, C, layer, tl, expense_spec, prem_key,
            loss_key, name, label)
    if econ is not None:
        pnl.economics = dict(econ)
    subject = ('the net-of-occurrence density' if has_occ
               else 'the gross density')
    if walk and layer is not None and has_occ:
        face, ladder = ('per-atom walk over the occurrence joint (xpnl)',
                        'scenario (κ) ladder -- one shared joint')
    elif walk and layer is not None:
        face, ladder = ('walk (xpnl)',
                        'scenario (κ) ladder -- one shared source')
    else:
        face, ladder = ('consolidated',
                        'scenario (κ) ladder -- one shared source')
    pnl._construction_description = (
        f'Variable-rating {feature} pnl, {face} face; the feature\'s subject '
        f'is {subject}; {ladder}; the feature changes one leg\'s map, never '
        'the machinery.')
    pnl._construction_explanation = (
        pnl._construction_description + '\n'
        f'Economics: gross premium {P_G:g}, feature deposit {P_C:g}, '
        f'commission {C:g}'
        + (f'; inuring occurrence program ceded premium {pc_occ:g}, '
           f'commission {c_occ:g}' if has_occ else '')
        + f'; terms {terms!r}.\n'
        'Booking: sell books +consideration / -obligation'
        + (', buy the contra' if walk and layer is not None else '')
        + ('; loss-basis LAE books deterministic rate * E[gross loss] on '
           'the consolidated face (the gross loss is not measurable on the '
           'net-of-occurrence subject) and stochastic rate * l on the walk '
           '(axis 0 of the joint carries it).'
           if has_occ else
           '; loss-basis LAE is stochastic rate * x (the gross loss is '
           'on-source).')
        + '\n\n' + pnl._replay_block())
    return pnl


def _build_variable_consolidated(agg, terms, P_G, P_C, C, layer, tl,
                                 expense_spec, consideration_label, prem_key,
                                 loss_key, name, label, *, pc_occ=0.0,
                                 c_occ=0.0):
    """The consolidated (single-group) face of a variable-rating program.

    The source is the engine's own density -- the feature's subject (gross,
    or net-of-occ when an occurrence program inures). The inuring program's
    guaranteed-cost economics fold into the net premium as the constants
    ``- pc_occ + c_occ``; loss-basis LAE books deterministic when the gross
    loss is off-source ([Var-Feature-Composed-With-Occ-Program]).
    """
    from . import _reinsurance
    has_occ = getattr(agg, 'occ_reins', None) is not None
    if layer is None:
        # retro: nothing is ceded, so the ledger is the plain shape with the
        # stochastic phi premium; labels stay untouched
        # ([Flag-Net-Premium-Leg-Label]).
        cons = [Leg(prem_key, terms.phi if tl == 'gross_premium' else P_G, kind='premium')]
        obl = [Leg(loss_key, lambda x: x, kind='loss')]
        obl += _expense_legs(agg, expense_spec, P_G, on_source_loss=True,
                             taken={leg.label for leg in cons + obl})
        return PnL(name=name or agg.name, role='sell', source=agg,
                   consideration=cons, obligation=obl, result_name='margin',
                   label=label)
    g_ceder, _netter = _reinsurance.make_ceder_netter([layer])
    net_prem_key = (f'{consideration_label} (net)' if consideration_label
                    else 'net premium')
    net_loss_key = f'{loss_key} (net)'
    occ_shift = c_occ - pc_occ            # the inuring GC occ constants
    if tl == 'ceded_premium':                # swing: stochastic ceded premium
        net_prem = lambda x: P_G + occ_shift - terms.phi(g_ceder(x)) + C
    elif tl == 'expense':                    # slide / pc: stochastic credit
        net_prem = lambda x: (P_G + occ_shift - P_C
                              + terms.phi(g_ceder(x) / P_C) * P_C)
    else:                                    # corridor: fixed split
        net_prem = P_G + occ_shift - P_C + C
    if tl == 'ceded_loss':                   # corridor: adjusted recovery
        net_loss = lambda x: x - terms.phi(g_ceder(x) / P_C) * P_C
    else:
        net_loss = lambda x: x - g_ceder(x)
    cons = [Leg(net_prem_key, net_prem, kind='premium')]
    obl = [Leg(net_loss_key, net_loss, kind='loss')]
    obl += _expense_legs(agg, expense_spec, P_G,
                         on_source_loss=not has_occ,
                         taken={net_prem_key, net_loss_key})
    return PnL(name=name or agg.name, role='sell', source=agg,
               consideration=cons, obligation=obl, result_name='margin',
               label=label)


def _build_variable_walk(agg, terms, P_G, P_C, C, layer, tl, expense_spec,
                         prem_key, loss_key, name, label):
    """The walk (xpnl) face of a variable-rating program: gross ``sell``
    group + the cession ``buy`` group over the shared gross atoms."""
    from . import _reinsurance
    cons = [Leg(prem_key, P_G, kind='premium')]
    obl = [Leg(loss_key, lambda x: x, kind='loss')]
    obl += _expense_legs(agg, expense_spec, P_G, on_source_loss=True,
                         taken={leg.label for leg in cons + obl})
    groups = [Group('Gross', 'sell', cons, obl)]
    g_ceder, _netter = _reinsurance.make_ceder_netter([layer])
    base = _tier_label(agg, 'agg_reins', 'ceded agg')
    if tl == 'ceded_premium':            # swing: stochastic ceded premium
        c_cons = [Leg(f'{base} premium',
                      lambda x: terms.phi(g_ceder(x)), kind='premium')]
    else:
        c_cons = [Leg(f'{base} premium', P_C, kind='premium')]
    if tl == 'ceded_loss':               # corridor: adjusted recovery
        c_obl = [Leg(f'{base} recovery',
                     lambda x: terms.phi(g_ceder(x) / P_C) * P_C, kind='recovery')]
    else:
        c_obl = [Leg(f'{base} recovery', g_ceder, kind='recovery')]
    if tl == 'expense':                  # slide / profit commission
        comm_key = ('sliding commission'
                    if type(terms).__name__ == 'SlideTerms'
                    else 'profit commission')
        c_obl.append(Leg(comm_key,
                         lambda x: terms.phi(g_ceder(x) / P_C) * P_C, kind='commission'))
    elif C:
        c_obl.append(Leg(f'{base} commission', C, kind='commission'))
    groups.append(Group(base, 'buy', c_cons, c_obl))
    return PnL(name=name or agg.name, source=agg, groups=groups,
               result_name='margin', label=label)


def _build_variable_walk_occ(agg, terms, P_G, P_C, C, layer, tl,
                             expense_spec, prem_key, loss_key, name, label,
                             *, pc_occ=0.0, c_occ=0.0):
    """The walk face of a variable-rating program with an inuring
    guaranteed-cost occurrence program: a per-atom tower over the
    occurrence ``(gross, ceded)`` joint
    ([Var-Feature-Composed-With-Occ-Program]).

    Gross -> ceded occ -> feature cover -> All, every leg a function of the
    one joint (the feature's subject is the net-of-occ ``max(L - C, 0)``),
    so the ladder is the scenario (``\u03ba``) pass and every column foots.
    LAE books stochastic ``rate * l``. The joint's grid caveat and the
    :class:`~aggregate.constants.CoarseJointGridWarning` guard apply as for
    the guaranteed-cost walk.
    """
    from . import _reinsurance
    g_ceder, _netter = _reinsurance.make_ceder_netter([layer])
    prem_f, rec_f, comm_f = _feature_maps(terms, tl, g_ceder, P_C, C)
    occ_base = _tier_label(agg, 'occ_reins', 'ceded occ')
    agg_base = _tier_label(agg, 'agg_reins', 'ceded agg')
    biv = agg.occ_bivariate(views=('gross', 'ceded'))
    check_joint_grid_adequacy(biv.bivariate.bs_ceded, biv.bivariate.bs_net,
                              None, [layer])

    def n(l, c):
        # the feature's subject: the net-of-occ aggregate (clip at 0 for
        # off-support joint cells c > l, which carry ~no mass)
        return np.maximum(l - c, 0.0)

    gross_obl = [Leg(loss_key, lambda l: l, kind='loss')]
    gross_obl += _expense_legs(agg, expense_spec, P_G, on_source_loss=True,
                               taken={prem_key, loss_key})
    groups = [Group('Gross', 'sell', [Leg(prem_key, P_G, kind='premium')], gross_obl)]
    occ_obl = [Leg(f'{occ_base} recovery', lambda l, c: c, is2d=True, kind='recovery')]
    if c_occ:
        occ_obl.append(Leg(f'{occ_base} commission', c_occ, kind='commission'))
    groups.append(Group(occ_base, 'buy',
                        [Leg(f'{occ_base} premium', pc_occ, kind='premium')], occ_obl))
    feat_obl = [Leg(f'{agg_base} recovery',
                    lambda l, c: rec_f(n(l, c)), is2d=True, kind='recovery')]
    if tl == 'expense':
        comm_key = ('sliding commission'
                    if type(terms).__name__ == 'SlideTerms'
                    else 'profit commission')
        feat_obl.append(Leg(comm_key, lambda l, c: comm_f(n(l, c)),
                            is2d=True, kind='commission'))
    elif C:
        feat_obl.append(Leg(f'{agg_base} commission', C, kind='commission'))
    if tl == 'ceded_premium':                     # swing
        feat_cons = [Leg(f'{agg_base} premium',
                         lambda l, c: prem_f(n(l, c)), is2d=True, kind='premium')]
    else:
        feat_cons = [Leg(f'{agg_base} premium', P_C, kind='premium')]
    groups.append(Group(agg_base, 'buy', feat_cons, feat_obl))
    return PnL(name=name or agg.name, source=biv.bivariate, groups=groups,
               result_name='margin', label=label)


def agg_tier_maps(terms, agg_recovery, agg_ceded_premium, agg_feature_terms):
    """The aggregate tier's three cash-flow maps over the ``(L, R)`` joint.

    Relocated off ``ReinstatementAnalysis._agg_tier_maps`` at the
    [Decommission-Analysis-Classes] refactor. Returns
    ``(recovery, premium, commission)`` -- vectorized ``f(l, r)`` closures
    (``commission`` is ``None`` when there is no stochastic commission). The
    raw tier recovery is ``g(max(L - A(R), 0))`` (the ceder input clipped at
    0: off-support joint cells ``r > l`` carry ~no mass but still evaluate,
    and the piecewise-linear ceder is only defined on ``[0, inf)``). A
    variable feature ([Var-Feature-Composed-With-Occ-Program]) swaps one map:

    * swing (``target_leg == 'ceded_premium'``): premium ``phi(g_rec)``;
    * slide / profit commission (``'expense'``): commission
      ``phi(g_rec / P_C) * P_C``;
    * corridor (``'ceded_loss'``): recovery ``phi(g_rec / P_C) * P_C``.

    Guaranteed cost: premium is the constant ``agg_ceded_premium`` (returned
    broadcast-shaped so composite legs stay vectorized).

    Parameters
    ----------
    terms : ReinstatementTerms
        The occurrence basis owning the annual-cap recovery ``A(R)``.
    agg_recovery : callable
        The aggregate cover's vectorized ceder ``g`` on the net-of-occurrence
        loss.
    agg_ceded_premium : float
        The deterministic aggregate ceded premium ``P_C``.
    agg_feature_terms : ContractTerms or None
        A variable feature on the aggregate cover, or ``None`` (guaranteed cost).
    """
    A = terms.recovery
    g0 = agg_recovery
    pc = agg_ceded_premium
    ft = agg_feature_terms

    def g_raw(l, r):
        return g0(np.maximum(l - A(r), 0.0))

    tl = getattr(ft, 'target_leg', None)
    if tl == 'ceded_loss':                    # corridor
        rec = lambda l, r: ft.phi(g_raw(l, r) / pc) * pc
    else:
        rec = g_raw
    if tl == 'ceded_premium':                 # swing
        prem = lambda l, r: ft.phi(g_raw(l, r))
    else:
        prem = lambda l, r: pc + 0.0 * np.asarray(r)
    comm = ((lambda l, r: ft.phi(g_raw(l, r) / pc) * pc)
            if tl == 'expense' else None)
    return rec, prem, comm


def build_reinstatement_source(agg, *, terms=None, gross_premium=None,
                               bs=None, log2_x=None, log2_y=None):
    r"""Build the ``(L, R)`` joint source and resolved economics for a
    reinstatements program.

    Relocated off ``Aggregate.reinstatement_analysis`` at the
    [Decommission-Analysis-Classes] refactor -- the orchestration the P&L
    builders need. Validates the single occurrence layer, resolves the
    reinstatement terms and gross premium from the engine (or the passed
    overrides), builds the ``('gross', 'ceded')`` occurrence joint (axis 0
    gross loss ``L``, axis 1 unlimited ceded recovery ``R``), runs the
    joint-grid adequacy guard, and builds the subsequent-aggregate-cover
    ceder ``g``.

    Parameters
    ----------
    agg : Aggregate
        The reinstated engine (carries ``occ_reins`` / ``agg_reins`` and the
        DecL-attached ``reinstatement_terms`` / ``reinstatement_gross_premium``).
    terms : ReinstatementTerms, optional
        Override for ``agg.reinstatement_terms``.
    gross_premium : float, optional
        Override for the resolved gross premium ``P_G``.
    bs, log2_x, log2_y : optional
        Forwarded to :meth:`Aggregate.occ_bivariate` for the joint grid.

    Returns
    -------
    (source, terms, gross_premium, agg_recovery)
        The ``BivariateDistribution`` joint, the resolved terms, the resolved
        gross premium, and the aggregate-cover ceder (``None`` if no agg cover).
    """
    from . import _reinsurance
    if agg.occ_reins is None:
        raise ValueError(
            'reinstatements require occurrence reinsurance; none configured '
            'on this aggregate.')
    if len(agg.occ_reins) != 1:
        raise ValueError(
            f'reinstatements require a single occurrence layer; found '
            f'{len(agg.occ_reins)} ([reins-single-layer]). A reinstated '
            'layer cannot share the occurrence tier with other variable '
            'layers (the engine ceiling is the 2-D (L, R) joint).')
    if terms is None:
        terms = getattr(agg, 'reinstatement_terms', None)
    if terms is None:
        raise ValueError(
            'reinstatements need a ReinstatementTerms, or a DecL '
            'reinstatements clause that sets self.reinstatement_terms.')
    if gross_premium is None:
        # the gross premium stashed by a DecL ``pnl ... reinstatements``
        # build (the consideration), then the exposure ``premium`` clause.
        gross_premium = getattr(agg, 'reinstatement_gross_premium', None)
    if gross_premium is None:
        gross_premium = getattr(agg, 'exp_premium', None)
    if gross_premium is None or gross_premium == 0:
        raise ValueError(
            'reinstatements need a gross premium (no premium clause on the '
            'aggregate to default from).')
    biv = agg.occ_bivariate(views=('gross', 'ceded'), bs=bs,
                            log2_x=log2_x, log2_y=log2_y)
    # [Reinst-Joint-Grid-Adequacy]: warn when a treaty kink region spans too
    # few buckets of the (coarse, budget-sized) joint -- a kinked map on too
    # few buckets carries an O(bs) bias the audits cannot see.
    check_joint_grid_adequacy(biv.bivariate.bs_ceded, biv.bivariate.bs_net,
                              terms, agg.agg_reins)
    # a subsequent aggregate cover is a deterministic pushforward of the SAME
    # joint via the net-of-occurrence loss L - A(R); build its ceder g. R
    # stays unlimited (the cap lives only in terms).
    agg_recovery = None
    if agg.agg_reins is not None:
        ceder, _netter = _reinsurance.make_ceder_netter(agg.agg_reins)
        agg_recovery = ceder
    return biv.bivariate, terms, gross_premium, agg_recovery


def build_reinstatement_pnl(agg, *, source, terms, gross_premium,
                            agg_recovery=None, agg_ceded_premium=0.0,
                            agg_feature_terms=None, occ_commission=0.0,
                            agg_commission=0.0, walk=False, expense_spec=None,
                            gcn_economics=None, consideration_label=None,
                            loss_label=None, name=None, label=None):
    """An occurrence-reinstatements program as a :class:`PnL`, either face.

    Both faces are per-atom ledgers over the one shared ``(L, R)`` joint
    ([One-2D-Source]) -- gross annual loss ``L`` on axis 0, unlimited
    occurrence recovery ``R`` on axis 1 -- so the stats ladder is the
    scenario (``κ``) pass and loss-basis LAE stays stochastic ``rate * l``
    in both:

    * ``walk=True`` (the ``xpnl`` face) -- the step tower: the gross
      ``sell`` group, the occurrence cession ``buy`` group with the
      genuinely stochastic ceded premium ``D + h(R)`` and recovery ``A(R)``,
      and -- for a subsequent aggregate cover -- a third ``buy`` group whose
      cash flows ride the same joint via the tier maps (no new dimension;
      a variable feature swaps exactly one map,
      [Var-Feature-Composed-With-Occ-Program]).
    * ``walk=False`` (the ``pnl`` face) -- the **consolidated** single-group
      net view ([Decision-PnL-Is-Consolidated], closing [2D-Deferred]): one
      stochastic net-premium leg ``P_G - D - h(R) - P_agg(L, R) +
      commissions`` and the net-of-everything loss
      ``L - A(R) - REC_agg(L, R)``, both deterministic pushforwards of the
      same joint -- so the two faces agree on the net position **exactly**
      (no engine drift; one joint).


    Parameters
    ----------
    agg : Aggregate
        The engine (label source: ``occ_reins`` / ``agg_reins`` labels).
    source : BivariateDistribution
        The ``(L, R)`` joint (from :func:`build_reinstatement_source`).
    terms : ReinstatementTerms
        The reinstatement basis owning ``recovery`` / ``reinstatement_premium``.
    gross_premium : float
        Fixed gross premium ``P_G``.
    agg_recovery : callable, optional
        The subsequent aggregate cover's ceder ``g`` (``None`` if no agg tier).
    agg_ceded_premium : float, default 0.0
        The aggregate cover's deterministic ceded premium ``P_C``.
    agg_feature_terms : ContractTerms, optional
        A variable feature on the aggregate cover (rides the same joint).
    occ_commission, agg_commission : float, default 0.0
        Deterministic ceding commissions on the occurrence / aggregate covers.
    walk : bool, default False
        ``True`` -> the step tower (``xpnl``); ``False`` -> the consolidated
        net view (``pnl``).
    gcn_economics : dict, optional
        The resolved cession economics; attached as ``pnl.economics``.

    Returns
    -------
    PnL
    """
    if not walk:
        return _build_reinstatement_consolidated(
            agg, source=source, terms=terms, gross_premium=gross_premium,
            agg_recovery=agg_recovery, agg_ceded_premium=agg_ceded_premium,
            agg_feature_terms=agg_feature_terms, occ_commission=occ_commission,
            agg_commission=agg_commission, expense_spec=expense_spec,
            gcn_economics=gcn_economics,
            consideration_label=consideration_label, loss_label=loss_label,
            name=name, label=label)
    t = terms
    P_G, D = gross_premium, t.deposit
    A, h = t.recovery, t.reinstatement_premium
    occ_base = _tier_label(agg, 'occ_reins', 'ceded occ')
    agg_base = _tier_label(agg, 'agg_reins', 'ceded agg')
    prem_key = consideration_label or 'premium'
    loss_key = loss_label or 'Loss'
    cons = [Leg(prem_key, P_G, kind='premium')]
    obl = [Leg(loss_key, lambda l: l, kind='loss')]       # 1-D: reads axis 0 = gross loss
    obl += _expense_legs(agg, expense_spec, P_G, on_source_loss=True,
                         taken={leg.label for leg in cons + obl})
    groups = [Group('Gross', 'sell', cons, obl)]
    occ_obl = [Leg(f'{occ_base} recovery', lambda l, r: A(r), is2d=True, kind='recovery')]
    if occ_commission:
        occ_obl.append(Leg(f'{occ_base} commission', occ_commission, kind='commission'))
    groups.append(Group(
        occ_base, 'buy',
        [Leg(f'{occ_base} premium', lambda l, r: D + h(r), is2d=True, kind='premium')],
        occ_obl))
    pc = 0.0
    if agg_recovery is not None:
        # the tier's three cash-flow maps ride the same joint; a variable
        # feature on the agg cover swaps exactly one of them
        # ([Var-Feature-Composed-With-Occ-Program]).
        pc = agg_ceded_premium
        rec, prem, comm = agg_tier_maps(terms, agg_recovery, agg_ceded_premium,
                                        agg_feature_terms)
        ft = agg_feature_terms
        a_obl = [Leg(f'{agg_base} recovery', rec, is2d=True, kind='recovery')]
        if comm is not None:
            comm_key = ('sliding commission'
                        if type(ft).__name__ == 'SlideTerms'
                        else 'profit commission')
            a_obl.append(Leg(comm_key, comm, is2d=True, kind='commission'))
        elif agg_commission:
            a_obl.append(Leg(f'{agg_base} commission', agg_commission, kind='commission'))
        if getattr(ft, 'target_leg', None) == 'ceded_premium':   # swing
            a_cons = [Leg(f'{agg_base} premium', prem, is2d=True, kind='premium')]
        else:
            a_cons = [Leg(f'{agg_base} premium', pc, kind='premium')]
        groups.append(Group(agg_base, 'buy', a_cons, a_obl))
    pnl = PnL(name=name or agg.name, source=source, groups=groups,
              result_name='margin', label=label)
    if gcn_economics is not None:
        pnl.economics = dict(gcn_economics)
    feat = type(agg_feature_terms).__name__ \
        if agg_feature_terms is not None else None
    pnl._construction_description = (
        f'Occurrence-reinstatements walk (xpnl): 2-D group ledger over the '
        f'shared (L, R) joint ([One-2D-Source]): {len(groups)} groups; the '
        f'ceded premium D + h(R) = {D:g} + reinstatement premium is '
        'genuinely stochastic; '
        + (f'the aggregate cover is {feat}-rated (its map rides the same '
           'joint); ' if feat else '')
        + 'scenario (κ) ladder (one shared joint). The consolidated net '
        'view is the pnl face.')
    pnl._construction_explanation = (
        pnl._construction_description + '\n'
        f'Economics: gross premium {P_G:g}, deposit {D:g}'
        + (f', aggregate ceded premium {pc:g}' if pc else '')
        + f'; committed scale = gross - deposit - pc_agg = {P_G - D - pc:g}.'
        '\nRows: the gross loss reads axis 0 of the joint (loss-basis LAE '
        'stochastic rate * l); the occurrence recovery A(R) and premium '
        'D + h(R) read axis 1; a subsequent aggregate cover recovers '
        'g(max(L - A(R), 0)) on the same joint (no new dimension).\n\n'
        + pnl._replay_block())
    return pnl


def _build_reinstatement_consolidated(agg, *, source, terms, gross_premium,
                                      agg_recovery=None, agg_ceded_premium=0.0,
                                      agg_feature_terms=None, occ_commission=0.0,
                                      agg_commission=0.0, expense_spec=None,
                                      gcn_economics=None,
                                      consideration_label=None,
                                      loss_label=None, name=None,
                                      label=None):
    """The consolidated (``pnl``) face of a reinstatements program.

    One ``sell`` group of 2-D legs over the same ``(L, R)`` joint the walk
    uses ([Decision-PnL-Is-Consolidated], closing [2D-Deferred]):

    * **net premium** ``P_G - D - h(R) - P_agg(L, R) + c_occ +
      commissions`` -- genuinely stochastic (the reinstatement premium, and
      a swing-rated or slide/pc-commissioned aggregate tier ride along);
    * **loss (net)** ``-(L - A(R) - REC_agg(L, R))`` -- net of everything;
    * expenses; loss-basis LAE stays stochastic ``rate * l`` (axis 0
      carries the gross loss -- unlike the guaranteed-cost consolidated
      face, nothing is off-source here; cf. [Consolidated-LAE-Off-Source]).

    Because both faces are pushforwards of the ONE joint, the consolidated
    net position equals the walk's grand result **exactly**.
    """
    t = terms
    P_G, D = gross_premium, t.deposit
    A, h = t.recovery, t.reinstatement_premium
    c_occ = occ_commission
    c_agg = agg_commission
    prem_key = (f'{consideration_label} (net)' if consideration_label
                else 'net premium')
    loss_key = f'{loss_label or "Loss"} (net)'
    if agg_recovery is not None:
        pc = agg_ceded_premium
        rec, prem, comm = agg_tier_maps(terms, agg_recovery, agg_ceded_premium,
                                        agg_feature_terms)
        if comm is not None:
            net_prem = lambda l, r: (P_G - D - h(r) - prem(l, r)
                                     + c_occ + comm(l, r))
        else:
            net_prem = lambda l, r: (P_G - D - h(r) - prem(l, r)
                                     + c_occ + c_agg)
        net_loss = lambda l, r: (l - A(r)) - rec(l, r)
    else:
        pc = 0.0
        net_prem = lambda l, r: P_G - D - h(r) + c_occ
        net_loss = lambda l, r: l - A(r)
    cons = [Leg(prem_key, net_prem, is2d=True, kind='premium')]
    obl = [Leg(loss_key, net_loss, is2d=True, kind='loss')]
    obl += _expense_legs(agg, expense_spec, P_G, on_source_loss=True,
                         taken={prem_key, loss_key})
    pnl = PnL(name=name or agg.name, role='sell', source=source,
              consideration=cons, obligation=obl, result_name='margin',
              label=label)
    if gcn_economics is not None:
        pnl.economics = dict(gcn_economics)
    feat = type(agg_feature_terms).__name__ \
        if agg_feature_terms is not None else None
    pnl._construction_description = (
        'Consolidated reinstatements pnl over the shared (L, R) joint '
        '([Decision-PnL-Is-Consolidated], closing [2D-Deferred]): one sell '
        f'group; net premium P_G - D - h(R){" - P_agg" if pc or feat else ""}'
        ' + commissions is genuinely stochastic; '
        + (f'the aggregate cover is {feat}-rated (its map rides the same '
           'joint); ' if feat else '')
        + 'scenario (κ) ladder (one shared joint). The step walk is one '
        'xpnl away, and agrees on the net position exactly (one joint).')
    pnl._construction_explanation = (
        pnl._construction_description + '\n'
        f'Economics: gross premium {P_G:g}, deposit {D:g}'
        + (f', aggregate ceded premium {pc:g}' if pc else '')
        + f'; committed scale = gross - deposit - pc_agg = {P_G - D - pc:g}.'
        '\nBooking: net premium and net loss are deterministic pushforwards '
        'of the joint; loss-basis LAE is stochastic rate * l (axis 0 '
        'carries the gross loss).\n\n' + pnl._replay_block())
    return pnl
