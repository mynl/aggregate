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
    consolidated over the gross density: the net premium / net loss legs are
    the feature's maps folded together.

* **``xpnl`` -- "how did I get there?"** The walk: gross -> each cover ->
  Total, a multi-group :class:`PnL` ([Decision-XPnL-Is-A-Recipe] -- a way of
  building, not a new type).

  * :func:`build_xpnl_walk` -- the guaranteed-cost walk,
    **marginal-stitched** ([GC-Tower-Marginal-Stitch]): every row is an
    affine transform of an exact ``reins_density_df`` marginal, derived rows
    read the engine's own net marginals (never cross-row sums), means foot by
    linearity, the ladder stays marginal (plain ``P`` headers).
  * :func:`build_variable_pnl` (``walk=True``) -- the variable-rating
    two-group per-atom ledger over the gross density.
  * :func:`build_reinstatement_pnl` -- the occurrence-reinstatements per-atom
    2-D ledger over the ``(L, R)`` joint; serves ``xpnl``, and (transitional
    [2D-Deferred] carve-out) ``pnl`` as well until the consolidated 2-D route
    lands.

The expense resolvers (:func:`resolve_expense` / :func:`_resolve_expense_split`)
relocated here from ``_pnl.py`` unchanged: ``_resolve_expense_split`` is the
only resolver feeding legs (loss-basis expense is stochastic ``rate * x``
wherever the loss is on-source); ``resolve_expense`` survives for scalar needs
(the analyses' deterministic economics) but never feeds a leg.
"""
from __future__ import annotations

import numpy as np

from ._pnl import Leg, Group, PnL, _ledger_plan

__all__ = ['build_plain_pnl', 'build_consolidated_pnl', 'build_xpnl_walk',
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
# Stitched-row helpers ([GC-Tower-Marginal-Stitch]): every guaranteed-cost
# walk row is a signed affine transform of one exact engine marginal
# ----------------------------------------------------------------------
def _affine_marginal_entry(agg, perspective, a, b, label):
    """One stitched ledger row ``a * X + b`` of an exact engine marginal.

    Returns the ``(GridDistribution, exact_mean, exact_sd)`` triple the
    kernel's stitched construction consumes. The marginal's probabilities are
    normalized (a clipped tail is a discretization artifact -- the same
    rationale as the kernel's atoms route); the transform is monotone, so the
    pushforward is a relabeling of the grid (reversed when ``a < 0``), exact
    -- no re-gridding, ever.
    """
    from ._grid_distribution import GridDistribution
    rd = agg.reins_density_df
    x = rd['loss'].to_numpy(dtype=float)
    p = rd[_GCN_LOSS_MARGINAL[perspective]].to_numpy(dtype=float)
    tot = float(p.sum())
    if tot > 0:
        p = p / tot
    v = a * x + b
    if a < 0:
        v = v[::-1]
        p = p[::-1]
    m = float((v * p).sum())
    var = float((v * v * p).sum()) - m * m
    sd = var ** 0.5 if var > 0 else 0.0
    gd = GridDistribution(v, p, bs=None, name=label, is_loss_value=False)
    return gd, m, sd


def _constant_entry(b, label):
    """A constant stitched ledger row (a fixed premium / expense): a
    one-atom :class:`GridDistribution` with exact mean ``b`` and sd 0."""
    from ._grid_distribution import GridDistribution
    gd = GridDistribution(np.array([float(b)]), np.array([1.0]), bs=None,
                          name=label, is_loss_value=False)
    return gd, float(b), 0.0


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
    pnl = PnL(name=name, role='sell', source=engine,
              consideration=[Leg(consideration_label or 'consideration',
                                 cons)],
              obligation=obligation, result_name='margin',
              label=label)
    pnl._construction_description = (
        f'Plain pnl: one sell group over the engine\'s own density '
        f'({type(engine).__name__}); consideration '
        f'{"loss-sensitive f(x)" if callable(cons) else f"{cons:g}"}; '
        'scenario (κ) ladder (single shared source).')
    pnl._construction_explanation = (
        pnl._construction_description + '\n'
        'Booking: sell -- +consideration, -obligation; loss-basis LAE is '
        'stochastic rate * x (the loss is on-source). For a '
        'reinsurance-bearing engine the density here is the net -- what '
        'comes out of the engine.\n\n' + pnl._replay_block())
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
    loss_key = f'{loss_label or "loss"} (net)'
    persp = 'net_agg' if has_agg else 'net_occ'
    source = _marginal_gd(agg, persp)
    cons = [Leg(prem_key, net_premium)]
    obl = [Leg(loss_key, lambda x: x)]
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
    """The guaranteed-cost ``xpnl`` **walk**: a marginal-stitched multi-group
    :class:`PnL` ([XPnL-Walk-Recipe] with [GC-Tower-Marginal-Stitch]).

    The step tower -- gross -> each cover -> Total -- assembled over the
    generic group ledger the kernel already has. No joint exists (the
    cessions are not functions of one observable), so every row is supplied
    **gd-backed**: an affine transform of one exact engine marginal from
    ``reins_density_df`` (gross, ceded-occ, net-occ, ceded-agg, net-net).
    Derived rows (step results, running nets, grand rows) read the engine's
    own net marginals -- never cross-row sums, which do not exist without
    atoms. Consequences, all documented on the sheets:

    * means foot by linearity; SDs / percentiles are exact **per row**;
    * the stats ladder stays **marginal** (plain ``P`` headers -- the
      on-sheet signature of [Decision-Kappa-Shared-Source-Rule]);
    * a loss-basis (LAE) expense books as its deterministic
      ``rate * E[gross loss]`` (any stitched tower is all-marginal -- no
      hybrid, even where a sub-chain could be per-atom);
    * the ``total impact`` row is a per-statistic delta (grand result and
      gross step ride different marginals): the mean is exact, the other
      cells are deltas of row statistics.

    Step labels per [Decision-Total-Step-Stays-Total]: the base step is the
    engine's declared label (fallback ``'gross'``); cover steps are the reins
    ``as`` labels (fallback ``'ceded occ'`` / ``'ceded agg'``); the grand
    step's index key stays the structural ``'Total'``.

    Returns
    -------
    PnL
        A plain multi-group :class:`PnL` -- xpnl is a way of building, not a
        new type ([Decision-XPnL-Is-A-Recipe]).
    """
    rd = agg.reins_density_df
    if rd is None:
        raise ValueError(
            "'xpnl' requires reinsurance on the wrapped engine; the aggregate "
            'carries no occurrence / aggregate treaty.')
    has_occ = agg.occ_reins is not None
    has_agg = agg.agg_reins is not None
    p_gross, pc_occ, pc_agg, c_occ, c_agg = _ledger_economics(
        agg, gross, ceded, gcn_economics)
    occ_base = _first_reins_label(agg, 'occ_reins') or 'ceded occ'
    agg_base = _first_reins_label(agg, 'agg_reins') or 'ceded agg'
    base_step = loss_label or 'gross'
    prem_key = consideration_label or 'premium'
    loss_key = loss_label or 'loss'

    # the gross sell group; LAE books deterministic (all-marginal, no hybrid)
    gross_obl = [Leg(loss_key, lambda x: x)]
    gross_obl += _expense_legs(agg, expense_spec, p_gross,
                               on_source_loss=False,
                               taken={prem_key, loss_key})
    e_const = float(sum(leg.func for leg in gross_obl[1:]))
    groups = [Group(base_step, 'sell', [Leg(prem_key, p_gross)], gross_obl)]
    # one buy group per cover, walk order: occ inures before agg
    covers = []                    # (group index, perspective, pc, commission)
    if has_occ:
        obl = [Leg(f'{occ_base} recovery', lambda x: x)]
        if c_occ:
            obl.append(Leg(f'{occ_base} commission', c_occ))
        groups.append(Group(occ_base, 'buy',
                            [Leg(f'{occ_base} premium', pc_occ)], obl))
        covers.append((len(groups) - 1, 'ceded_occ', 'net_occ', pc_occ,
                       c_occ))
    if has_agg:
        obl = [Leg(f'{agg_base} recovery', lambda x: x)]
        if c_agg:
            obl.append(Leg(f'{agg_base} commission', c_agg))
        groups.append(Group(agg_base, 'buy',
                            [Leg(f'{agg_base} premium', pc_agg)], obl))
        covers.append((len(groups) - 1, 'ceded_agg', 'net_agg', pc_agg,
                       c_agg))
    final_net = 'net_agg' if has_agg else 'net_occ'

    # ------------------------------------------------------------------
    # supply every plan row as (marginal perspective | constant, a, b):
    # the sum of a stage's rows is always measurable on one engine marginal
    # (loss_g - R_occ = X_net-occ, etc.) -- the stitch's footing contract.
    # The exact MEAN of every derived row is the signed sum of its
    # constituent leg means (linearity -- the EX column foots exactly); its
    # distribution (SD / percentiles) reads the engine's own marginal, which
    # agrees with that mean up to the engine's FFT / tail-clipping dust
    # (the massive-route pattern: exact mean alongside the realized gd).
    # ------------------------------------------------------------------
    result_name = 'margin'
    plan = _ledger_plan(groups, result_name)
    # per-cover running constants: premium in, expenses and ceded premiums
    # out, commissions back in
    desc = {}
    # gross group (index 0)
    desc[('leg', (0, 'cons', 0))] = (None, 0.0, p_gross)
    desc[('leg', (0, 'obl', 0))] = ('gross', -1.0, 0.0)
    for li, leg in enumerate(gross_obl[1:], start=1):
        desc[('leg', (0, 'obl', li))] = (None, 0.0, -float(leg.func))
    desc[('group_total', (0, 'cons'))] = (None, 0.0, p_gross)
    desc[('group_total', (0, 'obl'))] = ('gross', -1.0, -e_const)
    desc[('group_result', 0)] = ('gross', -1.0, p_gross - e_const)
    run_const = p_gross - e_const
    for gi, ceded_persp, net_persp, pc, comm in covers:
        desc[('leg', (gi, 'cons', 0))] = (None, 0.0, -pc)
        desc[('leg', (gi, 'obl', 0))] = (ceded_persp, 1.0, 0.0)
        if comm:
            desc[('leg', (gi, 'obl', 1))] = (None, 0.0, comm)
        desc[('group_total', (gi, 'cons'))] = (None, 0.0, -pc)
        desc[('group_total', (gi, 'obl'))] = (ceded_persp, 1.0, comm)
        desc[('group_result', gi)] = (ceded_persp, 1.0, comm - pc)
        run_const += comm - pc
        desc[('running_net', gi)] = (net_persp, -1.0, run_const)
    desc[('grand_total', 'cons')] = (None, 0.0, p_gross - pc_occ - pc_agg)
    desc[('grand_total', 'obl')] = (final_net, -1.0,
                                    -e_const + c_occ + c_agg)
    desc[('grand_result', None)] = (final_net, -1.0, run_const)

    # exact leg means, then derived means by linearity (the EX column
    # foots exactly by construction)
    leg_mean = {}
    for key, (persp, a, b) in desc.items():
        if key[0] != 'leg':
            continue
        if persp is None:
            leg_mean[key[1]] = b
        else:
            _gd, m, _sd = _affine_marginal_entry(agg, persp, a, b, 'tmp')
            leg_mean[key[1]] = m

    def _side_mean(gi, side, n_legs):
        return sum(leg_mean[(gi, side, li)] for li in range(n_legs))

    group_mean = {}
    for gi, g in enumerate(groups):
        group_mean[gi] = (_side_mean(gi, 'cons', len(g.consideration))
                          + _side_mean(gi, 'obl', len(g.obligation)))

    def _linear_mean(kind, payload):
        if kind == 'leg':
            return leg_mean[payload]
        if kind == 'group_total':
            gi, side = payload
            g = groups[gi]
            return _side_mean(gi, side, len(g.consideration if side == 'cons'
                                            else g.obligation))
        if kind == 'group_result':
            return group_mean[payload]
        if kind == 'running_net':
            return sum(group_mean[gi] for gi in range(payload + 1))
        if kind == 'grand_total':
            return sum(_side_mean(gi, payload,
                                  len(g.consideration if payload == 'cons'
                                      else g.obligation))
                       for gi, g in enumerate(groups))
        return sum(group_mean.values())          # grand_result

    entries = {}
    for row_label, kind, payload in plan:
        if kind == 'total_impact':
            entries[row_label] = ('delta', result_name,
                                  f'{groups[0].label} result')
            continue
        persp, a, b = desc[(kind, payload)]
        gd, _m, sd = (_constant_entry(b, row_label) if persp is None
                      else _affine_marginal_entry(agg, persp, a, b,
                                                  row_label))
        entries[row_label] = (gd, _linear_mean(kind, payload), sd)

    pnl = PnL(name=name or agg.name, source=agg, groups=groups,
              result_name=result_name, label=label, stitched_rows=entries)
    pnl.economics = dict(gcn_economics) if gcn_economics is not None \
        else {'gross': float(gross), 'ceded': float(ceded)}
    pnl._construction_description = (
        f'xpnl walk: {len(groups)}-step marginal-stitched tower '
        f'({" -> ".join(g.label for g in groups)} -> Total) over the exact '
        'reins_density_df marginals; means foot by linearity, SDs / '
        'percentiles exact per row; marginal (P) ladder -- no shared joint.')
    pnl._construction_explanation = _walk_explanation(
        agg, pnl, groups, covers, p_gross, e_const, final_net)
    return pnl


def _walk_explanation(agg, pnl, groups, covers, p_gross, e_const, final_net):
    """The full [Construction-Introspection] story for a stitched walk."""
    econ = pnl.economics
    step_lines = [
        f"  step {groups[0].label!r} (sell): premium {p_gross:g} in; loss "
        "reads reins_density_df['p_agg_gross']; expenses "
        f'{e_const:g} (loss-basis LAE deterministic -- off-source on a '
        'stitched tower).']
    for gi, ceded_persp, net_persp, pc, comm in covers:
        step_lines.append(
            f'  step {groups[gi].label!r} (buy): ceded premium {pc:g} out; '
            f'recovery reads '
            f'reins_density_df[{_GCN_LOSS_MARGINAL[ceded_persp]!r}]'
            + (f'; commission {comm:g} back' if comm else '')
            + f'; running net reads '
              f'reins_density_df[{_GCN_LOSS_MARGINAL[net_persp]!r}].')
    lines = [
        f'xpnl walk {pnl.label!r} ([XPnL-Walk-Recipe], marginal-stitched).',
        f'Engine: {agg.name!r}'
        + (f' -- occurrence reinsurance {agg.occ_reins}'
           if agg.occ_reins is not None else '')
        + (f' -- aggregate reinsurance {agg.agg_reins}'
           if agg.agg_reins is not None else '') + '.',
        f'Economics resolution: {econ!r}.',
        'Rows, and the marginal each reads:',
        *step_lines,
        'Booking: sell books +consideration / -obligation, buy the contra; '
        'the EX column adds down the sheet exactly (linearity).',
        'Ladder: marginal (plain P headers) -- the rows ride separate '
        'engine marginals, so there is no joint to condition on '
        '([Decision-Kappa-Shared-Source-Rule]); the total impact row is a '
        'per-statistic delta (mean exact).',
        '',
        pnl._replay_block(),
    ]
    return '\n'.join(lines)


# ----------------------------------------------------------------------
# Variable-rating / reinstatement feature builders
# ([Builders-Variable-Features]: a feature never changes the machinery --
#  it changes one leg's function)
# ----------------------------------------------------------------------
def build_variable_pnl(agg, *, walk=False, expense_spec=None,
                       consideration_label=None, loss_label=None, name=None,
                       label=None):
    """A retro / swing / slide / pc / corridor program as a :class:`PnL`.

    Reads the feature attached to ``agg`` by the underwriter
    (``variable_terms`` / ``variable_layer`` / ``variable_gross_premium`` /
    ``variable_ceded_premium`` / ``variable_commission``) and books the
    cash-flow parts over the **gross** 1-D density. Two faces
    ([Decision-2D-Is-Computation-Only] -- these features stay 1-D):

    * ``walk=False`` (the ``pnl`` face) -- the **consolidated** single-group
      net view: one net-premium consideration leg (the feature's map folded
      in: ``P_G - phi(ceded(x))`` for swing, the sliding / profit commission
      credited, the fixed split otherwise) and the net loss
      (``x - recovery(x)``, corridor-adjusted where applicable). Retro has no
      cession, so its consolidated ledger is the plain shape with the
      stochastic ``terms.phi`` premium (the acceptance pair).
    * ``walk=True`` (the ``xpnl`` face) -- the step tower: the gross ``sell``
      group plus a real cession ``buy`` group, with exactly **one** leg
      swapped for the feature's ``terms.phi``-driven map (stochastic ceded
      premium; sliding / profit commission; the corridor-adjusted recovery).

    Either face rides the gross atoms (one shared source), so the stats
    ladder is the scenario (``κ``) pass and loss-basis LAE stays stochastic
    ``rate * x``. Attach the drill-down analysis afterwards
    (``pnl.analysis``).

    Returns
    -------
    PnL
    """
    terms = agg.variable_terms
    P_G = float(agg.variable_gross_premium)
    P_C = float(getattr(agg, 'variable_ceded_premium', 0.0))
    C = float(getattr(agg, 'variable_commission', 0.0))
    layer = getattr(agg, 'variable_layer', None)
    prem_key = consideration_label or 'premium'
    loss_key = loss_label or 'loss'
    tl = terms.target_leg
    feature = type(terms).__name__
    if layer is None or not walk:
        pnl = _build_variable_consolidated(
            agg, terms, P_G, P_C, C, layer, tl, expense_spec,
            consideration_label, prem_key, loss_key, name, label)
    else:
        pnl = _build_variable_walk(
            agg, terms, P_G, P_C, C, layer, tl, expense_spec, prem_key,
            loss_key, name, label)
    face = 'walk (xpnl)' if walk and layer is not None else 'consolidated'
    pnl._construction_description = (
        f'Variable-rating {feature} pnl, {face} face over the gross density '
        f'(one shared source -> scenario (κ) ladder); the feature changes '
        'one leg\'s map, never the machinery.')
    pnl._construction_explanation = (
        pnl._construction_description + '\n'
        f'Economics: gross premium {P_G:g}, ceded premium {P_C:g}, '
        f'commission {C:g}; terms {terms!r}.\n'
        'Booking: sell books +consideration / -obligation'
        + (', buy the contra' if walk and layer is not None else '')
        + '; loss-basis LAE is stochastic rate * x (the gross loss is '
        'on-source).\n\n' + pnl._replay_block())
    return pnl


def _build_variable_consolidated(agg, terms, P_G, P_C, C, layer, tl,
                                 expense_spec, consideration_label, prem_key,
                                 loss_key, name, label):
    """The consolidated (single-group) face of a variable-rating program."""
    from . import _reinsurance
    if layer is None:
        # retro: nothing is ceded, so the ledger is the plain shape with the
        # stochastic phi premium; labels stay untouched
        # ([Flag-Net-Premium-Leg-Label]).
        cons = [Leg(prem_key, terms.phi if tl == 'gross_premium' else P_G)]
        obl = [Leg(loss_key, lambda x: x)]
        obl += _expense_legs(agg, expense_spec, P_G, on_source_loss=True,
                             taken={leg.label for leg in cons + obl})
        return PnL(name=name or agg.name, role='sell', source=agg,
                   consideration=cons, obligation=obl, result_name='margin',
                   label=label)
    g_ceder, _netter = _reinsurance.make_ceder_netter([layer])
    net_prem_key = (f'{consideration_label} (net)' if consideration_label
                    else 'net premium')
    net_loss_key = f'{loss_key} (net)'
    if tl == 'ceded_premium':                # swing: stochastic ceded premium
        net_prem = lambda x: P_G - terms.phi(g_ceder(x)) + C
    elif tl == 'expense':                    # slide / pc: stochastic credit
        net_prem = lambda x: P_G - P_C + terms.phi(g_ceder(x) / P_C) * P_C
    else:                                    # corridor: fixed split
        net_prem = P_G - P_C + C
    if tl == 'ceded_loss':                   # corridor: adjusted recovery
        net_loss = lambda x: x - terms.phi(g_ceder(x) / P_C) * P_C
    else:
        net_loss = lambda x: x - g_ceder(x)
    cons = [Leg(net_prem_key, net_prem)]
    obl = [Leg(net_loss_key, net_loss)]
    obl += _expense_legs(agg, expense_spec, P_G, on_source_loss=True,
                         taken={net_prem_key, net_loss_key})
    return PnL(name=name or agg.name, role='sell', source=agg,
               consideration=cons, obligation=obl, result_name='margin',
               label=label)


def _build_variable_walk(agg, terms, P_G, P_C, C, layer, tl, expense_spec,
                         prem_key, loss_key, name, label):
    """The walk (xpnl) face of a variable-rating program: gross ``sell``
    group + the cession ``buy`` group over the shared gross atoms."""
    from . import _reinsurance
    cons = [Leg(prem_key, P_G)]
    obl = [Leg(loss_key, lambda x: x)]
    obl += _expense_legs(agg, expense_spec, P_G, on_source_loss=True,
                         taken={leg.label for leg in cons + obl})
    groups = [Group('gross', 'sell', cons, obl)]
    g_ceder, _netter = _reinsurance.make_ceder_netter([layer])
    base = _first_reins_label(agg, 'agg_reins') or 'ceded agg'
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
    pnl._construction_description = (
        f'Occurrence-reinstatements 2-D group ledger over the shared (L, R) '
        f'joint ([One-2D-Source]): {len(groups)} groups; the ceded premium '
        f'D + h(R) = {D:g} + reinstatement premium is genuinely stochastic; '
        'scenario (κ) ladder (one shared joint). Serves pnl and xpnl alike '
        'until the consolidated 2-D route lands ([2D-Deferred]).')
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
