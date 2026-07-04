r"""The domain-agnostic profit-and-loss kernel: a source plus a group ledger.

A **P&L** is a probability space plus named accounting functionals
(``dev/plan-yapnl.md``):

* a **source** -- the stochastic generator (1-D, or one shared 2-D joint);
* an ordered list of **groups**, each a mini-P&L: a label, a ``role``
  (``'sell'`` / ``'buy'``), and ordered lists of consideration / obligation
  **legs** (:class:`Leg`) -- every leg an actual cash flow;
* everything else **derived**: per-leg exact :class:`GridDistribution` rows,
  per-group totals and results, running nets, the grand total, signed
  additive exhibits.

All the domain magic happens at the **caller** level (defining the source and
the leg functions); inside :class:`PnL` everything is generic. Insurance is a
caller, not a special case -- the DecL builders live in
:mod:`aggregate._pnl_builders` and translate insurance programs into sources +
groups; the kernel never sees the words gross / ceded / net.

Evaluation invariants
---------------------

* **Per-atom, then group.** Every leg func evaluates over the source atoms;
  totals, results and running nets are per-atom partial sums of the *signed*
  legs. The result is never arithmetic on marginal distributions -- the
  covariance rides for free ("means add, SDs don't" holds on-sheet
  automatically).
* **Signed throughout.** The group ``role`` books each leg into the holder's
  ledger (``sell`` -> ``+consideration, -obligation``; ``buy`` -> the contra),
  so the ``EX`` column adds straight down the sheet and each row's
  distribution is built on the signed values -- ``Pq(-X) = -P(1-q)(X)`` puts
  the adverse tail where the reader expects for both roles, automatically.
* **Exact by default.** A ``bs=0`` leg is an exact irregular
  :class:`GridDistribution` (group atoms by value, sum probability -- GDs need
  no equal spacing and never touch the FFT machinery). A ``bs>0`` leg
  rebuckets onto a regular grid via the shared pushforward machinery and is
  audited in :attr:`PnL.validation_df`.
* **[One-2D-Source].** Any number of ``is2d`` legs, all reading the single
  shared joint; an ``is2d`` leg over a 1-D source is an error, and a 2-D
  source with 1-D legs is fine (they read axis 0). Only one latent dimension
  exists.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd

from .moments import VALIDATION_NOISE, _snap_noise
from ._labeled import LabeledMixin

__all__ = ['Leg', 'Group', 'PnL', 'stack_marginal_pnls']


#: The detailed percentile ladder used by :attr:`PnL.stats_df` /
#: :attr:`PnL.scaled_stats_df` and :func:`stack_marginal_pnls` (the full
#: ladder; the headline :attr:`PnL.summary_df` reports only ``P1`` / ``Median``
#: / ``P99`` off it). The exact ladder is standardized in
#: ``[Reporting-Guidelines]``.
PERCENTILE_LADDER = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)

#: Distortion families evaluated by :meth:`PnL.evaluate` (the standard set minus
#: ``ccoc``, which needs an asset level a P&L does not carry).
_EVAL_FAMILIES = ('ph', 'wang', 'dual', 'tvar')


# ----------------------------------------------------------------------
# The leg / group spec objects
# ----------------------------------------------------------------------
class Leg:
    """One declared cash flow: a label and a map over the source atoms.

    Parameters
    ----------
    label : str
        The ledger row label -- reads like the exhibit row it becomes (label
        first, so ``Leg('LAE', lambda x: 0.05 * x)`` reads naturally). Leg
        labels are the row keys directly; ledger order is declaration order.
    func : float or callable
        The cash-flow magnitude per atom: a constant, a vectorized ``f(x)``
        over a 1-D source (or axis 0 of a 2-D source), or ``f(l, r)`` over the
        shared joint when ``is2d=True``. Magnitudes are as the caller writes
        them; the group's ``role`` supplies the sign.
    bs : float, default 0
        ``0`` -> the leg's distribution is the exact irregular
        :class:`GridDistribution` (group atoms by value, sum probability).
        ``> 0`` -> mean-preserving rebucket onto a regular ``bs`` grid via the
        shared pushforward machinery, audited in :attr:`PnL.validation_df`.
    is2d : bool, default False
        ``func`` reads both axes of the shared 2-D joint. An ``is2d`` leg over
        a 1-D source is an error ([One-2D-Source]).
    """

    def __init__(self, label, func, bs=0, is2d=False):
        if not isinstance(label, str) or not label:
            raise ValueError(
                f'Leg label must be a non-empty string, got {label!r}.')
        if bs < 0:
            raise ValueError(f'Leg bs must be >= 0, got {bs!r} ({label!r}).')
        self.label = label
        self.func = func
        self.bs = float(bs)
        self.is2d = bool(is2d)

    def __repr__(self):
        extra = (f', bs={self.bs:g}' if self.bs else '') + \
            (', is2d=True' if self.is2d else '')
        return f'Leg({self.label!r}{extra})'


def _as_legs(spec, default_label):
    """Normalize a consideration / obligation argument to a list of :class:`Leg`.

    Accepts an ordered ``{label: func}`` dict (the simple-case shorthand), a
    list / tuple of :class:`Leg`, a single :class:`Leg`, or a bare constant /
    callable (one leg named ``default_label``). ``None`` -> ``[]``.
    """
    if spec is None:
        return []
    if isinstance(spec, Leg):
        return [spec]
    if isinstance(spec, dict):
        return [Leg(k, v) for k, v in spec.items()]
    if isinstance(spec, (list, tuple)):
        out = list(spec)
        for leg in out:
            if not isinstance(leg, Leg):
                raise TypeError(
                    f'expected a list of Leg objects, got {type(leg).__name__} '
                    f'({leg!r}); use the {{label: func}} dict shorthand for '
                    'bare functions.')
        return out
    return [Leg(default_label, spec)]


class Group:
    """One mini-P&L inside a ledger: a label, a role, and its legs.

    Parameters
    ----------
    label : str
        The group's name; qualifies its total / result / running-net rows in a
        multi-group ledger.
    role : {'sell', 'buy'}
        Books the group in the holder's ledger: ``sell`` ->
        ``+consideration, -obligation``; ``buy`` -> the contra. A cession is
        declared conceptually (consideration = ceded premium, obligation =
        recovery + commission) and the ``buy`` role books it
        ``-premium / +recovery``.
    consideration, obligation : list of Leg, dict, Leg, or scalar/callable
        The group's cash flows; see :func:`_as_legs` for the accepted
        shorthands. At least one leg between the two.
    """

    def __init__(self, label, role, consideration=None, obligation=None):
        if role not in ('sell', 'buy'):
            raise ValueError(f"role must be 'sell' or 'buy', got {role!r}.")
        if not isinstance(label, str) or not label:
            raise ValueError(
                f'Group label must be a non-empty string, got {label!r}.')
        self.label = label
        self.role = role
        self.consideration = _as_legs(consideration, 'consideration')
        self.obligation = _as_legs(obligation, 'obligation')
        if not self.consideration and not self.obligation:
            raise ValueError(f'Group {label!r} has no legs.')

    def __repr__(self):
        return (f'Group({self.label!r}, {self.role!r}: '
                f'{len(self.consideration)} consideration, '
                f'{len(self.obligation)} obligation)')


# ----------------------------------------------------------------------
# Source atomization: every source reduces to (coords, probs)
# ----------------------------------------------------------------------
def _is_massive(source):
    """A disk-backed :class:`MassiveBivariateDistribution` source (never
    atomized in memory -- evaluated by one pushforward band sweep instead)."""
    return type(source).__name__ == 'MassiveBivariateDistribution'


def _source_atoms(source):
    """Reduce an in-memory P&L ``source`` to ``(coords, probs, shape, bs, is2d)``.

    The opaque source slot is resolved here; leg funcs are then evaluated by
    :func:`_eval_leg` over ``coords`` and the result raveled to align with the
    flat ``probs`` vector.

    Parameters
    ----------
    source : object
        A :class:`~aggregate.bivariate.BivariateDistribution` (axes 0/1 + 2-D
        ``density``); a :class:`~aggregate._grid_distribution.GridDistribution`
        (``x`` / ``p``); an :class:`Aggregate` (read ``density_df``); or a bare
        ``(values, probs)`` pair. A
        :class:`~aggregate.bivariate.MassiveBivariateDistribution` takes the
        one-sweep pushforward route instead (:func:`_is_massive`).

    Returns
    -------
    coords : tuple of ndarray
        The argument arrays a leg func is called with: ``(x,)`` for a 1-D
        source, ``(axis0[:, None], axis1[None, :])`` for a joint.
    probs : ndarray
        Flat (raveled) probability vector, one entry per atom.
    shape : tuple
        The broadcast shape of a leg's values.
    bs : float or None
        The source bucket size if it carries one (used by :meth:`PnL.evaluate`).
    is2d : bool
        Whether the source is a 2-D joint.
    """
    from ._grid_distribution import GridDistribution
    # bivariate: a full 2-D joint over (axis0, axis1)
    if hasattr(source, 'axis0') and hasattr(source, 'axis1') \
            and hasattr(source, 'density'):
        a0 = np.asarray(source.axis0, dtype=float)
        a1 = np.asarray(source.axis1, dtype=float)
        dens = np.asarray(source.density, dtype=float)
        return ((a0[:, None], a1[None, :]), dens.ravel(), dens.shape,
                getattr(source, 'bs', None), True)
    # a GridDistribution slot
    if isinstance(source, GridDistribution):
        x = np.asarray(source.x, dtype=float)
        return (x,), np.asarray(source.p, dtype=float), x.shape, source.bs, False
    # an Aggregate (or anything exposing density_df): the loss grid
    if hasattr(source, 'density_df'):
        dd = source.density_df
        x = (dd['loss'].to_numpy(dtype=float) if 'loss' in dd.columns
             else dd.index.to_numpy(dtype=float))
        p = dd['p_total'].to_numpy(dtype=float)
        return (x,), p, x.shape, getattr(source, 'bs', None), False
    # a bare (values, probs) pair
    vals = np.asarray(source[0], dtype=float)
    p = np.asarray(source[1], dtype=float)
    return (vals,), p, vals.shape, None, False


def _eval_leg(leg, coords, shape, is2d_source):
    """Evaluate one :class:`Leg`'s func over the source atoms -> flat magnitudes.

    A constant broadcasts to every atom. A 1-D func receives axis 0 (the only
    axis of a 1-D source; the first axis of a joint). An ``is2d`` func receives
    both broadcast axes and requires a 2-D source ([One-2D-Source]).
    """
    if leg.is2d and not is2d_source:
        raise ValueError(
            f'leg {leg.label!r} is declared is2d but the source is 1-D; an '
            'is2d leg needs the shared 2-D joint ([One-2D-Source]).')
    if callable(leg.func):
        args = coords if leg.is2d else (coords[0],)
        v = np.asarray(leg.func(*args), dtype=float)
    else:
        v = np.asarray(float(leg.func), dtype=float)
    return np.broadcast_to(v, shape).ravel()


def _signed_sweep_fn(leg, sign):
    """One leg as a signed sweep entry: a constant stays a (signed) constant
    (the sweep short-circuits it); a 1-D func reads axis 0 of the joint; an
    ``is2d`` func reads both broadcast axes."""
    if not callable(leg.func):
        return sign * float(leg.func)
    f = leg.func
    if leg.is2d:
        return lambda x, y, f=f, s=sign: s * f(x, y)
    return lambda x, y, f=f, s=sign: s * np.broadcast_to(
        np.asarray(f(x), dtype=float), np.broadcast_shapes(x.shape, y.shape))


def _sum_sweep_fns(fns):
    """A derived ledger row as its own signed-sum sweep function.

    Never a sum of bucketed legs: the sum is evaluated exactly per cell inside
    the band sweep and bucketed once, so ``mean(row) == sum(leg means)``
    exactly. Constants fold into a scalar shift; an all-constant row returns
    the bare constant (the sweep's constant short-circuit handles it).
    """
    consts = sum(f for f in fns if not callable(f))
    callables = [f for f in fns if callable(f)]
    if not callables:
        return float(consts)
    if len(callables) == 1 and not consts:
        return callables[0]

    def f(x, y):
        acc = None
        for fn in callables:
            v = fn(x, y)
            acc = v if acc is None else acc + v
        return acc + consts if consts else acc

    return f


def _moments_of(values, probs):
    """Exact ``(mean, sd, cv, skew)`` of per-atom ``values`` under ``probs``.

    Taken straight off the atoms (no rebucketing), so these are the
    ground-truth moments the reports add to. Two guards keep degenerate legs
    clean:

    * a **constant** leg (all ``values`` equal -- e.g. a fixed premium) returns
      ``sd = 0`` **exactly** regardless of any probability defect, so a fixed
      consideration reports ``SD = 0`` rather than the spurious
      ``sqrt(c^2 * Sum(p) * (1 - Sum(p)))`` picked up when the source density
      does not sum to exactly 1;
    * a **break-even** mean (``|mean| <= VALIDATION_NOISE``) returns ``cv = nan``
      -- CV is not meaningful when the mean is indistinguishable from zero.

    ``probs`` is assumed normalized (:class:`PnL` normalizes at construction).
    """
    values = np.asarray(values, dtype=float)
    if values.size and float(values.max()) == float(values.min()):
        m = float(values.flat[0])
        cv = 0.0 if abs(m) > VALIDATION_NOISE else float('nan')
        return m, 0.0, cv, float('nan')
    m = float((values * probs).sum())
    var = float((values * values * probs).sum()) - m * m
    var = var if var > 0 else 0.0
    sd = var ** 0.5
    cv = (sd / m) if abs(m) > VALIDATION_NOISE else float('nan')
    if sd > 0:
        skew = float((((values - m) ** 3) * probs).sum()) / sd ** 3
    else:
        skew = float('nan')
    return m, sd, cv, skew


# ----------------------------------------------------------------------
# An evaluated ledger row: signed per-atom values + the shared probs
# ----------------------------------------------------------------------
class _EvaluatedLeg:
    """One ledger row over the source: **signed** values + cached
    :class:`GridDistribution` / exact moments.

    Internal to :class:`PnL`. Declared legs and every derived row (totals,
    group results, running nets, grand rows) all evaluate to this shape, so
    the exhibits iterate one row type. Two backings:

    * **atoms** (in-memory sources): per-atom signed ``values`` + the shared
      ``probs``; ``bs=0`` builds the exact irregular GD (group-by-value,
      sum-prob), ``bs>0`` (declared legs only) rebuckets via the shared
      pushforward machinery. Moments are exact off the atoms.
    * **sweep** (massive sources): a precomputed regular-``bs``
      :class:`GridDistribution` from the one-pass band sweep plus the sweep's
      **exact** streamed mean / sd (the audit's EX basis); the skewness is
      read off the realized grid (the exact third moment is not streamed).
    """

    def __init__(self, label, values=None, probs=None, *, bs=0.0, sign=1.0,
                 gd=None, exact_mean=None, exact_sd=None):
        self.label = label
        self.values = None if values is None \
            else np.asarray(values, dtype=float)
        self.probs = probs
        self.bs = float(bs)
        #: The booking sign the group role applied (+1 / -1); ``values`` are
        #: already signed, ``sign`` recovers the caller's magnitude when needed
        #: (:meth:`PnL.evaluate` prices the obligation magnitude).
        self.sign = float(sign)
        self._gd = gd
        self._moms = None
        if gd is not None:
            m = float(exact_mean)
            sd = float(exact_sd)
            cv = sd / m if abs(m) > VALIDATION_NOISE else float('nan')
            if sd > 0:
                x, p = gd.x, gd.p
                tot = float(p.sum())
                skew = float((((x - m) ** 3) * p).sum()) / (tot * sd ** 3) \
                    if tot > 0 else float('nan')
            else:
                skew = float('nan')
            self._moms = (m, sd, cv, skew)

    @property
    def gd(self):
        """The row's signed :class:`GridDistribution` -- exact irregular when
        ``bs == 0``, the audited regular-``bs`` rebucket otherwise. Signed cash
        to the holder, so payoff orientation (``is_loss_value=False``)."""
        if self._gd is None:
            if self.bs:
                from .bivariate import _finalize_pushforward
                self._gd = _finalize_pushforward(
                    self.values, self.probs, bs=self.bs, log2=None,
                    window=None, scheme='linear', name=self.label,
                    is_loss_value=False, source='pnl-leg')
            else:
                from ._grid_distribution import GridDistribution
                ser = (pd.Series(self.probs, index=self.values)
                       .groupby(level=0).sum().sort_index())
                self._gd = GridDistribution(
                    ser.index.to_numpy(dtype=float),
                    ser.to_numpy(dtype=float),
                    bs=None, name=self.label, is_loss_value=False)
        return self._gd

    @property
    def moments(self):
        """Exact signed ``(mean, sd, cv, skew)`` (atoms or sweep basis)."""
        if self._moms is None:
            self._moms = _moments_of(self.values, self.probs)
        return self._moms

    @property
    def mean(self):
        return self.moments[0]

    def stat_vector(self):
        """``[EX, SD, CV, Skew] + percentile ladder`` -- one exhibit row."""
        m, sd, cv, skew = self.moments
        gd = self.gd
        return [m, sd, cv, skew] + [float(gd.q(q)) for q in PERCENTILE_LADDER]


class _EvaluatedGroup:
    """One evaluated :class:`Group`: its signed leg rows + derived rows."""

    def __init__(self, label, role, cons_rows, obl_rows, probs):
        self.label = label
        self.role = role
        self.cons = cons_rows
        self.obl = obl_rows
        if probs is None:
            # sweep-backed rows carry no atoms; the derived rows come back
            # from the band sweep as their own signed-sum functions.
            self.cons_total_values = self.obl_total_values = None
            self.result_values = None
            return
        n = len(probs)
        cons_vals = (sum(r.values for r in cons_rows) if cons_rows
                     else np.zeros(n))
        obl_vals = (sum(r.values for r in obl_rows) if obl_rows
                    else np.zeros(n))
        self.cons_total_values = cons_vals
        self.obl_total_values = obl_vals
        #: the group's signed net = its step delta in the running net.
        self.result_values = cons_vals + obl_vals


def _ledger_plan(groups, result_name):
    """The ledger row template as ``(label, kind, payload)`` triples.

    The **single source of truth** for the row set both evaluation routes
    materialize (in-memory atoms and the massive one-sweep pushforward), so
    the two ledgers cannot drift. Per group: consideration legs, the group
    total consideration (only if more than one), obligation legs, the group
    total obligation (ditto), the group result (= its step delta); in a
    multi-group ledger a running-net row (``net through <group>``) follows
    each group after the first, the per-group total / result rows are
    qualified by the group label, and the sheet closes with the grand rows:
    ``total consideration`` / ``total obligation`` / the grand result /
    ``total impact`` (grand result vs the first group's result).

    Kinds and payloads: ``'leg'`` ``(gi, side, li)``; ``'group_total'``
    ``(gi, side)``; ``'group_result'`` / ``'running_net'`` ``gi``;
    ``'grand_total'`` ``side``; ``'grand_result'`` / ``'total_impact'``
    ``None``. Duplicate labels raise here, once, for both routes.
    """
    multi = len(groups) > 1
    plan = []
    seen = set()

    def add(label, kind, payload):
        if label in seen:
            raise ValueError(
                f'duplicate ledger row label {label!r}; leg labels must be '
                'unique across the ledger.')
        seen.add(label)
        plan.append((label, kind, payload))

    for gi, g in enumerate(groups):
        qual = f'{g.label} ' if multi else ''
        for li, leg in enumerate(g.consideration):
            add(leg.label, 'leg', (gi, 'cons', li))
        if len(g.consideration) > 1:
            add(f'{qual}total consideration', 'group_total', (gi, 'cons'))
        for li, leg in enumerate(g.obligation):
            add(leg.label, 'leg', (gi, 'obl', li))
        if len(g.obligation) > 1:
            add(f'{qual}total obligation', 'group_total', (gi, 'obl'))
        add(f'{g.label} result' if multi else result_name, 'group_result', gi)
        if multi and gi > 0:
            add(f'net through {g.label}', 'running_net', gi)
    if multi:
        add('total consideration', 'grand_total', 'cons')
        add('total obligation', 'grand_total', 'obl')
        add(result_name, 'grand_result', None)
        add('total impact', 'total_impact', None)
    return plan


def _pct_label(q):
    """Percentile row/column label: ``f'P{q*100:.3g}'`` (``P1`` / ``P50`` / ``P99.5``)."""
    return f'P{q * 100:.3g}'


def _stat_names():
    """Column labels for a ledger row: ``EX / SD / CV / Skew`` + the ladder."""
    return ['EX', 'SD', 'CV', 'Skew'] + [_pct_label(q) for q in PERCENTILE_LADDER]


#: Default ``View`` level names for the :attr:`PnL.stats_df` row MultiIndex,
#: keyed by the internal side codes (``margin`` covers every result-flavored
#: row: group results, running nets, the grand result, total impact).
#: Capitalized, presentation-ready.
_VIEW_DEFAULTS = {'cons': 'Consideration', 'obl': 'Obligation',
                  'margin': 'Margin'}


#: ``stats_df`` / ``scaled_stats_df`` columns that are scale-invariant: a ratio
#: (``CV``) and a shape (``Skew``) pass through ``X / scale`` unchanged; every
#: other column divides by the committed scale.
_SCALE_INVARIANT_STATS = frozenset({'CV', 'Skew'})


# ----------------------------------------------------------------------
# The P&L value object
# ----------------------------------------------------------------------
class PnL(LabeledMixin):
    """A P&L position: a source plus an ordered ledger of signed groups.

    A lightweight value object -- signed per-atom rows + their exact moments +
    the caller's labels, with **no** retained engine dependency (the source is
    kept as an opaque reference only). Build one directly, through
    :meth:`Aggregate.make_pnl` (object sugar), or via ``build('pnl ...')``
    (the DecL builders in :mod:`aggregate._pnl_builders`).

    Two construction forms::

        PnL(name=..., source=..., groups=[Group(...), ...])
        PnL(name=..., role='sell', source=..., consideration=..., obligation=...)

    the second being sugar for exactly one :class:`Group`. Two P&Ls over the
    same source concatenate with ``+`` (their group ledgers chain).

    Parameters
    ----------
    name : str or None
        The position's name (identity handle).
    source : object
        The stochastic generator: a :class:`GridDistribution`, an
        :class:`Aggregate`, a bare ``(values, probs)`` pair, or a
        :class:`~aggregate.bivariate.BivariateDistribution` joint.
    groups : sequence of Group, optional
        The ordered ledger. Mutually exclusive with the single-group form.
    role, consideration, obligation : optional
        The single-group sugar (see :class:`Group`).
    scale : float or str, optional
        The committed unitizer for :attr:`summary_df` / :attr:`scaled_stats_df`.
        ``None`` (default) uses the expected signed grand total consideration;
        a number is an explicit scale; a string names a consideration leg.
    result_name : str, default 'result'
        Label for the grand result row.
    label : str, optional
        Optional human display label (the DecL ``as`` clause); presentation
        only, preferred over ``name`` in repr / titles.

    Notes
    -----
    The ledger rows are **signed cash flows**: the group role books each leg
    (+ / -) so the ``EX`` column adds straight down the sheet to the result
    rows, and each row's distribution sits on the signed values (the adverse
    tail lands where the reader expects for both roles). All derived rows --
    per-group totals, group results (= step deltas), running nets, the grand
    total -- are per-atom partial sums, never arithmetic on marginals, so the
    covariance is carried for free.
    """

    def __init__(self, *, name, source, groups=None, role=None,
                 consideration=None, obligation=None, scale=None,
                 result_name='result', label=None, label_map=None):
        if groups is None:
            if role is None:
                raise ValueError(
                    'PnL needs either groups= (the ledger form) or role= with '
                    'consideration=/obligation= (the single-group form).')
            groups = [Group(name or 'position', role, consideration, obligation)]
        else:
            if role is not None or consideration is not None \
                    or obligation is not None:
                raise ValueError(
                    'pass groups= or the single-group role/consideration/'
                    'obligation form, not both.')
            groups = list(groups)
            for g in groups:
                if not isinstance(g, Group):
                    raise TypeError(
                        f'groups must be Group objects, got {type(g).__name__}.')
            if not groups:
                raise ValueError('PnL needs at least one Group.')
        self.name = name
        # Object-level display label + interior label_map (leg-level labels
        # are the ledger row keys directly). Presentation only; ``name`` stays
        # the identity handle. See dev/done/plan-labels.md.
        self._init_labels(label=label, label_map=label_map)
        self.result_name = result_name
        #: The attached domain analysis object (a
        #: :class:`~aggregate.reinstatement.ReinstatementAnalysis` /
        #: :class:`~aggregate.variable_rating.VariableRatingAnalysis`), for
        #: drill-down only -- no exhibit is forwarded from it. ``None`` on a
        #: plain P&L.
        self.analysis = None
        #: matplotlib figure handle set by :meth:`plot`.
        self.figure = None
        self._group_specs = groups
        self._source = source
        #: the shared row template -- one source of truth for both routes
        self._plan = _ledger_plan(groups, self.result_name)
        if _is_massive(source):
            self._init_massive(source, groups)
        else:
            coords, probs, shape, source_bs, is2d = _source_atoms(source)
            self._coords = coords
            self._shape = shape
            self._source_bs = source_bs
            self._is2d_source = is2d
            # Normalize the shared probability vector so it sums to exactly 1:
            # a source density that clips a little tail mass (Sum(p) = 1 - eps)
            # would otherwise give a *constant* leg a spurious variance. A P&L
            # *is* a probability distribution; the clipped tail is a
            # discretization artifact, so renormalize.
            p = np.asarray(probs, dtype=float)
            total = float(p.sum())
            self._probs = p / total if total > 0 else p
            # evaluate the ledger
            self._egroups = []
            for g in groups:
                sign_cons = 1.0 if g.role == 'sell' else -1.0
                cons_rows = [
                    _EvaluatedLeg(leg.label,
                                  sign_cons * _eval_leg(leg, coords, shape,
                                                        is2d),
                                  self._probs, bs=leg.bs, sign=sign_cons)
                    for leg in g.consideration]
                obl_rows = [
                    _EvaluatedLeg(leg.label,
                                  -sign_cons * _eval_leg(leg, coords, shape,
                                                         is2d),
                                  self._probs, bs=leg.bs, sign=-sign_cons)
                    for leg in g.obligation]
                self._egroups.append(
                    _EvaluatedGroup(g.label, g.role, cons_rows, obl_rows,
                                    self._probs))
            self._assemble_rows()
        self._scale_arg = scale
        self._scale_value, self._scale_label = self._resolve_scale(scale)

    # ------------------------------------------------------------------
    # ledger assembly (in-memory atoms route)
    # ------------------------------------------------------------------
    def _assemble_rows(self):
        """Materialize the :func:`_ledger_plan` template from the evaluated
        groups: every derived row is a per-atom partial sum of the signed
        legs, so the covariance is carried for free."""
        probs = self._probs
        n = len(probs)
        egs = self._egroups
        # prefix running nets: net_after[gi] = sum of group results 0..gi
        net_after = []
        running = np.zeros(n)
        for g in egs:
            running = running + g.result_values
            net_after.append(running)
        rows = OrderedDict()
        self._group_result_rows = []
        grand_total_rows = {}
        for label, kind, payload in self._plan:
            if kind == 'leg':
                gi, side, li = payload
                g = egs[gi]
                row = (g.cons if side == 'cons' else g.obl)[li]
            elif kind == 'group_total':
                gi, side = payload
                vals = (egs[gi].cons_total_values if side == 'cons'
                        else egs[gi].obl_total_values)
                row = _EvaluatedLeg(label, vals, probs)
            elif kind == 'group_result':
                row = _EvaluatedLeg(label, egs[payload].result_values, probs)
                self._group_result_rows.append(row)
            elif kind == 'running_net':
                row = _EvaluatedLeg(label, net_after[payload].copy(), probs)
            elif kind == 'grand_total':
                vals = sum((g.cons_total_values if payload == 'cons'
                            else g.obl_total_values for g in egs),
                           np.zeros(n))
                row = _EvaluatedLeg(label, vals, probs)
                grand_total_rows[payload] = row
            elif kind == 'grand_result':
                row = _EvaluatedLeg(label, net_after[-1], probs)
            else:                                     # 'total_impact'
                row = _EvaluatedLeg(label,
                                    net_after[-1] - egs[0].result_values,
                                    probs)
            rows[label] = row
        self._rows = rows
        # grand references -- always available (the scale and the accessors
        # read them); on a single-group sheet they are not ledger rows.
        self._grand_cons = grand_total_rows.get('cons') or _EvaluatedLeg(
            'total consideration',
            sum((g.cons_total_values for g in egs), np.zeros(n)), probs)
        self._grand_obl = grand_total_rows.get('obl') or _EvaluatedLeg(
            'total obligation',
            sum((g.obl_total_values for g in egs), np.zeros(n)), probs)
        self._grand_result = (rows[self.result_name] if len(egs) > 1
                              else self._group_result_rows[0])

    # ------------------------------------------------------------------
    # ledger assembly (massive one-sweep pushforward route)
    # ------------------------------------------------------------------
    def _init_massive(self, source, groups):
        """Evaluate the whole ledger in **one**
        :meth:`MassiveBivariateDistribution.pushforward` band sweep.

        Every declared leg needs ``bs > 0`` (the sweep scatters onto regular
        output grids -- the caller knows their output scale); each derived row
        (totals, group results, running nets, grand rows) is pushed as its
        **own signed-sum function** with the coarsest constituent ``bs`` --
        never as a sum of bucketed legs -- so ``mean(result) ==
        sum(signed leg means)`` exactly (the a126 dict-pushforward contract).
        Exact means / sds come back from the sweep's streamed audit.
        """
        from collections import OrderedDict as _OD
        self._coords = None
        self._shape = None
        self._source_bs = None
        self._is2d_source = True
        self._probs = None
        # signed per-leg functions (constants stay constants: the sweep
        # short-circuits them before touching the disk)
        leg_entries = {}                       # (gi, side, li) -> (fn, bs)
        group_entries = {}                     # gi -> [fn, ...]
        for gi, g in enumerate(groups):
            sign_cons = 1.0 if g.role == 'sell' else -1.0
            group_entries[gi] = []
            for side, legs, sign in (('cons', g.consideration, sign_cons),
                                     ('obl', g.obligation, -sign_cons)):
                for li, leg in enumerate(legs):
                    if not leg.bs:
                        raise ValueError(
                            f'leg {leg.label!r}: a massive source evaluates '
                            'via one pushforward band sweep, so every leg '
                            'needs an explicit bs > 0 (the output scale).')
                    fn = _signed_sweep_fn(leg, sign)
                    leg_entries[(gi, side, li)] = (fn, leg.bs)
                    group_entries[gi].append((fn, leg.bs))
        # one sweep entry per ledger row, derived rows as signed sums
        funcs = _OD()
        bs_list = []
        signs = {}
        for label, kind, payload in self._plan:
            if kind == 'leg':
                fn, bs = leg_entries[payload]
                gi, side, _li = payload
                signs[label] = (1.0 if side == 'cons' else -1.0) * \
                    (1.0 if groups[gi].role == 'sell' else -1.0)
                entries = [(fn, bs)]
            elif kind == 'group_total':
                gi, side = payload
                g = groups[gi]
                legs = g.consideration if side == 'cons' else g.obligation
                entries = [leg_entries[(gi, side, li)]
                           for li in range(len(legs))]
            elif kind == 'group_result':
                entries = group_entries[payload]
            elif kind == 'running_net':
                entries = [e for gi in range(payload + 1)
                           for e in group_entries[gi]]
            elif kind == 'grand_total':
                entries = [(fn, bs) for (gi, side, li), (fn, bs)
                           in leg_entries.items() if side == payload]
            elif kind == 'grand_result':
                entries = [e for fns in group_entries.values() for e in fns]
            else:                                     # 'total_impact'
                entries = [e for gi in range(1, len(groups))
                           for e in group_entries[gi]]
            funcs[label] = _sum_sweep_fns([fn for fn, _bs in entries])
            bs_list.append(max(bs for _fn, bs in entries))
        # the sweep computes its own total over ALL entries -- meaningless
        # for a ledger (rows overlap), so park it under a reserved key.
        total_key = '__sweep_total__'
        results = source.pushforward(funcs, bs_list, bs_total=max(bs_list),
                                     total_key=total_key,
                                     is_loss_value=False)
        audit = results[next(iter(funcs))].pushforward_audit_df
        rows = OrderedDict()
        self._egroups = []
        self._group_result_rows = []
        grand_total_rows = {}
        group_rows = {gi: {'cons': [], 'obl': []} for gi in range(len(groups))}
        for label, kind, payload in self._plan:
            row = _EvaluatedLeg(
                label, gd=results[label], sign=signs.get(label, 1.0),
                bs=results[label].bs,
                exact_mean=audit.loc[label, 'EX'],
                exact_sd=audit.loc[label, 'SD'])
            rows[label] = row
            if kind == 'leg':
                gi, side, _li = payload
                group_rows[gi][side].append(row)
            elif kind == 'group_result':
                self._group_result_rows.append(row)
            elif kind == 'grand_total':
                grand_total_rows[payload] = row
        for gi, g in enumerate(groups):
            self._egroups.append(_EvaluatedGroup(
                g.label, g.role, group_rows[gi]['cons'],
                group_rows[gi]['obl'], None))
        self._rows = rows
        self._grand_result = (rows[self.result_name] if len(groups) > 1
                              else self._group_result_rows[0])
        # grand totals for the scale / accessors: expectations are linear, so
        # the exact means sum across the constituent legs -- no extra sweep
        # keys needed on a single-group sheet.
        from types import SimpleNamespace
        self._grand_cons = grand_total_rows.get('cons') or SimpleNamespace(
            mean=float(sum(r.mean for g in self._egroups for r in g.cons)))
        self._grand_obl = grand_total_rows.get('obl') or SimpleNamespace(
            mean=float(sum(r.mean for g in self._egroups for r in g.obl)))

    def _resolve_scale(self, scale):
        """Commit the exhibit scale at construction.

        Returns ``(value, label)``. ``scale`` is ``None`` (the default -- the
        expected signed grand total consideration, the natural unitizer), a
        bare number (an explicit scale, e.g. the reinstatement-program
        ``gross - deposit``), or a consideration leg label.
        """
        if scale is None:
            return self._grand_cons.mean, 'consideration'
        if isinstance(scale, str):
            for g in self._egroups:
                for r in g.cons:
                    if r.label == scale:
                        return r.mean, scale
            raise ValueError(
                f'scale {scale!r} names no consideration leg; legs are '
                f'{[r.label for g in self._egroups for r in g.cons]}.')
        return float(scale), 'scale'

    # ------------------------------------------------------------------
    # composition: same source, concatenated ledgers
    # ------------------------------------------------------------------
    def __add__(self, other):
        """Concatenate two group ledgers over the **same** source."""
        if not isinstance(other, PnL):
            return NotImplemented
        same = self._source is other._source
        if not same and self._probs is not None and other._probs is not None:
            same = (self._shape == other._shape
                    and np.array_equal(self._probs, other._probs)
                    and all(np.array_equal(a, b) for a, b
                            in zip(self._coords, other._coords)))
        if not same:
            raise ValueError(
                'pnl_a + pnl_b requires the same source (identical atoms); '
                f'{self.name!r} and {other.name!r} differ.')
        return PnL(name=f'{self.name} + {other.name}', source=self._source,
                   groups=self._group_specs + other._group_specs,
                   scale=self._scale_arg, result_name=self.result_name)

    # ------------------------------------------------------------------
    # structure access
    # ------------------------------------------------------------------
    @property
    def groups(self):
        """The declared :class:`Group` specs, in ledger order."""
        return list(self._group_specs)

    @property
    def role(self):
        """The single group's role (the sugar form); ``None`` on a
        multi-group ledger."""
        return self._egroups[0].role if len(self._egroups) == 1 else None

    @property
    def source(self):
        """The opaque stochastic generator the P&L was built over.

        A P&L *has a* generator (an :class:`Aggregate`, a
        :class:`GridDistribution`, a joint, ...) but never *depends* on it:
        the exhibits are read off the per-atom values captured at
        construction. Retained for reference / drill-down / ``+`` composition.
        """
        return self._source

    def _iter_legs(self):
        """Yield the declared (evaluated) legs across groups, ledger order."""
        for g in self._egroups:
            yield from g.cons
            yield from g.obl

    # ------------------------------------------------------------------
    # distribution accessors -- all delegate to the grand result row
    # ------------------------------------------------------------------
    @property
    def result(self):
        """The grand result as a :class:`GridDistribution`."""
        return self._grand_result.gd

    @property
    def gd(self):
        """Alias for :attr:`result` -- the grand result is the P&L's
        distribution."""
        return self._grand_result.gd

    @property
    def mean(self):
        """``E[result]`` (signed grand net)."""
        return self._grand_result.moments[0]

    @property
    def sd(self):
        """SD of the grand result."""
        return self._grand_result.moments[1]

    @property
    def cv(self):
        """CV of the grand result (meaningless near break-even; reported for
        symmetry)."""
        return self._grand_result.moments[2]

    @property
    def skew(self):
        """Skewness of the grand result."""
        return self._grand_result.moments[3]

    @property
    def prob_loss(self):
        """``P(result < 0)`` -- the probability the position loses money."""
        if self._probs is None:                # sweep-backed (massive source)
            gd = self._grand_result.gd
            return float(gd.p[gd.x < 0].sum())
        v = self._grand_result.values
        return float(self._probs[v < 0].sum())

    def q(self, p, kind='lower'):
        """Quantile (value at risk) of the result. Delegates to the result GD."""
        return self._grand_result.gd.q(p, kind)

    def var(self, p):
        """Value at risk = lower quantile of the result."""
        return self._grand_result.gd.var(p)

    def cdf(self, x):
        """``P(result <= x)``."""
        return self._grand_result.gd.cdf(x)

    def sf(self, x):
        """``P(result > x)``."""
        return self._grand_result.gd.sf(x)

    @property
    def E_consideration(self):
        """``E[total consideration]`` (signed, across groups) -- the default
        scale."""
        return self._grand_cons.mean

    @property
    def scale(self):
        """The committed ``(value, label)`` scale for the scaled exhibits.

        Fixed at construction (default: the expected signed grand total
        consideration). Every scaled cell divides by this one number, so the
        reports never re-derive it per call.
        """
        return self._scale_value, self._scale_label

    # ------------------------------------------------------------------
    # the row ledger, three views
    # ------------------------------------------------------------------
    @property
    def summary_df(self):
        """The headline ledger: one row per ledger line, headline columns.

        Rows = the ledger (legs, totals, results, running nets, grand rows --
        see :meth:`_assemble_rows`); columns ``EX`` / ``Scaled`` (``EX`` over
        the committed :attr:`scale`) / ``SD`` / ``CV`` / ``Skew`` / ``P1`` /
        ``Median`` / ``P99``. Signed cash flows; the ``EX`` column adds down
        the sheet to the result rows.

        Returns
        -------
        pandas.DataFrame
            Indexed by row label (index name ``'P&L'``).
        """
        denom = self._scale_value
        scalable = denom and abs(denom) > VALIDATION_NOISE
        rows = OrderedDict()
        for label, row in self._rows.items():
            m, sd, cv, skew = row.moments
            gd = row.gd
            rows[label] = [
                m, (m / denom if scalable else float('nan')),
                _snap_noise(sd), cv, _snap_noise(skew),
                float(gd.q(0.01)), float(gd.q(0.50)), float(gd.q(0.99))]
        df = pd.DataFrame.from_dict(
            rows, orient='index',
            columns=['EX', 'Scaled', 'SD', 'CV', 'Skew', 'P1', 'Median', 'P99'])
        df.index.name = 'P&L'
        return df

    def _view_index(self):
        """The ``(View, Line)`` row MultiIndex for :attr:`stats_df` /
        :attr:`scaled_stats_df`, aligned with the ledger plan order.

        ``View`` buckets every ledger row: legs and totals under their side
        (:data:`_VIEW_DEFAULTS` -- ``Consideration`` / ``Obligation``), all
        result-flavored rows (group results, running nets, the grand result,
        total impact) under ``Margin``. ``Line`` is the presentation label:
        leg labels as declared; total rows read ``Total`` (qualified as
        ``'<group> total'`` per group on a multi-group sheet); group results
        read the group label; the grand result reads ``Total``. Presentation
        only -- the flat plan labels stay the canonical row keys everywhere
        else (:attr:`summary_df`, :attr:`density_df`, :attr:`validation_df`,
        the sweep result keys).
        """
        groups = self._group_specs
        multi = len(groups) > 1
        v = _VIEW_DEFAULTS
        tuples = []
        for label, kind, payload in self._plan:
            if kind == 'leg':
                _gi, side, _li = payload
                t = (v[side], label)
            elif kind == 'group_total':
                gi, side = payload
                t = (v[side],
                     f'{groups[gi].label} total' if multi else 'Total')
            elif kind == 'group_result':
                t = (v['margin'],
                     groups[payload].label if multi else 'Total')
            elif kind == 'running_net':
                t = (v['margin'], f'Net through {groups[payload].label}')
            elif kind == 'grand_total':
                t = (v[payload], 'Total')
            elif kind == 'grand_result':
                t = (v['margin'], 'Total')
            else:                                     # 'total_impact'
                t = (v['margin'], 'Total impact')
            tuples.append(t)
        return pd.MultiIndex.from_tuples(tuples, names=['View', 'Line'])

    @property
    def stats_df(self):
        """The full ledger x metrics table, in currency units.

        Same rows as :attr:`summary_df` (ledger order preserved), indexed by
        the two-level ``(View, Line)`` MultiIndex of :meth:`_view_index`:
        ``View`` groups the sheet into ``Consideration`` / ``Obligation`` /
        ``Margin``, ``Line`` is the presentation row label (leg labels as
        declared; total rows read ``Total``, so ``'total obligation'``
        displays as ``('Obligation', 'Total')`` and the result as
        ``('Margin', 'Total')``). Columns are ``EX`` / ``SD`` / ``CV`` /
        ``Skew`` and the full :data:`PERCENTILE_LADDER` (``P1`` ... ``P99``).

        Returns
        -------
        pandas.DataFrame
            ``(View, Line)`` MultiIndexed rows in ledger order; columns the
            metric names.
        """
        data = [[_snap_noise(v) for v in row.stat_vector()]
                for row in self._rows.values()]
        return pd.DataFrame(data, index=self._view_index(),
                            columns=_stat_names())

    @property
    def scaled_stats_df(self):
        """The statistics of ``X / scale`` -- :attr:`stats_df` unitized.

        ``EX`` / ``SD`` and every percentile divide by the committed
        :attr:`scale`; ``CV`` and ``Skew`` are scale-invariant and pass
        through unchanged. Every cell defined (a break-even scale yields
        ``nan`` in the divided columns -- there is no unit to measure in).
        Carries the same ``(View, Line)`` row MultiIndex as :attr:`stats_df`.
        """
        df = self.stats_df
        denom = self._scale_value
        scalable = denom and abs(denom) > VALIDATION_NOISE
        for col in df.columns:
            if col in _SCALE_INVARIANT_STATS:
                continue
            df[col] = df[col] / denom if scalable else float('nan')
        return df

    @property
    def density_df(self):
        """An ordered ``{row label: GridDistribution}`` -- the declared legs,
        the group results (multi-group), and the grand result.

        **Not** a single stapled frame: each row keeps its own (irregular,
        exact) grid, so there is no lossy rebucketing onto a shared axis. The
        main customer is :meth:`plot`, which iterates and reads each GD's
        Series view (:meth:`GridDistribution.to_series`).
        """
        out = OrderedDict()
        for r in self._iter_legs():
            out[r.label] = r.gd
        if len(self._egroups) > 1:
            for r in self._group_result_rows:
                out[r.label] = r.gd
        out[self._grand_result.label] = self._grand_result.gd
        return out

    @property
    def validation_df(self):
        """Est-vs-EX audit for every ``bs > 0`` leg.

        ``EX`` is the exact signed mean straight off the atoms; ``Est`` the
        mean of the rebucketed :class:`GridDistribution`. The linear scheme
        matches means, so this reads *very* close -- but it is visible. Legs
        with ``bs = 0`` are exact and absent; no reaching through to the
        source's own validation, which may not exist.

        Returns
        -------
        pandas.DataFrame
            Indexed by leg label; columns ``EX`` / ``Est`` / ``abs_err`` /
            ``rel_err``. Empty (same columns) when every leg is exact.
        """
        recs = []
        for r in self._iter_legs():
            if not r.bs:
                continue
            ex = r.mean
            est = float(r.gd.mean())
            err = abs(ex - est)
            recs.append({'leg': r.label, 'EX': ex, 'Est': est,
                         'abs_err': err,
                         'rel_err': err / abs(ex) if ex else err})
        df = pd.DataFrame(recs, columns=['leg', 'EX', 'Est', 'abs_err',
                                         'rel_err'])
        return df.set_index('leg')

    # ------------------------------------------------------------------
    # Evaluation: the Cherny--Madan breakeven acceptability panel
    # ------------------------------------------------------------------
    def evaluate(self, names=None):
        """Evaluate the position: the Cherny--Madan breakeven acceptability panel.

        A P&L is **evaluated, not priced**: you price the **obligation** and
        ask which distortion drives the risk-adjusted net to zero -- the
        breakeven stress the position survives. That is the
        ``rho_g(obligation) = consideration`` calibration
        :meth:`Distortion.calibrate_set` solves over the full support, with
        the expected total consideration the held magnitude. The breakeven
        ``gini_p`` is the single family-agnostic acceptability index (Cherny &
        Madan).

        Parameters
        ----------
        names : sequence of str, optional
            Distortion families. Defaults to :data:`_EVAL_FAMILIES` (``ph`` /
            ``wang`` / ``dual`` / ``tvar``; ``ccoc`` is excluded).

        Returns
        -------
        pandas.DataFrame
            One row per family: ``param_name`` / ``param`` / ``error`` /
            ``gini_p`` / ``area``.

        Notes
        -----
        Requires a **single** obligation leg over a regular (``bs``-lattice)
        source -- the ordinary insurance case. Unchanged constraint; revisit
        post-beta if needed.
        """
        obl_legs = [r for g in self._egroups for r in g.obl]
        if self._source_bs is None or len(obl_legs) != 1:
            raise NotImplementedError(
                'evaluate requires a single obligation leg over a regular-grid '
                'source (an Aggregate / GridDistribution with bs); not '
                'available for this P&L.')
        from .spectral import Distortion
        from ._grid_distribution import GridDistribution
        from ._pricing import (_canonical_loss_frame, _calibration_survival,
                               _limited_ev)
        if names is None:
            names = _EVAL_FAMILIES
        # price the obligation *magnitude* (the caller's loss orientation):
        # the ledger row is signed, so divide the booking sign back out.
        leg = obl_legs[0]
        mag = leg.values * leg.sign
        ser = (pd.Series(self._probs, index=mag)
               .groupby(level=0).sum().sort_index())
        obl_gd = GridDistribution(ser.index.to_numpy(dtype=float),
                                  ser.to_numpy(dtype=float), bs=None,
                                  name=leg.label, is_loss_value=True)
        dz_index = obl_gd.x
        frame = pd.DataFrame({'loss': dz_index, 'p_total': obl_gd.p},
                             index=pd.Index(dz_index, name='loss'))
        shim = _LossFrameShim(frame, self._source_bs)
        dz, c, _reverse = _canonical_loss_frame(shim)
        bs = self._source_bs
        a_full = float(dz.index[-1])
        S, ess_sup = _calibration_survival(dz, bs, a_full)
        el = _limited_ev(dz, bs, a_full + bs)
        P = self.E_consideration
        target = P + c
        dists = Distortion.calibrate_set(
            S=S, bs=bs, premium_target=target, ess_sup=ess_sup,
            assets=ess_sup or a_full, el=el, names=names)
        rows = []
        for nm in names:
            d = dists[nm]
            param_name = getattr(d, 'param_name', None) or 'param'
            rows.append([param_name, d.shape, d.error, d.gini_p,
                         (d.gini_p + 1) / 2])
        return pd.DataFrame(
            rows, columns=['param_name', 'param', 'error', 'gini_p', 'area'],
            index=pd.Index(list(names), name='distortion'))

    # ------------------------------------------------------------------
    # Plot: the net result density + distribution
    # ------------------------------------------------------------------
    def plot(self, axd=None, **kwargs):
        """Plot the grand result density and distribution (CDF).

        Two panels: the result density (A) and distribution (B), with the
        break-even line at 0 marked.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from .plots import plot_pnl
        return plot_pnl(self, axd=axd, **kwargs)

    # ``label`` comes from ``LabeledMixin`` (the shared label surface);
    # ``name`` stays the identity handle. See dev/done/plan-labels.md.

    def __repr__(self):
        ng = len(self._egroups)
        nl = sum(len(g.cons) + len(g.obl) for g in self._egroups)
        if ng == 1:
            g = self._egroups[0]
            return (f'PnL({self.label!r}: role={g.role}, '
                    f'{len(g.cons)} consideration, {len(g.obl)} obligation '
                    f'-> {self.result_name!r})')
        return (f'PnL({self.label!r}: {ng} groups, {nl} legs '
                f'-> {self.result_name!r})')


class _LossFrameShim:
    """A minimal stand-in carrying ``density_df`` / ``bs`` / ``_is_loss_value``
    for :func:`_canonical_loss_frame` -- so :meth:`PnL.evaluate` reuses the
    pricing helpers without retaining an :class:`Aggregate`."""

    def __init__(self, density_df, bs):
        self.density_df = density_df
        self.bs = bs
        self._is_loss_value = True


# ----------------------------------------------------------------------
# The marginal (no-joint) assembler
# ----------------------------------------------------------------------
def stack_marginal_pnls(perspectives, *, impacts=None, name=None):
    """Stack independent marginal :class:`PnL` perspectives into one frame.

    The **no-joint** case: plain guaranteed-cost perspectives, each a separate
    :class:`PnL` over its **own** marginal -- no shared atoms, so no per-atom
    groups. Each perspective contributes its grand-result statistics as a row;
    ``impacts`` add per-statistic delta rows (``target - base``). Means add
    across the stack; SDs and percentiles are per-row ("means add, SDs
    don't").

    Parameters
    ----------
    perspectives : dict or sequence of (str, PnL)
        Ordered ``{label: PnL}`` (or ``(label, pnl)`` pairs).
    impacts : sequence of (str, str, str), optional
        ``(label, target_label, base_label)`` delta rows, computed
        per-statistic from the perspective rows.
    name : str, optional
        Stashed as ``df.index.name`` prefix context only; the frame itself is
        the deliverable.

    Returns
    -------
    pandas.DataFrame
        Rows = perspectives (+ impact rows); columns ``EX`` / ``SD`` / ``CV``
        / ``Skew`` + the :data:`PERCENTILE_LADDER`.
    """
    items = (list(perspectives.items()) if isinstance(perspectives, dict)
             else list(perspectives))
    rows = OrderedDict()
    for label, pnl in items:
        rows[label] = [_snap_noise(v)
                       for v in pnl._grand_result.stat_vector()]
    for label, tgt, base in (impacts or []):
        rows[label] = [a - b for a, b in zip(rows[tgt], rows[base])]
    df = pd.DataFrame.from_dict(rows, orient='index', columns=_stat_names())
    df.index.name = 'perspective' if name is None else f'{name} perspective'
    return df
