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

Two faces: pnl and xpnl
-----------------------

The DecL builders serve two objects for two questions
(``dev/plan-pnl-consolidated-xpnl-walk.md``):

* **``pnl`` -- "what is my position?"** The consolidated net-in-to-net-out
  view: always one group, always the flat three-row card. Ceded economics
  are netted out and not shown -- a reinsured aggregate's default output is
  its net.
* **``xpnl`` -- "how did I get there?"** The walk: gross -> each cover ->
  Total, a plain multi-group :class:`PnL` (a way of building, not a new
  type) carrying the exploded ``(Step, Side)`` card and
  ``(Step, Side, Label)`` stats sheet.

**Kappa shared-source rule** ([Decision-Kappa-Shared-Source-Rule]): scenario
(``κ``) ladder columns exist exactly when the ledger shares one source --
``pnl`` (single source, trivially) and per-atom towers get them; a
marginal-stitched guaranteed-cost ``xpnl`` (and the massive one-sweep route)
has no joint, so its ladder stays **marginal** under plain ``P`` headers.
The header *is* the flag.

Reading the P&L sheets
----------------------

Two exhibits, deliberately different in kind:

* :attr:`PnL.summary_df` -- the **card**: fixed rows (``Consideration`` /
  ``Obligation`` / ``Margin`` per step) that never vary with the ledger. Its
  percentiles are **marginal** quantiles of each row's own distribution --
  the card answers "how big is each total" (range feel). Marginal quantiles
  never add, so the card's percentile cells do **not** foot down the card;
  that is a property of quantiles, not an error.
* :attr:`PnL.economic_df` -- the **sheet**: every ledger row, and ``κ`` ladder
  columns that are **scenario states** anchored on the grand result
  ([Kappa-Scenario-Percentiles]): column ``κq`` shows every row's
  conditional mean given the result lands at its ``q``-quantile, so each
  column is one internally consistent state and **foots exactly** down the
  sheet. P&Ls are always in payoff sign convention, **left tail bad**
  ([Decision-Kappa-Outcome-Direction]): ``κ01`` is the adverse state, full
  stop -- a loss-sensitive premium correctly shows high in the bad columns.
  The header signals the semantics: a ``κ`` column *means* conditioning
  happened; ladders with no shared source (the massive one-sweep route,
  :func:`stack_marginal_pnls`, the stitched guaranteed-cost tower) stay
  **marginal** and keep plain ``P`` headers
  ([Decision-Kappa-Shared-Source-Rule]).

Where the result is non-monotone in the source (slides, swings, humps -- the
"switcheroo") a scenario cell is the exact mean over the level set of that
result value; well-defined, but read with care. The default ``Side`` labels
(``Consideration`` / ``Obligation`` / ``Margin``) rename at serve time::

    pnl.economic_df.rename({'Obligation': 'Loss & LAE'}, level='Side')
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd

from ._help import HelpMixin
from .constants import INFO_NA, info_row
from .moments import VALIDATION_NOISE, _snap_noise
from ._labeled import LabeledMixin
from ._program import ProgramMixin

__all__ = ['Leg', 'Group', 'PnL', 'stack_marginal_pnls', 'LEG_KINDS']


#: What a declared :class:`Leg` *is*, economically. Accounting metadata the
#: ledger itself never reads: :attr:`PnL.economic_ratios_df` uses it to split a group's
#: obligation into loss and expense (there is no other way to tell ``'Loss'``
#: from ``'LAE'`` but the label text), and :attr:`PnL.legs_df` reports it.
#: ``'premium'`` and ``'commission'`` are consideration-side flows on a ``sell``
#: and ``buy`` group respectively; ``'recovery'`` is what a cession pays back.
LEG_KINDS = ('premium', 'loss', 'expense', 'recovery', 'commission')


#: The detailed percentile ladder used by :attr:`PnL.economic_df` and
#: :func:`stack_marginal_pnls`. On the stats
#: sheets the ladder columns are **scenario states** (conditional means given
#: the grand result lands at its ``q``-quantile -- [Kappa-Scenario-Percentiles])
#: and carry ``κ`` headers; in :func:`stack_marginal_pnls` they stay marginal
#: (independent perspectives share no joint) under plain ``P`` headers. The
#: headline :attr:`PnL.summary_df` card computes its own marginal ``P01`` /
#: ``Median`` / ``P99`` -- deliberately different in kind
#: ([Decision-Card-Percentiles-Stay-Marginal]). The exact ladder is
#: standardized in ``[Reporting-Guidelines]``.
PERCENTILE_LADDER = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


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
    kind : str, optional
        What the flow *is*, one of :data:`LEG_KINDS`. Accounting metadata, not
        behavior: nothing in the ledger reads it, but :attr:`PnL.economic_ratios_df` needs
        it to separate loss from expense inside a group's obligation, and
        :attr:`PnL.legs_df` reports it. ``None`` (the default) leaves the leg
        unclassified, and ``economic_ratios_df`` folds an unclassified obligation into
        ``L``, so a ledger that declares no expense legs correctly reports
        ``E = 0``. Distinct from the ledger **row** kinds of
        :func:`_ledger_plan`, which describe a row's role in the sheet rather
        than a declared flow's economic nature.
    """

    def __init__(self, label, func, bs=0, is2d=False, kind=None):
        if not isinstance(label, str) or not label:
            raise ValueError(
                f'Leg label must be a non-empty string, got {label!r}.')
        if bs < 0:
            raise ValueError(f'Leg bs must be >= 0, got {bs!r} ({label!r}).')
        if kind is not None and kind not in LEG_KINDS:
            raise ValueError(
                f'Leg kind must be one of {", ".join(LEG_KINDS)}, not '
                f'{kind!r} ({label!r}).')
        self.label = label
        self.func = func
        self.bs = float(bs)
        self.is2d = bool(is2d)
        self.kind = kind

    def __repr__(self):
        extra = (f', bs={self.bs:g}' if self.bs else '') + \
            (', is2d=True' if self.is2d else '') + \
            (f', kind={self.kind!r}' if self.kind else '')
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
    margin_label : str, optional
        What this group's **own result** row is called under ``Margin`` on
        :attr:`PnL.economic_df`. Only the direct (``sell``) block of a ledger that
        buys something reads it, and only then does it appear
        ([First-Step-Label]): the subject business names its own margin, the way
        it already names its loss leg. ``None`` falls back to ``'Gross'``
        there, and everywhere else the label is structural (``'Total'``).
    """

    def __init__(self, label, role, consideration=None, obligation=None,
                 margin_label=None):
        if role not in ('sell', 'buy'):
            raise ValueError(f"role must be 'sell' or 'buy', got {role!r}.")
        if not isinstance(label, str) or not label:
            raise ValueError(
                f'Group label must be a non-empty string, got {label!r}.')
        self.label = label
        self.role = role
        self.margin_label = margin_label
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
        """``[EX, SD, CV, Skew] + marginal percentile ladder`` -- one row.

        The ladder here is **marginal** (each ``Pq`` is this row's own
        quantile). Serves :func:`stack_marginal_pnls` (independent
        perspectives share no joint, so there is nothing to condition on) and
        the massive one-sweep :attr:`PnL.economic_df` (conditioning needs a
        second sweep -- [Massive-Kappa-Second-Sweep] in ``dev/TODO.md``); the
        in-memory :attr:`PnL.economic_df` ladder is the conditional
        [Kappa-Scenario-Percentiles] pass instead.
        """
        m, sd, cv, skew = self.moments
        gd = self.gd
        return [m, sd, cv, skew] + [float(gd.q(q)) for q in PERCENTILE_LADDER]


class _DeltaGD:
    """Quantile shim for a :class:`_DeltaRow`: ``q(p)`` is the **difference of
    the two rows' quantiles**, not the quantile of the difference (which would
    need a joint the stitched route does not have)."""

    __slots__ = ('_t', '_b')

    def __init__(self, target_gd, base_gd):
        self._t = target_gd
        self._b = base_gd

    def q(self, p, kind='lower'):
        return float(self._t.q(p, kind)) - float(self._b.q(p, kind))


class _DeltaRow:
    """A per-statistic delta ledger row: ``target - base``, cell by cell.

    Serves the rows of a **stitched** ledger that no engine marginal backs
    (the ``total impact`` of a marginal-stitched tower: grand result and first
    step ride *different* marginals, and their difference has no distribution
    without a joint). Semantics follow the historical
    :func:`stack_marginal_pnls` impact rows: every statistic is the plain
    difference of the two rows' statistics -- the mean is exact by linearity;
    SD / CV / Skew / percentile cells are **deltas of row statistics**, not
    statistics of the delta.
    """

    def __init__(self, label, target, base):
        self.label = label
        self._t = target
        self._b = base
        self.bs = 0.0
        self.sign = 1.0
        self.values = None
        self.probs = None

    @property
    def gd(self):
        return _DeltaGD(self._t.gd, self._b.gd)

    @property
    def moments(self):
        return tuple(a - b for a, b in zip(self._t.moments, self._b.moments))

    @property
    def mean(self):
        return self.moments[0]

    def stat_vector(self):
        return [a - b for a, b in
                zip(self._t.stat_vector(), self._b.stat_vector())]


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


def _ledger_plan(groups, result_name, tier_spans=()):
    """The ledger row template as ``(label, kind, payload)`` triples.

    The **single source of truth** for the row set every evaluation route
    materializes (in-memory atoms, the massive one-sweep pushforward, and the
    gd-backed stitch), so the ledgers cannot drift. Per group: consideration
    legs, the group total consideration (only if more than one), obligation
    legs, the group total obligation (ditto), the group result (= its step
    delta); in a multi-group ledger a running-net row (``net through
    <group>``) follows each group after the first, the per-group total /
    result rows are qualified by the group label, and the sheet closes with
    the grand rows: ``total consideration`` / ``total obligation`` / the
    grand result / ``total impact`` (grand result vs the first group's
    result).

    Kinds and payloads: ``'leg'`` ``(gi, side, li)``; ``'group_total'``
    ``(gi, side)``; ``'group_result'`` / ``'running_net'`` ``gi``;
    ``'tier_total'`` ``(lo, hi, side)``; ``'tier_result'`` ``(lo, hi)``;
    ``'grand_total'`` ``side``; ``'grand_result'`` / ``'total_impact'``
    ``None``. Duplicate labels raise here, once, for every route. (A forced
    single-group tower -- the one-step walk -- keeps this single-group
    template: the tower shape is presentation only, no grand rows.)

    Parameters
    ----------
    groups : list of Group
        The declared groups, in ledger order.
    result_name : str
        Name of the single-group result row (the grand result on a tower).
    tier_spans : tuple, optional
        ``(label, lo, hi)`` triples naming contiguous **spans** of groups that
        earn their own subtotal block, emitted after group ``hi - 1`` with
        ``hi`` exclusive ([Tier-Subtotal-Rows]). A layer-peeled walk passes one
        span per reinsurance tier, so a peeled tower still shows the whole
        occurrence and whole aggregate program. A span of fewer than two groups
        emits nothing, mirroring the group totals: with one group the group's
        own rows already *are* the subtotal, which is why the plain tier walk
        is untouched.

    Notes
    -----
    Payloads must be hashable: they are half of the ``(kind, payload)`` key
    into ``PnL._by_kind``, hence tuples for the span kinds.
    """
    multi = len(groups) > 1
    plan = []
    seen = set()
    # spans that earn a block, keyed by the group they follow
    spans_after = {}
    for label, lo, hi in tier_spans:
        if hi - lo >= 2:
            spans_after.setdefault(hi - 1, []).append((label, lo, hi))

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
        for label, lo, hi in spans_after.get(gi, ()):
            add(f'{label} total consideration', 'tier_total', (lo, hi, 'cons'))
            add(f'{label} total obligation', 'tier_total', (lo, hi, 'obl'))
            add(f'{label} result', 'tier_result', (lo, hi))
    if multi:
        add('total consideration', 'grand_total', 'cons')
        add('total obligation', 'grand_total', 'obl')
        add(result_name, 'grand_result', None)
        add('total impact', 'total_impact', None)
    return plan


def _pct_label(q):
    """Marginal percentile column label, zero-padded: ``P01`` / ``P50`` /
    ``P99`` (``.3g`` fallback for fractional points: ``P99.5``).

    The header signals the semantics ([Decision-Ladder-Column-Names]): a plain
    ``P`` column is a **marginal** quantile / statistic of that row; a ``κ``
    column (:func:`_kappa_label`) means conditioning happened.
    """
    v = q * 100
    return f'P{v:02.0f}' if float(v).is_integer() else f'P{v:.3g}'


def _kappa_label(q):
    """Scenario (kappa) column label: ``κ01`` / ``κ50`` / ``κ99`` (``.3g``
    fallback for fractional points).

    The ``κ`` marks the [Kappa-Scenario-Percentiles] semantics on the sheet
    itself: the column is the conditional mean of each row given the grand
    result lands at its ``q``-quantile -- the library's kappa function
    ``E[X_i | X = x]`` applied to the ledger. P&Ls are always in payoff sign
    convention, **left tail bad**: ``κ01`` is the adverse state, ``κ99`` the
    favorable one ([Decision-Kappa-Outcome-Direction]).
    """
    v = q * 100
    return f'κ{v:02.0f}' if float(v).is_integer() else f'κ{v:.3g}'


def _stat_names(scenario=False):
    """Column labels for a ledger row: ``EX / SD / CV / Skew`` + the ladder.

    ``scenario=True`` gives the ladder ``κ`` headers (the conditional
    [Kappa-Scenario-Percentiles] columns of the in-memory
    :attr:`PnL.economic_df`); the default plain ``P`` headers mark a **marginal**
    ladder (the massive one-sweep route, :func:`stack_marginal_pnls`, the
    stitched guaranteed-cost ``xpnl`` tower). Positional access or
    ``df.filter(like=...)`` avoids typing the ``κ`` glyph.
    """
    lab = _kappa_label if scenario else _pct_label
    return ['EX', 'SD', 'CV', 'Skew'] + [lab(q) for q in PERCENTILE_LADDER]


#: Fixed column set of the :attr:`PnL.summary_df` card (headline moments +
#: the three marginal range percentiles; ``P01`` zero-pads to pair with
#: ``P99`` -- [Decision-Ladder-Column-Names]).
_CARD_COLS = ['EX', 'SD', 'CV', 'Skew', 'P01', 'Median', 'P99']


#: The four signed amounts :attr:`PnL.economic_ratios_df` accumulates per block, and which
#: :data:`LEG_KINDS` feeds each. An unclassified leg falls back on its side:
#: consideration to ``P``, obligation to ``L`` (the residual), so a ledger that
#: declares no expense legs reports ``E = 0`` rather than guessing.
_RATIO_AMOUNTS = ('P', 'L', 'E', 'C')
_RATIO_BUCKET = {'premium': 'P', 'loss': 'L', 'recovery': 'L',
                 'expense': 'E', 'commission': 'C'}

#: Fixed column order of :attr:`PnL.economic_ratios_df`: the amounts, the block's signed
#: result, the three ratios of means, their three mean-of-ratio twins, then the
#: two shares of the gross block. ``L``, ``M``, ``P`` and ``LR`` keep the
#: :data:`aggregate.pentagon.PENTAGON_STATS` spelling so a P&L ratio frame
#: concatenates and diffs against a pricing frame; ``E`` / ``C`` / ``ER`` / ``CR``
#: extend it, and ``Q`` / ``a`` / ``PQ`` / ``ROE`` have no meaning on a ledger.
_RATIO_COLS = ('P', 'L', 'E', 'C', 'M', 'LR', 'ER', 'CR',
               'E_LR', 'E_ER', 'E_CR', 'P_share', 'M_share')


#: Default ``Side`` level names for the :attr:`PnL.economic_df` row MultiIndex,
#: keyed by the internal side codes (``margin`` covers every result-flavored
#: row: group results, running nets, the grand result, total impact).
#: Capitalized, presentation-ready.
_SIDE_DEFAULTS = {'cons': 'Consideration', 'obl': 'Obligation',
                  'margin': 'Margin'}



# ----------------------------------------------------------------------
# The P&L value object
# ----------------------------------------------------------------------
class PnL(HelpMixin, LabeledMixin, ProgramMixin):
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
    result_name : str, default 'result'
        Label for the grand result row -- the flat ledger key only
        (:attr:`density_df`, the sweep result keys). The card and stats
        sheets display ``Margin`` regardless, over ``'Net'`` on a ledger that
        buys something and ``'Total'`` on one that does not.
    tier_spans : tuple, optional
        ``(label, lo, hi)`` triples naming contiguous spans of groups that earn
        their own subtotal block, ``hi`` exclusive ([Tier-Subtotal-Rows]). See
        :func:`_ledger_plan`; a span of fewer than two groups emits nothing.
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
                 consideration=None, obligation=None,
                 result_name='result', tier_spans=(), label=None,
                 label_map=None, stitched_rows=None, force_tower=False):
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
        # DecL trailer metadata. A PnL is assembled by the builders rather than
        # splatted from a spec, so these start empty and :meth:`_adopt_engine`
        # fills them from the statement's recipe. See dev/plan-meta-data.md.
        self.note = ''
        self.tags = ()
        self.hints = ''
        self.doc = ''
        self.result_name = result_name
        #: The wrapped stochastic engine (the DecL ``pnl`` / ``xpnl`` inner
        #: :class:`Aggregate` / :class:`Portfolio`), kept for drill-down --
        #: the ``source`` is the *simplest sufficient object* (a
        #: GridDistribution / bivariate joint), this is the full engine
        #: behind it ([Engine-Reference-On-PnL],
        #: ``dev/done/plan-pnl-faces-punchlist.md``). ``None``
        #: on a hand-built kernel P&L.
        self.engine = None
        # Backing store for the ``program`` property: ``build`` stamps the
        # declaration here; it otherwise falls back to the engine's.
        self._program = ''
        #: The resolved cession economics dict (``pc_occ`` / ``pc_agg`` /
        #: ``c_occ`` / ``c_agg`` / ``gross`` / ``ceded``) -- the observable of
        #: the DecL ``deposit`` / ``rol`` / ``rate`` / ``cede`` resolution,
        #: set by the reinsurance builders. ``None`` on a plain P&L.
        self.economics = None
        # Construction narratives ([Construction-Introspection]): recorded by
        # the DecL builders at construction (they alone know the why); the
        # resolved properties fall back to a generic structural narrative so
        # a hand-built kernel P&L is never without one.
        self._construction_description = None
        self._construction_explanation = None
        #: matplotlib figure handle set by :meth:`plot`.
        self.figure = None
        self._group_specs = groups
        self._source = source
        #: stitched ledgers carry gd-backed rows with no shared atoms
        #: (no per-atom values, no ``+`` composition) -- see :meth:`_init_stitched`.
        self._stitched = stitched_rows is not None
        #: tower presentation: multi-group, or a forced single-group walk
        #: ([Decision-XPnL-Plain-Is-One-Step-Walk] -- ``force_tower=True``
        #: presents the one-group ledger as its single (Step, Side) block /
        #: (Step, Side, Label) sheet; no grand rows, no impact).
        self._tower = len(groups) > 1 or bool(force_tower)
        #: contiguous group spans carrying their own subtotal block
        #: ([Tier-Subtotal-Rows]); ``{(lo, hi): label}`` for the presentation
        #: lookup, spans of fewer than two groups dropped as the plan drops them.
        self._tier_spans = tuple(tier_spans)
        self._span_labels = {(lo, hi): lbl for lbl, lo, hi in self._tier_spans
                             if hi - lo >= 2}
        #: the shared row template -- one source of truth for all routes
        self._plan = _ledger_plan(groups, self.result_name, self._tier_spans)
        if stitched_rows is not None:
            self._init_stitched(groups, stitched_rows)
        elif _is_massive(source):
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
        #: ``(kind, payload) -> row`` -- structural access for the fixed card
        #: (:attr:`summary_df` reads rows by ledger role, not label).
        self._by_kind = {}
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
            elif kind == 'tier_total':
                # a span subtotal is the grand total restricted to egs[lo:hi]
                lo, hi, side = payload
                vals = sum((g.cons_total_values if side == 'cons'
                            else g.obl_total_values for g in egs[lo:hi]),
                           np.zeros(n))
                row = _EvaluatedLeg(label, vals, probs)
            elif kind == 'tier_result':
                lo, hi = payload
                vals = sum((g.result_values for g in egs[lo:hi]), np.zeros(n))
                row = _EvaluatedLeg(label, vals, probs)
            elif kind == 'grand_total':
                vals = sum((g.cons_total_values if payload == 'cons'
                            else g.obl_total_values for g in egs),
                           np.zeros(n))
                row = _EvaluatedLeg(label, vals, probs)
                grand_total_rows[payload] = row
            elif kind == 'grand_result':
                row = _EvaluatedLeg(label, net_after[-1], probs)
            elif kind == 'total_impact':
                row = _EvaluatedLeg(label,
                                    net_after[-1] - egs[0].result_values,
                                    probs)
            else:
                raise ValueError(
                    f'unknown ledger row kind {kind!r} for row {label!r}; '
                    '_assemble_rows must handle every kind _ledger_plan emits.')
            rows[label] = row
            self._by_kind[(kind, payload)] = row
        self._rows = rows
        # grand references -- always available (the scale and the accessors
        # read them); on a single-group sheet they are not ledger rows.
        self._grand_cons = grand_total_rows.get('cons') or _EvaluatedLeg(
            'total consideration',
            sum((g.cons_total_values for g in egs), np.zeros(n)), probs)
        self._grand_obl = grand_total_rows.get('obl') or _EvaluatedLeg(
            'total obligation',
            sum((g.obl_total_values for g in egs), np.zeros(n)), probs)
        self._grand_result = (rows[self.result_name] if self._tower
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
            elif kind == 'tier_total':
                lo, hi, side = payload
                entries = [(fn, bs) for (gi, sd, li), (fn, bs)
                           in leg_entries.items()
                           if sd == side and lo <= gi < hi]
            elif kind == 'tier_result':
                lo, hi = payload
                entries = [e for gi in range(lo, hi)
                           for e in group_entries[gi]]
            elif kind == 'grand_total':
                entries = [(fn, bs) for (gi, side, li), (fn, bs)
                           in leg_entries.items() if side == payload]
            elif kind == 'grand_result':
                entries = [e for fns in group_entries.values() for e in fns]
            elif kind == 'total_impact':
                entries = [e for gi in range(1, len(groups))
                           for e in group_entries[gi]]
            else:
                raise ValueError(
                    f'unknown ledger row kind {kind!r} for row {label!r}; '
                    '_init_massive must handle every kind _ledger_plan emits.')
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
        self._by_kind = {}
        grand_total_rows = {}
        group_rows = {gi: {'cons': [], 'obl': []} for gi in range(len(groups))}
        for label, kind, payload in self._plan:
            row = _EvaluatedLeg(
                label, gd=results[label], sign=signs.get(label, 1.0),
                bs=results[label].bs,
                exact_mean=audit.loc[label, 'EX'],
                exact_sd=audit.loc[label, 'SD'])
            rows[label] = row
            self._by_kind[(kind, payload)] = row
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
        self._grand_result = (rows[self.result_name] if self._tower
                              else self._group_result_rows[0])
        # grand totals for the scale / accessors / the summary card. On a
        # multi-group sheet these are the grand_total sweep rows; single-group
        # they coincide with the group's own side total, which is always
        # reachable without extra sweep keys (>1 legs -> the group_total row;
        # exactly 1 -> the leg row *is* the total; 0 -> a constant zero).
        self._grand_cons = grand_total_rows.get('cons') \
            or self._card_side_row(0, 'cons')
        self._grand_obl = grand_total_rows.get('obl') \
            or self._card_side_row(0, 'obl')

    # ------------------------------------------------------------------
    # ledger assembly (stitched gd-backed route -- internal)
    # ------------------------------------------------------------------
    def _init_stitched(self, groups, entries):
        """Materialize the ledger from **caller-supplied gd-backed rows**
        ([GC-Tower-Marginal-Stitch]; internal -- not a public construction
        surface).

        The marginal-stitched route: every plan row arrives as its own exact
        :class:`GridDistribution` + exact mean / sd (the same shape the
        massive one-sweep route returns), with **no shared atoms** -- the
        builder reads each row off an engine marginal (an affine transform of
        ``reins_density_df``), never off cross-row sums, which do not exist
        without a joint. ``entries`` maps every :func:`_ledger_plan` row label
        to either

        * ``(gd, exact_mean, exact_sd)`` -- a gd-backed row, or
        * ``('delta', target_label, base_label)`` -- a :class:`_DeltaRow`
          (per-statistic difference; targets must precede in plan order).

        Consequences of having no atoms: the stats ladder is **marginal**
        (plain ``P`` headers -- the on-sheet signature of
        [Decision-Kappa-Shared-Source-Rule]) and ``+`` composition is
        unavailable. :meth:`evaluate` works (a187): it reads each row's own
        marginal, and the one row that has no law, the ``_DeltaRow`` impact, is
        not an evaluated position anyway (a190).
        """
        self._coords = None
        self._shape = None
        self._source_bs = None
        self._is2d_source = False
        self._probs = None
        rows = OrderedDict()
        self._egroups = []
        self._group_result_rows = []
        self._by_kind = {}
        grand_total_rows = {}
        group_rows = {gi: {'cons': [], 'obl': []} for gi in range(len(groups))}
        for label, kind, payload in self._plan:
            try:
                e = entries[label]
            except KeyError:                       # pragma: no cover
                raise ValueError(
                    f'stitched construction: no entry supplied for ledger row '
                    f'{label!r}; the builder must supply every plan row.')
            if e[0] == 'delta':
                row = _DeltaRow(label, rows[e[1]], rows[e[2]])
            else:
                gd, m, sd = e
                row = _EvaluatedLeg(label, gd=gd, exact_mean=m, exact_sd=sd)
            rows[label] = row
            self._by_kind[(kind, payload)] = row
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
        self._grand_result = (rows[self.result_name] if self._tower
                              else self._group_result_rows[0])
        self._grand_cons = grand_total_rows.get('cons') \
            or self._card_side_row(0, 'cons')
        self._grand_obl = grand_total_rows.get('obl') \
            or self._card_side_row(0, 'obl')


    # ------------------------------------------------------------------
    # composition: same source, concatenated ledgers
    # ------------------------------------------------------------------
    def __add__(self, other):
        """Concatenate two group ledgers over the **same** source."""
        if not isinstance(other, PnL):
            return NotImplemented
        if self._stitched or other._stitched:
            raise ValueError(
                'a stitched P&L carries gd-backed rows with no shared atoms, '
                'so its ledger cannot concatenate with another; rebuild from '
                'the engine instead.')
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
        # ``other``'s groups shift right by our count, so its spans shift too
        offset = len(self._group_specs)
        spans = self._tier_spans + tuple(
            (lbl, lo + offset, hi + offset) for lbl, lo, hi in other._tier_spans)
        return PnL(name=f'{self.name} + {other.name}', source=self._source,
                   groups=self._group_specs + other._group_specs,
                   result_name=self.result_name, tier_spans=spans)

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
        """The grand result as a :class:`GridDistribution`.

        Notes
        -----
        The one public name for the P&L's distribution. A ``PnL.gd`` alias
        existed until ``a171``, when it was retired under the one-name-per-concept
        rule: ``gd`` stays the *internal* vocabulary on ledger rows (a leg's or a
        group's ``.gd``), and a first-class object's own distribution reads as its
        ``result``.
        """
        return self._grand_result.gd

    # The four moments carry the ``est_`` prefix, not ``actual_``: a P&L is
    # evaluated per-atom over the *source's realised grid*, so every moment
    # inherits that grid's discretization. (Within the ledger those per-atom
    # values are exact -- that is the ``EX`` basis ``validation_df`` audits a
    # rebucketed ``bs > 0`` leg against -- but relative to the analytic
    # ``actual_*`` moments of the generating Aggregate they are estimates.)
    @property
    def est_m(self):
        """``E[result]`` (signed grand net), off the realised grid."""
        return self._grand_result.moments[0]

    @property
    def est_sd(self):
        """SD of the grand result."""
        return self._grand_result.moments[1]

    @property
    def est_cv(self):
        """CV of the grand result (meaningless near break-even; reported for
        symmetry)."""
        return self._grand_result.moments[2]

    @property
    def est_skew(self):
        """Skewness of the grand result."""
        return self._grand_result.moments[3]

    @property
    def prob_eq_0(self):
        """``P(result == 0)`` -- the probability the position exactly breaks even.

        The sign-neutral break-even probability, shared with
        :attr:`~aggregate.distributions.Aggregate.prob_eq_0` and
        :attr:`~aggregate.portfolio.Portfolio.prob_eq_0`. On a loss object it
        reads as "no loss"; on a payoff / P&L object as "exactly break even".

        Notes
        -----
        Replaces the old ``prob_loss`` (``P(result < 0)``), which had no
        meaning on a loss-valued object -- the probability of *a loss* and the
        probability of *losing money* are opposite tails of the same number.
        Mass is summed over the atoms at exactly zero, so a continuous P&L
        straddling zero reports ``0.0``; the atom is genuine (and often large)
        for discrete and reinsurance-net books.
        """
        if self._probs is None:                # sweep-backed (massive source)
            gd = self._grand_result.gd
            return float(gd.p[gd.x == 0].sum())
        v = self._grand_result.values
        return float(self._probs[v == 0].sum())

    def q(self, p, kind='lower'):
        """Quantile (value at risk) of the result. Delegates to the result GD."""
        return self._grand_result.gd.q(p, kind)

    def tvar(self, p):
        """Tail value at risk of the result. Delegates to the result GD.

        Notes
        -----
        The P&L is a **payoff**, so the interesting tail is the *low* one: the
        result GD carries the orientation (:attr:`GridDistribution.is_loss_value`
        is ``False`` here) and ``tvar`` reads off the correct side, exactly as
        ``q`` does. Added in 1.0.0a150 -- ``q`` / ``var`` / ``cdf`` / ``sf``
        delegated from the start but ``tvar`` did not, leaving the library's
        flagship risk measure unavailable on a P&L.
        """
        return self._grand_result.gd.tvar(p)

    def cdf(self, x):
        """``P(result <= x)``."""
        return self._grand_result.gd.cdf(x)

    def sf(self, x):
        """``P(result > x)``."""
        return self._grand_result.gd.sf(x)

    # ------------------------------------------------------------------
    # the row ledger, three views
    # ------------------------------------------------------------------
    def _zero_row(self):
        """A constant-zero ledger row -- the side total of a group with no
        legs on that side (e.g. a pure hedge declares no consideration)."""
        if self._probs is not None:
            return _EvaluatedLeg('zero', np.zeros(len(self._probs)),
                                 self._probs)
        from ._grid_distribution import GridDistribution
        return _EvaluatedLeg(
            'zero', gd=GridDistribution(np.array([0.0]), np.array([1.0]),
                                        bs=None, name='zero',
                                        is_loss_value=False),
            exact_mean=0.0, exact_sd=0.0)

    def _card_side_row(self, gi, side):
        """Group ``gi``'s total row for one side (``'cons'`` / ``'obl'``).

        Always resolvable from the ledger as evaluated -- no extra
        computation (and, on the massive route, no extra sweep keys): with
        more than one leg the ``group_total`` row exists; with exactly one
        the leg row *is* the side total; with none the total is a constant
        zero.
        """
        row = self._by_kind.get(('group_total', (gi, side)))
        if row is not None:
            return row
        g = self._egroups[gi]
        side_rows = g.cons if side == 'cons' else g.obl
        return side_rows[0] if side_rows else self._zero_row()

    def _card_stat_row(self, row):
        """One :attr:`summary_df` card row: ``EX / SD / CV / Skew`` +
        **marginal** ``P1 / Median / P99`` of the row's own distribution."""
        m, sd, cv, skew = row.moments
        gd = row.gd
        return [m, _snap_noise(sd), cv, _snap_noise(skew),
                float(gd.q(0.01)), float(gd.q(0.50)), float(gd.q(0.99))]

    @property
    def summary_df(self):
        """The headline card: fixed rows that never vary with the ledger.

        Single-group: a flat three-row card ``Consideration`` /
        ``Obligation`` / ``Margin`` read from the grand references -- the
        exact structural mirror of the flat :attr:`Aggregate.summary_df`.
        Multi-group (a tower): one ``(Step, Side)`` block per step plus a
        closing ``'All'`` block (the a140 author rename -- too many things
        were already called Total), mirroring the per-unit blocks of
        :attr:`Portfolio.summary_df`::

            ('Gross',     'Consideration' | 'Obligation' | 'Margin')
            ('ceded occ', 'Consideration' | 'Obligation' | 'Margin' | 'Net')
            ('All',       'Consideration' | 'Obligation' | 'Margin' | 'Impact')

        Per step, ``Margin`` is the group result (the step delta) and ``Net``
        the running net through the step (omitted on the first step, where
        net = margin); the ``'All'`` block closes with ``Impact`` = grand
        result minus the first step's result. A forced single-group tower
        (the one-step walk) is just its one block -- no ``'All'`` block, no
        impact. Rows scale with steps, never with legs -- the fixed-shape
        contract. The card always displays ``Margin`` regardless of
        :attr:`result_name` (which names the flat ledger key only).

        A layer-peeled walk adds a three-row block per reinsurance tier that
        spans two or more steps, after the last step it spans
        ([Tier-Subtotal-Rows])::

            ('All occurrence', 'Consideration' | 'Obligation' | 'Margin')

        so a peeled tower still shows the whole occurrence and whole aggregate
        program. There is no ``Net`` row on a tier block: the running net
        through the tier is already the last layer's ``Net``.

        Columns ``EX`` / ``SD`` / ``CV`` / ``Skew`` / ``P01`` / ``Median`` /
        ``P99``. The percentiles are **marginal quantiles of each card row's
        own distribution** -- the card answers "how big is each total" (range),
        so its percentile cells do **not** add down the card (marginal
        quantiles never add). The footing sheet is :attr:`economic_df`, whose
        scenario columns condition on the grand result and foot exactly. The
        card is currency only: ratios live in :attr:`economic_ratios_df`, their own
        table, per the reporting rule that a column carries one unit.

        Returns
        -------
        pandas.DataFrame
            Flat index named ``'Side'`` (single group) or a ``(Step, Side)``
            MultiIndex (tower); columns as above. The card's ``Side`` level
            merges the two :attr:`economic_df` levels: ``Net`` and ``Impact`` are
            ``Label`` values there, but on a fixed-shape card they are rows of
            their own, so both frames name the level the same way rather than
            inventing a second word for it.

        See Also
        --------
        economic_df : the full ledger sheet (every leg, footing columns).
        """
        if not self._tower:
            rows = OrderedDict((
                ('Consideration', self._grand_cons),
                ('Obligation', self._grand_obl),
                ('Margin', self._grand_result)))
            df = pd.DataFrame.from_dict(
                {k: self._card_stat_row(r) for k, r in rows.items()},
                orient='index', columns=_CARD_COLS)
            df.index.name = 'Side'
            return df
        recs = []
        index = []
        for gi, g in enumerate(self._group_specs):
            index.append((g.label, 'Consideration'))
            recs.append(self._card_stat_row(self._card_side_row(gi, 'cons')))
            index.append((g.label, 'Obligation'))
            recs.append(self._card_stat_row(self._card_side_row(gi, 'obl')))
            index.append((g.label, 'Margin'))
            recs.append(self._card_stat_row(
                self._by_kind[('group_result', gi)]))
            if gi > 0:
                index.append((g.label, 'Net'))
                recs.append(self._card_stat_row(
                    self._by_kind[('running_net', gi)]))
            # a tier's own block, after the last group it spans
            # ([Tier-Subtotal-Rows]); the grand block below is the template
            for (lo, hi), lbl in self._span_labels.items():
                if hi - 1 != gi:
                    continue
                for view, key in (
                        ('Consideration', ('tier_total', (lo, hi, 'cons'))),
                        ('Obligation', ('tier_total', (lo, hi, 'obl'))),
                        ('Margin', ('tier_result', (lo, hi)))):
                    index.append((lbl, view))
                    recs.append(self._card_stat_row(self._by_kind[key]))
        if len(self._group_specs) > 1:
            # the closing grand block ('All' since a140 -- too many things
            # were already called Total); a forced single-group tower (the
            # one-step walk) is just its one block.
            for view, row in (('Consideration', self._grand_cons),
                              ('Obligation', self._grand_obl),
                              ('Margin', self._grand_result),
                              ('Impact',
                               self._by_kind[('total_impact', None)])):
                index.append(('All', view))
                recs.append(self._card_stat_row(row))
        return pd.DataFrame(
            recs, columns=_CARD_COLS,
            index=pd.MultiIndex.from_tuples(index, names=['Step', 'Side']))

    def _side_index(self):
        """The row MultiIndex for :attr:`economic_df`, aligned with the ledger
        plan order.

        Single-group: two-level ``(Side, Label)``. ``Side`` buckets every
        ledger row: legs and totals under their side (:data:`_SIDE_DEFAULTS`
        -- ``Consideration`` / ``Obligation``), the result under ``Margin``.
        ``Label`` is the presentation label: leg labels as declared; total
        rows and the result read ``Total``.

        Multi-group (a tower): three-level ``(Step, Side, Label)``. ``Step``
        is the group label, in ledger order; the grand rows close the sheet
        under step ``'All'`` (the a140 author rename -- too many things were
        already called Total). Running nets sit at ``(step, 'Margin', 'Net')``
        and the total impact at ``('All', 'Margin', 'Impact')`` -- the a132
        qualified-string lines (``'<group> total'``, ``'Net through <g>'``)
        became levels. A forced single-group tower (the one-step walk) is just
        its one block -- no grand rows, no impact.

        **The direct block and Net** ([Ledger-Side-Label-Levels], a189;
        [First-Step-Label], a191). A ledger that buys something distinguishes
        three margins that all used to read ``Total``: the grand result and the
        grand totals are ``Net``, a cession's own result stays ``Total``, and
        the ``sell`` group's own result takes :attr:`Group.margin_label` (the
        subject business's own name, defaulting to ``'Gross'``) so the direct
        block names its margin the way it already names its loss leg. Both are
        gated on the ledger actually containing a ``buy`` group, because only
        then is there anything to be direct or net *of*: a plain single-group
        ``pnl`` is one ``sell`` group whose legs are already net, and a ledger
        merging two sold books has no net to take, so both keep ``Total``
        throughout.

        A tier subtotal ([Tier-Subtotal-Rows]) takes its span's own ``Step``
        (``'All occurrence'`` / ``'All aggregate'`` on a layer-peeled walk) and
        reads ``Label == 'Total'``, like the group totals it sits between, so
        exhibits that filter ``Label`` on ``'Total'`` still treat it as the
        summary row it is.

        Presentation only -- the flat plan labels stay the canonical row keys
        everywhere else (:attr:`density_df`, :attr:`validation_df`, the sweep
        result keys).
        """
        groups = self._group_specs
        multi = self._tower
        v = _SIDE_DEFAULTS
        # only a ledger with a purchase in it has a direct block and a net
        has_buy = any(g.role == 'buy' for g in groups)
        net = 'Net' if has_buy else 'Total'
        tuples = []
        for label, kind, payload in self._plan:
            if kind == 'leg':
                gi, side, _li = payload
                t = (groups[gi].label, v[side], label)
            elif kind == 'group_total':
                gi, side = payload
                t = (groups[gi].label, v[side], 'Total')
            elif kind == 'group_result':
                g = groups[payload]
                lbl = ((g.margin_label or 'Gross')
                       if (has_buy and g.role == 'sell') else 'Total')
                t = (g.label, v['margin'], lbl)
            elif kind == 'running_net':
                t = (groups[payload].label, v['margin'], 'Net')
            elif kind == 'tier_total':
                lo, hi, side = payload
                t = (self._span_labels[(lo, hi)], v[side], 'Total')
            elif kind == 'tier_result':
                t = (self._span_labels[payload], v['margin'], 'Total')
            elif kind == 'grand_total':
                t = ('All', v[payload], net)
            elif kind == 'grand_result':
                t = ('All', v['margin'], net)
            elif kind == 'total_impact':
                t = ('All', v['margin'], 'Impact')
            else:
                raise ValueError(
                    f'unknown ledger row kind {kind!r} for row {label!r}; '
                    '_side_index must handle every kind _ledger_plan emits.')
            tuples.append(t)
        if not multi:
            return pd.MultiIndex.from_tuples(
                [t[1:] for t in tuples], names=['Side', 'Label'])
        return pd.MultiIndex.from_tuples(
            tuples, names=['Step', 'Side', 'Label'])

    def _scenario_ladder(self):
        """The [Kappa-Scenario-Percentiles] pass: ``{label: [cell per q]}``.

        For each ladder point ``q``, anchor the scenario on the **grand
        result**: ``x_q = gd.q(q)`` off the grand-result GD, form the exact
        atom slice ``result == x_q`` (the GD support *is* the set of exact
        result values, so float equality is exact), and take the
        probability-weighted mean of every row's signed values over the
        slice::

            cell(row, κq) = E[row | result == x_q]

        This is the library's kappa function ``E[X_i | X = x]`` (Portfolio's
        ``exeqa_*``) applied to the ledger. By linearity of conditional
        expectation every column **foots exactly** -- legs -> totals ->
        result add down the sheet to ``x_q``; the grand-result cell is
        automatically its own marginal quantile (``E[result | result = x] =
        x``), no special case. In-memory atoms route only (the massive
        one-sweep route keeps marginal ladders --
        [Massive-Kappa-Second-Sweep]).
        """
        res_vals = self._grand_result.values
        gd = self._grand_result.gd
        slices = []
        for q in PERCENTILE_LADDER:
            mask = res_vals == float(gd.q(q))
            pw = self._probs[mask]
            slices.append((mask, pw, float(pw.sum())))
        return {label: [float((row.values[mask] * pw).sum() / pm)
                        for mask, pw, pm in slices]
                for label, row in self._rows.items()}

    @property
    def stats_df(self):
        """The wrapped engine's canonical moment store, or an empty frame.

        ``stats_df`` means **one thing** across every first-class citizen: the
        ``(component, measure)`` by view moment store of a book
        ([PnL-Economic-Frames], 1.0.0a204). A P&L is a ledger over a book, so
        the natural reading is the book's own moments, and this delegates to
        :attr:`engine`.

        The ledger sheet that used to answer to this name is
        :attr:`economic_df`, which is what it always was: an accounting view,
        not a statistics frame.

        Returns
        -------
        pandas.DataFrame
            ``self.engine.stats_df`` when a stochastic engine is attached; an
            **empty** frame on a hand-built kernel P&L, which carries no
            engine (see :attr:`engine`). Empty rather than ``None`` or a
            raise: the FCC contract says the member exists, callers reach it
            defensively, and greater_tables renders an empty frame cleanly.
        """
        engine = self.engine
        if engine is None:
            return pd.DataFrame()
        return engine.stats_df

    @property
    def economic_df(self):
        """The full ledger x metrics sheet, in currency units -- the
        alignment/footing exhibit.

        Renamed from ``economic_df`` at 1.0.0a204 ([PnL-Economic-Frames]): this
        is an **accounting** view of the ledger, never a statistics frame, and
        the old name both misdescribed it and collided with the moment store
        every other first-class citizen serves under that name. See
        :attr:`economic_df`, which now delegates to the engine.

        Rows = the whole ledger (legs, totals, results, running nets, grand
        rows), in ledger order, indexed by :meth:`_side_index`: two-level
        ``(Side, Label)`` single-group, three-level ``(Step, Side, Label)`` on
        a tower. Columns are ``EX`` / ``SD`` / ``CV`` / ``Skew`` and the full
        :data:`PERCENTILE_LADDER`.

        The ``κ`` ladder columns are **scenario states, not per-row
        quantiles**: column ``κq`` is the state in which the grand result
        lands at its ``q``-quantile, and each cell is the conditional mean
        ``E[row | result == x_q]`` (:meth:`_scenario_ladder`). The header
        signals the semantics ([Decision-Ladder-Column-Names]): ``κ`` *means*
        conditioning happened. Consequences:

        * every column **foots** -- legs -> totals -> result add down the
          sheet to ``x_q``;
        * direction is uniform ([Decision-Kappa-Outcome-Direction]): a column
          is one state ordered by the **outcome** -- payoff sign convention,
          left tail bad, so ``κ01`` is the adverse state, full stop -- and
          e.g. a loss-sensitive (retro) premium correctly shows *high* in the
          bad columns. (This is deliberately reversed from the usual
          actuarial *loss* view, where the right tail is bad.)
        * the grand-result row's cells are its own marginal quantiles.

        ``EX / SD / CV / Skew`` are row properties and stay **marginal**.
        Per-row marginal quantiles remain one line away via
        ``density_df[row].q(p)``. Positional access or
        ``df.filter(like='κ')`` avoids typing the glyph.

        Two subtleties, documented rather than engineered away: where the
        result is **non-monotone** in the source (slides, swings, humps --
        the "switcheroo") the cell is the exact mean over the level set
        ``{result == x_q}``, well-defined but subtler to interpret; and a
        **constant** grand result (fully hedged) makes the conditioning event
        everything, so every cell equals its ``EX``. Ledgers with **no shared
        atoms** keep **marginal** ladders under plain ``P`` headers
        ([Decision-Kappa-Shared-Source-Rule]): a massive (one-sweep) source
        (conditioning needs a second sweep -- [Massive-Kappa-Second-Sweep] in
        ``dev/TODO.md``) and the stitched guaranteed-cost ``xpnl`` tower
        (independent marginals, no joint).

        Returns
        -------
        pandas.DataFrame
            MultiIndexed rows in ledger order; columns the metric names.

        See Also
        --------
        summary_df : the fixed headline card (marginal range percentiles).
        """
        if self._probs is None:               # no shared atoms: marginal
            # ladder, plain ``P`` headers (the massive one-sweep route and
            # the stitched guaranteed-cost tower -- the on-sheet signature of
            # [Decision-Kappa-Shared-Source-Rule]).
            data = [[_snap_noise(v) for v in row.stat_vector()]
                    for row in self._rows.values()]
            cols = _stat_names()
        else:
            ladder = self._scenario_ladder()
            data = [[_snap_noise(v) for v in
                     list(row.moments) + ladder[label]]
                    for label, row in self._rows.items()]
            cols = _stat_names(scenario=True)
        return pd.DataFrame(data, index=self._side_index(), columns=cols)

    # ------------------------------------------------------------------
    # raw materials for ratio exhibits ([PnL-Ratio-Frame])
    # ------------------------------------------------------------------
    def _blocks(self):
        """The ``(step label, group indices)`` pairs :attr:`economic_ratios_df` reports.

        Every group, then each tier span, then the whole ledger under ``'All'``
        (only where the grand rows exist, i.e. a genuinely multi-group ledger).
        Spans sit after the last group they cover, matching the sheet.
        """
        out = []
        for gi, g in enumerate(self._group_specs):
            out.append((g.label, (gi,)))
            for (lo, hi), lbl in self._span_labels.items():
                if hi - 1 == gi:
                    out.append((lbl, tuple(range(lo, hi))))
        if len(self._group_specs) > 1:
            out.append(('All', tuple(range(len(self._group_specs)))))
        return out

    def _block_amounts(self, gis):
        """Signed ``P / L / E / C`` and the per-atom vectors, over a group span.

        The four amounts carry the sign of their contribution **in the gross
        direction**: consideration enters as booked, obligations negated. A
        cession's ceded premium and recovery are therefore both negative, so
        every ratio built from them comes out with its conventional sign, the
        amounts add across blocks, and ``M == P - L - E - C`` holds identically
        (it *is* the signed row sum).

        Returns ``(amounts, vectors, m, fixed_p)``. ``vectors`` is ``None`` when
        the ledger has no shared atoms (the stitched and massive routes).
        ``fixed_p`` reports whether every leg feeding ``P`` is deterministic,
        which is what decides whether a mean-of-ratio can be formed without a
        joint: a constant denominator factors straight out of ``E[X / P]``.
        Every route carries an exact per-row standard deviation, so the test
        works off the rows rather than off the atoms.
        """
        n = 0 if self._probs is None else len(self._probs)
        amounts = {k: 0.0 for k in _RATIO_AMOUNTS}
        vectors = {k: np.zeros(n) for k in _RATIO_AMOUNTS} if n else None
        m = 0.0
        fixed_p = True
        for gi in gis:
            spec, g = self._group_specs[gi], self._egroups[gi]
            for legs, rows, side in ((spec.consideration, g.cons, 'cons'),
                                     (spec.obligation, g.obl, 'obl')):
                factor = 1.0 if side == 'cons' else -1.0
                for leg, r in zip(legs, rows):
                    bucket = _RATIO_BUCKET.get(
                        leg.kind, 'P' if side == 'cons' else 'L')
                    amounts[bucket] += factor * r.mean
                    if vectors is not None:
                        vectors[bucket] += factor * r.values
                    if bucket == 'P' and r.moments[1] > VALIDATION_NOISE:
                        fixed_p = False
                    m += r.mean
        return amounts, vectors, m, fixed_p

    @staticmethod
    def _mean_of_ratio(num, den, probs):
        """``E[num / den]`` per atom, or ``nan`` where the ratio is undefined.

        The mean of the ratio, as distinct from the ratio of the means. They
        differ exactly when the denominator is random and correlated with the
        numerator, which is what a variable-rated premium is. Undefined, hence
        ``nan``, if any atom carrying probability has a vanishing denominator.
        """
        if den is None:
            return float('nan')
        live = probs > 0
        if np.any(np.abs(den[live]) <= VALIDATION_NOISE):
            return float('nan')
        return float(np.sum(probs[live] * num[live] / den[live]))

    @property
    def economic_ratios_df(self):
        """Amounts and ratios per ledger block -- **raw materials**, not a card.

        Renamed from ``economic_ratios_df`` at 1.0.0a204 ([PnL-Economic-Frames]), which
        joins :attr:`economic_df` in naming the accounting family. It is
        **not** a view of that frame: splitting expense from commission needs
        :attr:`Leg.kind`, and the ``E_`` columns need the per-atom vectors,
        neither of which survives into the ledger sheet.

        One row per block: each group, each tier subtotal, and ``'All'`` on a
        multi-group ledger. Deliberately unformatted and absent from ``qd`` /
        the notebook repr, which render :attr:`summary_df`: this is the frame to
        slice, unstack and build presentation tables from, in the spirit of
        ``Portfolio.analyze_distortions``' ``pricing_df``. Transpose for the
        stat-down-the-side orientation.

        Returns
        -------
        pandas.DataFrame
            Indexed by ``'Step'``, with columns

            ``P``, ``L``, ``E``, ``C``
                Premium, loss, expense and commission, signed in the **gross
                direction** (see :meth:`_block_amounts`), so they add across
                blocks and ``M == P - L - E - C`` identically. ``L`` absorbs
                cession recoveries and any unclassified obligation leg; ``E``
                and ``C`` need :attr:`Leg.kind`.
            ``M``
                The block's signed result.
            ``LR``, ``ER``, ``CR``
                ``L / P``, ``E / P`` and ``(L + E + C) / P``: **ratios of
                means**, the convention of ``pricing_df`` and of a rate filing.
                Re-derived from this row's amounts, never averaged from the
                blocks below it. ``CR`` satisfies ``1 - CR == M / P``.
            ``E_LR``, ``E_ER``, ``E_CR``
                The same three as **means of ratios**, ``E[L / P]`` and so on.
                These part company with the plain ratios exactly when premium
                is random and correlated with loss, which is what a retro-rated
                account or a swing / slide / profit-commission cession is; a
                deterministic premium makes the pairs agree identically.

                Availability turns on whether the **denominator** is random,
                not on whether a joint happens to exist. A constant premium
                factors out of ``E[X / P]``, so these are exact on every route
                and simply repeat the plain ratios. A random premium needs the
                atoms to average over, so it is ``nan`` on a route that has
                none (the stitched peel, the massive sweep), and ``nan`` where
                some atom carrying probability has a vanishing premium, the
                ratio being undefined there. Never a silent fallback to the
                ratio of the means.
            ``P_share``, ``M_share``
                Premium and margin over the **first** block's, that block being
                the gross book in every builder. Ratios of means.

        See Also
        --------
        legs_df : the itemized companion, one row per declared leg.
        summary_df : the presentation-ready card.
        """
        probs = self._probs
        recs, index = [], []
        first = None
        pairs = (('LR', 'E_LR'), ('ER', 'E_ER'), ('CR', 'E_CR'))
        for label, gis in self._blocks():
            a, v, m, fixed_p = self._block_amounts(gis)
            p, ell, e, c = a['P'], a['L'], a['E'], a['C']
            if first is None:
                first = (p, m)
            live = abs(p) > VALIDATION_NOISE
            row = {'P': p, 'L': ell, 'E': e, 'C': c, 'M': m}
            for name, num in (('LR', ell), ('ER', e), ('CR', ell + e + c)):
                # ``or 0.0`` normalizes the signed zero a zero numerator over a
                # negative (cession) premium would otherwise leave on the sheet
                row[name] = (num / p or 0.0) if live else float('nan')
            # Three-way, on whether the DENOMINATOR is random -- not on whether
            # a joint happens to exist. A constant premium factors out of
            # E[X / P] exactly, so the mean of the ratio is available on every
            # route; only a random premium needs the atoms to average over.
            if not live:
                for _r, name in pairs:
                    row[name] = float('nan')
            elif fixed_p:
                for plain, name in pairs:
                    row[name] = row[plain]
            elif v is None:
                for _r, name in pairs:
                    row[name] = float('nan')
            else:
                for name, num in (('E_LR', v['L']), ('E_ER', v['E']),
                                  ('E_CR', v['L'] + v['E'] + v['C'])):
                    row[name] = self._mean_of_ratio(num, v['P'], probs)
            row['P_share'] = (p / first[0]
                              if abs(first[0]) > VALIDATION_NOISE
                              else float('nan'))
            row['M_share'] = (m / first[1]
                              if abs(first[1]) > VALIDATION_NOISE
                              else float('nan'))
            recs.append([row[c_] for c_ in _RATIO_COLS])
            index.append(label)
        return pd.DataFrame(recs, columns=list(_RATIO_COLS),
                            index=pd.Index(index, name='Step'))

    @property
    def legs_df(self):
        """One row per **declared** leg: where it sits, what it is, what it costs.

        The itemized companion to :attr:`economic_ratios_df`, and the only place
        :attr:`Leg.kind` surfaces. Raw materials, like ``economic_ratios_df``: the frame
        to group and pivot when the ratio you want is not one ``economic_ratios_df``
        carries. Derived rows (totals, results, running nets) are absent by
        design -- they are sums of these.

        Returns
        -------
        pandas.DataFrame
            Columns ``Step`` / ``Side`` / ``Label`` / ``kind`` / ``EX`` /
            ``SD``, in ledger order, matching the :attr:`economic_df` level names.
            ``EX`` is the **signed booked** mean, as on :attr:`economic_df`;
            ``kind`` is ``None`` for an unclassified leg.
        """
        recs = []
        for label, kind, payload in self._plan:
            if kind != 'leg':
                continue
            gi, side, li = payload
            spec = self._group_specs[gi]
            legs = spec.consideration if side == 'cons' else spec.obligation
            row = self._by_kind[(kind, payload)]
            sd = row.stat_vector()[1] if self._probs is None \
                else row.moments[1]
            recs.append([spec.label, _SIDE_DEFAULTS[side], label,
                         legs[li].kind, row.mean, _snap_noise(sd)])
        return pd.DataFrame(
            recs, columns=['Step', 'Side', 'Label', 'kind', 'EX', 'SD'])

    @property
    def density_df(self):
        """An ordered ``{row label: GridDistribution}`` -- the declared legs,
        the group results (multi-group), any tier subtotal results, and the
        grand result.

        **Not** a single stapled frame: each row keeps its own (irregular,
        exact) grid, so there is no lossy rebucketing onto a shared axis. The
        main customer is :meth:`plot`, which iterates and reads each GD's
        Series view (:meth:`GridDistribution.to_series`).
        """
        out = OrderedDict()
        for r in self._iter_legs():
            out[r.label] = r.gd
        if self._tower:
            for r in self._group_result_rows:
                out[r.label] = r.gd
            for span in self._span_labels:
                r = self._by_kind[('tier_result', span)]
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
    #: Ledger row kinds whose row is an evaluable **position**: each group's own
    #: result, the running net after each group, a tier subtotal result, and the
    #: grand result.
    #:
    #: ``total_impact`` is deliberately absent. It is a margin, but it is not a
    #: position: it is the difference between two of them, what the ledger's
    #: purchases did to the bottom line. Nobody holds it, so the stress it
    #: survives is not a question with an answer. (On a stitched peel it does
    #: not even have a law, being a delta of two statistics whose sides ride
    #: different marginals.) The ceded program *as a position* is already in the
    #: sheet, under the tier subtotal rows.
    _MARGIN_KINDS = ('group_result', 'running_net', 'tier_result',
                     'grand_result')

    def _row_role(self, kind, payload):
        """The role a margin row is evaluated under: ``'buy'`` when the row
        **is** a cession, so its margin is read from the seller's side.

        A purchased layer's margin to the buyer is negative by construction
        (premium paid less recoveries received), so it has no breakeven of its
        own. The question worth asking about a layer is what stress the seller's
        position survives, and :data:`~aggregate._pricing.EVAL_SIGN` answers it
        by negating a ``buy`` margin.

        Row by row: a group result takes its own group's role. A tier subtotal
        reads ``'buy'`` when every group it spans is a ``buy`` and ``'sell'``
        when every one is a ``sell``; a mixed span nets the two, so it reads
        ``'net'`` like the rows below. A running net and the grand result **are**
        a netting of buying against selling, which is neither of the two sides,
        so they read ``'net'``: already in payoff orientation, evaluated as
        booked.

        Parameters
        ----------
        kind, payload : str, object
            A ledger row's kind and payload, as :func:`_ledger_plan` emits them.
            Only the :data:`_MARGIN_KINDS` are meaningful here.

        Returns
        -------
        str
            A key of :data:`~aggregate._pricing.EVAL_SIGN`: ``'sell'``,
            ``'buy'`` or ``'net'``.
        """
        groups = self._group_specs
        if kind == 'group_result':
            return groups[payload].role
        if kind != 'tier_result':                 # running_net, grand_result
            return 'net'
        lo, hi = payload
        roles = {groups[i].role for i in range(lo, hi)}
        return roles.pop() if len(roles) == 1 else 'net'

    def evaluate(self, names=None):
        """Evaluate the position: the Cherny--Madan breakeven acceptability panel.

        A P&L is **evaluated, not priced**: you ask which distortion drives the
        risk-adjusted margin to zero, the breakeven stress the position
        survives. The breakeven ``gini_p`` is the single family-agnostic
        acceptability index (Cherny & Madan).

        Every **margin row of the ledger** is evaluated, not just the grand
        result, so a tower reads as a story: the gross deal, each reinsurance
        layer as a position in its own right, and the running net after each
        purchase. Reading a ``gini_p`` column down the ``net through ...`` rows
        is watching what buying cover does to the deal.

        A ceded layer is evaluated **from the seller's side** (:meth:`_row_role`
        and :data:`~aggregate._pricing.EVAL_SIGN`), because the buyer's margin
        on it is negative by construction and has no breakeven. The ``role``
        column says which rows those are: ``buy`` for a cession, ``sell`` for a
        book written, ``net`` for a running net or the grand result, which net
        the two against each other. That is what makes the panel a buy decision:
        a layer whose ``gini_p`` sits **above** the ``net`` row immediately over
        it is priced above the holder's own acceptability, so buying it lowers
        the net, and one below it raises the net.

        Only **positions** appear. The ``total impact`` row does not: it is the
        difference between two positions, what the purchases did to the bottom
        line, and nobody holds it, so the stress it survives is not a question
        with an answer. The ceded program as a position is already here, under
        the tier subtotal rows.

        Parameters
        ----------
        names : sequence of str, optional
            Distortion families. Defaults to
            :data:`~aggregate._pricing.EVAL_FAMILIES` (``ph`` / ``wang`` /
            ``dual`` / ``tvar``; ``ccoc`` is excluded).

        Returns
        -------
        pandas.DataFrame
            Tidy (long) form, ``MultiIndex`` rows ``(Step, distortion)`` and
            columns ``role`` / ``param_name`` / ``param`` / ``gini_p`` /
            ``error`` / ``status``. ``.unstack('distortion')`` gives the wide
            comparison view.

        Warns
        -----
        DegenerateEvaluationWarning
            Once per call, naming every step with no breakeven level. Since a
            cession is read from the seller's side this is now rare: it fires
            on a layer priced below its own expected recovery, which is a real
            finding rather than the routine consequence of paying for cover.

        See Also
        --------
        aggregate._pricing.evaluate_margin : the per-row solve and its math.

        Notes
        -----
        Each row's :attr:`~aggregate.GridDistribution` is already the signed
        margin in payoff orientation, exact and irregular for a ``bs = 0`` leg
        and a regular rebucket otherwise, so the quadrature adapts per row and
        no ledger shape is excluded. In particular there is no
        single-obligation-leg restriction: a margin is one random variable
        however many legs feed it, so an expense ledger evaluates like any
        other.
        """
        from ._pricing import evaluate_margin, warn_degenerate
        blocks, steps = [], []
        for label, kind, payload in self._plan:
            if kind not in self._MARGIN_KINDS:
                continue
            blocks.append(evaluate_margin(
                self._rows[label].gd,
                role=self._row_role(kind, payload), names=names))
            steps.append(label)
        panel = pd.concat(blocks, keys=steps, names=['Step'])
        warn_degenerate(panel, self.label)
        return panel

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

    # ------------------------------------------------------------------
    # Construction introspection ([Construction-Introspection])
    # ------------------------------------------------------------------
    def _route_name(self):
        """The evaluation route this ledger was built on, for narratives."""
        if self._stitched:
            return 'marginal-stitched (gd-backed rows, no shared atoms)'
        if self._probs is None:
            return 'massive one-sweep pushforward'
        return 'per-atom in-memory'

    def _generic_construction_description(self):
        """The structural fallback narrative for a hand-built kernel P&L."""
        ng = len(self._egroups)
        nl = sum(len(g.cons) + len(g.obl) for g in self._egroups)
        ladder = ('scenario (κ, conditional on the grand result)'
                  if self._probs is not None
                  else 'marginal (plain P headers)')
        return (f'Hand-built P&L {self.label!r}: {ng} group(s), {nl} declared '
                f'leg(s) over a {type(self._source).__name__} source; '
                f'{self._route_name()} evaluation; stats ladder {ladder}.')

    def _replay_block(self):
        """The closing executable-replay block of
        :attr:`construction_explanation`: the literal ``PnL(...)`` call that
        reproduces the object from its retained ``_group_specs`` and source
        reference. Constant legs render verbatim; function legs render as
        ``<fn>`` placeholders (a lambda has no literal form)."""
        def fn_repr(f):
            if callable(f):
                fname = getattr(f, '__name__', '<lambda>')
                return fname if fname != '<lambda>' else '<fn>'
            return repr(f)

        def legs_repr(legs):
            return '[' + ', '.join(
                f'Leg({leg.label!r}, {fn_repr(leg.func)}'
                + (f', bs={leg.bs:g}' if leg.bs else '')
                + (', is2d=True' if leg.is2d else '')
                + (f', kind={leg.kind!r}' if leg.kind else '') + ')'
                for leg in legs) + ']'

        groups = ',\n        '.join(
            f'Group({g.label!r}, {g.role!r}, '
            f'consideration={legs_repr(g.consideration)}, '
            f'obligation={legs_repr(g.obligation)})'
            for g in self._group_specs)
        return (f'Replay (function legs as <fn> placeholders):\n'
                f'    PnL(name={self.name!r},\n'
                f'        source=<{type(self._source).__name__}'
                f' {getattr(self._source, "name", "")}>,\n'
                f'        groups=[{groups}],\n'
                f'        result_name={self.result_name!r})')

    @property
    def construction_description(self):
        """One paragraph: how this P&L was built -- route, source, group
        count. Recorded by the DecL builders at construction; a hand-built
        kernel P&L gets a generic structural narrative, so the property is
        never absent. The full story is
        :attr:`construction_explanation`."""
        return (self._construction_description
                or self._generic_construction_description())

    @property
    def construction_explanation(self):
        """The full construction story, recorded by the builder that made
        this P&L: the engine and its clauses; the economics resolution
        (``deposit 2000 -> pc_occ = 2000``); which source each row reads; the
        booking signs; whether the stats ladder is scenario (``κ``) or
        marginal (``P``) and why; any ignored clauses (same wording as the
        :class:`~aggregate.constants.IgnoredDecLClauseWarning`). Closes with
        the executable replay block. Hand-built kernel P&Ls get a minimal
        generic narrative."""
        if self._construction_explanation is not None:
            return self._construction_explanation
        return (self._generic_construction_description() + '\n\n'
                + self._replay_block())

    @property
    def info(self):
        """Fixed-layout multi-line summary string (terse).

        Every row is always present, in the same order, for every ``PnL``; a
        value that does not apply (no engine, no economics) renders as
        ``n/a``. Shares the label/value convention
        (:func:`aggregate.constants.info_row`) with ``Aggregate`` /
        ``Portfolio``. The row catalogue is documented in
        ``dev/info-strings.rst``; the narrative twin is
        :attr:`construction_description`.
        """
        engine = self.engine
        rows = [
            ('pnl object name', self.name),
            ('label', self.label),
            ('groups', len(self._egroups)),
            ('legs', sum(len(g.cons) + len(g.obl) for g in self._egroups)),
            ('role', self.role or 'multi-group'),
            ('result name', self.result_name),
            ('E[result]', f'{self.est_m:,.6g}'),
            ('SD(result)', f'{self.est_sd:,.6g}'),
            ('CV(result)', f'{self.est_cv:,.6g}'),
            ('skew(result)', f'{self.est_skew:,.6g}'),
            ('P(X=0)', f'{self.prob_eq_0:.6g}'),
            ('engine', f'{type(engine).__name__} {engine.name}'
                       if engine is not None else INFO_NA),
            ('source', type(self._source).__name__),
            ('economics', 'resolved' if self.economics else INFO_NA),
        ]
        return '\n'.join(info_row(label, value) for label, value in rows)

    @property
    def program(self) -> str:
        """The DecL text this P&L was declared with.

        ``build`` stamps the declaration here directly; a P&L snapshotted from
        an engine falls back to :attr:`engine`'s program (the inner
        ``Aggregate`` / ``Portfolio`` the DecL ``pnl`` / ``xpnl`` statement
        built). ``''`` for a hand-built kernel P&L with neither.
        """
        return self._program or getattr(self.engine, 'program', '') or ''

    def _adopt_engine(self, engine, recipe=None):
        """Attach the wrapped engine and take the statement's trailer metadata.

        The DecL ``pnl`` / ``xpnl`` statement's own ``note`` / ``tags`` /
        ``hints`` / ``doc`` ride on its build recipe, not on the engine: for an
        agg engine the two happen to coincide (the parser merges the pnl spec
        into the inner :class:`Aggregate`), but for a ``port.NAME`` engine the
        Portfolio has metadata of its *own* which is not the P&L's. Reading
        from the recipe keeps both cases right.

        Parameters
        ----------
        engine : Aggregate or Portfolio
            The wrapped stochastic engine.
        recipe : dict, optional
            The ``_pnl_recipe`` the engine was carrying; its ``trailer_meta``
            entry supplies the metadata. Missing or absent leaves the empty
            defaults set in ``__init__``.
        """
        self.engine = engine
        meta = (recipe or {}).get('trailer_meta') or {}
        self.note = meta.get('note', '')
        self.tags = tuple(meta.get('tags', ()))
        self.hints = meta.get('hints', '')
        self.doc = meta.get('doc', '')

    @program.setter
    def program(self, value):
        self._program = value or ''

    # ``format_program`` / ``pprogram`` / ``pprogram_html`` come from
    # ``ProgramMixin``; the ``program`` property above overrides the mixin's
    # class-level default so a snapshotted P&L falls through to its engine.

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

    def _repr_html_(self):
        """HTML view: identity, ledger shape, headline moments, summary card.

        Notes
        -----
        Added at ``a171`` [PnL-Repr-HTML], which closed the last hole in the
        display surface: ``PnL`` was the only first-class class without one, so a
        bare ``pnl`` in a Jupyter cell fell back to ``__repr__``. Built on the
        same two pieces as :meth:`Portfolio._repr_html_`, an intro paragraph and
        ``summary_df``, so a P&L renders like everything else.

        The card is the fixed one settled at ``a134``: its percentiles are
        **marginal** and deliberately do not foot, because the footing sheet is
        ``economic_df``. That is said in the intro rather than left for the reader
        to discover from a column that does not add up.
        """
        ng = len(self._egroups)
        nl = sum(len(g.cons) + len(g.obl) for g in self._egroups)
        _gs = '' if ng == 1 else 's'
        _ls = '' if nl == 1 else 's'
        parts = [f'{ng} group{_gs}, {nl} leg{_ls}, resolving to '
                 f'<code>{self.result_name}</code>.',
                 f'E[result] {self.est_m:,.6g}, SD {self.est_sd:,.6g}, '
                 f'P(result = 0) {self.prob_eq_0:.4g}.',
                 'Percentile columns are marginal and do not foot; '
                 '<code>stats_df</code> is the footing sheet.']
        fmt = lambda x: f'{x:,.5g}'
        return '\n'.join([
            # ``label``, not ``_title_name``: the builders pass ``label=name``,
            # so ``_label`` is never None on a P&L and the ``label (handle)``
            # form would print the name twice. Matches __repr__ above.
            f'<h3>PnL object: {self.label}</h3>',
            '<p>' + ' '.join(parts) + '</p>',
            '<h4>Summary</h4>',
            self.summary_df.to_html(float_format=fmt, na_rep=''),
        ])


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
    don't"). The ladder columns here are **marginal by construction** --
    independent perspectives share no joint, so there is nothing to
    condition on (contrast the scenario columns of :attr:`PnL.economic_df`).

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
