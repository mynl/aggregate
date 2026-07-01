r"""The domain-agnostic profit-and-loss API over the pushforward engine.

A **P&L position** is *money in minus money out*, as a function of a random
state::

    result(state) = consideration(state) - obligation(state)     [role='sell']
    result(state) = obligation(state) - consideration(state)      [role='buy']

with ``state ~ source``. Group each component map's values over the source atoms
by output value and sum probability, and you get a :class:`GridDistribution` per
leg -- **exact**, with all its moments and percentiles. That triple
(consideration, obligation, result) **is** the whole object.

A :class:`PnL` **consumes and throws away** its stochastic engine: it is neither
an :class:`Aggregate` (no subclass) nor *has* one (no retained reference).
:func:`create_pnl` reads the probabilities and the component values off whatever
``source`` you hand it, builds the leg distributions **eagerly**, and discards
the source. A P&L is then just signed distributions + their exact moments + the
caller's labels -- a lightweight *accounting* value object that any domain can
build, with insurance as one caller among many.

Two things make this cross-domain:

1. **Labels are data, not code.** The caller names the legs through the **dict
   keys** of ``consideration`` / ``obligation`` (``'revenue'``, ``'fuel cost'``,
   ``'premium'``, ``'loss'``); there is no built-in perspective / category
   taxonomy.
2. **The constructor takes an opaque source slot.** A :class:`GridDistribution`,
   an :class:`Aggregate`, a :class:`~aggregate.bivariate.BivariateDistribution`,
   or a bare ``(values, probs)`` pair -- the maps decide what it means.

All magnitudes are **non-negative**; the ``role`` (``'sell'`` / ``'buy'``)
supplies the sign, so the cession sign-flip falls out of the role, never a hand
edit. See ``dev/plan-pnl-api.md``.
"""
from __future__ import annotations

import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd


#: The detailed percentile ladder used by :attr:`PnL.stats_df` (the full ladder;
#: the headline :attr:`PnL.summary_df` reports only ``P01`` / ``Median`` / ``P99``
#: off it). The exact ladder is standardized in ``[Reporting-Guidelines]``.
PERCENTILE_LADDER = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)

#: Distortion families evaluated by :meth:`PnL.evaluate` (the standard set minus
#: ``ccoc``, which needs an asset level a P&L does not carry).
_EVAL_FAMILIES = ('ph', 'wang', 'dual', 'tvar')


# ----------------------------------------------------------------------
# Source atomization: every source reduces to (coords, probs)
# ----------------------------------------------------------------------
def _source_atoms(source, probs):
    """Reduce a P&L ``source`` to ``(coords, probs, shape, bs)``.

    The opaque source slot is resolved here; the component maps are then
    evaluated by :func:`_eval_component` over ``coords`` (a tuple of the
    coordinate arrays a map is called with) and the result raveled to align with
    the flat ``probs`` vector.

    Parameters
    ----------
    source : object
        One of: a :class:`~aggregate.bivariate.BivariateDistribution` (axes 0/1
        + 2-D ``density``); a :class:`~aggregate._grid_distribution.GridDistribution`
        (``x`` / ``p``); an :class:`Aggregate` (read ``density_df``); or a bare
        ``(values, probs)`` pair.
    probs : array_like or None
        Probability override; ``None`` defaults from the slot.

    Returns
    -------
    coords : tuple of ndarray
        The argument arrays a component map is called with: ``(x,)`` for a 1-D
        source, ``(axis0[:, None], axis1[None, :])`` for a bivariate.
    probs : ndarray
        Flat (raveled) probability vector, one entry per atom.
    shape : tuple
        The broadcast shape of a component map's values (so a scalar component
        fills it).
    bs : float or None
        The source bucket size if it carries one (used by :meth:`PnL.evaluate`);
        ``None`` otherwise.
    """
    from ._grid_distribution import GridDistribution
    # bivariate: a full 2-D joint over (axis0, axis1)
    if hasattr(source, 'axis0') and hasattr(source, 'axis1') \
            and hasattr(source, 'density'):
        a0 = np.asarray(source.axis0, dtype=float)
        a1 = np.asarray(source.axis1, dtype=float)
        dens = np.asarray(source.density, dtype=float)
        p = dens.ravel() if probs is None else np.asarray(probs, dtype=float).ravel()
        return (a0[:, None], a1[None, :]), p, dens.shape, getattr(source, 'bs', None)
    # a GridDistribution slot
    if isinstance(source, GridDistribution):
        x = np.asarray(source.x, dtype=float)
        p = source.p if probs is None else np.asarray(probs, dtype=float)
        return (x,), np.asarray(p, dtype=float), x.shape, source.bs
    # an Aggregate (or anything exposing density_df): the gross loss grid
    if hasattr(source, 'density_df'):
        dd = source.density_df
        x = (dd['loss'].to_numpy(dtype=float) if 'loss' in dd.columns
             else dd.index.to_numpy(dtype=float))
        p = (dd['p_total'].to_numpy(dtype=float) if probs is None
             else np.asarray(probs, dtype=float))
        return (x,), p, x.shape, getattr(source, 'bs', None)
    # a bare (values, probs) pair
    vals = np.asarray(source[0], dtype=float)
    p = np.asarray(source[1], dtype=float) if probs is None \
        else np.asarray(probs, dtype=float)
    return (vals,), p, vals.shape, None


def _eval_component(spec, coords, shape):
    """Evaluate one component spec over the source atoms -> a flat value array.

    ``spec`` is a constant (broadcast to every atom) or a vectorized callable
    ``f(*coords)``. The result is broadcast to ``shape`` and raveled so it aligns
    with the flat ``probs`` vector.
    """
    if callable(spec):
        v = np.asarray(spec(*coords), dtype=float)
    else:
        v = np.asarray(float(spec), dtype=float)
    return np.broadcast_to(v, shape).ravel()


def _as_components(spec, default_name):
    """Normalize a component arg to an ordered ``{name: spec}`` dict.

    A dict is taken verbatim (its keys *are* the leg names, in order); a bare
    scalar / callable becomes a single ``{default_name: spec}`` leg.
    """
    if isinstance(spec, dict):
        return OrderedDict(spec)
    return OrderedDict([(default_name, spec)])


def _moments_of(values, probs):
    """Exact ``(mean, sd, cv, skew)`` of per-atom ``values`` under ``probs``.

    Taken straight off the atoms (no rebucketing), so these are the ground-truth
    moments the reports add to. A point mass has ``sd = cv = 0`` and ``skew =
    nan``; a zero mean has ``cv = nan``.
    """
    m = float((values * probs).sum())
    var = float((values * values * probs).sum()) - m * m
    var = var if var > 0 else 0.0
    sd = var ** 0.5
    cv = (sd / m) if m else float('nan')
    if sd > 0:
        skew = float((((values - m) ** 3) * probs).sum()) / sd ** 3
    else:
        skew = float('nan')
    return m, sd, cv, skew


# ----------------------------------------------------------------------
# A single leg: a labelled per-atom value array + the shared probs
# ----------------------------------------------------------------------
class _Leg:
    """One labelled component over a source: per-atom values + cached GD / moments.

    Internal to :class:`PnL`. Holds the **per-atom** value array (so dependent
    legs can be summed per atom by the result and the tower), and builds its
    exact :class:`GridDistribution` (group-by-value, sum-prob -- no rebucketing)
    and exact moments lazily.
    """

    def __init__(self, name, values, probs, *, is_loss_value):
        self.name = name
        self.values = np.asarray(values, dtype=float)
        self.probs = probs
        self.is_loss_value = is_loss_value
        self._gd = None
        self._moms = None

    @property
    def gd(self):
        """The leg's exact :class:`GridDistribution` (``bs=None``, irregular)."""
        if self._gd is None:
            from ._grid_distribution import GridDistribution
            ser = (pd.Series(self.probs, index=self.values)
                   .groupby(level=0).sum().sort_index())
            self._gd = GridDistribution(
                ser.index.to_numpy(dtype=float), ser.to_numpy(dtype=float),
                bs=None, name=self.name, is_loss_value=self.is_loss_value)
        return self._gd

    @property
    def moments(self):
        """Exact ``(mean, sd, cv, skew)`` straight off the atoms."""
        if self._moms is None:
            self._moms = _moments_of(self.values, self.probs)
        return self._moms

    @property
    def mean(self):
        return self.moments[0]


# ----------------------------------------------------------------------
# The P&L value object
# ----------------------------------------------------------------------
class PnL:
    """A profit-and-loss position: consideration legs, obligation legs, a result.

    A lightweight value object -- signed leg distributions + their exact moments
    + the caller's labels, with **no** retained stochastic engine. Build one with
    :func:`create_pnl` (the general entry) or :meth:`Aggregate.make_pnl` (object
    sugar); never construct directly.

    Parameters
    ----------
    name : str or None
        The position's name.
    role : {'sell', 'buy'}
        ``'sell'`` -- you receive the consideration and owe the obligation
        (``result = consideration - obligation``); ``'buy'`` -- you pay the
        consideration and receive the obligation (``result = obligation -
        consideration``).
    probs : ndarray
        The shared per-atom probability vector.
    consideration, obligation : OrderedDict
        ``{name: per-atom value array}`` for the magnitude legs (non-negative).
    result_name : str
        Label for the result leg.
    bs : float or None
        The source bucket size, retained **only** for :meth:`evaluate` (a scalar,
        not the engine).

    Notes
    -----
    The leg distributions are **exact** (group-by-value, sum-prob -- no FFT, no
    rebucketing), so a P&L has nothing of its own to validate: there is no
    ``validation_df``, and every moment and percentile is exact. The result is
    computed **per atom, then grouped** (never arithmetic on the two marginal
    GDs, which are dependent), so the per-atom difference carries the covariance
    for free -- means add and SDs do not.
    """

    def __init__(self, *, name, role, probs, consideration, obligation,
                 result_name, bs=None):
        if role not in ('sell', 'buy'):
            raise ValueError(f"role must be 'sell' or 'buy', got {role!r}.")
        self.name = name
        self.role = role
        self.result_name = result_name
        self._probs = np.asarray(probs, dtype=float)
        self._bs = bs
        # consideration is money-in (a payoff/asset, is_loss_value=False);
        # obligation is a loss (is_loss_value=True).
        self._cons = OrderedDict(
            (k, _Leg(k, v, self._probs, is_loss_value=False))
            for k, v in consideration.items())
        self._obl = OrderedDict(
            (k, _Leg(k, v, self._probs, is_loss_value=True))
            for k, v in obligation.items())
        # per-atom totals and the role-signed result
        self._cons_total_vals = sum(l.values for l in self._cons.values())
        self._obl_total_vals = sum(l.values for l in self._obl.values())
        sign = 1.0 if role == 'sell' else -1.0
        self._result = _Leg(result_name,
                            sign * (self._cons_total_vals - self._obl_total_vals),
                            self._probs, is_loss_value=False)
        self._cons_total = _Leg('Total consideration', self._cons_total_vals,
                                self._probs, is_loss_value=False)
        self._obl_total = _Leg('Total obligation', self._obl_total_vals,
                               self._probs, is_loss_value=True)

    # ------------------------------------------------------------------
    # Leg access
    # ------------------------------------------------------------------
    @property
    def result(self):
        """The result (net) leg as a :class:`GridDistribution`."""
        return self._result.gd

    @property
    def gd(self):
        """Alias for :attr:`result` -- the net is the P&L's distribution."""
        return self._result.gd

    def _summary_legs(self):
        """Ordered ``(label, _Leg, is_total)`` for the summary / density reports.

        consideration leaves, Total consideration (only if >1 leaf), obligation
        leaves, Total obligation (only if >1 leaf), the result.
        """
        out = [(l.name, l, False) for l in self._cons.values()]
        if len(self._cons) > 1:
            out.append(('Total consideration', self._cons_total, True))
        out += [(l.name, l, False) for l in self._obl.values()]
        if len(self._obl) > 1:
            out.append(('Total obligation', self._obl_total, True))
        out.append((self.result_name, self._result, False))
        return out

    @property
    def E_consideration(self):
        """``E[Total consideration]`` -- the denominator of the ``% Consid`` column."""
        return float((self._cons_total_vals * self._probs).sum())

    # ------------------------------------------------------------------
    # FCC report 1: the short headline
    # ------------------------------------------------------------------
    @property
    def summary_df(self):
        """The headline P&L table: one row per leg, ratio-and-percentile columns.

        Rows, in order: each **consideration** component (named / ordered by the
        ``consideration`` dict keys), **Total consideration** (only if >1
        component); each **obligation** component, **Total obligation** (only if
        >1); the **result** (named by ``result_name``). Columns:

        ``EX`` (mean -- a magnitude for a consideration / obligation row, the
        signed net for the result), ``% Consid`` (``EX`` over
        :attr:`E_consideration` -- the loss-ratio / margin family), ``SD``,
        ``CV``, ``Skew``, ``P01`` / ``Median`` / ``P99`` (the ``0.01`` / ``0.5``
        / ``0.99`` quantiles of the row's own leg).

        Returns
        -------
        pandas.DataFrame
            Indexed by leg label (index name ``'P&L'``).
        """
        denom = self.E_consideration
        rows = OrderedDict()
        for label, leg, _is_total in self._summary_legs():
            m, sd, cv, skew = leg.moments
            gd = leg.gd
            rows[label] = [
                m, (m / denom if denom else float('nan')), sd, cv, skew,
                float(gd.q(0.01)), float(gd.q(0.50)), float(gd.q(0.99))]
        df = pd.DataFrame.from_dict(
            rows, orient='index',
            columns=['EX', '% Consid', 'SD', 'CV', 'Skew', 'P01', 'Median', 'P99'])
        df.index.name = 'P&L'
        return df

    # ------------------------------------------------------------------
    # FCC report 2: the detailed stats x legs table
    # ------------------------------------------------------------------
    def stats_df(self, scale='Total'):
        """The detailed ``stats x legs`` table, each leg paired with a ``%-of-scale``.

        Every statistic (``EX`` / ``SD`` / ``CV`` / ``Skew`` and the full
        :data:`PERCENTILE_LADDER`) for every leg, paired with the same value as a
        fraction of a **fixed consideration scale** -- two columns per leg
        (``value``, ``% of <scale>``). Read everything "per unit of a fixed gross
        premium".

        Parameters
        ----------
        scale : str, default ``'Total'``
            The consideration element to divide by: ``'Total'`` (Total
            consideration) or a single ``consideration`` key. The scale **must be
            deterministic** (``SD == 0``) -- scaling a distribution by a
            stochastic consideration needs the joint we never form, so a non-fixed
            scale **warns** and divides by its ``EX``.

        Returns
        -------
        pandas.DataFrame
            Index = statistics; columns = a ``(leg, {'value', '% of <scale>'})``
            MultiIndex.
        """
        scale_leg = self._cons_total if scale == 'Total' else self._cons[scale]
        scale_mean, scale_sd, _, _ = scale_leg.moments
        if scale_sd > 0:
            warnings.warn(
                f'stats_df scale {scale!r} is stochastic (SD={scale_sd:.4g} != 0); '
                'a distribution cannot be scaled by a random consideration without '
                'the joint, so dividing by its mean EX instead.')
        denom = scale_mean if scale_mean else float('nan')
        scale_label = 'Total' if scale == 'Total' else scale
        stat_names = ['EX', 'SD', 'CV', 'Skew'] + \
            [f'P{int(round(q * 100)):02d}' for q in PERCENTILE_LADDER]
        data = OrderedDict()
        for label, leg, _is_total in self._summary_legs():
            m, sd, cv, skew = leg.moments
            gd = leg.gd
            vals = [m, sd, cv, skew] + [float(gd.q(q)) for q in PERCENTILE_LADDER]
            data[(label, 'value')] = vals
            data[(label, f'% of {scale_label}')] = [v / denom for v in vals]
        df = pd.DataFrame(data, index=stat_names)
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=['leg', ''])
        df.index.name = 'stat'
        return df

    # ------------------------------------------------------------------
    # FCC report 3: the per-leg distributions (a dict of GDs, no shared staple)
    # ------------------------------------------------------------------
    @property
    def density_df(self):
        """An ordered ``{leg_name: GridDistribution}`` -- consideration(s),
        obligation(s), result.

        **Not** a single stapled frame: each leg keeps its own (irregular, exact)
        grid, so there is no lossy rebucketing onto a shared axis. The main
        customer is :meth:`plot`, which iterates and reads each GD's Series view
        (:meth:`GridDistribution.to_series`).
        """
        out = OrderedDict()
        for l in self._cons.values():
            out[l.name] = l.gd
        for l in self._obl.values():
            out[l.name] = l.gd
        out[self.result_name] = self._result.gd
        return out

    # ------------------------------------------------------------------
    # Distribution accessors -- all delegate to the net result GD
    # ------------------------------------------------------------------
    @property
    def mean(self):
        """``E[result]`` (signed net)."""
        return self._result.moments[0]

    @property
    def sd(self):
        """SD of the result."""
        return self._result.moments[1]

    @property
    def cv(self):
        """CV of the result (meaningless near break-even; reported for symmetry)."""
        return self._result.moments[2]

    @property
    def skew(self):
        """Skewness of the result."""
        return self._result.moments[3]

    @property
    def prob_loss(self):
        """``P(result < 0)`` -- the probability the position loses money."""
        v = self._result.values
        return float(self._probs[v < 0].sum())

    def q(self, p, kind='lower'):
        """Quantile (value at risk) of the result. Delegates to the result GD."""
        return self._result.gd.q(p, kind)

    def var(self, p):
        """Value at risk = lower quantile of the result."""
        return self._result.gd.var(p)

    def cdf(self, x):
        """``P(result <= x)``."""
        return self._result.gd.cdf(x)

    def sf(self, x):
        """``P(result > x)``."""
        return self._result.gd.sf(x)

    # ------------------------------------------------------------------
    # Evaluation: the Cherny--Madan breakeven acceptability panel
    # ------------------------------------------------------------------
    def evaluate(self, names=None):
        """Evaluate the position: the Cherny--Madan breakeven acceptability panel.

        A P&L is **evaluated, not priced**: you price the **obligation** and ask
        which distortion drives the risk-adjusted net to zero -- the breakeven
        stress the position survives. That is the ``rho_g(obligation) =
        consideration`` calibration :meth:`Distortion.calibrate_set` solves over
        the full support, with ``consideration`` the held magnitude (no asset cap,
        no cost-of-capital inversion). The breakeven ``gini_p`` is the single
        family-agnostic acceptability index (Cherny & Madan).

        Parameters
        ----------
        names : sequence of str, optional
            Distortion families. Defaults to :data:`_EVAL_FAMILIES` (``ph`` /
            ``wang`` / ``dual`` / ``tvar`` -- ``ccoc`` is excluded, it needs an
            asset level a P&L does not carry).

        Returns
        -------
        pandas.DataFrame
            One row per family: ``param_name`` / ``param`` / ``error`` / ``gini_p``
            / ``area``.

        Notes
        -----
        Requires a **single** obligation over a regular (``bs``-lattice) source --
        the ordinary insurance case. Multi-obligation or non-uniform-grid
        evaluation is deferred.
        """
        if self._bs is None or len(self._obl) != 1:
            raise NotImplementedError(
                'evaluate requires a single obligation over a regular-grid source '
                '(an Aggregate / GridDistribution with bs); not available for this '
                'P&L.')
        from .spectral import Distortion
        from ._pricing import (_canonical_loss_frame, _calibration_survival,
                               _limited_ev)
        if names is None:
            names = _EVAL_FAMILIES
        # the obligation as a loss frame on its (regular) grid
        obl_gd = self._obl_total.gd
        dz_index = obl_gd.x
        frame = pd.DataFrame({'loss': dz_index, 'p_total': obl_gd.p},
                             index=pd.Index(dz_index, name='loss'))
        shim = _LossFrameShim(frame, self._bs)
        dz, c, _reverse = _canonical_loss_frame(shim)
        bs = self._bs
        a_full = float(dz.index[-1])
        S, ess_sup = _calibration_survival(dz, bs, a_full)
        el = _limited_ev(dz, bs, a_full + bs)
        # the consideration available to absorb the obligation
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
        """Plot the net result density and distribution (CDF).

        Two panels: the result density (A) and distribution (B), with the
        break-even line at 0 marked. Component overlays and the cession waterfall
        live on the :class:`PnLTower`, not the single leg.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from .plots import plot_pnl
        return plot_pnl(self, axd=axd, **kwargs)

    def __repr__(self):
        nc = len(self._cons)
        no = len(self._obl)
        return (f'PnL({self.name!r}: role={self.role}, '
                f'{nc} consideration, {no} obligation -> {self.result_name!r})')


class _LossFrameShim:
    """A minimal stand-in carrying ``density_df`` / ``bs`` / ``_is_loss_value``
    for :func:`_canonical_loss_frame` -- so :meth:`PnL.evaluate` reuses the
    pricing helpers without retaining an :class:`Aggregate`."""

    def __init__(self, density_df, bs):
        self.density_df = density_df
        self.bs = bs
        self._is_loss_value = True


# ----------------------------------------------------------------------
# The general constructor
# ----------------------------------------------------------------------
def create_pnl(source, *, consideration, obligation, role='sell',
               result_name='result', name=None, probs=None):
    """Build a :class:`PnL` from a source and its consideration / obligation maps.

    ``result(state) = consideration(state) - obligation(state)`` for ``role
    ='sell'`` (you are paid and owe the risky leg), or the negation for
    ``role='buy'`` (you pay and receive the risky leg). Both magnitudes are
    **non-negative**; the role supplies the sign.

    Parameters
    ----------
    source : object
        The opaque state slot: a :class:`~aggregate._grid_distribution.GridDistribution`,
        an :class:`Aggregate`, a :class:`~aggregate.bivariate.BivariateDistribution`,
        or a bare ``(values, probs)`` pair. Probabilities default from the slot.
    consideration, obligation : float, callable, or dict
        Each a non-negative magnitude: a constant, a vectorized callable
        ``f(*coords)`` of the source coordinate arrays, or an ordered
        ``{name: ...}`` dict of such (multi-component -- the parts surface as
        rows). A dict key names the leg; a bare value is named ``'consideration'``
        / ``'obligation'``.
    role : {'sell', 'buy'}, default 'sell'
        Whether you are selling (writing) or buying the obligation.
    result_name : str, default 'result'
        Label for the net result leg.
    name : str, optional
        The position name.
    probs : array_like, optional
        Probability override; defaults from the source slot.

    Returns
    -------
    PnL
    """
    coords, p, shape, bs = _source_atoms(source, probs)
    cons = _as_components(consideration, 'consideration')
    obl = _as_components(obligation, 'obligation')
    cons_vals = OrderedDict(
        (k, _eval_component(v, coords, shape)) for k, v in cons.items())
    obl_vals = OrderedDict(
        (k, _eval_component(v, coords, shape)) for k, v in obl.items())
    return PnL(name=name, role=role, probs=p,
               consideration=cons_vals, obligation=obl_vals,
               result_name=result_name, bs=bs)


# ----------------------------------------------------------------------
# The tower: legs sharing a source, with running net + benefit deltas
# ----------------------------------------------------------------------
class PnLTower:
    """A stack of :class:`PnL` legs over **one shared source**: the inuring waterfall.

    Every leg is a :func:`create_pnl` over the same source (same probability
    vector / atom order), so their results add **per atom**: the running net after
    leg ``k`` is the group-by of the summed per-atom results of legs ``0..k`` --
    means add, and the covariance is carried for free.

    The tower exposes the cascade: each leg's result, the **running net** after
    it, each **one-step delta** (the layer's benefit), and a **final total delta**
    (net-of-all vs the first leg). "Try this or that" is adding or dropping a leg.

    Parameters
    ----------
    legs : sequence of PnL
        The ordered legs, sharing a source.
    delta_names : sequence of str, optional
        Labels for the one-step deltas (benefit of each inuring layer). Defaults
        to ``'<leg> benefit'``.
    name : str, optional
        The tower name.
    """

    def __init__(self, legs, *, delta_names=None, name=None):
        legs = list(legs)
        if not legs:
            raise ValueError('a PnLTower needs at least one leg.')
        n = len(legs[0]._probs)
        for leg in legs:
            if len(leg._probs) != n:
                raise ValueError(
                    'all tower legs must share one source (equal-length probs).')
        self.legs = legs
        self.name = name
        self.delta_names = list(delta_names) if delta_names is not None else None
        self._probs = legs[0]._probs

    @property
    def net(self):
        """The final running-net :class:`PnL` (all legs combined per atom)."""
        return self._running_net(len(self.legs) - 1)

    def _running_net(self, k):
        """A :class:`PnL` for the net after legs ``0..k`` (combined per atom)."""
        total = sum(leg._result.values for leg in self.legs[:k + 1])
        return PnL(
            name=f'{self.name} net' if self.name else 'net', role='sell',
            probs=self._probs,
            consideration=OrderedDict([('net result', total)]),
            obligation=OrderedDict([('zero', np.zeros_like(total))]),
            result_name='net result', bs=self.legs[0]._bs)

    def _columns(self):
        """Ordered ``(label, _Leg)`` columns: legs, running nets, deltas, total.

        Layout: the base leg; then for each inuring leg its own column, the
        **running net** after it (labelled ``'net'`` for the final inuring leg,
        ``'net/<leg>'`` for an intermediate one so labels stay unique), and the
        **one-step benefit** delta; a final **total benefit** when there are >2
        legs.
        """
        legs = self.legs
        cols = [(legs[0].name or 'leg 0', legs[0]._result)]
        running = legs[0]._result.values
        for i in range(1, len(legs)):
            leg = legs[i]
            cols.append((leg.name or f'leg {i}', leg._result))
            new_running = running + leg._result.values
            is_final = i == len(legs) - 1
            net_label = 'net' if is_final else f'net/{leg.name or i}'
            cols.append((net_label, _Leg(net_label, new_running, self._probs,
                                         is_loss_value=False)))
            dn = (self.delta_names[i - 1]
                  if self.delta_names and i - 1 < len(self.delta_names)
                  else f'{leg.name} benefit')
            cols.append((dn, _Leg(dn, new_running - running, self._probs,
                                  is_loss_value=False)))
            running = new_running
        if len(legs) > 2:
            total = running - legs[0]._result.values
            cols.append(('total benefit',
                         _Leg('total benefit', total, self._probs,
                              is_loss_value=False)))
        return cols

    @property
    def gcn_df(self):
        """The cascade exhibit -- statistics (rows) x tower columns.

        Columns: each leg's result, the running **net** after it, each one-step
        **delta** (benefit), and a final **total benefit** (when >2 legs). Rows:
        ``EX`` / ``SD`` / ``CV`` / ``Skew`` and the full :data:`PERCENTILE_LADDER`.
        Only ``EX`` adds across the leg + net columns (means add); SD / CV / Skew /
        percentiles are per-column.
        """
        stat_names = ['EX', 'SD', 'CV', 'Skew'] + \
            [f'P{int(round(q * 100)):02d}' for q in PERCENTILE_LADDER]
        data = OrderedDict()
        for label, leg in self._columns():
            m, sd, cv, skew = leg.moments
            gd = leg.gd
            data[label] = [m, sd, cv, skew] + \
                [float(gd.q(q)) for q in PERCENTILE_LADDER]
        df = pd.DataFrame(data, index=stat_names)
        df.index.name = 'stat'
        return df

    @property
    def summary_df(self):
        """The net leg's :attr:`PnL.summary_df` (the headline of the cascade)."""
        return self.net.summary_df

    @property
    def density_df(self):
        """The net leg's :attr:`PnL.density_df`."""
        return self.net.density_df

    def q(self, p, kind='lower'):
        """Quantile of the final net."""
        return self.net.q(p, kind)

    def __repr__(self):
        return f'PnLTower({self.name!r}: {len(self.legs)} legs)'


def create_pnl_tower(legs, *, delta_names=None, name=None):
    """Stack :class:`PnL` legs into a :class:`PnLTower` (the inuring waterfall).

    Parameters
    ----------
    legs : sequence of PnL
        The ordered legs, all over one shared source.
    delta_names : sequence of str, optional
        Labels for the one-step benefit deltas.
    name : str, optional
        The tower name.

    Returns
    -------
    PnLTower
    """
    return PnLTower(legs, delta_names=delta_names, name=name)
