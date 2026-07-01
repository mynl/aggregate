r"""The domain-agnostic profit-and-loss API over the pushforward engine.

A **P&L position** is *money in minus money out*, as a function of a random
state::

    result(state) = consideration(state) - obligation(state)     [role='sell']
    result(state) = obligation(state) - consideration(state)      [role='buy']

with ``state ~ source``. Group each component map's values over the source atoms
by output value and sum probability, and you get a :class:`GridDistribution` per
leg -- **exact**, with all its moments and percentiles. That triple
(consideration, obligation, result) **is** the whole object.

A :class:`PnL` **does not depend on** its stochastic engine: it is not an
:class:`Aggregate` (no subclass), and although it keeps an opaque reference to
the ``source`` it was built over (:attr:`PnL.stochastic_engine`, for drill-down),
it never reads back through it. :func:`create_pnl` reads the probabilities and
the component values off whatever ``source`` you hand it and builds the leg
distributions **eagerly**; from then on a P&L is just signed distributions +
their exact moments + the caller's labels -- a lightweight *accounting* value
object that any domain can build, with insurance as one caller among many.

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
edit.

**Insurance is a caller, not a special case.** The Gross / Ceded / Net waterfall
is a :class:`PnLTower` of legs (:func:`gcn_tower_from_aggregate`); reinstatement
and variable rating are tower builders too (in :mod:`aggregate.reinstatement` /
:mod:`aggregate.variable_rating`). The gross/ceded/net vocabulary is *data* --
leg names -- the kernel never needs the word "ceded." See ``dev/plan-pnl-api.md``.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd

from .moments import VALIDATION_NOISE, _snap_noise

__all__ = ['create_pnl', 'create_pnl_tower', 'PnL', 'PnLTower']


#: The detailed percentile ladder used by :attr:`PnL.stats_df` and
#: :attr:`PnLTower.gcn_df` (the full ladder; the headline :attr:`PnL.summary_df`
#: reports only ``P01`` / ``Median`` / ``P99`` off it). The exact ladder is
#: standardized in ``[Reporting-Guidelines]``.
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
        A :class:`~aggregate.bivariate.BivariateDistribution` (axes 0/1 + 2-D
        ``density``); a :class:`~aggregate._grid_distribution.GridDistribution`
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
        The broadcast shape of a component map's values.
    bs : float or None
        The source bucket size if it carries one (used by :meth:`PnL.evaluate`).
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
    moments the reports add to. Two guards keep degenerate legs clean:

    * a **constant** leg (all ``values`` equal -- e.g. a fixed premium) returns
      ``sd = 0`` **exactly** regardless of any probability defect, so a fixed
      consideration reports ``SD = 0`` rather than the spurious
      ``sqrt(c^2 * Sum(p) * (1 - Sum(p)))`` picked up when the source density does
      not sum to exactly 1;
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
# A single leg: a labelled per-atom value array + the shared probs
# ----------------------------------------------------------------------
class _Leg:
    """One labelled component over a source: per-atom values + cached GD / moments.

    Internal to :class:`PnL` / :class:`PnLTower`. Holds the **per-atom** value
    array (so dependent legs sum per atom) and builds its exact
    :class:`GridDistribution` (group-by-value, sum-prob -- no rebucketing) and
    exact moments lazily.
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

    def stat_vector(self):
        """``[EX, SD, CV, Skew] + percentile ladder`` -- one exhibit column."""
        m, sd, cv, skew = self.moments
        gd = self.gd
        return [m, sd, cv, skew] + [float(gd.q(q)) for q in PERCENTILE_LADDER]


def _pct_label(q):
    """Percentile row/column label: ``f'P{q*100:.3g}'`` (``P1`` / ``P50`` / ``P99.5``)."""
    return f'P{q * 100:.3g}'


def _stat_names():
    """Row labels for a stats column: ``EX`` / ``SD`` / ``CV`` / ``Skew`` + ladder."""
    return ['EX', 'SD', 'CV', 'Skew'] + [_pct_label(q) for q in PERCENTILE_LADDER]


#: Rows on which a ``Scaled`` (per-unit-of-consideration) value is **not**
#: meaningful -- a ratio (``CV``) or a shape (``Skew``) does not scale, so its
#: scaled cell is ``nan``.
_UNSCALABLE_STATS = frozenset({'CV', 'Skew'})


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
        The position's name (identity handle).
    display_label : str or None
        Optional human display label (the DecL ``as`` clause); presentation only,
        preferred over ``name`` in repr / titles (see :attr:`display_name`).
    role : {'sell', 'buy'}
        ``'sell'`` -- you receive the consideration and owe the obligation
        (``result = consideration - obligation``); ``'buy'`` -- you pay the
        consideration and receive the obligation (``result = obligation -
        consideration``).
    probs : ndarray
        The shared per-atom probability vector.
    consideration, obligation : OrderedDict
        ``{name: per-atom value array}`` for the magnitude legs.
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
                 result_name, bs=None, stochastic_engine=None, scale=None,
                 display_label=None):
        if role not in ('sell', 'buy'):
            raise ValueError(f"role must be 'sell' or 'buy', got {role!r}.")
        self.name = name
        #: Optional human display label (the DecL ``as`` clause). Presentation
        #: only -- preferred over ``name`` in repr / exhibit titles via
        #: :attr:`display_name`; ``name`` stays the identity handle.
        self.display_label = display_label
        self.role = role
        self.result_name = result_name
        # Normalize the shared probability vector so it sums to exactly 1: a
        # source density that clips a little tail mass (Sum(p) = 1 - eps) would
        # otherwise give a *constant* consideration leg a spurious mean (c*Sum(p))
        # and variance (c^2*Sum(p)*(1-Sum(p))). A P&L *is* a probability
        # distribution; the clipped tail is a discretization artifact, so renormalize.
        p = np.asarray(probs, dtype=float)
        total = float(p.sum())
        self._probs = p / total if total > 0 else p
        self._bs = bs
        #: The opaque stochastic generator the P&L was built over (an
        #: :class:`Aggregate`, a :class:`GridDistribution`, ...). A P&L *has a*
        #: generator but never *depends* on it for its exhibits (those are read
        #: off the leg value arrays); retained for reference / drill-down only.
        self._stochastic_engine = stochastic_engine
        #: Optional attached domain waterfall producing :attr:`margin_df` -- a
        #: :class:`PnLTower` (Gross/Ceded/Net cession) or an analysis object
        #: (:class:`~aggregate.reinstatement.ReinstatementAnalysis` /
        #: :class:`~aggregate.variable_rating.VariableRatingAnalysis`). Both expose
        #: a ``gcn_df``. ``None`` on a plain P&L.
        self._waterfall = None
        self._cons = OrderedDict(
            (k, _Leg(k, v, self._probs, is_loss_value=False))
            for k, v in consideration.items())
        self._obl = OrderedDict(
            (k, _Leg(k, v, self._probs, is_loss_value=True))
            for k, v in obligation.items())
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
        self._scale_value, self._scale_label = self._resolve_scale(scale)

    def _resolve_scale(self, scale):
        """Commit the ``stats_df`` / ``summary_df`` scale at construction.

        Returns ``(value, label)``. ``scale`` is ``None`` (the default -- the
        **expected total consideration**, the natural unitizer), a bare number
        (an explicit scale, e.g. the RP case's ``gross - deposit``), or a
        consideration leg name.
        """
        if scale is None:
            return self._cons_total.mean, 'consideration'
        if isinstance(scale, str):
            return self._cons[scale].mean, scale
        return float(scale), 'scale'

    @property
    def stochastic_engine(self):
        """The opaque stochastic generator the P&L was built over (or ``None``).

        A P&L *has a* generator (an :class:`Aggregate`, a
        :class:`GridDistribution`, a bivariate, ...) but is agnostic about which,
        and never *depends* on it: the exhibits are read off the leg value arrays
        captured at construction. Retained for reference / drill-down only.
        """
        return self._stochastic_engine

    @property
    def tower(self):
        """The attached :class:`PnLTower` (Gross/Ceded/Net cession), or ``None``.

        Present on a cession-bearing P&L built through the Gross/Ceded/Net path;
        ``None`` on a plain P&L or one backed by an analysis object (see
        :attr:`analysis`).
        """
        return self._waterfall if isinstance(self._waterfall, PnLTower) else None

    @property
    def analysis(self):
        """The attached domain analysis object, or ``None``.

        A :class:`~aggregate.reinstatement.ReinstatementAnalysis` or
        :class:`~aggregate.variable_rating.VariableRatingAnalysis` when the P&L was
        built from a reinstatement / variable-rating program (the drill-down home
        for the treaty maps, ``validation_df``, ``tail_df``, ``plot``); ``None``
        otherwise.
        """
        return None if isinstance(self._waterfall, (PnLTower, type(None))) \
            else self._waterfall

    @property
    def margin_df(self):
        """The Gross / Ceded / Net **margin waterfall** (stats x perspective).

        Rows are ``EX / SD / CV / Skew`` and the :data:`PERCENTILE_LADDER`;
        columns are the ``Gross`` / ``Ceded`` / ``Net`` perspectives plus the
        one-step benefit / impact deltas. Only ``EX`` adds across the split; SD /
        CV / Skew / percentiles are per-column ("means add, SDs don't"). Forwarded
        from the attached cession waterfall (:attr:`tower` / :attr:`analysis`); a
        **plain** P&L has no cession, so it raises.

        Returns
        -------
        pandas.DataFrame

        Raises
        ------
        AttributeError
            If the P&L carries no cession waterfall.
        """
        if self._waterfall is None:
            raise AttributeError(
                f'{self.name!r} is a plain P&L with no cession; margin_df is only '
                'defined for a Gross/Ceded/Net position (a reinsurance / '
                'reinstatement / variable-rating program).')
        return self._waterfall.gcn_df

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
        """``E[Total consideration]`` -- the default scale (``role='sell'``)."""
        return float((self._cons_total_vals * self._probs).sum())

    @property
    def scale(self):
        """The committed ``(value, label)`` scale for the ``Scaled`` columns.

        Fixed at construction (default: the expected total consideration). Every
        ``Scaled`` cell in :attr:`summary_df` / :attr:`stats_df` divides by this
        one number, so the reports never re-derive it per call.
        """
        return self._scale_value, self._scale_label

    # ------------------------------------------------------------------
    # FCC report 1: the short headline
    # ------------------------------------------------------------------
    @property
    def summary_df(self):
        """The headline P&L table: one row per leg, ratio-and-percentile columns.

        **Fixed shape.** Rows, in order: each **consideration** component (named /
        ordered by the ``consideration`` dict keys), **Total consideration** (only
        if >1 component); each **obligation** component, **Total obligation** (only
        if >1); the **result** (named by ``result_name``) -- so the row set varies
        only with the number of legs, never with the P&L's flavor. Columns are
        always ``EX`` (mean -- a magnitude for a consideration / obligation row,
        the signed net for the result), ``Scaled`` (``EX`` over the committed
        :attr:`scale`), ``SD``, ``CV``, ``Skew``, ``P1`` / ``Median`` / ``P99``.

        Returns
        -------
        pandas.DataFrame
            Indexed by leg label (index name ``'P&L'``).
        """
        denom = self._scale_value
        scalable = denom and abs(denom) > VALIDATION_NOISE
        rows = OrderedDict()
        for label, leg, _is_total in self._summary_legs():
            m, sd, cv, skew = leg.moments
            gd = leg.gd
            rows[label] = [
                m, (m / denom if scalable else float('nan')),
                _snap_noise(sd), cv, _snap_noise(skew),
                float(gd.q(0.01)), float(gd.q(0.50)), float(gd.q(0.99))]
        df = pd.DataFrame.from_dict(
            rows, orient='index',
            columns=['EX', 'Scaled', 'SD', 'CV', 'Skew', 'P1', 'Median', 'P99'])
        df.index.name = 'P&L'
        return df

    # ------------------------------------------------------------------
    # FCC report 2: the detailed stats x legs table
    # ------------------------------------------------------------------
    @property
    def stats_df(self):
        """The detailed ``stats x legs`` table, each leg paired with a ``Scaled`` value.

        **Fixed shape.** Every statistic (``EX`` / ``SD`` / ``CV`` / ``Skew`` and
        the full :data:`PERCENTILE_LADDER`, labelled ``P1`` / ``P5`` / ... ) for
        every leg, paired with the same value divided by the committed
        :attr:`scale` -- two columns per leg (``value``, ``Scaled``). The scale is
        a single number fixed at construction (default: the expected total
        consideration), so this is a property, not a method.

        ``CV`` and ``Skew`` are unitless already, so their ``Scaled`` cells are
        ``nan`` (scaling a ratio / shape is meaningless); a break-even scale
        (``|value| <= VALIDATION_NOISE``) makes every ``Scaled`` cell ``nan``.

        Returns
        -------
        pandas.DataFrame
            Index = statistics (single level); columns = a ``(leg, {'value',
            'Scaled'})`` MultiIndex.
        """
        denom = self._scale_value
        scalable = denom and abs(denom) > VALIDATION_NOISE
        stat_names = _stat_names()
        data = OrderedDict()
        for label, leg, _is_total in self._summary_legs():
            vals = [_snap_noise(v) for v in leg.stat_vector()]
            scaled = [
                (float('nan') if (name in _UNSCALABLE_STATS or not scalable)
                 else v / denom)
                for name, v in zip(stat_names, vals)]
            data[(label, 'value')] = vals
            data[(label, 'Scaled')] = scaled
        df = pd.DataFrame(data, index=stat_names)
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=['leg', ''])
        df.index.name = 'stat'
        return df

    # ------------------------------------------------------------------
    # FCC report 3: the per-leg distributions (a dict of GDs)
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
        the full support, with ``consideration`` the held magnitude. The breakeven
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
            One row per family: ``param_name`` / ``param`` / ``error`` / ``gini_p``
            / ``area``.

        Notes
        -----
        Requires a **single** obligation over a regular (``bs``-lattice) source --
        the ordinary insurance case.
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
        break-even line at 0 marked.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from .plots import plot_pnl
        return plot_pnl(self, axd=axd, **kwargs)

    @property
    def display_name(self):
        """The human display label if set (the DecL ``as`` clause), else ``name``.

        Presentation only -- repr / exhibit titles prefer it; ``name`` stays the
        identity handle. See dev/plan-decl-labels.md.
        """
        return self.display_label or self.name

    def __repr__(self):
        nc = len(self._cons)
        no = len(self._obl)
        return (f'PnL({self.display_name!r}: role={self.role}, '
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
               result_name='result', name=None, probs=None, scale=None,
               display_label=None):
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
        ``{name: ...}`` dict of such. A dict key names the leg; a bare value is
        named ``'consideration'`` / ``'obligation'``.
    role : {'sell', 'buy'}, default 'sell'
        Whether you are selling (writing) or buying the obligation.
    result_name : str, default 'result'
        Label for the net result leg.
    name : str, optional
        The position name (identity handle).
    display_label : str, optional
        Optional human display label (the DecL ``as`` clause); presentation only,
        preferred over ``name`` in repr / exhibit titles. See dev/plan-decl-labels.md.
    probs : array_like, optional
        Probability override; defaults from the source slot.
    scale : float or str, optional
        The unitizer for the ``Scaled`` columns, fixed at construction. ``None``
        (default) uses the expected total consideration; a number is an explicit
        scale (e.g. the reinstatement-programme ``gross - deposit``); a string
        names a consideration leg.

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
               result_name=result_name, bs=bs, stochastic_engine=source,
               scale=scale, display_label=display_label)


# ----------------------------------------------------------------------
# The tower: an ordered set of exhibit columns + a net value object
# ----------------------------------------------------------------------
class PnLTower:
    """An ordered set of P&L exhibit columns with a net value object.

    A tower is the inuring **waterfall** exhibit. Two builders feed it:

    * :meth:`from_legs` -- legs over **one shared source** (the joint / degenerate
      cases: reinstatement, variable rating, and the generic
      :func:`create_pnl_tower`). Their results add **per atom**, so the columns
      are each leg, the **running net** after it, each **one-step benefit** delta,
      and a **total benefit**. Means add and the covariance is carried for free.
    * :meth:`from_perspectives` -- independent columns each over their **own**
      marginal (the plain Gross / Ceded / Net case, which has no joint). The
      columns are the perspectives plus **impact** columns (per-statistic
      ``target - base`` -- means add, SDs do not).

    Do not construct directly; use a builder or :func:`create_pnl_tower`.

    Parameters
    ----------
    columns : list
        Ordered exhibit columns, each ``('leg', label, _Leg)`` or
        ``('impact', label, target_label, base_label)``.
    net : PnL
        The net position, for the forwarded value-object surface
        (:attr:`summary_df` / :attr:`density_df` / :meth:`q`).
    name : str, optional
        The tower name.
    """

    def __init__(self, columns, net, *, name=None):
        self._cols = columns
        self._net = net
        self.name = name

    # -- builders -------------------------------------------------------
    @classmethod
    def from_legs(cls, legs, *, delta_names=None, name=None):
        """A per-atom tower of :class:`PnL` legs over one shared source."""
        legs = list(legs)
        if not legs:
            raise ValueError('a PnLTower needs at least one leg.')
        probs = legs[0]._probs
        n = len(probs)
        for leg in legs:
            if len(leg._probs) != n:
                raise ValueError(
                    'all tower legs must share one source (equal-length probs).')
        cols = [('leg', legs[0].name or 'leg 0', legs[0]._result)]
        running = legs[0]._result.values
        for i in range(1, len(legs)):
            leg = legs[i]
            cols.append(('leg', leg.name or f'leg {i}', leg._result))
            new_running = running + leg._result.values
            is_final = i == len(legs) - 1
            net_label = 'net' if is_final else f'net/{leg.name or i}'
            cols.append(('leg', net_label,
                         _Leg(net_label, new_running, probs, is_loss_value=False)))
            dn = (delta_names[i - 1] if delta_names and i - 1 < len(delta_names)
                  else f'{leg.name} benefit')
            cols.append(('leg', dn,
                         _Leg(dn, new_running - running, probs, is_loss_value=False)))
            running = new_running
        if len(legs) > 2:
            total = running - legs[0]._result.values
            cols.append(('leg', 'total benefit',
                         _Leg('total benefit', total, probs, is_loss_value=False)))
        net = PnL(name=f'{name} net' if name else 'net', role='sell', probs=probs,
                  consideration=OrderedDict([('net result', running)]),
                  obligation=OrderedDict([('zero', np.zeros(n))]),
                  result_name='net result', bs=legs[0]._bs)
        return cls(cols, net, name=name)

    @classmethod
    def from_perspectives(cls, perspectives, net, impacts, *, name=None):
        """A perspective table: independent columns + per-statistic impact columns.

        Parameters
        ----------
        perspectives : list of (str, _Leg)
            ``(label, result-leg)`` for each waterfall perspective.
        net : PnL
            The net-perspective P&L (the forwarded value-object surface).
        impacts : list of (str, str, str)
            ``(label, target_label, base_label)`` impact columns.
        """
        cols = [('leg', label, leg) for label, leg in perspectives]
        cols += [('impact', label, tgt, base) for label, tgt, base in impacts]
        return cls(cols, net, name=name)

    # -- the exhibit ----------------------------------------------------
    @property
    def gcn_df(self):
        """The cascade exhibit -- statistics (rows) x waterfall columns.

        Rows: ``EX`` / ``SD`` / ``CV`` / ``Skew`` and the full
        :data:`PERCENTILE_LADDER`. Columns are the tower's leg columns plus any
        impact columns (per-statistic ``target - base``). Only ``EX`` adds across
        the leg + net columns; SD / CV / Skew / percentiles are per-column
        ("means add, SDs don't").
        """
        data = OrderedDict()
        vecs = {}
        for entry in self._cols:
            if entry[0] == 'leg':
                _, label, leg = entry
                v = leg.stat_vector()
                vecs[label] = v
                data[label] = v
            else:
                _, label, tgt, base = entry
                data[label] = [a - b for a, b in zip(vecs[tgt], vecs[base])]
        df = pd.DataFrame(data, index=_stat_names())
        df.index.name = 'stat'
        return df

    # -- forwarded net value-object surface -----------------------------
    @property
    def net(self):
        """The net :class:`PnL` (the final waterfall position)."""
        return self._net

    @property
    def summary_df(self):
        """The net leg's :attr:`PnL.summary_df` (the headline of the cascade)."""
        return self._net.summary_df

    @property
    def density_df(self):
        """The net leg's :attr:`PnL.density_df`."""
        return self._net.density_df

    def q(self, p, kind='lower'):
        """Quantile of the final net."""
        return self._net.q(p, kind)

    def plot(self, axd=None, **kwargs):
        """Plot the net position (delegates to the net :class:`PnL`)."""
        return self._net.plot(axd=axd, **kwargs)

    def __repr__(self):
        n = sum(1 for c in self._cols if c[0] == 'leg')
        return f'PnLTower({self.name!r}: {n} columns)'


def create_pnl_tower(legs, *, delta_names=None, name=None):
    """Stack :class:`PnL` legs into a per-atom :class:`PnLTower` (inuring waterfall).

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
    return PnLTower.from_legs(legs, delta_names=delta_names, name=name)


# ----------------------------------------------------------------------
# The insurance preset: Gross / Ceded / Net as a PnLTower of legs
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
    mixed-basis group is the generic ``'expense'``. See dev/plan-decl-labels.md.
    """
    bases = {basis for basis, _ in terms}
    return f'{next(iter(bases))} expense' if len(bases) == 1 else 'expense'


def _expected_gross_loss(agg):
    """The expected **gross** loss E[X] read off the aggregate's exact density."""
    if agg.occ_reins is not None or agg.agg_reins is not None:
        rd = agg.reins_density_df
        return float((rd['loss'].to_numpy() * rd['p_agg_gross'].to_numpy()).sum())
    dd = agg.density_df
    return float((dd['loss'].to_numpy() * dd['p_total'].to_numpy()).sum())


def resolve_expense(agg, expense_spec, gross_premium):
    """The gross expense ``E_G`` as a scalar from a DecL ``expense`` spec.

    Sums **all** terms across **all** groups (grouping is a plain-path reporting
    nicety; the Gross/Ceded/Net scalar is the total). ``expense_spec`` may be the
    grouped form, a legacy flat list of ``(basis, value)`` terms, or a bare
    ``(basis, value)`` tuple (see :func:`_normalize_expense_groups`). For each term
    ``'fixed'`` is a currency amount, ``'premium'`` a fraction of ``gross_premium``,
    ``'loss'`` a fraction of the **expected gross loss** (deterministic).
    ``None`` / empty -> ``0``.
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

    (Contrast :func:`resolve_expense`, which returns the single deterministic
    scalar the Gross/Ceded/Net commission split needs -- summed across groups.)
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


def _gcn_magnitudes(agg, gross, ceded, net, expense_spec, econ):
    """Per-perspective premium / expense magnitudes for the GCN waterfall.

    Ported from the legacy ``PnL._gcn_magnitudes``: premium per perspective (a
    DecL ceded-premium clause gives a per-side split; a scalar API GCN books its
    single ceded amount on the side present), gross expense on the gross leg, a
    commission credit on each cession, so net expense ``= E_G - C_occ - C_agg``
    and means add across the split.
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


def _first_reins_label(labels):
    """The first non-``None`` cession display label from a per-layer label list.

    The tower consolidates a basis's cessions into a single ``ceded`` column, so
    the first labeled layer names it. ``None`` / empty -> ``None``.
    """
    if not labels:
        return None
    return next((x for x in labels if x is not None), None)


def _perspective_pnl(agg, persp, prem, exp, probs_col, loss_x):
    """One waterfall perspective as a :class:`PnL` over its own loss marginal.

    A cession (``role='buy'``) pays the ceded premium and receives the recovery
    (+ commission credit); a retained perspective (``role='sell'``) receives the
    premium and owes the loss (+ expense). The perspective's ``result`` is its
    underwriting result.
    """
    from ._grid_distribution import GridDistribution
    gd = GridDistribution(loss_x, probs_col, bs=None)
    if persp in _GCN_CEDED:
        return create_pnl(
            gd, role='buy', name=persp,
            consideration={'ceded premium': prem},
            obligation={'recovery': lambda x: x, 'commission': exp},
            result_name=persp)
    return create_pnl(
        gd, role='sell', name=persp,
        consideration={'premium': prem},
        obligation={'loss': lambda x: x, 'expense': exp},
        result_name=persp)


def gcn_tower_from_aggregate(agg, *, gross, ceded, net=None, expense_spec=None,
                             gcn_economics=None, name=None):
    """The Gross / Ceded / Net waterfall as a :class:`PnLTower` (perspective table).

    Each perspective is a :func:`create_pnl` over its own exact aggregate
    **marginal** from ``Aggregate.reins_density_df`` (no ``(L, R)`` joint is
    formed for the plain case -- means add across the split; SD / percentiles are
    per-column). The tower columns are the present perspectives plus the inuring
    **impact** columns; :attr:`PnLTower.summary_df` forwards the net perspective.

    Parameters
    ----------
    agg : Aggregate
        The updated risky leg, carrying occurrence and/or aggregate reinsurance.
    gross, ceded : float
        The gross premium received and the (scalar API) ceded premium paid.
    net : float, optional
        Override the retained net premium (default ``gross - ceded``).
    expense_spec : list or tuple, optional
        A DecL ``expense`` spec (see :func:`resolve_expense`).
    gcn_economics : dict, optional
        Per-side ceded premiums / commissions from DecL clauses
        (``pc_occ`` / ``pc_agg`` / ``c_occ`` / ``c_agg``).
    name : str, optional
        The tower name.

    Returns
    -------
    PnLTower
    """
    rd = agg.reins_density_df
    if rd is None:
        raise ValueError(
            'the Gross/Ceded/Net view requires reinsurance on the risky leg; the '
            "aggregate carries no occurrence / aggregate treaty.")
    loss_x = rd['loss'].to_numpy(dtype=float)
    prem_mag, exp_mag, has_occ, has_agg = _gcn_magnitudes(
        agg, gross, ceded, net, expense_spec, gcn_economics)
    both = has_occ and has_agg
    final_net = 'net_agg' if has_agg else 'net_occ'

    def persp_pnl(persp):
        col = rd[_GCN_LOSS_MARGINAL[persp]].to_numpy(dtype=float)
        return _perspective_pnl(agg, persp, prem_mag[persp], exp_mag[persp],
                                col, loss_x)

    # A reins-clause ``as`` label (first labeled layer on each basis) renames that
    # basis's **cession** column in the margin waterfall; the ``net`` columns keep
    # their structural names. See dev/plan-decl-labels.md ([Labels-Reins]).
    occ_ceded_label = _first_reins_label(getattr(agg, 'occ_reins_label', None))
    agg_ceded_label = _first_reins_label(getattr(agg, 'agg_reins_label', None))
    # ordered display: (label, perspective-key) for present tiers
    perspectives = [('gross', 'gross')]
    impacts = []
    if has_occ:
        q = ' occ' if both else ''
        ceded_col = occ_ceded_label or f'ceded{q}'
        perspectives += [(ceded_col, 'ceded_occ'), (f'net{q}', 'net_occ')]
        if both:
            impacts.append(('occ impact', f'net{q}', 'gross'))
    if has_agg:
        q = ' agg' if both else ''
        ceded_col = agg_ceded_label or f'ceded{q}'
        perspectives += [(ceded_col, 'ceded_agg'), (f'net{q}', 'net_agg')]
        if both:
            impacts.append(('agg impact', f'net{q}', f'net occ'))
    final_label = perspectives[-1][0]
    impacts.append(('impact', final_label, 'gross'))
    # build each perspective's result leg (labelled by its display label)
    persp_legs = []
    net_pnl = None
    for label, key in perspectives:
        pnl = persp_pnl(key)
        persp_legs.append((label, pnl._result))
        if key == final_net:
            net_pnl = pnl
    tower = PnLTower.from_perspectives(persp_legs, net_pnl, impacts, name=name)
    #: the resolved Gross/Ceded/Net economics (a DecL clause's per-side split, or
    #: the scalar API's single amounts) -- the observable of the DecL premium /
    #: commission resolution (``deposit`` / ``rol`` / ``rate`` / ``cede``).
    tower.economics = dict(gcn_economics) if gcn_economics is not None \
        else {'gross': float(gross), 'ceded': float(ceded)}
    return tower
