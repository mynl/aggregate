"""Property-catastrophe reinstatement premiums: terms and stochastic-ceded analysis.

Occurrence excess-of-loss reinsurance with paid / free reinstatements, where the
**ceded premium is stochastic**: the reinstatement premium ``h(R)`` is a function
of the unlimited annual occurrence recovery ``R``. This module holds two objects:

* :class:`ReinstatementTerms` -- an immutable description of one reinstatement
  basis (occurrence limit ``y``, price multipliers, deposit premium) that owns
  the deterministic recovery ``A(R)`` and reinstatement-premium ``h(R)`` maps and
  the annual recovery cap ``(m+1)y``. No FFT code -- pure vectorized arithmetic.
* :class:`ReinstatementAnalysis` -- the engine that pairs the terms with the
  joint ``(L, R)`` law (via :meth:`Aggregate.occ_bivariate`) and pushes the
  signed accounting legs forward into gross / ceded / net underwriting
  distributions and exhibits.

The mathematics and worked examples live in ``dev/reinstatements.md`` (correct
arithmetic) and ``dev/pre-plan-reinstatements.md`` (full spec); the integration
design is ``dev/plan-reinstatements.md`` with the shared engine seam in
``dev/plan-variable-rating-appendix.md``.

Submodule access only (no top-level re-export), per the project layout::

    from aggregate.reinstatement import ReinstatementTerms
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

__all__ = ['ReinstatementTerms', 'ReinstatementAnalysis']

#: Adverse-tail levels for the headline ``summary_df`` (pre-plan section 13).
SUMMARY_PERCENTILES = (0.90, 0.95, 0.99, 0.995, 0.996, 0.999)


@dataclass(frozen=True)
class ReinstatementTerms:
    r"""Immutable terms of one occurrence reinstatement basis.

    Describes a ``y`` xs ``a`` occurrence layer with ``m`` paid reinstatements,
    pro rata as to amount but not as to time. The reinstatement premium charged
    on tranche ``j`` of unlimited recovery is ``alpha_j`` times the base rate on
    line ``r = deposit / limit``; the annual recovery is capped at ``(m+1) y``.

    Parameters
    ----------
    limit : float
        Occurrence limit ``y`` (also the base for the rate on line ``r`` and the
        default reinstatement tranche width).
    rates : tuple of float
        Price multipliers ``(alpha_1, ..., alpha_m)`` relative to ``r``; ``len``
        is the number of reinstatements ``m``. ``0`` is a free reinstatement,
        ``1`` a full-rate one, ``0.5`` half rate. The **empty** tuple is the
        zero-reinstatements case (``m = 0``): a single annual limit ``y`` with no
        reinstatement premium (DecL ``no reinstatements``).
    deposit : float
        Base (deposit) ceded premium ``D`` in currency -- the layer's
        Phase-1 ``deposit | rol | rate`` premium. The base rate on line is
        ``r = deposit / limit``.
    widths : tuple of float, optional
        Per-tranche widths ``(w_1, ..., w_m)`` for irregular schedules. Default
        (``None``) is equal full-limit reinstatements (``w_j = limit``). Set by
        :meth:`from_tranches`.
    recovery_cap : float, optional
        Total annual recovery capacity ``Y``. Default (``None``) is
        ``limit + sum(widths)`` (``= (m+1) y`` for equal full reinstatements).
    premium_function : callable, optional
        Escape hatch: an arbitrary vectorized nondecreasing ``h(R)`` overriding
        the tranche sum (set by :meth:`from_callable`).

    Notes
    -----
    The reinstatement-premium function is the tranche sum (pre-plan section 5.1)

    .. math::

        h(R) = r \sum_{j=1}^m \alpha_j \big[(R - b_j)_+ \wedge w_j\big],
        \qquad b_j = \sum_{i<j} w_i,

    nonnegative, nondecreasing and piecewise linear, constant once all
    reinstatement capacity ``sum(w_j)`` is consumed. Recovery is
    ``A(R) = R \wedge Y``. Both are deterministic nondecreasing functions of the
    single random variable ``R``, hence comonotone (``dev/reinstatements.md``).
    """

    limit: float
    rates: tuple
    deposit: float
    widths: Optional[tuple] = None
    recovery_cap: Optional[float] = None
    premium_function: Optional[Callable] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    def __post_init__(self):
        if not (self.limit > 0 and np.isfinite(self.limit)):
            raise ValueError(
                f'ReinstatementTerms: limit must be finite and > 0, '
                f'got {self.limit!r}.')
        if not (self.deposit >= 0 and np.isfinite(self.deposit)):
            raise ValueError(
                f'ReinstatementTerms: deposit must be finite and >= 0, '
                f'got {self.deposit!r}.')
        rates = tuple(float(a) for a in self.rates)
        object.__setattr__(self, 'rates', rates)
        if self.premium_function is None:
            # An empty ``rates`` tuple is the *zero-reinstatements* case: ``m = 0``,
            # a single annual limit ``y``, no reinstatement premium (DecL ``no
            # reinstatements``). The omitted-clause / free + unlimited case carries
            # no terms object at all, so it never reaches here.
            if any((not np.isfinite(a)) or a < 0 for a in rates):
                raise ValueError(
                    f'ReinstatementTerms: rates must be finite and nonnegative, '
                    f'got {rates!r}.')
        if self.widths is not None:
            widths = tuple(float(w) for w in self.widths)
            object.__setattr__(self, 'widths', widths)
            if len(widths) != len(rates):
                raise ValueError(
                    f'ReinstatementTerms: widths ({len(widths)}) and rates '
                    f'({len(rates)}) must have the same length.')
            if any((not np.isfinite(w)) or w <= 0 for w in widths):
                raise ValueError(
                    f'ReinstatementTerms: widths must be finite and > 0, '
                    f'got {widths!r}.')
        if self.recovery_cap is not None and not (
                self.recovery_cap > 0 and np.isfinite(self.recovery_cap)):
            raise ValueError(
                f'ReinstatementTerms: recovery_cap must be finite and > 0, '
                f'got {self.recovery_cap!r}.')
        if self.premium_function is not None:
            self._validate_callable()

    def _validate_callable(self):
        """Check the escape-hatch ``premium_function`` is vectorized, finite,
        nonnegative and nondecreasing on a sample of the recovery range."""
        probe = np.linspace(0.0, self.total_recovery_capacity, 257)
        try:
            vals = np.asarray(self.premium_function(probe), dtype=float)
        except Exception as e:                       # noqa: BLE001
            raise ValueError(
                'ReinstatementTerms.from_callable: premium_function must accept '
                'and return a NumPy array (it failed to evaluate on a vector). '
                f'Underlying error: {e!r}') from e
        if vals.shape != probe.shape:
            raise ValueError(
                'ReinstatementTerms.from_callable: premium_function must be '
                f'vectorized (input shape {probe.shape}, output shape '
                f'{vals.shape}).')
        if not np.all(np.isfinite(vals)):
            raise ValueError(
                'ReinstatementTerms.from_callable: premium_function must be '
                'finite over the recovery range.')
        if np.any(vals < -1e-9):
            raise ValueError(
                'ReinstatementTerms.from_callable: premium_function must be '
                'nonnegative.')
        if np.any(np.diff(vals) < -1e-9):
            raise ValueError(
                'ReinstatementTerms.from_callable: premium_function must be '
                'nondecreasing.')

    # ------------------------------------------------------------------
    # alternative constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_tranches(cls, *, limit, widths, rates, deposit, recovery_cap=None):
        """Build terms with general (possibly unequal) reinstatement tranche widths.

        Parameters
        ----------
        limit : float
            Occurrence limit ``y`` (base for ``r = deposit / limit``).
        widths : sequence of float
            Reinstatement tranche widths ``(w_1, ..., w_m)``.
        rates : sequence of float
            Price multipliers, same length as ``widths``.
        deposit : float
            Base ceded premium ``D``.
        recovery_cap : float, optional
            Total recovery capacity; default ``limit + sum(widths)``.
        """
        return cls(limit=float(limit), rates=tuple(rates), deposit=float(deposit),
                   widths=tuple(widths), recovery_cap=recovery_cap)

    @classmethod
    def from_callable(cls, *, limit, premium_function, deposit, recovery_cap):
        """Build terms from an arbitrary vectorized nondecreasing ``h(R)``.

        For advanced / irregular reinstatement pricing that the tranche schedule
        cannot express. ``premium_function`` must accept and return a NumPy
        array and be nonnegative and nondecreasing over ``[0, recovery_cap]``
        (validated; :class:`ValueError` otherwise).

        Parameters
        ----------
        limit : float
            Occurrence limit ``y``.
        premium_function : callable
            Vectorized ``h(R) -> reinstatement premium``.
        deposit : float
            Base ceded premium ``D``.
        recovery_cap : float
            Total annual recovery capacity ``Y``.
        """
        return cls(limit=float(limit), rates=(), deposit=float(deposit),
                   recovery_cap=float(recovery_cap),
                   premium_function=premium_function)

    # ------------------------------------------------------------------
    # derived quantities
    # ------------------------------------------------------------------
    @property
    def rol(self):
        """Base rate on line ``r = deposit / limit`` ([decl-rol])."""
        return self.deposit / self.limit

    @property
    def n_reinstatements(self):
        """Number of reinstatements ``m = len(rates)``."""
        return len(self.rates)

    @property
    def _tranche_widths(self):
        """Per-tranche widths array (defaulting to equal full limits)."""
        if self.widths is not None:
            return np.asarray(self.widths, dtype=float)
        return np.full(self.n_reinstatements, float(self.limit))

    @property
    def _breakpoints(self):
        """Recovery consumed before each tranche, ``b_j = sum_{i<j} w_i``."""
        w = self._tranche_widths
        return np.concatenate(([0.0], np.cumsum(w)[:-1])) if len(w) else \
            np.zeros(0)

    @property
    def reinstatement_capacity(self):
        """Reinstated capacity ``sum(w_j)`` (``= m * y`` for full reinstatements)."""
        if self.premium_function is not None:
            return self.total_recovery_capacity - self.limit
        return float(self._tranche_widths.sum())

    @property
    def total_recovery_capacity(self):
        """Total annual recovery capacity ``Y`` (``= (m+1) y`` standard)."""
        if self.recovery_cap is not None:
            return float(self.recovery_cap)
        return float(self.limit + self._tranche_widths.sum())

    @property
    def maximum_reinstatement_premium(self):
        """Largest reinstatement premium ``h(Y)`` (all capacity consumed)."""
        return float(self.reinstatement_premium(self.total_recovery_capacity))

    # ------------------------------------------------------------------
    # the deterministic maps (vectorized, no Python loops)
    # ------------------------------------------------------------------
    def recovery(self, R):
        r"""Actual annual recovery ``A(R) = R \wedge Y``.

        Parameters
        ----------
        R : float or ndarray
            Unlimited annual occurrence recovery.

        Returns
        -------
        float or ndarray
            ``min(R, total_recovery_capacity)``.
        """
        return np.minimum(np.asarray(R, dtype=float),
                          self.total_recovery_capacity)

    def reinstatement_premium(self, R):
        r"""Reinstatement premium ``h(R)`` (the tranche sum, or the callable).

        Parameters
        ----------
        R : float or ndarray
            Unlimited annual occurrence recovery.

        Returns
        -------
        float or ndarray
            ``r * sum_j alpha_j [(R - b_j)_+ wedge w_j]`` (pre-plan section 5.1),
            or ``premium_function(R)`` for the escape-hatch form.
        """
        R = np.asarray(R, dtype=float)
        if self.premium_function is not None:
            return self.premium_function(R)
        a = np.asarray(self.rates, dtype=float)
        b = self._breakpoints
        w = self._tranche_widths
        # broadcast R over the m tranches; clip each into [0, w_j]
        tranche = np.clip(R[..., None] - b, 0.0, w)
        return self.rol * np.sum(a * tranche, axis=-1)

    def ceded_premium(self, R):
        """Total ceded premium ``D + h(R)`` (deposit plus reinstatement premium)."""
        return self.deposit + self.reinstatement_premium(R)

    def __repr__(self):
        if self.premium_function is not None:
            return (f'ReinstatementTerms(limit={self.limit:g}, callable h, '
                    f'deposit={self.deposit:g}, '
                    f'Y={self.total_recovery_capacity:g})')
        return (f'ReinstatementTerms(limit={self.limit:g}, '
                f'rates={self.rates}, deposit={self.deposit:g}, '
                f'rol={self.rol:.4g}, m={self.n_reinstatements}, '
                f'Y={self.total_recovery_capacity:g})')


class ReinstatementAnalysis:
    r"""Stochastic-ceded reinstatement analysis: the engine behind the GCN exhibits.

    Pairs a :class:`ReinstatementTerms` with the joint law of
    ``(L, R)`` -- gross annual loss ``L`` and unlimited annual occurrence
    recovery ``R`` -- and pushes the signed accounting legs forward into the
    eleven named one-dimensional underwriting distributions and the
    gross / ceded / net exhibits. Because both the recovery ``A(R)`` and the
    reinstatement premium ``h(R)`` are deterministic functions of ``R``, **every**
    output is a deterministic pushforward of the *one* joint FFT2 -- no second
    convolution (pre-plan sections 8-11).

    Construct via :meth:`Aggregate.reinstatement_analysis` (the convenience
    wrapper) or directly from a ``('gross', 'ceded')`` joint::

        joint = agg.occ_bivariate(views=('gross', 'ceded'))
        analysis = ReinstatementAnalysis(joint, terms, gross_premium=P_G)

    Parameters
    ----------
    source : BivariateDistribution
        The ``(L, R)`` joint -- axis 0 gross loss, axis 1 unlimited ceded
        recovery (the ``('gross', 'ceded')`` view of :meth:`Aggregate.occ_bivariate`).
    terms : ReinstatementTerms
        The reinstatement basis owning ``recovery`` / ``reinstatement_premium``.
    gross_premium : float
        Fixed gross premium ``P_G``.
    percentiles : tuple of float, optional
        Adverse-tail levels for :attr:`summary_df`. Default
        :data:`SUMMARY_PERCENTILES`.

    Notes
    -----
    The per-``(L, R)`` signed accounting (pre-plan section 6), with payoff signs
    (premium received ``+``, paid ``-``; loss a negative obligation; recovery
    ``+``), ``A = terms.recovery(R)``, ``RP = terms.reinstatement_premium(R)``,
    ``D = terms.deposit``:

    ======  =====================  ================  ===========================
    view    premium                loss              underwriting
    ======  =====================  ================  ===========================
    gross   ``P_G``                ``-L``            ``P_G - L``
    ceded   ``-(D + RP)``          ``+A``            ``A - D - RP``
    net     ``P_G - D - RP``       ``-(L - A)``      ``P_G - D - RP - L + A``
    ======  =====================  ================  ===========================

    Means add across the split (``gross + ceded = net`` in the exhibit's signed
    convention) and down to UW; SDs and percentiles do not, so the volatility and
    percentile rows are read off the joint pushforward.
    """

    #: The exhibit column legs, as ``(label, ceded?, premium_fn, loss_fn, uw_fn)``
    #: builders are created per-instance in ``_build`` (they close over terms).

    def __init__(self, source, terms, gross_premium, *,
                 percentiles=SUMMARY_PERCENTILES):
        self.source = source
        self.terms = terms
        self.gross_premium = float(gross_premium)
        self.percentiles = tuple(percentiles)
        # axis 0 is gross loss L, axis 1 is unlimited ceded recovery R
        names = source.axis_names
        if names not in (('gross_loss', 'unlimited_ceded_loss'),
                         ('gross (G)', 'ceded (C)')) and 'gross' not in str(names[0]):
            # not fatal -- occ_bivariate(('gross','ceded')) labels axis0 'gross'
            pass
        self._distributions = None
        self._stats = None

    # ------------------------------------------------------------------
    # the eleven named legs (deterministic pushforwards of the joint)
    # ------------------------------------------------------------------
    def _leg_functions(self):
        """Return ``{name: (function, is_loss_value)}`` for the eleven legs.

        Each ``function(l, r)`` is vectorized on the broadcast axes; ``l`` is
        gross loss, ``r`` unlimited ceded recovery (pre-plan section 6).
        """
        P_G = self.gross_premium
        D = self.terms.deposit
        A = self.terms.recovery
        h = self.terms.reinstatement_premium
        return {
            'gross_loss':            (lambda l, r: l, True),
            'gross_uw':              (lambda l, r: P_G - l, False),
            'unlimited_ceded_loss':  (lambda l, r: r, True),
            'ceded_loss':            (lambda l, r: A(r), True),
            'reinstatement_premium': (lambda l, r: h(r), True),
            'ceded_premium':         (lambda l, r: D + h(r), True),
            # ceded column UW = recovery - ceded premium (cedant's cession effect)
            'ceded_uw':              (lambda l, r: A(r) - (D + h(r)), False),
            'net_premium':           (lambda l, r: P_G - D - h(r), True),
            'net_loss':              (lambda l, r: l - A(r), True),
            'net_uw':                (lambda l, r: P_G - D - h(r) - l + A(r), False),
        }

    @property
    def distributions(self):
        """Dict of the eleven named 1-D leg :class:`GridDistribution` objects.

        Built lazily by pushing each accounting leg forward over the joint; the
        fixed ``gross_premium`` is a degenerate one-point distribution so the
        reporting API stays uniform (pre-plan section 11.3).
        """
        if self._distributions is None:
            from ._grid_distribution import GridDistribution
            d = {}
            for name, (fn, is_loss) in self._leg_functions().items():
                d[name] = self.source.pushforward(fn, name=name,
                                                  is_loss_value=is_loss)
            # fixed gross premium: degenerate point mass
            d['gross_premium'] = GridDistribution(
                np.array([self.gross_premium]), np.array([1.0]),
                name='gross_premium', is_loss_value=True)
            self._distributions = d
        return self._distributions

    # ------------------------------------------------------------------
    # exact moment store (the EX column; single source of truth)
    # ------------------------------------------------------------------
    @property
    def stats_df(self):
        """Exact per-leg moments on the source joint grid (mean / sd / cv / skew).

        The canonical moment store (the audit's "EX" column): each leg's exact
        ``sum p_ij f(l_i, r_j)^k`` straight off the joint (pre-plan section 15),
        not the rebucketed pushforward. Means here are the ground truth the GCN
        Mean rows add to.
        """
        if self._stats is None:
            rows = {}
            for name, (fn, _) in self._leg_functions().items():
                rows[name] = self.source.transformed_moments(fn, max_order=3)
            df = pd.DataFrame(rows).T
            df.loc['gross_premium'] = {'mass': 1.0, 'mean': self.gross_premium,
                                       'var': 0.0, 'sd': 0.0, 'cv': 0.0,
                                       'skew': np.nan}
            self._stats = df
        return self._stats

    def _exact(self, name):
        """``(mean, sd, skew)`` of a leg from the exact :attr:`stats_df`."""
        row = self.stats_df.loc[name]
        sk = row.get('skew', np.nan)
        return (float(row['mean']), float(row['sd']),
                float(sk) if pd.notna(sk) else 0.0)

    # ------------------------------------------------------------------
    # the reused GCN waterfall (via the shared assembler)
    # ------------------------------------------------------------------
    @property
    def gcn_df(self):
        """Gross / Ceded / Net underwriting waterfall (the reused PnL exhibit).

        Built through the shared :func:`aggregate._pnl.gcn_assemble_column` so it
        carries the identical sections, signs, CV rows and percentile placement
        as :meth:`PnL.gcn_df` -- the difference is real but contained: the ceded
        and net **premium** legs are stochastic (nonzero ``CV Premium``), and the
        UW percentiles are read off the ``(L, R)`` pushforward, not a 1-D
        marginal (decision 2). Means add ``gross + ceded = net``.
        """
        from ._pnl import gcn_assemble_column, GCN_PERCENTILES
        d = self.distributions
        cols = [
            ('gross', False, 'gross_premium', 'gross_loss', 'gross_uw'),
            ('ceded', True, 'ceded_premium', 'ceded_loss', 'ceded_uw'),
            ('net', False, 'net_premium', 'net_loss', 'net_uw'),
        ]
        data = {}
        for label, ceded, prem, loss, uw in cols:
            pm, psd, _ = self._exact(prem)
            lm, lsd, lsk = self._exact(loss)
            uw_dist = d[uw]
            pct = {lvl: float(uw_dist.q(lvl)) for lvl in GCN_PERCENTILES}
            data[label] = gcn_assemble_column(
                ceded=ceded, prem_mean=pm, prem_sd=psd,
                loss_mean=lm, loss_sd=lsd, loss_skew=lsk,
                exp_mean=0.0, exp_sd=0.0, uw_pctiles=pct)
        # impact column: net vs gross (percent change; points on Ratio rows)
        from ._pnl import PnL, _GCN_RATIO_ROWS
        data['impact'] = PnL._gcn_impact(data['net'], data['gross'],
                                         _GCN_RATIO_ROWS)
        df = pd.DataFrame(data)
        df.index = pd.MultiIndex.from_tuples(df.index, names=['section', 'item'])
        return df

    # ------------------------------------------------------------------
    # the headline summary (pre-plan section 13)
    # ------------------------------------------------------------------
    @property
    def summary_df(self):
        """Headline Gross / Ceded / Net / Impact / Pct Impact table (pre-plan section 13).

        Rows: Premium, CV(Premium), Loss, CV(Loss), Underwriting Result,
        SD(Underwriting Result), then the adverse-tail percentile rows
        (:attr:`percentiles`). The Impact column is Net - Gross (``= -Ceded`` for
        premium / loss); Pct Impact is the percent change vs gross. Reports
        SD(UW) (a near-zero margin) but CV(premium) / CV(loss) (materially
        nonzero), following the PnL convention.
        """
        prem = {'gross': self._exact('gross_premium'),
                'ceded': self._exact('ceded_premium'),
                'net': self._exact('net_premium')}
        loss = {'gross': self._exact('gross_loss'),
                'ceded': self._exact('ceded_loss'),
                'net': self._exact('net_loss')}
        uw = {'gross': self._exact('gross_uw'),
              'ceded': self._exact('ceded_uw'),
              'net': self._exact('net_uw')}
        d = self.distributions
        uw_dist = {'gross': d['gross_uw'], 'ceded': d['ceded_uw'],
                   'net': d['net_uw']}
        cols = ['gross', 'ceded', 'net']
        rows = {}
        rows[('', 'Premium')] = {c: prem[c][0] for c in cols}
        rows[('', 'CV(Premium)')] = {
            c: (prem[c][1] / prem[c][0] if prem[c][0] else np.nan) for c in cols}
        rows[('', 'Loss')] = {c: loss[c][0] for c in cols}
        rows[('', 'CV(Loss)')] = {
            c: (loss[c][1] / loss[c][0] if loss[c][0] else np.nan) for c in cols}
        rows[('', 'Underwriting')] = {c: uw[c][0] for c in cols}
        rows[('', 'SD(Underwriting)')] = {c: uw[c][1] for c in cols}
        for lvl in self.percentiles:
            # adverse tail of an underwriting *result* (payoff): the bad outcome
            # is the low quantile, so the lvl-adverse row is q(1 - lvl).
            rows[('Percentile', f'{lvl:g}')] = {
                c: float(uw_dist[c].q(1.0 - lvl)) for c in cols}
        df = pd.DataFrame(rows).T
        df['Impact'] = df['net'] - df['gross']
        with np.errstate(divide='ignore', invalid='ignore'):
            df['Pct Impact'] = np.where(df['gross'] != 0,
                                        df['net'] / df['gross'] - 1.0, np.nan)
        df.index = pd.MultiIndex.from_tuples(df.index, names=['section', 'item'])
        df.columns = ['Gross', 'Ceded', 'Net', 'Impact', 'Pct Impact']
        return df

    # ------------------------------------------------------------------
    # the audit (pre-plan section 16)
    # ------------------------------------------------------------------
    @property
    def validation_df(self):
        """Numerical audit: exact (EX) vs rebucketed (Est) and the GCN identities.

        Cross-checks (pre-plan section 16): total mass; the premium / loss / UW
        additive identities ``gross = ceded + net`` *in the signed leg sense*
        (here written ``net - gross - ceded`` so 0 is a pass); and EX vs Est means
        for the headline legs. Computed on the joint grid **before** rebucketing
        is the ground truth; the pushforward (Est) should match to grid accuracy.
        """
        d = self.distributions
        s = self.stats_df
        recs = []

        def add(item, ex, est):
            err = abs(ex - est)
            rel = err / abs(ex) if ex else err
            recs.append({'item': item, 'EX': ex, 'Est': est,
                         'abs_err': err, 'rel_err': rel})

        # mass of each pushforward leg
        for name in ('gross_uw', 'ceded_uw', 'net_uw', 'ceded_premium',
                     'net_loss'):
            add(f'mass[{name}]', 1.0, float(d[name].p.sum()))
        # EX vs Est means
        for name in ('ceded_premium', 'ceded_loss', 'net_loss', 'net_uw'):
            add(f'mean[{name}]', float(s.loc[name, 'mean']), d[name].mean())
        # signed additive identities: gross + ceded = net (means)
        gp, cp, npm = (s.loc['gross_premium', 'mean'],
                       s.loc['ceded_premium', 'mean'],
                       s.loc['net_premium', 'mean'])
        add('premium: P_G - (D+h) - net_prem', float(gp - cp), float(npm))
        gl, cl, nl = (s.loc['gross_loss', 'mean'], s.loc['ceded_loss', 'mean'],
                      s.loc['net_loss', 'mean'])
        add('loss: gross - ceded - net', float(gl - cl), float(nl))
        # UW: gross + ceded(col) = net, with ceded col = A-(D+h)
        guw = self.gross_premium - gl
        cuw = s.loc['ceded_uw', 'mean']
        nuw = s.loc['net_uw', 'mean']
        add('uw: gross + ceded - net', float(guw + cuw), float(nuw))
        df = pd.DataFrame(recs).set_index('item')
        return df

    # ------------------------------------------------------------------
    # capital exhibit: return-period table of net underwriting loss
    # ------------------------------------------------------------------
    def tail_df(self, periods=None):
        """Return-period table of **net underwriting loss** (the capital exhibit).

        For each return period ``T`` the net underwriting result at the adverse
        ``1 - 1/T`` tail, plus gross for comparison and the cession benefit
        (pre-plan section 18.4 / 13.7). The killer capital number: how bad the net
        year is at the 1-in-``T`` level once reinstatement premium is paid.

        Parameters
        ----------
        periods : sequence of float, optional
            Return periods. Default ``(10, 20, 50, 100, 200, 250, 1000)``.

        Returns
        -------
        pandas.DataFrame
            Indexed by return period; columns ``gross_uw`` / ``net_uw`` /
            ``benefit`` (net - gross).
        """
        if periods is None:
            periods = (10, 20, 50, 100, 200, 250, 1000)
        d = self.distributions
        recs = []
        for T in periods:
            lvl = 1.0 - 1.0 / T
            guw = float(d['gross_uw'].q(1.0 - lvl))
            nuw = float(d['net_uw'].q(1.0 - lvl))
            recs.append({'return_period': T, 'gross_uw': guw, 'net_uw': nuw,
                         'benefit': nuw - guw})
        return pd.DataFrame(recs).set_index('return_period')

    # ------------------------------------------------------------------
    # narratives
    # ------------------------------------------------------------------
    @property
    def reins_description(self):
        """One-line treaty summary: layer, ``m``, rol, deposit, capacity."""
        t = self.terms
        return (f'{t.limit:g} xs layer, {t.n_reinstatements} reinstatement(s) '
                f'@ rates {t.rates}, deposit {t.deposit:g} '
                f'(rol {t.rol:.1%}); recovery capacity {t.total_recovery_capacity:g}, '
                f'max reinstatement premium {t.maximum_reinstatement_premium:g}.')

    @property
    def reins_explanation(self):
        """Multi-line treaty narrative."""
        t = self.terms
        return (
            f'Occurrence layer limit {t.limit:g} with {t.n_reinstatements} paid '
            f'reinstatement(s).\n'
            f'Base deposit premium {t.deposit:g}, rate on line {t.rol:.2%}.\n'
            f'Reinstatement price multipliers {t.rates}.\n'
            f'Total annual recovery capacity {t.total_recovery_capacity:g} '
            f'(= {t.n_reinstatements + 1} x {t.limit:g}).\n'
            f'Maximum reinstatement premium {t.maximum_reinstatement_premium:g} '
            f'(charged on the first {t.reinstatement_capacity:g} of recovery).\n'
            f'Gross premium {self.gross_premium:g}; ceded premium is stochastic '
            f'(deposit + reinstatement premium h(R)).')

    def __repr__(self):
        return (f'ReinstatementAnalysis(P_G={self.gross_premium:g}, '
                f'{self.terms!r})')
