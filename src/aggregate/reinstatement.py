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

from .contract_terms import ContractTerms

__all__ = ['ReinstatementTerms', 'ReinstatementAnalysis']

#: Adverse-tail levels for the headline ``summary_df`` (pre-plan section 13).
SUMMARY_PERCENTILES = (0.90, 0.95, 0.99, 0.995, 0.996, 0.999)


@dataclass(frozen=True)
class ReinstatementTerms(ContractTerms):
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

    Reinstatement is the one **two-map** :class:`~aggregate.contract_terms.ContractTerms`
    feature: :meth:`reinstatement_premium` is the premium decorator exposed as the
    base :meth:`phi` (filling the ceded-premium leg), while :meth:`recovery` is the
    annual-cap loss transform consumed separately by :class:`ReinstatementAnalysis`.
    """

    #: ``ContractTerms`` metadata: ``h(R)`` fills the ceded-premium leg, reading
    #: the unlimited annual occurrence recovery ``R`` (appendix sections 1, 4).
    target_leg = 'ceded_premium'
    loss_basis = 'occurrence_recovery'

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
        nonnegative and nondecreasing on a sample of the recovery range.

        Delegates to the shared :meth:`ContractTerms._check_vectorized` probe.
        """
        self._check_vectorized(
            self.premium_function, 0.0, self.total_recovery_capacity,
            where='ReinstatementTerms.from_callable: premium_function',
            nonnegative=True, nondecreasing=True)

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

    def phi(self, R):
        """``ContractTerms`` leg map: the reinstatement premium ``h(R)``.

        The premium decorator filling the ceded-premium leg (the annual-cap loss
        transform :meth:`recovery` is the feature's second map, handled separately
        by :class:`ReinstatementAnalysis`).
        """
        return self.reinstatement_premium(R)

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
                 percentiles=SUMMARY_PERCENTILES, joint_aggregate=None,
                 agg_recovery=None, agg_ceded_premium=0.0,
                 gross_expense=0.0, occ_commission=0.0, agg_commission=0.0):
        self.source = source
        self.terms = terms
        self.gross_premium = float(gross_premium)
        self.percentiles = tuple(percentiles)
        #: optional subsequent aggregate cover (decision 3): a vectorized ceder
        #: ``g(net-of-occurrence loss) -> aggregate recovery`` applied to
        #: ``L - A(R)``, hence a deterministic pushforward of the *same* joint.
        #: ``None`` => no aggregate tier (the 3-column gross/ceded/net waterfall).
        self.agg_recovery = agg_recovery
        #: deterministic aggregate-cover ceded premium (the agg layer's
        #: ``deposit | rol | rate``); ``0`` when unknown (programmatic path).
        self.agg_ceded_premium = float(agg_ceded_premium)
        #: deterministic Phase-1 expense / commission economics threaded from the
        #: ``PnL`` (gross expense ``E_G``; ceding commissions ``c_occ`` on the
        #: reinstated layer's deposit and ``c_agg`` on the agg cover). All ``0``
        #: on the programmatic path (a plain aggregate has no premium / expense).
        #: The reinstatement premium ``h(R)`` is non-commissionable, so the
        #: commissions stay deterministic (on the base premiums). The
        #: *stochastic* commission features (slide / profit commission) are
        #: Phase 3.
        self.gross_expense = float(gross_expense)
        self.occ_commission = float(occ_commission)
        self.agg_commission = float(agg_commission)
        #: the ``BivariateAggregate`` holder (when built via
        #: :meth:`Aggregate.reinstatement_analysis`); carries the joint-grid
        #: sizing audit reused by :attr:`bs_window_df` / :attr:`bs_description`.
        self.joint_aggregate = joint_aggregate
        #: matplotlib figure handle set by :meth:`plot`.
        self.figure = None
        # axis 0 is gross loss L, axis 1 is unlimited ceded recovery R
        names = source.axis_names
        if names not in (('gross_loss', 'unlimited_ceded_loss'),
                         ('gross (G)', 'ceded (C)')) and 'gross' not in str(names[0]):
            # not fatal -- occ_bivariate(('gross','ceded')) labels axis0 'gross'
            pass
        #: cached leg pushforwards / exact moments over the (L, R) joint source.
        #: Each accounting leg is pushed forward (rebucketed, for the audit) and
        #: has its exact moments taken straight off the joint; the fixed gross
        #: premium rides as an exact one-point distribution.
        self._dists = None
        self._stats = None

    # ------------------------------------------------------------------
    # the named legs (deterministic pushforwards of the joint)
    # ------------------------------------------------------------------
    def _leg_functions(self):
        """Return ``{name: (function, is_loss_value)}`` for every accounting leg.

        Each ``function(l, r)`` is vectorized on the broadcast axes; ``l`` is
        gross loss, ``r`` unlimited ceded recovery (pre-plan section 6). The
        eleven occurrence legs are always present; a subsequent aggregate cover
        (decision 3) adds five more, all pushforwards of the **same** joint via
        the net-of-occurrence loss ``L - A(R)``.
        """
        P_G = self.gross_premium
        D = self.terms.deposit
        A = self.terms.recovery
        h = self.terms.reinstatement_premium
        legs = {
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
        if self.agg_recovery is not None:
            # The agg cover attaches on the net-of-occurrence loss L - A(R)
            # (occurrence inures to aggregate). ``g`` is its ceder; the agg
            # recovery g(L - A(R)) and the twice-net loss are deterministic
            # pushforwards of the same joint. Premium pc_agg is deterministic.
            # The ceder input is clipped at 0: off-support joint cells (r > l)
            # carry ~no mass but still evaluate, and the piecewise-linear ceder
            # is only defined on [0, inf).
            g0 = self.agg_recovery
            pc = self.agg_ceded_premium

            def g_rec(l, r):
                return g0(np.maximum(l - A(r), 0.0))

            legs.update({
                'ceded_agg_loss':  (lambda l, r: g_rec(l, r), True),
                'net_agg_loss':    (lambda l, r: (l - A(r)) - g_rec(l, r), True),
                'net_agg_premium': (lambda l, r: P_G - D - h(r) - pc, True),
                # ceded-agg column UW = recovery - agg ceded premium
                'ceded_agg_uw':    (lambda l, r: g_rec(l, r) - pc, False),
                # net-of-everything UW
                'net_agg_uw':      (lambda l, r: P_G - D - h(r) - pc
                                    - (l - A(r)) + g_rec(l, r), False),
                # total cession across both tiers (the headline Ceded column)
                'total_ceded_premium': (lambda l, r: D + h(r) + pc, True),
                'total_ceded_loss':    (lambda l, r: A(r) + g_rec(l, r), True),
                'total_ceded_uw':      (lambda l, r: (A(r) + g_rec(l, r))
                                        - (D + h(r) + pc), False),
            })
        return legs

    # ------------------------------------------------------------------
    # the leg evaluation: pushforwards + exact moments over the (L, R) joint.
    # INTERNAL -- the engine behind the kept domain extras (validation_df /
    # tail_df / plot / info); the human-facing exhibits are the returned
    # PnL's group ledger ([Builders-Variable-Features] demotion).
    # ------------------------------------------------------------------
    @property
    def _distributions(self):
        """Dict of the named leg :class:`GridDistribution` objects (internal).

        Each accounting leg is **pushed forward** (scattered, rebucketed onto a
        regular grid -- the form :attr:`validation_df` audits against the exact
        moments) over the ``(L, R)`` joint; the fixed ``gross_premium`` is a
        degenerate one-point distribution so the internal API stays uniform.
        """
        if self._dists is None:
            from ._grid_distribution import GridDistribution
            d = {name: self.source.pushforward(fn, name=name, is_loss_value=isv)
                 for name, (fn, isv) in self._leg_functions().items()}
            d['gross_premium'] = GridDistribution(
                np.array([self.gross_premium]), np.array([1.0]),
                name='gross_premium', is_loss_value=True)
            self._dists = d
        return self._dists

    # ------------------------------------------------------------------
    # exact moment store (the audit's EX basis; internal)
    # ------------------------------------------------------------------
    @property
    def _stats_df(self):
        """Exact per-leg moments on the source joint grid (internal).

        Each leg's exact ``sum p_ij f(l_i, r_j)^k`` straight off the joint
        (:meth:`~aggregate.bivariate.BivariateDistribution.transformed_moments`),
        not the rebucketed pushforward -- the ground truth
        :attr:`validation_df` audits against.
        """
        if self._stats is None:
            rows = {name: self.source.transformed_moments(fn)
                    for name, (fn, _isv) in self._leg_functions().items()}
            df = pd.DataFrame(rows).T
            df.loc['gross_premium'] = {
                'mass': 1.0, 'mean': self.gross_premium, 'var': 0.0, 'sd': 0.0,
                'cv': 0.0, 'skew': np.nan}
            self._stats = df
        return self._stats

    def _exact(self, name):
        """``(mean, sd, skew)`` of a leg from the exact :attr:`_stats_df`."""
        row = self._stats_df.loc[name]
        sk = row.get('skew', np.nan)
        return (float(row['mean']), float(row['sd']),
                float(sk) if pd.notna(sk) else 0.0)

    @property
    def _final_net_uw(self):
        """Leg name of the net-of-everything underwriting result.

        ``net_agg_uw`` when a subsequent aggregate cover is present (decision 3),
        else the occurrence-net ``net_uw``.
        """
        return 'net_agg_uw' if self.agg_recovery is not None else 'net_uw'

    @property
    def _exp_mag(self):
        """Per-perspective deterministic expense / commission magnitude.

        The per-side expense split: the gross expense ``E_G`` books on
        the gross column; each cession credits its commission, so net expense is
        ``E_G - c_occ - c_agg``. The reinstatement premium ``h(R)`` is
        non-commissionable, so these stay deterministic (Phase 3 makes the
        commission leg stochastic via slide / profit commission). All zero unless
        a DecL ``pnl`` threads expenses / ``cede``.
        """
        e, co, ca = self.gross_expense, self.occ_commission, self.agg_commission
        return {'gross': e, 'ceded_occ': co, 'net_occ': e - co,
                'ceded_agg': ca, 'net_agg': e - co - ca,
                'total_ceded': co + ca}

    @property
    def _net_uw_expense_shift(self):
        """Signed expense added to the final-net pure UW (``-`` net expense cost)."""
        return -self._exp_mag['net_agg' if self.agg_recovery is not None
                              else 'net_occ']

    def _uw_with_expense(self, uw_leg, perspective, ceded):
        """Net-expense-shifted copy of a pure UW leg (a deterministic translation).

        The leg distributions stay *pure underwriting* (premium + loss); the
        deterministic expense / commission is a constant shift applied here for
        the exhibits -- ``+`` a commission credit on a cession column, ``-`` an
        expense cost on a retained column.
        """
        from ._grid_distribution import GridDistribution
        gd = self._distributions[uw_leg]
        shift = self._exp_mag[perspective] * (1.0 if ceded else -1.0)
        if shift == 0.0:
            return gd
        return GridDistribution(gd.x + shift, gd.p, bs=gd.bs, name=gd.name,
                                is_loss_value=gd.is_loss_value)

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
        d = self._distributions
        s = self._stats_df
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
        # aggregate-cover tier (decision 3): the waterfall identity
        # net_occ + ceded_agg = net_agg holds for the means.
        if self.agg_recovery is not None:
            nocc = s.loc['net_uw', 'mean']
            cagg = s.loc['ceded_agg_uw', 'mean']
            nagg = s.loc['net_agg_uw', 'mean']
            add('agg uw: net_occ + ceded_agg - net_agg',
                float(nocc + cagg), float(nagg))
            # total cession ties to the final net: gross + total_ceded = net_agg
            tcuw = s.loc['total_ceded_uw', 'mean']
            add('total uw: gross + total_ceded - net_agg',
                float(guw + tcuw), float(nagg))
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
            ``benefit`` (net - gross). ``net_uw`` is net of everything -- it
            reads the net-of-aggregate result when a subsequent aggregate cover
            is present (decision 3).
        """
        if periods is None:
            periods = (10, 20, 50, 100, 200, 250, 1000)
        d = self._distributions
        net_leg = d[self._final_net_uw]
        # net of deterministic expense / commission (constant shift), so the
        # capital number is the true bad net year (premium - loss - expense).
        g_shift = -self._exp_mag['gross']
        n_shift = -self._exp_mag['net_agg' if self.agg_recovery is not None
                                 else 'net_occ']
        recs = []
        for T in periods:
            lvl = 1.0 - 1.0 / T
            guw = float(d['gross_uw'].q(1.0 - lvl)) + g_shift
            nuw = float(net_leg.q(1.0 - lvl)) + n_shift
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

    @property
    def validation_explanation(self):
        """Prose verdict on the numerical audit (mirrors ``Aggregate``).

        Summarises :attr:`validation_df`: the largest absolute error across the
        mass / EX-vs-Est / additive-identity checks against the 1e-6 grid
        tolerance, and lists any item that fails.
        """
        v = self.validation_df
        worst = float(v['abs_err'].max())
        passes = worst < 1e-6
        lines = [
            f'Reinstatement audit for the {self.terms.limit:g} xs layer '
            f'(P_G = {self.gross_premium:g}).',
            f'Largest absolute error across {len(v)} checks: {worst:.2e} '
            f"-- {'PASS' if passes else 'CHECK'} at the 1e-6 grid tolerance.",
            'Checks: total mass of each pushforward leg; EX (exact joint-grid '
            'moment) vs Est (rebucketed pushforward) means; and the signed '
            'additive identities gross + ceded = net.']
        if not passes:
            bad = list(v[v['abs_err'] >= 1e-6].index)
            lines.append('Above tolerance: ' + ', '.join(bad))
        return '\n'.join(lines)

    def _validation_passes(self):
        """``True`` when every audit check is within the 1e-6 grid tolerance."""
        return float(self.validation_df['abs_err'].max()) < 1e-6

    # ------------------------------------------------------------------
    # joint-grid sizing audit (reused from the BivariateAggregate holder)
    # ------------------------------------------------------------------
    @property
    def bs_window_df(self):
        """Per-axis joint-grid sizing summary (reused from the joint holder).

        Delegates to :attr:`BivariateAggregate.bs_window_df` when the analysis
        was built via :meth:`Aggregate.reinstatement_analysis`; ``None`` for a
        directly-constructed analysis with no holder.
        """
        if self.joint_aggregate is None:
            return None
        return self.joint_aggregate.bs_window_df

    @property
    def bs_description(self) -> str:
        """One-line summary of the joint ``(L, R)`` grid (reused from the holder)."""
        if self.joint_aggregate is not None:
            return self.joint_aggregate.bs_description
        return (f'reinstatement joint grid: L bs={self.source.bs_ceded:g}, '
                f'R bs={self.source.bs_net:g}')

    @property
    def bs_explanation(self) -> str:
        """Verbose note on the joint grid plus the audit tolerance."""
        return (f'{self.bs_description}\nThe analysis pushes the signed '
                'accounting legs forward over this one joint FFT2; every leg '
                'distribution is a deterministic rebucketing of it (no second '
                'convolution). The rebucketing error is audited in '
                'validation_df against a 1e-6 tolerance.')

    # ------------------------------------------------------------------
    # structured summary / display
    # ------------------------------------------------------------------
    def info(self):
        """Fixed-layout text summary (terms, premium, headline means, audit)."""
        from .constants import info_row
        t, s = self.terms, self._stats_df
        rows = [
            ('reinstatement analysis', f'{t.limit:g} xs layer'),
            ('gross premium', f'{self.gross_premium:,.6g}'),
            ('deposit (base premium)', f'{t.deposit:,.6g}'),
            ('rate on line', f'{t.rol:.4%}'),
            ('reinstatements (m)', t.n_reinstatements),
            ('rates', str(t.rates)),
            ('recovery capacity Y', f'{t.total_recovery_capacity:,.6g}'),
            ('max reinstatement prem', f'{t.maximum_reinstatement_premium:,.6g}'),
            ('E[gross UW]', f'{s.loc["gross_uw", "mean"] - self._exp_mag["gross"]:,.6g}'),
            ('E[ceded UW]', f'{s.loc["ceded_uw", "mean"] + self._exp_mag["ceded_occ"]:,.6g}'),
            ('E[net UW]', f'{s.loc[self._final_net_uw, "mean"] + self._net_uw_expense_shift:,.6g}'),
            ('E[ceded premium]', f'{s.loc["ceded_premium", "mean"]:,.6g}'),
            ('CV[ceded premium]', f'{s.loc["ceded_premium", "cv"]:.4f}'),
            ('validation', 'ok' if self._validation_passes() else 'CHECK'),
        ]
        return '\n'.join(info_row(label, value) for label, value in rows)

    def _text_info_blob(self):
        """Short text intro for :func:`aggregate.qd`."""
        t = self.terms
        return (f'Reinstatement analysis: {t.limit:g} xs layer, '
                f'{t.n_reinstatements} reinstatement(s) @ {t.rates}, '
                f'gross premium {self.gross_premium:g}, deposit {t.deposit:g}.')

    def _repr_html_(self):
        """Jupyter display: treaty intro and the return-period tail table.

        The Gross/Ceded/Net exhibit is the owning PnL's ledger
        ([Builders-Variable-Features] demotion); the analysis displays only
        its domain extras.
        """
        intro = self._text_info_blob()
        parts = [f'<h4>Reinstatement analysis &mdash; {self.terms.limit:g} xs '
                 f'layer</h4>', f'<p>{intro}</p>', self.tail_df().to_html()]
        if not self._validation_passes():
            parts.append('<p><b>VALIDATION FAILS:</b> '
                         f'{self.validation_explanation}</p>')
        return '\n'.join(parts)

    def plot(self, axd=None, **kwargs):
        """One mosaic of the reinstatement story (pre-plan section 18).

        Four panels: **A** the joint ``(L, R)`` log density with the recovery
        breakpoints / cap overlaid; **B** the deterministic maps ``A(R)``,
        ``h(R)``, ``D + h(R)`` and the ceded underwriting ``D + h - A``; **C**
        the gross vs net underwriting-result return-period (Lee) curves; and
        **D** the cession impact ``q_p(net) - q_p(gross)``.

        Parameters
        ----------
        axd : dict of str to Axes, optional
            Mosaic with keys ``'A'``, ``'B'``, ``'C'``, ``'D'``; a new figure is
            created if omitted and stored on :attr:`figure`.
        **kwargs
            Passed to the canvas creator (e.g. ``figsize``).

        Returns
        -------
        matplotlib.figure.Figure
        """
        from .plots._aggregate import plot_reinstatement
        return plot_reinstatement(self, axd=axd, **kwargs)

    def __repr__(self):
        return (f'ReinstatementAnalysis(P_G={self.gross_premium:g}, '
                f'{self.terms!r})')
