r"""Variable-rating analysis: one stochastic leg from a :class:`ContractTerms` feature.

The feature-agnostic engine behind retro / swing / slide / profit-commission /
corridor (``dev/plan-variable-rating.md``). Where Phase 2's
:class:`aggregate.reinstatement.ReinstatementAnalysis` is bespoke to occurrence
reinstatement premiums, this is generic over **any**
:class:`~aggregate.contract_terms.ContractTerms`: the feature's vectorized
``phi`` is pushed forward over the loss distribution to make exactly **one** P&L
leg stochastic (decision 0: one variable feature per program, no stacking), and
the Gross / Ceded / Net waterfall is a group ledger of :class:`~aggregate.PnL`
legs over the degenerate 1-D source.

Every accounting leg is written as a function of ``(l, a)`` -- gross loss ``l``
and ceded loss ``a`` -- so the **same** code serves both bases (appendix
section 2):

* **aggregate basis (1-D)** -- ``a = ceder(l)`` is deterministic, so each leg is a
  1-D pushforward of the gross density (:func:`~aggregate.bivariate.pushforward_1d`).
  This module implements that path.
* **occurrence basis (2-D)** -- ``a`` is a genuine second axis (the recovery), so
  the legs push over the ``(L, R)`` joint; that path reuses
  :meth:`~aggregate.bivariate.BivariateDistribution.pushforward` (follow-up).

Submodule access only (no top-level re-export)::

    from aggregate.variable_rating import VariableRatingAnalysis
"""

import numpy as np
import pandas as pd

from .contract_terms import ContractTerms

__all__ = ['VariableRatingAnalysis']

#: Adverse-tail levels for the headline ``summary_df`` (mirrors reinstatement).
SUMMARY_PERCENTILES = (0.90, 0.95, 0.99, 0.995, 0.996, 0.999)


def _layer_ceder(share, limit, attachment):
    """Vectorized occurrence/aggregate ceder ``a(l) = share * clip(l - attach, 0, limit)``."""
    share = float(share)
    limit = float(limit)
    attachment = float(attachment)

    def ceder(l):
        l = np.asarray(l, dtype=float)
        return share * np.clip(l - attachment, 0.0, limit)

    return ceder


class VariableRatingAnalysis:
    r"""Gross / Ceded / Net analysis with one :class:`ContractTerms`-driven leg.

    Pairs the gross loss distribution of an account / layer with a single
    variable-rating feature and pushes the signed accounting legs forward into the
    Gross / Ceded / Net underwriting distributions and the GCN exhibit. The feature
    selects which leg is stochastic via its
    :attr:`~aggregate.contract_terms.ContractTerms.target_leg`:

    ===============  =========================  ==========================================
    ``target_leg``   feature(s)                 stochastic leg
    ===============  =========================  ==========================================
    ``gross_premium``  retro                    gross premium ``= phi(net account loss)``
    ``ceded_premium``  swing                    ceded premium ``= phi(ceded loss)``
    ``expense``        slide, profit commission  ceding commission ``= phi(ceded LR) * P_C``
    ``ceded_loss``     corridor                 ceded loss ``= phi(ceded LR) * P_C``
    ===============  =========================  ==========================================

    All other legs stay deterministic (scalar premiums / expense, the layer ceder).
    This is the **1-D aggregate-basis** implementation: ceded loss ``a = ceder(l)``
    is a deterministic function of gross loss, so every leg is a 1-D pushforward of
    the gross density.

    Parameters
    ----------
    gross_grid : ndarray
        The gross aggregate loss index (``Aggregate.density_df.index``).
    gross_density : ndarray
        Gross aggregate probability mass aligned with ``gross_grid``.
    terms : ContractTerms
        The single variable-rating feature.
    gross_premium : float
        Fixed gross premium ``P_G`` (the base, before any retro variation).
    ceded_premium : float, default 0.0
        Fixed ceded premium ``P_C`` (resolved ``deposit | rol | rate``); the base
        before any swing variation, and the loss-ratio denominator for slide / pc /
        corridor.
    layer : tuple of float, optional
        ``(share, limit, attachment)`` of the single priced layer; ``None`` for an
        account-level feature with no reinsurance (e.g. retro alone).
    ceder : callable, optional
        Vectorized ``a = ceder(l)`` overriding ``layer``.
    gross_expense : float, default 0.0
        Deterministic gross expense ``E_G`` (books on the gross column).
    commission : float, default 0.0
        Deterministic ceding commission (the fixed ``cede`` case); ignored when the
        feature itself fills the expense leg (slide / pc).
    percentiles : tuple of float, optional
        Adverse-tail levels for :attr:`summary_df`. Default :data:`SUMMARY_PERCENTILES`.

    Notes
    -----
    Payoff signs follow the reinstatement convention (premium received ``+``, paid
    ``-``; loss a negative obligation; recovery / commission ``+``). The per-``l``
    accounting, with ``A = ceded_loss(l)``, ``P_C = ceded_premium(l)``,
    ``C = commission(l)``, ``E_G`` the gross expense:

    ======  ===================  ================  ===================
    view    premium              loss              expense
    ======  ===================  ================  ===================
    gross   ``P_G``              ``-l``            ``-E_G``
    ceded   ``-P_C``             ``+A``            ``+C``
    net     ``P_G - P_C``        ``-(l - A)``      ``-(E_G - C)``
    ======  ===================  ================  ===================

    Means add across the split and down to underwriting; the stochastic leg carries
    the volatility (nonzero CV).
    """

    def __init__(self, gross_grid, gross_density, terms, *, gross_premium,
                 ceded_premium=0.0, layer=None, ceder=None, gross_expense=0.0,
                 commission=0.0, percentiles=SUMMARY_PERCENTILES):
        if not isinstance(terms, ContractTerms):
            raise TypeError(
                f'VariableRatingAnalysis: terms must be a ContractTerms, got '
                f'{type(terms).__name__}.')
        self.grid = np.asarray(gross_grid, dtype=float)
        self.density = np.asarray(gross_density, dtype=float)
        self.terms = terms
        self.gross_premium = float(gross_premium)
        self.ceded_premium = float(ceded_premium)
        self.gross_expense = float(gross_expense)
        self.commission = float(commission)
        self.percentiles = tuple(percentiles)
        if ceder is not None:
            self.ceder = ceder
        elif layer is not None:
            self.ceder = _layer_ceder(*layer)
        else:
            self.ceder = lambda l: np.zeros_like(np.asarray(l, dtype=float))
        self.layer = layer
        self._validate_feature()
        #: cached leg pushforwards / exact moments over the degenerate 1-D source
        #: (all mass on the curve ``a = ceder(l)``; each leg ``psi(l) = f(l,
        #: ceder(l))`` is a 1-D pushforward of the gross density).
        self._dists = None
        self._stats = None

    def _validate_feature(self):
        """Check the single feature has the economics its leg needs."""
        tl = self.terms.target_leg
        if tl not in ('gross_premium', 'ceded_premium', 'expense', 'ceded_loss'):
            raise ValueError(
                f'VariableRatingAnalysis: unsupported target_leg '
                f'{tl!r} on {type(self.terms).__name__}.')
        if tl in ('expense', 'ceded_loss') and self.ceded_premium <= 0:
            raise ValueError(
                f'{type(self.terms).__name__} reads the ceded loss ratio, which '
                f'needs a positive ceded premium denominator; got '
                f'ceded_premium={self.ceded_premium}.')

    # ------------------------------------------------------------------
    # the four feature component maps f(l, a) (one is feature-varied)
    # ------------------------------------------------------------------
    def _components(self):
        """The four accounting component maps ``f(l, a)`` -- gross / ceded premium,
        ceded loss, commission -- with exactly one varied by the feature's ``phi``.

        ``a = ceder(l)`` is the base ceded loss. ``target_leg`` steers which map
        reads ``terms.phi``: ``gross_premium`` (retro, on net account loss
        ``l - a``); ``ceded_premium`` (swing, on ceded loss ``a``); ``expense``
        (slide / profit commission, on ceded LR ``a / P_C``); ``ceded_loss``
        (corridor, on ceded LR). These decompose every accounting leg the
        group ledger and the exhibits read.
        """
        terms = self.terms
        tl = terms.target_leg
        P_G = self.gross_premium
        P_C = self.ceded_premium
        C = self.commission

        if tl == 'ceded_loss':                              # corridor
            def ceded_loss(l, a):
                return terms.phi(a / P_C) * P_C             # phi reads ceded LR
        else:
            def ceded_loss(l, a):
                return a

        if tl == 'gross_premium':                           # retro
            def gross_premium(l, a):
                return terms.phi(np.asarray(l, dtype=float) - a)
        else:
            def gross_premium(l, a):
                return np.full_like(np.asarray(l, dtype=float), P_G)

        if tl == 'ceded_premium':                           # swing
            def ceded_premium(l, a):
                return terms.phi(a)
        else:
            def ceded_premium(l, a):
                return np.full_like(np.asarray(l, dtype=float), P_C)

        if tl == 'expense':                                 # slide, profit comm.
            def commission(l, a):
                return terms.phi(a / P_C) * P_C
        else:
            def commission(l, a):
                return np.full_like(np.asarray(l, dtype=float), C)

        return {'gross_premium': gross_premium, 'ceded_premium': ceded_premium,
                'ceded_loss': ceded_loss, 'commission': commission}

    # ------------------------------------------------------------------
    # the named legs (maps f(l, a) over the degenerate (l, a=ceder(l)) source)
    # ------------------------------------------------------------------
    def _leg_functions(self):
        """Return ``{name: (map, is_loss_value)}`` for every accounting leg.

        Each ``map(l, a)`` is vectorized on the gross-loss grid ``l`` and the
        **base ceded loss** ``a = ceder(l)`` -- the degenerate aggregate basis
        puts all mass on the curve ``a = ceder(l)``, so each leg ``psi(l) = f(l,
        ceder(l))`` is a 1-D pushforward. The single feature makes exactly one leg
        loss-sensitive; the rest read scalars or the base cession ``a``. Legs
        carry the universal ``f(l, a)`` signature, so the occurrence (2-D) path is
        the same maps over a full joint.
        """
        E_G = self.gross_expense
        comp = self._components()
        ceded_loss = comp['ceded_loss']
        gross_premium = comp['gross_premium']
        ceded_premium = comp['ceded_premium']
        commission = comp['commission']

        def net_premium(l, a):
            return gross_premium(l, a) - ceded_premium(l, a)

        def net_loss(l, a):
            return np.asarray(l, dtype=float) - ceded_loss(l, a)

        def net_expense(l, a):                            # gross expense net of commission
            return E_G - commission(l, a)

        # underwriting legs INCLUDE the (possibly stochastic) expense / commission
        # so the percentile rows are correct; signs per the docstring table. The
        # ceded leg already carries cedant-perspective signs (recovery +, ceded
        # premium -), so net underwriting is gross_uw + ceded_uw (means add).
        def gross_uw(l, a):
            return gross_premium(l, a) - np.asarray(l, dtype=float) - E_G

        def ceded_uw(l, a):
            return ceded_loss(l, a) - ceded_premium(l, a) + commission(l, a)

        def net_uw(l, a):
            return gross_uw(l, a) + ceded_uw(l, a)

        return {
            'gross_loss':     (lambda l, a: np.asarray(l, dtype=float), True),
            'ceded_loss':     (ceded_loss, True),
            'net_loss':       (net_loss, True),
            'gross_premium':  (gross_premium, True),
            'ceded_premium':  (ceded_premium, True),
            'net_premium':    (net_premium, True),
            'gross_expense':  (lambda l, a: np.full_like(np.asarray(l, float), E_G), True),
            'commission':     (commission, True),
            'net_expense':    (net_expense, True),
            'gross_uw':       (gross_uw, False),
            'ceded_uw':       (ceded_uw, False),
            'net_uw':         (net_uw, False),
        }

    # ------------------------------------------------------------------
    # the leg evaluation: 1-D pushforwards + exact moments over a = ceder(l)
    # ------------------------------------------------------------------
    def _psi(self, fn):
        """Collapse a leg map ``f(l, a)`` onto the degenerate curve ``a=ceder(l)``."""
        ceder = self.ceder
        return lambda l: fn(l, ceder(l))

    @property
    def _distributions(self):
        """Dict of the named leg :class:`GridDistribution` objects (internal).

        Each leg ``psi(l) = f(l, ceder(l))`` is a **1-D pushforward** of the gross
        density (:func:`~aggregate.bivariate.pushforward_1d`), the degenerate
        aggregate-basis path. The internal engine behind :meth:`tail_df`; the
        human-facing exhibits are the owning PnL's group ledger.
        """
        if self._dists is None:
            from .bivariate import pushforward_1d
            self._dists = {
                name: pushforward_1d(self.grid, self.density, self._psi(fn),
                                     name=name, is_loss_value=isv)
                for name, (fn, isv) in self._leg_functions().items()}
        return self._dists

    @property
    def _stats_df(self):
        """Exact per-leg moments on the gross grid (internal).

        Each leg's exact ``sum p_i psi(l_i)^k`` straight off the gross density
        (no rebucketing) -- the ground-truth moment store.
        """
        if self._stats is None:
            rows = {name: self._moments(fn)
                    for name, (fn, _isv) in self._leg_functions().items()}
            self._stats = pd.DataFrame(rows).T
        return self._stats

    def _moments(self, fn):
        """Exact ``mass/mean/var/sd/cv/skew`` of ``psi(l)=f(l,ceder(l))``."""
        l = self.grid
        p = self.density
        v = np.broadcast_to(np.asarray(self._psi(fn)(l), dtype=float), l.shape)
        mass = float(p.sum())
        m1 = float((p * v).sum())
        m2 = float((p * v * v).sum())
        var = max(m2 - m1 * m1, 0.0)
        sd = float(np.sqrt(var))
        cv = sd / m1 if m1 else 0.0
        skew = float((p * (v - m1) ** 3).sum()) / sd ** 3 if sd > 0 else np.nan
        return pd.Series({'mass': mass, 'mean': m1, 'var': var, 'sd': sd,
                          'cv': cv, 'skew': skew})

    def _exact(self, name):
        """``(mean, sd, skew)`` of a leg from the exact :attr:`_stats_df`."""
        row = self._stats_df.loc[name]
        sk = row.get('skew', np.nan)
        return (float(row['mean']), float(row['sd']),
                float(sk) if pd.notna(sk) else 0.0)

    def tail_df(self, periods=None):
        """Return-period table of **net underwriting result** (the capital exhibit).

        For each return period ``T`` the net underwriting result at the adverse
        ``1 - 1/T`` tail, plus gross for comparison and the feature benefit
        (net - gross). The underwriting legs already carry the expense /
        commission.

        Parameters
        ----------
        periods : sequence of float, optional
            Return periods. Default ``(10, 20, 50, 100, 200, 250, 1000)``.

        Returns
        -------
        pandas.DataFrame
            Indexed by return period; columns ``gross_uw`` / ``net_uw`` /
            ``benefit``.
        """
        if periods is None:
            periods = (10, 20, 50, 100, 200, 250, 1000)
        d = self._distributions
        recs = []
        for T in periods:
            lvl = 1.0 / T
            guw = float(d['gross_uw'].q(lvl))
            nuw = float(d['net_uw'].q(lvl))
            recs.append({'return_period': T, 'gross_uw': guw, 'net_uw': nuw,
                         'benefit': nuw - guw})
        return pd.DataFrame(recs).set_index('return_period')
