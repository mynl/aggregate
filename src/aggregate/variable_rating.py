r"""Variable-rating analysis: one stochastic leg from a :class:`ContractTerms` feature.

The feature-agnostic engine behind retro / swing / slide / profit-commission /
corridor (``dev/plan-variable-rating.md``). Where Phase 2's
:class:`aggregate.reinstatement.ReinstatementAnalysis` is bespoke to occurrence
reinstatement premiums, this is generic over **any**
:class:`~aggregate.contract_terms.ContractTerms`: the feature's vectorized
``phi`` is pushed forward over the loss distribution to make exactly **one** P&L
leg stochastic (decision 0: one variable feature per program, no stacking), and
the Gross / Ceded / Net waterfall is assembled through the shared
:func:`aggregate._pnl.gcn_assemble_column`.

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
        #: cached InsuranceView over the leg set + degenerate GraphSource (the
        #: aggregate-basis source: all mass on the curve a = ceder(l)). One
        #: kernel evaluation backs distributions / stats_df / gcn_df.
        self._view = None

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
    # the named legs (maps f(l, a) over the degenerate (l, a=ceder(l)) source)
    # ------------------------------------------------------------------
    def _leg_functions(self):
        """Return ``{name: (map, is_loss_value)}`` for every accounting leg.

        Each ``map(l, a)`` is vectorized on the gross-loss grid ``l`` and the
        **base ceded loss** ``a = ceder(l)`` -- the second coordinate of the
        degenerate :class:`~aggregate.legs.GraphSource` (the aggregate basis puts
        all mass on the curve ``a = kappa(l)``). The single feature makes exactly
        one leg loss-sensitive; the rest read scalars or the base cession ``a``.
        Legs carry the **universal** ``f(X, Y)`` signature of the kernel, so the
        occurrence (2-D) path is the same maps over a full joint.
        """
        terms = self.terms
        tl = terms.target_leg
        P_G = self.gross_premium
        P_C = self.ceded_premium
        E_G = self.gross_expense
        C = self.commission

        # --- ceded loss A(l, a): the base cession a, corridor-modified if so ---
        if tl == 'ceded_loss':                              # corridor
            def ceded_loss(l, a):
                return terms.phi(a / P_C) * P_C             # phi reads ceded LR
        else:
            def ceded_loss(l, a):
                return a

        # --- gross premium: retro varies it on net account loss l - a -----------
        if tl == 'gross_premium':                           # retro
            def gross_premium(l, a):
                return terms.phi(np.asarray(l, dtype=float) - a)
        else:
            def gross_premium(l, a):
                return np.full_like(np.asarray(l, dtype=float), P_G)

        # --- ceded premium: swing varies it on ceded loss a ---------------------
        if tl == 'ceded_premium':                           # swing
            def ceded_premium(l, a):
                return terms.phi(a)
        else:
            def ceded_premium(l, a):
                return np.full_like(np.asarray(l, dtype=float), P_C)

        # --- ceding commission (expense credit): slide / pc vary it on ceded LR -
        if tl == 'expense':                                 # slide, profit comm.
            def commission(l, a):
                return terms.phi(a / P_C) * P_C
        else:
            def commission(l, a):
                return np.full_like(np.asarray(l, dtype=float), C)

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
    # the kernel evaluation: one InsuranceView over the leg set + source
    # ------------------------------------------------------------------
    @property
    def view(self):
        """The backing :class:`~aggregate._insurance_view.InsuranceView` (lazy).

        Wraps the leg set and the degenerate
        :class:`~aggregate.legs.GraphSource` (``kappa = ceder``); every exhibit
        reads its cached :attr:`distributions` / :attr:`stats_df`.
        """
        if self._view is None:
            from .legs import Leg, LegSet, GraphSource
            from ._insurance_view import InsuranceView
            legs = LegSet(Leg(name, fn, is_value)
                          for name, (fn, is_value) in self._leg_functions().items())
            source = GraphSource(self.grid, self.density, kappa=self.ceder)
            self._view = InsuranceView(legs, source)
        return self._view

    @property
    def distributions(self):
        """Dict of the named leg :class:`GridDistribution` objects (via the kernel)."""
        return self.view.distributions

    @property
    def stats_df(self):
        """Exact per-leg moments on the gross grid (mean / sd / cv / skew).

        The canonical moment store: each leg's exact ``sum p_i f(l_i, a_i)^k``
        straight off the gross density (the degenerate source's
        ``transformed_moments``), not the rebucketed pushforward. Means here are
        the ground truth the GCN Mean rows add to.
        """
        return self.view.stats_df

    def _exact(self, name):
        """``(mean, sd, skew)`` of a leg from the exact :attr:`stats_df`."""
        return self.view.exact(name)

    # ------------------------------------------------------------------
    # the reused GCN waterfall (via the shared assembler)
    # ------------------------------------------------------------------
    @property
    def gcn_df(self):
        """Gross / Ceded / Net underwriting waterfall (the reused PnL exhibit).

        Built through the shared :func:`aggregate._pnl.gcn_assemble_column`, so it
        carries the identical sections, signs, CV rows and percentile placement as
        :meth:`PnL.gcn_df`. The feature's leg is stochastic (nonzero CV in its
        section); means add ``gross + ceded = net`` down to underwriting.
        """
        from ._pnl import gcn_assemble_column, GCN_PERCENTILES, PnL, _GCN_RATIO_ROWS
        d = self.distributions

        def column(ceded, prem_leg, loss_leg, exp_leg, uw_leg):
            pm, psd, _ = self._exact(prem_leg)
            lm, lsd, lsk = self._exact(loss_leg)
            em, esd, _ = self._exact(exp_leg)             # nonnegative magnitude
            uw_dist = d[uw_leg]
            pct = {lvl: float(uw_dist.q(lvl)) for lvl in GCN_PERCENTILES}
            return gcn_assemble_column(
                ceded=ceded, prem_mean=pm, prem_sd=psd,
                loss_mean=lm, loss_sd=lsd, loss_skew=lsk,
                exp_mean=em, exp_sd=esd, uw_pctiles=pct)

        # expense magnitude per column (gcn_assemble_column signs it: + on the
        # ceded credit, - elsewhere): gross books E_G; ceded credits commission;
        # net books E_G - commission. The percentile rows read the matching UW leg.
        data = {}
        data['gross'] = column(False, 'gross_premium', 'gross_loss',
                               'gross_expense', 'gross_uw')
        data['ceded'] = column(True, 'ceded_premium', 'ceded_loss',
                               'commission', 'ceded_uw')
        data['net'] = column(False, 'net_premium', 'net_loss',
                             'net_expense', 'net_uw')
        data['impact'] = PnL._gcn_impact(data['net'], data['gross'],
                                         _GCN_RATIO_ROWS)
        df = pd.DataFrame(data)
        df.index = pd.MultiIndex.from_tuples(df.index, names=['section', 'item'])
        return df
