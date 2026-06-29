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
        self._distributions = None
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
    # the named legs (deterministic 1-D pushforwards of the gross density)
    # ------------------------------------------------------------------
    def _leg_functions(self):
        """Return ``{name: (function, is_loss_value)}`` for every accounting leg.

        Each ``function(l)`` is vectorized on the gross-loss grid. The single
        feature makes exactly one leg loss-sensitive; the rest are deterministic
        (scalar premiums / expense, the base layer ceder).
        """
        terms = self.terms
        tl = terms.target_leg
        P_G = self.gross_premium
        P_C = self.ceded_premium
        E_G = self.gross_expense
        ceder = self.ceder

        # --- ceded loss A(l): the base ceder, corridor-modified if applicable ---
        if tl == 'ceded_loss':                              # corridor
            def ceded_loss(l):
                a0 = ceder(l)
                return terms.phi(a0 / P_C) * P_C            # phi reads ceded LR
        else:
            def ceded_loss(l):
                return ceder(l)

        # --- gross premium: retro varies it on net account loss l - A0(l) -------
        if tl == 'gross_premium':                           # retro
            def gross_premium(l):
                net_account = np.asarray(l, dtype=float) - ceder(l)
                return terms.phi(net_account)
        else:
            def gross_premium(l):
                return np.full_like(np.asarray(l, dtype=float), P_G)

        # --- ceded premium: swing varies it on ceded loss -----------------------
        if tl == 'ceded_premium':                           # swing
            def ceded_premium(l):
                return terms.phi(ceder(l))
        else:
            def ceded_premium(l):
                return np.full_like(np.asarray(l, dtype=float), P_C)

        # --- ceding commission (expense credit): slide / pc vary it on ceded LR -
        if tl == 'expense':                                 # slide, profit comm.
            def commission(l):
                a0 = ceder(l)
                return terms.phi(a0 / P_C) * P_C
        else:
            C = self.commission

            def commission(l):
                return np.full_like(np.asarray(l, dtype=float), C)

        def net_premium(l):
            return gross_premium(l) - ceded_premium(l)

        def net_loss(l):
            return np.asarray(l, dtype=float) - ceded_loss(l)

        def net_expense(l):                               # gross expense net of commission
            return E_G - commission(l)

        # underwriting legs INCLUDE the (possibly stochastic) expense / commission
        # so the percentile rows are correct; signs per the docstring table. The
        # ceded leg already carries cedant-perspective signs (recovery +, ceded
        # premium -), so net underwriting is gross_uw + ceded_uw (means add).
        def gross_uw(l):
            return gross_premium(l) - np.asarray(l, dtype=float) - E_G

        def ceded_uw(l):
            return ceded_loss(l) - ceded_premium(l) + commission(l)

        def net_uw(l):
            return gross_uw(l) + ceded_uw(l)

        return {
            'gross_loss':     (lambda l: np.asarray(l, dtype=float), True),
            'ceded_loss':     (ceded_loss, True),
            'net_loss':       (net_loss, True),
            'gross_premium':  (gross_premium, True),
            'ceded_premium':  (ceded_premium, True),
            'net_premium':    (net_premium, True),
            'gross_expense':  (lambda l: np.full_like(np.asarray(l, float), E_G), True),
            'commission':     (commission, True),
            'net_expense':    (net_expense, True),
            'gross_uw':       (gross_uw, False),
            'ceded_uw':       (ceded_uw, False),
            'net_uw':         (net_uw, False),
        }

    @property
    def distributions(self):
        """Dict of the named leg :class:`GridDistribution` objects (lazy)."""
        if self._distributions is None:
            from .bivariate import pushforward_1d
            d = {}
            for name, (fn, is_loss) in self._leg_functions().items():
                d[name] = pushforward_1d(self.grid, self.density, fn,
                                         name=name, is_loss_value=is_loss)
            self._distributions = d
        return self._distributions

    # ------------------------------------------------------------------
    # exact moment store (the EX column; single source of truth)
    # ------------------------------------------------------------------
    @property
    def stats_df(self):
        """Exact per-leg moments on the gross grid (mean / sd / cv / skew).

        The canonical moment store: each leg's exact ``sum p_i f(l_i)^k`` straight
        off the gross density, not the rebucketed pushforward. Means here are the
        ground truth the GCN Mean rows add to.
        """
        if self._stats is None:
            p = self.density
            rows = {}
            for name, (fn, _) in self._leg_functions().items():
                v = np.broadcast_to(np.asarray(fn(self.grid), dtype=float),
                                    self.grid.shape)
                m1 = float(np.sum(p * v))
                m2 = float(np.sum(p * v * v))
                var = max(m2 - m1 * m1, 0.0)
                sd = np.sqrt(var)
                cv = sd / m1 if m1 else 0.0
                if sd > 0:
                    m3 = float(np.sum(p * (v - m1) ** 3))
                    skew = m3 / sd ** 3
                else:
                    skew = np.nan
                rows[name] = {'mass': float(np.sum(p)), 'mean': m1, 'var': var,
                              'sd': sd, 'cv': cv, 'skew': skew}
            self._stats = pd.DataFrame(rows).T
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
