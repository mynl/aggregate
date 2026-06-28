"""First-class profit-and-loss veneer over an :class:`Aggregate`.

A :class:`PnL` is a thin *composition* over a **pure-loss** :class:`Aggregate`
``X`` (the risky leg, the *obligation*) and a *consideration* ``C`` (the amount
changing hands at inception -- a held position). The net P&L is **always
payoff** ("more money is better"), which is what the name means::

    net = C - X    if X is a loss    (more loss   -> less net)
    net = C + X    if X is a payoff  (more payoff -> more net)

so the combine sign is **read from** ``X.value_type`` -- there is no separate
``sign`` knob. Long/short and buy/sell are expressed by *which orientation you
give X* and by the **sign of the consideration** (``+`` received, ``-`` paid).

The risky leg ``X`` is left **untouched**: ``self.agg`` is the honest
obligation, with its own density, moments and plot. The net is a *cheap
deterministic* transform of ``X``'s density (a 1-D relabel onto the net grid,
comonotone with ``X`` -- no convolution, no FFT, no windowing), derived lazily
in :attr:`pnl_df`. This keeps a single source of truth: ``pnl.agg.density_df``
is the obligation, ``pnl.pnl_df`` is the net.

See ``dev/plan-pnl.md``. The signed additive ``summary_df``, ``plot``,
``evaluate`` panel, reinsurance-aware GCN view, and the function-valued
consideration *reporting* land in later stages; constant **and** callable
consideration are both supported in :attr:`pnl_df` here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


#: UW-result percentile levels for the GCN exhibit (payoff convention, so a
#: *low* level is the bad tail). Deliberate regulatory anchors: 1/1000, 1/200
#: (Solvency II), 1/250 (~US RBC / rating agency), 1/100, then the body and the
#: good side. Used verbatim -- no rounding.
GCN_PERCENTILES = (0.001, 0.005, 0.006, 0.01, 0.1, 0.5, 0.9, 0.99)

#: Each waterfall perspective reads its own exact aggregate **marginal** from
#: ``Aggregate.reins_density_df`` (means add across the split; SD / skew /
#: percentiles are per-column marginals and do *not* add -- appendix S2).
_GCN_LOSS_MARGINAL = {
    'gross': 'p_agg_gross',
    'ceded_occ': 'p_agg_ceded_occ',
    'net_occ': 'p_agg_net_occ',
    'ceded_agg': 'p_agg_ceded',
    'net_agg': 'p_agg_net',
}

#: The cession perspectives (premium paid, recovery / commission received).
_GCN_CEDED = frozenset({'ceded_occ', 'ceded_agg'})

#: Exhibit row layout (section, item); the Ratio rows take a percentage-*point*
#: impact, every other row a percent-change impact (decision 4).
_GCN_RATIO_ROWS = (('Ratio', 'LR'), ('Ratio', 'ER'), ('Ratio', 'CR'))


def gcn_assemble_column(*, ceded, prem_mean, prem_sd, loss_mean, loss_sd,
                        loss_skew, exp_mean, exp_sd, uw_pctiles):
    """Assemble one GCN waterfall column's full row vector from per-leg stats.

    The shared signed-additive table builder ([pnl-share], appendix S1): both
    :meth:`PnL._gcn_perspective_rows` (Phase-1 scalar premium + loss marginal)
    and :class:`aggregate.reinstatement.ReinstatementAnalysis` (stochastic ceded
    premium ``D + h(R)``, joint-sourced UW) feed it per-leg statistics and get
    back one column of the exhibit. Keeping one builder guarantees the two
    exhibits share rows, sign convention, and the CV-not-SD margin rule.

    Parameters
    ----------
    ceded : bool
        Cession column (premium paid ``-``, recovery received ``+``) vs a
        retained column (premium ``+``, loss borne ``-``).
    prem_mean, prem_sd : float
        Premium leg mean (magnitude) and SD. ``prem_sd`` is ``0`` for a fixed
        premium; a stochastic ceded premium makes it nonzero.
    loss_mean, loss_sd, loss_skew : float
        Loss / recovery leg moments (magnitudes).
    exp_mean, exp_sd : float
        Expense leg mean (magnitude) and SD (``0`` for a deterministic expense).
    uw_pctiles : dict
        ``{level: signed UW value}`` for this column, already computed by the
        caller in the adverse-tail direction.

    Returns
    -------
    dict
        ``(section, item) -> value`` for one column, ready to assemble into the
        ``gcn_df`` frame.
    """
    nan = float('nan')
    P, E, lm = prem_mean, exp_mean, loss_mean
    # --- Mean section (signed; adds down to UW and across the GCN split) ---
    premium = (-P) if ceded else P
    loss = (+lm) if ceded else (-lm)            # recovery is a gain (+)
    expense = (+E) if ceded else (-E)
    uw = premium + loss + expense
    # --- Ratio section (magnitudes) ----------------------------------------
    lr = lm / P if P else nan
    er = E / P if P else nan
    cr = (lm + E) / P if P else nan
    # --- Volatility section ------------------------------------------------
    # CV of each Mean-section leg (order premium, loss, expense). Premium /
    # expense are deterministic (CV 0) in Phase 1; reinstatements make premium
    # stochastic, Phase 3 slide / pc make expense stochastic.
    cv_premium = prem_sd / P if P else nan
    cv_loss = loss_sd / lm if lm else nan
    cv_expense = exp_sd / E if E else 0.0
    sd_lr = loss_sd / P if P else nan
    sd_cr = loss_sd / P if P else nan           # expense deterministic in Phase 1
    out = {
        ('Mean', 'Premium'): premium, ('Mean', 'Loss'): loss,
        ('Mean', 'Expense'): expense, ('Mean', 'UW'): uw,
        ('Ratio', 'LR'): lr, ('Ratio', 'ER'): er, ('Ratio', 'CR'): cr,
        ('Volatility', 'CV Premium'): cv_premium,
        ('Volatility', 'CV Loss'): cv_loss,
        ('Volatility', 'CV Expense'): cv_expense,
        ('Volatility', 'SD LR'): sd_lr, ('Volatility', 'SD CR'): sd_cr,
        ('Volatility', 'Skew LR'): loss_skew, ('Volatility', 'Skew CR'): loss_skew,
    }
    for lvl, val in uw_pctiles.items():
        out[('UW %ile', f'{lvl:g}')] = val
    return out


class PnL:
    """A profit-and-loss position: a consideration held against a risky leg.

    Parameters
    ----------
    agg : Aggregate
        The risky leg ``X`` (the obligation). Its ``value_type`` (``'loss'`` or
        ``'payoff'``) fixes the combine sign; it is otherwise untouched.
    consideration : float, array-like, or callable
        The amount changing hands at inception ``C``, **signed** (``+``
        received, ``-`` paid). A scalar or vector is a constant consideration
        (a vector sums to one book amount). A callable ``f(x)`` is an increasing
        function of the loss outcome (swing / slide / profit commission),
        applied bucket-wise on ``X``'s grid.

    Notes
    -----
    Construct via :meth:`Aggregate.make_pnl` or ``build('pnl ...')``; the two
    coincide. A ``PnL`` is **evaluated** (Cherny--Madan breakeven), not priced
    -- see a later stage for ``evaluate``.
    """

    def __init__(self, agg, consideration=None, *, gross=None, ceded=None, net=None,
                 expense_spec=None, gcn_economics=None):
        self.agg = agg
        self.program = ''
        self._pnl_df = None
        #: cached backing ReinstatementAnalysis (built lazily on first exhibit
        #: access when the aggregate carries DecL reinstatement terms).
        self._reins_analysis = None
        #: gross-expense spec: a list of ``(basis, value)`` terms (basis in
        #: ``{'premium', 'loss', 'fixed'}``) that sum (decision 1); a bare
        #: ``(basis, value)`` tuple is also accepted; ``None`` => no expense.
        self._expense_spec = expense_spec
        #: per-side GCN economics from DecL ceded-premium clauses --
        #: ``{'pc_occ', 'pc_agg', 'c_occ', 'c_agg'}`` (ceded premiums and
        #: commissions by waterfall side); ``None`` => scalar / no commission.
        self._gcn_econ = gcn_economics
        gcn = gross is not None or ceded is not None
        if gcn:
            if gross is None or ceded is None:
                raise ValueError(
                    'a Gross/Ceded/Net PnL needs both gross= and ceded= premiums.')
            if consideration is not None:
                raise ValueError(
                    'pass either consideration= or gross=/ceded=, not both.')
            if agg.agg_reins is None and agg.occ_reins is None:
                raise ValueError(
                    'the Gross/Ceded/Net view requires reinsurance on the risky '
                    'leg (gross/ceded/net loss distributions); the aggregate '
                    "carries no 'occurrence'/'aggregate net of/ceded to' treaty.")
            # net consideration = gross received - ceded paid (or stated directly)
            self._gcn = {'gross': float(gross), 'ceded': float(ceded)}
            self.consideration = float(net) if net is not None \
                else float(gross) - float(ceded)
        else:
            if consideration is None:
                raise ValueError(
                    'PnL needs a consideration= (or gross=/ceded= for the '
                    'Gross/Ceded/Net view).')
            self._gcn = None
            self.consideration = consideration

    # ------------------------------------------------------------------
    # Identity / lifecycle
    # ------------------------------------------------------------------
    @property
    def name(self):
        """The name of the underlying risky leg."""
        return self.agg.name

    @property
    def value_type(self):
        """A P&L is **always** payoff -- more net money is better."""
        return 'payoff'

    def update(self, log2=16, bs=0, **kwargs):
        """Update the underlying loss aggregate; invalidate the derived net.

        Delegates to :meth:`Aggregate.update` (the risky leg carries all the
        FFT / windowing machinery); the net :attr:`pnl_df` is a lazy transform
        of the result, so it is simply invalidated here.

        Parameters
        ----------
        log2, bs : int, float
            Passed through to :meth:`Aggregate.update`.
        **kwargs
            Passed through to :meth:`Aggregate.update`.

        Returns
        -------
        PnL
            ``self`` (for chaining), matching the build/update protocol.
        """
        self.agg.update(log2=log2, bs=bs, **kwargs)
        self._pnl_df = None
        self._reins_analysis = None
        return self

    # ------------------------------------------------------------------
    # Reinstatement analysis (stochastic ceded premium) -- lazy backing engine
    # ------------------------------------------------------------------
    @property
    def reinstatement_analysis(self):
        """The backing :class:`ReinstatementAnalysis`, or ``None``.

        Present when the underlying aggregate carries DecL reinstatement terms
        (``self.agg.reinstatement_terms``, set by a ``reinstatements`` clause);
        built on first access and cached. The GCN exhibit (:attr:`gcn_df`, and
        hence :attr:`summary_df`) delegates to it so the stochastic ceded
        premium ``D + h(R)`` is reflected. Returns ``None`` for an ordinary
        P&L. See ``dev/plan-reinstatements.md`` decisions 1-2.
        """
        terms = getattr(self.agg, 'reinstatement_terms', None)
        if terms is None:
            return None
        if self._reins_analysis is None:
            pg = getattr(self.agg, 'reinstatement_gross_premium', None)
            self._reins_analysis = self.agg.reinstatement_analysis(
                gross_premium=pg, terms=terms)
        return self._reins_analysis

    # ------------------------------------------------------------------
    # The net distribution (derived, cached)
    # ------------------------------------------------------------------
    def _consideration_at(self, x):
        """Consideration value(s) for loss outcome(s) ``x``.

        A callable is applied elementwise (loss-sensitive consideration); a
        constant scalar / vector collapses to one book amount (the vector sums,
        matching the single deterministic inception cash flow).
        """
        c = self.consideration
        if callable(c):
            return np.asarray(c(x), dtype=float)
        return float(np.sum(np.asarray(c, dtype=float)))

    def _gross_expense(self):
        """The gross expense ``E_G`` as a scalar (deterministic in Phase 1).

        ``_expense_spec`` is a **list** of ``(basis, value)`` terms (``and``-joined
        in DecL) that **sum**; a bare ``(basis, value)`` tuple is accepted too (the
        Python-API single-term form). For each term ``fixed`` is the currency
        amount, ``premium`` a fraction of the gross premium, ``loss`` a fraction of
        the **expected gross loss** (so the leg stays deterministic -- it becomes a
        distribution only with slide / PC in Phase 3). ``None`` / empty => ``0``.
        """
        spec = self._expense_spec
        if not spec:
            return 0.0
        # Normalize a bare ``(basis, value)`` tuple to a one-term list.
        terms = [spec] if isinstance(spec[0], str) else spec
        return float(sum(self._one_expense(basis, val) for basis, val in terms))

    def _one_expense(self, basis, val):
        """One expense term resolved to currency (see :meth:`_gross_expense`)."""
        if basis == 'fixed':
            return float(val)
        if basis == 'premium':
            pg = float(self._gcn['gross']) if self._gcn is not None \
                else float(self._consideration_at(0.0))
            return float(val) * pg
        if basis == 'loss':
            has_reins = self.agg.occ_reins is not None or self.agg.agg_reins is not None
            if has_reins:
                rd = self.agg.reins_density_df
                eg = float((rd['loss'].to_numpy() * rd['p_agg_gross'].to_numpy()).sum())
            else:
                x, p = self._xp()
                eg = float((x * p).sum())
            return float(val) * eg
        raise ValueError(f'unknown expense basis {basis!r}')

    def _total_commission(self):
        """Total ceding commission ``C`` (a credit reducing net expense).

        Deterministic in Phase 1 (the ``cede`` clause); ``0`` without one. Slide
        / profit commission make this a distribution in Phase 3 (appendix S1).
        """
        econ = self._gcn_econ
        if econ is None:
            return 0.0
        return float(econ.get('c_occ', 0.0) + econ.get('c_agg', 0.0))

    def _net_expense(self):
        """Net expense ``= gross expense - commission`` (decision 2)."""
        return self._gross_expense() - self._total_commission()

    def _net_values(self, x):
        """Map loss-grid outcomes ``x`` to net P&L outcomes (``C -/+ X - E_net``)."""
        c = self._consideration_at(x)
        base = (c - x) if self.agg._is_loss_value else (c + x)
        return base - self._net_expense()          # net of expense less commission

    @property
    def pnl_df(self):
        """The net P&L distribution -- a deterministic transform of ``X``.

        ``net = C -/+ X`` is comonotone with ``X``, so this is a 1-D relabel of
        the loss density onto the net grid (sorted ascending, duplicate net
        outcomes summed) -- no convolution, no FFT, no windowing, no dropped
        mass. Cached; invalidated by :meth:`update`.

        Returns
        -------
        pandas.DataFrame
            Indexed by net outcome ``net``; columns ``p_total`` (mass), ``F``
            (cdf), ``S`` (survival).
        """
        if self._pnl_df is None:
            x, p = self._xp()
            self._pnl_df = self._frame_from(self._net_values(x), p)
        return self._pnl_df

    @staticmethod
    def _frame_from(values, probs):
        """A net frame (``p_total``/``F``/``S``, index ``net``) from ``(values,
        probs)`` -- duplicate outcomes summed, sorted ascending.
        """
        ser = pd.Series(probs, index=values).groupby(level=0).sum().sort_index()
        df = pd.DataFrame({'p_total': ser.to_numpy()},
                          index=pd.Index(ser.index.to_numpy(), name='net'))
        df['F'] = df['p_total'].cumsum()
        df['S'] = 1.0 - df['F']
        return df

    # ------------------------------------------------------------------
    # Moments (all on the shared loss grid (x, p), so the summary EX column
    # adds exactly: Consideration + Obligation = Margin)
    # ------------------------------------------------------------------
    def _xp(self):
        """The loss grid ``(x, p)`` from the obligation's density."""
        dd = self.agg.density_df
        return dd['loss'].to_numpy(dtype=float), dd['p_total'].to_numpy(dtype=float)

    @staticmethod
    def _moms(vals, p):
        """``(mean, sd, skew)`` of ``vals`` under weights ``p``."""
        m = float((vals * p).sum())
        var = float((vals * vals * p).sum()) - m * m
        sd = var ** 0.5 if var > 0 else 0.0
        skew = float((((vals - m) ** 3) * p).sum()) / sd ** 3 if sd > 0 else 0.0
        return m, sd, skew

    def _moments(self):
        """``(mean, sd, skew)`` of the net P&L."""
        x, p = self._xp()
        return self._moms(self._net_values(x), p)

    @property
    def mean(self):
        """E[net] = C - E[X] (loss) or C + E[X] (payoff)."""
        return self._moments()[0]

    @property
    def sd(self):
        """SD of the net (invariant under the constant-consideration shift)."""
        return self._moments()[1]

    @property
    def var(self):
        """Variance of the net."""
        return self._moments()[1] ** 2

    @property
    def skew(self):
        """Skewness of the net (sign-flipped vs the loss for ``X`` a loss)."""
        return self._moments()[2]

    @property
    def prob_loss(self):
        """``P(net < 0)`` -- the probability the position loses money."""
        x, p = self._xp()
        y = self._net_values(x)
        return float(p[y < 0].sum())

    # ------------------------------------------------------------------
    # Reporting: the signed, additive P&L summary
    # ------------------------------------------------------------------
    @property
    def summary_df(self):
        """Signed, additive P&L rows: ``Consideration + Obligation = Margin``.

        Three rows, each its **signed contribution** to the net P&L (so the
        ``EX`` column *adds*), reported with the **SD** spread, *not* CV -- the
        margin sits near break-even where ``CV = sd / mean`` is meaningless:

        - **Consideration** -- the amount changing hands at inception
          (``+`` received, ``-`` paid).
        - **Obligation** -- the risky leg's signed contribution (``-`` a loss
          borne, ``+`` a payoff held).
        - **Margin** -- the net position (``= Consideration + Obligation``).

        A sold cover shows Consideration ``+``, Obligation ``-``, Margin their
        sum; **buying flips both signs** (you pay at inception *and* hold the
        payoff). Freq / Sev / Agg detail lives on ``pnl.agg.summary_df`` (the
        honest obligation table) -- not here. See dev/plan-pnl.md S3.1.

        Returns
        -------
        pandas.DataFrame
            Rows ``Consideration`` / ``Obligation`` / ``Margin``; columns
            ``EX`` / ``SD`` / ``Sk``. For a Gross/Ceded/Net position
            (``make_pnl(gross=, ceded=)``) this is instead the additive GCN
            exhibit -- see :meth:`gcn_df`.
        """
        if self._gcn is not None:
            return self.gcn_df
        x, p = self._xp()
        # consideration leg (constant -> certain, SD 0; callable -> f(X))
        if callable(self.consideration):
            cm, csd, csk = self._moms(np.asarray(self.consideration(x), dtype=float), p)
        else:
            cm, csd, csk = self._consideration_at(x), 0.0, 0.0
        # obligation = the risky leg's SIGNED contribution to the net
        sign = -1.0 if self.agg._is_loss_value else 1.0
        om, osd, osk = self._moms(sign * x, p)
        # expense leg: a signed cost (deterministic in Phase 1, so SD 0)
        eg = self._gross_expense()
        em, esd, esk = -eg, 0.0, 0.0
        mm, msd, msk = self._moms(self._net_values(x), p)
        df = pd.DataFrame(
            {'EX': [cm, om, em, mm], 'SD': [csd, osd, esd, msd],
             'Sk': [csk, osk, esk, msk]},
            index=['Consideration', 'Obligation', 'Expense', 'Margin'])
        df.index.name = 'P&L'
        # Combined-ratio line (loss + expense over premium); a loss-leg notion,
        # NaN for a payoff or a non-positive premium.
        lm = float((x * p).sum())
        prem = cm
        if self.agg._is_loss_value and prem > 0:
            df.loc['Combined ratio'] = [(lm + eg) / prem, np.nan, np.nan]
        return df

    # ------------------------------------------------------------------
    # The Gross / Ceded / Net exhibit (the leg model; appendix S1)
    # ------------------------------------------------------------------
    def _gcn_marginal(self, perspective):
        """The ``(x, p)`` loss/recovery marginal for a waterfall perspective.

        Each perspective reads its **own** exact aggregate marginal from
        ``Aggregate.reins_density_df`` (``p_agg_gross`` etc.), so a cession
        perspective carries the recovery distribution and a retained perspective
        the net-loss distribution -- no ``(L, R)`` joint is needed (appendix S2).
        """
        rd = self.agg.reins_density_df
        x = rd['loss'].to_numpy(dtype=float)
        p = rd[_GCN_LOSS_MARGINAL[perspective]].to_numpy(dtype=float)
        return x, p

    @staticmethod
    def _quantile(x, p, level):
        """Lower ``level``-quantile of a discrete ``(x, p)`` marginal."""
        F = np.cumsum(p)
        idx = int(np.searchsorted(F, level, side='left'))
        return float(x[min(idx, len(x) - 1)])

    def _gcn_perspective_rows(self, perspective, prem_mag, exp_mag):
        """The full row vector (all sections) for one waterfall **column**.

        Means are signed so they add down to ``UW`` *and* across the GCN split
        (a cession pays premium ``-`` and receives recovery / commission ``+``).
        Ratios use magnitudes (a ceded ``LR`` is recovery / ceded-premium).
        Volatility and percentiles come from this column's own marginal; in
        Phase 1 the expense is deterministic so ``SD(ER) = 0`` and
        ``SD(CR) = SD(loss) / premium`` (decision 4).
        """
        x, p = self._gcn_marginal(perspective)
        lm, lsd, lsk = self._moms(x, p)            # loss (or recovery) moments
        ceded = perspective in _GCN_CEDED
        P, E = prem_mag, exp_mag
        # --- UW percentiles, loss-severity aligned ---------------------
        # The rows index the loss-severity direction, so every column reads as
        # one scenario: a *low* level is the bad-loss tail. Retained UW (gross /
        # net) = P - loss - E falls with loss, so it reads the (1 - lvl) loss
        # quantile. A cession UW = recovery + E - P *rises* with loss, so it
        # reads the lvl recovery quantile -- i.e. the cession column is reversed
        # (1 - lvl), giving the cedant's largest benefit at the worst-loss row.
        # Percentiles never add across columns (different sort orders), so this
        # alignment costs no additivity.
        pct = {}
        for lvl in GCN_PERCENTILES:
            if ceded:
                pct[lvl] = self._quantile(x, p, 1.0 - lvl) + E - P
            else:
                pct[lvl] = P - self._quantile(x, p, 1.0 - lvl) - E
        # Phase-1 premium and expense are deterministic (SD 0); the shared
        # builder applies the signs, ratios, CV rows and percentile placement.
        return gcn_assemble_column(
            ceded=ceded, prem_mean=P, prem_sd=0.0,
            loss_mean=lm, loss_sd=lsd, loss_skew=lsk,
            exp_mean=E, exp_sd=0.0, uw_pctiles=pct)

    @staticmethod
    def _gcn_impact(target, base, ratio_rows):
        """A percent-change (or percentage-*point*, for ratio rows) impact column."""
        out = {}
        for key in target:
            t, b = target[key], base[key]
            if key in ratio_rows:
                out[key] = t - b                   # points
            elif b:
                out[key] = t / b - 1.0             # signed fraction
            else:
                out[key] = float('nan')
        return out

    @property
    def gcn_df(self):
        """The Gross / Ceded / Net exhibit -- a multi-section waterfall (decision 4).

        Columns are the reinsurance waterfall (inuring **occurrence -> aggregate**,
        the aggregate cover applying to net-of-occurrence), shown only for the
        sides that are configured::

            gross | ceded occ | net occ | occ impact | ceded agg | net agg | agg impact | impact

        With only one side present the single delta column is just ``impact`` and
        the ``ceded`` / ``net`` columns drop the occ / agg qualifier. The
        ``*impact`` columns are **percent change** of one perspective versus its
        predecessor (``occ`` vs gross, ``agg`` vs net-occ, ``impact`` = net vs
        gross) -- a percentage-*point* change on the Ratio rows.

        Rows, in sections: **Mean** (Premium / Loss / Expense / UW, signed --
        adds down to UW *and* across the GCN split); **Ratio** (LR / ER / CR from
        the means); **Volatility** (SD and Skew of LR / CR); and **UW
        percentiles** (:data:`GCN_PERCENTILES`), **loss-severity aligned** so each
        row reads as one scenario direction: a low level is the bad-loss tail, so
        the cession columns run reversed (largest cedant benefit at the worst-loss
        row). Only the Mean section adds across columns -- SD / skew / percentiles
        are per-column marginals and do **not** add ("means add, SDs don't").

        Returns
        -------
        pandas.DataFrame
            Row ``MultiIndex`` ``(section, item)``; one column per present
            waterfall perspective plus the ``*impact`` columns.

        Notes
        -----
        When the underlying aggregate carries a DecL ``reinstatements`` clause
        the ceded premium ``D + h(R)`` is stochastic, so this delegates to the
        backing :class:`~aggregate.reinstatement.ReinstatementAnalysis` (whose
        columns read off the ``(L, R)`` joint pushforward, not a 1-D loss
        marginal). See ``dev/plan-reinstatements.md`` decision 2.
        """
        analysis = self.reinstatement_analysis
        if analysis is not None:
            return analysis.gcn_df
        agg = self.agg
        if agg.reins_density_df is None:
            raise ValueError(
                'no gross/ceded/net loss views on the risky leg; update the '
                'aggregate (it must carry occurrence or aggregate reinsurance).')
        has_occ = agg.occ_reins is not None
        has_agg = agg.agg_reins is not None
        both = has_occ and has_agg
        # Premium magnitudes per perspective. DecL ceded-premium clauses give a
        # per-side split (`_gcn_econ`); the scalar Python-API GCN books its single
        # ceded amount on the side actually present.
        p_gross = float(self._gcn['gross'])
        econ = self._gcn_econ
        if econ is not None:
            pc_occ = float(econ.get('pc_occ', 0.0))
            pc_agg = float(econ.get('pc_agg', 0.0))
        else:
            ceded_total = float(self._gcn['ceded'])
            pc_agg = ceded_total if has_agg else 0.0
            pc_occ = ceded_total if (has_occ and not has_agg) else 0.0
        prem_mag = {
            'gross': p_gross,
            'ceded_occ': pc_occ, 'net_occ': p_gross - pc_occ,
            'ceded_agg': pc_agg, 'net_agg': p_gross - pc_occ - pc_agg,
        }
        # Gross expense books on the gross leg; each cession credits a commission
        # (Phase 1: deterministic `cede`, absent here, so 0) so net expense
        # = E_G - C_occ - C_agg. Means then add across the split.
        e_gross = self._gross_expense()
        c_occ = float(econ.get('c_occ', 0.0)) if econ is not None else 0.0
        c_agg = float(econ.get('c_agg', 0.0)) if econ is not None else 0.0
        exp_mag = {
            'gross': e_gross,
            'ceded_occ': c_occ, 'net_occ': e_gross - c_occ,
            'ceded_agg': c_agg, 'net_agg': e_gross - c_occ - c_agg,
        }
        # Ordered display columns: (label, kind, spec).
        final_net = 'net_agg' if has_agg else 'net_occ'
        # The final net premium honors a `net=` override (== gross - ceded
        # otherwise, so the no-override snapshot is unchanged).
        prem_mag[final_net] = float(self.consideration)
        ordered = [('gross', 'persp', 'gross')]
        if has_occ:
            q = ' occ' if both else ''
            ordered += [(f'ceded{q}', 'persp', 'ceded_occ'),
                        (f'net{q}', 'persp', 'net_occ')]
            if both:
                ordered.append(('occ impact', 'impact', ('net_occ', 'gross')))
        if has_agg:
            q = ' agg' if both else ''
            ordered += [(f'ceded{q}', 'persp', 'ceded_agg'),
                        (f'net{q}', 'persp', 'net_agg')]
            if both:
                ordered.append(('agg impact', 'impact', ('net_agg', 'net_occ')))
        ordered.append(('impact', 'impact', (final_net, 'gross')))
        # Compute each perspective's rows once, then assemble columns.
        rows = {k: self._gcn_perspective_rows(k, prem_mag[k], exp_mag[k])
                for _, kind, k in ordered if kind == 'persp'}
        data = {}
        for label, kind, spec in ordered:
            if kind == 'persp':
                data[label] = rows[spec]
            else:
                tgt, base = spec
                data[label] = self._gcn_impact(rows[tgt], rows[base],
                                               _GCN_RATIO_ROWS)
        df = pd.DataFrame(data)
        df.index = pd.MultiIndex.from_tuples(df.index, names=['section', 'item'])
        return df

    # ------------------------------------------------------------------
    # Distribution functions (read the net frame)
    # ------------------------------------------------------------------
    def q(self, p):
        """Lower ``p``-quantile of the net: smallest outcome with ``F >= p``."""
        df = self.pnl_df
        y = df.index.to_numpy(dtype=float)
        F = df['F'].to_numpy(dtype=float)
        idx = int(np.searchsorted(F, p, side='left'))
        idx = min(idx, len(y) - 1)
        return float(y[idx])

    def cdf(self, x):
        """``P(net <= x)`` from the net frame (right-continuous step)."""
        df = self.pnl_df
        y = df.index.to_numpy(dtype=float)
        F = df['F'].to_numpy(dtype=float)
        idx = int(np.searchsorted(y, x, side='right')) - 1
        return float(F[idx]) if idx >= 0 else 0.0

    def sf(self, x):
        """``P(net > x)`` -- the survival function of the net."""
        return 1.0 - self.cdf(x)

    # ------------------------------------------------------------------
    # Evaluation: the Cherny--Madan breakeven acceptability panel
    # ------------------------------------------------------------------
    def evaluate(self, names=None):
        """Evaluate the position: the Cherny--Madan breakeven acceptability panel.

        A P&L is **evaluated, not priced**. You never price the net; you price
        the risky leg ``X`` and ask: which distortion drives the risk-adjusted
        net to **zero** -- the breakeven stress the position survives? With
        translation and duality both orientations reduce to the *same* equation,

            ``P = g(loss-version of the risky leg)``

        -- exactly the ``rho_g(S) = premium_target`` calibration
        :meth:`Distortion.calibrate_set` already solves, with
        ``premium_target = the held consideration P`` over the **full** support
        (no asset cap, no cost-of-capital inversion). Hence a ``PnL`` exposes
        ``evaluate``, not the ``coc`` price methods, and ``ccoc`` (which needs an
        asset level) is **excluded** from the panel.

        The raw breakeven parameter is family-specific (``wang`` lambda is not
        ``tvar`` p), but ``gini_p = 2 integral g - 1`` is family-agnostic,
        comparable, and monotone in loading -- so the breakeven ``gini_p`` is the
        single **acceptability index** (Cherny & Madan; do not abbreviate it
        "AI"): a more profitable position survives a larger stress and so scores a
        larger ``gini_p``. The ``calibrate_set`` family loop therefore yields one
        acceptability panel in a common currency.

        Parameters
        ----------
        names : sequence of str, optional
            Distortion families to evaluate. Defaults to the standard set minus
            ``ccoc`` (``ph``, ``wang``, ``dual``, ``tvar``).

        Returns
        -------
        pandas.DataFrame
            One row per family, columns ``param_name`` / ``param`` (the
            family-specific breakeven parameter), ``error`` (calibration
            residual), ``gini_p`` (the acceptability index), and ``area``
            (``= (gini_p + 1) / 2 = integral g``).

        Notes
        -----
        Constant consideration only in this release; function-valued
        (loss-sensitive) evaluation is deferred. See dev/plan-pnl.md S1.
        """
        if callable(self.consideration):
            raise NotImplementedError(
                'evaluate requires a constant consideration; function-valued '
                '(loss-sensitive) evaluation is deferred.')
        from .spectral import Distortion
        from ._pricing import (_canonical_loss_frame, _calibration_survival,
                               _limited_ev, DEFAULT_CALIBRATION_DISTORTIONS)
        if names is None:
            names = tuple(n for n in DEFAULT_CALIBRATION_DISTORTIONS
                          if n != 'ccoc')
        # Canonical 0-based loss frame of the risky leg (reverses a payoff).
        dz, c, _reverse = _canonical_loss_frame(self.agg)
        bs = self.agg.bs
        a_full = float(dz.index[-1])               # full support: no asset cap
        S, ess_sup = _calibration_survival(dz, bs, a_full)
        el = _limited_ev(dz, bs, a_full + bs)      # E[loss-version] (uncapped)
        # breakeven g(loss-version) = P  <=>  g(Z) = P + c  (translation by c).
        # The premium available to absorb loss is net of expense less commission.
        P = float(self._consideration_at(0.0)) - self._net_expense()
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
    # Plot: the net (Margin) density + distribution, no severity panel
    # ------------------------------------------------------------------
    def plot(self, axd=None, **kwargs):
        """Plot the net P&L (Margin) density and distribution.

        A P&L is an **affine of its aggregate**, not a compound of a severity,
        so there is **no severity / density-derivative panel** (that belongs to
        an :class:`Aggregate` -- use ``pnl.agg.plot()`` for the bare risky leg).
        Two panels: the Margin density (A) and distribution (B), with the
        break-even line at 0 marked.

        Parameters
        ----------
        axd : dict of str to Axes, optional
            Mosaic with keys ``'A'`` (density) and ``'B'`` (distribution); a new
            figure is created if omitted and stored on ``pnl.figure``.
        **kwargs
            Passed to the canvas creator (e.g. ``figsize``).

        Returns
        -------
        matplotlib.figure.Figure
        """
        from .plots import plot_pnl
        return plot_pnl(self, axd=axd, **kwargs)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    def __repr__(self):
        c = self.consideration
        cstr = '<fn>' if callable(c) else f'{c}'
        return (f'PnL({self.name!r}: consideration={cstr}, '
                f'X={self.agg.value_type})')
