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

    def __init__(self, agg, consideration=None, *, gross=None, ceded=None, net=None):
        self.agg = agg
        self.program = ''
        self._pnl_df = None
        gcn = gross is not None or ceded is not None
        if gcn:
            if gross is None or ceded is None:
                raise ValueError(
                    'a Gross/Ceded/Net PnL needs both gross= and ceded= premiums.')
            if consideration is not None:
                raise ValueError(
                    'pass either consideration= or gross=/ceded=, not both.')
            if agg.agg_reins is None:
                raise ValueError(
                    'the Gross/Ceded/Net view requires aggregate reinsurance on '
                    'the risky leg (gross/ceded/net loss distributions); the '
                    "aggregate carries no 'aggregate net of/ceded to' treaty.")
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
        return self

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

    def _net_values(self, x):
        """Map loss-grid outcomes ``x`` to net P&L outcomes (``C -/+ X``)."""
        c = self._consideration_at(x)
        return (c - x) if self.agg._is_loss_value else (c + x)

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
        mm, msd, msk = self._moms(self._net_values(x), p)
        df = pd.DataFrame(
            {'EX': [cm, om, mm], 'SD': [csd, osd, msd], 'Sk': [csk, osk, msk]},
            index=['Consideration', 'Obligation', 'Margin'])
        df.index.name = 'P&L'
        return df

    @property
    def gcn_df(self):
        """The Gross / Ceded / Net exhibit -- **doubly additive**, signed.

        For a ``make_pnl(gross=, ceded=)`` position on an aggregate-reinsurance
        risky leg, a 3x3 table whose rows are the three legs and columns the
        signed P&L parts, additive **both ways**:

        - rows add: ``Net = Gross + Ceded`` (the legs are comonotone -- all
          deterministic functions of the one gross loss -- so this stays 1-D, no
          joint model); and
        - columns add: ``Margin = Consideration + Obligation`` (the
          :meth:`summary_df` convention).

        The ceded leg is literally negative relative to gross: you **pay** the
        ceded premium (Consideration ``-Pc``) and **receive** the recovery
        (Obligation ``+E[R]``, a gain), so it nets the gross down to the
        retained position. ``Net`` is the headline (it drives the net
        :attr:`pnl_df`, moments, plot and :meth:`evaluate`).

        Returns
        -------
        pandas.DataFrame
            Rows ``Gross`` / ``Ceded`` / ``Net``; columns ``Consideration`` /
            ``Obligation`` / ``Margin`` (expected values).
        """
        xs = self.agg.xs
        if self.agg.agg_density_gross is None:
            raise ValueError(
                'no aggregate gross/ceded/net loss views on the risky leg; '
                'update the aggregate (it must carry aggregate reinsurance).')
        e_gross = float((xs * self.agg.agg_density_gross).sum())
        e_recov = float((xs * self.agg.agg_density_ceded).sum())  # recovery R
        e_net = float((xs * self.agg.agg_density_net).sum())
        pg = self._gcn['gross']
        pc = self._gcn['ceded']
        pn = float(self.consideration)
        # signed P&L contributions (loss subtracts, recovery adds).
        df = pd.DataFrame(
            {'Consideration': [pg, -pc, pn],
             'Obligation': [-e_gross, e_recov, -e_net],
             'Margin': [pg - e_gross, e_recov - pc, pn - e_net]},
            index=['Gross', 'Ceded', 'Net'])
        df.index.name = 'leg'
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
        P = float(self._consideration_at(0.0))
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
