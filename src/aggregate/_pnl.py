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

    def __init__(self, agg, consideration):
        self.agg = agg
        self.consideration = consideration
        self.program = ''
        self._pnl_df = None

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
            dd = self.agg.density_df
            x = dd['loss'].to_numpy(dtype=float)
            p = dd['p_total'].to_numpy(dtype=float)
            y = self._net_values(x)
            # group duplicate net outcomes (a callable consideration may not be
            # injective) and sort ascending.
            ser = pd.Series(p, index=y).groupby(level=0).sum().sort_index()
            df = pd.DataFrame({'p_total': ser.to_numpy()},
                              index=pd.Index(ser.index.to_numpy(), name='net'))
            df['F'] = df['p_total'].cumsum()
            df['S'] = 1.0 - df['F']
            self._pnl_df = df
        return self._pnl_df

    # ------------------------------------------------------------------
    # Moments of the net (from the derived density; exact for discrete)
    # ------------------------------------------------------------------
    def _moments(self):
        df = self.pnl_df
        y = df.index.to_numpy(dtype=float)
        p = df['p_total'].to_numpy(dtype=float)
        m = float((y * p).sum())
        var = float((y * y * p).sum()) - m * m
        sd = var ** 0.5 if var > 0 else 0.0
        skew = float((((y - m) ** 3) * p).sum()) / sd ** 3 if sd > 0 else 0.0
        return m, sd, skew

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
        df = self.pnl_df
        y = df.index.to_numpy(dtype=float)
        p = df['p_total'].to_numpy(dtype=float)
        return float(p[y < 0].sum())

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
    # Display
    # ------------------------------------------------------------------
    def __repr__(self):
        c = self.consideration
        cstr = '<fn>' if callable(c) else f'{c}'
        return (f'PnL({self.name!r}: consideration={cstr}, '
                f'X={self.agg.value_type})')
