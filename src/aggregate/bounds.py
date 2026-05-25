"""
Pricing-bounds analysis (Mildenhall, IME 2022).

The :class:`Bounds` class is constructed in one shot from a distribution and a
target premium. It computes the bounding pricing distortions consistent with
the premium, exposes the min/max envelope of that family, and renders the
three-panel "cloud" figure from the paper.

Naming convention used throughout
---------------------------------

============  =============  ============================================
Name          Shape          Meaning
============  =============  ============================================
``p_knots``   ``(n_p,)``     TVaR threshold values — the p axis
``s_grid``    ``(n_s,)``     distortion evaluation points — the s axis
``tvar_x_p``  ``(n_p,)``     ``tvar_x_p[i] = TVaR_{p_knots[i]}(min(X, a))``
``tvar_hinges`` ``(n_p, n_s)`` ``min(1, s_grid[j] / (1 - p_knots[i]))``
``cloud_df``  ``(n_s, K)``   each column is a convex combination of two
                              rows of ``tvar_hinges``; ``K`` = number of
                              ``(p_lo, p_hi)`` pairs straddling ``p_star``
============  =============  ============================================

The "hinge family" is the set of TVaR distortions parameterised by p:
``TVaR_p(s) = min(1, s / (1 - p))``. p indexes the family; s is the
distortion argument.
"""
from functools import cached_property
from itertools import cycle
import logging

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import brentq

from .constants import FIG_W
from .spectral import Distortion

logger = logging.getLogger(__name__)


def _resolve_obj(obj, line):
    """
    Coerce *obj* into ``(tvar_x, F, name)`` where

    - ``tvar_x(p)`` returns ``TVaR_p(X)`` (unbounded — capping at ``a`` is
      applied separately in :meth:`Bounds._tvar_x_a`).
    - ``F(x)`` returns ``P(X <= x)``.
    - ``name`` is a display string.

    Accepted obj types: ``Portfolio``, ``Aggregate``, ``pd.Series``,
    ``pd.DataFrame``. For Series/DataFrame the index is interpreted as outcomes
    and values as the pmf; for DataFrame the first column is the pmf.
    """
    # Local imports to keep this module decoupled at import time.
    from .distributions import Aggregate
    from .portfolio import Portfolio
    from .utilities import make_var_tvar

    if isinstance(obj, Portfolio):
        if line == 'total':
            return obj.tvar, obj.cdf, f'{obj.name}.total'
        if line not in obj.line_names_ex:
            raise ValueError(f'line {line!r} not in portfolio {obj.name!r}')
        ag = getattr(obj, line)
        return ag.tvar, ag.cdf, f'{obj.name}.{line}'

    if isinstance(obj, Aggregate):
        return obj.tvar, obj.cdf, obj.name

    if isinstance(obj, pd.DataFrame):
        ser = obj.iloc[:, 0]
        name = obj.columns[0] if hasattr(obj.columns[0], '__str__') else 'frame'
    elif isinstance(obj, pd.Series):
        ser = obj
        name = ser.name if ser.name is not None else 'series'
    else:
        raise TypeError(
            f'Bounds: unsupported obj type {type(obj).__name__}. '
            'Accepted: Portfolio, Aggregate, pd.Series, pd.DataFrame.')

    if not ser.index.is_unique:
        raise ValueError('pmf index must be unique')
    if not ser.index.is_monotonic_increasing:
        raise ValueError('pmf index must be monotonic increasing')
    ser = ser[ser > 0]
    qf = make_var_tvar(ser)
    cdf_ser = ser.cumsum()

    def F(x):
        if x >= cdf_ser.index[-1]:
            return 1.0
        if x < cdf_ser.index[0]:
            return 0.0
        return float(cdf_ser.loc[:x].iloc[-1])

    return qf.tvar, F, str(name)


class Bounds:
    """
    Pricing bounds (IME 2022).

    Parameters
    ----------
    obj : Portfolio, Aggregate, pd.Series, or pd.DataFrame
        The risk X.
    premium : float
        Target premium. Required: ``E[X] < premium <= a``.
    a : float, default ``np.inf``
        Asset cap. The class bounds prices of ``min(X, a)``.
    line : str, default ``'total'``
        Only used when ``obj`` is a ``Portfolio``.
    n_p : int, default ``257``
        Base p-grid size. Adaptive refinement adds a handful of knots
        around ``p_star``.
    n_s : int, default ``513``
        s-grid size. Binary, ``np.linspace(0, 1, n_s)``.

    Attributes
    ----------
    p_star : float
        TVaR threshold where ``TVaR_{p_star}(min(X, a)) = premium``.
    p_knots : ndarray, shape (n_p,)
    s_grid : ndarray, shape (n_s,)
    tvar_x_p : ndarray, shape (n_p,)
    tvar_hinges : ndarray, shape (n_p, n_s)
    weight_df : DataFrame
        One row per bracketing ``(p_lo, p_hi)`` pair with columns
        ``t_lower, t_upper, weight``.
    cloud_df : DataFrame, shape ``(n_s, K)``
        Columns are MultiIndex ``(p_lo, p_hi)``.
    min_envelope : :class:`Distortion`
        Pointwise minimum of the cloud. Min-of-concaves is concave, so this
        is itself a coherent distortion.
    max_envelope : callable
        Pointwise maximum of the cloud, as an ``interp1d`` callable. NOT
        a Distortion (max of concaves is not concave in general).
    min_envelope_hinges : DataFrame, shape ``(n_s, 4)``
        Columns ``s, p_lo, p_hi, weight`` — at each ``s``, the BiTVaR
        bracket from :attr:`weight_df` that achieves the pointwise
        minimum, plus that bracket's convex-combo weight.
    """

    def __init__(self, obj, premium, *, a=np.inf, line='total',
                 n_p=257, n_s=513):
        self._obj = obj
        self.premium = float(premium)
        self.a = float(a) if not np.isinf(a) else np.inf
        self.line = line
        self.n_p = int(n_p)
        self.n_s = int(n_s)

        tvar_x_unb, F, name = _resolve_obj(obj, line)
        self._tvar_x_unb = tvar_x_unb
        self._F = F
        self.name = name
        self.Fb = 1.0 if np.isinf(self.a) else float(F(self.a))

        mean = float(tvar_x_unb(0))
        if self.premium < mean:
            raise ValueError(
                f'premium {self.premium} below mean {mean}; pricing bound undefined')
        if not np.isinf(self.a) and self.premium > self.a:
            raise ValueError(
                f'premium {self.premium} exceeds asset cap {self.a}')

    # ------------------------------------------------------------------
    # Bounded TVaR — TVaR_p(min(X, a))
    # ------------------------------------------------------------------

    def _tvar_x_a(self, p):
        """TVaR_p of min(X, a). Scalar or array p."""
        tvar = self._tvar_x_unb(p)
        if np.isinf(self.a):
            return tvar
        # For p >= F(a) the conditional tail of min(X, a) is exactly a.
        # For p < F(a), TVaR_p(min(X,a)) = TVaR_p(X) - (1-F(a))(TVaR_{F(a)}(X) - a) / (1-p).
        gap = (1.0 - self.Fb) * (self._tvar_x_unb(self.Fb) - self.a)
        return np.where(np.asarray(p) < self.Fb,
                        tvar - gap / (1.0 - p),
                        self.a)

    # ------------------------------------------------------------------
    # p_star — root of TVaR_p(min(X, a)) = premium
    # ------------------------------------------------------------------

    @cached_property
    def p_star(self):
        """The unique p in (0, 1) where ``TVaR_p(min(X, a)) = premium``."""
        f = lambda p: float(self._tvar_x_a(p)) - self.premium
        # Coarse bracket on dyadic grid k/256.
        coarse = np.arange(1, 256) / 256.0
        vals = np.array([f(p) for p in coarse])
        sign_changes = np.where(np.diff(np.sign(vals)) != 0)[0]
        if len(sign_changes) == 0:
            # Bracket on the open interval as fallback.
            lo, hi = 2.0 ** -10, 1.0 - 2.0 ** -10
        else:
            idx = sign_changes[0]
            lo, hi = coarse[idx], coarse[idx + 1]
        return float(brentq(f, lo, hi, xtol=2 ** -17, rtol=2 ** -30))

    # ------------------------------------------------------------------
    # Grids
    # ------------------------------------------------------------------

    @cached_property
    def p_knots(self):
        """The TVaR-threshold grid, shape ``(n_p,)``-ish (adaptive adds knots)."""
        base = np.linspace(0.0, 1.0, self.n_p, endpoint=False)
        # Densification around p_star — dyadic offsets at 2**-8 .. 2**-11.
        offsets = 2.0 ** -np.arange(8, 12)
        extras = np.concatenate([self.p_star + offsets, self.p_star - offsets,
                                 [self.p_star, 1.0]])
        extras = extras[(extras > 0) & (extras <= 1.0)]
        knots = np.unique(np.concatenate([base, extras]))
        return knots

    @cached_property
    def s_grid(self):
        """The distortion-evaluation grid, shape ``(n_s,)``."""
        return np.linspace(0.0, 1.0, self.n_s)

    @cached_property
    def tvar_x_p(self):
        """``tvar_x_p[i] = TVaR_{p_knots[i]}(min(X, a))``, shape ``(n_p,)``."""
        return np.asarray(self._tvar_x_a(self.p_knots), dtype=float)

    @cached_property
    def tvar_hinges(self):
        """``tvar_hinges[i, j] = min(1, s_grid[j] / (1 - p_knots[i]))``."""
        with np.errstate(divide='ignore', invalid='ignore'):
            h = np.minimum(1.0, self.s_grid[None, :] / (1.0 - self.p_knots[:, None]))
        # p == 1 produces inf; the TVaR-1 distortion is g(s)=1 for s>0, g(0)=0.
        h = np.where(self.p_knots[:, None] >= 1.0,
                     (self.s_grid > 0).astype(float)[None, :],
                     h)
        return h

    # ------------------------------------------------------------------
    # Weight table — bracketing (p_lo, p_hi) pairs
    # ------------------------------------------------------------------

    @cached_property
    def weight_df(self):
        """
        Bracketing-pair weights.

        For each pair ``(p_lo, p_hi)`` with ``p_lo <= p_star < p_hi``, the
        weight ``w`` satisfies
        ``(1-w) tvar_x(p_lo) + w tvar_x(p_hi) = premium``.

        Index: MultiIndex ``(p_lo, p_hi)``.
        Columns: ``t_lower, t_upper, weight``.
        """
        ps = self.p_knots
        tps = self.tvar_x_p
        lhs = ps <= self.p_star
        rhs = ps > self.p_star
        pl, pu = np.meshgrid(ps[lhs], ps[rhs], indexing='ij')
        tl, tu = np.meshgrid(tps[lhs], tps[rhs], indexing='ij')
        w = (self.premium - tl) / np.where(tu == tl, 1.0, tu - tl)
        df = pd.DataFrame({
            'p_lower': pl.ravel(),
            'p_upper': pu.ravel(),
            't_lower': tl.ravel(),
            't_upper': tu.ravel(),
            'weight': w.ravel(),
        }).set_index(['p_lower', 'p_upper'])
        return df.sort_index()

    @cached_property
    def cloud_df(self):
        """
        The cloud of weighted-TVaR distortions, shape ``(n_s, K)``.

        ``cloud_df[s, (p_lo, p_hi)] = (1-w) min(1, s/(1-p_lo)) + w min(1, s/(1-p_hi))``
        where ``w`` is the bracket weight from :attr:`weight_df`.
        """
        # Build via vectorised gather + linear combo on tvar_hinges.
        ps = self.p_knots
        idx_lo = np.searchsorted(ps, self.weight_df.index.get_level_values('p_lower'))
        idx_hi = np.searchsorted(ps, self.weight_df.index.get_level_values('p_upper'))
        w = self.weight_df['weight'].values
        lo_rows = self.tvar_hinges[idx_lo, :]    # (K, n_s)
        hi_rows = self.tvar_hinges[idx_hi, :]    # (K, n_s)
        cloud = (1.0 - w[:, None]) * lo_rows + w[:, None] * hi_rows
        return pd.DataFrame(cloud.T, index=self.s_grid,
                            columns=self.weight_df.index).rename_axis('s', axis=0)

    # ------------------------------------------------------------------
    # Envelopes
    # ------------------------------------------------------------------

    @cached_property
    def min_envelope(self):
        """
        Pointwise minimum of the cloud, as a :class:`Distortion`.

        Min-of-concaves is concave, so this is a coherent distortion. Built
        through :meth:`Distortion.s_gs_distortion` (convex-envelope kind).
        """
        s = self.s_grid
        g = self.cloud_df.min(axis=1).values
        return Distortion.s_gs_distortion(
            s, g, display_name=f'min env({self.name}, prem={self.premium:.4g})')

    @cached_property
    def max_envelope(self):
        """
        Pointwise maximum of the cloud, as a linear-interpolation callable.

        Max-of-concaves is generally not concave, so this is NOT a
        :class:`Distortion`. Use as a function: ``bd.max_envelope(s)``.
        """
        s = self.s_grid
        g = self.cloud_df.max(axis=1).values
        return interp1d(s, g, kind='linear', bounds_error=False,
                        fill_value=(0.0, 1.0))

    @cached_property
    def min_envelope_hinges(self):
        """
        The active bracketing BiTVaR at each ``s`` along the minimum envelope.

        For every ``s`` in :attr:`s_grid`, the minimum envelope's value
        ``min_envelope.g(s) = cloud_df.loc[s].min()`` is achieved by *one*
        of the cloud columns — i.e. by exactly one ``(p_lo, p_hi)`` bracket
        from :attr:`weight_df`. This frame records, for each ``s``, which
        bracket that is, plus its weight.

        Each row fully specifies the BiTVaR realising the envelope at that
        ``s``:

            ``g_s(u) = (1 - w) * min(1, u / (1 - p_lo))
                          + w * min(1, u / (1 - p_hi))``

        evaluated at ``u = s``, where ``w`` is the bracket weight
        (chosen so that the BiTVaR prices ``min(X, a)`` to ``premium``).

        Returns
        -------
        DataFrame
            shape ``(n_s, 4)`` with columns:

            ============  ==========================================
            ``s``         the evaluation point (== :attr:`s_grid`)
            ``p_lo``      lower TVaR threshold of the active bracket
            ``p_hi``      upper TVaR threshold of the active bracket
            ``weight``    convex-combo weight on the upper threshold
            ============  ==========================================

        Notes
        -----
        ``p_lo`` and ``p_hi`` are the *labels* of the cloud_df column
        achieving the minimum at ``s`` — they're values from
        :attr:`p_knots`, not computed extrema. The weight is looked up in
        :attr:`weight_df`. To rebuild the active BiTVaR as a
        :class:`Distortion`, call ``self.distortion(p_lo, p_hi)``.

        Useful as the data artifact behind the "min envelope as a
        weighted TVaR" paper extension: the envelope is the lower
        boundary of the set of BiTVaRs pricing to ``premium``, and this
        table tells you *which* BiTVaR is binding at each point.
        """
        argmin = self.cloud_df.idxmin(axis=1)
        df = pd.DataFrame(argmin.tolist(), columns=['p_lo', 'p_hi'])
        df.insert(0, 's', argmin.index.values)
        df['weight'] = self.weight_df.loc[
            list(zip(df['p_lo'], df['p_hi'])), 'weight'].values
        return df

    # ------------------------------------------------------------------
    # Convenience views
    # ------------------------------------------------------------------

    @cached_property
    def tvar_df(self):
        """``DataFrame`` with index ``p_knots`` and column ``tvar``."""
        return pd.DataFrame({'tvar': self.tvar_x_p}, index=self.p_knots) \
                 .rename_axis('p')

    def distortion(self, pl, pu):
        """Return the BiTVaR with knots ``(pl, pu)`` and the matching weight."""
        if (pl, pu) not in self.weight_df.index:
            raise KeyError(f'({pl}, {pu}) not in weight_df index — '
                           'must be one of the bracketing pairs')
        w = self.weight_df.at[(pl, pu), 'weight']
        return Distortion('bitvar', w, df=[pl, pu])

    def __repr__(self):
        return (f'Bounds({self.name!r}, premium={self.premium:.6g}, '
                f'a={self.a}, p_star={self.p_star:.6g})')

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_envelope(self, *, axs=None, n_resamples=0, alpha=0.05,
                      distortions='ordered', title='',
                      lim=(-0.025, 1.025)):
        """
        Three-panel envelope figure (formerly ``cloud_view``).

        Panel 1: scatter of sampled cloud columns shaded by weight, plus the
        min/max envelope band.
        Panels 2-3: the calibrated distortions overlaid on the envelope band.

        Parameters
        ----------
        axs : array of 3 Axes, optional
            If omitted, a new ``1 x 3`` figure is created.
        n_resamples : int, default 0
            If positive, draw this many bracket columns from ``cloud_df``,
            restricted to ``p_lo == 0`` (pricing distortions, those that pin
            the mean), and overplot them coloured by weight.
        alpha : float, default 0.05
            Opacity of the resampled curves.
        distortions : ``'ordered'``, list of dict, or ``'space'``
            What to overlay in panels 2-3. ``'ordered'`` only works for
            ``Portfolio`` objects with calibrated distortions.
        title : str, default ``''``
            Suptitle (applied to all panels).
        lim : tuple, default ``(-0.025, 1.025)``
            x and y axis limits.

        Returns
        -------
        fig, axs : matplotlib figure and array of three Axes.
        """
        if axs is None:
            fig, axs = plt.subplots(1, 3, figsize=(3 * FIG_W, FIG_W),
                                    constrained_layout=True, squeeze=False)
            axs = axs[0]
        else:
            axs = np.atleast_1d(axs).flatten()
            fig = axs[0].get_figure()

        norm = mpl.colors.Normalize(0, 1)
        cm = mpl.cm.ScalarMappable(norm=norm, cmap='viridis_r')
        mapper = cm.get_cmap()
        s_eval = np.linspace(0, 1, 1001)

        def _band(ax):
            ax.fill_between(self.cloud_df.index, self.cloud_df.min(1),
                            self.cloud_df.max(1), facecolor='C7', alpha=.15)
            self.cloud_df.min(1).plot(ax=ax, label='_nolegend_', lw=1, c='k')
            self.cloud_df.max(1).plot(ax=ax, label='_nolegend_', lw=1, c='k')

        if distortions == 'ordered':
            from .portfolio import Portfolio
            if not isinstance(self._obj, Portfolio):
                raise ValueError("distortions='ordered' requires a Portfolio")
            distortions = [
                {k: self._obj.distortions[k] for k in ['ccoc', 'tvar']},
                {k: self._obj.distortions[k] for k in ['ph', 'wang', 'dual']},
            ]

        ax = axs[0]
        if n_resamples > 0:
            bit = self.weight_df.xs(0, drop_level=False) \
                                .sample(n=n_resamples, replace=True) \
                                .reset_index()
            for _, row in bit.iterrows():
                pl, pu = row['p_lower'], row['p_upper']
                w = row['weight']
                self.cloud_df[(pl, pu)].plot(ax=ax, lw=1, c=mapper(w),
                                             alpha=alpha, label=None)
            fig.colorbar(cm, ax=ax, shrink=.5, aspect=16,
                         label='Weight to upper threshold')
        _band(ax)
        ax.plot([0, 1], [0, 1], c='k', lw=.25)
        ax.set(xlim=lim, ylim=lim, aspect='equal')

        if isinstance(distortions, dict):
            distortions = [distortions]
        if isinstance(distortions, list):
            name_mapper = {'ccoc': 'CCoC', 'tvar': 'TVaR(p*)',
                           'ph': 'PH', 'wang': 'Wang', 'dual': 'Dual'}
            ls_cycle = list(mpl.lines.lineStyles.keys())
            for ax, dist_dict in zip(axs[1:], distortions):
                lssi = iter(cycle(ls_cycle))
                for k, d in dist_dict.items():
                    ax.plot(s_eval, d.g(s_eval), lw=1, ls=next(lssi),
                            label=name_mapper.get(k, k))
                _band(ax)
                ax.plot([0, 1], [0, 1], c='k', lw=.25)
                ax.legend(loc='lower right', ncol=3, fontsize='large')
                ax.set(xlim=lim, ylim=lim, aspect='equal')
            # Average extreme overlay on the last panel
            self.cloud_df.mean(1).plot(ax=axs[-1], c=f'C{len(distortions[-1])}',
                                        ls='-.', lw=.5, label='Avg extreme')

        if title:
            for ax in axs:
                ax.set(title=title)

        return fig, axs

    def plot_weights(self, ax=None, *, levels=20, colorbar=True):
        """
        Contour plot of the bracketing weight as a function of ``(p_lo, p_hi)``.

        Parameters
        ----------
        ax : Axes, optional
            Target axes; created if omitted.
        levels : int, default 20
            Contour levels.
        colorbar : bool, default True
            Attach a colorbar.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(FIG_W, FIG_W),
                                 constrained_layout=True)
        bit = self.weight_df['weight'].unstack()
        img = ax.contourf(bit.columns, bit.index, bit,
                          cmap='viridis_r', levels=levels)
        ax.set(xlabel='p_upper', ylabel='p_lower',
               title='Weight for p_upper', aspect='equal')
        if colorbar:
            ax.get_figure().colorbar(img, ax=ax, shrink=.5, aspect=16,
                                     label='Weight to p_upper')
        return ax
