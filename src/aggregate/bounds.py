"""
Pricing-bounds analysis.

Two related classes, two related papers:

- :class:`Bounds` (Mildenhall, IME 2022) is constructed in one shot from a
  distribution and a target premium. It computes the bounding pricing
  distortions consistent with the premium, exposes the min/max envelope of
  that family, and renders the three-panel "cloud" figure from the paper.

- :class:`AllocationBounds` (similar-risks paper) takes the next step: given
  that the total is priced to P, what is the range of *natural allocation*
  premiums to each unit of a Portfolio? It slices the convex hull of the
  exact ``(TVaR_p(X), a_i(p))`` curve at ``T = P``.

How they differ
---------------

==============  ===================================  ===================================
Aspect          ``Bounds``                           ``AllocationBounds``
==============  ===================================  ===================================
Object of       the *distortion family* G_P:         the *allocation image* of G_P:
study           envelopes of g in (s, g(s)) space    ranges of NA premium by unit
Input           Portfolio, Aggregate, or pmf         Portfolio only (needs ``exeqa_*``)
                Series (needs ``tvar``, ``cdf``)
Premium         baked in at construction             argument at call time; hulls are
                                                     P-independent
Machinery       p-knot grid + brentq root find,      CDF-breakpoint vertices and
                approximate                          monotone-chain hulls, exact for
                                                     the discretized distribution
``p_star``      cached property at the fixed         ``p_star(P)`` method, exact
                premium, generic root find           per-atom inversion
==============  ===================================  ===================================

Both parameterize the extreme consistent distortions as biTVaRs
``(1 - w1) TVaR_{p0} + w1 TVaR_{p1}`` with ``p0 <= p_star <= p1``.

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

__all__ = ['AllocationBounds', 'Bounds']


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
    n_p : int, default ``256``
        Base p-grid size. Adaptive refinement adds a handful of knots
        around ``p_star``. Base excludes endpoint so use power of 2.
    n_s : int, default ``513``
        s-grid size. Binary, ``np.linspace(0, 1, n_s)`` includes
        endpoint so power of 2 plus 1.

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
                 n_p=256, n_s=513):
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
        through :func:`aggregate.spectral.convex_distortion`, which fits a
        piecewise-linear weighted-TVaR to the (s, gs) sample points.
        """
        from .spectral import convex_distortion
        s = self.s_grid
        g = self.cloud_df.min(axis=1).values
        return convex_distortion(
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
        return Distortion('bitvar', p0=pl, p1=pu, w1=w)

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


def _monotone_hull(t, y, side):
    """
    Indices of the lower or upper convex envelope of points sorted by t.

    Parameters
    ----------
    t : ndarray
        Strictly increasing x-coordinates (the TVaR values).
    y : ndarray
        y-coordinates (the unit allocations).
    side : {'lower', 'upper'}
        Which envelope to build.

    Returns
    -------
    ndarray of int
        Indices into ``t``/``y`` of the envelope vertices, left to right.

    Notes
    -----
    Andrew's monotone-chain algorithm, single left-to-right pass.  For the
    lower hull pop the middle point while the turn o -> a -> b is clockwise
    or collinear (cross <= 0); for the upper hull pop while counterclockwise
    or collinear (cross >= 0).  Exactly-collinear interior vertices are
    removed; near-collinear vertices are kept as short edges, which is
    harmless (the envelope is unchanged) and keeps the achieving bitvar
    report local.  No tolerance machinery is needed because the input
    points are exact vertices of a piecewise-linear curve.
    """
    sgn = 1.0 if side == 'lower' else -1.0
    idx = []
    for i in range(len(t)):
        while len(idx) >= 2:
            o, a = idx[-2], idx[-1]
            cross = (t[a] - t[o]) * (y[i] - y[o]) - (y[a] - y[o]) * (t[i] - t[o])
            if sgn * cross <= 0.0:
                idx.pop()
            else:
                break
        idx.append(i)
    return np.asarray(idx, dtype=np.intp)


class AllocationBounds:
    """
    Natural-allocation premium ranges consistent with a total premium.

    Given a :class:`~aggregate.portfolio.Portfolio` with total
    ``X = sum_i X_i``, determine for each unit the range of natural
    allocation (NA) premiums over all distortions g that price the total to
    P, ``G_P = { g : rho_g(X) = P }`` — the allocation counterpart of
    :class:`Bounds` (see the module docstring for the comparison).

    Build once from a Portfolio; thereafter :meth:`bounds`, :meth:`bitvars`,
    etc. evaluate for any target premium P by slicing precomputed convex
    envelopes — the P-dependence is a cheap piecewise-linear lookup.

    Parameters
    ----------
    port : Portfolio
        Must be updated (``density_df`` with ``exeqa_*`` columns available).
    units : list of str, optional
        Unit (line) names to include.  Default: all of ``port.line_names``.
    s_floor : float, default 1e-14
        Drop curve vertices with tail probability ``S < s_floor``.  Deep in
        the tail both the tail sums and ``exeqa`` are dominated by FFT noise
        (and ``add_exa`` zeroes ``exeqa`` below its own cut), so conditional
        expectations there are unreliable.  Truncation shrinks the feasible
        premium range upper end from ``ess sup X`` to ``T(1 - s_floor)``;
        set ``s_floor=0`` to keep everything.

    Attributes
    ----------
    curve_df : DataFrame
        Vertex table indexed by ``p`` with columns ``exeqa_total`` (=
        ``TVaR_p(X)``) and ``exeqa_<unit>`` (= NA to the unit at that p).
        Exact at every index value; the underlying curve is linear in the
        (T, a_i) plane between consecutive rows.
    premium_range : tuple of float
        ``(E[X], T_max)`` — the feasible P values.
    units : list of str
        Included unit names.
    additivity_error : float
        ``max_p |sum_i a_i(p) - T(p)|`` over the kept vertices.  Inherited
        from ``density_df`` (``add_exa`` zeroes ``exeqa`` where ``p_total``
        is below its cut, and ``exeqa`` carries FFT noise), so it bounds the
        absolute accuracy of the allocation bounds — typically ~1e-6
        relative, concentrated in the deep tail.

    Notes
    -----
    **Theory** (similar-risks paper).  Represent a distortion by its Kusuoka
    measure ``mu``, so ``rho_mu(X) = int TVaR_p(X) mu(dp)`` and the NA to
    unit i is ``A_i(mu) = int a_i(p) mu(dp)`` where ``a_i(p)`` is the
    TVaR_p natural allocation.  The pricing constraint is one affine
    constraint on probability measures, so the extreme consistent measures
    are two-point TVaR mixtures — *biTVaRs* ``(p0, p1, w1)`` with
    ``TVaR_{p0}(X) <= P <= TVaR_{p1}(X)`` and weight ``w1`` on ``p1``
    chosen to hit P exactly.  Every scalar extremum ``min/max A_i`` (indeed
    any linear allocation score) is attained at a biTVaR.  The biTVaR
    allocation is the height at ``T = P`` of the chord joining
    ``(T(p0), a_i(p0))`` and ``(T(p1), a_i(p1))`` on the parametric curve
    ``C_i = {(T(p), a_i(p))}``, hence

        [min A_i, max A_i] = vertical slice at T = P through the convex
        hull of C_i,

    lower envelope giving the minimum and upper the maximum, with the hull
    edge crossing ``T = P`` identifying the achieving biTVaR.

    **Exactness.**  For the discrete (FFT-grid) distribution the curve is
    exactly piecewise linear with kinks only at CDF breakpoints: within the
    atom at ``x_k``, writing ``u = 1/(1 - p)``,

        ``T(p) = x_k + (A_k - S_k x_k) u``,
        ``a_i(p) = kappa_ik + (A_ik - S_k kappa_ik) u``,

    (``S_k = Pr(X > x_k)``, ``kappa_ij = E[X_i | X = x_j]`` = ``exeqa``,
    ``A_k, A_ik`` strict-tail sums) — both coordinates affine in the *same*
    parameter u, so eliminating p leaves a straight segment: the 1/(1-p)
    nonlinearity is common to both axes and cancels.  The hull of the curve
    therefore equals the hull of its breakpoint vertices,

        vertex m = ( E[X | X >= x_m], E[X_i | X >= x_m] ),  p_m = F_{m-1},

    plain conditional tail expectations computed with reverse cumulative
    sums.  No p-grid, no interpolation error.  Vertices arrive sorted by T,
    so each envelope is one monotone-chain pass, O(n).

    All computations use the *linear* natural allocation
    ``sum_x kappa_i(x) Delta g(S(x))``; the total is unbounded (no asset
    cap).

    Examples
    --------
    ::

        from aggregate.bounds import AllocationBounds
        ab = AllocationBounds(port)        # build once: vertices + hulls
        ab.curve_df                        # the (p, T(p), a_i(p)) vertex table
        ab.bounds([1200, 1300])            # tidy (P, unit) lower/upper frame
        ab.bitvars(1200)                   # achieving biTVaRs (p0, p1, w1)
        ab.p_star(1200)                    # p with TVaR_p(X) = P (exact)
        ab.distortion(1200, 'A', 'upper')  # the achieving Distortion object
        ab.check(1200)                     # audit: reprice with each biTVaR
        ab.plot(P=1200)                    # curves, hulls, slice
    """

    def __init__(self, port, units=None, s_floor=1e-14):
        self.port = port
        self.s_floor = float(s_floor)

        if units is None:
            units = list(port.line_names)
        self.units = list(units)

        df = getattr(port, 'density_df', None)
        if df is None:
            raise ValueError(f'Portfolio {port.name!r} has no density_df; call update() first.')
        kcols = [f'exeqa_{u}' for u in self.units]
        missing = [c for c in kcols if c not in df.columns]
        if missing:
            raise ValueError(f'density_df missing columns {missing}; check unit names {self.units}.')

        # ------------------------------------------------------------------
        # Extract the discrete distribution: outcomes x, masses prob, and the
        # conditional allocations kappa_i(x) = E[X_i | X = x] = exeqa_i.
        # ------------------------------------------------------------------
        x = df.index.to_numpy(dtype=float)
        prob = df['p_total'].to_numpy(dtype=float)
        # kappa matrix: first column is the total (kappa_total(x) = x), so the
        # total's "allocation" IS TVaR and row sums give a built-in audit.
        kappa = np.column_stack([x, df[kcols].to_numpy(dtype=float)])

        # FFT densities carry tiny negative noise; tolerate at machine scale
        # only (tight threshold by project convention), error on real mass.
        tol = 64 * np.finfo(float).eps * max(1.0, np.abs(prob).sum())
        if np.any(prob < -tol):
            raise ValueError('p_total has negative mass beyond floating-point tolerance.')
        prob = np.where(np.abs(prob) <= tol, 0.0, prob)

        keep = prob > 0.0
        if not keep.any():
            raise ValueError('p_total has no positive mass.')
        x, prob, kappa = x[keep], prob[keep], kappa[keep]

        # Normalize so the masses sum to exactly 1 (FFT total is ~1 + eps).
        prob = prob / prob.sum()

        # ------------------------------------------------------------------
        # Curve vertices.  Vertex m corresponds to p_m = F_{m-1} (so m = 0 is
        # p = 0) and carries the conditional tail expectations
        #     T_m = E[X | X >= x_m],   a_im = E[X_i | X >= x_m].
        # Reverse cumsums accumulate from the tail end, which is the accurate
        # direction for tail quantities (no 1 - F cancellation).
        # ------------------------------------------------------------------
        S = np.cumsum(prob[::-1])[::-1]                       # S_m = Pr(X >= x_m)
        T = np.cumsum((prob * x)[::-1])[::-1] / S             # E[X | X >= x_m]
        A = np.cumsum((prob[:, None] * kappa)[::-1], axis=0)[::-1] / S[:, None]
        F = np.cumsum(prob)
        p_vert = np.concatenate([[0.0], F[:-1]])              # p at vertex m

        # Drop noise-dominated deep-tail vertices (see s_floor docstring).
        ok = S >= self.s_floor
        if not ok.all():
            logger.info('AllocationBounds: dropping %d tail vertices with S < %g',
                        (~ok).sum(), self.s_floor)

        # T must be strictly increasing for hulling and slicing.  It is in
        # exact arithmetic (conditional tail expectations increase); guard
        # against floating-point wobble in the deep tail by keeping only
        # vertices that strictly advance T.
        Tm = np.where(ok, T, -np.inf)
        runmax = np.maximum.accumulate(np.concatenate([[-np.inf], Tm[:-1]]))
        ok &= Tm > runmax

        if ok.sum() < 2:
            raise ValueError('Fewer than two usable curve vertices; '
                             'total is degenerate or s_floor too aggressive.')

        self._x = x
        self._prob = prob
        self._kappa = kappa                                   # (n, 1 + n_units)
        self._S = S[ok]                                       # = 1 - p, exact
        self._T = T[ok]
        self._A = A[ok]
        self._p_vert = p_vert[ok]
        # Map kept-vertex position -> original row, for p_star atom lookup.
        self._row = np.flatnonzero(ok)

        self.curve_df = pd.DataFrame(
            self._A,
            index=pd.Index(self._p_vert, name='p'),
            columns=['exeqa_total'] + kcols)

        # ------------------------------------------------------------------
        # Per-unit envelopes: indices into the vertex arrays.  P-independent,
        # so all later queries are O(log n) slices.
        # ------------------------------------------------------------------
        self._hulls = {}
        for j, u in enumerate(self.units):
            y = self._A[:, j + 1]                             # col 0 is total
            self._hulls[u] = {'lower': _monotone_hull(self._T, y, 'lower'),
                              'upper': _monotone_hull(self._T, y, 'upper')}

        self.premium_range = (float(self._T[0]), float(self._T[-1]))

        # Diagnostic: sum of unit allocations should equal T at every vertex.
        # Any residual is inherited from density_df; see class docstring.
        self.additivity_error = float(
            np.abs(self._A[:, 1:].sum(axis=1) - self._T).max())

    # ----------------------------------------------------------------------
    # Representation
    # ----------------------------------------------------------------------

    def __repr__(self):
        lo, hi = self.premium_range
        return (f'AllocationBounds({self.port.name!r}, units={self.units}, '
                f'{len(self._T)} vertices, premium range [{lo:.6g}, {hi:.6g}], '
                f'additivity error {self.additivity_error:.3g})')

    # ----------------------------------------------------------------------
    # p_star — exact inversion of TVaR_p(X) = P on the grid
    # ----------------------------------------------------------------------

    def p_star(self, P):
        """
        The p with ``TVaR_p(X) = P``, exact for the discrete distribution.

        Compare :attr:`Bounds.p_star`, the generic root-found counterpart
        (premium fixed at construction, any input object).

        Parameters
        ----------
        P : float
            Target premium, within :attr:`premium_range`.

        Returns
        -------
        float

        Notes
        -----
        Locate the bracketing vertices ``T_m <= P <= T_{m+1}``; within that
        atom ``TVaR_p = x_m + (A - S x_m)/(1 - p)`` with
        ``A = S_{m+1} T_{m+1}`` and ``S = S_{m+1}`` the strict-tail mass, so

            1 - p* = S_{m+1} (T_{m+1} - x_m) / (P - x_m).
        """
        P = float(P)
        self._validate_P(np.array([P]))
        m = int(np.searchsorted(self._T, P))
        if self._T[m] == P:
            return float(self._p_vert[m])
        # P strictly between T[m-1] and T[m]: inside the atom at x of row m-1.
        xk = self._x[self._row[m - 1]]
        s = self._S[m] * (self._T[m] - xk) / (P - xk)
        return float(1.0 - s)

    # ----------------------------------------------------------------------
    # Core slicing
    # ----------------------------------------------------------------------

    def _validate_P(self, P):
        """Raise if any target premium lies outside the feasible range."""
        lo, hi = self.premium_range
        bad = (P < lo) | (P > hi)
        if bad.any():
            raise ValueError(
                f'P={P[bad][:5]} outside feasible premium range [{lo:.6g}, {hi:.6g}] '
                f'(= [E[X], TVaR at s_floor truncation]).')

    def _slice(self, unit, side, P):
        """
        Evaluate one envelope at premiums P.

        Returns
        -------
        value, p0, p1, w1, i0, i1 : ndarrays
            Envelope height at each P; the achieving biTVaR (TVaR levels
            ``p0 < p1``, weight ``w1`` on ``p1``); and the kept-vertex
            indices of the edge endpoints (used internally for exact-S
            repricing).  When the slice lands exactly on a vertex the
            biTVaR is degenerate (``w1`` 0 or 1 — a pure TVaR).
        """
        h = self._hulls[unit][side]
        xh, yh = self._T[h], self._A[h, self.units.index(unit) + 1]
        # Edge e spans [xh[e], xh[e+1]]; right side puts P == xh[e] on the
        # edge to its left except at the first vertex; clip handles ends.
        e = np.clip(np.searchsorted(xh, P, side='right') - 1, 0, len(xh) - 2)
        x0, x1 = xh[e], xh[e + 1]
        w1 = (P - x0) / (x1 - x0)
        value = (1.0 - w1) * yh[e] + w1 * yh[e + 1]
        return value, self._p_vert[h[e]], self._p_vert[h[e + 1]], w1, h[e], h[e + 1]

    # ----------------------------------------------------------------------
    # Public queries
    # ----------------------------------------------------------------------

    def bounds(self, P):
        """
        Lower/upper natural-allocation bounds by unit at total premium P.

        Parameters
        ----------
        P : float or array_like
            Target total premium(s), each within :attr:`premium_range`.

        Returns
        -------
        DataFrame
            Indexed by ``(P, unit)`` with columns ``lower``, ``upper``,
            ``width``.  Within each P the lower (resp. upper) column sums to
            at most (at least) P; equality holds only when every unit's
            bound is achieved by the same biTVaR.
        """
        P = np.atleast_1d(np.asarray(P, dtype=float))
        self._validate_P(P)
        blocks = []
        for u in self.units:
            lo, *_ = self._slice(u, 'lower', P)
            hi, *_ = self._slice(u, 'upper', P)
            blocks.append(pd.DataFrame(
                {'lower': lo, 'upper': hi, 'width': hi - lo},
                index=pd.MultiIndex.from_arrays([P, [u] * len(P)], names=['P', 'unit'])))
        return pd.concat(blocks).sort_index()

    def bitvars(self, P):
        """
        Achieving biTVaRs for each unit's lower and upper bound at P.

        Parameters
        ----------
        P : float or array_like
            Target total premium(s).

        Returns
        -------
        DataFrame
            Indexed by ``(P, unit, bound)`` with columns ``value`` (the
            bound), ``p0``, ``p1``, ``w1`` — the biTVaR
            ``(1 - w1) TVaR_{p0} + w1 TVaR_{p1}``, weight on the upper level
            per the :class:`~aggregate.spectral.Distortion` convention.

        Notes
        -----
        When the achieving hull edge joins *adjacent* curve vertices the
        curve itself is on the envelope there, and the biTVaR is equivalent
        (for every unit's allocation simultaneously) to the pure
        ``TVaR_{p_star(P)}`` inside that atom — the two-point representation
        is then one of many optimizers.  Edges that skip vertices are
        genuine two-point biTVaRs.
        """
        P = np.atleast_1d(np.asarray(P, dtype=float))
        self._validate_P(P)
        recs = []
        for u in self.units:
            for side in ('lower', 'upper'):
                v, p0, p1, w1, *_ = self._slice(u, side, P)
                for i in range(len(P)):
                    recs.append((P[i], u, side, v[i], p0[i], p1[i], w1[i]))
        return (pd.DataFrame(recs, columns=['P', 'unit', 'bound', 'value', 'p0', 'p1', 'w1'])
                .set_index(['P', 'unit', 'bound'])
                .sort_index())

    def __call__(self, P):
        """Shorthand for :meth:`bounds`."""
        return self.bounds(P)

    def distortion(self, P, unit, bound):
        """
        The achieving distortion for one unit/bound at premium P.

        Parameters
        ----------
        P : float
        unit : str
        bound : {'lower', 'upper'}

        Returns
        -------
        Distortion
            ``Distortion('bitvar', p0, p1, w1)``, or ``Distortion('tvar', p)``
            when the optimizer is degenerate (slice exactly at a vertex).
        """
        v, p0, p1, w1, *_ = self._slice(unit, bound, np.array([float(P)]))
        p0, p1, w1 = float(p0[0]), float(p1[0]), float(w1[0])
        # Degenerate: slice landed on a vertex -> pure TVaR at that level.
        if w1 == 0.0 or p0 == p1:
            return Distortion('tvar', p=p0, display_name=f'TVaR({p0:.5g})')
        if w1 == 1.0:
            return Distortion('tvar', p=p1, display_name=f'TVaR({p1:.5g})')
        return Distortion('bitvar', p0=p0, p1=p1, w1=w1,
                          display_name=f'bitvar({p0:.5g}, {p1:.5g}; {w1:.4g})')

    # ----------------------------------------------------------------------
    # Audits
    # ----------------------------------------------------------------------

    def na_grid(self, p_grid):
        """
        TVaR and natural allocations at arbitrary p values (audit helper).

        Direct evaluation of ``T(p)`` and ``a_i(p)`` from the discrete
        distribution — independent of the vertex/hull machinery, so it
        cross-checks :attr:`curve_df` (at vertex p's) and interior linearity.

        Parameters
        ----------
        p_grid : array_like
            Values in [0, 1].

        Returns
        -------
        DataFrame
            Indexed by p with columns ``exeqa_total``, ``exeqa_<unit>``.

        Notes
        -----
        For p with VaR atom k (the smallest x with F(x) > p, located via
        ``searchsorted(F, p, side='right')``), the TVaR allocation is

            [ sum_{j>k} prob_j kappa_ij + (F_k - p) kappa_ik ] / (1 - p),

        and at p = 1 it is kappa_i at the largest positive-mass loss.

        .. warning:: Parametrizing by p is ill-conditioned near p = 1: the
           denominator ``1 - p`` suffers catastrophic cancellation once
           ``1 - p`` approaches the cumulative rounding error of F (~1e-11
           on a 2**16 grid), so values there diverge from :attr:`curve_df`
           like noise/(1-p).  ``curve_df``, built from reverse cumulative
           sums indexed by the outcome, is the accurate representation;
           use this method for auditing at moderate p only.
        """
        prob, kappa = self._prob, self._kappa
        F = np.cumsum(prob)
        F[-1] = 1.0
        # Tail sums excluding the current row: sum_{j>k} prob_j kappa_ij.
        pk = prob[:, None] * kappa
        tail_excl = np.vstack([np.cumsum(pk[::-1], axis=0)[::-1][1:],
                               np.zeros((1, kappa.shape[1]))])

        p = np.asarray(p_grid, dtype=float)
        if np.any((p < 0) | (p > 1)):
            raise ValueError('p_grid values must lie in [0, 1].')

        out = np.empty((len(p), kappa.shape[1]))
        at_one = p == 1.0
        pp = p[~at_one]
        if pp.size:
            k = np.searchsorted(F, pp, side='right')
            atom = np.maximum(F[k] - pp, 0.0)               # mass of the split atom
            out[~at_one] = (tail_excl[k] + atom[:, None] * kappa[k]) / (1.0 - pp)[:, None]
        out[at_one] = kappa[-1]                              # TVaR_1 = ess sup

        return pd.DataFrame(out, index=pd.Index(p, name='p'),
                            columns=self.curve_df.columns)

    def check(self, P):
        """
        Audit the bounds at P by direct repricing with the achieving biTVaRs.

        For each unit and bound, take the achieving biTVaR and compute its
        linear natural allocation ``sum_x kappa_i(x) Delta g(S(x))`` from
        first principles, then compare to the hull-slice value.

        Parameters
        ----------
        P : float

        Returns
        -------
        DataFrame
            Indexed by ``(unit, bound)`` with columns ``value`` (hull),
            ``repriced`` (direct), ``err`` (difference), ``total``
            (distortion price of X — should equal P), ``total_err``.

        Notes
        -----
        Evaluates the biTVaR hinge ``g(s) = (1-w1) min(1, s/s0) +
        w1 min(1, s/s1)`` in closed form using the *exact vertex tail
        probabilities* ``s_m = Pr(X > x_{m-1}) = 1 - p_m`` from the reverse
        cumsum — both more accurate than ``1 - p`` (which cancels
        catastrophically for p near 1) and sharper than ``Distortion.g``
        (interp1d-backed for bitvar, smearing the kink at tiny ``s1``).
        """
        P = float(P)
        # Strict-tail survival S(x_j) = Pr(X > x_j) at every kept outcome.
        Sx = np.append(np.cumsum(self._prob[::-1])[::-1][1:], 0.0)

        def hinge(s, s_level):
            # TVaR distortion with 1 - p = s_level; s_level = 0 is the max
            # distortion 1_{s > 0}.
            if s_level <= 0.0:
                return (s > 0).astype(float)
            return np.minimum(1.0, s / s_level)

        recs = []
        for u in self.units:
            for side in ('lower', 'upper'):
                v, p0, p1, w1, i0, i1 = self._slice(u, side, np.array([P]))
                v, w1, i0, i1 = float(v[0]), float(w1[0]), int(i0[0]), int(i1[0])
                gS = ((1.0 - w1) * hinge(Sx, self._S[i0])
                      + w1 * hinge(Sx, self._S[i1]))
                # Risk-adjusted mass on each outcome: Delta g(S) telescopes
                # from g(S(-inf)) = g(1) = 1 down the tail.
                gp = -np.diff(gS, prepend=1.0)
                j = self.units.index(u) + 1                  # col 0 = total
                repriced = float(gp @ self._kappa[:, j])
                total = float(gp @ self._x)
                recs.append((u, side, v, repriced, repriced - v,
                             total, total - P))
        return (pd.DataFrame(recs, columns=['unit', 'bound', 'value', 'repriced',
                                            'err', 'total', 'total_err'])
                .set_index(['unit', 'bound']))

    # ----------------------------------------------------------------------
    # Plotting
    # ----------------------------------------------------------------------

    def plot(self, units=None, P=None, axs=None, max_t=None):
        """
        Plot each unit's allocation curve, envelopes, and optional P-slice.

        Parameters
        ----------
        units : list of str, optional
            Default: all units.
        P : float, optional
            Draw the vertical slice at T = P and mark the bounds.
        axs : array of Axes, optional
            One per unit; created if omitted.
        max_t : float, optional
            Truncate the T axis (the far tail compresses the picture).

        Returns
        -------
        array of Axes
        """
        if units is None:
            units = self.units
        if axs is None:
            n = len(units)
            ncols = min(n, 3)
            nrows = -(-n // ncols)
            fig, axs = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.8 * nrows),
                                    constrained_layout=True, squeeze=False)
            axs = axs.flat
        for ax, u in zip(axs, units):
            j = self.units.index(u) + 1
            t, y = self._T, self._A[:, j]
            if max_t is not None:
                mask = t <= max_t
                t, y = t[mask], y[mask]
            ax.plot(t, y, lw=0.75, c='C0', label=r'$a_i(p)$ vs $T(p)$')
            for side, c in (('lower', 'C2'), ('upper', 'C3')):
                h = self._hulls[u][side]
                th, yh = self._T[h], self._A[h, j]
                if max_t is not None:
                    m = th <= max_t
                    th, yh = th[m], yh[m]
                ax.plot(th, yh, lw=1.25, c=c, ls='--', label=side)
            if P is not None:
                lo, *_ = self._slice(u, 'lower', np.array([float(P)]))
                hi, *_ = self._slice(u, 'upper', np.array([float(P)]))
                ax.axvline(P, lw=0.5, c='k')
                ax.plot([P, P], [lo[0], hi[0]], lw=2.5, c='k', solid_capstyle='butt')
                ax.plot([P, P], [lo[0], hi[0]], 'o', ms=4, c='k')
            ax.set(title=u, xlabel='$T(p)$ = total premium', ylabel='NA premium')
            ax.legend(fontsize='x-small')
        return axs
