"""
Pricing-bounds analysis.

Three related classes, two related papers:

- :class:`Bounds` (Mildenhall, IME 2022) is constructed in one shot from a
  distribution and a target premium. It computes the bounding pricing
  distortions consistent with the premium, exposes the min/max envelope of
  that family, and renders the three-panel "cloud" figure from the paper.

- :class:`AllocationBounds` (similar-risks paper) takes the next step: given
  that the total is priced to P, what is the range of *natural allocation*
  premiums to each unit of a Portfolio? It slices the convex hull of the
  exact ``(TVaR_p(X), a_i(p))`` curve at ``T = P``.

- :class:`PricingBounds` (similar-risks paper) generalizes the allocation
  question to *any* second risk: given the reference risk X is priced to P,
  what is the range of the price of another risk Y under the same family of
  consistent distortions? It slices the convex hull of the exact
  ``(TVaR_p(X), TVaR_p(Y))`` curve at ``T_X = P``.  The special case
  ``X = U[0, 1]`` is the *Gini lens*: ``TVaR_p(X) = (1 + p) / 2`` is affine in
  p, so the pricing constraint collapses to a mean-Kusuoka-level condition
  ``E_mu[p] = 2P - 1`` and the bounds read off the envelope gap of Y's own
  TVaR curve.

How they differ
---------------

.. list-table:: Bounds vs AllocationBounds vs PricingBounds
   :header-rows: 1
   :widths: 14 28 29 29

   * - Aspect
     - ``Bounds``
     - ``AllocationBounds``
     - ``PricingBounds``
   * - Object of study
     - the distortion family G_P (envelopes of g)
     - the allocation image of G_P (NA premium by unit)
     - the price image of G_P (price of another risk Y)
   * - Input
     - Portfolio, Aggregate, or pmf Series
     - Portfolio only (needs ``exeqa_*``)
     - two risks X, Y (or callable TVaR sources)
   * - Premium
     - baked in at construction
     - argument at call time; hulls P-independent
     - argument at call time; hulls P-independent
   * - Machinery
     - p-knot grid + brentq, approximate
     - CDF-breakpoint vertices + monotone hulls, exact
     - union-of-breakpoints vertices + hulls, exact
   * - ``p_star``
     - cached property, generic root find
     - ``p_star(P)`` exact per-atom inversion
     - ``p_star(P)`` via the X-source inverse

Both parameterize the extreme consistent distortions as biTVaRs
``(1 - w1) TVaR_{p0} + w1 TVaR_{p1}`` with ``p0 <= p_star <= p1``.

Naming convention used throughout
---------------------------------

===============  ==============  ============================================
Name             Shape           Meaning
===============  ==============  ============================================
``p_knots``      ``(n_p,)``      TVaR threshold values, the p axis
``s_grid``       ``(n_s,)``      distortion evaluation points, the s axis
``tvar_x_p``     ``(n_p,)``      ``tvar_x_p[i] = TVaR_{p_knots[i]}(min(X, a))``
``tvar_hinges``  ``(n_p, n_s)``  ``min(1, s_grid[j] / (1 - p_knots[i]))``
``cloud_df``     ``(n_s, K)``    each column is a convex combination of two
                                 rows of ``tvar_hinges``; ``K`` = number of
                                 ``(p_lo, p_hi)`` pairs straddling ``p_star``
===============  ==============  ============================================

The "hinge family" is the set of TVaR distortions parameterised by p:
``TVaR_p(s) = min(1, s / (1 - p))``. p indexes the family; s is the
distortion argument.
"""
from functools import cached_property
import logging

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import brentq

from ._help import HelpMixin
from .constants import INFO_NA, info_row
from .spectral import Distortion
from ._grid_distribution import GridDistribution

logger = logging.getLogger(__name__)

__all__ = ['AllocationBounds', 'Bounds', 'PricingBounds']


def _resolve_obj(obj, unit):
    """
    Coerce *obj* into ``(gd, name)`` where

    - ``gd`` is a :class:`~aggregate._grid_distribution.GridDistribution` over
      the risk ``X``. The (unbounded) ``TVaR_p(X)`` is ``gd.tvar(p)``, ``P(X <=
      x)`` is ``gd.cdf(x)``, and the capped ``TVaR_p(min(X, a))`` is
      ``gd.tvar_of_limited(p, a)`` (used by :meth:`Bounds._tvar_x_a`).
    - ``name`` is a display string.

    Accepted obj types: ``Portfolio``, ``Aggregate``, ``pd.Series``,
    ``pd.DataFrame``. For Series/DataFrame the index is interpreted as outcomes
    and values as the pmf; for DataFrame the first column is the pmf.

    For ``Aggregate`` / ``Portfolio`` the object's own ``GridDistribution`` view
    is reused (the same value type the risk measures already flow through).
    """
    # Local imports to keep this module decoupled at import time.
    from .distributions import Aggregate
    from .portfolio import Portfolio

    if isinstance(obj, Portfolio):
        if unit == 'total':
            return obj._grid_distribution(), f'{obj.name}.total'
        if unit not in obj.unit_names_ex:
            raise ValueError(f'unit {unit!r} not in portfolio {obj.name!r}')
        ag = getattr(obj, unit)
        return ag._grid_distribution(), f'{obj.name}.{unit}'

    if isinstance(obj, Aggregate):
        return obj._grid_distribution(), obj.name

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
    return GridDistribution.from_series(ser, name=str(name)), str(name)


class Bounds(HelpMixin):
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
    unit : str, default ``'total'``
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

    def __init__(self, obj, premium, *, a=np.inf, unit='total',
                 n_p=256, n_s=513):
        self._obj = obj
        self.premium = float(premium)
        self.a = float(a) if not np.isinf(a) else np.inf
        self.unit = unit
        self.n_p = int(n_p)
        self.n_s = int(n_s)

        gd, name = _resolve_obj(obj, unit)
        self._dist = gd
        self.name = name
        self.Fb = 1.0 if np.isinf(self.a) else float(gd.cdf(self.a))

        mean = float(gd.tvar(0))
        if self.premium < mean:
            raise ValueError(
                f'premium {self.premium} below mean {mean}; pricing bound undefined')
        if not np.isinf(self.a) and self.premium > self.a:
            raise ValueError(
                f'premium {self.premium} exceeds asset cap {self.a}')

    @property
    def info(self):
        """Fixed-layout multi-line summary string (terse).

        Every row is always present, in the same order, for every ``Bounds``;
        a value that does not apply renders as ``n/a``. Shares the label/value
        convention (:func:`aggregate.constants.info_row`) with ``Aggregate`` /
        ``Portfolio``. The row catalogue is documented in
        ``dev/info-strings.rst``. Deliberately cheap: it reports the grid
        sizes rather than materialising :attr:`cloud_df`.
        """
        mean = float(self._dist.tvar(0))
        rows = [
            ('bounds object name', self.name),
            ('kind', 'pricing bounds (IME 2022)'),
            ('unit', self.unit),
            ('E[X]', f'{mean:,.6g}'),
            ('premium', f'{self.premium:,.6g}'),
            ('margin', f'{self.premium - mean:,.6g}'),
            ('asset cap', f'{self.a:,.6g}' if np.isfinite(self.a) else 'unlimited'),
            ('F(a)', f'{self.Fb:.6g}'),
            ('p_star', f'{self.p_star:.6g}'),
            ('n_p', self.n_p),
            ('n_s', self.n_s),
        ]
        return '\n'.join(info_row(label, value) for label, value in rows)

    # ------------------------------------------------------------------
    # Bounded TVaR — TVaR_p(min(X, a))
    # ------------------------------------------------------------------

    def _tvar_x_a(self, p):
        """TVaR_p of min(X, a). Scalar or array p.

        Delegates to :meth:`GridDistribution.tvar_of_limited` -- the analytic
        composite ``TVaR_p(X) - (1-F(a))(TVaR_{F(a)}(X) - a)/(1-p)`` (for
        ``p < F(a)``, else ``a``) now lives in one place on the value type. O(1),
        no grid rebuild, so it stays cheap inside the ``p_star`` root-find.
        """
        return self._dist.tvar_of_limited(p, self.a)

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
            s, g, label=f'min env({self.name}, prem={self.premium:.4g})')

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
        ``s``::

            g_s(u) = (1 - w) * min(1, u / (1 - p_lo))
                        + w * min(1, u / (1 - p_hi))

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
        from .plots import plot_bounds_envelope
        return plot_bounds_envelope(self, axs=axs, n_resamples=n_resamples,
                                    alpha=alpha, distortions=distortions,
                                    title=title, lim=lim)

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
        from .plots import plot_bounds_weights
        return plot_bounds_weights(self, ax=ax, levels=levels, colorbar=colorbar)


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


def _bitvar_g(s, s0, s1, w1):
    """
    Cumulative biTVaR distortion ``g`` evaluated at survival probabilities ``s``.

    ``g(s) = (1 - w1) min(1, s / s0) + w1 min(1, s / s1)`` where ``s0 = 1 - p0``
    and ``s1 = 1 - p1`` are the tail levels of the two TVaR knots.  A level of
    zero is the maximal distortion ``1_{s > 0}`` (the TVaR-1 / essential-sup
    pricing measure).
    """
    def hinge(lvl):
        if lvl <= 0.0:
            return (np.asarray(s) > 0).astype(float)
        return np.minimum(1.0, np.asarray(s) / lvl)
    return (1.0 - w1) * hinge(s0) + w1 * hinge(s1)


def _extract_pmf(obj, a=np.inf):
    """
    Coerce *obj* into ``(x, prob, name)`` — a discrete distribution of
    ``min(obj, a)`` on a strictly increasing grid with positive masses
    summing to 1.

    Parameters
    ----------
    obj : Aggregate, Portfolio, pd.Series, or pd.DataFrame
        For ``Aggregate`` / ``Portfolio`` the total ``p_total`` pmf is used;
        for ``Series`` / ``DataFrame`` the index is outcomes and the values
        (first column for a frame) the pmf.
    a : float, default ``np.inf``
        Asset cap.  Mass at outcomes ``>= a`` collapses into a single atom at
        ``a`` — the distribution of ``min(obj, a)``.  Only the *distribution*
        is capped (no allocation bookkeeping), so this is an exact, cheap
        transform.

    Returns
    -------
    x : ndarray
    prob : ndarray
    name : str
    """
    from .distributions import Aggregate
    from .portfolio import Portfolio

    if isinstance(obj, (Aggregate, Portfolio)):
        ser = obj.density_df.query('p_total > 0').p_total
        name = obj.name
    elif isinstance(obj, pd.DataFrame):
        ser = obj.iloc[:, 0]
        ser = ser[ser > 0]
        name = str(obj.columns[0])
    elif isinstance(obj, pd.Series):
        ser = obj[obj > 0]
        name = str(ser.name) if ser.name is not None else 'series'
    else:
        raise TypeError(
            f'PricingBounds: unsupported source type {type(obj).__name__}. '
            'Accepted: Aggregate, Portfolio, pd.Series, pd.DataFrame, '
            "a TVaR source, or 'uniform'.")

    if not ser.index.is_unique:
        raise ValueError('pmf index must be unique')
    if not ser.index.is_monotonic_increasing:
        raise ValueError('pmf index must be monotonic increasing')

    x = ser.index.to_numpy(dtype=float)
    prob = ser.to_numpy(dtype=float)
    if np.isfinite(a):
        # Collapse the tail X >= a into one atom at a (distribution of min(X,a)).
        x = np.minimum(x, a)
        # Sum masses sharing the (now repeated) cap value; np.add.reduceat on
        # the unique grid keeps it O(n).
        uniq, inv = np.unique(x, return_inverse=True)
        pooled = np.zeros_like(uniq)
        np.add.at(pooled, inv, prob)
        x, prob = uniq, pooled
    prob = prob / prob.sum()
    return x, prob, name


class _TVaRSource:
    """
    Axis adapter for :class:`PricingBounds`: a thing that returns ``TVaR_p``.

    Two concrete kinds, both usable on either axis (so the uniform reference
    can be X *or* Y):

    - :class:`_RiskSource` wraps a discrete risk (Aggregate, Portfolio, pmf);
      ``tvar`` and ``breakpoints`` come from its pmf, and it can be repriced
      from first principles for :meth:`PricingBounds.check`.
    - :class:`_CallableSource` wraps a closed-form ``(T, T_inv)`` pair with no
      breakpoints — e.g. :func:`uniform_source`.
    """

    name = '?'
    breakpoints = None      # ndarray of interior p where VaR jumps, or None

    def tvar(self, p):
        """Vectorised ``TVaR_p`` at the (array of) probability level(s) ``p``."""
        raise NotImplementedError

    def tvar_inv(self, t):
        """The p with ``TVaR_p = t`` (scalar), or ``None`` if unavailable."""
        return None

    def reprice(self, g):
        """``rho_g`` via the survival hinge, or ``None`` if no pmf is held."""
        return None


class _RiskSource(_TVaRSource):
    """A :class:`_TVaRSource` backed by a discrete risk's pmf."""

    def __init__(self, x, prob, name):
        self.name = name
        self._x = np.asarray(x, dtype=float)
        self._prob = np.asarray(prob, dtype=float)
        F = np.cumsum(self._prob)
        F[-1] = 1.0
        self._F = F
        # Interior breakpoints: p where the VaR steps to the next atom.
        self.breakpoints = F[:-1]
        # var/tvar via the shared GridDistribution kernel (zero-mass atoms are
        # filtered internally but leave tvar unchanged -- a property of X).
        self._tvar = GridDistribution(self._x, self._prob).tvar
        # Own-grid vertices for exact tvar inversion (p_star).
        S = np.cumsum(self._prob[::-1])[::-1]            # S_m = Pr(X >= x_m)
        self._S_own = S
        self._T_own = np.cumsum((self._prob * self._x)[::-1])[::-1] / S
        self._pv_own = np.concatenate([[0.0], F[:-1]])

    def tvar(self, p):
        return np.asarray(self._tvar(np.asarray(p, dtype=float)), dtype=float)

    def tvar_inv(self, P):
        """
        Exact p with ``TVaR_p(X) = P`` from the own-grid vertices.

        Locate the bracketing vertices ``T_m <= P <= T_{m+1}``; within the
        atom at the VaR ``x_{m-1}`` the level solves
        ``1 - p = S_m (T_m - x_{m-1}) / (P - x_{m-1})``.
        """
        P = float(P)
        T = self._T_own
        m = int(np.searchsorted(T, P))
        if m >= len(T):
            return float(self._pv_own[-1])
        if T[m] == P:
            return float(self._pv_own[m])
        xk = self._x[m - 1]
        s = self._S_own[m] * (T[m] - xk) / (P - xk)
        return float(1.0 - s)

    def reprice(self, g):
        # rho_g(X) = sum_x x * Delta g(S(x)), S the strict survival function.
        Sx = np.append(np.cumsum(self._prob[::-1])[::-1][1:], 0.0)
        gp = -np.diff(g(Sx), prepend=1.0)
        return float(gp @ self._x)


class _CallableSource(_TVaRSource):
    """A :class:`_TVaRSource` backed by closed-form ``T(p)`` and ``T_inv(t)``."""

    def __init__(self, tvar, tvar_inv=None, name='callable', breakpoints=None):
        self._tvar = tvar
        self._tvar_inv = tvar_inv
        self.name = name
        self.breakpoints = breakpoints

    def tvar(self, p):
        return np.asarray(self._tvar(np.asarray(p, dtype=float)), dtype=float)

    def tvar_inv(self, t):
        return None if self._tvar_inv is None else float(self._tvar_inv(float(t)))


def uniform_source(name='U[0,1]'):
    """
    The standard-uniform TVaR source — the Gini reference.

    ``TVaR_p(U[0,1]) = (1 + p) / 2`` is affine in p, so its inverse is
    ``T_inv(t) = 2t - 1``.  Used as the X-source it turns the pricing
    constraint into the mean-Kusuoka-level condition ``E_mu[p] = 2P - 1``;
    used as a Y-source it reports a normalised dispersion (Gini) reading.
    """
    return _CallableSource(lambda p: (1.0 + p) / 2.0,
                           tvar_inv=lambda t: 2.0 * t - 1.0,
                           name=name)


def _coerce_source(obj, *, a=np.inf, name=None):
    """Coerce a user argument into a :class:`_TVaRSource`."""
    if isinstance(obj, _TVaRSource):
        return obj
    if isinstance(obj, str):
        if obj.lower() in ('uniform', 'unif', 'u'):
            return uniform_source()
        raise ValueError(f"unknown source string {obj!r}; did you mean 'uniform'?")
    x, prob, nm = _extract_pmf(obj, a)
    return _RiskSource(x, prob, name or nm)


class _HullEngine(HelpMixin):
    """
    Shared convex-hull / slice engine for :class:`AllocationBounds` and
    :class:`PricingBounds`.

    Both classes reduce to the same geometry: a parametric curve
    ``(T_X(p), y_j(p))`` per item ``j``, exactly piecewise linear in
    ``(T_X, y_j)`` with vertices at the CDF breakpoints, whose vertical slice
    at ``T_X = P`` through the convex hull gives ``[min, max]`` of the item's
    score over all distortions pricing X to P.  Subclasses build the vertex
    table and call :meth:`_init_engine`; everything below queries it.

    The vertex table is held as

    - ``_T`` : (n,) strictly increasing x-axis = ``TVaR_p(X)`` at the vertices,
    - ``_A`` : (n, 1 + k), column 0 == ``_T``, columns 1.. the k item curves,
    - ``_p_vert`` / ``_S`` : the vertex p-values and exact tail masses ``1 - p``,
    - ``_y_names`` : the k item (unit / risk) names,
    - ``_hulls`` : per item the lower/upper monotone-chain envelope indices.

    Subclasses provide :meth:`_invert_T` (the ``p_star`` inverse) and their own
    ``check`` / ``curve_df``.
    """

    _item_label = 'unit'
    _curve_label = r'$a_i(p)$ vs $T(p)$'
    _xlabel = '$T(p)$ = total premium'
    _ylabel = 'NA premium'

    def _hull_info_rows(self):
        """The shared ``info`` rows: the slice geometry both subclasses carry.

        Subclass ``info`` properties lead with their own identity rows and then
        splice these in, so the two objects read the same below the fold.
        """
        lo, hi = self.premium_range
        return [
            ('asset cap', f'{self.a:,.6g}' if np.isfinite(self.a) else 'unlimited'),
            (f'{self._item_label}s', ', '.join(self._y_names)),
            ('vertices', len(self._T)),
            ('premium range', f'[{lo:,.6g}, {hi:,.6g}]'),
            ('s_floor', f'{self.s_floor:.3g}'),
        ]

    def _init_engine(self, T, A, p_vert, S, y_names):
        """Store the vertex table and build the per-item convex envelopes."""
        self._T = np.asarray(T, dtype=float)
        self._A = np.asarray(A, dtype=float)
        self._p_vert = np.asarray(p_vert, dtype=float)
        self._S = np.asarray(S, dtype=float)
        self._y_names = list(y_names)
        self._hulls = {}
        for j, u in enumerate(self._y_names):
            y = self._A[:, j + 1]
            self._hulls[u] = {'lower': _monotone_hull(self._T, y, 'lower'),
                              'upper': _monotone_hull(self._T, y, 'upper')}
        self.premium_range = (float(self._T[0]), float(self._T[-1]))

    # ----------------------------------------------------------------------
    # p_star — exact inversion of TVaR_p(X) = P
    # ----------------------------------------------------------------------

    def _invert_T(self, P):
        """Return the p with ``TVaR_p(X) = P`` (subclass hook)."""
        raise NotImplementedError

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
        """
        P = float(P)
        self._validate_P(np.array([P]))
        return self._invert_T(P)

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

    def _slice(self, name, side, P):
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
        h = self._hulls[name][side]
        j = self._y_names.index(name) + 1
        xh, yh = self._T[h], self._A[h, j]
        # Edge e spans [xh[e], xh[e+1]]; clip handles the endpoints.
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
        Lower/upper bounds by item at reference premium P.

        Parameters
        ----------
        P : float or array_like
            Target reference premium(s), each within :attr:`premium_range`.

        Returns
        -------
        DataFrame
            Indexed by ``(P, item)`` with columns ``lower``, ``upper``,
            ``width``.
        """
        P = np.atleast_1d(np.asarray(P, dtype=float))
        self._validate_P(P)
        blocks = []
        for u in self._y_names:
            lo, *_ = self._slice(u, 'lower', P)
            hi, *_ = self._slice(u, 'upper', P)
            blocks.append(pd.DataFrame(
                {'lower': lo, 'upper': hi, 'width': hi - lo},
                index=pd.MultiIndex.from_arrays(
                    [P, [u] * len(P)], names=['P', self._item_label])))
        return pd.concat(blocks).sort_index()

    def bitvars(self, P):
        """
        Achieving biTVaRs for each item's lower and upper bound at P.

        Parameters
        ----------
        P : float or array_like
            Target reference premium(s).

        Returns
        -------
        DataFrame
            Indexed by ``(P, item, bound)`` with columns ``value`` (the
            bound), ``p0``, ``p1``, ``w1`` — the biTVaR
            ``(1 - w1) TVaR_{p0} + w1 TVaR_{p1}``, weight on the upper level
            per the :class:`~aggregate.spectral.Distortion` convention.

        Notes
        -----
        When the achieving hull edge joins *adjacent* curve vertices the curve
        itself is on the envelope there, and the biTVaR is equivalent to the
        pure ``TVaR_{p_star(P)}`` inside that atom — the two-point
        representation is then one of many optimizers.  Edges that skip
        vertices are genuine two-point biTVaRs.
        """
        P = np.atleast_1d(np.asarray(P, dtype=float))
        self._validate_P(P)
        recs = []
        for u in self._y_names:
            for side in ('lower', 'upper'):
                v, p0, p1, w1, *_ = self._slice(u, side, P)
                for i in range(len(P)):
                    recs.append((P[i], u, side, v[i], p0[i], p1[i], w1[i]))
        return (pd.DataFrame(recs, columns=['P', self._item_label, 'bound',
                                            'value', 'p0', 'p1', 'w1'])
                .set_index(['P', self._item_label, 'bound'])
                .sort_index())

    def __call__(self, P):
        """Shorthand for :meth:`bounds`."""
        return self.bounds(P)

    def distortion(self, P, item, bound):
        """
        The achieving distortion for one item/bound at premium P.

        Parameters
        ----------
        P : float
        item : str
            Unit (allocation) or risk (pricing) name.
        bound : {'lower', 'upper'}

        Returns
        -------
        Distortion
            ``Distortion('bitvar', p0, p1, w1)``, or ``Distortion('tvar', p)``
            when the optimizer is degenerate (slice exactly at a vertex).
        """
        v, p0, p1, w1, *_ = self._slice(item, bound, np.array([float(P)]))
        p0, p1, w1 = float(p0[0]), float(p1[0]), float(w1[0])
        # Degenerate: slice landed on a vertex -> pure TVaR at that level.
        if w1 == 0.0 or p0 == p1:
            return Distortion('tvar', p=p0, label=f'TVaR({p0:.5g})')
        if w1 == 1.0:
            return Distortion('tvar', p=p1, label=f'TVaR({p1:.5g})')
        return Distortion('bitvar', p0=p0, p1=p1, w1=w1,
                          label=f'bitvar({p0:.5g}, {p1:.5g}; {w1:.4g})')

    # ----------------------------------------------------------------------
    # Plotting
    # ----------------------------------------------------------------------

    def plot(self, items=None, P=None, axs=None, max_t=None):
        """
        Plot each item's curve, convex envelopes, and optional P-slice.

        Parameters
        ----------
        items : list of str, optional
            Default: all items.
        P : float, optional
            Draw the vertical slice at T = P and mark the bounds.
        axs : array of Axes, optional
            One per item; created if omitted.
        max_t : float, optional
            Truncate the T axis (the far tail compresses the picture).

        Returns
        -------
        array of Axes
        """
        from .plots import plot_hull_bounds
        return plot_hull_bounds(self, items=items, P=P, axs=axs, max_t=max_t)


class AllocationBounds(_HullEngine):
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
    a : float, default ``np.inf``
        Asset cap.  Finite ``a`` bounds the total at ``X ∧ a``: grid rows
        below ``a`` are kept as-is and all default states ``X >= a``
        collapse into a single atom at ``a`` carrying the *linear* natural
        allocation ``a · E[X_i/X | X >= a]`` (equal-priority proportional
        sharing; the lifted allocation is not offered).  The collapse is
        built here from the ``exi_xgta_*`` columns.  ``a`` is snapped to
        the loss grid.  Default ``np.inf`` is the unbounded total.
    units : list of str, optional
        Unit names to include.  Default: all of ``port.unit_names``.
    s_floor : float, default 1e-14
        Drop curve vertices with tail probability ``S < s_floor``.  Deep in
        the tail both the tail sums and ``exeqa`` are dominated by FFT noise
        (and ``add_exa`` zeroes ``exeqa`` below its own cut), so conditional
        expectations there are unreliable.  Truncation shrinks the feasible
        premium range upper end from ``ess sup X`` to ``T(1 - s_floor)``;
        set ``s_floor=0`` to keep everything.  Mostly moot when bounded:
        the collapsed default atom has macroscopic mass.

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
    ``sum_x kappa_i(x) Delta g(S(x))``.

    **Bounded totals.**  With finite ``a`` the input triple describes
    ``X ∧ a`` and the unit payouts ``X_i 1_{X<a} + a (X_i/X) 1_{X>=a}``;
    since ``sum_i X_i/X = 1`` the collapsed allocations sum to ``a`` by
    construction.  ``X ∧ a`` is just another discrete rv with its own
    kappa decomposition, so the exactness argument and all machinery apply
    verbatim; :attr:`premium_range` becomes ``[E[X ∧ a], a]`` and the
    range collapses to the default-state allocations at ``P = a``.  The
    p = 0 vertex equals ``(exa_total(a), exa_i(a))`` from ``density_df``
    (the PIR ``alpha S`` integral) — an independent cross-check.

    Examples
    --------
    ::

        from aggregate.bounds import AllocationBounds
        ab = AllocationBounds(port)        # build once: vertices + hulls
        ab = port.allocation_bounds(p=0.995)   # bounded at assets q(0.995)
        ab.curve_df                        # the (p, T(p), a_i(p)) vertex table
        ab.bounds([1200, 1300])            # tidy (P, unit) lower/upper frame
        ab.bitvars(1200)                   # achieving biTVaRs (p0, p1, w1)
        ab.p_star(1200)                    # p with TVaR_p(X) = P (exact)
        ab.distortion(1200, 'A', 'upper')  # the achieving Distortion object
        ab.check(1200)                     # audit: reprice with each biTVaR
        ab.plot(P=1200)                    # curves, hulls, slice
    """

    def __init__(self, port, *, a=np.inf, units=None, s_floor=1e-14):
        self.port = port
        self.s_floor = float(s_floor)
        self.a = float(a)

        if units is None:
            units = list(port.unit_names)
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
        # Bounded (finite a): X ∧ a with the default states X >= a collapsed
        # into one atom at a carrying the linear-NA allocations
        # a·E[X_i/X | X >= a] = a·exi_xgta_i(a - bs). The collapse forces
        # S(a) = 0, so the whole tail mass — including any pmf deficit —
        # lands in the atom. (Sole owner of this idiom since the linear
        # price engine moved onto the unified apply_distortion frame.)
        # ------------------------------------------------------------------
        if np.isfinite(self.a):
            self.a = float(port.snap(self.a))
            sub = df.loc[:self.a]
            x = sub.index.to_numpy(dtype=float)
            S_col = sub['S'].to_numpy(dtype=float).copy()
            S_col[-1] = 0.0
            prob = -np.diff(S_col, prepend=1.0)
            unit_kappa = sub[kcols].to_numpy(dtype=float).copy()
            # re-aim the atom only when real mass lies beyond a (not just
            # FFT leakage measured by the pmf deficit)
            if port.sf(self.a) > (1 - df.p_total.sum()):
                logger.info('Collapsing tail states: exeqa at the atom '
                            'becomes a * exi_xgta(a - bs)')
                acols = [f'exi_xgta_{u}' for u in self.units]
                unit_kappa[-1, :] = (
                    sub[acols].iloc[-2].to_numpy(dtype=float) * self.a)
        else:
            x = df.index.to_numpy(dtype=float)
            prob = df['p_total'].to_numpy(dtype=float)
            unit_kappa = df[kcols].to_numpy(dtype=float)
        # kappa matrix: first column is the total (kappa_total(x) = x — and
        # exactly a at the collapsed atom), so the total's "allocation" IS
        # TVaR and row sums give a built-in audit.
        kappa = np.column_stack([x, unit_kappa])

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
        # Map kept-vertex position -> original row, for p_star atom lookup.
        self._row = np.flatnonzero(ok)

        # Hand the vertex table to the shared engine (sets _T, _A, _p_vert,
        # _S, _y_names, _hulls, premium_range).  P-independent, so all later
        # queries are O(log n) slices.
        self._init_engine(T[ok], A[ok], p_vert[ok], S[ok], self.units)

        self.curve_df = pd.DataFrame(
            self._A,
            index=pd.Index(self._p_vert, name='p'),
            columns=['exeqa_total'] + kcols)

        # Diagnostic: sum of unit allocations should equal T at every vertex.
        # Any residual is inherited from density_df; see class docstring.
        self.additivity_error = float(
            np.abs(self._A[:, 1:].sum(axis=1) - self._T).max())

    # ----------------------------------------------------------------------
    # Representation
    # ----------------------------------------------------------------------

    def __repr__(self):
        lo, hi = self.premium_range
        cap = f'a={self.a:.6g}, ' if np.isfinite(self.a) else ''
        return (f'AllocationBounds({self.port.name!r}, {cap}units={self.units}, '
                f'{len(self._T)} vertices, premium range [{lo:.6g}, {hi:.6g}], '
                f'additivity error {self.additivity_error:.3g})')

    @property
    def info(self):
        """Fixed-layout multi-line summary string (terse).

        Every row is always present, in the same order, for every
        ``AllocationBounds``. The middle block is the shared ``_HullEngine``
        slice geometry (:meth:`_hull_info_rows`), so this and
        :attr:`PricingBounds.info` read alike below the identity rows. Shares
        the label/value convention (:func:`aggregate.constants.info_row`) with
        ``Aggregate`` / ``Portfolio``; catalogue in ``dev/info-strings.rst``.
        """
        rows = [
            ('allocation bounds object', self.port.name),
            ('kind', 'natural-allocation premium ranges by unit'),
        ]
        rows += self._hull_info_rows()
        rows.append(('additivity error', f'{self.additivity_error:.3g}'))
        return '\n'.join(info_row(label, value) for label, value in rows)

    # ----------------------------------------------------------------------
    # p_star — exact inversion of TVaR_p(X) = P on the grid
    # ----------------------------------------------------------------------

    def _invert_T(self, P):
        """
        The p with ``TVaR_p(X) = P``, exact for the discrete distribution.

        Locate the bracketing vertices ``T_m <= P <= T_{m+1}``; within that
        atom ``TVaR_p = x_m + (A - S x_m)/(1 - p)`` with
        ``A = S_{m+1} T_{m+1}`` and ``S = S_{m+1}`` the strict-tail mass, so

            1 - p* = S_{m+1} (T_{m+1} - x_m) / (P - x_m).
        """
        m = int(np.searchsorted(self._T, P))
        if self._T[m] == P:
            return float(self._p_vert[m])
        # P strictly between T[m-1] and T[m]: inside the atom at x of row m-1.
        xk = self._x[self._row[m - 1]]
        s = self._S[m] * (self._T[m] - xk) / (P - xk)
        return float(1.0 - s)

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


class PricingBounds(_HullEngine):
    """
    Price ranges of a second risk consistent with pricing a reference risk.

    Given a reference risk ``X`` priced to ``P`` by *some* distortion ``g``,
    determine for one or more other risks ``Y`` the range of the price
    ``rho_g(Y)`` over the whole family ``G_P = { g : rho_g(X) = P }`` — the
    cross-pricing counterpart of :class:`AllocationBounds` (see the module
    docstring for the comparison).

    A distortion with Kusuoka measure ``mu`` prices *any* risk by
    ``rho_mu(Z) = int TVaR_p(Z) mu(dp)``, so the pricing constraint
    ``int TVaR_p(X) mu(dp) = P`` is one affine constraint and the extreme
    consistent measures are biTVaRs.  Plotting ``TVaR_p(Y)`` against
    ``TVaR_p(X)`` as ``p`` sweeps ``[0, 1]`` gives a parametric curve whose
    vertical slice at ``T_X = P`` through the convex hull is the price range
    of ``Y``.

    Parameters
    ----------
    x_source : Aggregate, Portfolio, pd.Series, ``'uniform'``, or TVaR source
        The reference risk ``X`` carrying the pricing constraint.  Pass
        ``'uniform'`` (or :func:`uniform_source`) for the Gini lens, where
        ``TVaR_p(X) = (1 + p) / 2`` and the constraint becomes the mean
        Kusuoka level ``E_mu[p] = 2P - 1``.
    y_sources : source or list/dict of sources
        The risk(s) ``Y`` whose price range is wanted.  A dict supplies
        explicit names; a list/single uses each source's own name.  Any
        TVaR source is accepted (a risk, a layer, a standalone unit, or the
        uniform reference).
    a : float, default ``np.inf``
        Asset cap.  Finite ``a`` prices ``min(X, a)`` and ``min(Y, a)`` —
        only the *distributions* are capped (no allocation bookkeeping), an
        exact, cheap pmf transform.  Ignored for closed-form sources.
    s_floor : float, default 1e-14
        Drop vertices with tail probability ``1 - p < s_floor`` (deep-tail
        FFT noise); shrinks the feasible-premium upper end from ``ess sup X``.
    n_grid : int, default 1024
        Fallback p-grid size used only when *every* source is closed-form
        (no breakpoints); smooth curves converge fast so this is rarely hit.

    Attributes
    ----------
    curve_df : DataFrame
        Vertex table indexed by ``p`` with column ``tvar_<X>`` (= ``TVaR_p(X)``)
        and one ``price_<Y>`` column per Y (= ``TVaR_p(Y)``).  Exact at every
        index value; the curve is linear in the ``(T_X, T_Y)`` plane between
        rows.
    premium_range : tuple of float
        ``(E[X], T_X_max)`` — the feasible P values.
    gini_level : float or None
        For a uniform reference, ``2P - 1`` is the mean Kusuoka level; this
        attribute is ``None`` unless the X-source is the uniform reference,
        in which case :meth:`mean_kusuoka_level` gives it for any P.

    Notes
    -----
    **Unmatched p-grids.**  X and Y have different CDF breakpoints.  Within a
    single atom of X, ``TVaR_p(X)`` is affine in ``u = 1/(1 - p)``; within a
    single atom of Y, ``TVaR_p(Y)`` is affine in the *same* ``u``.  On the
    intersection of an X-atom and a Y-atom — between consecutive points of the
    **union of the two breakpoint sets** — both coordinates are affine in
    ``u``, so the curve is exactly a line segment in ``(T_X, T_Y)`` space.
    Merging the breakpoint grids and evaluating both TVaRs at every union
    point therefore gives the exact piecewise-linear curve with no
    interpolation.

    **Gini lens** (uniform reference).  With ``X = U[0, 1]`` the x-axis is
    affine in ``p`` itself, so the constraint reads ``E_mu[p] = 2P - 1`` and
    the bounds become the concave/convex envelope gap of ``Y``'s own
    ``TVaR``-vs-``p`` curve at the mean Kusuoka level — an X-free reading of
    why distortions pricing ``Y`` disagree.

    **Bounded vs unbounded.**  For a finite asset cap ``a`` the construction
    is exact and well-conditioned: the tail collapses into a macroscopic atom
    at ``a``, no vertex approaches ``p = 1``, and :meth:`check` reprices to
    ~1e-12.  Unbounded, the upper price bound of a heavy-tailed ``Y`` is
    genuinely tail-driven (a vanishing weight on the ess-sup pricing measure
    times a diverging ess sup), and on the FFT grid the deep-tail vertices
    (``1 - p`` below ~1e-7) are discretization noise rather than exact math.
    The upper bound there is grid-sensitive and :meth:`check` will show larger
    errors at extreme-tail biTVaR knots — diagnostic of that unreliability.
    Cap the risks (pass ``a`` / use :meth:`Portfolio.pricing_bounds` with
    ``p=``), or raise ``s_floor``, for stable answers.

    Examples
    --------
    ::

        from aggregate.bounds import PricingBounds
        pb = PricingBounds(port, layer_agg)      # price of layer given rho(port)=P
        pb = PricingBounds('uniform', Y)         # Gini lens on Y
        pb.curve_df                              # (p, T_X, T_Y) vertex table
        pb.bounds(1200)                          # (P, risk) -> lower/upper/width
        pb.bitvars(1200)                         # achieving biTVaRs (p0, p1, w1)
        pb.check(1200)                           # audit: reprice each Y directly
        pb.plot(P=1200)                          # curves, hulls, slice
    """

    _item_label = 'risk'
    _curve_label = r'$T_Y(p)$ vs $T_X(p)$'
    _xlabel = '$T_X(p)$ = price of X'
    _ylabel = 'price of Y'

    def __init__(self, x_source, y_sources, *, a=np.inf, s_floor=1e-14,
                 n_grid=1024):
        self.a = float(a)
        self.s_floor = float(s_floor)
        self.n_grid = int(n_grid)

        # Coerce the X (reference) source and the Y (target) source(s).
        self._x_source = _coerce_source(x_source, a=self.a, name='X')
        self._is_uniform_x = (isinstance(self._x_source, _CallableSource)
                              and self._x_source.breakpoints is None
                              and self._x_source.name == 'U[0,1]')

        if isinstance(y_sources, dict):
            raw = list(y_sources.items())
        elif isinstance(y_sources, (list, tuple)):
            raw = [(None, y) for y in y_sources]
        else:
            raw = [(None, y_sources)]
        y_items = []
        seen = set()
        for nm, y in raw:
            src = _coerce_source(y, a=self.a, name=nm)
            name = nm if nm is not None else src.name
            base, k = name, 1                    # disambiguate clashes
            while name in seen:
                k += 1
                name = f'{base}#{k}'
            seen.add(name)
            y_items.append((name, src))
        self._y_sources = dict(y_items)
        y_names = [nm for nm, _ in y_items]

        # ------------------------------------------------------------------
        # Vertex p-grid: the union of every source's CDF breakpoints (the p
        # where some VaR jumps), plus p = 0.  Between consecutive union points
        # every TVaR is affine in u = 1/(1 - p), so the curve is exactly
        # piecewise linear with these vertices.  If all sources are closed
        # form, fall back to a dense grid (approximate but fast-converging).
        # ------------------------------------------------------------------
        bps = [np.asarray(src.breakpoints, dtype=float)
               for src in [self._x_source] + [s for _, s in y_items]
               if src.breakpoints is not None]
        if bps:
            p_vert = np.unique(np.concatenate([[0.0]] + bps))
        else:
            p_vert = np.linspace(0.0, 1.0, self.n_grid, endpoint=False)
        p_vert = p_vert[(p_vert >= 0.0) & (1.0 - p_vert >= self.s_floor)]

        # Evaluate every TVaR at the shared vertices (x-vertices are common,
        # so adding Y columns is cheap — many Ys cost almost nothing).
        T = self._x_source.tvar(p_vert)
        ycols = [src.tvar(p_vert) for _, src in y_items]
        A = np.column_stack([T] + ycols)

        # T_X must be strictly increasing.  It is for distinct p (conditional
        # tail expectations increase) until X saturates at its ess sup, where
        # several top vertices tie; keep only strictly-advancing vertices
        # (the slicing/hull machinery needs a strict x-order).
        order = np.argsort(T, kind='mergesort')
        T, A, p_vert = T[order], A[order], p_vert[order]
        runmax = np.maximum.accumulate(np.concatenate([[-np.inf], T[:-1]]))
        ok = T > runmax
        if ok.sum() < 2:
            raise ValueError('Fewer than two usable curve vertices; the '
                             'x-source is degenerate or s_floor too aggressive.')
        T, A, p_vert = T[ok], A[ok], p_vert[ok]

        self._init_engine(T, A, p_vert, 1.0 - p_vert, y_names)

        self.curve_df = pd.DataFrame(
            A, index=pd.Index(p_vert, name='p'),
            columns=[f'tvar_{self._x_source.name}']
                    + [f'price_{nm}' for nm in y_names])
        self.gini_level = (self.mean_kusuoka_level(self.premium_range[0])
                           if self._is_uniform_x else None)

    # ----------------------------------------------------------------------
    # Representation
    # ----------------------------------------------------------------------

    def __repr__(self):
        lo, hi = self.premium_range
        cap = f'a={self.a:.6g}, ' if np.isfinite(self.a) else ''
        return (f'PricingBounds(X={self._x_source.name!r}, '
                f'Y={list(self._y_sources)}, {cap}'
                f'{len(self._T)} vertices, premium range [{lo:.6g}, {hi:.6g}])')

    @property
    def info(self):
        """Fixed-layout multi-line summary string (terse).

        Every row is always present, in the same order, for every
        ``PricingBounds``; the Gini rows render ``n/a`` unless the reference
        risk X is the uniform. The middle block is the shared ``_HullEngine``
        slice geometry (:meth:`_hull_info_rows`), so this and
        :attr:`AllocationBounds.info` read alike below the identity rows.
        Catalogue in ``dev/info-strings.rst``.
        """
        rows = [
            ('pricing bounds object',
             f'{self._x_source.name} vs {", ".join(self._y_names)}'),
            ('kind', 'price ranges of risks consistent with the reference'),
            ('reference risk', self._x_source.name),
            ('uniform reference', bool(self._is_uniform_x)),
        ]
        rows += self._hull_info_rows()
        rows.append(('n_grid', self.n_grid))
        rows.append(('gini_level',
                     f'{self.gini_level:.6g}' if self.gini_level is not None
                     else INFO_NA))
        return '\n'.join(info_row(label, value) for label, value in rows)

    # ----------------------------------------------------------------------
    # p_star via the X-source inverse
    # ----------------------------------------------------------------------

    def _invert_T(self, P):
        """The p with ``TVaR_p(X) = P`` via the X-source's exact inverse."""
        p = self._x_source.tvar_inv(P)
        if p is None:
            raise NotImplementedError(
                f'X-source {self._x_source.name!r} provides no TVaR inverse; '
                'p_star is unavailable for this reference.')
        return float(p)

    def mean_kusuoka_level(self, P):
        """
        The mean Kusuoka level ``pi = 2P - 1`` for the uniform (Gini) reference.

        Only meaningful when the X-source is :func:`uniform_source`: there the
        pricing constraint ``int TVaR_p(X) mu(dp) = P`` reduces to
        ``E_mu[p] = 2P - 1``, the average TVaR threshold the consistent
        distortions must carry.

        Parameters
        ----------
        P : float

        Returns
        -------
        float
        """
        if not self._is_uniform_x:
            raise ValueError('mean_kusuoka_level requires the uniform X-source '
                             "(pass x_source='uniform').")
        return 2.0 * float(P) - 1.0

    # ----------------------------------------------------------------------
    # Audit
    # ----------------------------------------------------------------------

    def check(self, P):
        """
        Audit the bounds at P by direct repricing with the achieving biTVaRs.

        For each Y and bound, build the achieving biTVaR distortion and
        reprice Y from first principles via the survival hinge
        ``rho_g(Y) = sum_y y Delta g(S_Y(y))`` — an independent path from the
        ``TVaR``-chord that produced the slice — then compare.  ``total``
        reprices the *reference* X with the same biTVaR and must equal P.

        Parameters
        ----------
        P : float

        Returns
        -------
        DataFrame
            Indexed by ``(risk, bound)`` with columns ``value`` (hull),
            ``repriced`` (direct), ``err``, ``total`` (price of X, == P),
            ``total_err``.  ``repriced``/``total`` are ``NaN`` for closed-form
            sources that hold no pmf.

        Notes
        -----
        The biTVaR hinge uses the *exact vertex tail probabilities*
        ``1 - p0``, ``1 - p1`` from the engine, avoiding the ``1 - p``
        cancellation near ``p = 1``.
        """
        P = float(P)
        recs = []
        for nm, src in self._y_sources.items():
            for side in ('lower', 'upper'):
                v, p0, p1, w1, i0, i1 = self._slice(nm, side, np.array([P]))
                v, w1, i0, i1 = float(v[0]), float(w1[0]), int(i0[0]), int(i1[0])
                s0, s1 = float(self._S[i0]), float(self._S[i1])
                g = lambda s: _bitvar_g(s, s0, s1, w1)
                repriced = src.reprice(g)
                total = self._x_source.reprice(g)
                recs.append((
                    nm, side, v,
                    np.nan if repriced is None else repriced,
                    np.nan if repriced is None else repriced - v,
                    np.nan if total is None else total,
                    np.nan if total is None else total - P))
        return (pd.DataFrame(recs, columns=['risk', 'bound', 'value', 'repriced',
                                            'err', 'total', 'total_err'])
                .set_index(['risk', 'bound']))
