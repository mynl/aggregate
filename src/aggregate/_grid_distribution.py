"""The shared discrete-grid distribution value type.

This module is a **leaf**: it imports only numpy / pandas / scipy and nothing
from elsewhere in :mod:`aggregate`. If it ever needs ``distributions`` or
``portfolio`` the design has gone wrong -- stop and reconsider.

:class:`GridDistribution` is the single home for the marginal-vector accessors
(``q`` / ``var`` / ``tvar`` / ``tvar_threshold`` / ``cdf`` / ``sf`` / ``pmf`` /
``mean`` / ``lev`` / ``tvar_of_limited``) that ``Aggregate``, ``Portfolio``,
``Severity``, ``Bounds`` and ``Bivariate`` all need. Each of those classes wraps
the same object underneath: a probability vector ``p`` over a loss index ``x``,
with an optional bucket size ``bs``.

**Spacing is not assumed.** The probability accessors are pure functions of
``(x, p)`` alone -- the :func:`make_var_tvar` kernel builds them from ``cumsum(p)``
and the index *values* via :func:`numpy.searchsorted`, never dividing by a step
width -- so they are already correct on a non-uniform index. Only the
width-dependent ops (``pdf`` = mass / width, ``snap`` = snap to a regular grid)
need ``bs``; leave it ``None`` for a genuinely non-uniform grid and those raise.

The ``make_var_tvar`` kernel lives **here** (relocated from ``utilities.py`` at
1.0.0a90): once every consumer goes through :class:`GridDistribution` the kernel
belongs with the type that owns var / tvar, and co-locating them makes this type
self-contained.
"""

from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.optimize import bisect

__all__ = ['GridDistribution', 'make_var_tvar', 'ProbLossAssets',
           'return_period_map']


def return_period_map(p, is_loss_value=True):
    """Return period ``T`` for non-exceedance probabilities ``p`` under a role.

    The loss/payoff branch in one place, so every consumer (the Lee/quantile
    plot worker and the summary ``tail_df``) reads the same map instead of
    re-deriving it.

    Parameters
    ----------
    p : float or array_like
        Non-exceedance probability ``F(x)``.
    is_loss_value : bool, default True
        Orientation. ``True`` (loss): the bad outcomes are rare *large* losses
        at large ``p``, so ``T = 1 / (1 - p)``. ``False`` (payoff): the bad
        outcomes are rare *low* payoffs at small ``p``, so ``T = 1 / p``.

    Returns
    -------
    float or ndarray
        Return period ``T``. Diverges (``inf``) at the saturating endpoint
        (``p -> 1`` loss, ``p -> 0`` payoff); the caller caps / drops it.

    Notes
    -----
    Pure function of ``(p, is_loss_value)``;
    :meth:`GridDistribution.return_period` is the bound form that supplies the
    distribution's own orientation.
    """
    p = np.asarray(p, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return 1.0 / (1.0 - p) if is_loss_value else 1.0 / p


def period_to_p(period, is_loss_value=True):
    """Non-exceedance probability ``p`` for return periods ``T`` under a role.

    The exact inverse of :func:`return_period_map`: it turns the return-period
    ladder of the summary ``tail_df`` (``1-in-200``, ``1-in-250``, ...) into the
    quantile levels ``p`` to feed to ``q`` / ``tvar``.

    Parameters
    ----------
    period : float or array_like
        Return period ``T`` (``T >= 1``).
    is_loss_value : bool, default True
        Orientation. ``True`` (loss): the bad outcomes are rare *large* losses,
        so the row sits in the upper tail, ``p = 1 - 1 / T``. ``False``
        (payoff): the bad outcomes are rare *low* payoffs, so the row sits in
        the lower tail, ``p = 1 / T`` -- the table reads off the downside.

    Returns
    -------
    float or ndarray
        Non-exceedance probability ``p = F(VaR)``.

    Notes
    -----
    Pure function of ``(period, is_loss_value)`` and the exact inverse of
    :func:`return_period_map`; the two share the single loss/payoff branch so
    the Lee plot and the ``tail_df`` ladder round-trip cleanly.
    """
    period = np.asarray(period, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return 1.0 - 1.0 / period if is_loss_value else 1.0 / period


QuantileFunctions = namedtuple("QuantileFunctions", 'q q_lower var q_upper tvar')

#: The three mutually-determining views of one point on the distribution:
#: VaR probability ``p``, limited expected loss ``L = E[min(X, a)]``, and asset
#: level ``a``. Returned by :meth:`GridDistribution.prob_loss_assets`.
ProbLossAssets = namedtuple('ProbLossAssets', 'p L a')


def make_var_tvar(ser):
    """Build var (lower quantile), upper quantile, and tvar functions from a series.

    ``ser`` has index given by losses and probability values. It must have a
    unique monotonic increasing index and all values > 0. Such a series comes
    from ``a.density_df.query('p_total > 0').p_total``, for example.

    The returned functions make **no equal-spacing assumption**: quantiles are
    found by :func:`numpy.searchsorted` on the cumulative ``ser`` against the
    index *values*, and the tvar integral accumulates ``x * p`` from the right,
    so the kernel is correct on a non-uniform index.

    Tested using numpy vs ``pd.Series`` lookup functions; this version is much
    faster. See ``tests/test_grid_distribution.py`` for brute-force checks.

    Parameters
    ----------
    ser : pandas.Series
        Realized pmf: index = outcomes, values = probabilities (all > 0), index
        unique and monotonic increasing.

    Returns
    -------
    QuantileFunctions
        Namedtuple ``(q, q_lower, var, q_upper, tvar)``; ``q`` / ``var`` are
        aliases for the lower quantile.

    Notes
    -----
    Relocated from ``aggregate.utilities`` at 1.0.0a90 (changed in v0.13.0
    originally). The body is unchanged: this is the var/tvar kernel that
    :class:`GridDistribution` wraps.
    """

    # audits
    assert ser.index.is_unique, 'index values must be unique'
    assert ser.index.is_monotonic_increasing, 'index values must be increasing'

    # detach from the outside scope
    ser = ser.copy()

    # create needed arrays
    x_np = np.array(ser.index)
    # better not to cumulate array when all elements are equal (because of
    # floating point issues). This does make some difference.
    if np.all(np.isclose(ser, ser.iloc[0], atol=2**-53)):
        d = 1 / len(ser)
        cser = pd.Series(np.linspace(d, 1, len(ser)), index=ser.index)
    else:
        cser = ser.cumsum()
    cser_F_np = cser.to_numpy()
    # detach the index values
    # cser_idx = pd.Index(cser.values)
    tvar_unconditional = ((ser * ser.index)[::-1].cumsum()[::-1]).to_numpy()

    # these last three are annoyting because np.where does not short circuit
    tvar_unconditional = np.hstack((tvar_unconditional, np.inf, np.inf))
    cser_F_np2 = np.hstack((cser_F_np, 1))
    x_np2l = np.hstack((x_np, x_np[-1]))
    x_np2u = np.hstack((x_np, np.inf))
    # x_max = cser_F_np[-2]

    # tests show this is about 6 times faster than
    # q = interp1d(cser, ser.index, kind='next', bounds_error=False, fill_value=(ser.index.min(), ser.index.max()))
    def q_lower(p):
        nonlocal x_np2l, cser_F_np
        return x_np2l[np.searchsorted(cser_F_np, p, side='left')]

    def q_upper(p):
        nonlocal x_np2u, cser_F_np
        return x_np2u[np.searchsorted(cser_F_np, p, side='right')]

    def tvar(p):
        """
        Vectorized TVaR computation.
        """
        nonlocal cser_F_np, x_np, tvar_unconditional
        if isinstance(p, (float, int)):
            # easy
            if p >= cser_F_np[-2]:
                return x_np[-1]
            else:
                idx = np.searchsorted(cser_F_np, p, side='right')
                return ((cser_F_np[idx] - p) * x_np[idx] + tvar_unconditional[idx + 1]) / (1 - p)
        else:
            # vectorized
            p = np.array(p)
            idx = np.searchsorted(cser_F_np, p, side='right')
            return np.where(idx >= len(cser_F_np) - 1,
                            x_np[-1],
                           ((cser_F_np2[idx] - p) * x_np2u[idx] + tvar_unconditional[idx + 1]) / (1 - p))

    return QuantileFunctions(q_lower, q_lower, q_lower, q_upper, tvar)


class GridDistribution:
    """A discrete distribution on a grid: mass ``p`` over index ``x``, optional ``bs``.

    Read-only value type owning the lazy var/tvar kernel cache and the marginal
    accessors. Construct one from any ``(x, p)`` pair (or a series via
    :meth:`from_series`); the holder rebuilds a fresh ``GridDistribution`` when
    its density changes, so there is no cache to invalidate.

    Parameters
    ----------
    x : array_like
        Loss index (outcomes); unique, monotonic increasing.
    p : array_like
        Probability mass at each ``x`` (need not sum to exactly 1; the kernel
        works on the realized cumulative).
    bs : float, optional
        Bucket size. Needed only by the width-dependent ops (:meth:`pdf`,
        :meth:`snap`); leave ``None`` for a genuinely non-uniform grid and those
        raise a clear error.
    name : str, optional
        Display name.
    is_loss_value : bool, default True
        Orientation of the cash flow ``X``: ``True`` if ``X = 1`` means "I pay
        1" (a **loss**, more is worse), ``False`` if it means "I receive 1" (a
        **payoff**, more is better). Fixed at construction and read-only -- part
        of the cash flow's identity, set by whoever builds the distribution.
        Defaults ``True`` for the sign-agnostic callers that do not care.

    Notes
    -----
    **Spacing-agnostic invariant.** The probability accessors (:meth:`q`,
    :meth:`var`, :meth:`tvar`, :meth:`tvar_threshold`, :meth:`cdf`, :meth:`sf`,
    :meth:`pmf`, :meth:`mean`, :meth:`lev`, :meth:`tvar_of_limited`) never divide
    by ``bs`` -- only :meth:`pdf` / :meth:`snap` (and :meth:`lev` when ``bs`` is
    supplied) touch spacing. A ``bs``-divide creeping into ``tvar`` / ``q`` is a
    bug.

    **Orientation is metadata, not a kernel input.** The objective accessors
    above describe the random variable and are identical for a loss or a payoff,
    so they never read :attr:`is_loss_value`. The flag is consulted only by the
    "which side is bad?" operations -- pricing (which tail a distortion loads)
    and :meth:`return_period` (which tail the return-period axis spreads).
    """

    def __init__(self, x, p, bs=None, name='', is_loss_value=True):
        self._x = np.asarray(x, dtype=float)
        self._p = np.asarray(p, dtype=float)
        self.bs = bs
        self.name = name
        self._is_loss_value = bool(is_loss_value)
        self._vt = None              # lazy var/tvar kernel cache, owned HERE
        self._cum = None             # lazy cumulative (for cdf/sf/lev)

    @classmethod
    def from_series(cls, ser, bs=None, name='', is_loss_value=True):
        """Build from a ``pd.Series`` (index = outcomes, values = mass)."""
        return cls(np.asarray(ser.index, dtype=float), ser.to_numpy(dtype=float),
                   bs, name or (ser.name if ser.name is not None else ''),
                   is_loss_value)

    def to_series(self, name=None):
        """The mass as a ``pd.Series`` ``outcome -> prob`` (index = ``x``).

        The thin Series view a :class:`~aggregate.PnL` ``density_df`` iterates
        over: a GD on its own (possibly irregular, exact) grid, ready to plot or
        tabulate without stapling onto a shared grid. The index is named
        ``'outcome'``; the series is named ``name`` (defaults to the GD's name).

        Parameters
        ----------
        name : str, optional
            Series name; defaults to :attr:`name`.

        Returns
        -------
        pandas.Series
            Index = outcomes ``x`` (named ``'outcome'``), values = mass ``p``.
        """
        return pd.Series(self._p, index=pd.Index(self._x, name='outcome'),
                         name=name if name is not None else (self.name or None))

    def __repr__(self):
        nm = f' {self.name!r}' if self.name else ''
        bs = '' if self.bs is None else f', bs={self.bs:g}'
        role = 'loss' if self._is_loss_value else 'payoff'
        return f'GridDistribution{nm} (n={len(self._x)}{bs}, {role})'

    @property
    def x(self):
        """The loss index (read-only view)."""
        return self._x

    @property
    def p(self):
        """The probability mass vector (read-only view)."""
        return self._p

    @property
    def is_loss_value(self):
        """Orientation (read-only): ``True`` loss, ``False`` payoff.

        Metadata fixed at construction -- the objective kernel never consults
        it; only :meth:`return_period` and pricing do.
        """
        return self._is_loss_value

    def return_period(self, p):
        """Return period ``T`` for non-exceedance probabilities ``p``.

        Maps ``p`` to ``T`` through this distribution's own orientation:
        ``T = 1 / (1 - p)`` for a loss (bad tail at large ``p``), ``T = 1 / p``
        for a payoff (bad tail at small ``p``). Delegates to the shared
        :func:`return_period_map` so the Lee plot and the summary ``tail_df``
        read one implementation. Vectorizes over an array of ``p``.
        """
        return return_period_map(p, self._is_loss_value)

    # ------------------------------------------------------------------
    # lazy cores
    # ------------------------------------------------------------------
    def _funcs(self):
        """Lazily build and cache the var/tvar kernel on the ``p > 0`` subset.

        The kernel is built on the positive-mass subset to match the historic
        ``density_df.query('p_total > 0')`` filter the holder classes applied
        before calling it.
        """
        if self._vt is None:
            mask = self._p > 0
            ser = pd.Series(self._p[mask], index=self._x[mask])
            self._vt = make_var_tvar(ser)
        return self._vt

    def _cumulative(self):
        """Lazily build and cache ``(x, cumsum(p))`` for cdf/sf/lev."""
        if self._cum is None:
            self._cum = np.cumsum(self._p)
        return self._cum

    # ------------------------------------------------------------------
    # probability accessors: pure functions of (x, p); spacing-agnostic
    # ------------------------------------------------------------------
    def q(self, p, kind='lower'):
        """Quantile (value at risk). ``kind`` in ``{'lower', 'upper'}``.

        Lower quantile ``inf{x : F(x) >= p}``; upper quantile
        ``inf{x : F(x) > p}``. Vectorizes over an array of ``p``.
        """
        if kind == 'middle':
            kind = 'lower'
        assert kind in ('lower', 'upper'), "kind must be 'lower' or 'upper'"
        qf = self._funcs()
        return qf.q_lower(p) if kind == 'lower' else qf.q_upper(p)

    def var(self, p):
        """Value at risk = lower quantile (alias for ``q(p, 'lower')``)."""
        return self.q(p, 'lower')

    def tvar(self, p):
        """Tail value at risk (expected shortfall) at level ``p``. Vectorizes."""
        return self._funcs().tvar(p)

    def tvar_threshold(self, p, kind='lower'):
        """Find ``pt`` such that ``TVaR(pt) == VaR(p)`` by bisection.

        Fails if ``p == 0`` (returns 0, the mean threshold) because the signs
        do not bracket a root there.
        """
        a = self.q(p, kind)
        if p == 0:
            return 0
        return bisect(lambda t: self.tvar(t) - a, 0, 1)

    def cdf(self, x):
        """Right-continuous CDF ``P(X <= x)``. Scalar or array ``x``."""
        cum = self._cumulative()
        x = np.asarray(x, dtype=float)
        idx = np.searchsorted(self._x, x, side='right')   # count of atoms <= x
        out = np.where(idx == 0, 0.0, cum[np.clip(idx - 1, 0, len(cum) - 1)])
        return out[()] if out.ndim == 0 else out

    def sf(self, x):
        """Survival function ``P(X > x) = 1 - cdf(x)`` -- the S vector."""
        return 1.0 - self.cdf(x)

    def pmf(self, x):
        """Probability mass at ``x`` (0 if ``x`` is not a grid point)."""
        x = np.asarray(x, dtype=float)
        idx = np.searchsorted(self._x, x, side='left')
        scalar = x.ndim == 0
        idx = np.atleast_1d(idx)
        xv = np.atleast_1d(x)
        out = np.zeros(idx.shape, dtype=float)
        hit = (idx < len(self._x)) & (self._x[np.clip(idx, 0, len(self._x) - 1)] == xv)
        out[hit] = self._p[idx[hit]]
        return float(out[0]) if scalar else out

    def mean(self):
        """Mean ``E[X] = Σ x p``."""
        return float(np.sum(self._x * self._p))

    def lev(self, a):
        """Limited expected value ``E[min(X, a)] = ∫₀ᵃ S(x) dx``.

        The left-Riemann sum of the survival function up to ``a``: with a bucket
        ``bs`` it is ``bs · Σ_{x < a} S(x)`` -- the exact convention
        :meth:`Portfolio.add_exa` uses (the datum a distortion calibrates
        against). With ``bs is None`` it degrades to local ``np.diff(x)`` step
        widths so it stays correct on a non-uniform grid.

        This equals ``E[min(X, a)]`` when the grid is **zero-based** (``x[0] ==
        0``, the aggregate / portfolio convention): the sum starts at ``x[0]``,
        so a grid whose first atom is ``> 0`` omits the ``[0, x[0])`` slab where
        ``S == 1``. The intent is byte-for-byte parity with ``add_exa``, not an
        origin-independent ``E[min(X, a)]``.

        Parameters
        ----------
        a : float
            Cap / limit.

        Returns
        -------
        float
            ``E[min(X, a)]``.
        """
        x = self._x
        S = self.sf(x)                 # survival at each grid point
        if self.bs is not None:
            mask = x < a
            return float(self.bs * np.sum(S[mask]))
        # non-uniform: width to the next grid point, last step capped at a
        widths = np.diff(x)
        total = 0.0
        for i in range(len(x) - 1):
            if x[i] >= a:
                break
            w = min(widths[i], a - x[i])
            total += S[i] * w
        return float(total)

    def tvar_of_limited(self, p, a):
        """``TVaR_p(min(X, a))`` -- the analytic composite, O(1), no grid rebuild.

        For ``p < F(a)``::

            TVaR_p(min(X, a)) = TVaR_p(X) - (1 - F(a))(TVaR_{F(a)}(X) - a) / (1 - p)

        and ``a`` for ``p >= F(a)`` (the conditional tail of ``min(X, a)`` is
        exactly ``a`` once past the cap). Reads :meth:`tvar` / :meth:`cdf` only,
        so it is safe inside a root-find hot loop (cf. :meth:`cap`, which
        rebuilds the grid). ``a`` is an *argument*, never stored -- the
        distribution stays cap-agnostic.

        Parameters
        ----------
        p : float or array_like
            Level(s).
        a : float
            Cap.

        Returns
        -------
        float or ndarray
            ``TVaR_p(min(X, a))``.
        """
        tvar = self.tvar(p)
        if np.isinf(a):
            return tvar
        Fb = float(self.cdf(a))
        if Fb >= 1.0:
            return tvar
        gap = (1.0 - Fb) * (float(self.tvar(Fb)) - a)
        return np.where(np.asarray(p) < Fb, tvar - gap / (1.0 - np.asarray(p)), a)

    # ------------------------------------------------------------------
    # capital anchor: free choice over {p, L, a}
    # ------------------------------------------------------------------
    def _invert_lev(self, L):
        """Solve ``lev(a) = L`` for ``a`` by safeguarded Newton (bisection fallback).

        The limited expected value ``L(a) = ∫₀ᵃ S(x) dx`` is monotone
        increasing with continuous-model slope ``dL/da = S(a) = sf(a)``, so a
        Newton step ``a ← a − (lev(a) − L) / S(a)`` is taken whenever it stays
        inside the running bracket and ``S(a) > 0``; otherwise the step bisects.
        The far tail (``S(a) → 0``) makes Newton ill-conditioned, so the
        bisection fallback carries the solve there. The bracket starts at
        ``[0, q(1)]`` (``L`` feasibility -- ``L < mean()`` -- is the caller's
        responsibility, guaranteeing a root in range).

        Returns the (unsnapped) root location; :meth:`prob_loss_assets` snaps it
        to the grid and recomputes ``L`` so the returned triple is exact.
        """
        lo, hi = 0.0, float(self.q(1))
        # Pin the bracket far tighter than one bucket so the midpoint snaps
        # unambiguously to the intended grid point.
        tol = (self.bs * 1e-3) if self.bs is not None else (hi - lo) * 1e-12
        a = 0.5 * (lo + hi)
        for _ in range(200):
            f = self.lev(a) - L
            if f > 0:
                hi = a
            else:
                # f <= 0: lev is a left-continuous step, so the flat lev == L
                # plateau (just below the target grid point) reads as f == 0 and
                # is pushed up toward the jump.
                lo = a
            if hi - lo <= tol:
                break
            s = float(self.sf(a))
            step = (a - f / s) if s > 0 else None
            if f != 0 and step is not None and lo < step < hi:
                a = step
            else:
                a = 0.5 * (lo + hi)
        return 0.5 * (lo + hi)

    def prob_loss_assets(self, *, p=None, L=None, a=None):
        """Given any one of ``p``, ``L``, ``a``, return all three (capital anchor).

        ``{p, L, a}`` are three views of the same point on the distribution --
        the VaR probability ``p``, the limited expected loss
        ``L = E[min(X, a)] =`` :meth:`lev`, and the asset level ``a`` -- and any
        one determines the other two. Pass **exactly one** keyword (all others
        ``None``); the result is snapped to the grid so the returned ``a`` is an
        exact grid point and the triple is mutually consistent
        (``L == lev(a)``, ``p == cdf(a)``).

        Parameters
        ----------
        p : float, optional
            VaR probability; ``a = q(p)``.
        L : float, optional
            Limited expected loss ``E[min(X, a)]``; ``a`` is root-found from
            ``lev(a) = L``. Must satisfy ``L < mean()`` (a LEV is bounded above
            by ``E[X]``, attained only as ``a → ∞``); raises otherwise.
        a : float, optional
            Asset level; snapped to the grid.

        Returns
        -------
        ProbLossAssets
            Namedtuple ``(p, L, a)``.

        Raises
        ------
        ValueError
            If the number of supplied (non-None) arguments is not exactly one,
            or if ``L >= mean()`` (infeasible).

        Notes
        -----
        The ``L`` anchor is the fragile case: ``L`` near ``E[X]`` is
        ill-conditioned because ``dL/da = S(a) → 0`` in the far tail. The
        feasibility guard and the safeguarded root-find (:meth:`_invert_lev`,
        Newton with a bisection fallback) keep that failure mode explicit.

        The capital anchor is a **loss-side** construction: ``a`` is an asset
        level and ``L = E[min(X, a)]`` a limited *loss*, so the ``{p, L, a}``
        identity is meaningful only for the loss orientation
        (``is_loss_value=True``) on a **zero-based** grid (``x[0] == 0``, the
        aggregate / portfolio convention that :meth:`lev` and :meth:`q` are
        built for). Like the rest of the kernel this method does **not** consult
        :attr:`is_loss_value` -- it cannot detect a misuse -- so the caller is
        responsible for invoking it (and :meth:`lev` / :meth:`_invert_lev`) only
        on a loss-oriented, zero-based GD; on a payoff GD the result is not a
        capital anchor.
        """
        n = sum(v is not None for v in (p, L, a))
        if n != 1:
            raise ValueError(
                'prob_loss_assets: pass exactly one of p=, L=, a= '
                f'(got {n} non-None).')
        if a is not None:
            # snap to a grid point self-consistent with p
            p = float(self.cdf(a))
            a = float(self.q(p))
            L = self.lev(a)
        elif p is not None:
            a = float(self.q(p))      # already a grid point
            L = self.lev(a)
        else:
            mean = self.mean()
            if not (L < mean):
                raise ValueError(
                    f'prob_loss_assets: L={L:.6g} is infeasible. The limited '
                    f'expected value E[min(X, a)] is bounded above by '
                    f'E[X]={mean:.6g} (attained only as a -> infinity).')
            a_root = self._invert_lev(L)
            # snap to the grid, then recompute so the returned triple is exact
            if self.bs is not None:
                a = float(self.snap(a_root))
            else:
                a = float(self.q(self.cdf(a_root)))
            p = float(self.cdf(a))
            L = self.lev(a)
        return ProbLossAssets(float(p), float(L), float(a))

    # alias
    pla = prob_loss_assets

    # ------------------------------------------------------------------
    # width-dependent ops: require bs (raise if bs is None)
    # ------------------------------------------------------------------
    def _require_bs(self, op):
        if self.bs is None:
            raise ValueError(
                f'{op} requires a bucket size bs, but this GridDistribution has '
                'bs=None (a non-uniform grid). Width-dependent ops are undefined.')

    def pdf(self, x):
        """Continuous-density reading ``mass / bs`` (linear interp). Requires ``bs``."""
        self._require_bs('pdf')
        return np.interp(x, self._x, self._p, left=0.0, right=0.0) / self.bs

    def snap(self, x):
        """Snap ``x`` to the nearest point of the regular ``bs`` grid. Requires ``bs``."""
        self._require_bs('snap')
        x0 = self._x[0]
        return x0 + np.round((np.asarray(x, dtype=float) - x0) / self.bs) * self.bs

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------
    def cap(self, a):
        """Return a fresh :class:`GridDistribution` for ``min(X, a)``.

        Pools all mass at points ``>= a`` onto a single atom at ``a``. Any
        accessor on the capped law then follows -- ``gd.cap(a).tvar(p)`` equals
        ``gd.tvar_of_limited(p, a)`` -- but :meth:`cap` rebuilds the grid, so it
        is the convenience path for one-off capped views, **not** the hot loop
        (use :meth:`tvar_of_limited` there).
        """
        x, p = self._x, self._p
        if a >= x[-1]:
            return GridDistribution(x.copy(), p.copy(), self.bs, self.name,
                                    self._is_loss_value)
        mask = x < a
        tail = float(np.sum(p[~mask]))
        new_x = np.append(x[mask], a)
        new_p = np.append(p[mask], tail)
        return GridDistribution(new_x, new_p, self.bs, self.name,
                                self._is_loss_value)
