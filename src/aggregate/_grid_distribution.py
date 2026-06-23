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

__all__ = ['GridDistribution', 'make_var_tvar']


QuantileFunctions = namedtuple("QuantileFunctions", 'q q_lower var q_upper tvar')


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

    Notes
    -----
    **Spacing-agnostic invariant.** The probability accessors (:meth:`q`,
    :meth:`var`, :meth:`tvar`, :meth:`tvar_threshold`, :meth:`cdf`, :meth:`sf`,
    :meth:`pmf`, :meth:`mean`, :meth:`lev`, :meth:`tvar_of_limited`) never divide
    by ``bs`` -- only :meth:`pdf` / :meth:`snap` (and :meth:`lev` when ``bs`` is
    supplied) touch spacing. A ``bs``-divide creeping into ``tvar`` / ``q`` is a
    bug.
    """

    def __init__(self, x, p, bs=None, name=''):
        self._x = np.asarray(x, dtype=float)
        self._p = np.asarray(p, dtype=float)
        self.bs = bs
        self.name = name
        self._vt = None              # lazy var/tvar kernel cache, owned HERE
        self._cum = None             # lazy cumulative (for cdf/sf/lev)

    @classmethod
    def from_series(cls, ser, bs=None, name=''):
        """Build from a ``pd.Series`` (index = outcomes, values = mass)."""
        return cls(np.asarray(ser.index, dtype=float), ser.to_numpy(dtype=float),
                   bs, name or (ser.name if ser.name is not None else ''))

    def __repr__(self):
        nm = f' {self.name!r}' if self.name else ''
        bs = '' if self.bs is None else f', bs={self.bs:g}'
        return f'GridDistribution{nm} (n={len(self._x)}{bs})'

    @property
    def x(self):
        """The loss index (read-only view)."""
        return self._x

    @property
    def p(self):
        """The probability mass vector (read-only view)."""
        return self._p

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
            return GridDistribution(x.copy(), p.copy(), self.bs, self.name)
        mask = x < a
        tail = float(np.sum(p[~mask]))
        new_x = np.append(x[mask], a)
        new_p = np.append(p[mask], tail)
        return GridDistribution(new_x, new_p, self.bs, self.name)
