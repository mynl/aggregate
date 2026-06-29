r"""The bivariate leg kernel: legs as pushforwards of a bivariate source.

A **leg** is one cash-flow stream -- a deterministic, vectorized function
``f(X, Y)`` of the two coordinates of a bivariate law, *pushed forward* onto its
own one-dimensional grid. This module is **domain-free**: it knows nothing about
insurance. Gross / ceded / net reinsurance, variable rating and reinstatement are
all *Views* assembled on top (``aggregate._insurance_view`` and the analysis
classes); the kernel only knows ``(X, Y) -> legs -> distributions``.

The one idea (``dev/plan-bivariate-legs.md`` section 3): a leg's distribution is
the **image (pushforward)** of the source measure through the leg map. The source
is a finite set of weighted atoms ``(state_k, p_k)``; for a leg ``f`` we

1. **evaluate** ``z_k = f(state_k)`` at every atom;
2. **bin by output** -- ``f`` is generally many-to-one (a saturating layer maps a
   range of ``X`` to one value), so atoms landing on the same ``z`` have their
   masses **added** (accumulate, never sort-and-relabel); then re-bucket ``z``
   onto a regular grid -> a :class:`~aggregate._grid_distribution.GridDistribution`
   so every ``q`` / ``var`` / ``tvar`` / ``cdf`` routes through the one canonical
   object;
3. in parallel take **exact moments** straight off the source,
   ``E[f^m] = sum_k p_k f(state_k)^m`` -- the ground-truth "EX" column (means add
   exactly, no rebucketing).

Dimensionality lives in the **source**, not the leg. Every leg has the universal
signature ``f(X, Y)``. A *degenerate* source (:class:`GraphSource`) puts all mass
on the curve ``Y = kappa(X)`` parameterized by a 1-D grid, so any leg factors
through ``X`` alone and pushes over the univariate density (the cheap 1-D path). A
*full* bivariate (e.g.
:class:`~aggregate.bivariate.BivariateDistribution`) makes ``Y`` independent
information, so a leg that touches it is genuinely 2-D. "1-D vs 2-D is a source
swap."

The **source protocol** is duck-typed -- any object exposing

* ``pushforward(function, *, name, is_loss_value) -> GridDistribution`` and
* ``transformed_moments(function, max_order=3) -> pandas.Series``

is a source. :class:`~aggregate.bivariate.BivariateDistribution` already
satisfies it; :class:`GraphSource` supplies the degenerate (univariate + ``kappa``)
case over :func:`~aggregate.bivariate.pushforward_1d`.

Submodule access only (no top-level re-export)::

    from aggregate.legs import Leg, LegSet, GraphSource
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

__all__ = ['Leg', 'LegSet', 'GraphSource']


# ----------------------------------------------------------------------
# The leg (kernel): one labelled cash-flow stream f(X, Y)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Leg:
    """One cash-flow stream: a named, vectorized map ``f(X, Y)`` of the source axes.

    The kernel leg is **label-free** -- ``name`` is an identity the View maps to
    insurance labels (``gross_loss`` -> perspective ``gross`` / category
    ``loss``); it carries no ``perspective`` / ``category`` / ``kind`` of its own.
    Signs live **in the map** (a paid premium is a negative map), so cash-flow
    algebra is just summation: any combination of legs (a margin, ``net = X - Y``)
    is itself a leg.

    Parameters
    ----------
    name : str
        Leg identity, e.g. ``'gross_loss'`` / ``'ceded_premium'``.
    map : callable
        Vectorized ``f(X, Y) -> z``, the leg's **signed** value. Receives the
        source's broadcast axis arrays (a full bivariate calls
        ``map(axis0[:, None], axis1[None, :])``; a :class:`GraphSource` calls
        ``map(x, kappa(x))``). A one-coordinate leg simply ignores ``Y``.
    is_value : bool, default True
        Orientation of the resulting :class:`GridDistribution` -- ``True`` a loss
        value (adverse tail at the high end), ``False`` an underwriting result /
        payoff (adverse tail at the low end). Passed through as
        ``is_loss_value``.
    """

    name: str
    map: Callable = field(repr=False)
    is_value: bool = True

    # -- leg algebra: signs in the maps, so combining is summation -------
    def __neg__(self) -> 'Leg':
        """The sign-flipped leg ``-f`` (a received cash flow becomes paid)."""
        f = self.map
        return Leg(f'-{self.name}', lambda *a: -np.asarray(f(*a), dtype=float),
                   self.is_value)

    def __add__(self, other: 'Leg') -> 'Leg':
        """The summed leg ``f + g`` (e.g. ``net_uw = gross_uw + ceded_uw``)."""
        f, g = self.map, other.map
        return Leg(f'{self.name}+{other.name}',
                   lambda *a: np.asarray(f(*a), dtype=float)
                   + np.asarray(g(*a), dtype=float),
                   self.is_value)

    def __sub__(self, other: 'Leg') -> 'Leg':
        """The difference leg ``f - g`` (e.g. ``net_loss = gross_loss - ceded_loss``)."""
        f, g = self.map, other.map
        return Leg(f'{self.name}-{other.name}',
                   lambda *a: np.asarray(f(*a), dtype=float)
                   - np.asarray(g(*a), dtype=float),
                   self.is_value)

    def scaled(self, c: float, *, name: str = None) -> 'Leg':
        """The scaled leg ``c * f`` (e.g. a quota share of an existing leg)."""
        f = self.map
        return Leg(name or f'{c:g}*{self.name}',
                   lambda *a: float(c) * np.asarray(f(*a), dtype=float),
                   self.is_value)

    def renamed(self, name: str, *, is_value: bool = None) -> 'Leg':
        """A copy of this leg under a new ``name`` (and optional orientation)."""
        return Leg(name, self.map,
                   self.is_value if is_value is None else is_value)

    @classmethod
    def combine(cls, name, legs, *, is_value=True) -> 'Leg':
        """A single leg whose map is the **sum** of ``legs`` (``Margin = sum legs``).

        Signs are already baked into each member map, so the margin is a plain
        sum. ``legs`` is any iterable of :class:`Leg`.
        """
        maps = [leg.map for leg in legs]

        def _summed(*a):
            total = 0.0
            for f in maps:
                total = total + np.asarray(f(*a), dtype=float)
            return total

        return cls(name, _summed, is_value)


# ----------------------------------------------------------------------
# The leg set: ordered named legs, evaluated over a source
# ----------------------------------------------------------------------
class LegSet:
    """An ordered collection of named :class:`Leg` objects, evaluated over a source.

    The leg set is the unit a View assembles and the kernel pushes forward. It is
    **stateless** with respect to any particular source -- :meth:`distributions`
    and :meth:`stats_df` take the source as an argument so the same set can be
    pushed over the degenerate and the full bivariate alike. (Consumers cache the
    results; the kernel does not.)
    """

    def __init__(self, legs=()):
        self._legs = OrderedDict()
        for leg in legs:
            self.add(leg)

    def add(self, leg: Leg) -> 'Leg':
        """Append a :class:`Leg` (later additions override an earlier same name)."""
        self._legs[leg.name] = leg
        return leg

    @property
    def names(self):
        """The leg names, in insertion order."""
        return list(self._legs)

    def __iter__(self):
        return iter(self._legs.values())

    def __len__(self):
        return len(self._legs)

    def __contains__(self, name):
        return name in self._legs

    def __getitem__(self, name) -> Leg:
        return self._legs[name]

    # -- the one operation: push every leg over the source --------------
    def distributions(self, source) -> dict:
        """``{name: GridDistribution}`` -- each leg pushed forward over ``source``.

        One :meth:`source.pushforward` per leg (step 1-2 of the module note):
        evaluate the map at every source atom, accumulate masses landing on the
        same output value, and re-bucket onto a regular grid.
        """
        return {leg.name: source.pushforward(
                    leg.map, name=leg.name, is_loss_value=leg.is_value)
                for leg in self}

    def stats_df(self, source, *, max_order=3) -> pd.DataFrame:
        """Exact per-leg moments straight off the source grid (the "EX" column).

        One :meth:`source.transformed_moments` per leg -- the exact
        ``sum_k p_k f(state_k)^m`` with no rebucketing (step 3 of the module
        note). The index is the leg names; columns are whatever the source's
        moment method reports (``mass`` / ``mean`` / ``var`` / ``sd`` / ``cv`` /
        ``skew``). Means here are the ground truth the additive reports add to.
        """
        rows = {leg.name: source.transformed_moments(leg.map, max_order=max_order)
                for leg in self}
        return pd.DataFrame(rows).T


# ----------------------------------------------------------------------
# The degenerate source: all mass on the graph Y = kappa(X)
# ----------------------------------------------------------------------
class GraphSource:
    r"""A degenerate bivariate source: all mass on the curve ``Y = kappa(X)``.

    The univariate fast path of the kernel. The source is a 1-D grid ``X`` with
    density ``p``; the second coordinate is the deterministic graph
    ``Y = kappa(X)`` (the cession map ``kappa`` -- e.g. the aggregate-reinsurance
    ceder). Because every state is ``(x_i, kappa(x_i))``, a leg ``f(X, Y)``
    reduces to ``psi(X) = f(X, kappa(X))`` and pushes over the **univariate**
    density via :func:`~aggregate.bivariate.pushforward_1d` -- the cheap 1-D path,
    returning the **same** :class:`GridDistribution` type as a full 2-D
    pushforward.

    Parameters
    ----------
    grid : ndarray
        The 1-D source index ``X`` (e.g. ``Aggregate.density_df.index``).
    density : ndarray
        Source probability mass aligned with ``grid``.
    kappa : callable, optional
        Vectorized ``kappa(x) -> y`` defining the graph. Default is the zero map
        (``Y = 0``), i.e. a bare univariate source whose legs ignore ``Y``.

    Notes
    -----
    Satisfies the kernel **source protocol** (``pushforward`` /
    ``transformed_moments``) so a :class:`LegSet` cannot tell it from a full
    :class:`~aggregate.bivariate.BivariateDistribution`. The moment conventions
    here (``cv = 0`` at a zero mean, central-third-moment ``skew``, ``NaN`` at a
    point mass) match the legacy aggregate-basis engine exactly.
    """

    def __init__(self, grid, density, kappa=None):
        self.grid = np.asarray(grid, dtype=float)
        self.density = np.asarray(density, dtype=float)
        self.kappa = kappa if kappa is not None else \
            (lambda x: np.zeros_like(np.asarray(x, dtype=float)))

    def _values(self, function):
        """Evaluate ``function(X, kappa(X))`` broadcast onto the grid shape."""
        x = self.grid
        v = np.asarray(function(x, self.kappa(x)), dtype=float)
        return np.broadcast_to(v, x.shape)

    def pushforward(self, function, *, name=None, is_loss_value=True, **kwargs):
        """Pushforward of ``Z = function(X, kappa(X))`` over the univariate density.

        Delegates to :func:`~aggregate.bivariate.pushforward_1d` with the graph
        collapsed in, so a degenerate leg never materializes a 2-D grid.
        """
        from .bivariate import pushforward_1d
        kappa = self.kappa
        return pushforward_1d(
            self.grid, self.density,
            lambda x: function(x, kappa(x)),
            name=name, is_loss_value=is_loss_value, **kwargs)

    def transformed_moments(self, function, max_order=3):
        """Exact moments of ``Z = function(X, kappa(X))`` on the source grid.

        The ground-truth "EX" numbers: ``sum_i p_i psi(x_i)^k`` with no
        rebucketing. Mirrors the legacy aggregate-basis conventions -- ``mass`` /
        ``mean`` / ``var`` (floored at 0) / ``sd`` / ``cv`` (``0`` at a zero mean)
        / ``skew`` (central third moment, ``NaN`` at a point mass).
        """
        p = self.density
        v = self._values(function)
        mass = float(np.sum(p))
        m1 = float(np.sum(p * v))
        m2 = float(np.sum(p * v * v))
        var = max(m2 - m1 * m1, 0.0)
        sd = float(np.sqrt(var))
        cv = sd / m1 if m1 else 0.0
        if sd > 0 and max_order >= 3:
            m3 = float(np.sum(p * (v - m1) ** 3))
            skew = m3 / sd ** 3
        else:
            skew = np.nan
        return pd.Series({'mass': mass, 'mean': m1, 'var': var, 'sd': sd,
                          'cv': cv, 'skew': skew},
                         name=getattr(function, '__name__', 'Z'))

    def __repr__(self):
        return (f'GraphSource(n={len(self.grid)}, '
                f'mass={float(self.density.sum()):.4g})')
