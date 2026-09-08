"""Bivariate aggregate distributions via copula + 2D FFT.

This module is the first-class home of the joint-aggregate machinery. It is
strictly **two-axis** by design: for three or more correlated units the right
path is independent components coupled by Iman--Conover and read back as a
sample (the "switcheroo"), not a native shared-frequency ``rfftn`` convolution.
It hosts:

* :class:`BivariateAggregate` -- the modelled object declared in DecL with the
  ``bivariate`` keyword. Two component ``agg`` / ``pnl`` severity factories are
  coupled per-claim by a :class:`aggregate.copula.Copula`, then accumulated by a
  **shared** outer frequency through a 2D FFT.
* :class:`BivariateDistribution` -- the lightweight joint-density container
  (marginals, mixed moments, correlation, contour). Originally introduced for
  :meth:`aggregate.distributions.Aggregate.occ_bivariate` (the joint law of
  occurrence ceded / net under a reinsurance program); kept here as the shared
  result container.
* :func:`_netceded_window_hi` / :func:`scatter_bivariate` -- the (one common bs)
  window measurement and the 2D rebucketing scatter used by ``occ_bivariate``.

Nothing here is re-exported at the top-level package namespace (submodule
access only, per the project layout convention): reach it as
``from aggregate.bivariate import BivariateAggregate``.

Notes
-----
**The construction.** Each component is an inner :class:`Aggregate` restricted
to a zero-inflated Bernoulli ``dfreq [0 1] [p0 p1]`` form, so its *aggregate*
density is the per-event severity ``g_i = (1 - p_i) delta_0 + p_i f_i`` ("peril
i triggered this event w.p. ``p_i``, with conditional severity ``f_i``"). With
marginal per-claim CDFs ``G_i = cumsum(g_i)`` and a copula ``C``, the joint
per-claim severity is the discrete-Sklar rectangle mass
``S[i,j] = C(G1[i],G2[j]) - C(G1[i-1],G2[j]) - C(G1[i],G2[j-1]) +
C(G1[i-1],G2[j-1])`` (see :meth:`Copula.rectangle_pmf`). The joint aggregate is
then ``density = iFFT2(freq_pgf(N, FFT2(S)))`` -- the ordinary compound-FFT with
the 1D transforms replaced by 2D transforms, valid because ``freq_pgf(n, z)`` is
*elementwise* in ``z``. Marginalising one axis recovers the corresponding
standalone aggregate (the exact validation target), because the zero-frequency
slice of ``FFT2(S)`` along an axis is just ``FFT`` of that axis's marginal
``g_i``.

**P&L axes are not supported here.** A ``pnl`` component (book-level / joint
P&L) is rejected at construction: a loss-sensitive consideration must be netted
per unit *before* the joint combine, which the 2D FFT does not preserve. Use
plain ``agg`` components, or build a standalone :class:`aggregate.PnL`.
"""

import json
import logging
import os
import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd
import scipy.fft as sfft
import scipy.sparse as ssp

from ._help import HelpMixin
from ._labeled import LabeledMixin
from ._program import ProgramMixin
from .constants import (DefectiveDistributionWarning, info_row, INFO_NA,
                        warn_once)
from .config import get_settings
from ._validation import DEFICIT_MATERIALITY
from .moments import (MomentAggregator, xsden_to_meancvskew,
                      _noise_aware_rel_error, _snap_noise)
from .utilities import round_bucket, balanced_window
from ._grid_distribution import GridDistribution

logger = logging.getLogger(__name__)

# Default output grid length (log2) for a pushforward when neither ``bs`` nor
# ``log2`` is pinned: the transformed variable lives on a single 1-D axis, so a
# 16-bit grid (65,536 buckets) over its realized range is ample and cheap.
_PUSHFORWARD_LOG2 = 16

# Noise floor for a finished joint density, as a fraction of the mass the grid
# carries. A 2-D FFT accumulates round-off proportional to the total it sums,
# not to the tallest cell it produced, so the floor is anchored to the sum: on
# a normalized joint that is 1e-15 in absolute terms, which is where the two
# de-fuzz sites here have always sat, and on an unnormalized one it still means
# the same depth. See :func:`_clip_density_fuzz`.
_DENSITY_FUZZ = 1e-15

# Coverage of the per-axis sizing window: 1 - 10**-_WINDOW_NINES per tail.
# First-class bivariate setting (see aggregate.config [bivariate]);
# independent of the 1-D distributions.WINDOW_NINES because the 2-D per-axis
# grid may want fewer nines for memory. Resolved once per session.
_WINDOW_NINES = get_settings().bivariate.window_nines
# Total 2-D grid budget in log2 cells (2**_TOTAL_LOG2 cells), split between the
# two axes by measured support; overridable via update(log2=...). Square-law
# memory lever (see dev/plan-mv.md §5.3).
_TOTAL_LOG2 = get_settings().bivariate.total_log2
# Smallest per-axis log2 the sizer will hand back (keeps a usable grid).
# First-class bivariate setting (see aggregate.config [bivariate]).
_MIN_AXIS_LOG2 = get_settings().bivariate.min_axis_log2
# Cap on the 1-D measurement grid used by _size_axes on the MASSIVE path only:
# a massive budget like (16, 16) would otherwise ask the standalone marginal
# for a 2**32-bucket update just to read its window. 2**20 buckets resolve the
# window edges to 10**-window_nines amply; the in-core path is untouched.
_MASSIVE_MEASURE_LOG2 = 20

# View-pair plumbing for the occurrence netceded family (the ``netceded`` /
# ``grossceded`` / ``grossnet`` DecL prefixes and ``occ_bivariate(views=...)``).
# Each view of {gross, ceded, net} maps to its realized occ margin column in
# ``reins_density_df`` (for the balanced_window measurement), its layering-summary
# column in ``reins_stats_df`` (for the theoretical moments), and its per-claim
# cession image map. ``gross`` is the identity map (no cession). The keyword
# names the pair x-then-y, so axis 0 is the first view, axis 1 the second
# (dev/plan-mv.md S3).
_VIEW_AGG_COL = {'gross': 'p_agg_gross',
                 'ceded': 'p_agg_ceded_occ',
                 'net': 'p_agg_net_occ'}
_VIEW_STAT_COL = {'gross': 'Gross', 'ceded': 'Ceded', 'net': 'Net'}


def _view_image_fn(agg, view):
    """Per-claim cession image map for an occurrence ``view`` of an aggregate.

    ``gross`` -> identity (the gross loss itself); ``ceded`` -> ``occ_ceder``;
    ``net`` -> ``occ_netter``. The image is sampled on the gross grid and
    scattered onto the (common-``bs``) view axis, so the gross axis rebuckets
    like the others rather than being special-cased.
    """
    if view == 'gross':
        return lambda x: x
    if view == 'ceded':
        return agg.occ_ceder
    if view == 'net':
        return agg.occ_netter
    raise ValueError(f'unknown reinsurance view {view!r}')


class ClashSolution(NamedTuple):
    """Solution of the independent-trigger clash model (see :func:`solve_clash_model`)."""

    n: float
    na: float
    nb: float
    nc: float
    n0: float
    pa: float
    pb: float


def solve_clash_model(na, nb, nc):
    """Derive the shared event count and per-event triggers of a clash model.

    The natural cat-clash baseline: a **shared** event drives two perils, each
    triggered within the event by an independent Bernoulli, and ``clash`` is the
    expected count of events that trigger *both*. Given the interpretable counts

    * ``na`` -- expected count of events triggering **A only**,
    * ``nb`` -- expected count triggering **B only**,
    * ``nc`` -- expected **clash** count (both A and B),

    the missing cell of the independent-trigger 2x2 table (neither peril) is fixed
    by the independence constraint ``nc * n0 = na * nb`` as ``n0 = na nb / nc``.
    The shared event count and the two per-event trigger probabilities are then

    .. math::

        n = na + nb + nc + n0, \\quad
        pa = (na + nc) / n, \\quad
        pb = (nb + nc) / n.

    These feed the two components as Bernoulli factories ``dfreq [0 1] [1-pa pa]``
    / ``[1-pb pb]`` coupled by the **independent** copula; the dependence between
    the two aggregates comes entirely from the shared frequency (plus any
    common-shock mixing on the outer frequency).

    Parameters
    ----------
    na, nb : float
        A-only and B-only expected event counts (non-negative).
    nc : float
        Clash (both) expected event count; must be positive for a finite,
        non-degenerate solution.

    Returns
    -------
    ClashSolution
        Named tuple ``(n, na, nb, nc, n0, pa, pb)``.

    Raises
    ------
    ValueError
        If ``nc <= 0`` or ``na`` / ``nb`` is negative.
    """
    na, nb, nc = float(na), float(nb), float(nc)
    if nc <= 0:
        raise ValueError('clash: nc (the clash count) must be positive.')
    if na < 0 or nb < 0:
        raise ValueError('clash: na and nb must be non-negative.')
    n0 = na * nb / nc
    n = na + nb + nc + n0
    return ClashSolution(n=n, na=na, nb=nb, nc=nc, n0=n0,
                         pa=(na + nc) / n, pb=(nb + nc) / n)


def _netceded_window_hi(occ_density, xs, prob):
    """Upper window edge of an occurrence-reins margin (``balanced_window``).

    Netceded sizing is **window selection** on the realized occ margins -- a
    column of ``reins_density_df`` (``p_agg_ceded_occ`` / ``p_agg_net_occ``,
    computed in one gross pass) read with the same
    :func:`~aggregate.utilities.balanced_window` primitive as the copula axes.
    Ceded / net are non-negative, so the grid is 0-based and need only reach this
    upper edge.

    Parameters
    ----------
    occ_density : ndarray
        The occ-reins aggregate margin on the gross grid ``xs`` (need not be
        normalised).
    xs : ndarray
        The gross loss grid.
    prob : float
        Discarded equal-tail mass for ``balanced_window``.

    Returns
    -------
    float
        The upper window edge ``q(1 - prob/2)`` of the margin.
    """
    ser = pd.Series(np.asarray(occ_density, dtype=float), index=np.asarray(xs))
    ser = ser[ser > 0]
    if len(ser) == 0:        # degenerate margin (no mass)
        return 0.0
    _lo, hi = balanced_window(ser, prob)
    return float(hi)


def scatter_bivariate(cv, nv, mass, bs_c, bs_n, n_c, n_n, scheme='linear'):
    """Scatter per-claim mass onto a 2D ``(ceded, net)`` grid.

    The 2D analogue of the reinsurance rebucketing scatter
    (:meth:`Aggregate._rebucket_to_grid`). Each gross claim of mass
    ``mass[k]`` is placed at the off-grid point ``(cv[k], nv[k])`` and
    distributed onto the surrounding grid cells.

    Parameters
    ----------
    cv, nv : ndarray
        Ceded and net loss for each gross grid point (``ceder(xs)`` /
        ``netter(xs)``).
    mass : ndarray
        Gross severity probability mass aligned with ``cv`` / ``nv``.
    bs_c, bs_n : float
        Ceded- and net-axis bucket sizes.
    n_c, n_n : int
        Ceded- and net-axis grid lengths.
    scheme : {'linear', 'nearest'}, default 'linear'
        ``'nearest'`` rounds each point to its closest cell (full mass in one
        cell). ``'linear'`` splits each point's mass bilinearly over the <= 4
        surrounding cells with weights ``(1-fc)(1-fn)``, ``fc(1-fn)``,
        ``(1-fc)fn`` and ``fc fn`` where ``fc``/``fn`` are the fractional grid
        positions; this preserves **both** marginal means exactly.

    Returns
    -------
    ndarray
        Bivariate severity density, shape ``(n_c, n_n)``; total mass equals
        ``mass.sum()``.

    Notes
    -----
    Points at or beyond the top of either axis pile into the edge cell -- the
    same overflow mode as a univariate aggregate deficit. Fractional offsets
    are clipped to ``[0, 1]`` against the clipped index, mirroring the 1D
    scatter so the overflow weight collapses cleanly onto the last bucket.
    """
    out = np.zeros((n_c, n_n))
    sc = np.asarray(cv, dtype=float) / bs_c
    sn = np.asarray(nv, dtype=float) / bs_n
    if scheme == 'nearest':
        ic = np.clip(np.round(sc).astype(int), 0, n_c - 1)
        inn = np.clip(np.round(sn).astype(int), 0, n_n - 1)
        np.add.at(out, (ic, inn), mass)
    else:  # 'linear' -- bilinear split, preserves both marginal first moments
        kc = np.clip(np.floor(sc).astype(int), 0, n_c - 1)
        kn = np.clip(np.floor(sn).astype(int), 0, n_n - 1)
        fc = np.clip(sc - kc, 0.0, 1.0)
        fn = np.clip(sn - kn, 0.0, 1.0)
        kc1 = np.clip(kc + 1, 0, n_c - 1)
        kn1 = np.clip(kn + 1, 0, n_n - 1)
        np.add.at(out, (kc, kn), mass * (1 - fc) * (1 - fn))
        np.add.at(out, (kc1, kn), mass * fc * (1 - fn))
        np.add.at(out, (kc, kn1), mass * (1 - fc) * fn)
        np.add.at(out, (kc1, kn1), mass * fc * fn)
    return out


def scatter_bivariate_sparse(cv, nv, mass, bs_c, bs_n, n_c, n_n, scheme='linear'):
    """Sparse (scipy CSR) form of :func:`scatter_bivariate` for the massive path.

    Identical placement math -- nearest / bilinear split, edge-clipped
    overflow -- but the result is a ``scipy.sparse.csr_matrix`` holding only
    the populated cells (~``len(mass)`` points, or 4x for ``'linear'``). The
    dense ``(n_c, n_n)`` form does not fit in RAM at massive grid sizes; the
    out-of-core kernel streams a sparse severity row-band by row-band
    (``dev/plan-bv.md`` §4.2 pass 1).

    Parameters
    ----------
    cv, nv, mass, bs_c, bs_n, n_c, n_n, scheme
        As :func:`scatter_bivariate`.

    Returns
    -------
    scipy.sparse.csr_matrix
        Bivariate severity mass, shape ``(n_c, n_n)``; duplicate placements
        are summed by the CSR conversion. Total mass equals ``mass.sum()``.
    """
    sc = np.asarray(cv, dtype=float) / bs_c
    sn = np.asarray(nv, dtype=float) / bs_n
    mass = np.asarray(mass, dtype=float)
    if scheme == 'nearest':
        rows = np.clip(np.round(sc).astype(int), 0, n_c - 1)
        cols = np.clip(np.round(sn).astype(int), 0, n_n - 1)
        w = mass
    else:  # 'linear' -- bilinear split, preserves both marginal first moments
        kc = np.clip(np.floor(sc).astype(int), 0, n_c - 1)
        kn = np.clip(np.floor(sn).astype(int), 0, n_n - 1)
        fc = np.clip(sc - kc, 0.0, 1.0)
        fn = np.clip(sn - kn, 0.0, 1.0)
        kc1 = np.clip(kc + 1, 0, n_c - 1)
        kn1 = np.clip(kn + 1, 0, n_n - 1)
        rows = np.concatenate([kc, kc1, kc, kc1])
        cols = np.concatenate([kn, kn, kn1, kn1])
        w = np.concatenate([mass * (1 - fc) * (1 - fn), mass * fc * (1 - fn),
                            mass * (1 - fc) * fn, mass * fc * fn])
    return ssp.coo_matrix((w, (rows, cols)), shape=(n_c, n_n)).tocsr()


# ----------------------------------------------------------------------
# Generic pushforward of a joint (or marginal) density through a scalar map
# ----------------------------------------------------------------------
# After a 2-D FFT has produced the joint law of ``(L, R)``, quantities such as
# net loss ``L - A(R)`` or net underwriting ``P_G - D - h(R) - L + A(R)`` are
# deterministic, generally nonlinear functions of both coordinates. Their
# marginals are *not* convolutions (the two axes are dependent), so the correct
# object is the **pushforward** of the probability matrix through the map: scatter
# every cell's mass onto a 1-D output grid at the transformed value. The same
# machinery serves a 1-D source density (an aggregate marginal) for the
# aggregate-basis variable-rating features. See ``dev/pre-plan-reinstatements.md``
# sections 9-10 and ``dev/plan-variable-rating-appendix.md`` section 6.

def _pushforward_grid(vmin, vmax, *, bs=None, log2=None, window=None):
    """Resolve the output ``(z0, bs, n_out)`` for a 1-D pushforward grid.

    The grid is regular (``z0 + bs * arange(n_out)``) and **may be signed** --
    a net underwriting result runs negative -- so ``z0`` is aligned *down* to a
    multiple of ``bs`` (``floor(lo / bs) * bs``) which keeps ``0`` on the grid
    when the range straddles it.

    Parameters
    ----------
    vmin, vmax : float
        The realized min / max of the transformed values (used when ``window``
        is not pinned).
    bs : float, optional
        Output bucket size. Default: sized from the range and ``log2``.
    log2 : int, optional
        Output grid length ``1 << log2``. Default: :data:`_PUSHFORWARD_LOG2`
        when ``bs`` is also unset; ignored once ``bs`` and ``window`` fix the
        count.
    window : (float, float), optional
        Explicit ``(lo, hi)`` output range, overriding the measured one.

    Returns
    -------
    z0, bs, n_out : float, float, int
    """
    if window is not None:
        lo, hi = float(window[0]), float(window[1])
    else:
        lo, hi = float(vmin), float(vmax)
    if not hi > lo:                     # degenerate (constant map): tiny span
        hi = lo + (abs(lo) + 1.0) * 1e-9
    span = hi - lo
    bs_pinned = bs is not None
    if not bs_pinned:
        target_log2 = int(log2) if log2 is not None else _PUSHFORWARD_LOG2
        bs = float(round_bucket(span / (1 << target_log2)))
    else:
        bs = float(bs)
    z0 = float(np.floor(lo / bs) * bs)
    if log2 is not None and not bs_pinned:
        # log2 pins the count; bs was sized to span/2**log2, so 1<<log2 buckets
        # cover the range with a little headroom.
        n_out = 1 << int(log2)
    else:
        n_out = int(np.ceil((hi - z0) / bs)) + 1
    return z0, bs, int(n_out)


def _scatter_1d(values, weights, z0, bs, n_out, scheme='linear'):
    """Scatter ``weights`` onto a regular 1-D grid at transformed ``values``.

    The 1-D analogue of :func:`scatter_bivariate`, built on :func:`numpy.bincount`
    (faster than ``np.add.at`` for dense 1-D output; pre-plan section 9.4). Mass at
    a value outside ``[z0, z0 + bs*(n_out-1)]`` piles onto the nearer edge bucket
    (the same overflow mode as a univariate aggregate deficit) -- so **total mass
    is retained**; the clipped amount is reported separately.

    Parameters
    ----------
    values : ndarray
        Transformed value at each source cell (any shape; flattened).
    weights : ndarray
        Source probability mass aligned with ``values`` (same shape).
    z0, bs : float
        Output grid origin and bucket size.
    n_out : int
        Output grid length.
    scheme : {'linear', 'nearest'}, default 'linear'
        ``'linear'`` splits each cell's mass over the two straddling buckets
        (preserves the mean exactly); ``'nearest'`` rounds to one bucket.

    Returns
    -------
    mass : ndarray
        Length-``n_out`` output mass.
    clipped : float
        Mass whose true value fell outside the grid (piled on an edge).
    """
    u = (np.asarray(values, dtype=float).ravel() - z0) / bs
    w = np.asarray(weights, dtype=float).ravel()
    clipped = float(w[(u < 0.0) | (u > n_out - 1)].sum())
    if scheme == 'nearest':
        k = np.clip(np.round(u).astype(int), 0, n_out - 1)
        mass = np.bincount(k, weights=w, minlength=n_out)
    else:                               # linear split, mean-preserving
        k0 = np.floor(u).astype(int)
        f = u - k0
        k0c = np.clip(k0, 0, n_out - 1)
        k1c = np.clip(k0 + 1, 0, n_out - 1)
        # clip fractional offset against the clipped index so out-of-range mass
        # collapses cleanly onto the edge bucket (mirrors scatter_bivariate).
        f = np.clip(f, 0.0, 1.0)
        mass = np.bincount(k0c, weights=w * (1.0 - f), minlength=n_out)
        mass += np.bincount(k1c, weights=w * f, minlength=n_out)
    return mass[:n_out], clipped


def _finalize_pushforward(values, weights, *, bs, log2, window, scheme,
                          name, is_loss_value, source):
    """Size the grid, scatter, and wrap the result as a :class:`GridDistribution`.

    Shared tail of both pushforward entry points. Warns (does not raise) when a
    material fraction of mass is clipped onto the grid edges, mirroring the
    aggregate deficit convention. The clipped mass and source provenance are
    stashed on the returned object (``.clipped_mass`` / ``.pushforward_source``)
    for the audit.
    """
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    z0, bs_out, n_out = _pushforward_grid(vmin, vmax, bs=bs, log2=log2,
                                          window=window)
    mass, clipped = _scatter_1d(values, weights, z0, bs_out, n_out, scheme)
    grid = z0 + bs_out * np.arange(n_out)
    total = float(weights.sum())
    if total > 0 and clipped / total > DEFICIT_MATERIALITY:
        warn_once(
            f'pushforward {name!r}: {clipped / total:.2e} of mass fell outside '
            f'the output window [{grid[0]:g}, {grid[-1]:g}] and was clipped onto '
            f'the edge buckets. Widen with window=/log2=/bs=.',
            DefectiveDistributionWarning,
            key='defective-construction', stacklevel=3)
    gd = GridDistribution(grid, mass, bs=bs_out, name=name or '',
                          is_loss_value=is_loss_value)
    gd.clipped_mass = clipped
    gd.pushforward_source = source
    return gd


def pushforward_1d(grid, density, function, *, bs=None, log2=None, window=None,
                   scheme='linear', name=None, is_loss_value=True):
    """Pushforward of a 1-D source density through a scalar map ``z = function(x)``.

    The aggregate-basis path of the shared variable-rating engine
    (``plan-variable-rating-appendix.md`` section 2): when ceded loss is a
    deterministic function of *gross* loss the joint is degenerate and a leg
    ``psi(L)`` is a 1-D pushforward of the gross density. Returns the **same**
    :class:`GridDistribution` result type as the 2-D :meth:`BivariateDistribution.pushforward`.

    Parameters
    ----------
    grid : ndarray
        The source loss index (e.g. ``Aggregate.density.index``).
    density : ndarray
        Source probability mass aligned with ``grid``.
    function : callable
        Vectorized ``function(x) -> z``; receives the ``grid`` array.
    bs, log2, window, scheme, name, is_loss_value
        As :meth:`BivariateDistribution.pushforward`.

    Returns
    -------
    GridDistribution
    """
    x = np.asarray(grid, dtype=float)
    p = np.asarray(density, dtype=float)
    values = np.broadcast_to(np.asarray(function(x), dtype=float), x.shape)
    return _finalize_pushforward(values, p, bs=bs, log2=log2, window=window,
                                 scheme=scheme, name=name,
                                 is_loss_value=is_loss_value, source='1d')


# ----------------------------------------------------------------------
# Dict pushforward: several f_i(X, Y) (+ their total) in ONE pass (plan-bv §6)
# ----------------------------------------------------------------------

def _probe_constant(f, xs0, xs1, n_probe=7):
    """Detect a constant ``f(x, y)`` on a coarse probe lattice.

    A constant function (common -- a premium) never enters the band sweep:
    its pushforward is a point mass at the constant. The probe evaluates
    ``f`` on an ``n_probe x n_probe`` lattice spanning the label ranges; a
    genuinely non-constant function that is constant on the probe is
    pathological and out of scope (documented in :meth:`pushforward`).

    Returns
    -------
    float or None
        The constant value, or ``None`` if ``f`` varies on the probe.
    """
    px = xs0[np.linspace(0, len(xs0) - 1, min(n_probe, len(xs0))).astype(int)]
    py = xs1[np.linspace(0, len(xs1) - 1, min(n_probe, len(xs1))).astype(int)]
    v = np.asarray(f(px[:, None], py[None, :]), dtype=float)
    v = np.broadcast_to(v, (len(px), len(py)))
    if np.all(v == v.flat[0]):
        return float(v.flat[0])
    return None


def _pushforward_functions(bands, xs0, xs1, total_mass, functions, bs, *,
                           bs_total=None, total_key='total', windows=None,
                           scheme='linear', is_loss_value=True, source=None,
                           row_chunk=1024):
    """Streamed pushforward of a dict of functions ``{key: f(x, y)}`` -- and,
    for two or more, their **total** ``t = sum f_i`` -- in one pass over the
    joint density (``dev/plan-bv.md`` §6).

    The single driver behind both containers:
    :meth:`MassiveBivariateDistribution.pushforward` feeds it disk bands;
    :meth:`BivariateDistribution.pushforward` (dict form) feeds the in-core
    density as one band. The total is accumulated **pre-bucketing** --
    evaluated exactly per cell, summed, then bucketed once with its own
    ``bs_total`` -- never as a sum of bucketed results, so
    ``mean(total) == sum(mean(f_i))`` is a hard invariant. Constants (given
    as numbers, or callables flagged by :func:`_probe_constant`) are
    short-circuited before the sweep: a point mass at the constant, a scalar
    shift inside the total.

    Parameters
    ----------
    bands : callable
        Zero-argument generator factory yielding ``(r0, r1, band)`` density
        row bands in physical order; called once for the sweep. (A factory,
        not an iterator, so the label-only window pre-sweep never touches
        the density.)
    xs0, xs1 : ndarray
        Axis label grids.
    total_mass : float
        Total joint mass (for the constant point masses and the audit).
    functions : dict[str, callable or float]
        Named maps ``f(x, y)`` (vectorized, broadcast contract as
        :meth:`BivariateDistribution.pushforward`) or constants.
    bs : float or array-like
        Output bucket size -- scalar (every key) or length ``len(functions)``
        in iteration order.
    bs_total : float, optional
        Output bucket size for the total. **Required** when two or more
        functions are given; must be omitted for a single function.
    total_key : str, default 'total'
        Key for the total entry; collision with a user key raises.
    windows : dict[str, (float, float)], optional
        Explicit output ranges per key (``total_key`` allowed); keys not
        pinned are measured by a label-only pre-sweep (CPU, no disk).
    scheme : {'linear', 'nearest'}, default 'linear'
        Scatter scheme (``'linear'`` preserves the mean).
    is_loss_value : bool, default True
        Orientation of the outputs.
    source : object, optional
        Stashed on each result as ``.pushforward_source``.
    row_chunk : int, default 1024
        Row band size for the label-only pre-sweep.

    Returns
    -------
    dict[str, GridDistribution]
        One entry per key, plus ``total_key`` when ``len(functions) >= 2``.
        Each carries ``.clipped_mass``, ``.pushforward_source`` and the
        shared Est-vs-EX audit frame ``.pushforward_audit_df`` (rows per key,
        exact streamed moments vs the realized output moments).
    """
    if not isinstance(functions, dict) or not functions:
        raise ValueError('functions must be a non-empty dict {key: f(x, y)}.')
    keys = list(functions)
    n = len(keys)
    if total_key in keys:
        raise ValueError(
            f'functions key {total_key!r} collides with the reserved total '
            f'key; rename it or pass total_key=.')
    if n >= 2 and bs_total is None:
        raise ValueError(
            f'bs_total is required with {n} functions: the total '
            f't = sum(f_i) is returned under {total_key!r} and needs its own '
            'output bucket size (explicit is better than implicit).')
    if n == 1 and bs_total is not None:
        raise ValueError('bs_total given with a single function: no total is '
                         'computed (nothing for it to size).')
    if np.isscalar(bs):
        bs_map = {k: float(bs) for k in keys}
    else:
        bs_arr = np.asarray(bs, dtype=float)
        if len(bs_arr) != n:
            raise ValueError(
                f'bs must be a scalar or length {n} (one per function in '
                f'iteration order); got length {len(bs_arr)}.')
        bs_map = dict(zip(keys, bs_arr))
    windows = dict(windows or {})

    # ------------------------------------------------------------------
    # classify: constants never enter the sweep (numbers, or probe-constant
    # callables); their total contribution is a scalar shift.
    # ------------------------------------------------------------------
    consts, funcs = {}, {}
    for k, f in functions.items():
        if callable(f):
            c = _probe_constant(f, xs0, xs1)
            if c is None:
                funcs[k] = f
            else:
                consts[k] = c
        else:
            consts[k] = float(f)
    const_shift = float(sum(consts.values()))
    with_total = n >= 2

    # ------------------------------------------------------------------
    # windows: label-only pre-sweep for any unpinned key (CPU, no disk I/O).
    # The total's range is measured cell-wise on the summed values -- NOT as
    # the sum of per-key extremes.
    # ------------------------------------------------------------------
    need = [k for k in funcs if k not in windows]
    need_total = with_total and total_key not in windows
    if need or need_total:
        lo = {k: np.inf for k in funcs}
        hi = {k: -np.inf for k in funcs}
        lo_t, hi_t = np.inf, -np.inf
        yb = xs1[None, :]
        for r0 in range(0, len(xs0), row_chunk):
            r1 = min(r0 + row_chunk, len(xs0))
            xb = xs0[r0:r1, None]
            t = None
            for k, f in funcs.items():
                v = np.asarray(f(xb, yb), dtype=float)
                lo[k] = min(lo[k], float(v.min()))
                hi[k] = max(hi[k], float(v.max()))
                if need_total:
                    vb = np.broadcast_to(v, (r1 - r0, len(xs1)))
                    t = vb.copy() if t is None else t + vb
            if need_total and t is not None:
                lo_t = min(lo_t, float(t.min()))
                hi_t = max(hi_t, float(t.max()))
        for k in need:
            windows[k] = (lo[k], hi[k])
        if need_total:
            if not funcs:                      # all constants
                lo_t = hi_t = 0.0
            windows[total_key] = (lo_t + const_shift, hi_t + const_shift)

    # ------------------------------------------------------------------
    # output grids + accumulators (mass, clip, raw moments E[Z^k], k <= 3)
    # ------------------------------------------------------------------
    specs = {}
    for k in funcs:
        z0, bs_k, n_out = _pushforward_grid(*windows[k], bs=bs_map[k])
        specs[k] = dict(z0=z0, bs=bs_k, n_out=n_out, mass=np.zeros(n_out),
                        clipped=0.0, raw=np.zeros(4))
    if with_total:
        z0, bs_t, n_out = _pushforward_grid(*windows[total_key],
                                            bs=float(bs_total))
        specs[total_key] = dict(z0=z0, bs=bs_t, n_out=n_out,
                                mass=np.zeros(n_out), clipped=0.0,
                                raw=np.zeros(4))

    # ------------------------------------------------------------------
    # THE sweep: one disk read per band, n+1 bincount scatters, the exact
    # ("EX") raw moments folded alongside for free.
    # ------------------------------------------------------------------
    def _fold(sp, values, dens):
        m, c = _scatter_1d(values, dens, sp['z0'], sp['bs'], sp['n_out'],
                           scheme)
        sp['mass'] += m
        sp['clipped'] += c
        vk = np.ones_like(values)
        for j in range(4):
            sp['raw'][j] += float(np.sum(dens * vk))
            if j < 3:
                vk = vk * values

    if funcs or with_total:
        yb = xs1[None, :]
        for r0, r1, dens in bands():
            xb = xs0[r0:r1, None]
            t = None
            for k, f in funcs.items():
                v = np.broadcast_to(np.asarray(f(xb, yb), dtype=float),
                                    dens.shape)
                _fold(specs[k], v, dens)
                if with_total:
                    t = v.copy() if t is None else t + v
            if with_total:
                if t is None:                  # all constants
                    t = np.zeros(dens.shape)
                if const_shift:
                    t = t + const_shift
                _fold(specs[total_key], t, dens)

    # ------------------------------------------------------------------
    # results + the shared Est-vs-EX audit frame
    # ------------------------------------------------------------------
    out = {}
    audit_rows = {}

    def _ex_stats(raw):
        m0 = raw[0]
        mean = raw[1] / m0 if m0 else np.nan
        var = raw[2] / m0 - mean ** 2 if m0 else np.nan
        return mean, (np.sqrt(var) if var > 0 else 0.0)

    def _est_stats(grid, mass):
        m0 = mass.sum()
        mean = (grid @ mass) / m0 if m0 else np.nan
        var = (grid ** 2) @ mass / m0 - mean ** 2 if m0 else np.nan
        return mean, (np.sqrt(var) if var > 0 else 0.0)

    for k in keys:
        if k in consts:
            c = consts[k]
            gd = GridDistribution(np.array([c]), np.array([total_mass]),
                                  bs=bs_map[k], name=str(k),
                                  is_loss_value=is_loss_value)
            gd.clipped_mass = 0.0
            audit_rows[k] = (c, 0.0, c, 0.0)
        else:
            sp = specs[k]
            grid = sp['z0'] + sp['bs'] * np.arange(sp['n_out'])
            _warn_pushforward_clip(k, sp['clipped'], total_mass, grid)
            gd = GridDistribution(grid, sp['mass'], bs=sp['bs'], name=str(k),
                                  is_loss_value=is_loss_value)
            gd.clipped_mass = sp['clipped']
            audit_rows[k] = (*_ex_stats(sp['raw']),
                             *_est_stats(grid, sp['mass']))
        gd.pushforward_source = source
        out[k] = gd
    if with_total:
        sp = specs[total_key]
        grid = sp['z0'] + sp['bs'] * np.arange(sp['n_out'])
        _warn_pushforward_clip(total_key, sp['clipped'], total_mass, grid)
        gd = GridDistribution(grid, sp['mass'], bs=sp['bs'],
                              name=str(total_key), is_loss_value=is_loss_value)
        gd.clipped_mass = sp['clipped']
        gd.pushforward_source = source
        out[total_key] = gd
        audit_rows[total_key] = (*_ex_stats(sp['raw']),
                                 *_est_stats(grid, sp['mass']))

    audit = pd.DataFrame(
        {k: {'EX': ex_m, 'Est EX': est_m,
             'Err EX': _noise_aware_rel_error(est_m, ex_m),
             'SD': ex_s, 'Est SD': est_s,
             'Err SD': _noise_aware_rel_error(est_s, ex_s)}
         for k, (ex_m, ex_s, est_m, est_s) in audit_rows.items()}).T
    audit = audit[['EX', 'Est EX', 'Err EX', 'SD', 'Est SD', 'Err SD']]
    audit.index.name = 'key'
    for gd in out.values():
        gd.pushforward_audit_df = audit
    return out


def _warn_pushforward_clip(name, clipped, total, grid):
    """Warn when a material fraction of mass clipped onto the window edges
    (the shared convention of :func:`_finalize_pushforward`)."""
    if total > 0 and clipped / total > DEFICIT_MATERIALITY:
        warn_once(
            f'pushforward {name!r}: {clipped / total:.2e} of mass fell outside '
            f'the output window [{grid[0]:g}, {grid[-1]:g}] and was clipped onto '
            f'the edge buckets. Widen with windows= or coarsen bs.',
            DefectiveDistributionWarning,
            key='defective-construction', stacklevel=3)


def _clip_density_fuzz(density, where):
    """Zero the FFT dust in a finished joint density, in place.

    Parameters
    ----------
    density : ndarray
        The joint mass matrix, modified in place.
    where : str
        The construction being finished, for the warning text.

    Returns
    -------
    ndarray
        The same array.

    Notes
    -----
    **Small in magnitude is noise; large and negative is not.** The two-sided
    predicate this replaces was inherited from
    :func:`aggregate.utilities.remove_fuzz`, whose docstring justifies the
    ``abs`` explicitly for the signed P&L columns of a frame. That does not
    transfer. A joint density is non-negative even where its **support** is
    signed, which is the ordinary case for a P&L axis, so a large negative
    cell is a broken construction rather than dust. A one-sided clip would
    zero it just as quietly as the two-sided one did, which is why the
    warning is the point of this function and the clipping is the
    housekeeping around it.

    The floor is a fraction of the mass the grid carries rather than an
    absolute constant, because the depth an absolute constant reaches moves
    with the grid: the same ``1e-15`` sits 12.4 decades under the peak on one
    joint and 10.7 on another. Anchoring it to the sum (see
    :data:`_DENSITY_FUZZ`) holds the depth still, and on a normalized joint
    reproduces the absolute constant these two sites have always used.

    **The deficit is a free detector.** Both callers compute
    ``1 - density.sum()`` after this returns, so clipping a genuine negative
    raises the sum and drives the deficit negative. A negative deficit means
    something was clipped upward and this warning should have fired.
    """
    total = float(np.abs(density).sum())
    floor = _DENSITY_FUZZ * total if total > 0 else _DENSITY_FUZZ
    density[np.abs(density) < floor] = 0.0
    if density.size and density.min() < 0:
        logger.warning(
            '%s: joint density has %d negative cells, worst %.3e; clipped to '
            'zero. A density is non-negative even where its support is '
            'signed, so this is a broken construction rather than FFT fuzz, '
            'and the reported deficit goes negative by what was clipped.',
            where, int((density < 0).sum()), float(density.min()))
        density[density < 0] = 0.0
    return density


def build_netceded_joint(agg, views=('net', 'ceded'), bs=None,
                         log2_x=None, log2_y=None, total_log2=None):
    """Joint per-occurrence density of two of {gross, ceded, net} for one aggregate.

    The severity builder for :class:`BivariateAggregate`'s ``netceded`` mode
    (and the engine behind
    :meth:`aggregate.distributions.Aggregate.occ_bivariate`). Per claim the
    cession map sends a gross loss ``X`` to a point ``(v0(X), v1(X))`` on the
    comonotone curve, where ``v0`` / ``v1`` are the chosen views' image maps
    (gross = identity, ceded = ``occ_ceder``, net = ``occ_netter``); the three
    views satisfy ``ceded + net = gross``, so any two determine the third.
    Placing the gross severity mass at ``(v0(x_k), v1(x_k))`` (rebucketed by the
    object's :attr:`reins_bucket` scheme) builds a **comonotone** bivariate
    severity ``S``, and the joint aggregate is ``iFFT2(freq_pgf(N, FFT2(S)))``.

    Parameters
    ----------
    agg : Aggregate
        An **updated** aggregate carrying occurrence reinsurance.
    views : (str, str), default ``('net', 'ceded')``
        The ``(x, y)`` axis view pair, each one of ``'gross'`` / ``'ceded'`` /
        ``'net'``. The keyword that selected this pair names it x-then-y, so axis
        0 is ``views[0]`` and axis 1 is ``views[1]``.
    bs : float, optional
        Bucket-size override (a single common ``bs`` for both axes). Default
        (``None``) takes the **exact common lattice** when the budget affords
        it (:func:`_netceded_exact_bs`, so the comonotone scatter never fires
        and the kappa curve is exact), and otherwise sizes one common ``bs``
        from the budget (``~ sqrt(hi0*hi1)/2**(total_log2/2)``), which is
        coarser than the gross ``bs`` and reports the loss of exactness
        through :attr:`BivariateAggregate.bs_explanation`.
    log2_x, log2_y : int, optional
        Axis-0 / axis-1 log2 grid-length overrides; measured from the realized
        occ margins of the two views via :func:`balanced_window` if omitted.
    total_log2 : int, optional
        Total 2-D cell budget. With nothing pinned the common ``bs`` is
        coarsened until the two windows fit; a pinned ``bs`` / ``log2`` pair
        that still overflows **raises**, since clipping an axis to fit drops
        most of the joint rather than a tail. ``None`` uses the
        :attr:`BivariateSettings.total_log2` default.

    Returns
    -------
    density : ndarray
        Joint ``(views[0], views[1])`` aggregate density, shape ``(n0, n1)``.
    grid_x, grid_y : ndarray
        The two axis grids.
    bs_x, bs_y : float
        The two axis bucket sizes (one common ``bs`` unless overridden).
    sev2 : ndarray
        The bivariate per-claim severity ``S`` (the comonotone scatter).
    deficit : float
        Tail mass lost beyond the grid.
    sizing : NetcededSizing
        The realized grid decision: the common ``bs``, the two axis lengths,
        the exact lattice that was available and whether it was taken. Carried
        the last slot of the tuple, where ``clipped`` sat until 1.0.0a277; a
        netceded joint no longer clips, it raises.

    Raises
    ------
    ValueError
        If ``agg`` carries no occurrence reinsurance, has not been updated, or
        carries a pinned grid that cannot be honored inside the budget.

    Notes
    -----
    Mirrors :meth:`Aggregate._fft_aggregate` (zero-risk and one-claim
    shortcuts, the latter through the shared
    :attr:`Aggregate.one_claim` predicate) with the 1D transforms replaced by
    2D, valid because ``freq_pgf(n, z)`` is elementwise in ``z`` (handled with
    ravel/reshape).
    """
    import scipy.fft as _sfft

    sizing = _netceded_sizing(
        agg, views, bs=bs, log2_x=log2_x, log2_y=log2_y,
        total_log2=total_log2)
    bs_x = bs_y = sizing.bs
    n0, n1 = sizing.n0, sizing.n1
    grid_x = bs_x * np.arange(n0)
    grid_y = bs_y * np.arange(n1)

    sev2 = scatter_bivariate(sizing.x0, sizing.x1, sizing.mass,
                             bs_x, bs_y, n0, n1, scheme=agg.reins_bucket)

    if agg.n == 0:
        density = np.zeros((n0, n1))
        density[0, 0] = 1.0
    elif agg.one_claim:
        density = sev2.copy()
    else:
        pad = agg.padding
        s_shape = (n0 << pad, n1 << pad)
        z = _sfft.rfft2(sev2, s=s_shape)
        # base_mean, not n: under zm / zt these differ (n is the shifted mean)
        ftagg = agg.frequency.freq_pgf(agg.base_mean, z.ravel()).reshape(z.shape)
        density = np.real(_sfft.irfft2(ftagg, s=s_shape))[:n0, :n1]

    _clip_density_fuzz(density, f'netceded joint {agg.name!r}')
    deficit = float(1.0 - density.sum())
    return density, grid_x, grid_y, bs_x, bs_y, sev2, deficit, sizing


class NetcededSizing(NamedTuple):
    """The realized netceded grid decision, and the severity inputs on it.

    Attributes
    ----------
    x0, x1 : ndarray
        The two views' image points sampled on the gross grid.
    mass : ndarray
        Gross severity mass aligned with ``x0`` / ``x1``.
    bs : float
        The common per-axis bucket size actually used.
    n0, n1 : int
        Axis grid lengths.
    bs_exact : float or None
        The largest bucket size placing the gross grid and both view images on
        one lattice, or ``None`` when no affordable one exists (a share
        cession's images generally sit at no usable common lattice at all).
    exact : bool
        Whether :attr:`bs` **is** that lattice, so the comonotone scatter never
        splits a per-claim point and the kappa curve is exact rather than
        accurate to the smear.
    """

    x0: object
    x1: object
    mass: object
    bs: float
    n0: int
    n1: int
    bs_exact: float
    exact: bool


def _netceded_axis_log2(hi, bs):
    """Axis log2 covering a window of width ``hi`` at bucket size ``bs``."""
    return max(int(np.ceil(np.log2(hi / bs + 1.0))), _MIN_AXIS_LOG2)


def _netceded_exact_bs(x0, x1, gross_bs, hi0, hi1, total_log2):
    """The exact common lattice for a netceded joint, or ``None`` if unaffordable.

    The largest bucket size placing every sampled view image, and the gross
    grid itself, on one lattice: the gcd of the atoms, which is what
    :func:`_lattice_bs` computes for discrete mode and what this reuses rather
    than reimplements.

    Parameters
    ----------
    x0, x1 : ndarray
        The two views' image points.
    gross_bs : float
        The source aggregate's own bucket size, folded in so the gross axis
        lands on the lattice as well as the images do.
    hi0, hi1 : float
        The two measured window widths, used only for the affordability test.
    total_log2 : int
        The 2-D cell budget.

    Returns
    -------
    float or None
        The exact lattice, or ``None`` when the grid it implies does not fit
        the budget.

    Notes
    -----
    Exactness is a property of excess of loss layers on an aligned lattice,
    not a general one: a **share** cession multiplies by a non-integer factor,
    and its images typically share no usable lattice at any bucket size a grid
    could carry. So the rule is prefer the exact lattice when it is affordable,
    report when it is not, never pretend.

    The fold runs over the sorted unique atoms and exits as soon as the running
    gcd is too fine to fit, which is the common case and makes the test cheap:
    a gcd only decreases, so once the budget is blown it stays blown, and the
    exact value of an unaffordable lattice is of no interest.
    """
    atoms = np.unique(np.abs(np.concatenate(
        [np.asarray(x0, dtype=float).ravel(),
         np.asarray(x1, dtype=float).ravel(),
         np.array([float(gross_bs)])])))
    atoms = atoms[atoms > 0]
    if len(atoms) == 0:
        return None
    g = float(atoms[0])
    for v in atoms[1:]:
        g = _float_gcd(g, float(v))
        if (_netceded_axis_log2(hi0, g)
                + _netceded_axis_log2(hi1, g)) > total_log2:
            return None
    return g


def _netceded_sizing(agg, views, bs=None, log2_x=None, log2_y=None,
                     total_log2=None):
    """Window measurement, common-``bs`` grid sizing and view-image sampling
    for the netceded joint -- the mode-specific severity *inputs*, shared by
    the in-core :func:`build_netceded_joint` (dense scatter + in-core FFT) and
    the massive update path (sparse scatter + out-of-core kernel).

    Parameters
    ----------
    agg, views, bs, log2_x, log2_y, total_log2
        As :func:`build_netceded_joint`.

    Returns
    -------
    NetcededSizing

    Raises
    ------
    ValueError
        When a caller-pinned ``bs`` / ``log2_x`` / ``log2_y`` cannot be honored
        inside ``total_log2``. Clipping an axis to fit was the behavior until
        1.0.0a277 and it is not a tail loss: with both axes pinned equal and
        over budget the rule cut axis 0 to :data:`_MIN_AXIS_LOG2`, 16 buckets,
        and the object then answered questions off a joint carrying half its
        mass. The caller has stated numbers that cannot all be honored, so this
        says which and stops.
    """
    if agg.occ_reins is None:
        raise ValueError(
            'netceded requires occurrence reinsurance; none configured.')
    if agg.sev_density_gross is None or agg.occ_ceder is None:
        raise ValueError(
            'netceded requires an updated object (no severity densities '
            'present). Call update() first.')
    if total_log2 is None:
        total_log2 = _TOTAL_LOG2

    rd = agg.reins_density_df
    prob = 10.0 ** -_WINDOW_NINES
    # ONE common bs across both axes (the comonotone curve couples them). The
    # window lengths are measured from the realized occ margins of the chosen
    # views (balanced_window; gross / ceded / net are all non-negative so the
    # grids are 0-based), and the common bs is sized straight from the budget --
    # NOT pinned to the gross bs. The gross bs auto-sizes fine to resolve the
    # cession layer (e.g. 0.125), which is far too fine for the 2-D grid; the
    # linear scatter rebuckets the gross-sampled view points onto whatever common
    # bs the budget affords. Sizing from the budget (not gross) also makes
    # occ_bivariate and the DecL prefix forms agree regardless of the gross grid.
    hi0 = _netceded_window_hi(rd[_VIEW_AGG_COL[views[0]]].to_numpy(), agg.xs, prob)
    hi1 = _netceded_window_hi(rd[_VIEW_AGG_COL[views[1]]].to_numpy(), agg.xs, prob)

    # The images do not depend on the grid, so they are sampled before it is
    # chosen: the exact lattice is measured off them.
    x0 = np.asarray(_view_image_fn(agg, views[0])(agg.xs), dtype=float)
    x1 = np.asarray(_view_image_fn(agg, views[1])(agg.xs), dtype=float)
    mass = np.asarray(agg.sev_density_gross, dtype=float)

    bs_pinned = bool(bs)
    bs_exact = _netceded_exact_bs(x0, x1, agg.bs, hi0, hi1, total_log2)
    exact = False
    if bs:
        bs = float(bs)
        exact = bs_exact is not None and abs(bs - bs_exact) <= 1e-12 * bs
    elif bs_exact is not None:
        # Nothing is lost at this bucket size and the budget can pay for it:
        # every per-claim point is a cell center, the bilinear scatter never
        # fires, and the kappa curve is exact rather than accurate to the smear.
        bs, exact = bs_exact, True
    else:
        # finest common bs that fits 2**total_log2: n0*n1 ~ (hi0*hi1)/bs**2,
        # so bs ~ sqrt(hi0*hi1) / 2**(total_log2/2). round_bucket up, then fit.
        floor_bs = max((hi0 * hi1) ** 0.5 / (2.0 ** (0.5 * total_log2)), 1e-12)
        bs = float(round_bucket(floor_bs))
    log2_0 = (int(log2_x) if log2_x
              else _netceded_axis_log2(hi0, bs))
    log2_1 = (int(log2_y) if log2_y
              else _netceded_axis_log2(hi1, bs))
    if not (bs_pinned or (log2_x and log2_y)):
        guard = 0
        while log2_0 + log2_1 > total_log2 and guard < 8:
            bs = float(round_bucket(bs * 2))
            exact = False
            log2_0 = int(log2_x) if log2_x else _netceded_axis_log2(hi0, bs)
            log2_1 = int(log2_y) if log2_y else _netceded_axis_log2(hi1, bs)
            guard += 1
    if log2_0 + log2_1 > total_log2:
        pins = ', '.join(
            f'{k}={v:g}' for k, v in
            (('bs', bs if bs_pinned else None), ('log2_x', log2_x),
             ('log2_y', log2_y)) if v)
        raise ValueError(
            f'{agg.name}: the netceded ({views[0]}, {views[1]}) windows need '
            f'2**{log2_0 + log2_1} cells at bs={bs:g} (axis 0 log2 {log2_0}, '
            f'axis 1 log2 {log2_1}), over the budget 2**{total_log2}, and the '
            f'pinned {pins} leaves nothing to coarsen. Clipping an axis to fit '
            f'drops most of the joint rather than a tail, so this refuses '
            f'rather than answering off it: pass '
            f'total_log2={log2_0 + log2_1} (with store_dir= at that size), or '
            f'relax the pin.')
    n0 = 1 << log2_0
    n1 = 1 << log2_1
    return NetcededSizing(x0=x0, x1=x1, mass=mass, bs=bs, n0=n0, n1=n1,
                          bs_exact=bs_exact, exact=exact)


def _float_gcd(a, b):
    """Greatest common divisor of two positive floats, by float Euclid.

    Stops when the remainder is rounding noise relative to the larger of the
    pair and 1, which is what makes it usable on measured lattice values rather
    than on integers. Shared by :func:`_lattice_bs` (discrete mode) and
    :func:`_netceded_exact_bs` (the netceded exact lattice), which want the
    same quantity and should not each carry a copy of the loop.
    """
    while b > 1e-9 * max(a, 1.0):
        a, b = b, a - np.floor(a / b) * b
    return a


def _lattice_bs(xs):
    """Largest bucket size that places every atom of ``xs`` exactly on the grid.

    The discrete bivariate severity (``dbvsev``) is exact only if each lattice
    value is an integer multiple of the per-axis ``bs`` -- then the joint matrix
    scatters onto the FFT grid with no rebucketing. This returns the greatest
    such ``bs`` (the gcd of the non-zero atoms): ``1`` for an integer lattice,
    the natural step for a uniform one, a common divisor otherwise.

    Parameters
    ----------
    xs : array-like
        The axis outcomes (a 1-D lattice; ``0`` is ignored, negatives by
        magnitude).

    Returns
    -------
    float
        The largest exact bucket size; ``1.0`` for an empty / all-zero lattice.
    """
    vals = np.abs(np.asarray(xs, dtype=float))
    vals = vals[vals > 0]
    if len(vals) == 0:
        return 1.0
    g = vals[0]
    for v in vals[1:]:
        g = _float_gcd(g, v)
    return float(g)


def _dense_1d(v):
    """Flatten a vector-like to a dense 1-D ndarray.

    The per-claim severity ``_S`` may be scipy.sparse after a massive update
    (its ``sum(axis=...)`` returns an ``np.matrix``); this normalises either
    form so the severity moment/dependence code is representation-agnostic.
    """
    return np.asarray(v, dtype=float).ravel()


class BivariateAggregate(HelpMixin, LabeledMixin, ProgramMixin):
    """Joint (bivariate) aggregate of two copula-coupled component aggregates.

    Declared in DecL with the ``bivariate`` keyword::

        bivariate Cat 25 claims
            agg Wind  dfreq [0 1] [.3 .7] sev lognorm 40 cv 1.2
            agg Flood dfreq [0 1] [.5 .5] sev lognorm 60 cv 1.5
            copula gumbel 0.4
            mixed gamma .2

    A **shared** outer frequency (here ``25`` events, gamma-mixed Poisson) drives
    two perils; within each event the two per-claim severities are coupled by the
    copula (here gumbel upper-tail dependence), and the per-event trigger
    probabilities come from the inner ``dfreq [0 1] [p0 p1]`` Bernoulli forms.
    The result is the joint law ``(A_Wind, A_Flood)``; either marginal reproduces
    the standalone aggregate for that unit.

    Parameters
    ----------
    name : str
        Object name.
    units : list of tuple
        Exactly two ``('agg', name, spec)`` tuples (the component severity
        factories), as produced by the DecL transformer. A component may be a
        ``pnl`` (its spec then carries ``agg_reflect`` / ``agg_shift``); the
        affine is applied per-axis after the 2D FFT.
    copula : Copula
        The fitted :class:`aggregate.copula.Copula` instance coupling the two
        per-claim severities.
    note : str, optional
        Free-text note.
    hints : str, optional
        Raw ``hints{...}`` build-settings string (``key=value;`` form);
        retained as annotation and consumed by the underwriter build path.
    exp_en, exp_el, exp_premium, exp_lr : float, optional
        Outer (shared-count) exposure, as parsed. ``exp_en`` (a claim count) is
        used directly; otherwise the count is derived from ``exp_el`` and the
        component per-event means.
    freq_name, freq_a, freq_b, freq_zm, freq_p0
        Outer (shared) frequency specification.

    Attributes
    ----------
    units : list of Aggregate
        The two component **loss** aggregates (the per-event severity factories;
        any ``pnl`` affine is stripped here and reapplied per-axis after the FFT).
    unit_names : list of str
        Component names.
    copula : Copula
    frequency : Frequency
        The shared outer frequency.
    en : float
        Expected outer event count.
    density : ndarray or None
        Joint aggregate density, shape ``(n0, n1)``; populated by :meth:`update`.
    axis_xs : list of ndarray
        Per-axis output grids (P&L-relabelled where a component is a ``pnl``).
    bs : list of float
        Per-axis bucket sizes.
    """

    def __init__(self, name, units=None, copula=None, note='', hints='',
                 tags=(), label=None, label_map=None, mode='copula',
                 nc_agg=None, nc_kwargs=None, nc_views=None, clash=None,
                 dbv_xs=None, dbv_ys=None, dbv_S=None,
                 exp_en=None, exp_el=None, exp_premium=None, exp_lr=None,
                 freq_name='poisson', freq_a=0.0, freq_b=0.0,
                 freq_zm=False, freq_p0=np.nan, **kwargs):
        # local import avoids a circular import at module load (distributions
        # imports nothing from here at import time, but be defensive).
        from .distributions import Aggregate, Frequency

        self.mode = mode
        self.name = name
        # Before the mode branches below, two of which return early.
        self._init_labels(label=label, label_map=label_map)
        self.note = note
        self.hints = hints
        #: Tag slugs from the DecL ``tags{...}`` trailer ('()' when none).
        self.tags = tuple(tags)
        self.program = ''
        # clash provenance (na, nb, nc, n0, pa, pb) when built from a `clash`
        # statement; None for an ordinary copula / netceded bivariate.
        self.clash = dict(clash) if clash else None
        # filled by update() (both modes)
        self.density = None
        self.axis_xs = [None, None]
        self.bs = [None, None]
        self.deficit = np.nan
        # massive (disk-backed) result surface; set by update(store_dir=...)
        self._massive = None
        self._massive_dist = None
        self._S = None
        # given joint per-claim severity matrix (discrete mode); None otherwise,
        # so update_work falls back to the copula rectangle_pmf.
        self._S_given = None
        self._sev_xs = [None, None]
        self._sev_moms = [None, None]      # per-axis per-event severity raw moms
        self._marg_theory = [None, None]   # per-axis (mean, sd, skew)
        self._total = None                 # cached .total GridDistribution
        # declared exposure terms for the stats_df meta rows
        # ([Stats-Frame-Parallel]); the count itself is resolved by
        # _resolve_en. lr is derived so el = prem * lr foots by construction.
        self._exp_el = (float(np.sum(np.asarray(exp_el, dtype=float)))
                        if exp_el is not None else np.nan)
        if exp_premium is not None:
            self._exp_prem = float(np.sum(np.asarray(exp_premium,
                                                     dtype=float)))
            if not np.isfinite(self._exp_el) and exp_lr is not None:
                self._exp_el = float(np.sum(
                    np.asarray(exp_premium, dtype=float)
                    * np.asarray(exp_lr, dtype=float)))
        else:
            self._exp_prem = np.nan
        self._exp_lr = (self._exp_el / self._exp_prem
                        if np.isfinite(self._exp_el)
                        and np.isfinite(self._exp_prem)
                        and self._exp_prem > 0 else np.nan)
        self.figure = None                 # set by plot()

        if mode == 'netceded':
            self._init_netceded(name, units, nc_agg, nc_kwargs, nc_views)
            return

        if mode == 'discrete':
            if units is not None:
                raise ValueError(
                    'component axes are not supported with dbvsev; use the '
                    "copula form 'bv ... agg ... agg ...'.")
            self._init_discrete(name, dbv_xs, dbv_ys, dbv_S,
                                freq_name, freq_a, freq_b, freq_zm, freq_p0,
                                exp_en, exp_el, exp_premium, exp_lr)
            return

        if units is None or len(units) != 2:
            raise ValueError(
                'bivariate (copula) requires exactly two components; '
                f'got {0 if units is None else len(units)}')
        self.copula = copula

        self._unit_specs = [t[2] for t in units]
        self.unit_names = [t[1] for t in units]
        # pnl components (book-level / joint P&L) are deferred -- a loss-sensitive
        # consideration must be netted per unit before the joint combine, which
        # the 2D FFT does not preserve. Reject with a clear message.
        pnl_units = [t[1] for t in units if t[0] == 'pnl']
        if pnl_units:
            raise NotImplementedError(
                "pnl components in a bivariate are not supported (joint P&L is "
                f"deferred). Offending component(s): {', '.join(pnl_units)}. "
                "Use plain agg components, or build a standalone PnL.")
        self.units = [Aggregate(**s) for s in self._unit_specs]
        # No per-axis affine: pnl components are rejected above, so both axes are
        # plain loss aggregates (inert affine, as the netceded path also sets).
        self._affine = [(False, 0.0), (False, 0.0)]

        # shared outer frequency + per-event severity raw moments (theoretical,
        # grid-independent, available at Aggregate.__init__)
        self.frequency = Frequency(freq_name, freq_a, freq_b, freq_zm, freq_p0)
        self.freq_name = freq_name
        # the shared-frequency kwargs, retained so a standalone marginal
        # aggregate can be rebuilt for measure-don't-guess axis sizing (§5).
        self._freq_kwargs = dict(freq_name=freq_name, freq_a=freq_a,
                                 freq_b=freq_b, freq_zm=freq_zm, freq_p0=freq_p0)
        self._sev_moms = [self._raw3(a.actual_m, a.actual_sd, a.actual_skew)
                          for a in self.units]
        self.en = self._resolve_en(exp_en, exp_el, exp_premium, exp_lr)
        self._gs = None

    def _init_netceded(self, name, units, nc_agg, nc_kwargs, nc_views):
        """Initialise the ``netceded`` mode: one reinsured aggregate split into
        the joint per-occurrence law of a chosen view-pair of {gross, ceded, net}.

        Parameters
        ----------
        name : str
            Object name.
        units : list of tuple or None
            Either ``None`` (when ``nc_agg`` is supplied directly, the
            :meth:`aggregate.distributions.Aggregate.occ_bivariate` path) or a
            single ``('agg', name, spec)`` tuple (the DecL prefix path), from
            which the inner aggregate is built.
        nc_agg : Aggregate or None
            A pre-built (already updated) aggregate carrying occurrence
            reinsurance; if given it is used as-is.
        nc_kwargs : dict or None
            Axis-sizing overrides (``bs`` / ``log2_x`` / ``log2_y``) forwarded
            to :func:`build_netceded_joint`.
        nc_views : (str, str) or None
            The ``(x, y)`` axis view pair, each one of ``'gross'`` / ``'ceded'``
            / ``'net'``. ``None`` defaults to ``('net', 'ceded')`` (the
            ``netceded`` keyword). Axis 0 is ``views[0]``, axis 1 is ``views[1]``.
        """
        from .distributions import Aggregate

        self.copula = None
        self._views = tuple(nc_views) if nc_views else ('net', 'ceded')
        self.unit_names = [v.capitalize() for v in self._views]
        self._affine = [(False, 0.0), (False, 0.0)]
        self._nc_kwargs = {k: v for k, v in (nc_kwargs or {}).items()
                           if v is not None}
        #: The realized grid decision, set by the update (:class:`NetcededSizing`).
        self._nc_sizing = None
        if nc_agg is not None:
            self._nc_agg = nc_agg
            self._nc_built_here = False
        else:
            if not units:
                raise ValueError('netceded requires one component aggregate.')
            self._nc_agg = Aggregate(**units[0][2])
            self._nc_built_here = True

    def _init_discrete(self, name, dbv_xs, dbv_ys, dbv_S,
                       freq_name, freq_a, freq_b, freq_zm, freq_p0,
                       exp_en, exp_el, exp_premium, exp_lr):
        """Initialise ``discrete`` mode: the joint per-claim severity given directly.

        A ``dbvsev`` lattice supplies the joint per-claim probability matrix
        ``S`` (rows = X outcomes, columns = Y outcomes) on the explicit lattice
        ``(dbv_xs, dbv_ys)``. The discrete analogue of copula mode: everything is
        reused except the *formation* of ``S`` -- here it is given, not built from
        a copula. The two per-axis marginal units are 1-claim discrete aggregates
        whose per-event severity is the row / column sum of ``S``; they play the
        same role the copula-mode component aggregates do (axis sizing + the
        standalone validation targets). Loss / loss only -- no ``pnl`` axes.

        Parameters
        ----------
        name : str
            Object name.
        dbv_xs, dbv_ys : array-like
            The X / Y axis outcomes (the lattice).
        dbv_S : ndarray
            The ``len(xs) x len(ys)`` joint per-claim probability matrix,
            ``S[i, j] = P(X = xs[i], Y = ys[j])`` (already validated /
            normalised by the transformer).
        freq_name, freq_a, freq_b, freq_zm, freq_p0
            The shared outer frequency (a ``dfreq`` empirical count, or a named
            distribution with a count from the exposure clause).
        exp_en, exp_el, exp_premium, exp_lr
            Outer (shared-count) exposure for a non-empirical frequency.
        """
        from .distributions import Aggregate, Frequency

        if dbv_xs is None or dbv_ys is None or dbv_S is None:
            raise ValueError('discrete bivariate requires dbv_xs, dbv_ys, dbv_S.')
        xs = np.asarray(dbv_xs, dtype=float)
        ys = np.asarray(dbv_ys, dtype=float)
        S = np.asarray(dbv_S, dtype=float)
        if S.shape != (len(xs), len(ys)):
            raise ValueError(
                f'discrete bivariate: S shape {S.shape} does not match the '
                f'lattice {(len(xs), len(ys))}.')

        self.copula = None
        self._S_given = S
        self._dbv_xs = xs
        self._dbv_ys = ys
        self._affine = [(False, 0.0), (False, 0.0)]

        # per-event marginal severities = row / column sums of S
        g0 = S.sum(axis=1)
        g1 = S.sum(axis=0)
        self.unit_names = ['X', 'Y']
        # dsev-shaped partial specs, so _standalone_marginal rebuilds each axis's
        # marginal (shared freq compounded with g_i) for measure-don't-guess sizing.
        self._unit_specs = [
            {'name': 'X', 'sev_name': 'dhistogram', 'sev_xs': xs, 'sev_ps': g0},
            {'name': 'Y', 'sev_name': 'dhistogram', 'sev_xs': ys, 'sev_ps': g1},
        ]
        # 1-claim discrete aggregates whose aggregate density IS the per-event
        # severity g_i (units[i].n == 1); supply the per-event severity moments
        # and the per-axis sizing inputs, exactly as the copula-mode units do.
        self.units = [
            Aggregate(name='X', freq_name='fixed', exp_en=1,
                      sev_name='dhistogram', sev_xs=xs, sev_ps=g0),
            Aggregate(name='Y', freq_name='fixed', exp_en=1,
                      sev_name='dhistogram', sev_xs=ys, sev_ps=g1),
        ]

        self.frequency = Frequency(freq_name, freq_a, freq_b, freq_zm, freq_p0)
        self.freq_name = freq_name
        self._freq_kwargs = dict(freq_name=freq_name, freq_a=freq_a,
                                 freq_b=freq_b, freq_zm=freq_zm, freq_p0=freq_p0)
        self._sev_moms = [self._raw3(a.actual_m, a.actual_sd, a.actual_skew)
                          for a in self.units]
        # shared event count (an empirical dfreq carries its own mean; see
        # _resolve_en).
        self.en = self._resolve_en(exp_en, exp_el, exp_premium, exp_lr)
        # per-axis exact bucket size: the lattice gcd, so S scatters onto the
        # FFT grid with no rebucketing (the discrete analogue of dsev exactness).
        self._lattice_bs = [_lattice_bs(xs), _lattice_bs(ys)]
        self._gs = None

    # ------------------------------------------------------------------
    # moment / sizing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _raw3(m, sd, skew):
        """Raw moments ``(E[X], E[X^2], E[X^3])`` from ``(mean, sd, skew)``."""
        var = float(sd) ** 2
        s2 = var + m * m
        s3 = (skew * sd ** 3 if np.isfinite(skew) else 0.0) + 3 * m * s2 - 2 * m ** 3
        return (float(m), float(s2), float(s3))

    def _resolve_en(self, exp_en, exp_el, exp_premium, exp_lr):
        """Expected outer event count from the parsed exposure clause.

        An empirical (``dfreq``) shared frequency carries its own count (the
        mean of the empirical pmf); the ``exp_en`` sentinel is ``-1`` there, so
        read the count off the frequency moments instead of the exposure clause.
        """
        if getattr(self, 'frequency', None) is not None \
                and self.frequency.freq_name == 'empirical':
            return float(self.frequency.freq_moms(0)[0])
        if exp_en is not None and np.sum(np.asarray(exp_en, dtype=float)) > 0:
            return float(np.sum(np.asarray(exp_en, dtype=float)))
        # fall back to a loss/premium exposure: events = loss / per-event mean
        el = 0.0
        if exp_el is not None:
            el = float(np.sum(np.asarray(exp_el, dtype=float)))
        elif exp_premium is not None and exp_lr is not None:
            el = float(np.sum(np.asarray(exp_premium, dtype=float)
                              * np.asarray(exp_lr, dtype=float)))
        if el > 0:
            mean_per_event = sum(s[0] for s in self._sev_moms)
            if mean_per_event > 0:
                return el / mean_per_event
        raise ValueError(
            'bivariate: cannot determine the shared event count; supply a '
            'claim count (e.g. "25 claims") on the bivariate statement.')

    def _marginal_moments(self, i):
        """Analytic ``(mean, sd, skew)`` of component ``i``'s **loss** marginal.

        The outer compound of the per-event severity ``g_i`` -- the standalone
        aggregate for that unit in loss terms (before any ``pnl`` affine).
        """
        f1, f2, f3 = self.frequency.freq_moms(self.en)
        s1, s2, s3 = self._sev_moms[i]
        a1, a2, a3 = MomentAggregator.agg_from_fs(f1, f2, f3, s1, s2, s3)
        m, cv, skew = MomentAggregator.static_moments_to_mcvsk(a1, a2, a3)
        sd = (cv * m) if np.isfinite(cv) else float(np.sqrt(max(a2 - a1 * a1, 0.0)))
        return float(m), float(sd), float(skew)

    def _standalone_marginal(self, i):
        """Build the standalone **loss** marginal aggregate for axis ``i``.

        The component-``i`` marginal is the shared outer frequency compounded
        with that component's per-event severity, i.e. an ordinary 1-D
        :class:`Aggregate` with the **shared** frequency and an expected count
        ``en * (per-event trigger mean)``. This is the validation target the
        joint marginal must reproduce (exact for Poisson / mixed-Poisson /
        negative-binomial outer frequencies, where thinning preserves the
        family). It is built but **not updated** here -- the caller sizes it.

        Any ``pnl`` affine is *not* applied: the loss marginal is what the 2-D
        FFT runs on, and the per-axis affine relabels it afterwards
        (:func:`_affine_axis`), so the loss marginal is the correct thing to
        size for both axis kinds.

        Returns
        -------
        Aggregate
            The unupdated standalone loss marginal.
        """
        from .distributions import Aggregate

        spec = {k: v for k, v in self._unit_specs[i].items()
                if k.startswith('sev_') or k in ('name', 'note')}
        spec['exp_en'] = float(self.en) * float(self.units[i].n)
        spec.update(self._freq_kwargs)
        return Aggregate(**spec)

    def _measure_marginal_window(self, i, prob, log2):
        """Equal-tail support window of component ``i``'s realized loss marginal.

        Runs the standalone loss marginal and reads its
        :func:`~aggregate.utilities.balanced_window` at tail probability
        ``prob``. A **signed** marginal needs care: the 1-D sizer places its grid
        origin at the window's lower edge, so all the power-of-2 slack lands
        *above* the band and the cheap lower tail clips -- which biases the
        equal-tail measurement upward (a symmetric axis comes out off-centre). So
        a signed marginal is rebuilt on a *centred* grid (slack both sides, sized
        from the SBJ-aware first build's realized extent so a heavy tail is still
        covered) before measuring. Non-negative axes are bounded at 0 and need no
        recentre.

        Returns
        -------
        (lo, hi, bs) : tuple of float
            The measured window edges and the marginal's bucket size (the latter
            a resolution floor for the width).
        """
        a = self._standalone_marginal(i)
        a.update(log2=log2)
        ser = a.density_df.query('p_total > 0').p_total
        reflect, shift = self._affine[i]
        if a.x_min < 0 and not (reflect or shift):
            nz = ser.index[ser.values > 1e-13]
            mlo, mhi = float(nz.min()), float(nz.max())
            margin = 0.5 * (mhi - mlo)        # slack on both sides
            lo_box, hi_box = mlo - margin, mhi + margin
            bs0 = float(round_bucket((hi_box - lo_box) / (1 << log2)))
            a.update(log2=log2, bs=bs0, x_min=float(np.floor(lo_box / bs0) * bs0))
            ser = a.density_df.query('p_total > 0').p_total
        lo, hi = balanced_window(ser, prob)
        return lo, hi, float(a.bs)

    def _size_axes(self, total_log2, bs_axes, log2_axes, measure_log2=None):
        """Measure each axis's grid from its realized standalone marginal (§5).

        The bv's privilege is to *measure*, not guess. Each standalone loss
        marginal is run once on a fine grid; an equal-tail
        :func:`~aggregate.utilities.balanced_window` (discarded mass
        ``10**-window_nines``) reads its **support window** directly off the
        realized pmf. The total budget ``2**total_log2`` is then a pure
        *resolution* choice: the per-axis ``log2`` split is allocated so the two
        bucket sizes come out comparable (``L0 - L1 ~ log2(W0 / W1)``), and each
        ``bs`` is fit to its window, ``bs_i = round_bucket(W_i / (2**L_i - 1))``.

        Because the measured window always covers the deep (``window_nines``)
        tail, the budget controls only *resolution*: a smaller budget coarsens
        ``bs`` rather than clipping the support, so the joint mass is conserved
        regardless of budget (copula axes have free ``bs``).

        Parameters
        ----------
        total_log2 : int
            Total log2 cell budget (``2**total_log2`` cells across both axes)
            for the auto split; ignored for an axis with an explicit ``log2``.
        bs_axes : list of (float or None)
            Per-axis explicit ``bs``; ``None`` to measure that axis.
        log2_axes : list of (int or None)
            Per-axis explicit ``log2``; ``None`` to allocate that axis from the
            budget. With one axis pinned the other takes the rest of the budget.
        measure_log2 : int, optional
            Grid length (log2) for the standalone-marginal *measurement*
            updates. Default ``total_log2`` (the historical in-core behavior);
            the massive path caps it at :data:`_MASSIVE_MEASURE_LOG2` so a
            ``(16, 16)`` budget does not ask a 1-D marginal for ``2**32``
            buckets just to read its window.

        Returns
        -------
        bss, log2s, x_mins, his : list of float/int/float/float
            Per-axis bucket size, output grid log2, window origin, and window
            top. ``x_min`` is the *measured* lower edge -- negative on a signed
            axis, positive on a non-negative book whose mass lives far from 0
            (no artificial 0-pin), and 0 only when the mass genuinely reaches the
            origin or the axis is a ``pnl`` loss (see below). ``his`` lets the
            caller size the FFT buffer to reach physical 0.
        clipped : bool
            ``True`` if explicit per-axis ``log2`` overrides exceed the budget.
        """
        prob = 10.0 ** -_WINDOW_NINES
        if measure_log2 is None:
            measure_log2 = total_log2
        # 1. Measure each marginal's support window off the realized pmf.
        los, his, widths = [], [], []
        for i in range(2):
            lo, hi, bs0 = self._measure_marginal_window(i, prob, measure_log2)
            # Use the MEASURED lower edge -- do not pin to 0. The only 0-pin is a
            # pnl axis, whose loss has a known lower bound of 0 *and* whose
            # _affine_axis relabel assumes a 0-based loss grid; there the affine
            # owns the (tight) P&L windowing, so the loss axis stays 0-based.
            reflect, shift = self._affine[i]
            if reflect or shift:
                lo = 0.0
            los.append(lo)
            his.append(hi)
            widths.append(max(hi - lo, bs0))

        # 2. Budget split: only when BOTH axes are fully auto does the total
        #    budget get balanced across them (comparable bs). The moment either
        #    axis is overridden, each axis is sized independently to honour its
        #    override and cover its measured window; a free axis then takes half
        #    the budget.
        def _cover_log2(width, bs):
            return max(int(np.ceil(np.log2(width / bs + 1.0))), _MIN_AXIS_LOG2)

        free = [log2_axes[i] is None and bs_axes[i] is None for i in range(2)]
        if free[0] and free[1]:
            delta = np.log2(widths[0] / widths[1])
            cap = total_log2 - _MIN_AXIS_LOG2
            L0 = int(np.clip(round((total_log2 + delta) / 2.0), _MIN_AXIS_LOG2, cap))
            alloc = [L0, total_log2 - L0]
        else:
            alloc = [total_log2 // 2, total_log2 // 2]

        # 3. Per-axis (bs, log2): a pinned bs grows log2 to cover the window; a
        #    pinned log2 fits bs to the window; pinning both is the user's call.
        #    bs is rounded UP (round_bucket), never to nearest, even though grid
        #    size is at a premium here: the grid length N is a power of two, so a
        #    measured window of (say) width 78 lands on a 128-wide grid whatever
        #    bs is -- rounding bs *down* to the nearest rung cannot tighten that,
        #    it only fails to cover the window (clip) or forces a larger log2
        #    (more memory). The dead space is split symmetrically by the centred
        #    placement below instead.
        bss, log2s, x_mins = [], [], []
        for i in range(2):
            if log2_axes[i] is not None:
                L_i = int(log2_axes[i])
                bs_i = (float(bs_axes[i]) if bs_axes[i] is not None
                        else float(round_bucket(widths[i] / ((1 << L_i) - 1))))
            elif bs_axes[i] is not None:
                bs_i = float(bs_axes[i])
                L_i = _cover_log2(widths[i], bs_i)
            else:
                L_i = alloc[i]
                bs_i = float(round_bucket(widths[i] / ((1 << L_i) - 1)))
            log2s.append(L_i)
            bss.append(bs_i)
            # Centre the measured window in the (power-of-two) grid: split the
            # unavoidable slack ``N*bs - width`` equally either side rather than
            # piling it above (which left a symmetric axis looking off-centre).
            # A non-negative axis is clamped at 0 (no mass below, no point wasting
            # grid there). Snap the origin down to the grid so physical 0 stays
            # an integer index (the signed compound's j0 roll needs that).
            slack = (1 << L_i) * bs_i - (his[i] - los[i])
            xc = los[i] - 0.5 * slack
            if los[i] >= 0:
                xc = max(0.0, xc)
            x_mins.append(float(np.floor(xc / bs_i) * bs_i))

        # Coverage can only fail when BOTH bs and log2 are pinned too small.
        clipped = any(((1 << log2s[i]) - 1) * bss[i] < widths[i] - bss[i]
                      for i in range(2))
        if clipped:
            # A visible warning, not a logger line. Coverage fails only when
            # BOTH bs and log2 are pinned too small, and the resulting deficit
            # is not a sliver: the 2-D grid can miss almost the whole joint
            # (a measured 0.9999 on a 64x64 pin). The logger is silent by
            # default, so the one structural certainty that the answer is
            # wrong was the one signal nobody saw.
            warn_once(
                f'bivariate {self.name}: pinned (bs, log2) does not cover the '
                f'measured window on an axis, so the joint loses mass off the '
                f'grid -- raise log2, coarsen bs, or unpin one of them.',
                DefectiveDistributionWarning,
                key='defective-construction', stacklevel=3)
        return bss, log2s, x_mins, his, clipped

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------

    def update(self, log2=0, bs=0, padding=None, store_dir=None,
               row_chunk=512, col_chunk=512, keep_transform=False, **kwargs):
        """Build the joint density: size axes, discretise ``g_i``, 2D FFT.

        Parameters
        ----------
        log2 : int or (int, int), optional
            A **scalar** is the *total* 2-D grid budget in log2 cells
            (``2**log2`` cells, split between the axes by measured support);
            ``0`` (default) uses the :attr:`BivariateSettings.total_log2`
            config value (20). A **2-tuple ``(log2_x, log2_y)``** pins the
            per-axis log2 directly (the budget is then their sum) -- use it to
            explore a split (e.g. ``(11, 9)`` vs ``(10, 10)``). The auto split
            *falls out* of each realized marginal's measured support -- the bv
            measures, it does not guess (``dev/plan-mv.md`` §5). In-core,
            memory is quadratic in the per-axis length, so raise only a
            little; with ``store_dir`` the joint lives on disk and budgets
            like ``(14, 14)``--``(16, 16)`` are the point (``dev/plan-bv.md``).
        bs : float or (float, float), optional
            A **scalar** is applied to both axes; a **2-tuple ``(bs_x, bs_y)``**
            pins the per-axis bucket size. ``0`` (default) measures each axis's
            ``bs`` from its standalone marginal. (Both ``log2`` and ``bs`` tuple
            forms pass straight through ``build(..., log2=(a, b), bs=(x, y))``.)
        padding : int, optional
            FFT zero-padding factor per axis (mirrors the 1D aggregate; ``1``
            doubles each axis length for the transform). Default ``1``
            in-core; **``0`` on the massive path** (``store_dir`` set) -- the
            measured window already covers the support to
            ``10**-window_nines``, so padding would quadruple disk and time
            for no useful protection; ``deficit`` is the guard
            (plan-bv §4.4, settled 2026-07-02).
        store_dir : str, optional
            Backing directory for a **massive (disk-backed) update**: the
            realized joint density is streamed to ``density.zarr`` in this
            directory and never materialises in RAM (peak memory is bounded
            by a band regardless of grid size). ``None`` (default) is today's
            in-core path, untouched. Requires the optional ``zarr`` extra
            (``pip install aggregate[massive]``). Results surface through
            :attr:`bivariate` as a :class:`MassiveBivariateDistribution`;
            ``self.density`` stays ``None``. Prefer a fast local drive with
            room for the transient staging store (~2x the density).
        row_chunk, col_chunk : int, optional
            Massive-path band / tile sizes (ignored in-core). Peak RAM ~
            ``max(row_chunk * M1, M0 * col_chunk) * 16`` bytes.
        keep_transform : bool, optional
            Massive path: keep the ``z1`` / ``z2`` staging stores after the
            update (default deletes them; ``z2`` is as large as the density).
        **kwargs
            Ignored (accepted for a uniform ``build`` call signature).

        Notes
        -----
        Axis sizing is *measured*, not guessed: each component's standalone loss
        marginal is run first, an equal-tail :func:`balanced_window` trims each
        realized pmf, and the per-axis ``(bs, log2, x_min)`` is read off the
        measured support (:meth:`_size_axes`). A signed (``ssev``) axis takes a
        two-sided window with a negative origin; :meth:`update_work` then
        compounds it with the same ``i0``/``j0`` wrap-and-roll the 1-D
        :meth:`aggregate.distributions.Aggregate._fft_aggregate` uses, so its
        negative tail no longer wraps. In ``netceded`` mode the joint is built
        from the single reinsured aggregate's comonotone scatter
        (:func:`build_netceded_joint`) instead, and ``log2`` / ``bs`` size
        whichever grid this object owns: the inner aggregate's on the DecL
        prefix route, the joint's on the
        :meth:`aggregate.distributions.Aggregate.occ_bivariate` route (see
        :meth:`_netceded_sizing_kwargs`).
        """
        # massive default padding 0 (measured window + deficit guard);
        # in-core default 1, both overridable.
        self.padding = (int(padding) if padding is not None
                        else (0 if store_dir is not None else 1))
        # a re-update replaces any previous result surface, either kind
        self._massive = None
        self._massive_dist = None
        self._total = None
        if self.mode == 'netceded':
            return self._update_netceded(log2=log2, bs=bs, store_dir=store_dir,
                                         row_chunk=row_chunk,
                                         col_chunk=col_chunk,
                                         keep_transform=keep_transform)
        # Parse scalar-or-(x, y) sizing args into per-axis overrides + budget.
        # Tuple entries may be None/0 to leave that axis auto.
        if isinstance(log2, (tuple, list)):
            log2_axes = [int(v) if v else None for v in log2]
            total_log2 = sum(int(v) for v in log2 if v) or _TOTAL_LOG2
        else:
            log2_axes = [None, None]
            total_log2 = int(log2) if log2 else _TOTAL_LOG2
        if isinstance(bs, (tuple, list)):
            bs_axes = [float(v) if v else None for v in bs]
        elif bs:
            bs_axes = [float(bs), float(bs)]
        else:
            bs_axes = [None, None]
        # discrete mode: default each axis's bs to the lattice gcd so the given
        # joint matrix scatters onto the grid exactly (a pinned bs lets _size_axes
        # grow log2 to cover the measured aggregate window). An explicit override
        # still wins.
        if self.mode == 'discrete':
            bs_axes = [bs_axes[i] if bs_axes[i] is not None else self._lattice_bs[i]
                       for i in range(2)]
        measure_log2 = (min(total_log2, _MASSIVE_MEASURE_LOG2)
                        if store_dir is not None else None)
        bss, log2s, x_mins, his, clipped = self._size_axes(
            total_log2, bs_axes, log2_axes, measure_log2=measure_log2)
        self._clipped = clipped
        self._gs = []
        self._i0 = []                    # per-event severity negative reach (lay-in wrap)
        self._j0 = []                    # output-window origin in buckets (final roll)
        self._mlog2 = []                 # log2 FFT buffer length per axis
        self._nout = [1 << L for L in log2s]
        for i, a in enumerate(self.units):
            # Build the per-event severity g_i on its OWN natural grid at the
            # measured resolution (x_min='auto' so a signed component keeps its
            # negative buckets). The output window is measured separately from
            # the marginal -- the severity's negative reach (i0) and the output
            # origin (j0) are independent, exactly as in the 1-D _fft_aggregate.
            a.update(log2=log2s[i], bs=bss[i])
            self.bs[i] = float(bss[i])
            self._gs.append(np.asarray(a.agg_density, dtype=float).copy())
            self._i0.append(int(round(-a.x_min / a.bs)) if a.x_min else 0)
            self._j0.append(int(round(x_mins[i] / bss[i])) if x_mins[i] else 0)
            self.axis_xs[i] = (x_mins[i]
                               + np.arange(self._nout[i], dtype=float) * bss[i])
            # The compound is anchored at physical 0 (non-negative severity) or
            # wraps the negatives (signed), so the FFT buffer must reach from
            # min(0, x_min) up to the grid top -- DECOUPLED from the output
            # length so a tight window far from 0 doesn't alias the upper tail.
            grid_top = x_mins[i] + self._nout[i] * bss[i]
            reach = grid_top - min(0.0, x_mins[i])
            need = int(np.ceil(np.log2(max(reach / bss[i], 2.0))))
            self._mlog2.append(max(need, log2s[i]) + self.padding)
        # severity grids are the per-event grids (signed where the component is),
        # captured for the severity panel of plot().
        self._sev_xs = [np.asarray(a.xs, dtype=float).copy() for a in self.units]
        if store_dir is not None:
            return self._update_massive(store_dir, row_chunk, col_chunk,
                                        keep_transform)
        self.update_work()
        return self

    def _netceded_sizing_kwargs(self, log2, bs):
        """Joint sizing keywords for this update, ``_nc_kwargs`` under overrides.

        ``log2`` and ``bs`` mean different things on the two netceded routes,
        and the difference is which grid the caller owns.

        On the DecL prefix route (``netceded agg ...``) this object built the
        inner aggregate, so ``build(..., log2=16, bs=1)`` is sizing **that**
        aggregate's 1-D grid, as it does for every other DecL form; the joint
        is sized from the constructor's ``nc_kwargs``. On the
        :meth:`aggregate.distributions.Aggregate.occ_bivariate` route the inner
        aggregate arrives already updated and its grid is not this object's to
        change, so the same two keywords size the **joint**: a scalar ``log2``
        is the 2-D cell budget, a ``(log2_x, log2_y)`` pair pins the axes, and
        ``bs`` is the common bucket size.

        That second reading is what makes ``update(log2=...)`` on a netceded
        joint do something. It was a no-op through 1.0.0a276 (``_nc_kwargs``
        was read whatever the caller passed), while the over-budget warning
        recommended it by name.
        """
        kw = dict(self._nc_kwargs)
        if self._nc_built_here:
            return kw
        if isinstance(log2, (tuple, list)):
            if len(log2) != 2:
                raise ValueError(
                    f'a netceded log2 pair is (log2_x, log2_y); got {log2!r}.')
            kw['log2_x'] = int(log2[0]) or None
            kw['log2_y'] = int(log2[1]) or None
        elif log2:
            kw['total_log2'] = int(log2)
        if bs:
            if isinstance(bs, (tuple, list)):
                raise ValueError(
                    'a netceded joint carries one common bs on both axes '
                    f'(the comonotone curve couples them); got {bs!r}.')
            kw['bs'] = float(bs)
        return kw

    def _update_netceded(self, log2=0, bs=0, store_dir=None,
                         row_chunk=512, col_chunk=512, keep_transform=False):
        """Build the joint (ceded, net) density of the single reinsured agg.

        Updates the inner aggregate (only if this object built it -- a
        pre-built agg passed by :meth:`Aggregate.occ_bivariate` must already be
        updated, so the precondition raises propagate), then assembles the joint
        via :func:`build_netceded_joint` and caches the per-axis ceded / net
        theoretical moments from the inner ``reins_stats_df``. With
        ``store_dir`` the comonotone scatter goes to a scipy.sparse matrix and
        the compound streams through the out-of-core kernel instead
        (``dev/plan-bv.md`` §4.4) -- the joint never materialises in RAM.

        ``log2`` and ``bs`` size the inner aggregate or the joint according to
        which route built this object; see :meth:`_netceded_sizing_kwargs`.
        """
        a = self._nc_agg
        if self._nc_built_here and a.sev_density_gross is None:
            kw = {}
            if log2:
                kw['log2'] = int(log2)
            if bs:
                kw['bs'] = bs
            a.update(**kw)
        nc = self._netceded_sizing_kwargs(log2, bs)
        # a netceded joint refuses an unhonorable pin rather than clipping one
        self._clipped = False
        if store_dir is not None:
            from ._aggregate_compute_massive import (
                massive_bivariate_convolution, PyramidBuilder)
            sizing = _netceded_sizing(a, self._views, **nc)
            self._nc_sizing = sizing
            bs_c, n0, n1 = sizing.bs, sizing.n0, sizing.n1
            sev2 = scatter_bivariate_sparse(sizing.x0, sizing.x1, sizing.mass,
                                            bs_c, bs_c, n0, n1,
                                            scheme=a.reins_bucket)
            gx = bs_c * np.arange(n0)
            gy = bs_c * np.arange(n1)
            mlog2 = (int(np.log2(n0)) + self.padding,
                     int(np.log2(n1)) + self.padding)
            pyramid = PyramidBuilder(store_dir, n0, n1,
                                     row_chunk=row_chunk, col_chunk=col_chunk)
            res = massive_bivariate_convolution(
                sev2, a.frequency.freq_pgf, float(a.base_mean),
                N0=n0, N1=n1, bs0=bs_c, bs1=bs_c,
                i0=(0, 0), j0=(0, 0), mlog2=mlog2,
                store_dir=store_dir, xs0=gx, xs1=gy,
                row_chunk=row_chunk, col_chunk=col_chunk,
                keep_transform=keep_transform, on_band=pyramid.on_band)
            pyramid.finish()
            self.density = None
            self.axis_xs = [gx, gy]
            self.bs = [bs_c, bs_c]
            self._S = sev2
            self._sev_xs = [gx, gy]   # severity panel = comonotone view scatter
            self.deficit = float(res.deficit)
            self._marg_theory = self._netceded_theory(a, self._views)
            self._finish_massive(res)
            return self
        density, gx, gy, bs_x, bs_y, sev2, deficit, sizing = build_netceded_joint(
            a, views=self._views, **nc)
        self.density = density
        self.axis_xs = [gx, gy]
        self.bs = [bs_x, bs_y]
        self._S = sev2
        self._sev_xs = [gx, gy]   # severity panel = comonotone view scatter
        self.deficit = deficit
        self._nc_sizing = sizing
        self._marg_theory = self._netceded_theory(a, self._views)
        return self

    @staticmethod
    def _netceded_theory(agg, views):
        """Per-axis ``(mean, sd, skew)`` of the chosen occ view aggregates,
        read from the inner aggregate's :attr:`reins_stats_df` (the matching
        ``Gross`` / ``Ceded`` / ``Net`` occurrence column for each view)."""
        rs = agg.reins_stats_df
        out = []
        for view in views:
            col = ('occ', _VIEW_STAT_COL[view])
            m = float(rs.loc[('agg', 'mean'), col])
            cv = float(rs.loc[('agg', 'cv'), col])
            sk = float(rs.loc[('agg', 'skew'), col])
            out.append((m, cv * m, sk))
        return out

    @staticmethod
    def _lay_signed_2d(S, i0_0, i0_1, M0, M1):
        """Lay per-claim severity ``S`` into an ``(M0, M1)`` FFT buffer, physical 0 at index 0.

        The 2-D analogue of the signed/windowed lay-in of
        :meth:`aggregate.distributions.Aggregate._fft_aggregate`: per axis the
        ``i0`` negative-physical buckets of the severity (array indices
        ``0..i0-1``) wrap to the top of the padded length ``M`` (period
        ``M*bs``), so the FFT-PGF compound treats physical ``0`` as the additive
        identity on each axis. Reduces to a plain zero-pad when both ``i0`` are 0.
        """
        n0, n1 = S.shape
        out = np.zeros((M0, M1))
        out[:n0 - i0_0, :n1 - i0_1] = S[i0_0:, i0_1:]
        if i0_1:
            out[:n0 - i0_0, M1 - i0_1:] = S[i0_0:, :i0_1]
        if i0_0:
            out[M0 - i0_0:, :n1 - i0_1] = S[:i0_0, i0_1:]
        if i0_0 and i0_1:
            out[M0 - i0_0:, M1 - i0_1:] = S[:i0_0, :i0_1]
        return out

    def _form_severity_matrix(self, N0, N1, massive=False):
        """Form the joint per-claim severity ``S`` (copula / discrete modes).

        The mode-specific severity *formation*, shared by the in-core
        :meth:`update_work` and the massive :meth:`_update_massive` paths.
        Discrete mode scatters the given lattice matrix onto the output grid
        at exact bucket indices (the lattice ``bs`` is the gcd); copula mode
        builds the discrete-Sklar rectangle mass from the per-event marginal
        CDFs (the Bernoulli zero-inflation is automatic as a jump in ``G``).

        With ``massive=True`` the result is RAM-bounded instead of full-grid
        dense: the discrete scatter goes to scipy.sparse, and the copula
        rectangle mass is **trimmed to the per-event support** -- ``G_i`` is
        cut where its remaining tail is below ``1e-15`` (the kernel's clamp
        threshold), so the dropped rectangle mass is ``< 2e-15`` and shows up
        (harmlessly) in ``deficit`` -- because the untrimmed ``(N0, N1)``
        rectangle matrix is exactly the allocation the disk-backed path
        exists to avoid. ``self._sev_xs`` is trimmed to match.

        Parameters
        ----------
        N0, N1 : int
            Output grid lengths.
        massive : bool, default False
            Return the RAM-bounded (sparse / trimmed) form.

        Returns
        -------
        ndarray or scipy.sparse.csr_matrix
            The per-claim severity, stored on ``self._S`` (retained for the
            severity panel of ``plot()`` and the ``Sev`` dependence row).
        """
        if self._S_given is not None:
            # discrete mode: scatter the given joint per-claim matrix onto the
            # (N0, N1) output grid at each lattice point's bucket index. The
            # lattice bs divides every atom (gcd), so the indices are exact and
            # the per-event-severity row/column sums reproduce g0 / g1 on the grid.
            # The i0 offset puts the scatter in the natural signed convention
            # the lay-in expects (row 0 = x_min = -i0*bs): _lay_signed_2d wraps
            # the first i0 rows/columns itself, so scattering a negative atom at
            # its raw (negative, numpy-wrapped) index would wrap it twice.
            ix = np.round(self._dbv_xs / self.bs[0]).astype(int) + self._i0[0]
            iy = np.round(self._dbv_ys / self.bs[1]).astype(int) + self._i0[1]
            if massive:
                rows = np.repeat(ix, len(iy))
                cols = np.tile(iy, len(ix))
                S = ssp.coo_matrix((np.asarray(self._S_given, dtype=float).ravel(),
                                    (rows, cols)), shape=(N0, N1)).tocsr()
            else:
                S = np.zeros((N0, N1))
                S[np.ix_(ix, iy)] = self._S_given
        else:
            g0, g1 = self._gs
            G0 = np.cumsum(g0)
            G1 = np.cumsum(g1)
            if massive:
                # trim each CDF where the remaining per-event tail is fp dust
                # (see the docstring); keep at least the signed lay-in rows.
                k0 = min(int(np.searchsorted(G0, 1.0 - 1e-15)) + 1, len(G0))
                k1 = min(int(np.searchsorted(G1, 1.0 - 1e-15)) + 1, len(G1))
                k0 = max(k0, self._i0[0] + 1)
                k1 = max(k1, self._i0[1] + 1)
                G0, G1 = G0[:k0], G1[:k1]
                self._sev_xs = [self._sev_xs[0][:k0], self._sev_xs[1][:k1]]
            # joint per-claim severity via the copula (marginals exact by
            # construction); retained for the severity panel of plot().
            S = self.copula.rectangle_pmf(G0, G1)
        self._S = S
        return S

    def _update_massive(self, store_dir, row_chunk, col_chunk, keep_transform):
        """Run the disk-backed (out-of-core) update for the copula / discrete
        modes and install the massive result surface.

        Same sizing, severity formation and ``i0``/``j0``/``mlog2`` windowing
        as :meth:`update_work`; only the FFT-PGF-iFFT core is delegated to
        :func:`~aggregate._aggregate_compute_massive.massive_bivariate_convolution`,
        which streams the joint to ``store_dir`` and folds the marginals /
        mixed moments band by band (``dev/plan-bv.md`` §4).
        """
        from ._aggregate_compute_massive import (massive_bivariate_convolution,
                                                 PyramidBuilder)

        N0, N1 = self._nout
        S = self._form_severity_matrix(N0, N1, massive=True)
        pyramid = PyramidBuilder(store_dir, N0, N1,
                                 row_chunk=row_chunk, col_chunk=col_chunk)
        res = massive_bivariate_convolution(
            S, self.frequency.freq_pgf, self.en,
            N0=N0, N1=N1, bs0=self.bs[0], bs1=self.bs[1],
            i0=tuple(self._i0), j0=tuple(self._j0), mlog2=tuple(self._mlog2),
            store_dir=store_dir, xs0=self.axis_xs[0], xs1=self.axis_xs[1],
            row_chunk=row_chunk, col_chunk=col_chunk,
            keep_transform=keep_transform, on_band=pyramid.on_band)
        pyramid.finish()
        self.density = None
        self.deficit = float(res.deficit)
        self._marg_theory = [self._marginal_moments(0), self._marginal_moments(1)]
        self._finish_massive(res)
        return self

    def _finish_massive(self, res):
        """Install the massive result surface: keep the kernel result, build
        the :class:`MassiveBivariateDistribution` container and persist the
        store metadata so :meth:`MassiveBivariateDistribution.reopen` can
        reconstruct it in a later session without re-running the FFT."""
        if self.mode == 'netceded':
            en, fname = float(self._nc_agg.n), self._nc_agg.frequency.freq_name
            cop = 'netceded'
        else:
            en, fname = float(self.en), self.freq_name
            cop = self.mode if self.copula is None else str(self.copula)
        meta = {'name': self.name, 'en': en, 'freq_name': fname,
                'copula': cop, 'deficit': float(res.deficit),
                'axis_names': tuple(self.unit_names), 'mode': self.mode,
                'padding': self.padding,
                'log2': [int(np.log2(len(res.xs0))), int(np.log2(len(res.xs1)))]}
        self._massive = res
        self._massive_dist = MassiveBivariateDistribution.from_result(res, meta)
        self._massive_dist.save()

    def update_work(self):
        """Assemble the joint density from the per-event severities ``g_i``.

        Builds the copula joint per-claim severity ``S`` (discrete Sklar), runs
        the 2D compound FFT (``freq_pgf`` applied elementwise to ``rfft2(S)``),
        then applies any per-axis ``pnl`` affine.

        The 2-D lift of the 1-D ``i0`` / ``j0`` machinery
        (:meth:`aggregate.distributions.Aggregate._fft_aggregate`): the
        per-claim severity is laid into the padded buffer with physical 0 at
        index 0 (each axis's ``i0`` negative-severity buckets wrapped to the top,
        :meth:`_lay_signed_2d`), the shared frequency is applied elementwise, and
        the finished density is relabelled onto each axis's output window by an
        ``np.roll`` of ``-j0``. Relabelling a finished array carries no ``N*s``
        shift, so it is correct for random as well as fixed frequency. The FFT
        buffer length ``M`` (:attr:`_mlog2`) is sized to reach physical 0 from
        the window, *decoupled* from the output length ``N`` so a tight window
        far from 0 does not alias. On a mass-at-0, 0-based grid (``i0 == j0 ==
        0``) the lay-in is a plain zero-pad and the roll is a no-op.
        """
        g0, g1 = self._gs
        i0_0, i0_1 = getattr(self, '_i0', [0, 0])
        j0_0, j0_1 = getattr(self, '_j0', [0, 0])
        N0, N1 = getattr(self, '_nout', [len(g0), len(g1)])
        mlog2 = getattr(self, '_mlog2', [int(np.log2(N0)) + self.padding,
                                         int(np.log2(N1)) + self.padding])
        S = self._form_severity_matrix(N0, N1)

        if self.en == 0:
            density = np.zeros((N0, N1))
            density[-j0_0, -j0_1] = 1.0   # point mass at physical (0, 0)
        else:
            M0, M1 = 1 << mlog2[0], 1 << mlog2[1]
            buf = self._lay_signed_2d(S, i0_0, i0_1, M0, M1)
            z = sfft.rfft2(buf)
            # freq_pgf is mathematically elementwise in z, but the empirical
            # implementation assumes a 1D argument -- flatten, apply, reshape.
            ftagg = self.frequency.freq_pgf(self.en, z.ravel()).reshape(z.shape)
            a = np.real(sfft.irfft2(ftagg, s=(M0, M1)))
            # relabel physical 0 (buffer index 0) onto each axis output window
            a = np.roll(a, (-j0_0, -j0_1), axis=(0, 1))
            density = a[:N0, :N1]

        _clip_density_fuzz(density, f'bivariate {self.name!r}')

        self.density = density
        self.deficit = float(1.0 - density.sum())
        # cache per-axis loss-marginal theory.
        self._marg_theory = [self._marginal_moments(0), self._marginal_moments(1)]

    # ------------------------------------------------------------------
    # results: marginals, moments, correlation, frames
    # ------------------------------------------------------------------

    def _require_density(self):
        if self.density is None and self._massive is None:
            raise ValueError('BivariateAggregate not updated; call update().')

    @property
    def bivariate(self):
        """A :class:`BivariateDistribution` view of the joint density.

        Reuses the shared container for moments, correlation, contour and the
        HTML repr. After a massive update (``update(store_dir=...)``) this is
        the :class:`MassiveBivariateDistribution` built at update time -- same
        role accessors, disk-backed density.
        """
        self._require_density()
        if self._massive_dist is not None:
            return self._massive_dist
        if self.mode == 'netceded':
            en, fname = float(self._nc_agg.n), self._nc_agg.frequency.freq_name
            cop = 'netceded'
        else:
            en, fname, cop = float(self.en), self.freq_name, str(self.copula)
        meta = {'name': self.name, 'en': en, 'freq_name': fname,
                'copula': cop, 'deficit': self.deficit,
                'axis_names': tuple(self.unit_names)}
        return BivariateDistribution(self.density, self.axis_xs[0],
                                     self.axis_xs[1], self.bs[0], self.bs[1],
                                     meta)

    # ------------------------------------------------------------------
    # Label surface (LabeledMixin hooks) -- a171. The exhibit axis is the two
    # component units, and each one's label is its component Aggregate's
    # resolved ``label``, exactly as Portfolio resolves its member units.
    #
    # A ``bivariate`` statement has no ``as`` clause in the grammar yet, so an
    # object-level label is set programmatically (``label=``) rather than in
    # DecL; see the [Bivariate-DecL-Label] item in dev/TODO.md. The component
    # labels DO come through DecL, because each unit is an ordinary ``agg``.
    # ------------------------------------------------------------------
    def _label_handles(self):
        """The exhibit axis: the two component unit handles."""
        return list(self.unit_names)

    def _resolve_handle_label(self, handle):
        """A unit handle -> its component ``Aggregate``'s resolved label.

        Falls back to the handle itself, which covers both an unlabeled
        component and the view-pair modes (``netceded`` / ``grossnet`` /
        ``grossceded``), whose axes are view names rather than units.
        """
        for a in getattr(self, 'units', None) or []:
            if getattr(a, 'name', None) == handle:
                return a.label
        return handle

    @property
    def marginals(self):
        """Return the two marginal densities ``(density.sum(1), density.sum(0))``.

        Each reproduces the standalone aggregate for that component (a ``pnl``
        axis reproduces the standalone ``pnl``). After a massive update these
        are the exact pass-3 accumulators -- no disk read.
        """
        self._require_density()
        if self._massive is not None:
            return self._massive.marg0, self._massive.marg1
        return self.density.sum(axis=1), self.density.sum(axis=0)

    def marginal(self, axis=0):
        """Axis marginal as a :class:`~aggregate._grid_distribution.GridDistribution`.

        The probability-object companion to the raw-array :attr:`marginals`:
        quantiles, TVaR and return periods on one axis are one call away.

        Parameters
        ----------
        axis : int or str, default 0
            ``0`` / ``1``, the aliases ``'x'`` / ``'y'``, or a component name
            from :attr:`unit_names` (case insensitive, so a netceded joint
            answers ``marginal('net')`` / ``marginal('ceded')``).

        Returns
        -------
        GridDistribution
            The realized marginal law on the axis grid, named with the
            component's resolved label. A ``pnl`` axis is returned with
            ``is_loss_value=False`` so its return periods read the correct
            tail.

        Notes
        -----
        Delegates through :attr:`bivariate`, so the in-core and massive
        routes share one entry point (the massive marginal is the free
        pass-3 accumulator, no disk read). Each marginal reproduces the
        standalone aggregate for its component up to the joint grid's
        rebucketing, which is the showpiece invariant of the 2-D FFT.
        """
        i = _resolve_axis(axis, self.unit_names)
        self._require_density()
        return self.bivariate.marginal(
            i, name=self._resolve_handle_label(self.unit_names[i]),
            is_loss_value=self._axis_kind(i) != 'pnl')

    def conditional(self, kind, value, report=None):
        """Conditional law of one axis given an event on the joint.

        Parameters
        ----------
        kind : {'x', 'y', 'x+y', 'x-y'}
            The conditioning variable: one axis (``'x'`` conditions on axis 0
            and reports axis 1; ``'y'`` the transpose), the total, or the
            difference; the event is the bucket containing ``value``.
        value : float
            The conditioning value, snapped to its bucket.
        report : int or str, optional
            Which axis's law the returned distribution is expressed in
            (``0`` / ``1`` / ``'x'`` / ``'y'`` or a name from
            :attr:`unit_names`). Determined for ``'x'`` / ``'y'`` (the non
            conditioning axis; a contradiction raises); defaults to axis 0
            for the diagonal kinds, whose two readings are affine images of
            each other (``y = value - x`` on the band).

        Returns
        -------
        GridDistribution
            The normalized conditional law on the reported axis's grid, with
            the orientation flag of that axis (``is_loss_value=False`` for a
            ``pnl`` axis).

        Raises
        ------
        ValueError
            On an unknown ``kind``, a contradictory ``report``, or a
            conditioning event carrying no mass.

        Notes
        -----
        Delegates to :meth:`JointBandsMixin.conditional`, which owns the
        band arithmetic and its documentation. The related sweeps:
        :meth:`exeqa_df` is the conditional **mean** over the whole
        conditioning grid, this is the full law at one point;
        ``[Bivariate-Total-Exeqa]`` (deferred) is the mean sweep on the
        total's own grid, whose band extraction the diagonal kinds here
        already implement one value at a time.
        """
        kindn = str(kind).lower().replace(' ', '')
        if kindn not in _CONDITIONAL_KINDS:
            raise ValueError(
                f'kind must be one of {_CONDITIONAL_KINDS}; got {kind!r}.')
        self._require_density()
        if kindn in ('x', 'y'):
            rep = 1 if kindn == 'x' else 0
        else:
            rep = 0 if report is None else _resolve_axis(report,
                                                         self.unit_names)
        return self.bivariate.conditional(
            kindn, value, report=report,
            is_loss_value=self._axis_kind(rep) != 'pnl')

    @property
    def total(self):
        """Realized law of ``X + Y`` as a cached ``GridDistribution``.

        The full distribution of the dependent total, not just its moments
        (those are :meth:`_total_agg_empirical`): percentiles, TVaR and
        return periods of the sum are read off this object.

        Returns
        -------
        GridDistribution
            The realized total on its own grid, named ``'total'``. Cached;
            recomputed after :meth:`update`, the same discipline as the
            other realized frames. Orientation: a loss unless **both** axes
            are ``pnl`` (payoff) axes.

        Notes
        -----
        Computed by :meth:`JointBandsMixin.total`: an exact anti-diagonal
        fold where the axes share a ``bs`` (the netceded and discrete
        modes), and mean-preserving :func:`_scatter_1d` routing onto
        ``bs_total = max(bs0, bs1)`` where they differ. In netceded mode
        the total has an exact reference, the gross aggregate
        (``ceded + net = gross`` per occurrence), which the tests pin.
        """
        if self._total is None:
            self._require_density()
            is_loss = any(self._axis_kind(i) != 'pnl' for i in range(2))
            self._total = self.bivariate.total(name='total',
                                               is_loss_value=is_loss)
        return self._total

    def moments(self, max_order=3):
        """Mixed raw moments ``E[A0^i A1^j]`` (delegates to the bivariate view)."""
        return self.bivariate.moments(max_order)

    @property
    def corr(self):
        """Pearson correlation of the two component aggregates.

        Notes
        -----
        This is the **realised output** correlation, which is *not* the copula
        parameter: compounding by the shared frequency attenuates the per-claim
        dependence (and a shared mixing frequency adds common-shock dependence on
        top). Compare with :meth:`Copula.tau` via :attr:`dependency_df`.
        """
        return self.bivariate.corr()

    @property
    def density_df(self):
        """Joint density as a DataFrame (a thin wrapper around ``density``).

        Returns
        -------
        DataFrame
            ``density[i, j]`` with the axis-0 grid as the index and the axis-1
            grid as the columns, labelled by the two component names.
        """
        self._require_density()
        if self.density is None:
            raise ValueError(
                'the joint density is disk-backed (massive update); a full '
                'DataFrame would read the whole store. Slice the zarr view '
                'instead: self.bivariate.density[rows, cols].')
        return pd.DataFrame(
            self.density,
            index=pd.Index(self.axis_xs[0], name=self.unit_names[0]),
            columns=pd.Index(self.axis_xs[1], name=self.unit_names[1]))

    @staticmethod
    def _quantile_column(level, other_name):
        """Column name for one conditional quantile: ``q99_Ceded``.

        Follows the existing ``exeqa_<axis>`` naming, which already carries the
        axis name, so a band frame reads as one family. The level is written as
        a percentage, zero padded to two digits where it is a whole one
        (``q01``, ``q99``) and with its decimals otherwise (``q0.5``), so two
        nearby levels cannot collide on one column.
        """
        pp = f'{level * 100:g}'
        if '.' not in pp and 'e' not in pp:
            pp = pp.zfill(2)
        return f'q{pp}_{other_name}'

    def exeqa_df(self, axis=0, levels=None, cdf_range=None):
        r"""The kappa curve: conditional means given one axis, over its whole grid.

        The bivariate answer to Portfolio's ``exeqa_*`` columns. Portfolio
        computes ``E[X_i | X = x]`` by the FFT trick, which assumes the units
        are independent; the two axes here are **dependent** by construction
        (a shared claim count, a comonotone per-claim cession), so that route
        is unavailable and the conditional mean comes straight out of the
        joint: one matrix vector product per direction.

        Parameters
        ----------
        axis : {0, 1}, default 0
            The **conditioning** axis. ``0`` conditions on the axis-0 variable
            and reports the conditional mean of axis 1; ``1`` is the transpose.
        levels : sequence of float, optional
            Probability levels in ``[0, 1]``. Each one adds a
            ``q<pp>_<other axis>`` column holding the conditional **quantile**
            of the other axis given the conditioning one, which is what turns
            the kappa curve into a curve with a band. ``None`` (the default)
            keeps today's columns exactly, so this is additive.
        cdf_range : (float, float), optional
            Crop to the rows between these two probabilities of the
            conditioning marginal, and compute the quantile columns only
            there. The quantiles are the expensive part of the sweep (one
            :class:`~aggregate._grid_distribution.GridDistribution` per live
            row: 42 s over a 65,536 row grid, against 0.8 s to build the joint
            it reads), and a plotted range is a few thousand rows rather than
            sixty five thousand. Cropping rather than blanking, so a ``NaN``
            keeps meaning "no mass here" and never "not measured here".
            ``F`` and ``S`` are the whole distribution's, not the crop's.

        Returns
        -------
        pandas.DataFrame
            Indexed by the conditioning axis grid (index named for that axis).
            Columns:

            ``p``
                The conditioning marginal mass at each grid point.
            ``F``, ``S``
                Its CDF and survival function, read through
                :class:`~aggregate.GridDistribution` so the probability
                vocabulary is the library's one implementation. Under a grid
                deficit ``S`` ends at the deficit rather than at zero, which
                is the honest reading of a truncated law.
            ``exeqa_<conditioning axis>``
                The identity ``E[X | X = x] = x``, carried so that a
                decomposition check is a column subtraction (Portfolio carries
                ``exeqa_total`` for the same reason).
            ``exeqa_<other axis>``
                The kappa curve ``E[Y | X = x]``.

            ``q<pp>_<other axis>``
                One per requested level: the conditional quantile of the other
                axis. Absent when ``levels`` is ``None``.

            Every conditional column is ``NaN`` where the conditioning row
            carries no mass, with ``p`` saying why: a conditional expectation
            given a null event has no value, and a zero filled there would be
            read as one. The joint is de-fuzzed at construction
            (:func:`_clip_density_fuzz`), so ``p > 0`` is an exact test rather
            than a threshold.

        Raises
        ------
        ValueError
            If the object has not been updated, if ``axis`` is not 0 or 1, or
            if a level is outside ``[0, 1]``.

        Notes
        -----
        **The arithmetic.** With joint mass ``d[i, j]`` on grids ``x``, ``y``,
        the conditioning marginal is ``p_i = sum_j d[i, j]`` and the numerator
        is ``num_i = sum_j y_j d[i, j]``, so ``kappa_i = num_i / p_i``. As a
        matrix vector product that is ``d @ y`` against ``d.sum(1)``, and the
        transpose for ``axis=1``. Reads the joint through the existing
        accessors and adds no state.

        **It is a row-wise fold, so where the density lives does not matter.**
        The sweep runs over :meth:`JointBandsMixin._row_bands`, which yields
        one band in core and the store's own row chunk on the massive route, so
        a disk backed joint answers this at bounded memory rather than refusing
        it. That refusal stood until 1.0.0a279 and took
        :meth:`natural_allocation` down with it, which was the one hole in an
        otherwise first class massive surface.

        **Why the quantiles route through GridDistribution.** One per live row,
        on that row's own normalized mass, so the probability vocabulary stays
        the library's single implementation and a lattice law's atoms are
        handled the way every other quantile in the package handles them. The
        cost is real (it is the per row construction, not the disk) and it is
        paid only when ``levels`` is asked for. Quantiles commute with the
        monotone map ``c -> c / g`` at fixed ``g``, so a **share** band is the
        value band divided by the index and needs no separate pass.

        **What a band says that no mean can.** The gross outcome does not
        determine the cession: the same 500 can arrive as one claim of 500,
        ceding 50, or as five claims of 100, ceding 250. The kappa curve
        averages that away by construction and the allocation inherits the
        averaging, which is correct as pricing and silent as description. Two
        percentiles off each row put the spread back.

        **What it is exact about, and what it is not.** The mass weighted mean
        of the kappa column reproduces the other axis's marginal mean to
        floating point, because both are the same sum of ``y_j d[i, j]`` taken
        in a different order. Pointwise the curve carries the rebucketing
        scatter: :func:`scatter_bivariate` splits each per-claim point
        bilinearly over up to four cells, which preserves **both** marginal
        means exactly but smears a conditional one, so ``kappa`` at a single
        grid point is accurate to the scatter rather than to the bit. Where
        the cession lands on the joint lattice (a share cession on a discrete
        severity, say) there is no split and the curve is exact.

        **A netceded joint is the motivating case.** With views
        ``('gross', 'ceded')`` this is the conditional ceded loss given the
        gross outcome, the input to
        :meth:`natural_allocation`. The third view follows by subtraction on
        the index, since ``ceded + net = gross`` holds pointwise.
        """
        self._require_density()
        axis = int(axis)
        if axis not in (0, 1):
            raise ValueError(f'axis must be 0 or 1, not {axis!r}')
        levels = () if levels is None else tuple(float(v) for v in levels)
        for level in levels:
            if not 0.0 <= level <= 1.0:
                raise ValueError(
                    f'a probability level lies in [0, 1]; got {level!r}.')

        grid = self.axis_xs[axis]
        other = self.axis_xs[1 - axis]
        self_name = self.unit_names[axis]
        other_name = self.unit_names[1 - axis]
        other_bs = self.bs[1 - axis]

        n = len(grid)
        lo_row, hi_row = self._cdf_row_window(axis, cdf_range)
        p = np.zeros(n)
        num = np.zeros(n)
        bands = np.full((len(levels), n), np.nan)
        for r0, r1, block in self.bivariate._row_bands(axis=axis):
            p[r0:r1] = block.sum(axis=1)
            num[r0:r1] = block @ other
            if not levels:
                continue
            for i in range(max(r0, lo_row), min(r1, hi_row)):
                mass = p[i]
                if mass <= 0:
                    continue
                # the row's own conditional law, normalized: the quantile of a
                # conditional distribution, not a quantile of the joint
                row = GridDistribution(other, block[i - r0] / mass, bs=other_bs)
                for k, level in enumerate(levels):
                    bands[k, i] = row.q(level)

        gd = GridDistribution(grid, p, bs=self.bs[axis], name=self_name)
        live = p > 0
        kappa = np.full(n, np.nan)
        kappa[live] = num[live] / p[live]
        out = {'p': p, 'F': gd.cdf(grid), 'S': gd.sf(grid),
               f'exeqa_{self_name}': np.where(live, grid, np.nan),
               f'exeqa_{other_name}': kappa}
        for k, level in enumerate(levels):
            out[self._quantile_column(level, other_name)] = bands[k]
        df = pd.DataFrame(out, index=pd.Index(grid, name=self_name))
        return df if cdf_range is None else df.iloc[lo_row:hi_row]

    def _cdf_row_window(self, axis, cdf_range):
        """Half open row range for a probability window on the conditioning axis.

        Read off the conditioning marginal, which costs no joint read on
        either route (the massive container carries its marginals as pass-3
        accumulators), so the expensive per row work in :meth:`exeqa_df` can be
        confined before the sweep rather than measured and then discarded.

        A probability window means the same thing on every grid, which a raw
        mass floor does not: the largest row mass on a fine joint measured in
        the notes is 8.9e-04, so a ``p > 1e-4`` threshold discards most of the
        picture there while keeping nearly all of it on a coarse one.
        """
        n = len(self.axis_xs[axis])
        if cdf_range is None:
            return 0, n
        lo, hi = (float(v) for v in cdf_range)
        if not 0.0 <= lo < hi <= 1.0:
            raise ValueError(
                f'cdf_range is an increasing probability pair inside [0, 1]; '
                f'got {cdf_range!r}.')
        marginal = np.asarray(self.marginals[axis], dtype=float)
        gd = GridDistribution(self.axis_xs[axis], marginal, bs=self.bs[axis])
        xs = np.asarray(self.axis_xs[axis], dtype=float)
        lo_row = int(np.clip(np.searchsorted(xs, gd.q(lo)), 0, n - 1))
        hi_row = int(np.clip(np.searchsorted(xs, gd.q(hi)), 0, n - 1)) + 1
        return lo_row, max(hi_row, lo_row + 1)

    def natural_allocation(self, distortion, P=None):
        r"""Allocate a gross distorted premium to the occurrence ceded and net.

        The decomposition that :func:`aggregate._reinsurance.reins_price_df`
        deliberately does not attempt. That function prices gross, ceded and
        net as three separate distributions, and says so: views are not a
        decomposition, and gross less net is the cedent's allowance rather
        than the reinsurer's price ([Difference-Is-A-Perspective]). This is
        the third question, the one neither row answers. Given a premium for
        the **gross** book, how much of it does each half of an occurrence
        program earn, on one consistent basis, adding up.

        Parameters
        ----------
        distortion : Distortion
            An **already calibrated** distortion. Calibration is the caller's
            choice and ``_pricing``'s business; nothing is calibrated here.
        P : float, optional
            The gross premium to split. Defaults to ``rho_g`` of the joint's
            own gross marginal. See the Notes on why a caller supplied ``P``
            is the usual call.

        Returns
        -------
        pandas.DataFrame
            Rows ``gross`` / ``ceded`` / ``net`` (index named ``view``, the
            :func:`aggregate._reinsurance.reins_price_df` vocabulary), columns
            the pentagon octet
            :data:`~aggregate.pentagon.PENTAGON_STATS`. ``L`` is the component
            mean off the joint, ``P`` the allocated premium, ``M = P - L`` the
            margin and ``LR = L / P``. There is no asset level in an
            unallocated-capital reading, so ``a`` is infinite and ``Q``,
            ``PQ`` and ``ROE`` are blank, exactly as an unlimited quote from
            ``reins_price_df``.

            Three diagnostics ride in ``.attrs``: ``rho_joint``, the risk
            measure read on the joint's own gross marginal; ``rho_fine``, the
            same measure on the source aggregate's fine 1-D gross density;
            and ``rho_gap``, the distance between them.

        Raises
        ------
        ValueError
            If the object is not a ``netceded`` joint, if neither axis is the
            ``gross`` view, or if it has not been updated.

        Notes
        -----
        **Why this needs the joint at all.** An occurrence program splits
        ``G = C + N`` claim by claim, so at the aggregate level ``N`` is *not*
        a comonotone function of ``G``: the random claim count decouples them.
        (Under an *aggregate* program it would be, and the split would be a
        1-D calculation.) The natural allocation of a Choquet premium
        ``rho_g(G) = P`` to the components is

        .. math::

            A_C = E[C\,g'(S_G(G))], \quad A_N = E[N\,g'(S_G(G))],
            \quad A_C + A_N = P,

        and conditioning on ``G`` reduces each to the kappa curve
        :meth:`exeqa_df` puts on the joint.

        **The increment form, which is why no derivative is evaluated.** On
        the lattice the weights are the distorted atom masses
        ``Delta_gS_i = g(S(g_{i-1})) - g(S(g_i))`` off the joint's own gross
        marginal, so

        .. math::

            A_C = \sum_i \kappa_C(g_i)\,\Delta_gS_i.

        That is the Lebesgue-Stieltjes statement directly: the usual
        conditions on ``g'(S)`` do not arise, atoms are handled exactly, and
        ``A_C + A_N`` equals ``rho_g`` of the marginal to floating point by
        construction, because ``kappa_C + kappa_N`` is the identity. The
        weights come from the one Choquet helper
        (:func:`~aggregate.spectral.choquet_weights`), the same convention
        :meth:`~aggregate.spectral.Distortion.price` uses, so the gross row
        **is** that function's answer on the same grid.

        **The calibration grid mismatch, and why the answer is a fraction.**
        A distortion is calibrated on the aggregate's fine 1-D gross density.
        The joint's gross marginal is a coarser rebucketed cousin carrying its
        own deficit, so ``rho_g`` of it does not hit the calibrated ``P`` to
        the bit. Rather than absorb that silently, the method computes
        allocation **fractions** on the joint's grid and applies them to the
        caller's ``P``. The fractions are robust to discretization and
        additivity stays exact whatever ``P`` is. Both readings are then taken
        and reported, ``rho_joint`` on the joint's marginal and ``rho_fine``
        on the fine 1-D density, with ``rho_gap`` their difference: a large
        gap says the joint's grid is too coarse to be pricing on, which is a
        judgment for the caller rather than a silent correction here.

        **The reading is unlimited.** With no asset level, a distortion
        carrying a mass at zero (``ccoc``) puts essentially all its weight on
        the largest outcome the grid happens to represent, so the gross
        premium moves with ``log2`` rather than with the risk. That bites on
        ordinary programs, an aggregate cession being unbounded whenever the
        claim count is. The fractions are far steadier than the level, being
        ratios on one grid, but pass a mass family through
        :meth:`~aggregate.spectral.Distortion.price` at a finite ``a`` and
        hand the result in as ``P`` if the level matters.

        **What the third view costs.** Whichever of ceded or net is on the
        joint's second axis is read from the kappa curve; the remaining view
        is ``g`` less that curve, taken on the index. It is a definition
        rather than a second measurement, which is what makes the rows foot
        exactly. The two builds therefore agree only to the rebucketing
        scatter, not to the bit.

        **On disk as well as in core.** This reads the joint only through
        :meth:`exeqa_df`, which is a row-wise fold over
        :meth:`JointBandsMixin._row_bands`, so a massive (disk backed) joint
        allocates at bounded memory. It refused until 1.0.0a279, inheriting the
        refusal from the kappa curve rather than from anything about the
        allocation.
        """
        from .pentagon import complete_pentagon
        from .spectral import choquet_weights

        # Structural refusals first, so an object of the wrong shape is told
        # that rather than told to call update(); the same ordering
        # ``reins_price_df`` uses for its no-cession case.
        if self.mode != 'netceded':
            raise ValueError(
                f'natural_allocation needs a netceded joint (one aggregate '
                f'split into two of gross / ceded / net); {self.name} is in '
                f'{self.mode!r} mode. Allocating a dependent two unit total '
                f'conditions on the total rather than on an axis, which is '
                f'the deferred [Bivariate-Total-Exeqa] work.')
        if 'gross' not in self._views:
            raise ValueError(
                f'natural_allocation conditions on the gross outcome, and '
                f'{self.name} carries views {self._views}. Rebuild with '
                f"views=('gross', 'ceded') or ('gross', 'net'); a "
                f'(net, ceded) joint has no gross axis to condition on.')
        self._require_density()

        axis = self._views.index('gross')
        other_view = self._views[1 - axis]
        df = self.exeqa_df(axis=axis)
        g = df.index.to_numpy(dtype=float)
        p = df['p'].to_numpy()
        # NaN off the support would poison the dot products; those rows carry
        # zero distorted weight anyway (p_k = 0 makes T_k = S_k, so gp_k = 0),
        # so zero is the arithmetically inert fill rather than a claim about
        # a conditional expectation that has no value.
        kappa = np.nan_to_num(
            df[f'exeqa_{self.unit_names[1 - axis]}'].to_numpy(), nan=0.0)

        # gross, ceded and net are all losses (a netceded axis carries no
        # affine), so the value-type role is fixed and ask-of-loss is g itself.
        gfn, _, _ = distortion.effective_g('ask', is_loss_value=True)
        gp = choquet_weights(g, p, gfn, allow_deficit=True).gp

        rho_joint = float(g @ gp)
        if rho_joint <= 0:
            raise ValueError(
                f'{self.name} prices at {rho_joint:g} on the gross basis, so '
                f'there are no shares to compute. A zero risk aggregate has '
                f'nothing to allocate.')
        rho_fine = float(distortion.price(
            self._nc_agg._reins_view_density('gross'), kind='ask').ask)
        premium = rho_joint if P is None else float(P)
        # fractions on the joint's grid, applied to the caller's premium
        share_other = float(kappa @ gp) / rho_joint
        loss_total = float(g @ p)
        loss_other = float(kappa @ p)

        amounts = {
            'gross': (loss_total, premium),
            other_view: (loss_other, share_other * premium),
        }
        third = 'net' if other_view == 'ceded' else 'ceded'
        amounts[third] = (loss_total - loss_other,
                          (1.0 - share_other) * premium)

        rows = [amounts[v] for v in ('gross', 'ceded', 'net')]
        out = complete_pentagon(pd.DataFrame(
            [[el, ask - el, ask, np.nan] for el, ask in rows],
            columns=['L', 'M', 'P', 'Q'],
            index=pd.Index(['gross', 'ceded', 'net'], name='view')))
        out['a'] = np.inf
        out.attrs['rho_joint'] = rho_joint
        out.attrs['rho_fine'] = rho_fine
        out.attrs['rho_gap'] = rho_joint - rho_fine
        return out

    @staticmethod
    def _mcs_to_raw(m, sd, skew):
        """Raw moments ``(ex1, ex2, ex3)`` from ``(mean, sd, skew)``.

        The inverse of ``static_moments_to_mcvsk``: ``ex2 = sd^2 + m^2`` and
        ``ex3 = mu3 + 3 m ex2 - 2 m^3`` with ``mu3 = skew sd^3``. ``nan``
        inputs propagate, so a partial theory fills a partial column.
        """
        ex2 = sd * sd + m * m
        ex3 = skew * sd ** 3 + 3.0 * m * ex2 - 2.0 * m ** 3
        return float(m), float(ex2), float(ex3)

    @property
    def stats_df(self):
        """Canonical moment store, parallel to :attr:`Portfolio.stats_df`.

        Restructured by ``[Bivariate-Punchup]`` (``1.0.0a330``): until then
        this frame was theoretical against empirical per marginal, a
        comparison that now lives in :attr:`validation_df`.

        **Rows** are Portfolio's ``(component, measure)`` MultiIndex: a
        ``meta`` block (limit, attachment, el, prem, lr, ...), then ``freq`` /
        ``sev`` / ``agg`` blocks each carrying the raw moments ``ex1``..``ex3``
        and ``mean`` / ``cv`` / ``skew``. Cells the mode does not carry hold
        ``NaN``, exactly as Portfolio leaves non-applicable cells.

        **Columns**: one per marginal (that component's theoretical moments,
        the analog of Portfolio copying each unit's ``stats_df['mixed']``),
        then ``independent``, then ``total``:

        * ``independent``: the moments ``X + Y`` would have were the axes
          independent, from the marginal theory (means, variances and third
          central moments add). Purely analytic, and the natural benchmark:
          ``total`` against ``independent`` reads off the dependence lift,
          which is the point of building a joint.
        * ``total``: the realized dependent total, from the joint mixed
          moments (:meth:`_total_agg_raw_moments`; no ``X + Y`` grid is
          formed). In netceded mode this column has an exact analytic
          reference, the gross aggregate.

        Notes
        -----
        The ``freq`` block repeats the **shared** outer frequency in every
        column: by construction each marginal is that count compounded with
        its own per-event severity, so the count is one fact, not four. The
        ``sev`` blocks are per-event: each marginal's own severity, the
        independent sum (cross moments factorize), and the dependent total
        off the joint per-claim matrix (:meth:`_total_sev_raw_moms`). The
        ``agg`` marginal columns carry the displayed theory
        (:meth:`_axis_theory`, any ``pnl`` affine applied), converted to raw
        moments by :meth:`_mcs_to_raw` so one column reports one variable on
        one basis. Portfolio's ``empirical`` / ``error`` columns are omitted
        deliberately: they would duplicate :attr:`validation_df`.
        """
        from ._portfolio import _PORT_STATS_ROW_INDEX
        self._require_density()
        cols = list(self.unit_names) + ['independent', 'total']
        df = pd.DataFrame(np.nan, index=_PORT_STATS_ROW_INDEX,
                          columns=cols, dtype=float)

        def put(col, comp, raw):
            m1, m2, m3 = raw
            m, cv, sk = MomentAggregator.static_moments_to_mcvsk(m1, m2, m3)
            df.loc[(comp, 'ex1'), col] = m1
            df.loc[(comp, 'ex2'), col] = m2
            df.loc[(comp, 'ex3'), col] = m3
            df.loc[(comp, 'mean'), col] = m
            df.loc[(comp, 'cv'), col] = cv
            df.loc[(comp, 'skew'), col] = sk

        # the shared outer count is one fact, repeated per column
        fmoms = self._shared_freq_moms()
        for col in cols:
            put(col, 'freq', fmoms)

        # per-event severity: each marginal, the independent sum, the total
        svs = [self._sev_raw_moms(i) for i in range(2)]
        for i, name in enumerate(self.unit_names):
            if svs[i] is not None:
                put(name, 'sev', svs[i])
        if svs[0] is not None and svs[1] is not None:
            (a1, a2, a3), (b1, b2, b3) = svs
            put('independent', 'sev', (a1 + b1,
                                       a2 + 2.0 * a1 * b1 + b2,
                                       a3 + 3.0 * a2 * b1
                                       + 3.0 * a1 * b2 + b3))
        tsev = self._total_sev_raw_moms()
        if tsev is not None:
            put('total', 'sev', tsev)

        # aggregate: displayed marginal theory, the independent-sum
        # benchmark (central moments add), and the realized dependent total
        theories = [self._axis_theory(i) for i in range(2)]
        for i, name in enumerate(self.unit_names):
            put(name, 'agg', self._mcs_to_raw(*theories[i]))
        (m0, sd0, sk0), (m1_, sd1, sk1) = theories
        var = sd0 * sd0 + sd1 * sd1
        sd = float(np.sqrt(var)) if var > 0 else np.nan
        mu3 = sk0 * sd0 ** 3 + sk1 * sd1 ** 3
        put('independent', 'agg', self._mcs_to_raw(
            m0 + m1_, sd, mu3 / sd ** 3 if sd and np.isfinite(sd) else np.nan))
        put('total', 'agg', self._total_agg_raw_moments())

        # meta: the terms the mode carries; NaN otherwise
        if self.mode == 'netceded':
            gross = self._nc_agg.stats_df['mixed']
            for key in ('limit', 'attachment', 'el', 'prem', 'lr'):
                if ('meta', key) in gross.index:
                    df.loc[('meta', key), 'total'] = float(gross[('meta',
                                                                  key)])
        else:
            df.loc[('meta', 'el'), 'total'] = self._exp_el
            df.loc[('meta', 'prem'), 'total'] = self._exp_prem
            df.loc[('meta', 'lr'), 'total'] = self._exp_lr
        return df

    def _axis_theory(self, i):
        """Displayed theoretical ``(mean, sd, skew)`` for axis ``i``.

        The cached marginal theory with any ``pnl`` affine applied (copula mode;
        ``netceded`` has no affine so the ceded / net moments pass through)."""
        mt, sdt, skt = self._marg_theory[i]
        reflect, shift = self._affine[i]
        if reflect:
            mt, skt = shift - mt, -skt
        elif shift:
            mt = shift + mt
        return mt, sdt, skt

    @property
    def summary_df(self):
        """At-a-glance risk view: moments and key percentiles, marginals and total.

        The daily-driver headline, matching :attr:`Aggregate.summary_df` and
        :attr:`Portfolio.summary_df` in role and columns. Until ``1.0.0a329``
        this name published the moment audit, which now lives at
        :attr:`validation_df` (the two names were crossed relative to the rest
        of the library; ``[Bivariate-Punchup]`` swapped them). One row per
        marginal, by resolved label, then a ``total`` row: the realized
        **dependent** ``X + Y``, which is the number the joint was built for.

        **Columns** ``Mean | SD | CV | Skew | P01 | Median | P99``. ``SD`` and
        ``CV`` are both always present (stable layout); ``CV`` blanks per row
        when ``|Mean|`` is ~0 relative to ``SD`` (a signed, near break even
        ``pnl`` axis; see :meth:`Aggregate._cv_or_nan`), and ``SD`` never
        blanks. Marginal rows carry the realized grid moments and exact grid
        percentiles read off :meth:`marginal`; the ``total`` row's moments
        come exact from the joint mixed moments
        (:meth:`_total_agg_empirical`, no grid formed) and its percentiles
        from the realized :attr:`total` distribution.

        Dependence (cov / corr / tau) is in :attr:`dependency_df`; the theory
        against realized audit is :attr:`validation_df`.

        Returns
        -------
        DataFrame
            Rows: the two marginals plus ``total`` (index named ``unit``);
            columns as above; the total mean rides in ``.attrs['mean']``.
        """
        from ._aggregate import (Aggregate, SUMMARY_PERCENTILES,
                                 _summary_pct_label)
        self._require_density()
        pcols = [_summary_pct_label(p) for p in SUMMARY_PERCENTILES]

        def _mcs(x, p):
            # realized central moments off one marginal's own grid; the mass
            # is normalized so a tail deficit does not read as a moment shift
            tot = p.sum()
            m1 = float(x @ p) / tot
            var = float(((x - m1) ** 2) @ p) / tot
            sd = float(np.sqrt(var)) if var > 0 else 0.0
            mu3 = float(((x - m1) ** 3) @ p) / tot
            return m1, sd, (mu3 / sd ** 3 if sd > 0 else np.nan)

        rows = {}
        for i in range(2):
            gd = self.marginal(i)
            m1, sd, skew = _mcs(gd.x, gd.p)
            rows[self.unit_names[i]] = [
                m1, sd, Aggregate._cv_or_nan(m1, sd), skew,
                *(gd.q(q) for q in SUMMARY_PERCENTILES)]
        tm, tsd, tsk = self._total_agg_empirical()
        t = self.total
        rows['total'] = [tm, tsd, Aggregate._cv_or_nan(tm, tsd), tsk,
                         *(t.q(q) for q in SUMMARY_PERCENTILES)]
        df = pd.DataFrame.from_dict(
            rows, orient='index', columns=['Mean', 'SD', 'CV', 'Skew',
                                           *pcols])
        df.index.name = 'unit'
        for c in ('Mean', 'SD', 'Skew', *pcols):
            df[c] = _snap_noise(df[c])
        df = self._relabel(df)
        df.attrs['mean'] = tm
        return df

    @property
    def validation_df(self):
        """Moment audit: reference against realized, Freq / Sev / Agg blocks.

        The bivariate answer to :attr:`Aggregate.validation_df` and
        :attr:`Portfolio.validation_df`. Until ``1.0.0a329`` this frame was
        published as ``summary_df`` (see there for the swap). A shared
        ``Freq`` block on top, a ``Sev`` / ``Agg`` block per component, then a
        ``total`` ``Sev`` / ``Agg`` block (the genuine ``X + Y`` aggregate).
        The eight columns are ``EX | Est EX | Err EX | <spread> | Est
        <spread> | Err <spread> | Sk | Est Sk``, with ``Est`` the realized
        (model output) value and ``Err`` the noise-aware relative error.

        ``Est`` is populated only where it is observable from the joint
        density: the per-component and ``total`` **Agg** rows. ``Freq`` and
        ``Sev`` rows are theory only (the 2-D convolution yields the joint
        aggregate, never an independent frequency or severity sample), and
        the shared ``Freq`` is reported once. The ``total Agg`` theory
        carries only the additive mean ``E[X] + E[Y]``; its spread and skew
        are emergent from the modeled dependence, so they appear on the
        ``Est`` side only. The spread column is **CV**, or **SD** when the
        book is signed (a ``pnl`` axis, a component with signed severity, or
        a signed netceded source; see :meth:`_signed`); the choice is frame
        wide, as in ``Portfolio``.

        The pass/fail gates are not here: they live behind the private
        ``_gate_checks`` and reach the reader through :attr:`info`,
        :attr:`validation_description` and :attr:`validation_explanation`,
        the way ``Aggregate`` manages its ``Validation`` flags.

        Returns
        -------
        DataFrame
            ``MultiIndex (component, part)`` rows; eight validation columns.
        """
        self._require_density()
        use_sd = self._signed()
        spread = 'SD' if use_sd else 'CV'

        def _spread(m, sd):
            return sd if use_sd else (sd / m if m else np.nan)

        def _row(mt, sdt, skt, me, sde, ske):
            spt, spe = _spread(mt, sdt), _spread(me, sde)
            return {
                'EX': mt, 'Est EX': me, 'Err EX': _noise_aware_rel_error(me, mt),
                spread: spt, f'Est {spread}': spe,
                f'Err {spread}': _noise_aware_rel_error(spe, spt),
                'Sk': skt, 'Est Sk': ske}

        nan3 = (np.nan, np.nan, np.nan)
        rows, index = [], []

        # shared frequency block (theory only -- count is an input, not sampled)
        fm, fcv, fsk = MomentAggregator.static_moments_to_mcvsk(
            *self._shared_freq_moms())
        rows.append(_row(fm, fcv * fm if np.isfinite(fcv) else np.nan, fsk, *nan3))
        index.append(('shared', 'Freq'))

        # per-component Sev (theory) + Agg (theory vs realised marginal)
        m0, m1 = self.marginals
        for i, (name, dens) in enumerate(zip(self.unit_names, (m0, m1))):
            sm, scv, ssk = self._sev_mcvsk(i)
            rows.append(_row(sm, scv * sm if np.isfinite(scv) else np.nan,
                             ssk, *nan3))
            index.append((name, 'Sev'))
            mt, sdt, skt = self._axis_theory(i)
            me, cve, ske = xsden_to_meancvskew(self.axis_xs[i], dens)
            sde = cve * me if np.isfinite(cve) else np.nan
            rows.append(_row(mt, sdt, skt, me, sde, ske))
            index.append((name, 'Agg'))

        # total Sev (theory) + total Agg (additive-mean theory vs realised)
        tsm, tscv, tssk = self._total_sev_mcvsk()
        rows.append(_row(tsm, tscv * tsm if np.isfinite(tscv) else np.nan,
                         tssk, *nan3))
        index.append(('total', 'Sev'))
        em, esd, esk = self._total_agg_empirical()
        tmean = sum(self._axis_theory(i)[0] for i in range(2))
        rows.append(_row(tmean, np.nan, np.nan, em, esd, esk))
        index.append(('total', 'Agg'))

        df = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(
            index, names=['component', 'part']))
        df = df[['EX', 'Est EX', 'Err EX', spread, f'Est {spread}',
                 f'Err {spread}', 'Sk', 'Est Sk']]
        # snap display dust in the value columns (Err columns keep their dust)
        for c in ('EX', 'Est EX', spread, f'Est {spread}', 'Sk', 'Est Sk'):
            df[c] = _snap_noise(df[c])
        return self._relabel(df)

    @property
    def dependency_df(self):
        """Joint dependence structure: ``cov`` / ``corr`` / ``tau`` by level.

        Two rows -- ``Sev`` (the per-claim joint severity) and ``Agg`` (the
        realised joint aggregate) -- and three columns:

        * ``cov`` -- covariance of the two components at that level;
        * ``corr`` -- linear (Pearson) correlation;
        * ``tau`` -- the input copula's Kendall tau (a per-claim property, so it
          sits on the ``Sev`` row; the realised aggregate ``tau`` is not computed
          -- it is an expensive concordance sum on the joint PMF).

        The ``Sev`` row is read from the modeled per-claim severity matrix
        ``self._S``; the ``Agg`` row from the realised joint mixed moments
        (:meth:`moments`). Higher mixed comoments are not tabulated -- they are a
        short user-side calc off :meth:`moments` (aggregate) or ``_S`` (severity).

        Returns
        -------
        DataFrame
            Index ``level`` in ``['Sev', 'Agg']``; columns ``cov`` / ``corr`` /
            ``tau``.
        """
        self._require_density()
        rows = {}
        # Agg level: realised joint aggregate
        mom = self.moments(2).to_numpy()
        tot = mom[0, 0]
        eX, eY = mom[1, 0] / tot, mom[0, 1] / tot
        vX, vY = mom[2, 0] / tot - eX ** 2, mom[0, 2] / tot - eY ** 2
        covA = mom[1, 1] / tot - eX * eY
        corrA = covA / np.sqrt(vX * vY) if vX > 0 and vY > 0 else np.nan
        rows['Agg'] = {'cov': covA, 'corr': corrA, 'tau': np.nan}
        # Sev level: per-claim joint severity (the FFT2 input matrix)
        if self._S is not None and self._sev_xs[0] is not None:
            S = self._S
            x0 = np.asarray(self._sev_xs[0], dtype=float)
            x1 = np.asarray(self._sev_xs[1], dtype=float)
            s0, s1 = _dense_1d(S.sum(axis=1)), _dense_1d(S.sum(axis=0))
            eA, eB = x0 @ s0, x1 @ s1
            vA, vB = (x0 ** 2) @ s0 - eA ** 2, (x1 ** 2) @ s1 - eB ** 2
            covS = x0 @ _dense_1d(S @ x1) - eA * eB
            corrS = covS / np.sqrt(vA * vB) if vA > 0 and vB > 0 else np.nan
            tau = self.copula.tau() if self.copula is not None else np.nan
            rows['Sev'] = {'cov': covS, 'corr': corrS, 'tau': tau}
        else:
            rows['Sev'] = {'cov': np.nan, 'corr': np.nan, 'tau': np.nan}
        df = pd.DataFrame(rows).T.reindex(['Sev', 'Agg'])[['cov', 'corr', 'tau']]
        df.index.name = 'level'
        return df

    def _signed(self):
        """Whether the audit frame spreads by SD rather than CV.

        True when any axis can sit near or below a zero mean, which makes CV
        meaningless: a ``pnl`` affine on either axis, a component aggregate
        that is itself signed (``ssev``, or a ``dsev`` with a negative atom;
        see :meth:`Aggregate._signed`), or, in netceded mode, a signed source
        aggregate (an occurrence view pair built on a signed gross book has
        signed views). Mirrors :meth:`Portfolio._signed`, which asks each
        component. Frame wide by design, as in ``Portfolio``; the per-row
        blanking lives in :attr:`summary_df` via ``_cv_or_nan``.
        """
        if self.mode == 'netceded':
            return self._nc_agg._signed()
        return (any(r or s for r, s in self._affine)
                or any(a._signed() for a in self.units))

    def _shared_freq_moms(self):
        """Raw moments ``(f1, f2, f3)`` of the shared outer frequency."""
        if self.mode == 'netceded':
            return self._nc_agg.frequency.freq_moms(float(self._nc_agg.n))
        return self.frequency.freq_moms(self.en)

    def _sev_raw_moms(self, i):
        """Per-event severity raw moments ``(s1, s2, s3)`` for component ``i``.

        The analytic per-event severity raw moments (copula mode); falls back
        to the per-event severity marginal of the joint severity matrix
        ``self._S`` (e.g. ``netceded``, where the net / ceded per-claim
        severity is read off the comonotone scatter). ``None`` when neither
        source is available.
        """
        sm = self._sev_moms[i] if self._sev_moms else None
        if sm is not None:
            return tuple(float(v) for v in sm)
        if self._S is not None and self._sev_xs[i] is not None:
            x = np.asarray(self._sev_xs[i], dtype=float)
            g = _dense_1d(self._S.sum(axis=1 - i))
            return (float(x @ g), float((x ** 2) @ g), float((x ** 3) @ g))
        return None

    def _sev_mcvsk(self, i):
        """Per-event severity ``(mean, cv, skew)`` for component ``i``.

        The mcvsk view of :meth:`_sev_raw_moms`; ``nan`` triple when the raw
        moments are unavailable.
        """
        sm = self._sev_raw_moms(i)
        if sm is None:
            return np.nan, np.nan, np.nan
        return MomentAggregator.static_moments_to_mcvsk(*sm)

    def _total_sev_raw_moms(self):
        """Raw moments ``(m1, m2, m3)`` of the total per-event severity ``S0 + S1``.

        From the modeled joint per-claim severity ``self._S`` (exact,
        dependence included). The cross moments ``E[S0^p S1^q]`` are formed as
        ``x0**p @ S @ x1**q`` so no dense ``n x n`` grid of pairwise sums is
        materialised. ``None`` when the severity matrix is unavailable.
        """
        S = self._S
        if S is None or self._sev_xs[0] is None:
            return None
        x0 = np.asarray(self._sev_xs[0], dtype=float)
        x1 = np.asarray(self._sev_xs[1], dtype=float)
        s0, s1 = _dense_1d(S.sum(axis=1)), _dense_1d(S.sum(axis=0))
        eA, eA2, eA3 = x0 @ s0, (x0 ** 2) @ s0, (x0 ** 3) @ s0
        eB, eB2, eB3 = x1 @ s1, (x1 ** 2) @ s1, (x1 ** 3) @ s1
        sx1, sx1_2 = _dense_1d(S @ x1), _dense_1d(S @ (x1 ** 2))
        eAB, eA2B, eAB2 = x0 @ sx1, (x0 ** 2) @ sx1, x0 @ sx1_2
        m1 = eA + eB
        m2 = eA2 + 2 * eAB + eB2
        m3 = eA3 + 3 * eA2B + 3 * eAB2 + eB3
        return float(m1), float(m2), float(m3)

    def _total_sev_mcvsk(self):
        """``(mean, cv, skew)`` of the total per-event severity ``S0 + S1``.

        The mcvsk view of :meth:`_total_sev_raw_moms`; ``nan`` triple when the
        severity matrix is unavailable.
        """
        m = self._total_sev_raw_moms()
        if m is None:
            return np.nan, np.nan, np.nan
        return MomentAggregator.static_moments_to_mcvsk(*m)

    def _total_agg_raw_moments(self):
        """Realized raw moments ``(m1, m2, m3)`` of the total ``X + Y`` aggregate.

        From the joint mixed moments ``E[X^i Y^j]`` (:meth:`moments`), so the
        ``X + Y`` grid is never formed: the raw moments of the sum follow from
        the marginal and cross raw moments by the binomial expansion.
        """
        mom = self.moments(3).to_numpy()
        tot = mom[0, 0]
        eX, eY = mom[1, 0] / tot, mom[0, 1] / tot
        eX2, eY2, eXY = mom[2, 0] / tot, mom[0, 2] / tot, mom[1, 1] / tot
        eX3, eY3 = mom[3, 0] / tot, mom[0, 3] / tot
        eX2Y, eXY2 = mom[2, 1] / tot, mom[1, 2] / tot
        m1 = eX + eY
        m2 = eX2 + 2 * eXY + eY2
        m3 = eX3 + 3 * eX2Y + 3 * eXY2 + eY3
        return float(m1), float(m2), float(m3)

    def _total_agg_empirical(self):
        """Realised ``(mean, sd, skew)`` of the total ``X + Y`` aggregate.

        The central-moment view of :meth:`_total_agg_raw_moments`.
        """
        m1, m2, m3 = self._total_agg_raw_moments()
        var = m2 - m1 * m1
        sd = float(np.sqrt(var)) if var > 0 else np.nan
        if sd and np.isfinite(sd) and sd > 0:
            mu3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
            skew = float(mu3 / sd ** 3)
        else:
            skew = np.nan
        return float(m1), sd, skew

    def _axis_kind(self, i):
        """Per-axis kind label (``agg`` / ``pnl`` / ``gross`` / ``ceded`` / ``net``)."""
        if self.mode == 'netceded':
            return self._views[i]
        reflect, shift = self._affine[i]
        return 'pnl' if (reflect or shift) else 'agg'

    def _bs_str(self, i):
        """``bs`` of axis ``i`` in the Agg/Port display form (``1/n`` for < 1)."""
        bs = self.bs[i]
        return f'{bs:.6g}' if bs >= 1 else f'1/{int(round(1.0 / bs))}'

    def _id(self):
        """Display-only 8-hex hash of the structural fields (mirrors Agg ``id``)."""
        import hashlib
        en = float(self._nc_agg.n) if self.mode == 'netceded' else float(self.en)
        key = repr((self.name, self.mode, tuple(self.unit_names),
                    str(self.copula), getattr(self, 'freq_name', ''), en))
        return hashlib.md5(key.encode()).hexdigest()[:8]

    @property
    def info(self):
        """Fixed-layout multi-unit summary string.

        Every row is always present, in the same order, for every
        ``BivariateAggregate``; a value that is not (yet) available -- e.g.
        the grid block before :meth:`update` -- renders as ``n/a``. Shares the
        label/value convention (:func:`aggregate.constants.info_row`) with
        ``Aggregate`` / ``Portfolio``; the bivariate row catalogue is documented
        in ``dev/info-strings.rst``.
        """
        updated = self.density is not None or self._massive is not None
        if self.mode == 'netceded':
            copula = 'comonotone (netceded)'
            freq_name = self._nc_agg.frequency.freq_name
            en = float(self._nc_agg.n)
            tau = INFO_NA
        elif self.mode == 'discrete':
            copula = 'discrete (dbvsev)'
            freq_name = self.freq_name
            en = float(self.en)
            tau = INFO_NA
        else:
            copula = repr(self.copula)
            freq_name = self.freq_name
            en = float(self.en)
            tau = INFO_NA if self.copula is None else f'{self.copula.tau():.4f}'
        rows = [
            ('bivariate object name', self.name),
            ('mode', self.mode),
            ('components', f'{self.unit_names[0]} x {self.unit_names[1]}'),
            ('copula', copula),
            ('shared frequency', freq_name),
            ('claim count', f'{en:,.3f}'),
            ('padding', self.padding if updated else INFO_NA),
        ]
        for i in range(2):
            lbl = f'axis {i}'
            rows.append((f'{lbl} name', self.unit_names[i]))
            rows.append((f'{lbl} kind', self._axis_kind(i)))
            if updated:
                xs = self.axis_xs[i]
                rows += [
                    (f'{lbl} bs', self._bs_str(i)),
                    (f'{lbl} log2', int(round(np.log2(len(xs))))),
                    (f'{lbl} x_min', f'{float(xs[0]):,.6g}'),
                    (f'{lbl} x_max', f'{float(xs[-1]):,.6g}'),
                ]
            else:
                rows += [(f'{lbl} bs', INFO_NA), (f'{lbl} log2', INFO_NA),
                         (f'{lbl} x_min', INFO_NA), (f'{lbl} x_max', INFO_NA)]
        rows += [
            ('correlation', f'{self.corr:.6f}' if updated else INFO_NA),
            ('copula tau', tau),
            ('tail deficit', f'{self.deficit:.2e}' if updated else INFO_NA),
            ('validation', self._explain_oneline() if updated else INFO_NA),
            ('id', self._id()),
        ]
        return '\n'.join(info_row(label, value) for label, value in rows)

    #: Loose gate on the per-axis marginal mean (relative error vs the analytic
    #: standalone). Deliberately loose: it catches gross misplacement, not the
    #: budget-dependent bs-discretization error, which is expected.
    MARGINAL_MEAN_GATE = 0.10
    #: Hard gate on the joint tail deficit. A measured grid conserves mass, so
    #: any deficit means a clipped or aliased window.
    TAIL_DEFICIT_GATE = 1e-5

    def _gate_checks(self):
        """The joint grid's private gate table: is this bivariate calculating correctly?

        The **one** computation behind the ``validation`` row of :attr:`info`,
        :attr:`validation_description` and :attr:`validation_explanation`
        (a172 [FCC-Contract-Gaps]; the three used to compute it three times).
        Public through those narrations only, the way ``Aggregate`` keeps its
        ``Validation`` flags behind ``explain_validation()``: the public
        :attr:`validation_df` is the moment audit, matching the other first
        class classes, and the exhibit layer derives its row emphasis from
        this table (``exhibits/_bivariate.py``).

        One row per check, in escalating order of severity:

        * **marginal mean**, one row per axis. Each marginal must reproduce its
          standalone aggregate, which is the showpiece invariant of the 2-D FFT.
          Measured against the analytic standalone mean at
          :attr:`MARGINAL_MEAN_GATE`.
        * **tail deficit**, one row. The hard correctness gate at
          :attr:`TAIL_DEFICIT_GATE`.

        Returns
        -------
        DataFrame
            Rows indexed by ``check``; columns ``Est`` (realized), ``Ref``
            (the target it is held to), ``Err`` (signed relative error, or the
            level itself for the deficit), ``Gate`` and ``Pass``.

        Notes
        -----
        Distinct from :attr:`validation_df`, which is the full theory against
        realized moment block over Freq / Sev / Agg. This table carries only
        what can *fail*, so a reader who wants the verdict does not have to
        know which of twenty numbers is load-bearing. The unit differs by row
        (currency for a mean, probability for the deficit) because a check
        table is a list of checks, not one measurement repeated.
        """
        self._require_density()
        rows, index = [], []
        m0, m1 = self.marginals
        for i, (name, dens) in enumerate(zip(self.unit_names, (m0, m1))):
            mt = self._axis_theory(i)[0]
            me = xsden_to_meancvskew(self.axis_xs[i], dens)[0]
            err = (me - mt) / abs(mt) if mt else np.nan
            rows.append({'Est': me, 'Ref': mt, 'Err': err,
                         'Gate': self.MARGINAL_MEAN_GATE,
                         'Pass': not (np.isfinite(err)
                                      and abs(err) > self.MARGINAL_MEAN_GATE)})
            index.append(f'marginal mean {name}')
        # The deficit IS its own error: there is no reference level to hit, the
        # target is zero, so Est and Err carry the same number and Ref is 0.
        rows.append({'Est': self.deficit, 'Ref': 0.0, 'Err': self.deficit,
                     'Gate': self.TAIL_DEFICIT_GATE,
                     'Pass': not self.deficit > self.TAIL_DEFICIT_GATE})
        index.append('tail deficit')
        df = pd.DataFrame(rows, index=pd.Index(index, name='check'))
        return df[['Est', 'Ref', 'Err', 'Gate', 'Pass']]

    def _explain_oneline(self):
        """One-unit validation summary for the ``info`` ``validation`` row.

        Reads the verdicts off :meth:`_gate_checks` rather than recomputing
        them, so the one-liner, the narrative and the table cannot disagree.
        """
        df = self._gate_checks()
        bad = []
        if not df.loc[df.index.str.startswith('marginal mean'), 'Pass'].all():
            bad.append('marginal mean')
        if not df.loc['tail deficit', 'Pass']:
            bad.append('tail deficit')
        return 'not unreasonable' if not bad else 'check: ' + ', '.join(bad)

    # ``program`` / ``format_program`` / ``pprogram`` / ``pprogram_html`` come
    # from ``ProgramMixin``; the two components render nested a level deeper.

    @property
    def bs_window_df(self):
        """Per-axis grid summary: the realized ``(kind, bs, log2, window)`` per axis.

        The bivariate analogue of :attr:`Aggregate.bs_window_df`. The bv
        *measures* its grid (it does not run the 1-D method ladder), so this is a
        two-row summary -- one per axis -- of the chosen grid, not the per-method
        decision journey. ``clipped`` flags a budget-forced window clip.
        """
        self._require_density()
        rows = {}
        for i, name in enumerate(self.unit_names):
            xs = self.axis_xs[i]
            rows[name] = {
                'kind': self._axis_kind(i), 'bs': self.bs[i],
                'log2': int(round(np.log2(len(xs)))),
                'x_min': float(xs[0]), 'x_max': float(xs[-1]),
                'clipped': bool(getattr(self, '_clipped', False)),
            }
        # ``from_dict(orient='index')``, not ``pd.DataFrame(rows).T``: the rows
        # mix str, float, int and bool, and the transposed form types the whole
        # block ``object``, which costs the served table its alignment and its
        # raw values. Same fix as the aggregate sizer.
        df = pd.DataFrame.from_dict(rows, orient='index')
        df['log2'] = df['log2'].astype('Int64')
        df.index.name = 'axis'
        return self._relabel(df)

    @property
    def bs_description(self) -> str:
        """One-unit summary of the chosen per-axis grids.

        The verbose form is :attr:`bs_explanation`.
        """
        parts = [f'{name} bs={self._bs_str(i)} log2={int(round(np.log2(len(self.axis_xs[i]))))}'
                 for i, name in enumerate(self.unit_names)]
        return 'bivariate grid: ' + '; '.join(parts) + f' (deficit {self.deficit:.2e})'

    @property
    def bs_explanation(self) -> str:
        """Verbose prose explaining the per-axis bivariate grid.

        The bivariate twin of
        :attr:`~aggregate.distributions.Aggregate.bs_explanation`. A bv
        *measures* its grid rather than running the 1-D method ladder, so this
        reports the realised per-axis ``(bs, log2, window)``, the joint memory
        footprint the pair implies, the tail deficit (the hard correctness
        gate: a measured grid conserves mass, so a deficit means a clipped or
        aliased window), and what to change if the deficit is material.
        """
        self._require_density()
        out = [f'Bivariate {self.mode} grid over '
               f'{self.unit_names[0]} x {self.unit_names[1]}.']
        cells = 1
        for i, name in enumerate(self.unit_names):
            xs = self.axis_xs[i]
            log2 = int(round(np.log2(len(xs))))
            cells *= len(xs)
            out.append(f'Axis {i} ({name}, {self._axis_kind(i)}): bs = '
                       f'{self._bs_str(i)}, log2 = {log2}, window '
                       f'[{float(xs[0]):,.6g}, {float(xs[-1]):,.6g}].')
        out.append(f'The joint grid is {cells:,d} cells.')
        sizing = getattr(self, '_nc_sizing', None)
        if sizing is not None:
            # Whether the comonotone curve lands on the lattice is the one
            # thing a netceded reader cannot see from the numbers above, and it
            # is what separates an exact kappa curve from an accurate one.
            if sizing.exact:
                out.append(
                    'Every per-claim view image is a grid point at this '
                    'bucket size, so the comonotone scatter never splits and '
                    'the conditional (kappa) curve is exact.')
            elif sizing.bs_exact is None:
                out.append(
                    'No common lattice for the view images fits the budget, '
                    'so the scatter splits each per-claim point and the '
                    'conditional (kappa) curve is accurate to that smear '
                    'rather than exact. Both marginal means are preserved '
                    'either way. Raise total_log2 to buy exactness, or accept '
                    'that a share cession usually has no usable lattice at '
                    'any size.')
            else:
                out.append(
                    f'The budget affords the exact lattice, bs = '
                    f'{sizing.bs_exact:g}, and the pinned bs = {sizing.bs:g} '
                    f'is not it, so the scatter splits each per-claim point '
                    f'and the conditional (kappa) curve is accurate to that '
                    f'smear rather than exact.')
        if getattr(self, '_clipped', False):
            out.append('The window was clipped to stay inside the memory '
                       'budget -- pass a larger budget or an explicit per-axis '
                       'bs / log2 to widen it.')
        deficit = self.deficit
        if deficit > 1e-5:
            out.append(f'Tail deficit {deficit:.2e} exceeds the 1e-5 gate: mass '
                       f'is falling off the grid, so raise log2 (or widen the '
                       f'window) before trusting the tail.')
        else:
            out.append(f'Tail deficit {deficit:.2e} is within the 1e-5 gate, so '
                       f'the window holds essentially all the mass.')
        return ' '.join(out)

    @property
    def axis_support_df(self):
        """Per-axis support summary of the realized marginals.

        One row per axis with the realized ``support_min`` / ``support_max``
        (where the marginal has mass), the theoretical ``mean`` / ``sd`` /
        ``skew``, and a ``right_heavy`` flag (``skew > 1``). The full 1-D
        tail-class ladder lives on each axis's standalone marginal aggregate.

        Notes
        -----
        Named ``tail_df`` until ``a171``, which was a collision rather than an
        analogy: :attr:`Aggregate.tail_df` and :attr:`Portfolio.tail_df` are
        **return-period** tables (p, VaR, TVaR, xsVaR by exceedance probability),
        and this frame is not one. It reports where the realized mass sits, per
        axis, which is what the name now says.
        """
        self._require_density()
        m0, m1 = self.marginals
        rows = {}
        for i, (name, m) in enumerate(zip(self.unit_names, (m0, m1))):
            xs = self.axis_xs[i]
            nz = np.flatnonzero(m > 1e-15)
            mt, sdt, skt = self._axis_theory(i)
            rows[name] = {
                'support_min': float(xs[nz[0]]) if len(nz) else np.nan,
                'support_max': float(xs[nz[-1]]) if len(nz) else np.nan,
                'mean': mt, 'sd': sdt, 'skew': skt,
                'right_heavy': bool(np.isfinite(skt) and skt > 1.0),
            }
        # Index-oriented construction, so the five float columns stay float and
        # ``right_heavy`` stays bool; the transposed form typed all six
        # ``object``. Same fix as :attr:`bs_window_df`.
        df = pd.DataFrame.from_dict(rows, orient='index')
        df.index.name = 'axis'
        return self._relabel(df)

    def tail_periods_df(self, periods=None):
        """Per-axis return period table on a caller's ladder.

        The parametrized worker behind :attr:`tail_df`, matching
        :meth:`Aggregate.tail_periods_df` and
        :meth:`Portfolio.tail_periods_df`.

        Parameters
        ----------
        periods : array_like of float, optional
            Return period ladder. Defaults to
            :data:`~aggregate._aggregate.DEFAULT_RETURN_PERIODS`.

        Returns
        -------
        pandas.DataFrame
            ``MultiIndex (axis, P)`` rows, columns ``T | VaR | TVaR | xsVaR |
            VaR/Mean``, the same shape :attr:`Portfolio.tail_df` uses for its
            per-unit blocks.

        Notes
        -----
        Read off the **realized marginals** of the joint grid, through a
        :class:`~aggregate._grid_distribution.GridDistribution` per axis, so
        the quantiles come from the same kernel as everywhere else and agree
        with the support :attr:`axis_support_df` reports. These are the axis
        *aggregate* distributions, compounded under the shared frequency, and
        so are not the distributions of the ``units`` the program names: a
        unit there is the per claim component (its own ``dfreq`` and
        severity), and the axis is that component compounded. The mean in
        ``.attrs`` and behind ``xsVaR`` is the realized first moment of the
        marginal, which is what the rest of the row is read from.

        **There is no total block**, unlike :attr:`Portfolio.tail_df`. The two
        axes are sized independently and routinely carry *different* bucket
        sizes (``bs`` is a list, one per axis), so the sum has no common
        lattice to land on and forming one would mean a rebucketing choice
        this class has never made. A dependent sum is also not a portfolio
        total, which assumes independence. Build the sum deliberately if you
        want it.
        """
        from ._aggregate import return_period_frame
        from ._grid_distribution import GridDistribution

        self._require_density()
        blocks, keys = [], []
        for i, (name, marginal) in enumerate(
                zip(self.unit_names, self.marginals)):
            gd = GridDistribution(self.axis_xs[i],
                                  np.asarray(marginal, dtype=float),
                                  bs=self.bs[i], name=name)
            mean = float(np.asarray(self.axis_xs[i], dtype=float)
                         @ np.asarray(marginal, dtype=float))
            blocks.append(return_period_frame(gd.q, gd.tvar, mean, periods))
            keys.append(name)
        df = pd.concat(blocks, keys=keys, names=['axis', 'P'])
        return self._relabel(df)

    @property
    def tail_df(self):
        """Per-axis return period ladder (``1.0.0a227``).

        The bivariate half of the deferred `[Loss-Lab-Round-3]` `kinds` item.
        ``BivariateAggregate`` had no ``tail_df`` at all after ``a171``
        renamed its old one to :attr:`axis_support_df` (which was a name
        collision, not an analogy), so a consumer holding four kinds had to
        dispatch on kind to know whether the frame existed.

        One block per axis; see :meth:`tail_periods_df`, which this calls with
        the default ladder, for why there is no total.

        Returns
        -------
        pandas.DataFrame
        """
        return self.tail_periods_df()

    @property
    def tail_description(self) -> str:
        """One-unit per-axis support summary.

        The verbose form is :attr:`tail_explanation`.
        """
        df = self.axis_support_df
        parts = [f'{name} [{r.support_min:.4g}, {r.support_max:.4g}]'
                 for name, r in df.iterrows()]
        return 'per-axis support: ' + '; '.join(parts)

    @property
    def tail_explanation(self) -> str:
        """Verbose prose over the per-axis realised tails and their dependence.

        The bivariate twin of
        :attr:`~aggregate.distributions.Aggregate.tail_explanation`: each
        axis's realised support and theoretical moments, whether it is
        right-heavy, and then the joint reading -- the realised Pearson
        correlation and (in copula mode) the copula's Kendall tau, which is
        what decides whether the two tails can blow out together.
        """
        df = self.axis_support_df
        out = []
        for name, r in df.iterrows():
            heavy = ('right-heavy (skew > 1)' if r.right_heavy
                     else 'not right-heavy')
            # bracket access: ``mean`` / ``skew`` are Series *methods*
            out.append(f'{name} lives on [{r.support_min:,.6g}, '
                       f'{r.support_max:,.6g}] with mean {r["mean"]:,.6g}, sd '
                       f'{r["sd"]:,.6g}, skew {r["skew"]:,.4g}: {heavy}.')
        out.append(f'The realised Pearson correlation is {self.corr:.4f}.')
        if self.mode == 'copula' and self.copula is not None:
            out.append(f'The {self.copula.name} copula has Kendall tau '
                       f'{self.copula.tau():.4f}, which is what sets how far '
                       f'into the joint tail the two axes travel together.')
        elif self.mode == 'netceded':
            out.append('The axes are comonotone views of one aggregate '
                       '(netceded mode), so the dependence is structural, not '
                       'modelled.')
        out.append('The full 1-D tail-class ladder lives on each axis\'s '
                   'standalone marginal aggregate.')
        return ' '.join(out)

    @property
    def validation_description(self) -> str:
        """One-line validation verdict, naming any check that failed.

        The short half of the pair (a172 [FCC-Contract-Gaps]); the verbose form
        is :attr:`validation_explanation` and the table behind both is the
        private :meth:`_gate_checks`.
        """
        return self._explain_oneline()

    @property
    def validation_explanation(self) -> str:
        """Long-narrative validation result for the joint grid.

        The bivariate twin of
        :attr:`~aggregate.distributions.Aggregate.validation_explanation`, and
        the verbose form of :attr:`validation_description` (the one-line
        ``validation`` row in :attr:`info`). Two checks: the **tail deficit**
        (the hard gate, since a measured grid conserves mass, so a deficit means
        a clipped or aliased window) and, loosely, that **each marginal
        reproduces its standalone aggregate** (the showpiece invariant). The
        marginal check is deliberately loose: the budget-dependent
        bs-discretization error is expected, and the exact per-axis errors are in
        :attr:`validation_df`.

        Reads :meth:`_gate_checks` rather than recomputing, so the table and the
        prose always agree.
        """
        out = []
        df = self._gate_checks()
        for check, r in df.iterrows():
            verdict = 'passes' if r['Pass'] else 'fails'
            if check == 'tail deficit':
                out.append(f'Tail deficit {r["Est"]:.2e} against the '
                           f'{r["Gate"]:.0e} gate: {verdict}.')
            elif np.isfinite(r['Err']):
                name = check[len('marginal mean '):]
                out.append(f'Marginal {name} mean {r["Est"]:,.6g} vs standalone '
                           f'{r["Ref"]:,.6g} (rel err {r["Err"]:.2e}): '
                           f'{verdict}.')
        one = self._explain_oneline()
        out.append('Not unreasonable.' if one == 'not unreasonable'
                   else f'Overall: {one}.')
        return ' '.join(out)

    def plot(self, axs=None, levels=14, log=False, **kwargs):
        """Two-panel contour plot: per-claim severity (left), aggregate (right).

        Parameters
        ----------
        axs : array of matplotlib Axes, optional
            Two target axes (e.g. ``ax0, ax1 = axs.flat``). A new ``1 x 2``
            figure is created if omitted; the figure is stored on
            :attr:`figure`.
        levels : int, default 14
            Number of contour levels.
        log : bool, default False
            Contour ``log10`` of each density (useful for heavy tails).
        **kwargs
            Passed through to ``Axes.contourf``.

        Returns
        -------
        None
            The drawn axes are reachable via ``self.figure.axes`` (or the
            ``axs`` you passed in). Returning ``None`` -- matching
            :meth:`Aggregate.plot` -- avoids the Jupyter inline backend
            rendering the figure twice (once via its post-execute hook and once
            from a returned matplotlib object).

        Notes
        -----
        The left panel is the joint **per-claim severity** ``S`` (the copula
        coupling -- or the comonotone ``(ceded, net)`` scatter in ``netceded``
        mode -- on the severity grids); the right panel is the joint
        **aggregate** density (on the output grids, P&L-relabelled for any
        ``pnl`` axis).

        After a massive update (``update(store_dir=...)``) this delegates to
        the disk-backed exhibit
        :meth:`MassiveBivariateDistribution.plot` (its own option set --
        ``window`` / ``contours`` / ``exceedance``, log color by default);
        call ``self.bivariate.plot(...)`` directly for full control.
        """
        if self._massive is not None:
            return self.bivariate.plot(**kwargs)
        from .plots import plot_bivariate
        return plot_bivariate(self, axs=axs, levels=levels, log=log, **kwargs)

    def __repr__(self):
        tag = self.mode if self.copula is None else repr(self.copula)
        if self.density is None and self._massive is None:
            return (f'BivariateAggregate(name={self.name!r}, '
                    f'units={self.unit_names!r}, {tag}, not updated)')
        shape = (len(self.axis_xs[0]), len(self.axis_xs[1]))
        massive = ', disk-backed' if self._massive is not None else ''
        return (f'BivariateAggregate(name={self.name!r}, '
                f'units={self.unit_names!r}, {tag}, '
                f'shape={shape}{massive}, corr={self.corr:.4f})')

    def _repr_html_(self):
        if self.density is None and self._massive is None:
            return f'<pre>{self!r}</pre>'
        return self.bivariate._repr_html_()


#: Conditioning kinds accepted by :meth:`JointBandsMixin.conditional` (and the
#: engine-level :meth:`BivariateAggregate.conditional` that delegates to it).
_CONDITIONAL_KINDS = ('x', 'y', 'x+y', 'x-y')


def _resolve_axis(axis, names):
    """Resolve an axis specification to the integer axis 0 or 1.

    The one resolver behind :meth:`JointBandsMixin.marginal`,
    :meth:`JointBandsMixin.conditional` and their engine-level twins, so every
    accessor accepts the same spellings.

    Parameters
    ----------
    axis : int or str
        ``0`` or ``1``; the aliases ``'x'`` (axis 0) and ``'y'`` (axis 1); or
        an axis name from ``names``. Name matching is case insensitive so a
        netceded view answers to its natural spelling (``'net'`` for the unit
        name ``'Net'``).
    names : sequence of str
        The two axis names, in axis order.

    Returns
    -------
    int
        The resolved axis, 0 or 1.

    Raises
    ------
    ValueError
        If ``axis`` is none of 0, 1, ``'x'``, ``'y'``, or a name in
        ``names``.
    """
    if isinstance(axis, str):
        low = axis.lower()
        if low == 'x':
            return 0
        if low == 'y':
            return 1
        lows = [str(n).lower() for n in names]
        if low in lows:
            return lows.index(low)
        raise ValueError(
            f"axis must be 0, 1, 'x', 'y' or one of {tuple(names)}; "
            f'got {axis!r}.')
    axis = int(axis)
    if axis not in (0, 1):
        raise ValueError(f'axis must be 0 or 1, not {axis!r}')
    return axis


class JointBandsMixin:
    """Row-wise reads of a joint density, wherever the density lives.

    The one place that knows whether a joint is a numpy array or a zarr store,
    so every row-wise consumer above it stops caring. Both containers already
    address their axes by role (``axis0`` / ``axis1`` / ``bs0`` / ``bs1`` /
    ``axis_names``, the positional ``.ceded`` / ``.net`` names lie for a
    ``('gross', 'ceded')`` joint), which is what makes one implementation
    possible: this mixin reads those and ``density`` and nothing else.

    Mixin, not a base class, and it defines no ``__init__``: the two containers
    are independently constructed and share no state beyond the surface named
    above (``CLAUDE.md``, naming conventions).
    """

    #: Rows per band on the massive route when the store cannot say. Never
    #: reached in practice: a zarr array carries its own chunking, which is the
    #: read size the store was written for.
    _DEFAULT_BAND_ROWS = 512

    def _row_bands(self, axis=0, band_rows=None):
        """Yield ``(r0, r1, block)`` row bands of the joint, in core or on disk.

        Parameters
        ----------
        axis : {0, 1}, default 0
            The conditioning axis. ``1`` yields bands of the **transpose**, so
            a consumer always folds along rows and the transpose is not a
            special case for the caller.
        band_rows : int, optional
            Rows per band. Defaults to the whole grid in core (one band) and to
            the store's own row chunk on the massive route, which is the read
            size the chunking was chosen for.

        Yields
        ------
        (int, int, ndarray)
            Half open row range and the dense float64 block for it.

        Notes
        -----
        The in core case yields exactly one band, so a consumer written against
        this iterator costs nothing there: no copy, no chunk arithmetic, one
        pass either way. On the massive route the band is the unit of disk
        read, and peak memory is ``band_rows * n_other * 8`` bytes regardless
        of grid size.
        """
        axis = int(axis)
        if axis not in (0, 1):
            raise ValueError(f'axis must be 0 or 1, not {axis!r}')
        dens = self.density
        n_rows = dens.shape[axis]
        in_core = isinstance(dens, np.ndarray)
        if band_rows is None:
            chunks = getattr(dens, 'chunks', None)
            band_rows = (n_rows if in_core
                         else (chunks[axis] if chunks
                               else self._DEFAULT_BAND_ROWS))
        band_rows = max(int(band_rows), 1)
        for r0 in range(0, n_rows, band_rows):
            r1 = min(r0 + band_rows, n_rows)
            block = (dens[r0:r1, :] if axis == 0 else dens[:, r0:r1])
            block = np.asarray(block, dtype=float)
            yield r0, r1, (block if axis == 0 else block.T)

    def slice(self, x=None, y=None):
        """Conditional distribution along one axis: ``P(Y | X ~ x)`` or ``P(X | Y ~ y)``.

        The analyst's probe: one row (or column) of the joint, instant at any
        grid size and on either route.

        Parameters
        ----------
        x, y : float, optional
            Exactly one must be given: the conditioning value, snapped to the
            nearest bucket.

        Returns
        -------
        GridDistribution
            The normalized conditional law on the other axis.

        Raises
        ------
        ValueError
            When neither or both values are given, or when the conditioning
            slice carries no mass.
        """
        if (x is None) == (y is None):
            raise ValueError('give exactly one of x= or y=.')
        axis = 0 if x is not None else 1
        value = x if x is not None else y
        grids = (self.axis0, self.axis1)
        steps = (self.bs0, self.bs1)
        own, other = grids[axis], grids[1 - axis]
        idx = int(np.clip(round((value - own[0]) / steps[axis]),
                          0, len(own) - 1))
        row = np.asarray(self.density[idx, :] if axis == 0
                         else self.density[:, idx], dtype=float).ravel()
        name = (f'{self.axis_names[1 - axis]} | '
                f'{self.axis_names[axis]}={own[idx]:g}')
        tot = row.sum()
        if tot <= 0:
            raise ValueError(
                f'no mass on the conditioning slice ({name}); pick a value '
                'inside the support (see .marginal()).')
        return GridDistribution(other, row / tot, bs=steps[1 - axis],
                                name=name)

    def marginal(self, axis=0, *, name=None, is_loss_value=True):
        """Axis marginal as a :class:`~aggregate._grid_distribution.GridDistribution`.

        Parameters
        ----------
        axis : int or str, default 0
            ``0`` / ``1``, the aliases ``'x'`` / ``'y'``, or an axis name from
            :attr:`axis_names` (case insensitive); see :func:`_resolve_axis`.
        name : str, optional
            Display name for the returned distribution; defaults to the axis
            name.
        is_loss_value : bool, default True
            Orientation flag passed through to the ``GridDistribution``
            (``False`` for a payoff axis, where more is better).

        Returns
        -------
        GridDistribution
            The realized marginal law on the axis grid.

        Notes
        -----
        Reads :meth:`marginals`, so the cost follows the route: one density
        fold in core, and the free pass-3 accumulators on the massive route
        (no disk read). The raw arrays stay available from :meth:`marginals`
        as the zero-copy primitive.
        """
        i = _resolve_axis(axis, self.axis_names)
        dens = np.asarray(self.marginals()[i], dtype=float)
        grid = (self.axis0, self.axis1)[i]
        bs = (self.bs0, self.bs1)[i]
        if name is None:
            name = str(self.axis_names[i])
        return GridDistribution(grid, dens, bs=bs, name=name,
                                is_loss_value=is_loss_value)

    def conditional(self, kind, value, report=None, *, is_loss_value=True):
        r"""Conditional law of one axis given an event, as a ``GridDistribution``.

        The full one-point conditional, where :meth:`~BivariateAggregate.exeqa_df`
        is the conditional **mean** swept over the whole conditioning grid.

        Parameters
        ----------
        kind : {'x', 'y', 'x+y', 'x-y'}
            The conditioning variable. ``'x'`` conditions on axis 0 and
            reports the law of axis 1 (delegating to :meth:`slice`); ``'y'``
            is the transpose. ``'x+y'`` and ``'x-y'`` condition on the total
            or the difference landing in the bucket containing ``value``.
        value : float
            The conditioning value, snapped to the bucket containing it.
        report : int or str, optional
            Which axis's law the returned distribution is expressed in
            (``0`` / ``1`` / ``'x'`` / ``'y'`` or an axis name). For
            ``'x'`` / ``'y'`` the reported axis is determined (the non
            conditioning axis) and a contradictory ``report`` raises; for the
            diagonal kinds it defaults to axis 0. The two diagonal readings
            are affine images of each other (``y = value - x`` on the band),
            so this is a fold-axis choice, not a second computation.
        is_loss_value : bool, default True
            Orientation flag for the returned ``GridDistribution``.

        Returns
        -------
        GridDistribution
            The normalized conditional law on the reported axis's grid.

        Raises
        ------
        ValueError
            On an unknown ``kind``, a contradictory ``report``, or a
            conditioning event carrying no mass.

        Notes
        -----
        The conditioning event for the diagonal kinds is the **total-grid
        bucket**: cells with ``round((x_i + s y_j - value) / bs_total) == 0``
        where ``s`` is the sign of the ``y`` term. Where the axes share a
        ``bs`` this is the exact lattice anti-diagonal (``bs_total = bs``);
        where they differ, ``bs_total = max(bs0, bs1)`` and the event is
        honest (a bucket of the total), consistent with :meth:`slice`
        conditioning on a bucket rather than on a measure-zero line. This is
        the single-value version of the deferred ``[Bivariate-Total-Exeqa]``
        sweep and shares its band arithmetic.

        Cost by route: ``'x'`` reads one row band (cheap on disk); ``'y'``
        and the diagonal kinds sweep :meth:`_row_bands` accumulating the
        masked fold, bounded memory, no new storage.
        """
        kind = str(kind).lower().replace(' ', '')
        if kind not in _CONDITIONAL_KINDS:
            raise ValueError(
                f'kind must be one of {_CONDITIONAL_KINDS}; got {kind!r}.')
        names = self.axis_names
        if kind in ('x', 'y'):
            determined = 1 if kind == 'x' else 0
            if report is not None and \
                    _resolve_axis(report, names) != determined:
                raise ValueError(
                    f'conditional({kind!r}, ...) reports the law of the '
                    f'other axis (axis {determined}); report={report!r} '
                    'contradicts that.')
            gd = self.slice(x=value) if kind == 'x' else self.slice(y=value)
            if is_loss_value:
                return gd
            return GridDistribution(gd.x, gd.p, bs=gd.bs, name=gd.name,
                                    is_loss_value=False)
        rep = 0 if report is None else _resolve_axis(report, names)
        sign = 1.0 if kind == 'x+y' else -1.0
        x0 = np.asarray(self.axis0, dtype=float)
        x1 = np.asarray(self.axis1, dtype=float)
        bs_total = (self.bs0 if self.bs0 == self.bs1
                    else max(self.bs0, self.bs1))
        value = float(value)
        out = np.zeros(len(x0) if rep == 0 else len(x1))
        for r0, r1, block in self._row_bands(axis=0):
            t = x0[r0:r1, None] + sign * x1[None, :]
            masked = np.where(np.round((t - value) / bs_total) == 0,
                              block, 0.0)
            if rep == 0:
                out[r0:r1] += masked.sum(axis=1)
            else:
                out += masked.sum(axis=0)
        op = '+' if kind == 'x+y' else '-'
        cname = f'{names[rep]} | {names[0]} {op} {names[1]}={value:g}'
        tot = out.sum()
        if tot <= 0:
            raise ValueError(
                f'no mass on the conditioning band ({cname}); pick a value '
                'the joint actually reaches (see .total()).')
        return GridDistribution((x0, x1)[rep], out / tot,
                                bs=(self.bs0, self.bs1)[rep], name=cname,
                                is_loss_value=is_loss_value)

    def total(self, *, name='total', is_loss_value=True):
        r"""Law of the sum of the two axes as a ``GridDistribution``.

        Parameters
        ----------
        name : str, default 'total'
            Display name for the returned distribution.
        is_loss_value : bool, default True
            Orientation flag for the returned ``GridDistribution``.

        Returns
        -------
        GridDistribution
            The realized law of ``X + Y`` on the total grid.

        Notes
        -----
        Where the two axes share a ``bs`` the anti-diagonal is lattice
        aligned and the fold is **exact**: the mass at total index ``k`` is
        ``sum_i d[i, k - i]``, accumulated row by row with no rebucketing.
        Where the ``bs`` differ, each cell's mass is routed onto the total
        grid (``bs_total = max(bs0, bs1)``) through the mean-preserving
        linear scatter :func:`_scatter_1d`, the ``[Bivariate-Total-Exeqa]``
        routing. Either way the sweep runs over :meth:`_row_bands`, so a
        disk-backed joint answers at bounded memory in one pass, and the
        cost is ``O(n m)``.
        """
        x0 = np.asarray(self.axis0, dtype=float)
        x1 = np.asarray(self.axis1, dtype=float)
        z0 = float(x0[0] + x1[0])
        if self.bs0 == self.bs1:
            bs_t = float(self.bs0)
            n_out = len(x0) + len(x1) - 1
            out = np.zeros(n_out)
            n1 = len(x1)
            for r0, r1, block in self._row_bands(axis=0):
                for i in range(r0, r1):
                    out[i:i + n1] += block[i - r0]
        else:
            bs_t = float(max(self.bs0, self.bs1))
            hi = float(x0[-1] + x1[-1])
            n_out = int(np.ceil((hi - z0) / bs_t)) + 1
            out = np.zeros(n_out)
            for r0, r1, block in self._row_bands(axis=0):
                t = (x0[r0:r1, None] + x1[None, :]).ravel()
                mass, _ = _scatter_1d(t, block.ravel(), z0, bs_t, n_out)
                out += mass
        z = z0 + bs_t * np.arange(n_out)
        return GridDistribution(z, out, bs=bs_t, name=name,
                                is_loss_value=is_loss_value)


class BivariateDistribution(JointBandsMixin):
    """Joint distribution of two aggregate quantities on a 2D grid.

    A lightweight container for a 2D density (e.g. the joint occurrence ceded
    ``C`` / net ``N`` returned by
    :meth:`aggregate.distributions.Aggregate.occ_bivariate`, or the joint of two
    copula-coupled components from :class:`BivariateAggregate`), with
    marginals, mixed moments, correlation, and a contour plot.

    Parameters
    ----------
    density : ndarray
        Joint probability mass, shape ``(len(ceded), len(net))``; entry
        ``[a, b]`` is ``P(C ~ ceded[a], N ~ net[b])``.
    ceded, net : ndarray
        1D axis grids.
    bs_ceded, bs_net : float
        Axis bucket sizes.
    meta : dict, optional
        Provenance (``name``, ``en``, ``freq_name``, ``deficit``, ...).

    Attributes
    ----------
    density, ceded, net, bs_ceded, bs_net, meta
        As constructed.
    """

    def __init__(self, density, ceded, net, bs_ceded, bs_net, meta=None):
        self.density = np.asarray(density, dtype=float)
        self.ceded = np.asarray(ceded, dtype=float)
        self.net = np.asarray(net, dtype=float)
        self.bs_ceded = float(bs_ceded)
        self.bs_net = float(bs_net)
        self.meta = dict(meta) if meta else {}

    # ------------------------------------------------------------------
    # axis access by role (not by the positional .ceded / .net names)
    # ------------------------------------------------------------------
    # The constructor stores the two grids positionally as ``.ceded`` (axis 0)
    # and ``.net`` (axis 1) for the original netceded use, but for a
    # ``('gross', 'ceded')`` joint axis 0 is *gross loss* L and axis 1 is the
    # *unlimited ceded recovery* R -- the positional names lie. Pushforward and
    # moment code must address axes by role through these accessors and
    # :attr:`axis_names`, never by ``.ceded`` / ``.net``.
    @property
    def axis0(self):
        """Axis-0 grid (the first ``views`` entry; positionally ``.ceded``)."""
        return self.ceded

    @property
    def axis1(self):
        """Axis-1 grid (the second ``views`` entry; positionally ``.net``)."""
        return self.net

    @property
    def axis_names(self):
        """``(name0, name1)`` axis roles from ``meta`` (default ceded/net)."""
        return self.meta.get('axis_names', ('ceded (C)', 'net (N)'))

    @property
    def bs0(self):
        """Axis-0 bucket size (positionally ``.bs_ceded``)."""
        return self.bs_ceded

    @property
    def bs1(self):
        """Axis-1 bucket size (positionally ``.bs_net``)."""
        return self.bs_net

    def pushforward(self, function, *, bs=None, log2=None, window=None,
                    scheme='linear', name=None, is_loss_value=True,
                    chunk_size=None, bs_total=None, total_key='total',
                    windows=None):
        """Pushforward of the joint density through a scalar map ``z = function(axis0, axis1)``.

        ``function`` may also be a **dict** ``{key: f(x, y)}`` -- the
        multi-function form (plan-bv §6, shared with
        :meth:`MassiveBivariateDistribution.pushforward`): every ``f_i`` (and,
        for two or more, their pre-bucket **total** under ``total_key``) is
        scattered in one pass and a ``dict[str, GridDistribution]`` returned.
        In the dict form ``bs`` is required (scalar or one per key),
        ``bs_total`` is required for two or more functions, ``windows=``
        optionally pins per-key output ranges, and ``log2`` / ``window`` /
        ``name`` / ``chunk_size`` do not apply.

        Scatters every joint cell's probability onto a 1-D output grid at its
        transformed value, yielding the exact distribution of ``Z =
        function(L, R)`` (pre-plan section 9). This is the **correct** object for a
        deterministic, generally nonlinear function of the two dependent axes --
        a marginal convolution would be wrong. The result is a
        :class:`~aggregate._grid_distribution.GridDistribution` carrying
        ``q`` / ``cdf`` / ``sf`` / ``tvar`` / moments, ready for
        :class:`~aggregate.PnL` reporting.

        Parameters
        ----------
        function : callable
            Vectorized ``function(x, y) -> z``, evaluated on broadcast axis
            arrays ``function(axis0[:, None], axis1[None, :])``. Must broadcast to
            the joint shape; addresses the axes **by role** (axis 0, axis 1), not
            by the misleading positional ``.ceded`` / ``.net`` names.
        bs : float, optional
            Output bucket size. Default: sized from the realized value range.
        log2 : int, optional
            Output grid length ``1 << log2`` (default :data:`_PUSHFORWARD_LOG2`
            when ``bs`` is unset).
        window : (float, float), optional
            Explicit ``(lo, hi)`` output range (may be signed).
        scheme : {'linear', 'nearest'}, default 'linear'
            Rebucketing scheme; ``'linear'`` preserves the mean.
        name : str, optional
            Name for the returned distribution.
        is_loss_value : bool, default True
            Orientation of ``Z`` (``True`` loss, ``False`` payoff) -- set
            ``False`` for an underwriting-result leg.
        chunk_size : int, optional
            Process axis-0 in row chunks of this many rows to bound temporary
            memory (pre-plan section 9.7); ``None`` evaluates the whole grid at
            once.

        Returns
        -------
        GridDistribution
            The law of ``Z``; ``.clipped_mass`` reports edge-clipped mass.

        Notes
        -----
        Complexity is ``O(n0 * n1)`` -- one pass over the joint cells -- using
        NumPy broadcasting and two :func:`numpy.bincount` scatters (no Numba).
        """
        a0 = self.axis0
        a1 = self.axis1
        dens = self.density
        n0 = len(a0)
        if isinstance(function, dict):
            if bs is None:
                raise ValueError('the dict pushforward requires bs= (the '
                                 'caller knows their output scale).')
            if log2 is not None or window is not None or name is not None:
                raise ValueError('log2 / window / name do not apply to the '
                                 'dict pushforward; use bs= and windows=.')
            step = chunk_size or n0

            def bands():
                for lo in range(0, n0, step):
                    hi = min(lo + step, n0)
                    yield lo, hi, dens[lo:hi]

            return _pushforward_functions(
                bands, np.asarray(a0, dtype=float), np.asarray(a1, dtype=float),
                float(dens.sum()), function, bs, bs_total=bs_total,
                total_key=total_key, windows=windows, scheme=scheme,
                is_loss_value=is_loss_value, source=self)
        if bs_total is not None or windows is not None:
            raise ValueError('bs_total / windows apply only to the dict '
                             'pushforward form.')
        if chunk_size is None or n0 <= chunk_size:
            values = np.broadcast_to(
                np.asarray(function(a0[:, None], a1[None, :]), dtype=float),
                dens.shape)
            return _finalize_pushforward(
                values, dens, bs=bs, log2=log2, window=window, scheme=scheme,
                name=name, is_loss_value=is_loss_value, source=self)
        # chunked: a first pass measures the range, a second scatters
        vmin, vmax = np.inf, -np.inf
        for lo in range(0, n0, chunk_size):
            v = function(a0[lo:lo + chunk_size, None], a1[None, :])
            vmin = min(vmin, float(np.min(v)))
            vmax = max(vmax, float(np.max(v)))
        z0, bs_out, n_out = _pushforward_grid(vmin, vmax, bs=bs, log2=log2,
                                              window=window)
        mass = np.zeros(n_out)
        clipped = 0.0
        for lo in range(0, n0, chunk_size):
            sl = slice(lo, lo + chunk_size)
            v = np.broadcast_to(
                np.asarray(function(a0[sl, None], a1[None, :]), dtype=float),
                dens[sl].shape)
            m, c = _scatter_1d(v, dens[sl], z0, bs_out, n_out, scheme)
            mass += m
            clipped += c
        grid = z0 + bs_out * np.arange(n_out)
        gd = GridDistribution(grid, mass, bs=bs_out, name=name or '',
                              is_loss_value=is_loss_value)
        gd.clipped_mass = clipped
        gd.pushforward_source = self
        return gd

    def transformed_moments(self, function, max_order=3):
        """Exact raw and central moments of ``Z = function(axis0, axis1)`` on the source grid.

        The **exact** ``E[Z^k] = sum p_ij function(l_i, r_j)^k`` taken directly on
        the joint grid -- no rebucketing -- so these are the headline "EX"
        (exact) numbers the audit validates the pushforward "Est" against
        (pre-plan section 15). Means add across legs by linearity; this is the
        ground truth they add to.

        Parameters
        ----------
        function : callable
            Vectorized ``function(x, y) -> z`` on broadcast axis arrays (same
            contract as :meth:`pushforward`).
        max_order : int, default 3
            Highest raw power taken.

        Returns
        -------
        pandas.Series
            ``mass``, ``mean``, and (to ``max_order``) ``var`` / ``sd`` / ``cv``
            / ``skew``.
        """
        a0 = self.axis0
        a1 = self.axis1
        p = self.density
        values = np.broadcast_to(
            np.asarray(function(a0[:, None], a1[None, :]), dtype=float), p.shape)
        tot = float(p.sum())
        raw = [float(np.sum(p * values ** k)) for k in range(max_order + 1)]
        out = {'mass': raw[0]}
        mean = raw[1] / tot if tot else np.nan
        out['mean'] = mean
        if max_order >= 2:
            var = raw[2] / tot - mean ** 2
            sd = np.sqrt(var) if var > 0 else 0.0
            out['var'] = var
            out['sd'] = sd
            out['cv'] = sd / mean if mean else np.nan
        if max_order >= 3 and out.get('sd', 0) > 0:
            m3 = raw[3] / tot - 3 * mean * (raw[2] / tot) + 2 * mean ** 3
            out['skew'] = m3 / out['sd'] ** 3
        return pd.Series(out, name=getattr(function, '__name__', 'Z'))

    def marginals(self):
        """Return the ceded and net marginal densities.

        Returns
        -------
        ceded_density : ndarray
            ``density.sum(axis=1)`` on the ``ceded`` grid.
        net_density : ndarray
            ``density.sum(axis=0)`` on the ``net`` grid.

        Notes
        -----
        These should match the univariate occurrence aggregate margins
        ``reins_density_df['p_agg_ceded_occ']`` and ``['p_agg_net_occ']``
        (rebucketed to the bivariate grids) and provide the primary numerical
        validation of the 2D convolution.
        """
        return self.density.sum(axis=1), self.density.sum(axis=0)

    def moments(self, max_order=3):
        """Mixed raw moments ``E[C^i N^j]`` for ``i, j in 0..max_order``.

        Parameters
        ----------
        max_order : int, default 3
            Highest power taken on each axis.

        Returns
        -------
        DataFrame
            ``(max_order + 1) x (max_order + 1)`` table; row ``i`` / column
            ``j`` is ``E[C^i N^j] = sum density * ceded**i * net**j``. The
            ``[0, 0]`` entry is the total probability (``~1``).

        Notes
        -----
        Computed as ``Ci @ density @ Nj.T`` where ``Ci[i] = ceded**i`` and
        ``Nj[j] = net**j`` -- a vectorised double sum over the grid.
        """
        ci = np.vstack([self.ceded ** i for i in range(max_order + 1)])
        nj = np.vstack([self.net ** j for j in range(max_order + 1)])
        m = ci @ self.density @ nj.T
        return pd.DataFrame(
            m,
            index=pd.Index([f'C^{i}' for i in range(max_order + 1)], name='C'),
            columns=pd.Index([f'N^{j}' for j in range(max_order + 1)], name='N'))

    def corr(self):
        """Pearson correlation of the two axes.

        Returns
        -------
        float
            ``Cov(C, N) / sqrt(Var(C) Var(N))``, in ``[-1, 1]``; ``nan`` if
            either margin is degenerate (zero variance).
        """
        m = self.moments(2).to_numpy()
        tot = m[0, 0]
        e_c = m[1, 0] / tot
        e_n = m[0, 1] / tot
        var_c = m[2, 0] / tot - e_c ** 2
        var_n = m[0, 2] / tot - e_n ** 2
        cov = m[1, 1] / tot - e_c * e_n
        denom = var_c * var_n
        if denom <= 0:
            return np.nan
        return float(cov / np.sqrt(denom))

    def contour(self, ax=None, levels=14, log=False, **kwargs):
        """Filled contour plot of the joint density.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            Target axes; a new figure is created if omitted (using the
            project ``FIG_W`` / ``FIG_H`` and constrained layout).
        levels : int, default 14
            Number of contour levels.
        log : bool, default False
            Contour ``log10`` of the density (clipped at the smallest positive
            value) -- useful for the heavy-tailed dependency structure.
        **kwargs
            Passed through to ``Axes.contourf``.

        Returns
        -------
        matplotlib Axes
            The axes drawn on.
        """
        from .plots import plot_bivariate_distribution
        return plot_bivariate_distribution(self, ax=ax, levels=levels, log=log, **kwargs)

    def _summary(self):
        """Return ``(E[C], E[N], corr, deficit)`` for the repr builders."""
        m = self.moments(1).to_numpy()
        tot = m[0, 0]
        return m[1, 0] / tot, m[0, 1] / tot, self.corr(), self.meta.get('deficit', 0.0)

    def __repr__(self):
        e_c, e_n, rho, deficit = self._summary()
        return (f'BivariateDistribution(name={self.meta.get("name", "")!r}, '
                f'shape={self.density.shape}, '
                f'bs=({self.bs_ceded:g}, {self.bs_net:g}), '
                f'E[C]={e_c:,.4g}, E[N]={e_n:,.4g}, corr={rho:.4f})')

    def _repr_html_(self):
        e_c, e_n, rho, deficit = self._summary()
        names = self.meta.get('axis_names', ('ceded (C)', 'net (N)'))
        rows = [
            ('name', self.meta.get('name', '')),
            ('grid shape', f'{self.density.shape[0]} &times; {self.density.shape[1]}'),
            ('bucket', f'{self.bs_ceded:g}, {self.bs_net:g}'),
            (f'E[{names[0]}]', f'{e_c:,.6g}'),
            (f'E[{names[1]}]', f'{e_n:,.6g}'),
            ('correlation', f'{rho:.6f}'),
            ('tail deficit', f'{deficit:.2e}'),
        ]
        body = ''.join(
            f'<tr><th style="text-align:left">{k}</th><td>{v}</td></tr>'
            for k, v in rows)
        return (f'<table class="aggregate bivariate">'
                f'<caption>BivariateDistribution</caption>{body}</table>')


class MassiveBivariateDistribution(JointBandsMixin):
    """Disk-backed joint distribution of two aggregate quantities.

    The massive sibling of :class:`BivariateDistribution`
    (``dev/plan-bv.md`` §5): the realized joint density lives in a zarr store
    (``density.zarr`` in ``store_dir``, physical order) and is only ever read
    band by band; the container carries the exact pass-3 accumulators --
    marginals, total mass / deficit, and the mixed raw moments
    ``E[X^a Y^b], a, b <= 3`` -- so the headline surface (marginals, moments,
    correlation, repr) costs no disk read at all. Same role accessors
    (:attr:`axis0`, :attr:`axis1`, :attr:`axis_names`) as the in-core
    container, so pushforward / moment code addresses axes by role.

    The store directory is self-describing (``meta.json`` +
    ``accumulators.npz`` written by :meth:`save`), so :meth:`reopen`
    reconstructs the container in a later session without re-running the FFT.

    Parameters
    ----------
    store_dir : str
        The backing directory.
    xs0, xs1 : ndarray
        Per-axis output label grids.
    bs0, bs1 : float
        Per-axis bucket sizes.
    marg0, marg1 : ndarray
        Exact axis marginals.
    total_mass, deficit : float
        Realized joint mass and its complement.
    raw_moments : ndarray
        ``(4, 4)`` unnormalised mixed raw moment sums.
    meta : dict, optional
        Provenance (``name``, ``en``, ``freq_name``, ``copula``,
        ``axis_names``, ``mode``, ...).
    density : zarr.Array, optional
        An already-open handle (fresh from the kernel); opened lazily from
        ``store_dir`` when omitted (the :meth:`reopen` path).
    """

    _META_FILE = 'meta.json'
    _ACC_FILE = 'accumulators.npz'

    def __init__(self, store_dir, xs0, xs1, bs0, bs1, marg0, marg1,
                 total_mass, deficit, raw_moments, meta=None, density=None):
        self.store_dir = str(store_dir)
        self.xs0 = np.asarray(xs0, dtype=float)
        self.xs1 = np.asarray(xs1, dtype=float)
        self.bs0 = float(bs0)
        self.bs1 = float(bs1)
        self.marg0 = np.asarray(marg0, dtype=float)
        self.marg1 = np.asarray(marg1, dtype=float)
        self.total_mass = float(total_mass)
        self.deficit = float(deficit)
        self.raw_moments = np.asarray(raw_moments, dtype=float)
        self.meta = dict(meta) if meta else {}
        self._density = density

    @classmethod
    def from_result(cls, res, meta=None):
        """Build from a kernel :class:`~aggregate._aggregate_compute_massive.MassiveResult`."""
        return cls(res.store_dir, res.xs0, res.xs1, res.bs0, res.bs1,
                   res.marg0, res.marg1, res.total_mass, res.deficit,
                   res.raw_moments, meta=meta, density=res.density)

    # ------------------------------------------------------------------
    # axis access by role (mirrors BivariateDistribution)
    # ------------------------------------------------------------------
    @property
    def axis0(self):
        """Axis-0 grid."""
        return self.xs0

    @property
    def axis1(self):
        """Axis-1 grid."""
        return self.xs1

    @property
    def axis_names(self):
        """``(name0, name1)`` axis roles from ``meta``."""
        return tuple(self.meta.get('axis_names', ('X', 'Y')))

    @property
    def density(self):
        """The joint density as a **lazy zarr array view** (never in RAM).

        Duck-types for slicing -- ``bv.density[1000:1010, :]`` reads just
        those tiles. A full ``[:]`` read is the user explicitly asking for
        the whole (potentially tens-of-GB) array and getting what they asked
        for.
        """
        if self._density is None:
            from ._aggregate_compute_massive import _require_zarr
            zarr = _require_zarr()
            self._density = zarr.open(
                os.path.join(self.store_dir, 'density.zarr'), mode='r+')
        return self._density

    # ------------------------------------------------------------------
    # accumulator-backed surface (no disk reads)
    # ------------------------------------------------------------------
    def marginals(self):
        """Return the two marginal densities (exact, precomputed)."""
        return self.marg0, self.marg1

    def moments(self, max_order=3):
        """Mixed raw moments ``E[X^i Y^j]``, same frame as the in-core container.

        Orders up to 3 per axis come from the pass-3 accumulators (free);
        a larger ``max_order`` streams one band sweep over the on-disk
        density.

        Parameters
        ----------
        max_order : int, default 3
            Highest power taken on each axis.

        Returns
        -------
        DataFrame
            As :meth:`BivariateDistribution.moments` (``C^i`` rows /
            ``N^j`` columns -- the historical labels).
        """
        if max_order < self.raw_moments.shape[0]:
            m = self.raw_moments[:max_order + 1, :max_order + 1]
        else:
            m = self._streamed_mixed_moments(max_order)
        return pd.DataFrame(
            m,
            index=pd.Index([f'C^{i}' for i in range(max_order + 1)], name='C'),
            columns=pd.Index([f'N^{j}' for j in range(max_order + 1)], name='N'))

    def _streamed_mixed_moments(self, max_order):
        """One band sweep over ``density.zarr`` for moments above order 3."""
        dens = self.density
        rc = int(dens.chunks[0])
        n0 = len(self.xs0)
        ypow = np.vstack([self.xs1 ** b for b in range(max_order + 1)])
        parts = []
        for r0 in range(0, n0, rc):
            r1 = min(r0 + rc, n0)
            band = np.asarray(dens[r0:r1, :])
            band_y = band @ ypow.T
            mom = np.empty((max_order + 1, max_order + 1))
            xa = np.ones(r1 - r0)
            for a in range(max_order + 1):
                mom[a, :] = xa @ band_y
                xa = xa * self.xs0[r0:r1]
            parts.append(mom)
        return np.sum(np.stack(parts), axis=0)

    def corr(self):
        """Pearson correlation of the two axes (from the accumulators)."""
        m = self.raw_moments
        tot = m[0, 0]
        e0, e1 = m[1, 0] / tot, m[0, 1] / tot
        v0 = m[2, 0] / tot - e0 ** 2
        v1 = m[0, 2] / tot - e1 ** 2
        cov = m[1, 1] / tot - e0 * e1
        denom = v0 * v1
        if denom <= 0:
            return np.nan
        return float(cov / np.sqrt(denom))

    # ------------------------------------------------------------------
    # streamed probes
    # ------------------------------------------------------------------
    # The probability surface (``marginal`` / ``conditional`` / ``total`` /
    # ``slice`` / ``_row_bands``) lives on JointBandsMixin: the same
    # implementation on both sides of the disk boundary, folding over row
    # bands. ``marginal`` reads :meth:`marginals`, which here returns the
    # free pass-3 accumulators, so it still costs no disk read.

    def pushforward(self, functions, bs, *, bs_total=None, total_key='total',
                    windows=None, scheme='linear', is_loss_value=True):
        """Streamed pushforwards of ``{key: f(X, Y)}`` -- one pass over the disk.

        The money method (``dev/plan-bv.md`` §6): every ``f_i`` -- and, for
        two or more, their **total** ``t = sum f_i`` -- is scattered onto its
        own 1-D output grid in a single band sweep of ``density.zarr``. The
        expensive FFT never reruns (update is decoupled), so this is
        repeatable at will: one full read of the store per call, regardless
        of how many functions ride along.

        Parameters
        ----------
        functions : dict[str, callable or float]
            Named vectorized maps ``f(x, y)`` on broadcast label arrays
            (role-addressed axes, the :meth:`BivariateDistribution.pushforward`
            contract), or plain numbers. Constants -- numbers, or callables
            constant on a coarse probe lattice (a genuinely non-constant
            function that fools the probe is pathological and out of scope)
            -- are short-circuited before the sweep: a point mass at the
            constant carrying ``total_mass``, a scalar shift inside the
            total, zero band-loop evaluations.
        bs : float or array-like
            Output bucket size(s) -- required, the caller knows their output
            scale. Scalar applies to every key; an array is matched to
            ``functions`` in iteration order.
        bs_total : float, optional
            Bucket size for the total -- **required** with two or more
            functions, forbidden with one (no total is computed then: the
            usual, fast case).
        total_key : str, default 'total'
            Key under which the total is returned; a collision with a user
            key raises. The total is accumulated **pre-bucketing** (evaluated
            exactly per cell, summed, bucketed once) -- never as a sum of
            bucketed results -- so ``mean(total) == sum(mean(f_i))`` exactly.
        windows : dict[str, (float, float)], optional
            Explicit ``(lo, hi)`` output ranges per key (including
            ``total_key``), skipping the label-only pre-sweep for those keys.
            Values outside a pinned window clip onto the edge buckets and are
            reported via ``.clipped_mass`` (the house convention).
        scheme : {'linear', 'nearest'}, default 'linear'
            Rebucketing scheme; ``'linear'`` preserves the mean.
        is_loss_value : bool, default True
            Orientation of the outputs.

        Returns
        -------
        dict[str, GridDistribution]
            One per key, plus ``total_key`` when two or more functions are
            given. Each carries ``.clipped_mass``, ``.pushforward_source``
            and the shared Est-vs-EX audit frame ``.pushforward_audit_df``
            (the streamed exact moments cost nothing extra -- they fold in
            the same sweep).
        """
        dens = self.density
        rc = int(dens.chunks[0])
        n0 = len(self.xs0)

        def bands():
            for r0 in range(0, n0, rc):
                r1 = min(r0 + rc, n0)
                yield r0, r1, np.asarray(dens[r0:r1, :])

        return _pushforward_functions(
            bands, self.xs0, self.xs1, self.total_mass, functions, bs,
            bs_total=bs_total, total_key=total_key, windows=windows,
            scheme=scheme, is_loss_value=is_loss_value, source=self)

    def transformed_moments(self, function, max_order=3):
        """Exact raw / central moments of ``Z = function(axis0, axis1)``, streamed.

        The massive version of
        :meth:`BivariateDistribution.transformed_moments` -- the same "EX"
        audit numbers, folded over one band sweep of the on-disk density.

        Parameters
        ----------
        function : callable
            Vectorized ``function(x, y) -> z`` on broadcast label arrays
            (role-addressed axes).
        max_order : int, default 3
            Highest raw power taken.

        Returns
        -------
        pandas.Series
            ``mass``, ``mean``, and (to ``max_order``) ``var`` / ``sd`` /
            ``cv`` / ``skew``.
        """
        dens = self.density
        rc = int(dens.chunks[0])
        n0 = len(self.xs0)
        yb = self.xs1[None, :]
        raw = np.zeros(max_order + 1)
        for r0 in range(0, n0, rc):
            r1 = min(r0 + rc, n0)
            band = np.asarray(dens[r0:r1, :])
            values = np.broadcast_to(
                np.asarray(function(self.xs0[r0:r1, None], yb), dtype=float),
                band.shape)
            vk = np.ones_like(band)
            for k in range(max_order + 1):
                raw[k] += float(np.sum(band * vk))
                if k < max_order:
                    vk = vk * values
        tot = float(self.total_mass)
        out = {'mass': raw[0]}
        mean = raw[1] / tot if tot else np.nan
        out['mean'] = mean
        if max_order >= 2:
            var = raw[2] / tot - mean ** 2
            sd = np.sqrt(var) if var > 0 else 0.0
            out['var'] = var
            out['sd'] = sd
            out['cv'] = sd / mean if mean else np.nan
        if max_order >= 3 and out.get('sd', 0) > 0:
            m3 = raw[3] / tot - 3 * mean * (raw[2] / tot) + 2 * mean ** 3
            out['skew'] = m3 / out['sd'] ** 3
        return pd.Series(out, name=getattr(function, '__name__', 'Z'))

    # ------------------------------------------------------------------
    # visualization (plan-bv §7)
    # ------------------------------------------------------------------
    @property
    def pyramid(self):
        """The sum/max/min decimation pyramid (a lazy zarr group), or ``None``.

        Built during pass 3 of the update (``pyramid.zarr`` beside the
        density); ``None`` if the store predates it or was cleaned. Levels
        and channel order are in ``pyramid.attrs``.
        """
        if getattr(self, '_pyramid', None) is None:
            path = os.path.join(self.store_dir, 'pyramid.zarr')
            if not os.path.exists(path):
                return None
            from ._aggregate_compute_massive import _require_zarr
            zarr = _require_zarr()
            self._pyramid = zarr.open_group(path, mode='r')
        return self._pyramid

    def plot(self, window=None, log=True, contours=False, exceedance=False,
             pixels=800, **kwargs):
        """Exploration-grade exhibit of the massive joint (plan-bv §7.2).

        Constant-cost at any zoom: only the pyramid level matching the pixel
        budget (or raw tiles at extreme zoom) is read -- never the full grid.
        Log-scale color by default (mass spans many decades; ``log=False``
        opts out), sum-channel body with a max-channel luminance boost so
        sub-pixel ridges and atoms glow, axis atom strips + origin badge,
        exact marginal panels. Delegates to
        :func:`aggregate.plots.plot_bivariate_massive`.

        Parameters
        ----------
        window : ((x0, x1), (y0, y1)), optional
            Value-coordinate zoom window; default the full grid.
        log : bool, default True
            ``log10`` color scale.
        contours : bool, default False
            Overlay decade contours of the log-density.
        exceedance : bool, default False
            Overlay joint-exceedance contours ``P(X > x, Y > y)``.
        pixels : int, default 800
            Pixel budget per axis -- picks the pyramid level.
        **kwargs
            Forwarded to the compositor.
        """
        from .plots import plot_bivariate_massive
        return plot_bivariate_massive(self, window=window, log=log,
                                      contours=contours,
                                      exceedance=exceedance, pixels=pixels,
                                      **kwargs)

    def plot_slice(self, x=None, y=None, ax=None, log=False):
        """Plot a conditional slice ``P(Y | X ~ x)`` / ``P(X | Y ~ y)`` (§7.2).

        Delegates to :func:`aggregate.plots.plot_bivariate_massive_slice`;
        the data is one row/column of tiles (:meth:`slice`).
        """
        from .plots import plot_bivariate_massive_slice
        return plot_bivariate_massive_slice(self, x=x, y=y, ax=ax, log=log)

    def explore(self, **kwargs):
        """Interactive pan / zoom explorer in JupyterLab (plan-bv §7.3).

        A holoviews + datashader + bokeh app over the decimation pyramid:
        dynamic re-aggregation on every viewport change, channel toggle
        (sum / max / min), hover readout, linked marginals re-windowed to
        the view, click-to-slice conditionals. Requires the optional
        ``aggregate[viz]`` extra; the matplotlib Tier-1 :meth:`plot` is
        always available. **Needs a live kernel**: display the result in a
        JupyterLab cell -- a saved-to-HTML export is a static snapshot and
        none of the interactivity (including tap-to-slice) runs there.

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`aggregate.plots._bivariate_explore.explore`
            (``pixels``, ``cmap``, ``width``, ``height``).

        Returns
        -------
        holoviews.Layout
            Display it in a notebook cell to launch the app.
        """
        from .plots._bivariate_explore import explore
        return explore(self, **kwargs)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self):
        """Persist the accumulators + metadata beside ``density.zarr``.

        Called automatically at update time; makes the store directory
        self-describing so :meth:`reopen` works in a later session.
        """
        from datetime import datetime
        np.savez(os.path.join(self.store_dir, self._ACC_FILE),
                 xs0=self.xs0, xs1=self.xs1,
                 marg0=self.marg0, marg1=self.marg1,
                 raw_moments=self.raw_moments)
        meta = dict(self.meta)
        meta.setdefault('created', datetime.now().isoformat(timespec='seconds'))
        meta['axis_names'] = list(self.axis_names)
        payload = {'bs0': self.bs0, 'bs1': self.bs1,
                   'total_mass': self.total_mass, 'deficit': self.deficit,
                   'meta': meta}
        with open(os.path.join(self.store_dir, self._META_FILE), 'w',
                  encoding='utf-8') as f:
            json.dump(payload, f, indent=1, default=str)

    @classmethod
    def reopen(cls, store_dir):
        """Reconstruct the container from a saved store directory.

        No FFT re-run: the accumulators and metadata are read back and the
        density opens lazily. Worth a lot after a 30-minute ``(16, 16)``
        update.

        Parameters
        ----------
        store_dir : str
            A directory previously written by ``update(store_dir=...)``.

        Returns
        -------
        MassiveBivariateDistribution
        """
        store_dir = str(store_dir)
        meta_path = os.path.join(store_dir, cls._META_FILE)
        acc_path = os.path.join(store_dir, cls._ACC_FILE)
        if not (os.path.exists(meta_path) and os.path.exists(acc_path)):
            raise FileNotFoundError(
                f'{store_dir} is not a massive bivariate store (missing '
                f'{cls._META_FILE} / {cls._ACC_FILE}).')
        with open(meta_path, encoding='utf-8') as f:
            payload = json.load(f)
        acc = np.load(acc_path)
        meta = payload.get('meta', {})
        if 'axis_names' in meta:
            meta['axis_names'] = tuple(meta['axis_names'])
        return cls(store_dir, acc['xs0'], acc['xs1'],
                   payload['bs0'], payload['bs1'],
                   acc['marg0'], acc['marg1'],
                   payload['total_mass'], payload['deficit'],
                   acc['raw_moments'], meta=meta)

    # ------------------------------------------------------------------
    # reprs
    # ------------------------------------------------------------------
    def _summary(self):
        """Return ``(E0, E1, corr, deficit)`` for the repr builders."""
        m = self.raw_moments
        tot = m[0, 0]
        return m[1, 0] / tot, m[0, 1] / tot, self.corr(), self.deficit

    def __repr__(self):
        e0, e1, rho, deficit = self._summary()
        shape = (len(self.xs0), len(self.xs1))
        return (f'MassiveBivariateDistribution('
                f'name={self.meta.get("name", "")!r}, shape={shape}, '
                f'bs=({self.bs0:g}, {self.bs1:g}), '
                f'E[{self.axis_names[0]}]={e0:,.4g}, '
                f'E[{self.axis_names[1]}]={e1:,.4g}, corr={rho:.4f}, '
                f'store={self.store_dir!r})')

    def _repr_html_(self):
        e0, e1, rho, deficit = self._summary()
        names = self.axis_names
        rows = [
            ('name', self.meta.get('name', '')),
            ('grid shape', f'{len(self.xs0)} &times; {len(self.xs1)}'),
            ('bucket', f'{self.bs0:g}, {self.bs1:g}'),
            (f'E[{names[0]}]', f'{e0:,.6g}'),
            (f'E[{names[1]}]', f'{e1:,.6g}'),
            ('correlation', f'{rho:.6f}'),
            ('tail deficit', f'{deficit:.2e}'),
            ('store', self.store_dir),
        ]
        body = ''.join(
            f'<tr><th style="text-align:left">{k}</th><td>{v}</td></tr>'
            for k, v in rows)
        return (f'<table class="aggregate bivariate">'
                f'<caption>MassiveBivariateDistribution (disk-backed)</caption>'
                f'{body}</table>')
