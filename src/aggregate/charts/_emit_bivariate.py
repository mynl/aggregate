"""Bivariate chart emitters: the joint density surface.

``chart_joint_surface`` pulls the joint matrix off a
:class:`~aggregate.bivariate.BivariateAggregate`, windows it on the fine
lattice, block-sums the crop to a display grid, and returns a
:class:`~aggregate.charts.ir.ChartDoc` with one 'surface' panel. Reduction is
meaning, so it happens here, mass-preservingly, never renderer-side.

**The order of operations is the whole design.** Measure the marginal
quantiles on the fine lattice, crop, *then* choose the block factor from the
cropped extent, then reduce. Cropping a grid that was already reduced cannot
recover the resolution the reduction averaged away: on a real Lomax axis the
same ``q(1e-4)`` window is 232 fine cells wide if it is taken first and 8
display cells wide if it is taken last, and those 8 start at 508 on a
distribution supported from 0. A factor of 29 in resolution and a support in
the wrong place, from one swap of two steps.

Everything derived from the grid, and everything the consumer will derive
from it, is a statement about the whole distribution rather than about the
box it is drawn in. That is why :attr:`SurfaceData.marginals` comes off the
object and :attr:`SurfaceData.moments` comes off the fine lattice: a
conditional mean read off a windowed joint is wrong by several percent, and
wrong with a sign that flips as the cut moves, which bends the shape of the
very curve the picture exists to show.

Pure numpy; no matplotlib (the plots boundary), no ECharts vocabulary.
"""

import numpy as np

from .._grid_distribution import GridDistribution
from ..bivariate import BivariateAggregate
from . import register_chart, _emitter_base
from .ir import (ChartAxis, ChartDoc, ChartSeries, Panel, SurfaceData,
                 complete_tex, encode_z_block)

__all__ = ['chart_joint_surface']

#: Default target cells per axis after reduction, migrated from the app's
#: ``CELLS = 128`` (surface.js:27): 128 x 128 is 16,384 vertices, WebGL-cheap
#: and finer than the eye resolves on a panel, while the block *sum* (never a
#: sample) preserves every gram of tail mass. A **target**, not a promise:
#: the reduction is by powers of two, so the realized count is what the
#: blocking could reach, and the document reports it.
DEFAULT_DETAIL = 128

#: Default window depth: keep ``q(1e-4)`` to ``q(1 - 1e-4)`` of each
#: marginal. Measured across four real joints this keeps 99.96% to 99.998% of
#: the mass, and drops the empty decades a heavy tail spends the rest of the
#: axis on. ``0`` means the whole grid.
DEFAULT_WINDOW = 4.0

#: Floor on the realized cells per axis. A Lomax on a lattice wide enough to
#: hold its tail carries 99.99% of its mass in the first bucket, so both ends
#: of a deep window land in that bucket and leave a grid one cell across,
#: with no spacing for a consumer to interpolate on.
MIN_CELLS = 8

#: How close to a reachable zero the low edge must land before it snaps
#: there, as a fraction of the window's width. A loss supported from the
#: origin should be drawn from the origin, and a window opening 3% above it
#: invents support the distribution does not have. Never applied to a signed
#: axis, whose lower bound is genuinely negative.
ZERO_SNAP_FRACTION = 0.05

#: How ``z`` is carried. ``'json'`` means no encoded block at all: the plain
#: ``x`` / ``y`` / ``z`` arrays are the payload, which is the behavior before
#: the block existed, kept for one release as the fallback.
ENCODINGS = ('f32b64', 'f64b64', 'u16log12b64', 'json')


def _window_indices(xs, marginal, bs, depth):
    """Half-open fine-lattice index window at depth ``10 ** -depth``.

    Parameters
    ----------
    xs : ndarray
        The fine lattice for this axis.
    marginal : ndarray
        The exact marginal mass on it.
    bs : float
        The fine bucket size (the lattice step).
    depth : float
        Window depth; ``<= 0`` means the whole axis.

    Returns
    -------
    (lo, hi) : tuple of int
        ``xs[lo:hi]`` is the window, ``hi`` exclusive and at least ``lo + 1``.

    Notes
    -----
    The quantiles come off a :class:`~aggregate._grid_distribution.
    GridDistribution` rather than a hand-rolled ``searchsorted``, which is the
    house rule for every quantile in the library: the lower quantile is
    ``inf{x : F(x) >= p}`` and getting the tie handling right on an atomic
    grid is exactly the kind of thing one implementation should own.

    Converting the value back to an index is exact lattice arithmetic and not
    a second search, because the quantile of a grid distribution is by
    construction one of its own atoms.
    """
    n = len(xs)
    if depth is None or depth <= 0:
        return 0, n
    p = 10.0 ** (-float(depth))
    gd = GridDistribution(xs, marginal, bs=bs)
    lo_v, hi_v = float(gd.q(p)), float(gd.q(1.0 - p))
    lo = int(np.clip(round((lo_v - xs[0]) / bs), 0, n - 1))
    hi = int(np.clip(round((hi_v - xs[0]) / bs), 0, n - 1)) + 1
    return (lo, hi) if hi > lo else (lo, min(lo + 1, n))


def _snap_to_zero(xs, lo, hi):
    """Pull the low edge to the origin when it is nearly there.

    Only on an axis whose lattice starts at zero: that is the one case where
    zero is both reachable (it is a grid point) and the distribution's real
    floor. An axis measured up from a positive lower bound has no zero to
    reach, and a signed axis has a genuinely negative bound, which is what
    ``IndepSigned`` is in the test set and why the rule is stated as a
    property of the lattice rather than as a guess from the data.
    """
    if xs[0] != 0.0 or lo == 0:
        return lo
    width = xs[hi - 1] - xs[lo]
    return 0 if width > 0 and xs[lo] < ZERO_SNAP_FRACTION * width else lo


def _block_factor(span, detail):
    """Smallest power of two ``k`` with ``ceil(span / k) <= detail``.

    A power of two per axis, chosen independently, so the blocks divide the
    axis exactly. An uneven last block makes the spacing non-uniform, which
    every interpolation downstream relies on, and puts the wide cell at the
    end of the axis, which is exactly where the tail is.

    ``detail`` is honored as a ceiling rather than as a target to straddle,
    because it is also the knob a deployment caps to bound a payload; a
    reduction that overshot it upward would make the cap mean nothing.
    """
    k = 1
    while -(-span // k) > detail:
        k <<= 1
    return k


def _align(lo, hi, k, n):
    """Widen ``[lo, hi)`` outward to whole blocks of ``k`` inside ``[0, n)``."""
    top = n - n % k
    lo = max(0, lo - lo % k)
    hi = min(top, hi + (-hi) % k)
    if hi <= lo:
        lo, hi = max(0, top - k), max(k, top)
    return lo, hi


def _axis_plan(xs, marginal, bs, depth, detail):
    """Window and block factor for one axis, in the order that matters.

    Returns
    -------
    (lo, hi, k) : tuple of int
        The fine index window, aligned to whole blocks, and the block factor.

    Notes
    -----
    Aligning the crop can add a block, which can put the realized count one
    over ``detail``; the loop takes the next power of two rather than
    reporting a count the caller capped. The floor then runs the other way,
    coarsening less until at least :data:`MIN_CELLS` cells exist, and widening
    the crop only when ``k`` is already 1 and there is nothing left to undo.

    **The floor outranks the ceiling**, and the two can collide. Counts
    reachable by a power-of-two blocking of a 528-cell window go 33, 17, 9,
    5, so a ``detail`` of 8 has nothing to land on: the choice is 9 cells or
    5, and 5 is a grid with no spacing to interpolate on. The floor takes it,
    and the overshoot is bounded by ``2 * MIN_CELLS - 1`` cells whatever
    ``detail`` was, so it cannot reach a payload budget. It can only happen
    at a target within a factor of two of the floor; every ordinary target
    is honored exactly. The document reports what was realized, through
    ``nx``, ``ny`` and ``k``, and the consumer displays that rather than what
    it asked for.
    """
    n = len(xs)
    lo, hi = _window_indices(xs, marginal, bs, depth)
    lo = _snap_to_zero(xs, lo, hi)
    k = _block_factor(hi - lo, detail)
    lo, hi = _align(lo, hi, k, n)
    while (hi - lo) // k > detail:
        k <<= 1
        lo, hi = _align(lo, hi, k, n)
    while (hi - lo) // k < MIN_CELLS and k > 1:
        k >>= 1
        lo, hi = _align(lo, hi, k, n)
    if (hi - lo) // k < MIN_CELLS:
        want = min(MIN_CELLS * k, n - n % k)
        lo = max(0, min(lo, n - want))
        hi = lo + want
    return lo, hi, k


def _reduce(values, k, axis=0):
    """Mass-preserving block sum by a factor that divides the axis exactly."""
    if k == 1:
        return values
    shape = list(values.shape)
    shape[axis:axis + 1] = [shape[axis] // k, k]
    return values.reshape(shape).sum(axis=axis + 1)


chart_joint_surface = _emitter_base('joint_surface')


@chart_joint_surface.register(BivariateAggregate)
def _joint_surface(bv, window=DEFAULT_WINDOW, detail=DEFAULT_DETAIL,
                   encoding='f32b64'):
    """Emit the joint density surface for a bivariate aggregate.

    Parameters
    ----------
    bv : BivariateAggregate
        An updated bivariate (in-memory joint density; the disk-backed
        massive pyramid stays bespoke, per the plan's scope).
    window : float, default :data:`DEFAULT_WINDOW`
        Keep ``q(10 ** -window)`` to ``q(1 - 10 ** -window)`` of each
        marginal, measured on the fine lattice before the reduction. ``0``
        keeps the whole grid.
    detail : int, default :data:`DEFAULT_DETAIL`
        Target cells per axis after the reduction, honored as a ceiling
        except that :data:`MIN_CELLS` outranks it (see :func:`_axis_plan`;
        the overshoot is at most ``2 * MIN_CELLS - 1`` cells and only at a
        target near the floor).
    encoding : str, default 'f32b64'
        One of :data:`ENCODINGS`; how ``z`` is carried in the encoded block.
        ``'json'`` emits no block, leaving the plain arrays as the payload.

    Returns
    -------
    ChartDoc
        One 'surface' panel: axes labeled with the resolved component
        labels, and a z axis read linearly by default that declares
        ``scales=('linear', 'log')``, because a joint density spans four or
        five orders of magnitude and the log reading is where tail
        dependence lives.

    Notes
    -----
    **Values are display-cell masses, not ordinates**, everywhere in the
    document: ``z``, the encoded block, and both marginals. Mass is exact, is
    what the block reduction preserves, and lets a consumer form either
    reading; a density is one division by ``dx * dy``, done once, at the
    consumer. Mixing the two is not a cosmetic error. A prototype that
    integrated a marginal as if it were a density while normalizing a
    conditional into a real one had them disagree by a factor of 1024, so the
    conditional lay flat on the floor at every cut while the marginal stood
    up beside it.

    **The coordinate is the block's first fine coordinate**, its low edge,
    which is the convention the fine lattice is already in, and the document
    says so through ``edge``. The previous convention filed a block covering
    ``[a, a + k * bs)`` under ``a + (k - 1) * bs``, its *last* fine
    coordinate, which reported a distribution supported on ``[0, inf)`` as
    starting at ``(k - 1) * bs`` and biased every mean taken against those
    coordinates up by close to a whole display bucket, 0.95 of one on the
    worst of the four test surfaces.

    A residual bias in the other direction is intrinsic and is not a defect:
    a display cell holds ``k`` atoms and labeling it with any single
    coordinate loses their spread, so a mean taken against the display
    lattice sits up to one display bucket low. That is what ``moments`` is
    for. The exact means travel with the document, off the fine lattice, and
    a consumer that needs a mean reads them rather than integrating the
    picture.
    """
    density = getattr(bv, 'density', None)
    if density is None:
        raise ValueError(
            'chart_joint_surface needs the in-memory joint density: '
            'update() first (a massive, disk-backed bivariate is out of '
            'scope for this chart).')
    if encoding not in ENCODINGS:
        raise ValueError(f'unknown encoding {encoding!r}; '
                         f'expected one of {ENCODINGS}')
    window = 0.0 if window is None else float(window)
    detail = int(detail)
    if detail < MIN_CELLS:
        raise ValueError(f'detail must be at least {MIN_CELLS}, got {detail}')

    density = np.nan_to_num(np.asarray(density, dtype=float),
                            nan=0.0, posinf=0.0, neginf=0.0)
    xs = np.asarray(bv.axis_xs[0], dtype=float)
    ys = np.asarray(bv.axis_xs[1], dtype=float)
    bs_x, bs_y = float(bv.bs[0]), float(bv.bs[1])
    marg_x, marg_y = (np.asarray(m, dtype=float) for m in bv.marginals)

    lo_x, hi_x, kx = _axis_plan(xs, marg_x, bs_x, window, detail)
    lo_y, hi_y, ky = _axis_plan(ys, marg_y, bs_y, window, detail)

    z = _reduce(_reduce(density[lo_x:hi_x, lo_y:hi_y], kx, axis=0), ky, axis=1)
    display_x = xs[lo_x:hi_x:kx]
    display_y = ys[lo_y:hi_y:ky]
    dx, dy = bs_x * kx, bs_y * ky
    nx, ny = len(display_x), len(display_y)

    total = float(density.sum())
    kept = float(z.sum() / total) if total > 0 else 0.0
    # The marginals are the object's own, reduced over this axis' crop alone:
    # the row sums of the cropped joint would be the marginals of a truncated
    # distribution, which is a different curve and the one a reader would be
    # misled by.
    display_marg_x = _reduce(marg_x[lo_x:hi_x], kx)
    display_marg_y = _reduce(marg_y[lo_y:hi_y], ky)

    names = list(bv.unit_names)
    if bv.use_labels:
        renamer = bv.renamer
        labels = [renamer.get(n, n) for n in names]
    else:
        labels = names

    # SurfaceData is row-major over y (z[r][c] at (x[c], y[r])), the reduced
    # matrix is [x-block, y-block]: transpose once, here.
    zt = np.ascontiguousarray(z.T)
    surface = SurfaceData(
        x=tuple(float(v) for v in display_x),
        y=tuple(float(v) for v in display_y),
        z=tuple(tuple(float(v) for v in row) for row in zt),
        x0=float(display_x[0]), dx=dx, nx=nx,
        y0=float(display_y[0]), dy=dy, ny=ny,
        edge='left',
        bs=(bs_x, bs_y),
        k=(kx, ky),
        window={
            'p': float(window),
            'x': (float(display_x[0]), float(display_x[0] + nx * dx)),
            'y': (float(display_y[0]), float(display_y[0] + ny * dy)),
            'kept': kept,
        },
        marginals={
            'x': tuple(float(v) for v in display_marg_x),
            'y': tuple(float(v) for v in display_marg_y),
        },
        moments={'mean': (float(xs @ marg_x), float(ys @ marg_y))},
        # A deficit that is not a number is not a fact about the grid, it is
        # an object that never measured one; omit the field rather than write
        # a NaN the canonical JSON would refuse anyway.
        deficit=(float(bv.deficit) if np.isfinite(bv.deficit) else None),
        z_block=(None if encoding == 'json'
                 else encode_z_block(zt, dtype=encoding)),
    )
    return complete_tex(ChartDoc(
        name='joint_surface',
        title=f'Joint density: {labels[0]} vs {labels[1]}',
        axes=(
            ChartAxis(id='x0', label=str(labels[0]), unit='currency'),
            ChartAxis(id='x1', label=str(labels[1]), unit='currency'),
            ChartAxis(id='z', label='density', unit='density',
                      scales=('linear', 'log')),
        ),
        panels=(
            Panel(id='joint', kind='surface', x_axis='x0', y_axis='x1',
                  z_axis='z'),
        ),
        series=(
            ChartSeries(name='joint density', role='joint',
                        panel_id='joint', surface=surface),
        ),
    ))


register_chart(
    'joint_surface', chart_joint_surface,
    predicate=lambda bv: getattr(bv, 'density', None) is not None,
    primary=BivariateAggregate)
