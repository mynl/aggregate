"""Bivariate chart emitters: the joint density surface, and the kappa band.

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

``chart_kappa`` draws the other half of what a joint knows: the conditional
cession given the gross outcome, as a curve **with a band**. A mean is the
wrong summary for the question a cedent actually asks about an occurrence
program, which is not "what do I cede on average when the year comes in at
500" but "having come in at 500, how much of that could I have been ceding,
and how much am I actually ceding". The gross outcome does not determine the
cession, and the band is where that story is.

Pure numpy; no matplotlib (the plots boundary), no ECharts vocabulary.
"""

import numpy as np

from .._aggregate import Aggregate
from .._grid_distribution import GridDistribution
from ..bivariate import BivariateAggregate
from . import register_chart, _emitter_base
from ._payload import lattice_payload
from .ir import (ChartAxis, ChartDoc, ChartSeries, Panel, SurfaceData,
                 complete_tex, encode_z_block)

__all__ = ['chart_joint_surface', 'chart_kappa']

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

    **The coordinate is the block's representative point**, the mean of the
    fine coordinates it covers: a block holding atoms at ``a, a + bs, ...,
    a + (k - 1) * bs`` is filed under ``a + (k - 1) * bs / 2``, and the
    document says so through ``edge='mid'``. Two conventions preceded it,
    both wrong in the same direction and by different amounts. The first
    filed the block under ``a + (k - 1) * bs``, its *last* fine coordinate,
    which reported a distribution supported on ``[0, inf)`` as starting at
    ``(k - 1) * bs`` and biased every mean taken against those coordinates
    up by close to a whole display bucket, 0.95 of one on the worst of the
    four test surfaces. The second filed it under ``a``, its low edge, which
    fixed the support but left the mean a half bucket low and invited a
    consumer to recover the middle by adding ``dx / 2`` when the truth wants
    ``(k - 1) * bs / 2``.

    The residual is then second order and of **either** sign: it is the
    deviation of the within-block mass from uniform, not a convention, so it
    vanishes as the density flattens across a block rather than shrinking
    toward a fixed side. Measured on ``Indep``'s y axis at the default
    window, where a block is two atoms, it is -0.015 display buckets against
    -0.265 for the low edge and +0.235 for a cell midpoint.

    **What the choice buys is the bound, not that measurement.** A block's
    conditional mean lies somewhere in ``[a, a + (k - 1) * bs]``, so filing
    the block at the middle of that span holds the error under ``dx / 2``
    whatever the density does inside it, and no other single coordinate
    does: the low edge is one-sided and its bound is a whole bucket, reached
    by a block whose mass sits at the far end. Being a bound rather than a
    tendency, it does not promise to win every case. Reduce the whole of
    ``Indep``'s y axis 128 to 1 and a Lomax puts each block's mass hard
    against its low end, where this convention reads 0.45 buckets high and
    the low edge, flattered by the shape, reads 0.05 low. Both are inside
    the bound only one of them has. Either way this is what ``moments`` is
    for: the exact means travel with the document, off the fine lattice, and
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
    # The representative point: a block covering k atoms is filed under their
    # mean, not under either end of the span they sit in.
    display_x = xs[lo_x:hi_x:kx] + (kx - 1) * bs_x / 2
    display_y = ys[lo_y:hi_y:ky] + (ky - 1) * bs_y / 2
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
        edge='mid',
        bs=(bs_x, bs_y),
        k=(kx, ky),
        window={
            'p': float(window),
            'x': (float(display_x[0] - dx / 2),
                  float(display_x[0] + (nx - 1) * dx + dx / 2)),
            'y': (float(display_y[0] - dy / 2),
                  float(display_y[0] + (ny - 1) * dy + dy / 2)),
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


# --- the kappa band ---------------------------------------------------------

#: The probability window on the conditioning marginal that the curves are
#: drawn over. A window, not a mass floor: a raw threshold like ``p > 1e-4``
#: means different things at different bucket sizes (on a fine measured joint
#: the largest row mass is 8.9e-04, so that threshold discards most of the
#: picture, while on a coarse one it keeps nearly all of it), and a CDF range
#: means the same thing on every grid.
KAPPA_CDF_RANGE = (1e-3, 0.999)

#: The band's two edges. A **percentile** band: nothing here is an estimate
#: with sampling error, the joint is the law, so the legend says percentile
#: and never "confidence interval".
KAPPA_LEVELS = (0.01, 0.99)

chart_kappa = _emitter_base('kappa')


def _kappa_ceiling(agg, g):
    """The most a single occurrence layer could cede at each gross total.

    Parameters
    ----------
    agg : Aggregate
        The reinsured aggregate behind the joint.
    g : ndarray
        Gross outcomes.

    Returns
    -------
    ndarray or None
        The ceiling at each ``g``, or ``None`` when the program is not a
        single finite layer.

    Notes
    -----
    For one layer ``limit xs attach`` at placement ``share``, the largest
    cession achievable with a gross total of ``g`` comes from splitting ``g``
    into claims of exactly ``attach + limit``, each ceding the full limit, plus
    whatever a remainder above the attachment cedes. So the ceiling is a comb
    with teeth every ``attach + limit`` and a maximum share of
    ``share * limit / (attach + limit)`` at each tooth.

    Drawn next to the upper band edge it says how much of the theoretically
    available cession the program actually delivers, and the answer is
    typically "nowhere near": getting several claims to land exactly at the top
    of the layer is a lot to ask. Measured on the notes' demo, the ceiling
    peaks at 0.5 while the realized 99th percentile share peaks at 0.400.

    Only for a single layer. A tower has no such simple envelope, since the
    optimal split of ``g`` across its layers is a different problem at every
    ``g``, so this answers ``None`` and the panel simply has one fewer curve.
    """
    program = getattr(agg, 'occ_reins', None)
    if not program or len(program) != 1:
        return None
    share, limit, attach = (float(v) for v in program[0])
    if not np.isfinite(limit) or limit <= 0:
        return None
    tooth = attach + limit
    g = np.asarray(g, dtype=float)
    whole = np.floor(g / tooth)
    remainder = g - whole * tooth
    return share * (whole * limit + np.maximum(remainder - attach, 0.0))


def _kappa_frame(bv, levels, cdf_range):
    """The band frame plus the axis roles, for a netceded joint.

    Refuses a joint with no gross axis in the same words
    :meth:`~aggregate.bivariate.BivariateAggregate.natural_allocation` uses:
    the curves condition on the gross outcome, and a ``(net, ceded)`` joint has
    no gross axis to condition on.
    """
    if bv.mode != 'netceded':
        raise ValueError(
            f'chart kappa reads a netceded joint (one aggregate split into two '
            f'of gross / ceded / net); {bv.name} is in {bv.mode!r} mode.')
    views = tuple(bv._views)
    if 'gross' not in views:
        raise ValueError(
            f'chart kappa conditions on the gross outcome, and {bv.name} '
            f"carries views {views}. Rebuild with views=('gross', 'ceded') "
            f"or ('gross', 'net').")
    axis = views.index('gross')
    return bv.exeqa_df(axis=axis, levels=levels, cdf_range=cdf_range), axis


@chart_kappa.register(BivariateAggregate)
def _kappa_band(bv, levels=KAPPA_LEVELS, cdf_range=KAPPA_CDF_RANGE,
                ceiling=True):
    """Emit the two-panel kappa band chart for a netceded joint.

    Parameters
    ----------
    bv : BivariateAggregate
        An updated ``netceded`` joint carrying a gross axis. A disk-backed
        (massive) joint is accepted: the band is a row-wise fold, which is the
        whole point of it.
    levels : (float, float), default :data:`KAPPA_LEVELS`
        The band's lower and upper probability.
    cdf_range : (float, float), default :data:`KAPPA_CDF_RANGE`
        The probability window on the gross marginal the curves are drawn over.
    ceiling : bool, default True
        Draw the deterministic ceiling on the share panel where the program is
        a single layer (:func:`_kappa_ceiling`).

    Returns
    -------
    ChartDoc
        Two 'xy' panels over one shared gross axis. **cession** carries the two
        kappa curves, their bands and the identity; **share** carries the same
        reading divided by the outcome, with the ceiling.

    Notes
    -----
    **The left panel is the conservation statement.** ``kappa_N = g -
    kappa_C`` pointwise, so the net band is the ceded band reflected in the
    identity, ``[g - q99, g - q01]``: two shaded regions of equal width, one
    hugging zero and one hugging the diagonal. Equal aspect is semantic there,
    as on the portfolio kappa panel, because both axes are losses and the
    reading is each curve's slope against 45 degrees.

    **The bands are ``y2`` series**, so a band *is* the region between two
    edges rather than two curves a reader has to associate.

    **No smoothing.** A rolling mean over a few tens of buckets would tidy the
    edges, which step by whole buckets because they are quantiles of a lattice
    law, but the visible structure below the mean is real: it is the comb of
    the ceiling, the k claims landing at the top of the layer. Smoothing it
    away removes the mechanism rather than noise. It is also a display choice,
    and this document carries meaning rather than drawing instructions.
    """
    levels = tuple(sorted(float(v) for v in levels))
    if len(levels) != 2:
        raise ValueError(
            f'the kappa band has two edges, a lower and an upper level; got '
            f'{levels!r}.')
    df, axis = _kappa_frame(bv, levels, cdf_range)
    other_name = bv.unit_names[1 - axis]
    other_view = bv._views[1 - axis]

    g = df.index.to_numpy(dtype=float)
    kappa = df[f'exeqa_{other_name}'].to_numpy(dtype=float)
    lo = df[bv._quantile_column(levels[0], other_name)].to_numpy(dtype=float)
    hi = df[bv._quantile_column(levels[1], other_name)].to_numpy(dtype=float)

    # The named axis carries one of ceded / net and the other view is g less
    # it, exactly as the allocation reads the third view: a definition on the
    # index rather than a second measurement, which is what makes the two
    # bands mirror images and the two curves sum to the diagonal.
    third_view = 'net' if other_view == 'ceded' else 'ceded'
    third = g - kappa
    third_lo, third_hi = g - hi, g - lo

    step = float(bv.bs[axis])
    x = lattice_payload(g, step)
    band_label = (f'{levels[0]:.0%} to {levels[1]:.0%} percentile band'
                  .replace('%%', '%'))
    series = [
        ChartSeries(name=f'E[{other_view} | gross]', role=other_view,
                    panel_id='cession', support='continuous',
                    y=tuple(float(v) for v in kappa), **x),
        ChartSeries(name=f'{other_view} {band_label}', role=other_view,
                    panel_id='cession', support='continuous',
                    y=tuple(float(v) for v in lo),
                    y2=tuple(float(v) for v in hi), **x),
        ChartSeries(name=f'E[{third_view} | gross]', role=third_view,
                    panel_id='cession', support='continuous',
                    y=tuple(float(v) for v in third), **x),
        ChartSeries(name=f'{third_view} {band_label}', role=third_view,
                    panel_id='cession', support='continuous',
                    y=tuple(float(v) for v in third_lo),
                    y2=tuple(float(v) for v in third_hi), **x),
        ChartSeries(name='gross', role='identity', panel_id='cession',
                    support='continuous',
                    y=tuple(float(v) for v in g), **x),
    ]

    # Shares. Quantiles commute with the monotone map c -> c / g at fixed g,
    # so the share band is the value band divided by the index: no second pass
    # over the joint, and no approximation either.
    with np.errstate(divide='ignore', invalid='ignore'):
        share = np.where(g > 0, kappa / g, np.nan)
        share_lo = np.where(g > 0, lo / g, np.nan)
        share_hi = np.where(g > 0, hi / g, np.nan)
    series += [
        ChartSeries(name=f'{other_view} share', role=other_view,
                    panel_id='share', support='continuous',
                    y=tuple(float(v) for v in share), **x),
        ChartSeries(name=f'{other_view} {band_label}', role=other_view,
                    panel_id='share', support='continuous',
                    y=tuple(float(v) for v in share_lo),
                    y2=tuple(float(v) for v in share_hi), **x),
    ]
    top = _kappa_ceiling(bv._nc_agg, g) if ceiling else None
    if top is not None:
        with np.errstate(divide='ignore', invalid='ignore'):
            top_share = np.where(g > 0, top / g, np.nan)
        series.append(ChartSeries(
            name='most the program could cede', role='ceiling',
            panel_id='share', support='continuous',
            y=tuple(float(v) for v in top_share), **x))

    window = (float(g[0]), float(g[-1]))
    return complete_tex(ChartDoc(
        name='kappa',
        title=f'Conditional cession: {bv.label}',
        axes=(
            ChartAxis(id='outcome', label='Gross outcome', unit='currency',
                      scales=('linear', 'log'), suggested_range=window),
            # Both axes of the left panel are losses on one scale, which is
            # what makes the identity line readable as 45 degrees.
            ChartAxis(id='cession', label='Conditional cession',
                      unit='currency', scales=('linear', 'log'),
                      suggested_range=(0.0, float(g[-1]))),
            ChartAxis(id='share', label='Share of the outcome ceded',
                      unit='ratio', suggested_range=(0.0, 1.0)),
        ),
        panels=(
            Panel(id='cession', kind='xy', x_axis='outcome',
                  y_axis='cession', aspect='equal',
                  title='Cession given the gross outcome'),
            Panel(id='share', kind='xy', x_axis='outcome', y_axis='share',
                  title='The same reading as a share'),
        ),
        series=tuple(series),
        # Lists, not tuples: meta travels as JSON and comes back as JSON, so
        # a tuple here would not survive load_chart_doc hash for hash.
        meta={'levels': [float(v) for v in levels],
              'cdf_range': [float(v) for v in cdf_range],
              'band': 'percentile'},
    ))


@chart_kappa.register(Aggregate)
def _kappa_from_aggregate(agg, **options):
    """The band chart for an occurrence program, off the joint it implies.

    A thin delegate. The curves are a property of the cession rather than of a
    calibration, so this is a chart on the built object and not a fourth
    pricing call; what it needs is the joint, and
    :meth:`~aggregate.distributions.Aggregate.occ_joint` holds one, so drawing
    after an allocation costs a lookup rather than a second 2-D FFT.
    """
    return _kappa_band(agg.occ_joint(views=('gross', 'ceded')), **options)


def _kappa_available(obj):
    """Availability for the three sources the ``kappa`` name serves.

    Duck-typed rather than by ``isinstance``, so this module stays free of an
    import of every class the chart draws for, and the type dispatch on the
    emitter is what actually decides which of the three a caller reaches.

    Order matters only in that each test is asked of the shape that can answer
    it: a book has units, a joint has a mode, and an aggregate has a program.
    """
    if getattr(obj, 'agg_list', None) is not None:
        return getattr(obj, 'density_df', None) is not None
    if getattr(obj, 'mode', None) == 'netceded':
        # accepts a massive joint, unlike joint_surface: the band is a
        # row-wise fold and surviving the disk route is the point of it
        return ('gross' in getattr(obj, '_views', ())
                and (obj.density is not None
                     or getattr(obj, '_massive', None) is not None))
    return (getattr(obj, 'occ_reins', None) is not None
            and getattr(obj, 'agg_density', None) is not None)


register_chart('kappa', chart_kappa, predicate=_kappa_available)
