"""Layer 1/2 for the massive (disk-backed) bivariate exhibit (plan-bv §7.2).

The constant-cost invariant: every render reads only the pyramid level whose
resolution matches the pixel budget (and, at extreme zoom, raw
``density.zarr`` tiles) -- never the full grid. Pan / zoom from the whole
distribution down to individual buckets at the same speed.

Layer 1 content workers render one panel each into a provided ``Axes``; the
Layer-2 compositor :func:`plot_bivariate_massive` builds the mosaic:

.. code-block:: text

    .  T  .      T = top marginal (axis 0, exact accumulator)
    A  H  M      H = heatmap (sum channel, max-channel glow), A = x=0 atom
    .  B  .      strip, M = right marginal (axis 1), B = y=0 atom strip

plus an origin badge ``P(0, 0)`` when both axes carry an atom at 0. The
main heatmap is ``log10`` density **by default** (mass spans many decades;
``log=False`` opts out), percentile-clipped color range, zeros masked to the
background. The **dual encoding** is the wow feature: the sum channel is the
color field; where the max channel shows sub-pixel concentration
(``max >> block mean``) the pixel gets a luminance boost, so one-cell-wide
ridges (the netceded comonotone filament), atoms and attachment kinks stay
lit at any zoom instead of washing out of a plain block-sum.
"""

import numpy as np

from ._style import make_mosaic, FIG_W, FIG_H, PLOT_FACE_COLOR

# glow tuning: boost starts when a block's max exceeds `_GLOW_ONSET` x the
# block mean and saturates `_GLOW_DECADES` decades above that.
_GLOW_ONSET = 10.0
_GLOW_DECADES = 3.0
_GLOW_STRENGTH = 0.85


# ----------------------------------------------------------------------
# data layer: window -> pyramid level -> channel tiles
# ----------------------------------------------------------------------

def _support_box(bd, floor=1e-13):
    """Default view window: the realized support of the exact marginals.

    The measured grid covers the ``10**-window_nines`` tails, which can leave
    half the canvas visually empty; the default exhibit trims to where the
    marginals actually carry mass. Pass an explicit ``window`` for anything
    else (including the full grid).
    """
    nz0 = np.flatnonzero(bd.marg0 > floor)
    nz1 = np.flatnonzero(bd.marg1 > floor)
    if len(nz0) == 0 or len(nz1) == 0:
        return None
    return ((float(bd.xs0[nz0[0]]), float(bd.xs0[nz0[-1]])),
            (float(bd.xs1[nz1[0]]), float(bd.xs1[nz1[-1]])))


def _resolve_window(bd, window):
    """Clip a value-coordinate ``((x0, x1), (y0, y1))`` window (or None ->
    the marginals' support box) to base-grid index ranges ``(i0, i1, j0, j1)``
    (half-open)."""
    n0, n1 = len(bd.xs0), len(bd.xs1)
    if window is None:
        window = _support_box(bd)
    if window is None:
        return 0, n0, 0, n1
    (x0, x1), (y0, y1) = window
    i0 = int(np.clip(np.floor((x0 - bd.xs0[0]) / bd.bs0), 0, n0 - 1))
    i1 = int(np.clip(np.ceil((x1 - bd.xs0[0]) / bd.bs0) + 1, i0 + 1, n0))
    j0 = int(np.clip(np.floor((y0 - bd.xs1[0]) / bd.bs1), 0, n1 - 1))
    j1 = int(np.clip(np.ceil((y1 - bd.xs1[0]) / bd.bs1) + 1, j0 + 1, n1))
    return i0, i1, j0, j1


def _select_level(bd, i0, i1, j0, j1, pixels):
    """Pick the pyramid level for a window at a pixel budget.

    Level ``k`` shows ``2^k x 2^k`` base cells per pyramid cell. The rule is
    **area-based** -- smallest ``k`` with ``window cells / 4^k`` within the
    ``pixels**2`` render budget -- rather than per-axis, so a strongly
    anisotropic grid (e.g. the netceded 16384 x 64) is not decimated to a
    couple of cells on its short axis just because its long axis is huge:
    the read stays bounded near ``pixels**2`` cells either way. Level 0
    reads ``density.zarr`` directly.
    """
    cells = (i1 - i0) * (j1 - j0)
    budget = max(int(pixels), 1) ** 2
    k = max(0, int(np.ceil(0.5 * np.log2(cells / budget))) if cells > budget
            else 0)
    pyr = bd.pyramid
    kmax = len(pyr.attrs.get('levels', [])) if pyr is not None else 0
    return min(k, kmax)


def _read_tiles(bd, i0, i1, j0, j1, level):
    """Read the (sum, max, min) channel tiles of a window at a level.

    Returns ``(mass, mx, mn, extent)``: block-mass, block-max and block-min
    arrays (level 0: ``mx = mn = mass``) and the matplotlib ``extent``
    ``(x_lo, x_hi, y_lo, y_hi)`` of the tile in value coordinates.
    """
    s = 1 << level
    # snap the window outward to whole blocks
    i0, i1 = (i0 // s) * s, -(-i1 // s) * s
    j0, j1 = (j0 // s) * s, -(-j1 // s) * s
    if level == 0:
        mass = np.asarray(bd.density[i0:i1, j0:j1], dtype=float)
        mx = mn = mass
    else:
        arr = bd.pyramid[f'L{level}']
        p0, p1, q0, q1 = i0 // s, i1 // s, j0 // s, j1 // s
        mass = np.asarray(arr[0, p0:p1, q0:q1], dtype=float)
        mx = np.asarray(arr[1, p0:p1, q0:q1], dtype=float)
        mn = np.asarray(arr[2, p0:p1, q0:q1], dtype=float)
    extent = (bd.xs0[0] + i0 * bd.bs0 - bd.bs0 / 2,
              bd.xs0[0] + i1 * bd.bs0 - bd.bs0 / 2,
              bd.xs1[0] + j0 * bd.bs1 - bd.bs1 / 2,
              bd.xs1[0] + j1 * bd.bs1 - bd.bs1 / 2)
    return mass, mx, mn, extent, (i0, i1, j0, j1)


def _axis_zero_index(xs, bs):
    """Index of physical 0 on a grid, or ``None`` if 0 is off-grid."""
    i = int(round((0.0 - xs[0]) / bs))
    if 0 <= i < len(xs) and abs(xs[i]) < bs / 2:
        return i
    return None


# ----------------------------------------------------------------------
# Layer 1: content workers
# ----------------------------------------------------------------------

def _render_heatmap(ax, mass, mx, extent, *, level, log, cmap, clip_pct):
    """The main panel: log10 sum-channel field with the max-channel glow."""
    import matplotlib as mpl

    cells = 4.0 ** level
    pos = mass[mass > 0]
    if pos.size == 0:
        ax.imshow(np.zeros_like(mass).T, origin='lower', extent=extent,
                  aspect='auto')
        return None
    if log:
        field = np.log10(np.where(mass > 0, mass, np.nan))
        lo, hi = np.nanpercentile(field, clip_pct)
    else:
        field = np.where(mass > 0, mass, np.nan)
        lo, hi = np.nanpercentile(field, clip_pct)
    norm = mpl.colors.Normalize(vmin=lo, vmax=hi)
    cm = mpl.colormaps[cmap]
    rgba = cm(norm(field))
    rgba[np.isnan(field)] = mpl.colors.to_rgba(PLOT_FACE_COLOR)
    if level > 0:
        # dual encoding: luminance boost where the block max dwarfs the
        # block mean -- sub-pixel filaments / atoms glow (plan-bv §7.2)
        with np.errstate(divide='ignore', invalid='ignore'):
            conc = np.where(mass > 0, mx * cells / mass, 0.0)
            glow = np.clip((np.log10(np.maximum(conc, 1.0))
                            - np.log10(_GLOW_ONSET)) / _GLOW_DECADES, 0, 1)
        rgba[..., :3] += (1.0 - rgba[..., :3]) * (glow * _GLOW_STRENGTH)[..., None]
    # density indexed [axis0, axis1]; imshow wants [row=y, col=x]
    ax.imshow(np.transpose(rgba, (1, 0, 2)), origin='lower', extent=extent,
              aspect='auto', interpolation='nearest')
    return norm


def _render_marginal(ax, xs, dens, *, vertical=False, color=None):
    """A flanking exact-marginal panel, aligned to the heatmap axis."""
    if vertical:
        ax.fill_betweenx(xs, 0.0, dens, alpha=0.35, color=color)
        ax.plot(dens, xs, lw=0.8, color=color)
        ax.set_xticks([])
    else:
        ax.fill_between(xs, 0.0, dens, alpha=0.35, color=color)
        ax.plot(xs, dens, lw=0.8, color=color)
        ax.set_yticks([])
    ax.margins(0)


# an axis-atom strip is only worth a panel when it carries real mass; the
# copula books' P(A=0) ~ exp(-en) is display noise, the netceded P(C=0) is a
# headline number.
_ATOM_MASS_FLOOR = 1e-6


def _render_atom_strip(ax, xs, mass, *, vertical=False, color='C3'):
    """A 1-D strip of the axis-atom mass (``P(X=0, .)`` / ``P(., Y=0)``).

    Rendered as a filled line (an imshow of near-zero values reads as a
    solid black bar); hidden entirely below :data:`_ATOM_MASS_FLOOR`. The
    strip total is annotated -- probability of zero is a headline number,
    not a color-scale nuisance (plan-bv §7.2).
    """
    total = 0.0 if mass is None else float(np.sum(mass))
    if total < _ATOM_MASS_FLOOR:
        ax.axis('off')
        return
    m = np.asarray(mass, dtype=float)
    if vertical:
        ax.fill_betweenx(xs, 0.0, m, color=color, alpha=0.6, lw=0)
        ax.set_xlim(0, max(m.max(), 1e-300))
        ax.invert_xaxis()
    else:
        ax.fill_between(xs, 0.0, m, color=color, alpha=0.6, lw=0)
        ax.set_ylim(0, max(m.max(), 1e-300))
        ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.annotate(f'{total:.3g}', xy=(0.5, 0.5), xycoords='axes fraction',
                fontsize=6, ha='center', va='center', color=color)
    ax.margins(0)


def _exceedance_grid(mass):
    """Joint exceedance ``P(X > x, Y > y)`` by 2-D suffix sums of block mass."""
    s = mass[::-1, ::-1].cumsum(axis=0).cumsum(axis=1)[::-1, ::-1]
    # shift: strictly-greater-than both coordinates
    e = np.zeros_like(s)
    e[:-1, :-1] = s[1:, 1:]
    return e


# ----------------------------------------------------------------------
# Layer 2: compositors
# ----------------------------------------------------------------------

def plot_bivariate_massive(bd, window=None, log=True, contours=False,
                           exceedance=False, pixels=800, cmap='viridis',
                           clip_pct=(0.5, 99.9), axs=None):
    """Exploration-grade exhibit of a massive joint density (plan-bv §7.2).

    Parameters
    ----------
    bd : MassiveBivariateDistribution
        The disk-backed joint.
    window : ((x0, x1), (y0, y1)), optional
        Value-coordinate zoom window; default the full grid. Same call
        contract as the rest of the subsystem: pick the level, read the
        tiles, done -- constant cost at any zoom.
    log : bool, default True
        ``log10`` color (the house default here -- linear hides everything
        but the mode).
    contours : bool, default False
        Overlay decade contours of the log-density.
    exceedance : bool, default False
        Overlay joint-exceedance contours ``P(X > x, Y > y)`` (suffix sums
        at tile resolution -- the joint-tail view a risk analyst reads).
    pixels : int, default 800
        Per-axis pixel budget; selects the pyramid level.
    cmap : str, default 'viridis'
        Colormap for the sum-channel field.
    clip_pct : (float, float), default (0.5, 99.9)
        Percentile clip of the color range (atoms / mode would otherwise
        own the scale).
    axs : dict of Axes, optional
        A mosaic dict with keys ``'H'`` (heatmap), ``'T'`` / ``'M'``
        (marginals), ``'A'`` / ``'B'`` (atom strips); created if omitted.

    Returns
    -------
    None
        The figure is stored on ``bd.figure`` (house convention -- avoids
        the Jupyter double render).
    """
    i0, i1, j0, j1 = _resolve_window(bd, window)
    level = _select_level(bd, i0, i1, j0, j1, pixels)
    mass, mx, mn, extent, (bi0, bi1, bj0, bj1) = _read_tiles(
        bd, i0, i1, j0, j1, level)

    # axis atoms (zero-inflated books carry finite mass ON the axes): shown
    # as flanking strips; at level 0 the atom row/col is excluded from the
    # interior field so it cannot saturate the color scale. At coarser
    # levels the atom sits inside its 2^k block and the max-channel glow
    # carries it (documented compromise).
    iz = _axis_zero_index(bd.xs0, bd.bs0)
    jz = _axis_zero_index(bd.xs1, bd.bs1)
    atom_x = np.asarray(bd.density[iz, :]) if iz is not None else None
    atom_y = np.asarray(bd.density[:, jz]) if jz is not None else None
    origin_p = float(bd.density[iz, jz]) if (iz is not None and jz is not None) \
        else None
    if level == 0:
        if iz is not None and bi0 <= iz < bi1:
            mass = mass.copy()
            mass[iz - bi0, :] = 0.0
        if jz is not None and bj0 <= jz < bj1:
            mass = mass.copy()
            mass[:, jz - bj0] = 0.0
        mx = mass

    if axs is None:
        fig, axd = make_mosaic('.T.\nAHM\n.B.', figsize=(1.6 * FIG_W, 1.4 * FIG_H),
                               width_ratios=[0.05, 1.0, 0.22],
                               height_ratios=[0.22, 1.0, 0.05])
    else:
        axd = axs
        fig = axd['H'].figure
    bd.figure = fig

    ax = axd['H']
    _render_heatmap(ax, mass, mx, extent, level=level, log=log, cmap=cmap,
                    clip_pct=clip_pct)
    names = bd.axis_names
    ax.set(xlabel=str(names[0]), ylabel=str(names[1]))

    if contours or exceedance:
        cx = bd.xs0[bi0] + (np.arange(mass.shape[0]) + 0.5) * bd.bs0 * (1 << level)
        cy = bd.xs1[bj0] + (np.arange(mass.shape[1]) + 0.5) * bd.bs1 * (1 << level)
    if contours:
        with np.errstate(divide='ignore'):
            lf = np.log10(np.where(mass > 0, mass, np.nan))
        lo = np.floor(np.nanmin(lf))
        hi = np.ceil(np.nanmax(lf))
        levels = np.arange(lo, hi + 1)
        if len(levels) >= 2:
            ax.contour(cx, cy, lf.T, levels=levels, colors='w',
                       linewidths=0.5, alpha=0.6)
    if exceedance:
        e = _exceedance_grid(mass) / max(mass.sum(), 1e-300)
        cs = ax.contour(cx, cy, e.T, levels=[0.001, 0.01, 0.1, 0.25, 0.5],
                        colors='crimson', linewidths=0.9)
        ax.clabel(cs, fmt='%g', fontsize=7)

    # exact marginal panels, windowed to the heatmap view
    _render_marginal(axd['T'], bd.xs0[bi0:bi1], bd.marg0[bi0:bi1])
    axd['T'].set_xlim(extent[0], extent[1])
    axd['T'].set_title(bd.meta.get('name', ''), fontsize=9)
    _render_marginal(axd['M'], bd.xs1[bj0:bj1], bd.marg1[bj0:bj1],
                     vertical=True)
    axd['M'].set_ylim(extent[2], extent[3])

    # atom strips + origin badge: P(X=0, .) is a function of y -> the left
    # vertical strip; P(., Y=0) a function of x -> the bottom strip.
    _render_atom_strip(axd['A'], bd.xs1, atom_x, vertical=True)
    _render_atom_strip(axd['B'], bd.xs0, atom_y, vertical=False)
    if origin_p is not None and origin_p > _ATOM_MASS_FLOOR:
        ax.annotate(f'P(0,0) = {origin_p:.3g}', xy=(0.02, 0.02),
                    xycoords='axes fraction', fontsize=7,
                    bbox=dict(boxstyle='round', fc='w', alpha=0.7))
    axd['A'].set_ylim(extent[2], extent[3])
    axd['B'].set_xlim(extent[0], extent[1])


def plot_bivariate_massive_slice(bd, x=None, y=None, ax=None, log=False):
    """Conditional strip chart ``P(Y | X ~ x)`` / ``P(X | Y ~ y)`` (§7.2).

    One row / column of tiles read (:meth:`MassiveBivariateDistribution.slice`)
    -- instant at any grid size.

    Parameters
    ----------
    bd : MassiveBivariateDistribution
    x, y : float, optional
        Exactly one: the conditioning value.
    ax : matplotlib Axes, optional
    log : bool, default False
        Log-scale the density axis.

    Returns
    -------
    matplotlib Axes
    """
    from ._style import make_grid
    gd = bd.slice(x=x, y=y)
    if ax is None:
        _, ax = make_grid(1, 1, squeeze=True)
    ax.plot(gd.x, gd.p, lw=1.0, drawstyle='steps-mid')
    ax.fill_between(gd.x, 0.0, gd.p, step='mid', alpha=0.3)
    if log:
        ax.set_yscale('log')
    ax.set(title=gd.name, xlabel='outcome', ylabel='conditional density')
    ax.margins(x=0)
    return ax
