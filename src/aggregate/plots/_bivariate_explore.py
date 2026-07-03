"""Tier-2 interactive explorer for the massive bivariate (plan-bv §7.3).

A **holoviews + datashader + bokeh** app over the decimation pyramid:
Google-Maps-style pan / zoom of a multi-gigacell pmf in JupyterLab. Every
viewport change re-reads only the pyramid level matching the pixel budget
(the same constant-cost data layer as the static Tier-1 exhibit), so a
``(16, 16)`` joint explores as fast as a ``(8, 8)`` one.

Panels:

- the main log10-density image with dynamic re-aggregation on pan / zoom, a
  channel selector (sum / max / min) and a hover readout;
- linked marginal panels that re-window to the viewport;
- click-to-slice: tap the joint to get the conditional ``P(Y | X ~ x)`` at
  the tapped abscissa;
- a pointer readout table with ``x, y, p, log10 p`` and the (block-resolution)
  joint cdf ``P(X <= x, Y <= y)``.

This module is imported **only** from
:meth:`aggregate.bivariate.MassiveBivariateDistribution.explore` -- the
matplotlib-only boundary of ``aggregate.plots.__init__`` is untouched. The
heavy stack is an optional extra: ``pip install aggregate[viz]``.
"""

import numpy as np

from ._bivariate_massive import (_resolve_window, _select_level, _read_tiles,
                                 _support_box)


def _require_viz():
    """Import the Tier-2 stack with an actionable error if missing."""
    try:
        import holoviews as hv
        import datashader                       # noqa: F401 -- regrid backend
        import bokeh                            # noqa: F401 -- render backend
    except ImportError as e:
        raise ImportError(
            "the interactive explorer requires holoviews, datashader and "
            "bokeh; install the optional extra: pip install aggregate[viz]"
        ) from e
    return hv


def explore(bd, *, pixels=800, cmap='viridis', width=650, height=500):
    """Build the interactive explorer app for a massive joint density.

    Display the returned object in JupyterLab (it renders itself); pan,
    zoom, switch channels, hover for values, tap for a conditional slice.

    Parameters
    ----------
    bd : MassiveBivariateDistribution
        The disk-backed joint.
    pixels : int, default 800
        Per-viewport pixel budget -- picks the pyramid level per redraw.
    cmap : str, default 'viridis'
        Colormap of the main image.
    width, height : int
        Main panel size in screen pixels.

    Returns
    -------
    holoviews.Layout
        The composed app: main image with linked marginals, the tap-slice
        panel and the pointer readout table.

    Notes
    -----
    The app needs a **live Python kernel**: display the returned layout in a
    JupyterLab cell (or serve it with panel/bokeh server). Every dynamic
    feature -- pan/zoom re-aggregation, the channel widget, hover, and
    click-to-slice -- is a Python callback reading the zarr store on demand.
    An ``hv.save(...)`` HTML export is a *static snapshot*: it shows the
    initial frames but none of the callbacks can run there.
    """
    hv = _require_viz()
    from holoviews import streams
    hv.extension('bokeh')

    names = tuple(str(n) for n in bd.axis_names)
    box = _support_box(bd) or ((float(bd.xs0[0]), float(bd.xs0[-1])),
                               (float(bd.xs1[0]), float(bd.xs1[-1])))

    def _window(x_range, y_range):
        xr = tuple(x_range) if x_range and np.all(np.isfinite(x_range)) \
            else box[0]
        yr = tuple(y_range) if y_range and np.all(np.isfinite(y_range)) \
            else box[1]
        return xr, yr

    # ---------------- main image: dynamic re-aggregation over the pyramid
    def tile(x_range, y_range, channel):
        xr, yr = _window(x_range, y_range)
        i0, i1, j0, j1 = _resolve_window(bd, (xr, yr))
        level = _select_level(bd, i0, i1, j0, j1, pixels)
        mass, mx, mn, extent, _ = _read_tiles(bd, i0, i1, j0, j1, level)
        arr = {'sum': mass, 'max': mx, 'min': mn}[channel]
        with np.errstate(divide='ignore', invalid='ignore'):
            img = np.log10(np.where(arr > 0, arr, np.nan))
        # hv.Image data is [row=y descending, col=x]; ours is [x, y]
        data = np.flipud(img.T)
        return hv.Image(data, bounds=(extent[0], extent[2],
                                      extent[1], extent[3]),
                        kdims=[names[0], names[1]], vdims=['log10 p'])

    rangexy = streams.RangeXY(x_range=box[0], y_range=box[1])
    channel_dim = hv.Dimension('channel', values=['sum', 'max', 'min'],
                               default='sum')
    main = hv.DynamicMap(tile, kdims=[channel_dim], streams=[rangexy])
    # NB: the tap / pointer streams below take main as their event source, so
    # main must be the object that is actually displayed -- no .relabel() /
    # clone after this point (a clone would receive no browser events and the
    # click-to-slice panel would sit dead).
    main = main.opts(
        'Image', cmap=cmap, width=width, height=height, colorbar=True,
        tools=['hover'], active_tools=['wheel_zoom'],
        title=str(bd.meta.get('name', '')),
        clipping_colors={'NaN': (0, 0, 0, 0)})

    # ---------------- linked marginals, re-windowed to the viewport
    def marg_top(x_range, y_range):
        xr, _ = _window(x_range, y_range)
        m = (bd.xs0 >= xr[0]) & (bd.xs0 <= xr[1])
        return hv.Curve((bd.xs0[m], bd.marg0[m]), names[0], 'p')

    def marg_right(x_range, y_range):
        _, yr = _window(x_range, y_range)
        m = (bd.xs1 >= yr[0]) & (bd.xs1 <= yr[1])
        return hv.Curve((bd.marg1[m], bd.xs1[m]), 'p', names[1])

    top = hv.DynamicMap(marg_top, streams=[rangexy]).opts(
        width=width, height=130)
    right = hv.DynamicMap(marg_right, streams=[rangexy]).opts(
        width=130, height=height)

    # ---------------- click-to-slice conditional
    tap = streams.SingleTap(x=None, y=None)

    def tap_slice(x, y):
        if x is None:
            return hv.Curve([], names[1],
                            'conditional p').opts(title='tap the joint to slice')
        gd = bd.slice(x=float(x))
        return hv.Curve((gd.x, gd.p), names[1], 'conditional p').opts(
            title=gd.name)

    slice_panel = hv.DynamicMap(tap_slice, streams=[tap]).opts(
        width=width, height=220)
    tap.source = main

    # ---------------- pointer readout: p, log10 p, joint cdf (block res)
    pyr = bd.pyramid
    lv = pyr.attrs['levels'] if pyr is not None else []
    if lv:
        coarse = np.asarray(pyr[lv[-1]['name']][0])
        step = 1 << len(lv)
    else:
        coarse = np.asarray(bd.density[:, :])
        step = 1
    cdf2 = coarse.cumsum(axis=0).cumsum(axis=1)

    pointer = streams.PointerXY(x=box[0][0], y=box[1][0])

    def readout(x, y):
        i = int(np.clip(round((x - bd.xs0[0]) / bd.bs0), 0, len(bd.xs0) - 1))
        j = int(np.clip(round((y - bd.xs1[0]) / bd.bs1), 0, len(bd.xs1) - 1))
        p = float(bd.density[i, j])
        f = float(cdf2[min(i // step, cdf2.shape[0] - 1),
                       min(j // step, cdf2.shape[1] - 1)])
        rows = [(names[0], float(bd.xs0[i])), (names[1], float(bd.xs1[j])),
                ('p', p), ('log10 p', np.log10(p) if p > 0 else float('-inf')),
                ('F(x, y)', f)]
        return hv.Table(rows, 'quantity', 'value')

    readout_panel = hv.DynamicMap(readout, streams=[pointer]).opts(
        width=280, height=220)
    pointer.source = main

    layout = (main << right << top) + slice_panel + readout_panel
    return layout.cols(1)
