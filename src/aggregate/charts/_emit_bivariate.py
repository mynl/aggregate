"""Bivariate chart emitters: the joint density surface (the pilot).

``chart_joint_surface`` pulls the joint matrix off a
:class:`~aggregate.bivariate.BivariateAggregate`, block-sums it to a display
grid inside the emitter, and returns a :class:`~aggregate.charts.ir.ChartDoc`
with one 'surface' panel. The reduction migrates the app's ``surfaceGrid``
(aggregate_api ``web/src/charts/surface.js``): reduction is meaning, so it
happens here, mass-preservingly, never renderer-side.

Pure numpy; no matplotlib (the plots boundary), no ECharts vocabulary.
"""

import math

import numpy as np

from ..bivariate import BivariateAggregate
from . import register_chart, _emitter_base
from .ir import (ChartAxis, ChartDoc, ChartSeries, Panel, SurfaceData,
                 complete_tex)

__all__ = ['chart_joint_surface']

#: Default display cells per side, migrated from the app's ``CELLS = 128``
#: (surface.js:27): 128 x 128 is 16,384 vertices, WebGL-cheap and finer than
#: the eye resolves on a panel, while the block *sum* (never a sample)
#: preserves every gram of tail mass.
DISPLAY_CELLS = 128


def _reduce_grid(density, x0, x1, cells):
    """Mass-preserving block-sum of a joint matrix to a display grid.

    Parameters
    ----------
    density : ndarray
        Joint mass matrix, ``density[i, j]`` on ``(x0[i], x1[j])``.
    x0, x1 : ndarray
        The two axis grids (independent components routinely land on
        different bucket sizes, so these are two vectors, never one step).
    cells : int
        Target cells per side. Axes at or under it pass through.

    Returns
    -------
    xs, ys, z : ndarray
        Display grids and the reduced matrix ``z[i, j]`` at
        ``(xs[i], ys[j])``.

    Notes
    -----
    Exactly the app's ``surfaceGrid`` algorithm: fixed-size prefix blocks
    ``bx = ceil(n / cells)`` with a short final block (every source cell
    lands in exactly one block, so mass is conserved), non-finite source
    cells contribute zero, and labels are the clamped block *right edges*
    (the inventory's judgment call J3; the divergence from the server's
    node-centered ``bin_density`` is recorded there for reconciliation).
    """
    nx, ny = density.shape
    bx = max(1, math.ceil(nx / cells))
    by = max(1, math.ceil(ny / cells))
    z = np.nan_to_num(np.asarray(density, dtype=float),
                      nan=0.0, posinf=0.0, neginf=0.0)
    if bx > 1:
        z = np.add.reduceat(z, np.arange(0, nx, bx), axis=0)
    if by > 1:
        z = np.add.reduceat(z, np.arange(0, ny, by), axis=1)
    xs = np.asarray(x0)[np.minimum(np.arange(1, z.shape[0] + 1) * bx - 1,
                                   nx - 1)]
    ys = np.asarray(x1)[np.minimum(np.arange(1, z.shape[1] + 1) * by - 1,
                                   ny - 1)]
    return xs, ys, z


chart_joint_surface = _emitter_base('joint_surface')


@chart_joint_surface.register(BivariateAggregate)
def _joint_surface(bv, display_log2=None):
    """Emit the joint density surface for a bivariate aggregate.

    Parameters
    ----------
    bv : BivariateAggregate
        An updated bivariate (in-memory joint density; the disk-backed
        massive pyramid stays bespoke, per the plan's scope).
    display_log2 : int, optional
        Log2 of the display cells per side; ``None`` means the migrated
        app default of 128 (:data:`DISPLAY_CELLS`).

    Returns
    -------
    ChartDoc
        One 'surface' panel: axes labeled with the resolved component
        labels, and a z axis read linearly by default that declares
        ``scales=('linear', 'log')``, because a joint density spans four or
        five orders of magnitude and the log reading is where tail
        dependence lives. Values are display-cell *masses*, not ordinates.
    """
    density = getattr(bv, 'density', None)
    if density is None:
        raise ValueError(
            'chart_joint_surface needs the in-memory joint density: '
            'update() first (a massive, disk-backed bivariate is out of '
            'scope for this chart).')
    cells = DISPLAY_CELLS if display_log2 is None else 1 << int(display_log2)
    xs, ys, z = _reduce_grid(density, bv.axis_xs[0], bv.axis_xs[1], cells)
    names = list(bv.unit_names)
    if bv.use_labels:
        renamer = bv.renamer
        labels = [renamer.get(n, n) for n in names]
    else:
        labels = names
    surface = SurfaceData(
        x=tuple(float(v) for v in xs),
        y=tuple(float(v) for v in ys),
        # SurfaceData is row-major over y (z[r][c] at (x[c], y[r])), the
        # reduced matrix is [x-block, y-block]: transpose once here.
        z=tuple(tuple(float(v) for v in row) for row in z.T),
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
    predicate=lambda bv: getattr(bv, 'density', None) is not None)
