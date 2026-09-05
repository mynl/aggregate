"""Approximation chart emitter: the five fits on the realized mass.

The teaching picture behind :attr:`~aggregate.Aggregate.approximation_df`:
how well does each method-of-moments family reproduce the book. One
density panel over the outcome axis, overlaying the five fitted laws on
the realized mass in grid-mass terms (``pdf * bs``, the
:attr:`~aggregate.Aggregate.approximation_density_df` columns), so the
curves and the bars are the same unit.

A tail (exceedance) panel with the sub-exponential implied tail
``E[N] * S_X(x)`` shipped from ``1.0.0a332`` to ``1.0.0a336`` and was
dropped by ruling at ``1.0.0a337``: the tail behavior is legible from the
``exact`` reading and the frame's quantile blocks, and the panel's twelve
full-grid curves dominated the document's byte count. The frame
(``approximation_df``) is unchanged. To study the tail explicitly, build
the fitted law as its own aggregate (``approximate`` object mode) and
read its charts.

Pure numpy and pandas; no matplotlib.
"""

import numpy as np

from .._aggregate import Aggregate
from . import register_chart, _emitter_base
from ._payload import collapse_empty_runs, lattice_payload
from ._two_panel import loss_window
from .ir import ChartAxis, ChartDoc, ChartSeries, Mark, Panel, complete_tex

__all__ = ['chart_approximation']

chart_approximation = _emitter_base('approximation')


def _updated(agg):
    """Availability: there is a realized grid to draw."""
    return getattr(agg, 'agg_density', None) is not None


@chart_approximation.register(Aggregate)
def _approximation(agg, xmax=None):
    """Emit the single-panel approximation chart for an aggregate.

    Parameters
    ----------
    agg : Aggregate
        Must be updated.
    xmax : float, optional
        Upper end of the outcome window, in place of the computed one (the
        same semantic option the ``agg`` chart takes).

    Returns
    -------
    ChartDoc
        One 'xy' panel, 'density': the realized mass with the five family
        densities overlaid in grid-mass terms, mean marked.

    Notes
    -----
    The family curves are the laws of the **emitted programs** (mirrored
    severity keyword, clamp included), read straight off
    ``approximation_density_df``, so the chart and the exhibit cannot
    disagree about which laws they describe. The tail comparison lives in
    the frame's ``quantiles`` / ``rel err`` blocks; the exceedance panel
    this chart carried through ``1.0.0a336`` was dropped by ruling (see
    the module docstring).
    """
    ddf = agg.approximation_density_df
    x = ddf.index.to_numpy(dtype=float)
    exact_mass = ddf['exact'].to_numpy(dtype=float)
    families = [c for c in ddf.columns if c != 'exact']

    series = []
    drawn_x, drawn_mass = collapse_empty_runs(x, exact_mass)
    series.append(ChartSeries(
        name='Exact', role='density', panel_id='density',
        y=tuple(float(v) for v in drawn_mass),
        **lattice_payload(drawn_x, agg.bs)))
    # A family the subject cannot admit (an unshifted gamma / lognorm fit
    # of a negative-mean book has no valid law) evaluates non-finite and is
    # not drawn; the frame keeps its NaN column, which is the reading.
    for kind in families:
        vals = ddf[kind].to_numpy(dtype=float)
        if not np.all(np.isfinite(vals)):
            continue
        series.append(ChartSeries(
            name=kind, role='density', panel_id='density',
            support='continuous',
            y=tuple(float(v) for v in vals),
            **lattice_payload(x, agg.bs)))

    window = (loss_window(agg.q, x[0]) if xmax is None
              else (min(0.0, float(x[0])), float(xmax)))
    peak = float(exact_mass.max())
    fam_peak = float(np.nanmax(ddf[families].to_numpy(dtype=float)))
    signed = bool(agg._signed())

    marks = [Mark(panel_id='density', orient='v', at=float(agg.est_m),
                  label='mean', role='mean')]

    return complete_tex(ChartDoc(
        name='approximation',
        title=f'{agg.label}: method-of-moments approximations',
        axes=(
            ChartAxis(id='outcome', label='Loss', unit='currency',
                      scales=('linear',) if signed else ('linear', 'log'),
                      suggested_range=window,
                      full_range=(min(0.0, float(x[0])), float(x[-1]))),
            ChartAxis(id='mass', label='Probability mass', unit='density',
                      scales=('linear', 'log'),
                      suggested_range=(0.0, max(peak, fam_peak))),
        ),
        panels=(
            Panel(id='density', kind='xy', x_axis='outcome', y_axis='mass',
                  title='Probability mass function'),
        ),
        series=tuple(series),
        marks=tuple(marks),
        meta={'ordinate': 'mass'},
    ))


register_chart('approximation', chart_approximation, predicate=_updated)
