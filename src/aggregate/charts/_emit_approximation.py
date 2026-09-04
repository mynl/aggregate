"""Approximation chart emitter: the five fits and the implied tail.

The teaching picture behind :attr:`~aggregate.Aggregate.approximation_df`:
how well does each method-of-moments family reproduce the book, and where
does every parametric family give out. Two panels over one outcome axis.
The density panel overlays the five fitted laws on the realized mass, in
grid-mass terms (``pdf * bs``, the
:attr:`~aggregate.Aggregate.approximation_density_df` columns), so the
curves and the bars are the same unit. The tail panel draws the exceedance
probabilities on a log-capable axis, which is where the Berry-Esseen story
is visible: the normal peels off first, the shifted families track
furthest, and the **sub-exponential implied tail** ``E[N] * S_X(x)``,
computed from the exact severity functions (:attr:`Aggregate.sev`, exact
rather than gridded), says where the aggregate stops being approximable by
any of them and becomes its own largest claim.

The implied-tail curve needs a single severity, so the chart registers on
:class:`Aggregate` only; ``Portfolio`` keeps the frames and the exhibit.
It is trimmed to where ``E[N] * S_X(x) <= 1``: it is an asymptotic tail
statement, not a distribution, and above 1 it says nothing.

Pure numpy and pandas; no matplotlib.
"""

import numpy as np

from .._aggregate import Aggregate
from . import register_chart, _emitter_base
from ._payload import collapse_empty_runs, lattice_payload
from ._two_panel import (RETURN_PERIOD_TOP, SURVIVAL_FLOOR, loss_window,
                         survival_window)
from .ir import ChartAxis, ChartDoc, ChartSeries, Mark, Panel, complete_tex

__all__ = ['chart_approximation']

chart_approximation = _emitter_base('approximation')


def _updated(agg):
    """Availability: there is a realized grid to draw."""
    return getattr(agg, 'agg_density', None) is not None


def _tail_series(name, x, survival, role='survival'):
    """One tail-panel series, trimmed to survivals worth drawing.

    Points at or below :data:`SURVIVAL_FLOOR` are float dust and are cut
    at the deep end, so the log reading stops where the curve stops
    meaning anything; ``NaN`` points (the implied tail where it exceeds 1)
    are cut at the near end. Both cuts are end trims on a monotone
    decreasing curve, so the kept run is contiguous.
    """
    keep = np.isfinite(survival) & (survival > SURVIVAL_FLOOR)
    if not np.any(keep):
        return None
    idx = np.nonzero(keep)[0]
    start, stop = int(idx[0]), int(idx[-1]) + 1
    return ChartSeries(
        name=name, role=role, panel_id='tail',
        x=tuple(float(v) for v in x[start:stop]),
        y=tuple(float(v) for v in survival[start:stop]))


@chart_approximation.register(Aggregate)
def _approximation(agg, xmax=None):
    """Emit the two-panel approximation chart for an aggregate.

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
        Two 'xy' panels over one shared outcome axis: 'density' (the
        realized mass with the five family densities overlaid in grid-mass
        terms) and 'tail' (the exceedance probabilities of all six laws
        plus the sub-exponential implied tail ``E[N] * S_X(x)``, on a
        log-capable survival axis).

    Notes
    -----
    The family curves are the laws of the **emitted programs** (mirrored
    severity keyword, clamp included), read straight off
    ``approximation_density_df`` and the same closed-form cumulatives that
    feed ``approximation_df``, so the chart and the exhibit cannot
    disagree about which laws they describe. The exact tail is
    ``1 - cumsum(p_total)`` at the bucket convention
    ``F(x_k) = P(X <= x_k)``.
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

    # Tail panel: exceedance of the exact law and of each emitted law, and
    # the implied tail off the exact severity functions. The exact running
    # sum under the round convention means P(X <= x_k + bs/2), so every
    # continuous curve is read at the same upper half-edge; a curve read at
    # the grid points themselves would sit half a bucket to the right of
    # the staircase it is compared against (see ``approximation_frame``).
    from .._aggregate import _approximation_laws
    laws = _approximation_laws(agg.est_m, agg.est_cv, agg.est_skew,
                               agg._signed())
    edges = x + 0.5 * float(agg.bs)
    exact_survival = 1.0 - np.cumsum(exact_mass)
    tail = [_tail_series('Exact', x, exact_survival)]
    with np.errstate(divide='ignore', invalid='ignore'):
        for kind in families:
            tail.append(_tail_series(kind, x,
                                     1.0 - laws[kind]['cdf'](edges)))
        implied = float(agg.n) * np.asarray(agg.sev.sf(edges), dtype=float)
    implied = np.where(implied <= 1.0, implied, np.nan)
    tail.append(_tail_series('Implied tail', x, implied, role='ceiling'))
    tail = [s for s in tail if s is not None]
    series.extend(tail)

    window = (loss_window(agg.q, x[0]) if xmax is None
              else (min(0.0, float(x[0])), float(xmax)))
    peak = float(exact_mass.max())
    fam_peak = float(np.nanmax(ddf[families].to_numpy(dtype=float)))
    signed = bool(agg._signed())

    marks = [Mark(panel_id='tail', orient='v', at=float(agg.est_m),
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
            # The survival axis is where the story is: log is the reading
            # that separates the families, so it is offered, and the
            # suggested window is rounded to whole decades over the
            # deepest survival drawn.
            ChartAxis(id='survival', label='Exceeding probability',
                      unit='probability', scales=('linear', 'log'),
                      suggested_range=survival_window(
                          [s.y for s in tail])),
            # The alternative reading of the survival axis, declared as
            # the other emitters declare it.
            ChartAxis(id='return_period', label='Return period',
                      unit='return_period', scales=('linear', 'log'),
                      reciprocal_of='survival',
                      suggested_range=(1.0, RETURN_PERIOD_TOP),
                      full_range=(1.0,
                                  float(round(1.0 / SURVIVAL_FLOOR)))),
        ),
        panels=(
            Panel(id='density', kind='xy', x_axis='outcome', y_axis='mass',
                  title='Probability mass function'),
            Panel(id='tail', kind='xy', x_axis='outcome', y_axis='survival',
                  title='Exceedance (survival) function'),
        ),
        series=tuple(series),
        marks=tuple(marks),
        meta={'ordinate': 'mass', 'return_period_map': 'reciprocal'},
    ))


register_chart('approximation', chart_approximation, predicate=_updated)
