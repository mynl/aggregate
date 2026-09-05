"""Approximation chart emitter: the five fits on the realized mass.

The teaching picture behind :attr:`~aggregate.Aggregate.approximation_df`:
how well does each method-of-moments family reproduce the book. One
density panel over the outcome axis, overlaying the five fitted laws on
the realized mass in grid-mass terms (``pdf * bs``, the
:attr:`~aggregate.Aggregate.approximation_density_df` columns), so the
curves and the bars are the same unit.

A tail (exceedance) panel with the sub-exponential implied tail
``E[N] * S_X(x)`` shipped from ``1.0.0a332`` to ``1.0.0a336`` and was
dropped by ruling at ``1.0.0a337``: the panel's twelve full-grid curves
dominated the *default* document's byte count. At ``1.0.0a344`` the
picture returned as its own chart, ``approximation_tails``, fetched only
when asked for: one survival curve per law (the cdf is the declared
complement reading of the same series, and the exchanged panel is the
upper Lee plot), plus the implied tail. The frame (``approximation_df``)
is unchanged throughout.

Pure numpy and pandas; no matplotlib.
"""

import numpy as np

from .._aggregate import Aggregate, _approximation_laws
from . import register_chart, _emitter_base
from ._payload import collapse_empty_runs, lattice_payload
from ._two_panel import (RETURN_PERIOD_TOP, SURVIVAL_FLOOR, loss_window,
                         survival_window)
from .ir import ChartAxis, ChartDoc, ChartSeries, Mark, Panel, complete_tex

__all__ = ['chart_approximation', 'chart_approximation_tails']

chart_approximation = _emitter_base('approximation')
chart_approximation_tails = _emitter_base('approximation_tails')


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
    disagree about which laws they describe. The tail comparison is the
    companion chart, ``approximation_tails``, its own document so this
    default fetch stays light (see the module docstring).
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


def _tail_series(name, x, survival, bs, role='survival'):
    """One tail-panel series, trimmed to survivals worth drawing.

    Points at or below :data:`SURVIVAL_FLOOR` are float dust and are cut
    at the deep end, so the log reading stops where the curve stops
    meaning anything; ``NaN`` points (the implied tail where it exceeds 1)
    are cut at the near end. Both cuts are end trims on a monotone
    decreasing curve, so the kept run is contiguous and its x stays a
    lattice.
    """
    keep = np.isfinite(survival) & (survival > SURVIVAL_FLOOR)
    if not np.any(keep):
        return None
    idx = np.nonzero(keep)[0]
    start, stop = int(idx[0]), int(idx[-1]) + 1
    return ChartSeries(
        name=name, role=role, panel_id='tails',
        y=tuple(float(v) for v in survival[start:stop]),
        **lattice_payload(x[start:stop], bs))


@chart_approximation_tails.register(Aggregate)
def _approximation_tails(agg, xmax=None):
    """Emit the tail (exceedance) chart for the five fits on an aggregate.

    Parameters
    ----------
    agg : Aggregate
        Must be updated.
    xmax : float, optional
        Upper end of the outcome window, in place of the computed one (the
        same semantic option the ``agg`` and ``approximation`` charts take).

    Returns
    -------
    ChartDoc
        One 'xy' panel, 'tails': the exceedance probability of the exact
        law and of each emitted law on a log-capable survival axis, plus
        the sub-exponential implied tail ``E[N] * S_X(x)`` (role
        'ceiling'), computed from the exact severity functions and trimmed
        to where it says something (``<= 1``).

    Notes
    -----
    The companion to the ``approximation`` density chart, reading the same
    view (the cached ``approximation_density_df`` grid and the same
    closed-form cumulatives that feed ``approximation_df``), split into
    its own document so the default fetch stays light (the ``1.0.0a337``
    byte-count ruling). Each law ships **one** curve: the non-exceedance
    reading is the declared complement axis, the return period the
    declared reciprocal axis, and the exchanged panel is the upper Lee
    (quantile) plot, so nothing travels twice.

    The exact tail is ``1 - cumsum(p_total)`` at the bucket convention
    ``F(x_k) = P(X <= x_k)``; the running sum under the round convention
    means ``P(X <= x_k + bs/2)``, so every continuous curve is read at the
    upper half-edge, or it would sit half a bucket right of the staircase
    it is compared against (see ``approximation_frame``). The implied-tail
    curve needs a single severity, so the chart registers on
    :class:`Aggregate` only; ``Portfolio`` keeps the frames and the
    exhibit.
    """
    ddf = agg.approximation_density_df
    x = ddf.index.to_numpy(dtype=float)
    exact_mass = ddf['exact'].to_numpy(dtype=float)
    families = [c for c in ddf.columns if c != 'exact']

    laws = _approximation_laws(agg.est_m, agg.est_cv, agg.est_skew,
                               agg._signed())
    edges = x + 0.5 * float(agg.bs)
    exact_survival = 1.0 - np.cumsum(exact_mass)
    tail = [_tail_series('Exact', x, exact_survival, agg.bs)]
    with np.errstate(divide='ignore', invalid='ignore'):
        for kind in families:
            survival = 1.0 - laws[kind]['cdf'](edges)
            if not np.any(np.isfinite(survival)):
                continue
            tail.append(_tail_series(kind, x, survival, agg.bs))
        implied = float(agg.n) * np.asarray(agg.sev.sf(edges), dtype=float)
    implied = np.where(implied <= 1.0, implied, np.nan)
    tail.append(_tail_series('Implied tail', x, implied, agg.bs,
                             role='ceiling'))
    tail = [s for s in tail if s is not None]

    window = (loss_window(agg.q, x[0]) if xmax is None
              else (min(0.0, float(x[0])), float(xmax)))
    signed = bool(agg._signed())

    marks = [Mark(panel_id='tails', orient='v', at=float(agg.est_m),
                  label='mean', role='mean')]

    return complete_tex(ChartDoc(
        name='approximation_tails',
        title=f'{agg.label}: approximation tails',
        axes=(
            ChartAxis(id='outcome', label='Loss', unit='currency',
                      scales=('linear',) if signed else ('linear', 'log'),
                      suggested_range=window,
                      full_range=(min(0.0, float(x[0])), float(x[-1]))),
            # The survival axis is where the story is: log is the reading
            # that separates the families, so it is offered, and the
            # suggested window is rounded to whole decades over the
            # deepest survival drawn.
            ChartAxis(id='survival', label='Exceeding probability',
                      unit='probability', scales=('linear', 'log'),
                      suggested_range=survival_window(
                          [s.y for s in tail])),
            # Undrawn: the reflected reading of the drawn survival axis,
            # which is the cdf the reader asked this chart for without a
            # second set of curves.
            ChartAxis(id='p', label='Non-exceeding probability',
                      unit='probability', complement_of='survival',
                      suggested_range=(0.0, 1.0)),
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
            # Exchanged, the exceedance curve is loss as a function of
            # survival probability, the upper Lee (quantile) plot.
            Panel(id='tails', kind='xy', x_axis='outcome', y_axis='survival',
                  invertible=True, title='Exceedance (survival) function',
                  inverse_title='Quantile (upper Lee) plot'),
        ),
        series=tuple(tail),
        marks=tuple(marks),
        meta={'ordinate': 'survival', 'return_period_map': 'reciprocal'},
    ))


register_chart('approximation', chart_approximation, predicate=_updated)
register_chart('approximation_tails', chart_approximation_tails,
               predicate=_updated)
