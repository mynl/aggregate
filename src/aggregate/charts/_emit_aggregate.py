"""Aggregate chart emitter: the mass and the Lee diagram.

Two panels, and the choice of which two is the whole design. The
compositor drew three, a density, a log density, and a Lee diagram, and the
middle one is not a third reading of the book: it is the first one read on
log. So it becomes a **declared reading** of the density panel, whose axes
both offer log, and the reader chooses it instead of being handed it
whether they wanted it or not.

The Lee diagram earns its panel because it is a genuinely different
question. The density asks what is likely; the Lee diagram is interrogated
the other way, at a chosen probability, and answers with a loss. Its
probability axis carries the paired return-period reading, which is the
same question asked in the language of reinsurance submissions and capital
standards, and which used to be the ``quantile_x='return'`` argument.

**The ordinate is a mass, not a density.** A discretized aggregate *is* the
distribution here rather than an approximation to a continuous ideal, which
is what ``support='atomic'`` says and what the renderer's ladder draws.
Dividing by the bucket size to manufacture a density ordinate would be that
continuous ideal creeping back in, and it is exactly the number a reader
cannot add up. The severity companion is a mass on the same grid, so the
two are comparable as drawn.

Pure numpy and pandas; no matplotlib.
"""

import numpy as np

from .._aggregate import Aggregate
from . import register_chart, _emitter_base
from ._two_panel import SURVIVAL_FLOOR, loss_window, quantile_curve
from .ir import ChartAxis, ChartDoc, ChartSeries, Mark, Panel, complete_tex

__all__ = ['chart_agg']

#: The return period marked on the density panel, in full weight. The
#: Solvency II standard, and the one number a reader looks for.
CAPITAL_ANCHOR = 200

#: The return periods marked faintly on the Lee panel, either side of the
#: one above: the US capital-adequacy and rating standard, and the round
#: number underneath it. Faint because they are a scale to read the panel
#: against rather than an answer.
LEE_ANCHORS = (100, 250)

#: How far above the aggregate's own peak the severity companion may reach
#: and still set the ordinate window. A severity peaks at its own small
#: losses, and on a long-tailed book that peak is orders of magnitude above
#: anything the aggregate reaches, so scaling to it flattens the subject of
#: the chart into the axis; on a small discrete book the two are within a
#: whisker of each other and clipping the companion would be gratuitous.
#: One rule, reproducing what the compositor did in its two branches.
COMPANION_HEADROOM = 2.0

chart_agg = _emitter_base('agg')


def _updated(agg):
    """Availability: there is a realized grid to draw."""
    return getattr(agg, 'agg_density', None) is not None


@chart_agg.register(Aggregate)
def _agg(agg, xmax=None):
    """Emit the two-panel mass and Lee chart for an aggregate.

    Parameters
    ----------
    agg : Aggregate
        Must be updated.
    xmax : float, optional
        Upper end of the outcome window, in place of the computed one. A
        semantic option rather than a view setting: it is how a gross and a
        net aggregate are read against one common scale, which is a
        comparison the reader is making and the document cannot know about.

    Returns
    -------
    ChartDoc
        Two 'xy' panels over one shared outcome axis: 'density' (the
        probability at each grid point, with the severity alongside) and
        'lee' (the quantile function drawn sideways, interrogated
        probability to loss). The outcome and mass axes declare a log
        reading, which is the compositor's old middle panel; the
        probability axis carries the paired return-period reading.

    Notes
    -----
    The outcome axis is one axis, referenced by the density panel as its x
    and by the Lee panel as its y, which is what makes the two panels one
    reading of one book rather than two pictures that happen to sit side by
    side: a window set on it moves both.

    The Lee curve is trimmed at the saturating top of the quantile function
    (``quantile_curve``), because the grid points past it carry no mass and
    drawing them reads as a loss the book can suffer.
    """
    df = agg.density_df
    x = df.loss.to_numpy(dtype=float)
    mass = df.p_total.to_numpy(dtype=float)
    sdf = agg.sev_density_df
    sev_x = sdf.loss.to_numpy(dtype=float)
    sev_mass = sdf.p_sev.to_numpy(dtype=float)

    window = (loss_window(agg.q, x[0]) if xmax is None
              else (min(0.0, float(x[0])), float(xmax)))
    peak, sev_peak = float(mass.max()), float(sev_mass.max())
    ordinate_top = (max(peak, sev_peak)
                    if sev_peak <= COMPANION_HEADROOM * peak else peak)
    lee_p, lee_x = quantile_curve(x, mass)
    sev_p, sev_lee_x = quantile_curve(sev_x, sev_mass)

    # A loss is read from its upper tail, so a return period is one over the
    # exceedance; a signed position is read from its shortfall, where the
    # probability on the axis already is the one being inverted. The same
    # branch ``Aggregate.tail_periods_df`` takes, from the same fact.
    signed = not agg._is_loss_value
    anchors = agg.tail_periods_df(periods=[CAPITAL_ANCHOR, *LEE_ANCHORS])
    marks = [Mark(panel_id='density', orient='v', at=float(agg.est_m),
                  label='mean', role='mean'),
             Mark(panel_id='density', orient='v',
                  at=float(anchors.loc[CAPITAL_ANCHOR, 'VaR']),
                  label=f'1-in-{CAPITAL_ANCHOR}', role='capital_anchor')]
    marks += [Mark(panel_id='lee', orient='v',
                   at=float(anchors.loc[t, 'p']), label=f'1-in-{t}',
                   role='capital_anchor', faint=True)
              for t in LEE_ANCHORS]

    return complete_tex(ChartDoc(
        name='agg',
        title=str(agg.label),
        axes=(
            ChartAxis(id='outcome', label='Loss', unit='currency',
                      scales=('linear', 'log'), suggested_range=window,
                      full_range=(min(0.0, float(x[0])), float(x[-1]))),
            # The ordinate window is the aggregate's own, widened for a
            # companion that nearly fits and clipping one that dwarfs it
            # (COMPANION_HEADROOM): the panel is about the aggregate. No
            # full_range, because that window *is* the whole extent of the
            # subject and a zoom-out button on it would do nothing.
            ChartAxis(id='mass', label='Probability mass', unit='density',
                      scales=('linear', 'log'),
                      suggested_range=(0.0, ordinate_top)),
            ChartAxis(id='p', label='Non-exceeding probability',
                      unit='probability', suggested_range=(0.0, 1.0)),
            # Not named by any panel: the alternative reading of 'p'. Its
            # window runs from the certain event to the deepest survival
            # worth a panel, past which the curve is a line of float dust.
            ChartAxis(id='return_period', label='Return period',
                      unit='return_period', scale='log',
                      reciprocal_of='p',
                      suggested_range=(1.0,
                                       float(round(1.0 / SURVIVAL_FLOOR)))),
        ),
        panels=(
            Panel(id='density', kind='xy', x_axis='outcome', y_axis='mass',
                  title='Probability mass function'),
            Panel(id='lee', kind='xy', x_axis='p', y_axis='outcome',
                  title='Quantile (Lee) plot'),
        ),
        series=(
            ChartSeries(name='Aggregate', role='density', panel_id='density',
                        x=tuple(float(v) for v in x),
                        y=tuple(float(v) for v in mass)),
            ChartSeries(name='Severity', role='density', panel_id='density',
                        x=tuple(float(v) for v in sev_x),
                        y=tuple(float(v) for v in sev_mass)),
            ChartSeries(name='Aggregate', role='cdf', panel_id='lee',
                        x=tuple(float(v) for v in lee_p),
                        y=tuple(float(v) for v in lee_x)),
            ChartSeries(name='Severity', role='cdf', panel_id='lee',
                        x=tuple(float(v) for v in sev_p),
                        y=tuple(float(v) for v in sev_lee_x)),
        ),
        marks=tuple(marks),
        meta={'ordinate': 'mass',
              'return_period_map': 'reciprocal' if signed else 'complement'},
    ))


register_chart('agg', chart_agg, predicate=_updated, primary=Aggregate)
