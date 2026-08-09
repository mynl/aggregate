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

from .._aggregate import Aggregate
from . import register_chart, _emitter_base
from ._payload import collapse_empty_runs, lattice_payload
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


def outcome_doc(name, title, subject, companion=None, *, window, full_range,
                ordinate_top, marks=(), outcome_label='Loss', step=None,
                outcome_scales=('linear', 'log'), is_loss_value=True):
    """The mass-and-Lee document, shared by the aggregate and the P&L.

    A P&L is an aggregate's shape over a different grid: the same two
    panels, the same one outcome axis read by both, the same quantile
    curve. What differs is what the outcome *means*, so the differences
    arrive as arguments rather than as a second copy of this.

    Parameters
    ----------
    name, title : str
        Registry name and the document's heading.
    subject : tuple
        ``(series name, outcome grid, mass)``, the chart's own subject.
    companion : tuple, optional
        A second curve in the same form, drawn alongside on both panels
        (an aggregate's severity). A P&L has none: it is an accounting
        result, not a compound of anything.
    window, full_range : tuple of float
        The outcome axis' suggested and full extents.
    ordinate_top : float
        Upper end of the mass axis.
    marks : iterable of Mark
    outcome_label : str
        What the outcome axis is called ('Loss', 'P&L').
    step : float, optional
        The grid spacing (a ``bs``), which lets the outcome coordinates go
        out as a lattice instead of as a list of evenly spaced numbers
        repeated once per series. Checked, not assumed: a run of empty
        buckets collapsed out of a density, or a quantile curve trimmed to
        its support, falls back to the explicit form on its own.
    outcome_scales : tuple of str
        The scales the outcome axis may be read on. A signed axis declares
        ``('linear',)``: half its values are negative and no log reading of
        it exists, which is the declaration doing exactly its job.
    is_loss_value : bool
        True where the adverse tail is the high one. It picks the
        return-period map, and it is the same fact ``tail_periods_df``
        branches on: a loss is interrogated at ``T = 1 / (1 - p)``, a
        payoff at its shortfall, ``T = 1 / p``.

    Returns
    -------
    ChartDoc
    """
    series = []
    for label, x, mass in filter(None, (subject, companion)):
        drawn_x, drawn_mass = collapse_empty_runs(x, mass)
        series.append(ChartSeries(
            name=label, role='density', panel_id='density',
            y=tuple(float(v) for v in drawn_mass),
            **lattice_payload(drawn_x, step)))
    for label, x, mass in filter(None, (subject, companion)):
        p, outcome = quantile_curve(x, mass)
        series.append(ChartSeries(
            name=label, role='cdf', panel_id='lee',
            x=tuple(float(v) for v in p),
            **lattice_payload(outcome, step, 'y')))
    return complete_tex(ChartDoc(
        name=name,
        title=title,
        axes=(
            ChartAxis(id='outcome', label=outcome_label, unit='currency',
                      scales=outcome_scales, suggested_range=window,
                      full_range=full_range),
            # No full_range on the ordinate: (0, the peak) already IS the
            # whole extent, and a zoom-out button on it would do nothing.
            ChartAxis(id='mass', label='Probability mass', unit='density',
                      scales=('linear', 'log'),
                      suggested_range=(0.0, float(ordinate_top))),
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
        series=tuple(series),
        marks=tuple(marks),
        meta={'ordinate': 'mass',
              'return_period_map': ('complement' if is_loss_value
                                    else 'reciprocal')},
    ))


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

    return outcome_doc(
        'agg', str(agg.label),
        ('Aggregate', x, mass), ('Severity', sev_x, sev_mass),
        window=window, full_range=(min(0.0, float(x[0])), float(x[-1])),
        ordinate_top=ordinate_top, marks=marks, step=agg.bs,
        is_loss_value=agg._is_loss_value)


register_chart('agg', chart_agg, predicate=_updated, primary=Aggregate)
