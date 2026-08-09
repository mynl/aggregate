"""Severity chart emitter: the density and the Lee diagram, on a quantile grid.

A :class:`~aggregate.distributions.Severity` is a look-through onto a
frozen scipy variable rather than a compute result, so it carries no
``density_df`` and there is no grid to read: the emitter synthesizes one,
absorbing the algorithm the app's server had been carrying.

The grid inverts the survival function over log-spaced exceedance
probabilities rather than walking loss linearly. A severity is routinely
heavy tailed and its support often unbounded, so a linear grid either
truncates the tail or spends nearly every point on it; quantile spacing
puts the points where the probability is. That choice is meaning, which is
why it belongs in an emitter and not in a renderer.

**Two panels, and the compositor's four collapse into them.** Its density
and log density are one quantity read two ways, so the density panel
declares both scales and the reader picks. Its distribution and its Lee
diagram are *inverses*, the same curve with the axes exchanged, so they
carry no different information and only one of them needs drawing; the Lee
orientation is the one kept, because it is the one that pairs with a
return-period reading, which is how a tail is actually quoted.

The ordinate is a **pdf**, not a mass. An aggregate's ``p_total`` is
probability per bucket and sums to one; this does not, and must never be
summed. The axis says ``pdf`` for that reason, and the series says
``support='continuous'``: a severity is the one place this library holds a
genuinely continuous law, because discretization happens in ``Aggregate``
and not here.

A discrete severity is the exception, and it has to be: it has no density
at all, so its pdf is identically zero and a pdf panel would draw a flat
line along the axis and call it a distribution. Where that happens the
emitter draws the probability mass instead, says so on the axis, records
which reading it gave in ``meta['ordinate']``, and the series is atomic.
The two are never mixed in one document.

Pure numpy and pandas; no matplotlib.
"""

import numpy as np

from .._severity import Severity
from . import register_chart, _emitter_base
from ._two_panel import SURVIVAL_FLOOR, pad_window
from .ir import ChartAxis, ChartDoc, ChartSeries, Panel, complete_tex

__all__ = ['chart_severity']

#: Grid points. Smooth at any plot width and trivial to serialize.
GRID_POINTS = 512

#: The grid runs between these exceedance probabilities, dense near the
#: median and still resolving the 1-in-100,000 tail without a huge grid.
GRID_P_EDGE = 1e-5

chart_severity = _emitter_base('severity')


def _quantile_grid(sev, n):
    """Loss points from inverting the survival at log-spaced probabilities.

    Two half runs meeting at the median, so both halves of the
    distribution are resolved: a single run from 1 to 1e-5 would crowd
    every point into the tail and leave the body a straight line.
    """
    ps = np.concatenate([
        np.logspace(np.log10(1 - GRID_P_EDGE), np.log10(0.5),
                    n // 2, endpoint=False),
        np.logspace(np.log10(0.5), np.log10(GRID_P_EDGE), n - n // 2),
    ])
    loss = np.asarray(sev.isf(ps), dtype=float)
    # A bounded or discrete severity repeats and can invert; unique() keeps
    # the grid monotone and drops the duplicates a flat stretch produces,
    # so no consumer has to defend against a bad axis.
    return np.unique(loss[np.isfinite(loss)])


@chart_severity.register(Severity)
def _severity(sev, n=GRID_POINTS):
    """Emit the two-panel density and tail chart for a severity.

    Parameters
    ----------
    sev : Severity
    n : int, default 512
        Grid points. The grid is quantile spaced, so this buys resolution
        in probability rather than in loss.

    Returns
    -------
    ChartDoc
        Two 'xy' panels over one loss axis: 'density' (the pdf ordinate, or
        the probability mass for a law that has no density, per
        ``meta['ordinate']``), both of whose axes declare a log reading;
        and 'lee' (the quantile function drawn sideways), whose probability
        axis carries the paired return-period reading.

    Notes
    -----
    The Lee curve costs nothing to build here and is exact: the grid is
    *already* a quantile grid, inverted from log-spaced exceedance
    probabilities, so the curve is the pair the grid was computed from
    rather than an accumulation of it.

    No marks: a severity chart carries no mean line and no capital anchors,
    because neither is a severity question (the app draws none either, and
    the inventory records the omission as deliberate).
    """
    loss = _quantile_grid(sev, n)
    if loss.size == 0:
        raise ValueError(f'severity {sev.label!r} produced no finite '
                         'quantiles to draw')
    with np.errstate(divide='ignore', invalid='ignore'):
        pdf = np.asarray(sev.pdf(loss), dtype=float)
        cdf = np.asarray(sev.cdf(loss), dtype=float)
    # No density anywhere on the grid means the law has none: read the
    # jumps of the step cdf, which are exactly the atoms. Tested on the
    # symptom rather than on the severity's kind, so a wrapper around a
    # discrete law is caught as surely as the discrete law itself.
    mass_reading = not np.any(pdf > 0)
    ordinate = np.diff(cdf, prepend=0.0) if mass_reading else pdf
    y_label = 'Probability mass' if mass_reading else 'pdf'
    # A severity is the one place this library holds a genuinely continuous
    # law: build one and it is a frozen scipy variable, with no ``xs`` and
    # no discrete density, because discretization happens in Aggregate and
    # not here. So a continuous severity says so, and only a discrete one
    # is atomic.
    support = 'atomic' if mass_reading else 'continuous'
    xs = tuple(float(v) for v in loss)
    name = str(sev.label)

    # An unsigned severity is read from zero: starting the axis at the
    # 0.1% quantile would hide the mass at and near zero that a layered or
    # spliced severity routinely has.
    lo = min(0.0, float(loss[0]))
    hi = float(sev.isf(0.001))

    return complete_tex(ChartDoc(
        name='severity',
        title=name,
        axes=(
            ChartAxis(id='loss', label='Loss', unit='currency',
                      scales=('linear', 'log'),
                      suggested_range=pad_window(lo, hi),
                      full_range=(lo, float(loss[-1]))),
            ChartAxis(id='pdf', label=y_label, unit='density',
                      scales=('linear', 'log')),
            ChartAxis(id='p', label='Non-exceeding probability',
                      unit='probability', suggested_range=(0.0, 1.0)),
            # Not named by any panel: the alternative reading of 'p'.
            ChartAxis(id='return_period', label='Return period',
                      unit='return_period', scale='log', reciprocal_of='p',
                      suggested_range=(1.0,
                                       float(round(1.0 / SURVIVAL_FLOOR)))),
        ),
        panels=(
            Panel(id='density', kind='xy', x_axis='loss', y_axis='pdf',
                  title='Severity density'),
            # Inverting it gives the distribution function, which is the
            # fourth panel the compositor drew as a picture of its own.
            Panel(id='lee', kind='xy', x_axis='p', y_axis='loss',
                  invertible=True, title='Quantile (Lee) plot',
                  inverse_title='Distribution function'),
        ),
        series=(
            ChartSeries(name=name, role='density', panel_id='density',
                        x=xs, y=tuple(float(v) for v in ordinate),
                        support=support),
            ChartSeries(name=name, role='cdf', panel_id='lee',
                        x=tuple(float(v) for v in cdf), y=xs,
                        support=support),
        ),
        meta={'ordinate': 'mass' if mass_reading else 'pdf',
              'return_period_map': 'complement'},
    ))


register_chart('severity', chart_severity, primary=Severity)
