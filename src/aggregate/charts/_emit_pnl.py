"""P&L chart emitter: the aggregate's two panels over a signed result.

A P&L's result is a :class:`GridDistribution` like any other, so the chart
is the aggregate's, and the emitter is a delegation rather than a second
drawing. Four things differ, and every one of them is a fact about what the
outcome *means* rather than about how it is drawn.

The outcome axis is **signed**, so its window is not anchored at zero: a
loss window reads from 0 because the mass at and near zero is real, and a
P&L's does not because half of its outcomes are on the other side of it.
For the same reason the outcome axis offers no log reading at all, which is
the declaration doing exactly its job.

The adverse tail is the **low** end, so the reader interrogates the
shortfall: the return period is one over the probability on the axis rather
than one over its complement, which is the ``reciprocal`` map, and the same
branch :meth:`PnL.tail_periods_df` already takes.

The break-even at zero is a **reading**, not decoration: it is the line the
whole chart is asked about, so it is a mark in both panels, vertical where
the outcome is on x and horizontal where it is on y.

There is no severity companion. A P&L is an accounting result, not a
compound of a severity, and there is nothing underneath it to draw.

Pure numpy and pandas; no matplotlib.
"""

from .._pnl import PnL
from . import register_chart, _emitter_base
from ._emit_aggregate import outcome_doc
from ._two_panel import loss_window
from .ir import Mark

__all__ = ['chart_pnl']

chart_pnl = _emitter_base('pnl')


def _has_result(pnl):
    """Availability: the ledger has a realized result grid to draw."""
    return getattr(pnl, 'result', None) is not None


@chart_pnl.register(PnL)
def _pnl(pnl):
    """Emit the two-panel mass and Lee chart for a P&L.

    Parameters
    ----------
    pnl : PnL
        Its :attr:`PnL.result` is the exact net result grid.

    Returns
    -------
    ChartDoc
        The same two 'xy' panels an aggregate emits, over one signed
        outcome axis: the mass at each grid point, and the quantile
        function drawn sideways. The mass axis declares a log reading; the
        outcome axis does not, because it is signed.

    Notes
    -----
    The marks are the reading. Break even at zero sits in both panels,
    vertical where the outcome is on x and horizontal where it is on y,
    with the mean beside it on the density panel. Neither is a point on
    the curve a reader can hover for: zero is where the sign of the result
    changes and the mean is a property of the whole distribution.
    """
    gd = pnl.result
    x, mass = gd.x, gd.p
    marks = [Mark(panel_id='density', orient='v', at=0.0,
                  label='break even', role='break_even'),
             Mark(panel_id='lee', orient='h', at=0.0,
                  label='break even', role='break_even'),
             Mark(panel_id='density', orient='v', at=float(gd.mean()),
                  label='mean', role='mean')]
    return outcome_doc(
        'pnl', str(pnl.label), (str(pnl.result_name), x, mass),
        window=loss_window(gd.q, float(x[0])),
        full_range=(float(x[0]), float(x[-1])),
        ordinate_top=float(mass.max()), marks=marks, step=gd.bs,
        outcome_label='P&L', outcome_scales=('linear',),
        is_loss_value=gd.is_loss_value)


register_chart('pnl', chart_pnl, predicate=_has_result, primary=PnL)
