"""The return-period reading, across the classes whose charts offer it.

Once ``quantile_x='return'`` was an argument threaded down to a drawing
worker; since ``[Chart-Aggregate]`` it is a reading the document declares
on its probability axis (``ChartAxis.reciprocal_of``) and the renderer
selects with ``return_period=True``. The transform itself is unit-tested in
``tests/test_chartdoc_readings.py``, against synthetic documents that say
exactly what is under test; what is checked here is that each class that
should offer the reading does, and that the linear default did not move.

The worker these grew from, ``plots/_quantile.py``, is gone: the renderer
owns the transform, the cap and the axis, and the emitters own the curve.
"""

import matplotlib
matplotlib.use('Agg')  # headless

from aggregate import build


def _lee_axis(fig):
    """The panel read probability to loss: titled Lee, or Aggregate."""
    for ax in fig.axes:
        if 'Lee' in ax.get_title() or ax.get_title() == 'Aggregate':
            return ax
    raise AssertionError('no quantile panel found')


def test_aggregate_return():
    a = build('agg RetCont 100 claims sev gamma 100 cv 1 poisson')
    ax = _lee_axis(a.plot(return_period=True))
    assert ax.get_xscale() == 'log'
    assert ax.get_xlabel() == 'Return period'


def test_aggregate_discrete_return():
    a = build('agg RetDice dfreq [3] dsev [1:6]')
    assert _lee_axis(a.plot(return_period=True)).get_xscale() == 'log'


def test_aggregate_linear_default_no_regression():
    a = build('agg RetLin 100 claims sev gamma 100 cv 1 poisson')
    ax = _lee_axis(a.plot())
    assert ax.get_xscale() == 'linear'
    assert ax.get_xlabel() == 'Non-exceeding probability'


def test_severity_return():
    a = build('sev RetSev lognorm 100 cv 2')
    ax = _lee_axis(a.plot(return_period=True))
    assert ax.get_xscale() == 'log'
    assert ax.get_xlabel() == 'Return period'


def test_pnl_return_reads_from_the_shortfall():
    """A payoff is interrogated at T = 1 / p, not 1 / (1 - p)."""
    p = build('pnl RetPnL 1000 premium less agg RetPnLe 100 claims '
              'sev lognorm 5 cv 2 poisson')
    assert p.result.is_loss_value is False
    ax = _lee_axis(p.plot(return_period=True))
    assert ax.get_xscale() == 'log'


def test_reins_occ_return():
    a = build('agg RetReins 100 claims 1000 xs 0 sev gamma 100 cv 1 '
              'occurrence net of 50 xs 50 poisson')
    ax = _lee_axis(a.reins_occ_plot(return_period=True))
    assert ax.get_xscale() == 'log'
    assert ax.get_xlabel() == 'Return period'


def test_the_cap_stands_in_where_no_window_is_declared():
    """The saturating end of a quantile function has to stop somewhere."""
    from aggregate.plots._chartdoc import MAX_RETURN_PERIOD
    a = build('agg RetCap 100 claims sev gamma 100 cv 1 poisson')
    ax = _lee_axis(a.plot(return_period=True))
    assert ax.get_xlim()[1] <= MAX_RETURN_PERIOD
