"""The return-period reading, across the classes whose charts offer it.

Once ``quantile_x='return'`` was an argument threaded down to a drawing
worker; since ``[Chart-Aggregate]`` it is a reading the document declares
on its probability axis (``ChartAxis.reciprocal_of``) and the renderer
selects with ``return_period=True``. The transform itself is unit-tested in
``tests/test_chartdoc_readings.py``, against synthetic documents that say
exactly what is under test; what is checked here is that each class that
should offer the reading does, and that the linear default did not move.

Since ``[Chart-2D-Punchups]`` the ladder itself is linear too. The axis
declares both scales and both windows and lets the reader choose, so the
opening view is 1 to :data:`RETURN_PERIOD_TOP` read linearly, and ``log``
and ``full_range`` are the two presses that recover the nine-decade log
picture this reading used to open with. The renderer pads a linear window
by a few percent either side and leaves a log one exact, which is why the
assertions below are shaped the way they are.

The worker these grew from, ``plots/_quantile.py``, is gone: the renderer
owns the transform, the cap and the axis, and the emitters own the curve.
"""

import matplotlib
matplotlib.use('Agg')  # headless

import pytest

from aggregate import build
from aggregate.charts._two_panel import RETURN_PERIOD_TOP, SURVIVAL_FLOOR


def _lee_axis(fig):
    """The panel read probability to loss: titled Lee, or Aggregate."""
    for ax in fig.axes:
        if 'Lee' in ax.get_title() or ax.get_title() == 'Aggregate':
            return ax
    raise AssertionError('no quantile panel found')


def test_aggregate_return():
    a = build('agg RetCont 100 claims sev gamma 100 cv 1 poisson')
    ax = _lee_axis(a.plot(return_period=True))
    assert ax.get_xscale() == 'linear'
    assert ax.get_xlabel() == 'Return period'
    assert ax.get_xlim()[1] == pytest.approx(RETURN_PERIOD_TOP, rel=0.1)


def test_aggregate_discrete_return():
    a = build('agg RetDice dfreq [3] dsev [1:6]')
    assert _lee_axis(a.plot(return_period=True)).get_xscale() == 'linear'


def test_the_reader_chooses_the_log_reading():
    """The axis offers log; it no longer decides for the reader.

    Both presses together are the picture the reading used to open with,
    a log axis over the whole declared extent, which is the point of
    declaring two scales and two windows rather than one of each.
    """
    a = build('agg RetLog 100 claims sev gamma 100 cv 1 poisson')
    ax = _lee_axis(a.plot(return_period=True, log=True))
    assert ax.get_xscale() == 'log'
    assert ax.get_xlim() == (1.0, RETURN_PERIOD_TOP)
    deep = _lee_axis(a.plot(return_period=True, log=True, full_range=True))
    assert deep.get_xlim() == (1.0, round(1.0 / SURVIVAL_FLOOR))


def test_aggregate_linear_default_no_regression():
    a = build('agg RetLin 100 claims sev gamma 100 cv 1 poisson')
    ax = _lee_axis(a.plot())
    assert ax.get_xscale() == 'linear'
    assert ax.get_xlabel() == 'Non-exceeding probability'


def test_severity_return():
    a = build('sev RetSev lognorm 100 cv 2')
    ax = _lee_axis(a.plot(return_period=True))
    assert ax.get_xscale() == 'linear'
    assert ax.get_xlabel() == 'Return period'
    assert _lee_axis(a.plot(return_period=True, log=True)).get_xscale() == 'log'


def test_pnl_return_reads_from_the_shortfall():
    """A payoff is interrogated at T = 1 / p, not 1 / (1 - p)."""
    p = build('pnl RetPnL 1000 premium less agg RetPnLe 100 claims '
              'sev lognorm 5 cv 2 poisson')
    assert p.result.is_loss_value is False
    ax = _lee_axis(p.plot(return_period=True))
    assert ax.get_xscale() == 'linear'
    assert ax.get_xlabel() == 'Return period'


def test_reins_occ_return():
    a = build('agg RetReins 100 claims 1000 xs 0 sev gamma 100 cv 1 '
              'occurrence net of 50 xs 50 poisson')
    ax = _lee_axis(a.reins_occ_plot(return_period=True))
    assert ax.get_xscale() == 'linear'
    assert ax.get_xlabel() == 'Return period'
    assert _lee_axis(a.reins_occ_plot(return_period=True,
                                      log=True)).get_xscale() == 'log'


def test_the_cap_stands_in_where_no_window_is_declared():
    """The saturating end of a quantile function has to stop somewhere.

    Every shipped emitter now declares both windows, so the renderer's
    own cap is a backstop for a document that declares neither rather
    than the mechanism. What is checked here is that the drawn axis stays
    under it, which the declared ladder achieves by two orders of
    magnitude and the full extent meets exactly.
    """
    from aggregate.plots._chartdoc import MAX_RETURN_PERIOD
    a = build('agg RetCap 100 claims sev gamma 100 cv 1 poisson')
    ax = _lee_axis(a.plot(return_period=True))
    assert ax.get_xlim()[1] <= MAX_RETURN_PERIOD
    full = _lee_axis(a.plot(return_period=True, log=True, full_range=True))
    assert full.get_xlim()[1] == MAX_RETURN_PERIOD
