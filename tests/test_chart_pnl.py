"""``chart_pnl``: the aggregate's two panels over a signed result.

A P&L's result is a ``GridDistribution`` like any other, so the chart is
the aggregate's and the emitter is a delegation. What is tested here is
the four things that differ, each of them a fact about what the outcome
means: the window is not anchored at zero, the outcome axis offers no log
reading, the adverse tail is the low one and so the return period is a
plain reciprocal, and break even is a mark rather than decoration.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')

from aggregate import build  # noqa: E402
from aggregate.charts import (  # noqa: E402
    available_charts, chart_pnl, human_strings, primary_chart,
)

_PNL = ('pnl CP.Book 1000 premium less agg CP.Loss 100 claims '
        'sev lognorm 5 cv 2 poisson')


@pytest.fixture(scope='module')
def pnl():
    return build(_PNL)


def axes_of(doc):
    return {a.id: a for a in doc.axes}


def test_the_aggregate_s_two_panels_without_the_companion(pnl):
    """One curve: an accounting result is not a compound of a severity."""
    doc = chart_pnl(pnl)
    assert doc.name == 'pnl'
    assert [p.id for p in doc.panels] == ['density', 'lee']
    assert [(s.panel_id, s.role) for s in doc.series] == [
        ('density', 'density'), ('lee', 'cdf')]
    assert len({s.name for s in doc.series}) == 1


def test_the_window_is_not_anchored_at_zero(pnl):
    """A loss window reads from 0 because the mass there is real.

    Half a P&L's outcomes sit on the other side of zero, so its window is
    the two-sided quantile crop instead.
    """
    lo, hi = axes_of(chart_pnl(pnl))['outcome'].suggested_range
    crop = float(pnl.result.q(0.001))
    assert lo < crop < 0                          # the crop, padded below it
    assert abs(lo - crop) < 0.05 * (hi - lo)      # by the pad, not by more


def test_a_signed_axis_offers_no_log_reading(pnl):
    """The declaration doing exactly its job: there is no log of a negative."""
    ax = axes_of(chart_pnl(pnl))
    assert ax['outcome'].scales == ('linear',)
    assert ax['mass'].scales == ('linear', 'log')       # the ordinate still can


def test_the_adverse_tail_is_the_low_one(pnl):
    """So the return period is 1 / p, not 1 / (1 - p).

    The declaration is the whole of it: what the map does to the drawn
    probabilities is the reflect test below.
    """
    assert chart_pnl(pnl).meta['return_period_map'] == 'reciprocal'


def test_reflecting_a_signed_chart_reads_the_upside_tail(pnl):
    """The one picture the switches reach only together.

    A payoff's return period is 1 / p, the shortfall. Reflected, the axis
    is the exceedance and the reading is 1 / (1 - p), the upside, which is
    the return period of doing *well*.
    """
    from aggregate.plots import plt
    ax = axes_of(chart_pnl(pnl))
    assert ax['survival'].complement_of == 'p'
    shortfall = pnl.plot(return_period=True).axes[1]
    upside = pnl.plot(reflect=True, return_period=True).axes[1]
    assert upside.get_xlabel() == shortfall.get_xlabel() == 'Return period'
    # Read against the document's own probabilities: recovering them from
    # the shortfall reading would cancel away the tail this is about.
    p = np.asarray(chart_pnl(pnl).series[1].x_values, dtype=float)
    plain = np.asarray(
        [ln for ln in shortfall.get_lines()
         if len(ln.get_xdata()) > 5][0].get_xdata(), dtype=float)
    drawn = np.asarray(
        [ln for ln in upside.get_lines()
         if len(ln.get_xdata()) > 5][0].get_xdata(), dtype=float)
    # p reaches 1 exactly where the grid saturates, and the reading there
    # is a gap rather than a number.
    seen = p < 1.0
    np.testing.assert_allclose(plain[seen], 1.0 / p[seen])
    np.testing.assert_allclose(drawn[seen], 1.0 / (1.0 - p[seen]))
    assert np.isnan(drawn[~seen]).all()
    plt.close('all')


def test_break_even_is_a_reading_in_both_panels(pnl):
    """Vertical where the outcome is on x, horizontal where it is on y."""
    marks = [m for m in chart_pnl(pnl).marks if m.role == 'break_even']
    assert {(m.panel_id, m.orient, m.at) for m in marks} == {
        ('density', 'v', 0.0), ('lee', 'h', 0.0)}


def test_tex_is_total(pnl):
    doc = chart_pnl(pnl)
    assert not set(human_strings(doc)) - set(doc.tex)


def test_it_is_the_ledger_s_own_picture(pnl):
    assert available_charts(pnl) == ['pnl']
    assert primary_chart(pnl) == 'pnl'


def test_the_readings_render(pnl):
    from aggregate.plots import plt
    plain, rp = pnl.plot(), pnl.plot(return_period=True)
    assert len(plain.axes) == 2
    density, lee = rp.axes
    assert lee.get_xscale() == 'log'
    assert density.get_xscale() == 'linear'            # signed, so untouched
    plt.close('all')


def test_log_acts_on_the_ordinate_alone(pnl):
    from aggregate.plots import plt
    fig = pnl.plot(log=True)
    density = fig.axes[0]
    assert density.get_yscale() == 'log'
    assert density.get_xscale() == 'linear'
    plt.close('all')
