"""[Chart-Severity] the severity's own picture: its density, and its Lee.

Four panels became two, and neither collapse loses anything. The density
and the log density were one quantity read two ways, so log is a declared
reading of the one panel. The distribution and the Lee diagram are
**inverses**, the same curve with its axes exchanged, so only one of them
carries information the other does not; the Lee orientation is kept,
because it is the one that pairs with a return-period reading.

A severity is also the one genuinely continuous law in this library, so its
series says so, and only a law with no density at all is atomic.

DecL programs mirror the ``CH.Sev*`` entries in
``src/aggregate/agg/decl-testers.agg``.
"""

import matplotlib
import numpy as np
import pytest

# Renderer cases draw to memory, never to a screen.
matplotlib.use('Agg')

from aggregate import build  # noqa: E402
from aggregate.charts import (  # noqa: E402
    available_charts, canonical_json, chart_severity, doc_hash, human_strings,
    primary_chart,
)
from aggregate.charts._emit_severity import GRID_POINTS  # noqa: E402

_LN = 'sev CH.SevLN lognorm 100 cv 2'
_BOUNDED = 'sev CH.SevB 500 * beta 2 3'
_DISCRETE = 'sev CH.SevD dhistogram xps [1 10 40] [.5 .3 .2]'


@pytest.fixture(scope='module')
def sev():
    return build(_LN)


def axes_of(doc):
    return {a.id: a for a in doc.axes}


def test_capability(sev):
    assert available_charts(sev) == ['severity']
    assert primary_chart(sev) == 'severity'


def test_two_panels_over_one_loss_axis(sev):
    """The density's x and the Lee's y are the same axis, so one window."""
    doc = chart_severity(sev)
    density, lee = doc.panels
    assert density.x_axis == lee.y_axis == 'loss'
    assert lee.x_axis == 'p'
    assert doc.marks == ()      # no mean, no anchors: not severity questions


def test_the_log_density_panel_is_a_reading_of_the_density(sev):
    ax = axes_of(chart_severity(sev))
    assert ax['loss'].scales == ('linear', 'log')
    assert ax['pdf'].scales == ('linear', 'log')


def test_the_distribution_panel_is_the_lee_panel_transposed(sev):
    """Inverses, so one of them is drawn and the other is that one read back.

    The curve kept is ``(F, loss)``. Its transpose is ``(loss, F)``, which
    is the distribution panel the compositor drew as a fourth picture, so
    nothing is lost by keeping one: the same pairs, read the other way.
    """
    doc = chart_severity(sev)
    lee = [s for s in doc.series if s.panel_id == 'lee'][0]
    p, loss = np.array(lee.x_values), np.array(lee.y_values)
    assert np.allclose(sev.cdf(loss), p, rtol=1e-8)
    assert (np.diff(loss) > 0).all()


def test_the_probability_axis_offers_the_return_period(sev):
    doc = chart_severity(sev)
    assert axes_of(doc)['return_period'].reciprocal_of == 'p'
    assert doc.meta['return_period_map'] == 'complement'   # a severity is a loss


def test_a_severity_is_continuous_unless_it_has_no_density(sev):
    assert {s.support for s in chart_severity(sev).series} == {'continuous'}
    assert {s.support for s in chart_severity(build(_DISCRETE)).series} == \
        {'atomic'}


def test_grid_is_quantile_spaced_not_linear(sev):
    """The grid is the point of this emitter: a linear grid over an
    unbounded heavy tail either truncates it or spends every point on it."""
    x = np.array(chart_severity(sev).series[0].x_values)
    assert len(x) == GRID_POINTS
    assert (np.diff(x) > 0).all()
    steps = np.diff(x)
    # quantile spacing means the step grows by orders of magnitude into the
    # tail; a linear grid would have a constant step
    assert steps.max() / steps.min() > 100


def test_ordinate_is_a_pdf_for_a_law_that_has_one(sev):
    doc = chart_severity(sev)
    assert doc.meta['ordinate'] == 'pdf'
    assert axes_of(doc)['pdf'].label == 'pdf'
    y = np.array(doc.series[0].y_values)
    assert (y >= 0).all() and y.max() > 0
    # a density ordinate is not a mass and must never be summable to one
    assert y.sum() > 1.0


def test_discrete_severity_reads_as_mass_not_a_flat_zero():
    """A discrete law has no density; a pdf panel would draw a flat line
    along the axis and call it a distribution."""
    sev = build(_DISCRETE)
    doc = chart_severity(sev)
    assert doc.meta['ordinate'] == 'mass'
    assert axes_of(doc)['pdf'].label == 'Probability mass'
    x = np.array(doc.series[0].x_values)
    y = np.array(doc.series[0].y_values)
    assert np.array_equal(x, [1.0, 10.0, 40.0])
    assert np.allclose(y, [0.5, 0.3, 0.2])      # the atoms, exactly
    assert y.sum() == pytest.approx(1.0)
    # the pdf really is identically zero there, which is what triggers it
    assert not np.any(sev.pdf(x) > 0)


def test_bounded_severity_grid_stays_monotone():
    """A bounded law repeats quantiles at its ceiling; unique() drops the
    duplicates rather than emitting a flat step or an inverted axis."""
    doc = chart_severity(build(_BOUNDED))
    x = np.array(doc.series[0].x_values)
    assert (np.diff(x) > 0).all()
    assert x[-1] <= 500.0


def test_windows(sev):
    ax = axes_of(chart_severity(sev))
    lo, hi = ax['loss'].suggested_range
    assert lo < 0                       # unsigned severity reads from zero, padded
    assert hi > float(sev.isf(0.001))   # the 0.1% quantile, padded
    assert ax['loss'].full_range[1] > hi     # out to the 1-in-100,000 loss
    assert ax['p'].suggested_range == (0.0, 1.0)


def test_tex_is_total(sev):
    doc = chart_severity(sev)
    assert not set(human_strings(doc)) - set(doc.tex)


def test_deterministic(sev):
    assert canonical_json(chart_severity(sev)) == \
        canonical_json(chart_severity(sev))
    assert doc_hash(chart_severity(sev, n=128)) != doc_hash(chart_severity(sev))


def test_the_readings_render(sev):
    from aggregate.plots import plt
    plain, log, rp = (sev.plot(), sev.plot(log=True),
                      sev.plot(return_period=True))
    assert len(plain.axes) == 2
    assert plain.axes[0].get_xscale() == 'linear'
    assert log.axes[0].get_xscale() == log.axes[0].get_yscale() == 'log'
    assert rp.axes[1].get_xscale() == 'log'
    assert rp.axes[1].get_xlabel() == 'Return period'
    # continuous: a line, never the atomic ladder's steps
    assert {ln.get_drawstyle() for ln in plain.axes[0].get_lines()} == \
        {'default'}
    plt.close('all')
