"""[Chart-Conversions] severity: the emitter's document contract.

The severity chart has no image gate. A ``plots/`` compositor exists
(``plot_severity``) but draws a *different* chart, four panels including a
Lee diagram, so there is nothing to compare pixel for pixel; the before
side is the app's server-synthesized frame, and that comparison belongs to
the paired app commit.

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
    available_charts, canonical_json, chart_severity, doc_hash,
)
from aggregate.charts._emit_severity import GRID_POINTS  # noqa: E402
from aggregate.constants import LOG_FLOOR  # noqa: E402

_LN = 'sev CH.SevLN lognorm 100 cv 2'
_BOUNDED = 'sev CH.SevB 500 * beta 2 3'
_DISCRETE = 'sev CH.SevD dhistogram xps [1 10 40] [.5 .3 .2]'


@pytest.fixture(scope='module')
def sev():
    return build(_LN)


def test_capability(sev):
    assert available_charts(sev) == ['severity']


def test_two_panels_share_one_loss_axis(sev):
    doc = chart_severity(sev)
    density, tail = doc.panels
    assert density.x_axis == tail.x_axis == 'loss'
    assert (density.read_axis, tail.read_axis) == ('x', 'y')
    assert {a.id: a for a in doc.axes}['survival'].scale == 'log'
    assert doc.marks == ()          # no mean, no anchors: not severity questions


def test_grid_is_quantile_spaced_not_linear(sev):
    """The grid is the point of this emitter: a linear grid over an
    unbounded heavy tail either truncates it or spends every point on it."""
    x = np.array(chart_severity(sev).series[0].x)
    assert len(x) == GRID_POINTS
    assert (np.diff(x) > 0).all()
    steps = np.diff(x)
    # quantile spacing means the step grows by orders of magnitude into the
    # tail; a linear grid would have a constant step
    assert steps.max() / steps.min() > 100


def test_grid_inverts_the_survival(sev):
    """Each grid point is the loss at its own exceedance probability."""
    doc = chart_severity(sev)
    x = np.array(doc.series[0].x)
    s = np.array([np.nan if v is None else v for v in doc.series[1].y])
    keep = ~np.isnan(s)
    assert np.allclose(sev.isf(s[keep]), x[keep], rtol=1e-8)


def test_ordinate_is_a_pdf_for_a_law_that_has_one(sev):
    doc = chart_severity(sev)
    assert doc.meta['ordinate'] == 'pdf'
    assert {a.id: a for a in doc.axes}['pdf'].label == 'pdf'
    y = np.array(doc.series[0].y)
    assert (y >= 0).all() and y.max() > 0
    # a density ordinate is not a mass and must never be summable to one
    assert y.sum() > 1.0


def test_discrete_severity_reads_as_mass_not_a_flat_zero():
    """A discrete law has no density; a pdf panel would draw a flat line
    along the axis and call it a distribution."""
    sev = build(_DISCRETE)
    doc = chart_severity(sev)
    assert doc.meta['ordinate'] == 'mass'
    assert {a.id: a for a in doc.axes}['pdf'].label == 'Probability mass'
    x = np.array(doc.series[0].x)
    y = np.array(doc.series[0].y)
    assert np.array_equal(x, [1.0, 10.0, 40.0])
    assert np.allclose(y, [0.5, 0.3, 0.2])      # the atoms, exactly
    assert y.sum() == pytest.approx(1.0)
    # the pdf really is identically zero there, which is what triggers it
    assert not np.any(sev.pdf(x) > 0)


def test_bounded_severity_grid_stays_monotone():
    """A bounded law repeats quantiles at its ceiling; unique() drops the
    duplicates rather than emitting a flat step or an inverted axis."""
    doc = chart_severity(build(_BOUNDED))
    x = np.array(doc.series[0].x)
    assert (np.diff(x) > 0).all()
    assert x[-1] <= 500.0


def test_dust_is_a_gap_not_a_zero():
    for series in (chart_severity(build(_DISCRETE)).series[1],
                   chart_severity(build(_BOUNDED)).series[1]):
        assert all(v is None or v > LOG_FLOOR for v in series.y)
    # the discrete law's top atom has exactly zero survival above it
    assert chart_severity(build(_DISCRETE)).series[1].y[-1] is None


def test_windows(sev):
    axes = {a.id: a for a in chart_severity(sev).axes}
    lo, hi = axes['loss'].suggested_range
    assert lo < 0                       # unsigned severity reads from zero, padded
    assert hi > float(sev.isf(0.001))   # the 0.1% quantile, padded
    s_lo, s_hi = axes['survival'].suggested_range
    assert s_hi == 1.0
    assert np.log10(s_lo) == int(np.log10(s_lo))


def test_deterministic(sev):
    assert canonical_json(chart_severity(sev)) == \
        canonical_json(chart_severity(sev))
    assert doc_hash(chart_severity(sev, n=128)) != doc_hash(chart_severity(sev))


def test_renderer_draws_two_panels_sharing_x(sev):
    from aggregate.plots import plot_chartdoc, plt
    fig = plot_chartdoc(chart_severity(sev))
    assert len(fig.axes) == 2
    assert fig.axes[0].get_xlim() == fig.axes[1].get_xlim()
    assert fig.axes[1].get_yscale() == 'log'
    plt.close('all')
