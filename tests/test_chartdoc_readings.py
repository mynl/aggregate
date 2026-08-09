"""The renderer's four switches, against documents that declare readings.

``[Chart-Declared-Readings]``. Which readings a quantity admits is a fact
about the quantity, so the document declares them; which one is on screen
is the caller's, so the renderer switches between them. The rule both sides
follow is that a switch acts on **every** axis or panel declaring the
reading and on no other, so a document that declares nothing draws the same
picture however it is called.

Synthetic documents throughout: these assert what the renderer does with a
declaration, not what any particular chart declares, and a hand-built
document says exactly what is under test. No matplotlib version pin, since
nothing here compares pixels.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')

from aggregate.charts import (  # noqa: E402
    ChartAxis, ChartCapabilityError, ChartDoc, ChartSeries, Mark, Panel,
    SurfaceData,
)


def one_panel(x_axis, y_axis, x, y, marks=(), meta=None, axes=()):
    """A one-panel xy document over the two axes given."""
    return ChartDoc(
        name='t', axes=(x_axis, y_axis) + tuple(axes),
        panels=(Panel(id='p', kind='xy', x_axis=x_axis.id, y_axis=y_axis.id),),
        series=(ChartSeries(name='s', role='density', panel_id='p',
                            x=tuple(x), y=tuple(y), support='continuous'),),
        marks=tuple(marks), meta=dict(meta or {}))


def drawn(doc, **kwargs):
    """``(axes, figure)`` of a rendered document; caller closes the figure."""
    from aggregate.plots import plot_chartdoc
    fig = plot_chartdoc(doc, **kwargs)
    return fig.axes[0], fig


def close(fig):
    from aggregate.plots import plt
    plt.close(fig)


# ------------------------------------------------------------------- log

def test_log_acts_only_where_it_is_declared():
    doc = one_panel(
        ChartAxis(id='loss', label='Loss', suggested_range=(1.0, 100.0),
                  scales=('linear', 'log')),
        ChartAxis(id='dens', label='Density'),
        x=(1.0, 10.0, 100.0), y=(0.5, 0.3, 0.2))
    ax, fig = drawn(doc, log=True)
    assert ax.get_xscale() == 'log'
    assert ax.get_yscale() == 'linear'        # declares one reading: fixed
    close(fig)


def test_a_document_that_declares_nothing_draws_the_same_either_way():
    doc = one_panel(ChartAxis(id='s', label='s', suggested_range=(0.0, 1.0)),
                    ChartAxis(id='g', label='g(s)'),
                    x=(0.0, 0.5, 1.0), y=(0.0, 0.7, 1.0))
    plain, fig1 = drawn(doc)
    limits = (plain.get_xlim(), plain.get_ylim(), plain.get_xscale())
    close(fig1)
    switched, fig2 = drawn(doc, log=True, full_range=True, return_period=True)
    assert (switched.get_xlim(), switched.get_ylim(),
            switched.get_xscale()) == limits
    close(fig2)


def test_log_floors_a_window_that_starts_at_zero():
    """The emitter cannot know a reader will ask for log, so the floor is here.

    A loss window starts at an exact zero, which no log axis can place. The
    renderer drops to the decade under the smallest positive value drawn,
    which crops nothing and keeps the gridlines on powers of ten.
    """
    doc = one_panel(
        ChartAxis(id='loss', label='Loss', suggested_range=(0.0, 100.0),
                  scales=('linear', 'log')),
        ChartAxis(id='dens', label='Density'),
        x=(0.0, 0.04, 100.0), y=(0.5, 0.3, 0.2))
    ax, fig = drawn(doc, log=True)
    assert ax.get_xlim() == (0.01, 100.0)     # decade under 0.04
    close(fig)


# ------------------------------------------------------------- full range

def test_full_range_widens_only_a_declared_axis():
    doc = one_panel(
        ChartAxis(id='loss', label='Loss', suggested_range=(0.0, 10.0),
                  full_range=(0.0, 1000.0)),
        ChartAxis(id='dens', label='Density', suggested_range=(0.0, 1.0)),
        x=(0.0, 5.0, 1000.0), y=(0.5, 0.3, 0.2))
    zoomed, fig1 = drawn(doc)
    assert zoomed.get_xlim()[1] < 20.0
    close(fig1)
    full, fig2 = drawn(doc, full_range=True)
    assert full.get_xlim()[1] > 1000.0        # the extent, plus the margin
    assert full.get_ylim() == zoomed.get_ylim()   # y offers no other reading
    close(fig2)


# ---------------------------------------------------------- return period

def _paired_doc(how, y=None):
    """A tail panel whose probability axis carries a return-period reading."""
    return one_panel(
        ChartAxis(id='loss', label='Loss'),
        ChartAxis(id='p', label='Probability', scale='log',
                  suggested_range=(1e-4, 1.0)),
        x=(1.0, 2.0, 3.0), y=y or (0.5, 0.1, 0.001),
        axes=(ChartAxis(id='rp', label='Return period', scale='log',
                        unit='return_period', reciprocal_of='p'),),
        meta={'return_period_map': how})


def test_reciprocal_map_reads_one_over_the_value():
    doc = _paired_doc('reciprocal')
    ax, fig = drawn(doc, return_period=True)
    line, = ax.get_lines()
    np.testing.assert_allclose(line.get_ydata(), (2.0, 10.0, 1000.0))
    assert ax.get_ylabel() == 'Return period'
    assert ax.get_yscale() == 'log'
    close(fig)


def test_complement_map_reads_one_over_the_exceedance():
    """A non-exceedance axis of a loss: T = 1 / (1 - p), never 1 / p."""
    doc = _paired_doc('complement', y=(0.5, 0.9, 0.999))
    ax, fig = drawn(doc, return_period=True)
    line, = ax.get_lines()
    np.testing.assert_allclose(line.get_ydata(), (2.0, 10.0, 1000.0))
    close(fig)


def test_the_map_defaults_to_the_literal_reciprocal():
    doc = one_panel(
        ChartAxis(id='loss', label='Loss'),
        ChartAxis(id='p', label='Survival', scale='log'),
        x=(1.0, 2.0), y=(0.5, 0.01),
        axes=(ChartAxis(id='rp', label='Return period', scale='log',
                        reciprocal_of='p'),))
    ax, fig = drawn(doc, return_period=True)
    line, = ax.get_lines()
    np.testing.assert_allclose(line.get_ydata(), (2.0, 100.0))
    close(fig)


def test_the_saturating_end_becomes_a_gap():
    """T diverges where the quantile function saturates, so the curve stops."""
    doc = _paired_doc('complement', y=(0.5, 0.9, 1.0))
    ax, fig = drawn(doc, return_period=True)
    line, = ax.get_lines()
    assert np.isnan(line.get_ydata()[-1])
    close(fig)


def test_marks_travel_with_the_axis_they_sit_on():
    doc = one_panel(
        ChartAxis(id='p', label='Probability', suggested_range=(0.0, 1.0)),
        ChartAxis(id='loss', label='Loss'),
        x=(0.5, 0.9, 0.99), y=(1.0, 2.0, 3.0),
        marks=(Mark(panel_id='p', orient='v', at=0.99, label='1-in-100',
                    role='capital_anchor'),),
        axes=(ChartAxis(id='rp', label='Return period', scale='log',
                        reciprocal_of='p'),),
        meta={'return_period_map': 'complement'})
    ax, fig = drawn(doc, return_period=True)
    vline, = [ln for ln in ax.get_lines() if len(set(ln.get_xdata())) == 1]
    assert vline.get_xdata()[0] == pytest.approx(100.0)
    close(fig)


def test_a_panel_with_no_pairing_is_untouched():
    doc = one_panel(ChartAxis(id='loss', label='Loss'),
                    ChartAxis(id='dens', label='Density'),
                    x=(1.0, 2.0), y=(0.5, 0.25))
    ax, fig = drawn(doc, return_period=True)
    line, = ax.get_lines()
    np.testing.assert_allclose(line.get_xdata(), (1.0, 2.0))
    close(fig)


# ---------------------------------------------------------- panel kinds

def grid_doc(kinds):
    surf = SurfaceData(x=(1.0, 2.0), y=(1.0, 2.0),
                       z=((0.1, 0.2), (0.3, 0.4)))
    return ChartDoc(
        name='joint', title='Joint',
        axes=(ChartAxis(id='x0', label='A'), ChartAxis(id='x1', label='B'),
              ChartAxis(id='z', label='density', scales=('linear', 'log'))),
        panels=(Panel(id='g', kind='surface', x_axis='x0', y_axis='x1',
                      z_axis='z', kinds=kinds),),
        series=(ChartSeries(name='joint density', role='joint', panel_id='g',
                            surface=surf),))


def test_a_declared_realization_beats_a_confession():
    """With a native kind on offer the renderer takes it, and says nothing."""
    ax, fig = drawn(grid_doc(('surface', 'heatmap')))
    assert not ax.get_title().endswith('(projection)')
    close(fig)


def test_a_declared_realization_satisfies_strict():
    ax, fig = drawn(grid_doc(('surface', 'heatmap')), strict=True)
    close(fig)
    with pytest.raises(ChartCapabilityError, match='projection'):
        drawn(grid_doc(('surface',)), strict=True)


def test_an_undeclared_kind_is_refused_by_name():
    with pytest.raises(ChartCapabilityError, match='does not declare'):
        drawn(grid_doc(('surface',)), kind='heatmap')
