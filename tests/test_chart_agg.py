"""``chart_agg``: the aggregate's own picture, mass and Lee.

The conversion that changes what ``Aggregate.plot`` draws. Three panels
become two, because the old middle one was the first one read on log and is
now a declared reading of it; the Lee diagram keeps its panel, because
interrogating a probability and reading back a loss is a different question
and not a rescaling.

Shape, semantics and the readings here; the picture itself is gated by the
image diff in ``tests/test_chartdoc_render.py``.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')

from aggregate import build  # noqa: E402
from aggregate.charts import (  # noqa: E402
    available_charts, build_chart_doc, chart_agg, human_strings,
    primary_chart,
)
from aggregate.charts._emit_aggregate import (  # noqa: E402
    CAPITAL_ANCHOR, COMPANION_HEADROOM, LEE_ANCHORS,
)

_CONT = 'agg CA.Cont 100 claims sev lognorm 50 cv 2 poisson'
_DICE = 'agg CA.Dice dfreq [3] dsev [1:6]'


@pytest.fixture(scope='module')
def cont():
    return build(_CONT)


@pytest.fixture(scope='module')
def dice():
    return build(_DICE)


def axes_of(doc):
    return {a.id: a for a in doc.axes}


# ------------------------------------------------------------------ shape

def test_two_panels_and_four_series(cont):
    doc = chart_agg(cont)
    assert doc.name == 'agg'
    assert [p.id for p in doc.panels] == ['density', 'lee']
    assert [(s.name, s.role, s.panel_id) for s in doc.series] == [
        ('Aggregate', 'density', 'density'),
        ('Severity', 'density', 'density'),
        ('Aggregate', 'cdf', 'lee'),
        ('Severity', 'cdf', 'lee'),
    ]


def test_one_outcome_axis_read_two_ways(cont):
    """The density's x and the Lee's y are the same axis, so one window."""
    doc = chart_agg(cont)
    density, lee = doc.panels
    assert density.x_axis == lee.y_axis == 'outcome'
    assert lee.x_axis == 'p'


def test_the_severity_is_a_companion_not_a_second_chart(cont):
    """Same name in both panels: one entity seen twice, toggled together."""
    doc = chart_agg(cont)
    named = [s for s in doc.series if s.name == 'Severity']
    assert {s.panel_id for s in named} == {'density', 'lee'}


# ------------------------------------------------------- declared readings

def test_the_old_log_panel_is_a_reading_of_the_first(cont):
    """Both axes, because it is log x *and* log y that made panel B."""
    ax = axes_of(chart_agg(cont))
    assert ax['outcome'].scales == ('linear', 'log')
    assert ax['mass'].scales == ('linear', 'log')
    assert ax['outcome'].scale == ax['mass'].scale == 'linear'


def test_the_lee_panel_inverts_to_the_distribution_function(cont):
    """The compositor's cdf panel, one click away rather than drawn twice."""
    lee = chart_agg(cont).panels[1]
    assert lee.invertible
    assert lee.inverse_title == 'Distribution function'


def test_the_probability_axis_offers_the_return_period(cont):
    doc = chart_agg(cont)
    ax = axes_of(doc)
    assert ax['return_period'].reciprocal_of == 'p'
    # a loss is interrogated from its upper tail: T = 1 / (1 - p)
    assert doc.meta['return_period_map'] == 'complement'
    # and the paired axis is not drawn, which ChartDoc itself enforces
    assert 'return_period' not in {p.x_axis for p in doc.panels} | \
        {p.y_axis for p in doc.panels}


def test_the_outcome_axis_offers_the_whole_grid(cont):
    ax = axes_of(chart_agg(cont))['outcome']
    assert ax.suggested_range[1] == pytest.approx(cont.q(0.999), rel=0.05)
    assert ax.full_range[1] == pytest.approx(float(cont.xs[-1]))
    assert ax.full_range[1] > ax.suggested_range[1]


def test_the_ordinate_offers_nothing_else(cont):
    """(0, the peak) is already the whole extent, so no zoom-out button."""
    assert axes_of(chart_agg(cont))['mass'].full_range is None


# --------------------------------------------------------------- semantics

def test_the_ordinate_is_a_mass(cont):
    """Not a density: a discretized aggregate IS the distribution here."""
    doc = chart_agg(cont)
    assert doc.meta['ordinate'] == 'mass'
    drawn = doc.series[0].y
    np.testing.assert_allclose(drawn[:50],
                               cont.density_df.p_total.to_numpy()[:50])


def test_a_towering_companion_is_clipped_not_flattened(cont, dice):
    """One rule where the compositor had two branches.

    A severity peaks at its own small losses. On a long-tailed book that is
    orders of magnitude above anything the aggregate reaches, so the
    aggregate sets the window and the companion runs off the top; on a
    small discrete book the two are within a whisker and clipping would be
    gratuitous.
    """
    top = axes_of(chart_agg(cont))['mass'].suggested_range[1]
    assert top == pytest.approx(cont.density_df.p_total.max())
    assert cont.sev_density_df.p_sev.max() > COMPANION_HEADROOM * top

    top = axes_of(chart_agg(dice))['mass'].suggested_range[1]
    assert top == pytest.approx(dice.sev_density_df.p_sev.max())


def test_the_lee_curve_is_trimmed_at_the_saturating_top(dice):
    """Past the top of the support the buckets carry no mass at all.

    Three dice reach 18 on a 32-bucket grid, and drawing the fourteen empty
    buckets above it runs the Lee line flat out to 31, which reads as a
    loss the book can suffer.
    """
    lee = [s for s in chart_agg(dice).series if s.panel_id == 'lee'][0]
    assert len(lee.x_values) < len(dice.density_df)
    assert lee.x_values[-1] == pytest.approx(1.0, abs=1e-9)
    assert lee.y_values[-1] == pytest.approx(18.0)


def test_marks_carry_their_reading(cont):
    doc = chart_agg(cont)
    marks = {(m.panel_id, m.label): m for m in doc.marks}
    assert marks[('density', 'mean')].at == pytest.approx(cont.est_m)
    anchor = marks[('density', f'1-in-{CAPITAL_ANCHOR}')]
    assert anchor.at == pytest.approx(cont.q(1 - 1 / CAPITAL_ANCHOR))
    assert anchor.role == 'capital_anchor' and not anchor.faint
    for t in LEE_ANCHORS:
        faint = marks[('lee', f'1-in-{t}')]
        assert faint.at == pytest.approx(1 - 1 / t)
        assert faint.faint


def test_xmax_is_a_semantic_option(cont):
    """How a gross and a net book are read against one scale."""
    ax = axes_of(chart_agg(cont, xmax=3000.0))['outcome']
    assert ax.suggested_range == (0.0, 3000.0)


def test_tex_is_total(cont):
    doc = chart_agg(cont)
    assert not set(human_strings(doc)) - set(doc.tex)


# -------------------------------------------------------------- capability

def test_it_is_the_aggregate_s_own_picture(cont):
    assert 'agg' in available_charts(cont)
    assert primary_chart(cont) == 'agg'


def test_no_picture_before_update():
    a = build(_CONT.replace('CA.Cont', 'CA.Raw'), update=False)
    assert 'agg' not in available_charts(a)
    assert primary_chart(a) is None
    with pytest.raises(ValueError, match='not available'):
        build_chart_doc(a, 'agg')


# ---------------------------------------------------------------- renderer

def drawn(doc, **kwargs):
    from aggregate.plots import plot_chartdoc
    return plot_chartdoc(doc, **kwargs)


def test_the_lee_panel_steps_the_other_way(dice):
    """A quantile function is the cumulative read sideways, so steps-pre.

    F steps *after* its atom; the same jump seen from the probability axis
    rises *before* it. Reading it off the axes rather than the series role
    is what keeps one rule for every chart.
    """
    from aggregate.plots import plt
    fig = drawn(chart_agg(dice))
    density, lee = fig.axes
    curves = [ln for ln in lee.get_lines() if ln.get_label()[0] != '_']
    assert {ln.get_drawstyle() for ln in curves} == {'steps-pre'}
    assert len(curves) == 2                    # the marks are not curves
    assert density.collections                 # the atoms drew as stems
    plt.close(fig)


def test_the_readings_render(cont):
    from aggregate.plots import plt
    doc = chart_agg(cont)
    plain, log, rp = (drawn(doc), drawn(doc, log=True),
                      drawn(doc, return_period=True))
    assert plain.axes[0].get_xscale() == 'linear'
    assert log.axes[0].get_xscale() == log.axes[0].get_yscale() == 'log'
    # the paired reading re-slices the panel, so the outcome axis follows
    # the data out to the deep tail rather than holding the p window
    assert rp.axes[1].get_xscale() == 'log'
    assert rp.axes[1].get_ylim()[1] > plain.axes[1].get_ylim()[1]
    plt.close('all')
