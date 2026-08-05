"""[Chart-Conversions] reinsurance triple: the emitter's document contract.

Structural, semantic and determinism checks on ``charts.chart_reins``, plus
the renderer's multi-panel realization. There is no ``plots/`` compositor
for this chart, so there is no image gate: the before side is the app's
client-side builder, and that comparison is the paired app commit's
fixture diff.

The DecL programs are the ``RR.*`` reinsurance cases from
``src/aggregate/agg/decl-testers.agg`` (one per stage combination), used
verbatim so the two stay in sync without a new corpus entry.
"""

import matplotlib
import numpy as np
import pytest

# Renderer cases draw to memory, never to a screen.
matplotlib.use('Agg')

from aggregate import build  # noqa: E402
from aggregate.charts import (  # noqa: E402
    available_charts, canonical_json, chart_reins, doc_hash,
)
from aggregate.charts._emit_reins import SURVIVAL_FLOOR  # noqa: E402
from aggregate.constants import LOG_FLOOR  # noqa: E402

_OCC = ('agg RR.Occ 10 claims sev lognorm 100 cv 2 '
        'occurrence net of 0.8 so 250 xs 175 poisson')
_AGG = ('agg RR.Agg 10 claims sev lognorm 100 cv 2 poisson '
        'aggregate net of 500 xs 750')
_BOTH = ('agg RR.Both 100 claims 5000 xs 0 sev lognorm 50 cv 1.5 '
         'occurrence net of 3500 po 4000 xs 1000 poisson '
         'aggregate net of 2000 xs 3000')
_GROSS = 'agg RR.Gross 10 claims sev lognorm 100 cv 2 poisson'


@pytest.fixture(scope='module')
def both():
    return build(_BOTH)


# ----------------------------------------------------------- capability

def test_available_only_with_a_cession(both):
    assert available_charts(both) == ['reins']
    assert available_charts(build(_GROSS)) == []


def test_default_basis_is_a_stage_that_cedes():
    """The default is never a triple with nothing in it."""
    assert chart_reins(build(_OCC)).meta['basis'] == 'occ'
    assert chart_reins(build(_AGG)).meta['basis'] == 'agg'


def test_basis_must_exist():
    with pytest.raises(ValueError, match='unknown reinsurance basis'):
        chart_reins(build(_OCC), basis='nope')
    with pytest.raises(ValueError, match='no cession on'):
        chart_reins(build(_AGG), basis='occ')
    with pytest.raises(ValueError, match='no cession on'):
        chart_reins(build(_OCC), basis='agg')


# ------------------------------------------------------------ structure

def test_two_panels_share_one_loss_axis(both):
    doc = chart_reins(both)
    density, tail = doc.panels
    assert (density.id, tail.id) == ('density', 'tail')
    # One axis id referenced twice IS the shared window.
    assert density.x_axis == tail.x_axis == 'loss'
    assert density.y_axis != tail.y_axis
    # At a chosen survival the answer wanted is the loss, not the reverse.
    assert (density.read_axis, tail.read_axis) == ('x', 'y')
    survival = {a.id: a for a in doc.axes}['survival']
    assert survival.scale == 'log'


def test_series_are_the_triple_twice_in_draw_order(both):
    doc = chart_reins(both)
    assert [s.panel_id for s in doc.series] == \
        ['density'] * 3 + ['tail'] * 3
    # net last so it draws on top: what did I keep must not hide under the
    # subject it came from
    assert [s.role for s in doc.series[:3]] == ['gross', 'ceded', 'net']
    assert [s.name for s in doc.series[:3]] == ['Gross', 'Ceded', 'Net']
    # the same entity seen twice carries the same name in both panels
    assert [s.name for s in doc.series[3:]] == ['Gross', 'Ceded', 'Net']


def test_subject_is_never_relabeled_gross(both):
    """On the aggregate triple the first series is the cover's subject.

    It equals true gross only with no occurrence program underneath, and
    this book has one.
    """
    doc = chart_reins(both, basis='agg')
    assert doc.series[0].role == 'subject'
    assert doc.series[0].name == 'Subject'
    assert 'gross' not in [s.role for s in doc.series]


def test_no_marks(both):
    """No tail_df and no mean on this chart, per the inventory."""
    assert chart_reins(both).marks == ()


# ------------------------------------------------------------- numerics

def test_survival_is_exactly_one_minus_cumsum(both):
    """The app accumulates client side; the emitter routes through the
    GridDistribution. They must agree to the last bit, or the conversion
    would silently move the curve."""
    doc = chart_reins(both)
    df = both.reins_density_df
    for series, column in zip(doc.series[3:],
                              ('p_agg_gross', 'p_agg_ceded_occ',
                               'p_agg_net_occ')):
        app = np.maximum(0.0, 1.0 - np.cumsum(df[column].to_numpy()))
        got = np.array([np.nan if v is None else v for v in series.y])
        keep = ~np.isnan(got)
        assert np.array_equal(got[keep], app[keep])


def test_dust_is_a_gap_not_a_zero(both):
    """A log axis cannot place float dust, and drawing it at the floor
    would read as tail that is not there."""
    for series in chart_reins(both).series[3:]:
        assert all(v is None or v > LOG_FLOOR for v in series.y)


def test_density_series_are_the_pmf_columns(both):
    doc = chart_reins(both, basis='sev')
    df = both.reins_density_df
    for series, column in zip(doc.series[:3],
                              ('p_sev_gross', 'p_sev_ceded', 'p_sev_net')):
        assert np.array_equal(np.array(series.y), df[column].to_numpy())


def test_windows(both):
    doc = chart_reins(both)
    axes = {a.id: a for a in doc.axes}
    lo, hi = axes['loss'].suggested_range
    # the window comes off the FIRST series, the widest of the triple: a
    # cession is bounded by its subject
    gross = both.reins_density_df['p_agg_gross'].to_numpy()
    x = both.reins_density_df['loss'].to_numpy()
    q999 = x[np.searchsorted(np.cumsum(gross), 0.999)]
    assert lo < 0 < q999 <= hi          # unsigned grid reads from zero, padded
    assert hi < x[-1]                   # and crops the heavy tail off
    s_lo, s_hi = axes['survival'].suggested_range
    assert s_hi == 1.0
    assert s_lo >= SURVIVAL_FLOOR
    assert np.log10(s_lo) == int(np.log10(s_lo))   # a round decade


def test_deterministic(both):
    assert canonical_json(chart_reins(both)) == canonical_json(chart_reins(both))
    assert doc_hash(chart_reins(both, basis='occ')) != \
        doc_hash(chart_reins(both, basis='agg'))


# ------------------------------------------------------------- renderer

def test_renderer_draws_two_panels_sharing_x(both):
    from aggregate.plots import plot_chartdoc, plt
    doc = chart_reins(both)
    fig = plot_chartdoc(doc)
    density, tail = fig.axes[0], fig.axes[1]
    assert len(fig.axes) == 2
    assert density.get_xlim() == tail.get_xlim()
    assert tail.get_yscale() == 'log'
    # the initial view honors the emitter's window rather than the grid
    lo, hi = {a.id: a for a in doc.axes}['loss'].suggested_range
    assert density.get_xlim() == pytest.approx((lo, hi))
    plt.close('all')


def test_renderer_refuses_a_supplied_axes_for_many_panels(both):
    from aggregate.charts import ChartCapabilityError
    from aggregate.plots import plot_chartdoc, plt
    _, ax = plt.subplots()
    with pytest.raises(ChartCapabilityError, match='cannot'):
        plot_chartdoc(chart_reins(both), ax=ax)
    plt.close('all')
