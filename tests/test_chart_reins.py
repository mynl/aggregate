"""[Chart-Reins] an occurrence program, per claim and in total.

The chart is the occurrence-reinsurance plot. Its two panels answer two
different questions and share nothing, not even a loss axis: a per-claim
loss and an annual aggregate are different quantities, and one window
across both would say they were the same.

Aggregate only at 1.0, by the author's decision. The three curves are
separate distributions and not a decomposition.

DecL programs mirror the ``CH.Reins*`` entries in
``src/aggregate/agg/decl-testers.agg``.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')

from aggregate import build  # noqa: E402
from aggregate.charts import (  # noqa: E402
    available_charts, canonical_json, chart_reins, human_strings,
    primary_chart,
)
from aggregate.charts._emit_reins import AGG_COLUMNS, NAMES  # noqa: E402

_OCC = ('agg CR.Occ 100 claims 1000 xs 0 sev lognorm 50 cv 2 '
        'occurrence net of 100 xs 100 poisson')
_GROSS = 'agg CR.Gross 100 claims 1000 xs 0 sev lognorm 50 cv 2 poisson'
_AGG_ONLY = ('agg CR.AggOnly 100 claims 1000 xs 0 sev lognorm 50 cv 2 '
             'poisson aggregate net of 2000 xs 3000')
_PORT = ('port CR.P agg A 50 claims 1000 xs 0 sev lognorm 50 cv 1.5 '
         'occurrence net of 100 xs 100 poisson')


@pytest.fixture(scope='module')
def occ():
    return build(_OCC)


def axes_of(doc):
    return {a.id: a for a in doc.axes}


def on(doc, panel):
    return [s for s in doc.series if s.panel_id == panel]


# ----------------------------------------------------------- capability

def test_available_only_with_an_occurrence_program(occ):
    """The aggregate's own chart is always there; reins joins it when the
    occurrence program does."""
    assert 'reins' in available_charts(occ)
    assert 'reins' not in available_charts(build(_GROSS))
    assert primary_chart(occ) == 'agg'      # reins is a view, not the picture


def test_aggregate_only_at_1_0():
    """A book's units cede on different stages, so there is no book triple,
    and an aggregate cover is a separate contract with its own picture."""
    assert 'reins' not in available_charts(build(_PORT))
    assert 'reins' not in available_charts(build(_AGG_ONLY))


# --------------------------------------------------------------- shape

def test_two_panels_sharing_nothing(occ):
    doc = chart_reins(occ)
    claim, annual = doc.panels
    assert (claim.id, annual.id) == ('occurrence', 'aggregate')
    assert claim.x_axis != annual.x_axis
    assert claim.y_axis != annual.y_axis


def test_the_triple_draws_on_both_panels_net_last(occ):
    doc = chart_reins(occ)
    for panel in ('occurrence', 'aggregate'):
        assert [s.role for s in on(doc, panel)] == ['gross', 'ceded', 'net']
        assert [s.name for s in on(doc, panel)] == list(NAMES)


def test_the_left_panel_is_the_claim_and_the_right_the_year(occ):
    doc = chart_reins(occ)
    ax = axes_of(doc)
    assert ax['claim'].label == 'Loss per claim'
    assert ax['annual'].label == 'Aggregate loss'
    # the claim window is the occurrence limit, which bounds the cession
    assert ax['claim'].suggested_range[1] == pytest.approx(1000.0, rel=0.05)
    assert ax['annual'].suggested_range[1] > 1000.0


def test_the_occurrence_panel_reads_on_log_and_only_on_log(occ):
    """A layered severity is a spike and a tail; linear is a spike."""
    ax = axes_of(chart_reins(occ))['sev_density']
    assert ax.scale == 'log'
    assert ax.scales == ('log',)


def test_the_aggregate_panel_carries_every_reading(occ):
    doc = chart_reins(occ)
    ax = axes_of(doc)
    assert ax['annual'].scales == ('linear', 'log')
    assert ax['return_period'].reciprocal_of == 'p'
    assert doc.meta['return_period_map'] == 'complement'
    annual = doc.panels[1]
    assert annual.invertible
    assert annual.inverse_title == 'Distribution function'


def test_the_probability_axis_offers_its_reflection(occ):
    """The survival function, and the log reading only it admits."""
    ax = axes_of(chart_reins(occ))
    assert ax['survival'].complement_of == 'p'
    assert ax['survival'].scales == ('linear', 'log')
    assert ax['p'].scales == ('linear',)


def test_the_right_panel_is_a_quantile_curve(occ):
    """Non-exceeding probability against loss, trimmed to the support."""
    doc = chart_reins(occ)
    gross = on(doc, 'aggregate')[0]
    p, loss = np.array(gross.x_values), np.array(gross.y_values)
    assert (np.diff(p) >= 0).all() and (np.diff(loss) > 0).all()
    assert p[-1] == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(
        np.cumsum(occ.reins_density_df[AGG_COLUMNS[0]].to_numpy())[:len(p)][-1],
        p[-1], rtol=1e-9)


def test_tex_is_total(occ):
    doc = chart_reins(occ)
    assert not set(human_strings(doc)) - set(doc.tex)


def test_deterministic(occ):
    assert canonical_json(chart_reins(occ)) == canonical_json(chart_reins(occ))


# ------------------------------------------------------------- renderer

def test_the_readings_render(occ):
    from aggregate.plots import plt
    plain = occ.reins_occ_plot()
    claim, annual = plain.axes
    assert claim.get_yscale() == 'log'
    assert annual.get_xscale() == 'linear'
    assert annual.get_title() == 'Aggregate'

    rp = occ.reins_occ_plot(return_period=True)
    assert rp.axes[1].get_xscale() == 'log'
    assert rp.axes[0].get_yscale() == 'log'      # untouched, already log

    inverted = occ.reins_occ_plot(invert=True)
    assert inverted.axes[1].get_title() == 'Distribution function'
    assert inverted.axes[1].get_xlabel() == 'Aggregate loss'
    plt.close('all')
