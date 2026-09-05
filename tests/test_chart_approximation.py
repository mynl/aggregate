"""``chart_approximation`` and ``chart_approximation_tails``: the five fits.

The teaching charts behind ``approximation_df``
(dev/done/plan-approximate-punchup.md). ``approximation``: one density
panel, the realized mass with the family densities overlaid in grid-mass
terms. The exceedance panel with the implied tail shipped ``1.0.0a332`` to
``1.0.0a336``, was dropped by ruling at ``1.0.0a337`` (byte count of the
default document), and returned at ``1.0.0a344`` as its own chart,
``approximation_tails``: one survival curve per law, the cdf as the
declared complement reading, the exchanged panel the upper Lee plot.
"""

import json

import pytest

from aggregate import build  # noqa: E402
from aggregate.charts import (  # noqa: E402
    available_charts, build_chart_doc, canonical_json, chart_approximation,
    chart_approximation_tails, primary_chart,
)

_CONT = 'agg CX.Cont 100 claims sev lognorm 50 cv 2 poisson'
_FAMILIES = ['norm', 'gamma', 'lognorm', 'sgamma', 'slognorm']


@pytest.fixture(scope='module')
def cont():
    return build(_CONT)


def test_registered_and_available(cont):
    assert 'approximation' in available_charts(cont)
    # the aggregate's own picture stays the agg chart
    assert primary_chart(cont) == 'agg'


def test_unavailable_before_update():
    a = build(_CONT, update=False)
    assert 'approximation' not in available_charts(a)


def test_one_panel_and_six_series(cont):
    doc = chart_approximation(cont)
    assert doc.name == 'approximation'
    assert [p.id for p in doc.panels] == ['density']
    assert [ax.id for ax in doc.axes] == ['outcome', 'mass']
    density = [(s.name, s.role) for s in doc.series]
    assert density == [('Exact', 'density')] + [(f, 'density')
                                                for f in _FAMILIES]
    # the mean mark rides the density panel now that it is the only one
    assert [(m.panel_id, m.role) for m in doc.marks] == [('density', 'mean')]


def test_family_series_are_lattice_payloads(cont):
    """Every series x rides the shared grid, so the lattice form applies:
    no series ships an explicit x tuple (the byte-count ruling behind the
    tail-panel drop)."""
    doc = chart_approximation(cont)
    for s in doc.series:
        assert s.x_lattice is not None, s.name
        assert s.x is None, s.name


def test_canonical_json_round_trips(cont):
    doc = build_chart_doc(cont, 'approximation')
    json.loads(canonical_json(doc))


def test_signed_subject_drops_log_and_inadmissible_families():
    """A signed (negative-mean) subject keeps a linear outcome axis and
    drops the unshifted positive-support families it cannot admit."""
    a = build('agg CX.S 10 claims ssev -lognorm 10 cv 0.5 poisson')
    doc = build_chart_doc(a, 'approximation')
    outcome = next(ax for ax in doc.axes if ax.id == 'outcome')
    assert outcome.scales == ('linear',)
    density_names = {s.name for s in doc.series if s.panel_id == 'density'}
    assert 'gamma' not in density_names and 'lognorm' not in density_names
    assert {'Exact', 'norm', 'sgamma', 'slognorm'} <= density_names
    json.loads(canonical_json(doc))


def test_left_skew_clamped_subject_serves_all(cont):
    a = build('agg CX.L 1 claim sev 100 * beta 5 1.3 fixed')
    doc = build_chart_doc(a, 'approximation')
    assert len(doc.series) == 6
    json.loads(canonical_json(doc))


# ----------------------------------------------------------------------
# approximation_tails
# ----------------------------------------------------------------------
def test_tails_registered_and_available(cont):
    assert 'approximation_tails' in available_charts(cont)
    a = build(_CONT, update=False)
    assert 'approximation_tails' not in available_charts(a)


def test_tails_one_panel_one_curve_per_law(cont):
    doc = chart_approximation_tails(cont)
    assert doc.name == 'approximation_tails'
    assert [p.id for p in doc.panels] == ['tails']
    names = [(s.name, s.role) for s in doc.series]
    assert names == ([('Exact', 'survival')]
                     + [(f, 'survival') for f in _FAMILIES]
                     + [('Implied tail', 'ceiling')])
    assert [(m.panel_id, m.role) for m in doc.marks] == [('tails', 'mean')]


def test_tails_readings_are_declared_not_shipped(cont):
    """The cdf, return period and quantile readings ride the one survival
    curve per law: a complement axis, a reciprocal axis, and an invertible
    panel, so nothing travels twice."""
    doc = chart_approximation_tails(cont)
    by_id = {ax.id: ax for ax in doc.axes}
    assert by_id['p'].complement_of == 'survival'
    assert by_id['return_period'].reciprocal_of == 'survival'
    panel = doc.panels[0]
    assert panel.invertible and panel.inverse_title
    assert doc.meta['return_period_map'] == 'reciprocal'


def test_tails_series_are_lattice_payloads_and_trimmed(cont):
    """Trims are end trims on the shared grid, so every kept run is still
    a lattice, and no drawn survival reaches the float-dust floor."""
    doc = chart_approximation_tails(cont)
    for s in doc.series:
        assert s.x_lattice is not None, s.name
        assert s.x is None, s.name
        assert min(s.y) > 1e-9, s.name
    implied = next(s for s in doc.series if s.role == 'ceiling')
    assert max(implied.y) <= 1.0


def test_tails_canonical_json_round_trips(cont):
    doc = build_chart_doc(cont, 'approximation_tails')
    json.loads(canonical_json(doc))


def test_tails_signed_subject_drops_inadmissible_families():
    a = build('agg CX.TS 10 claims ssev -lognorm 10 cv 0.5 poisson')
    doc = build_chart_doc(a, 'approximation_tails')
    outcome = next(ax for ax in doc.axes if ax.id == 'outcome')
    assert outcome.scales == ('linear',)
    names = {s.name for s in doc.series}
    assert 'gamma' not in names and 'lognorm' not in names
    assert {'Exact', 'norm', 'sgamma', 'slognorm'} <= names
    json.loads(canonical_json(doc))
