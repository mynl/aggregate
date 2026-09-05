"""``chart_approximation``: the five fits on the realized mass.

The teaching chart behind ``approximation_df``
(dev/done/plan-approximate-punchup.md): one density panel, the realized
mass with the family densities overlaid in grid-mass terms. The exceedance
panel with the implied tail shipped ``1.0.0a332`` to ``1.0.0a336`` and was
dropped by ruling at ``1.0.0a337``; the frame is unchanged.
"""

import json

import pytest

from aggregate import build  # noqa: E402
from aggregate.charts import (  # noqa: E402
    available_charts, build_chart_doc, canonical_json, chart_approximation,
    primary_chart,
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
